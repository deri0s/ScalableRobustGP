import os
from pathlib import Path

import gpytorch
import numpy as np
import pandas as pd
import torch
from gpytorch.likelihoods import GaussianLikelihood
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error

"""
NSG data

Do not adjust data for timelags.
"""

data_index = 0
# NSG post processes data location
ROOT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT_PATH / "data" / "processed" / "Training_data_partitions"
EXPERT_PATH = ROOT_PATH / "trained" / "experts"

apply_timelags = True


def align_inputs(x_df, y_df, t_series):
    xdeep = x_df.copy()
    ydeep = y_df.copy()
    # Ensure t_series values are numeric before finding max
    numeric_t_series = pd.to_numeric(t_series, errors="coerce").fillna(0)
    if numeric_t_series.empty:
        max_lag = 0
    else:
        max_lag = int(max(numeric_t_series))

    # X
    for name, lag in t_series.items():
        # Ensure lag is treated as integer for shift
        try:
            lag_int = int(float(lag))
            if lag_int > 0:  # Only shift if lag is positive
                xdeep[name] = xdeep[name].shift(lag_int)
        except ValueError:
            print(
                f"Warning: Could not convert lag '{lag}' for feature '{name}' to int. Skipping shift."
            )

    # Drop rows with NaNs introduced by shifting (only drop up to max_lag rows from top)
    xdeep = xdeep.iloc[max_lag:]  # More direct way to handle shift NaNs

    # y and date-time alignment
    # Ensure ydeep has enough rows before slicing
    if len(ydeep) >= max_lag:
        ydeep = ydeep.iloc[max_lag:].reset_index(drop=True)
    else:
        # Handle case where ydeep is shorter than max_lag (e.g., return empty DataFrames)
        print(
            f"Warning: y DataFrame length ({len(ydeep)}) is less than max_lag ({max_lag}). Alignment might be incorrect."
        )
        return pd.DataFrame(columns=x_df.columns), pd.DataFrame(columns=y_df.columns)

    # Ensure xdeep and ydeep have the same length after alignment
    common_len = min(len(xdeep), len(ydeep))
    xdeep = xdeep.iloc[:common_len].reset_index(drop=True)
    ydeep = ydeep.iloc[:common_len].reset_index(drop=True)

    return xdeep, ydeep


def get_hyper(gp):
    """Extract hyperparameters based on kernel type"""
    results = {}
    results["outputscale"] = gp.covar_module.outputscale.item()
    if hasattr(gp.covar_module.base_kernel, "kernels"):  # Additive kernel
        results["kernel_type"] = "additive"
        for i, k in enumerate(gp.covar_module.base_kernel.kernels):
            results[f"kernel_{i}_name"] = k.__class__.__name__
            if hasattr(k, "lengthscale"):
                results[f"kernel_{i}_lengthscales"] = k.lengthscale.squeeze().tolist()
            if hasattr(k, "alpha"):
                results[f"kernel_{i}_alpha"] = k.alpha.item()
    else:
        results["kernel_type"] = "single"
        results["kernel_name"] = gp.covar_module.base_kernel.__class__.__name__
        results["lengthscales"] = (
            gp.covar_module.base_kernel.lengthscale.squeeze().tolist()
        )
        if results["kernel_name"] == "RQKernel":
            results["alpha"] = gp.covar_module.base_kernel.alpha.item()

    results["noise"] = gp.likelihood.noise.item()
    return results


def predict_and_eval(gp, likelihood, scaler, X, X_train, X_test):
    gp.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(gp(X))
        observed_pred_train = likelihood(gp(X_train))
        observed_pred_test = likelihood(gp(X_test))

        # Unormalise predictions
        pred_mean = observed_pred.mean
        pred_mean_train = observed_pred_train.mean
        pred_mean_test = observed_pred_test.mean

    mu = scaler.inverse_transform(pred_mean.unsqueeze(1))[:, 0]
    mu_train = scaler.inverse_transform(pred_mean_train.unsqueeze(1))[:, 0]
    mu_test = scaler.inverse_transform(pred_mean_test.unsqueeze(1))[:, 0]
    stds = scaler.inverse_transform(observed_pred.stddev.unsqueeze(1))[:, 0]
    lower_stand, upper_stand = observed_pred.confidence_region()
    lower = scaler.inverse_transform(lower_stand.unsqueeze(1))[:, 0]
    upper = scaler.inverse_transform(upper_stand.unsqueeze(1))[:, 0]

    mse_all = mean_squared_error(y_processed, mu)
    mse_train = mean_squared_error(y_train, mu_train)
    mse_test = mean_squared_error(y_test, mu_test)

    return mse_all, mse_train, mse_test, mu, lower, upper


file = PROCESSED_PATH / f"data{data_index}.xlsx"

# Training df
X_df = pd.read_excel(file, sheet_name="X_stand")
y_df = pd.read_excel(file, sheet_name="y_nonstand")
t_df = pd.read_excel(file, sheet_name="timelags")
t_series = t_df.iloc[0, :]

# feature selection
X_df.drop(columns=["9282 Tweel Position"], inplace=True)
t_df.drop(columns=["9282 Tweel Position"], inplace=True)
X_df.drop(columns=["10091 Furnace Load"], inplace=True)
t_df.drop(columns=["10091 Furnace Load"], inplace=True)

X_df, y_df = align_inputs(X_df, y_df, t_df.iloc[0, :])

X_np = X_df.values
y_processed = y_df.y_processed.values
date_time = y_df.date_time.values

# Convert data to torch tensors
floating_point = torch.float64
X = torch.tensor(X_np, dtype=floating_point)


""" 2. Load trained experts """
expert_path = os.path.join(EXPERT_PATH, f"expert{data_index}.pth")
scaler_path = os.path.join(EXPERT_PATH, f"scaler{data_index}.pth")

print("Exper: ", data_index)
# Load train expert
gp = torch.load(expert_path, weights_only=False)
scaler = torch.load(scaler_path, weights_only=False)

likelihood = GaussianLikelihood()

print(f"\nEstimated Kernel:\n {gp.covar_module.base_kernel}")

hyperparams0 = get_hyper(gp)

print("\nHyperparameters:")
for key, value in hyperparams0.items():
    if not isinstance(value, list):
        print(f"{key}: {value}")

# feature importance
if hyperparams0["kernel_type"] == "additive":
    for i in range(2):
        print(f"Kernel: {hyperparams0[f'kernel_{i}_name']}")
        feature_importance = pd.DataFrame(
            {
                "inputs": X_df.columns.values,
                "lengthscales": hyperparams0[f"kernel_{i}_lengthscales"],
            }
        )
        print(feature_importance.sort_values(by="lengthscales"))
else:
    print(f"\nInputs: f{X_df.shape}, hyper: {len(hyperparams0)}")
    feature_importance = pd.DataFrame(
        {"inputs": X_df.columns.values, "lengthscales": hyperparams0["lengthscales"]}
    )
    print("\nFeature Importance (sorted by lengthscale):")
    print(feature_importance.sort_values(by="lengthscales"))

# Predictions
gp.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(gp(X))

    # Unormalise predictions
    pred_mean = observed_pred.mean
    mu = scaler.inverse_transform(pred_mean.unsqueeze(1))[:, 0]
    stds = scaler.inverse_transform(observed_pred.stddev.unsqueeze(1))[:, 0]
    lower_stand, upper_stand = observed_pred.confidence_region()
    lower = scaler.inverse_transform(lower_stand.unsqueeze(1))[:, 0]
    upper = scaler.inverse_transform(upper_stand.unsqueeze(1))[:, 0]

    print(f"MSE(train-test): {mean_squared_error(y_processed, mu)}")

# -----------------------------------------------------------------------------
# PLOT TRAINING DATA
# -----------------------------------------------------------------------------
end_indx = int(len(X) * 0.8)
fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc("xtick", labelsize=14)
plt.rc("ytick", labelsize=14)
fig.autofmt_xdate()

plt.title(f"Expert {data_index}")
ax.fill_between(
    date_time,
    mu - 1.96 * stds,
    mu + 1.96 * stds,
    alpha=0.3,
    color="coral",
    label="95% CI",
)
ax.plot(date_time, y_processed, "*", color="green", label="Val")
# ax.plot(date_time, y_filtered, color='blue', label='Filtered')
# ax.plot(date_time[i_clean], y_clean, 'o', color='green', label='furnace')
ax.plot(date_time, mu, color="red", label="GP")
ax.vlines(
    x=date_time[end_indx],
    ymin=0,
    ymax=max(y_processed),
    colors="black",
    ls="--",
    label="Test-data",
)
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size": 18}, facecolor="white", framealpha=1.0)

plt.show()
