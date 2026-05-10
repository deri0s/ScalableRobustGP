import os
import torch
import gpytorch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler as ss
from matplotlib import pyplot as plt
from pathlib import Path
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from models.svgp_auto_model_construction import GPTraining
import models.svgp_auto_model_construction
print("Executing module from file:")
print(models.svgp_auto_model_construction.__file__)

"""
NSG data

Do not adjust data for timelags.
"""
expert_index = 1
data_index = 0

# NSG post processes data location
ROOT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT_PATH / "data" / "processed" / "Training_data_partitions"
EXPERT_PATH = ROOT_PATH / "trained" / "experts"
file = PROCESSED_PATH / f'data{data_index}.xlsx'

apply_timelags = True

def align_inputs(x_df, y_df, t_series):
    xdeep = x_df.copy()
    ydeep = y_df.copy()
    # Ensure t_series values are numeric before finding max
    numeric_t_series = pd.to_numeric(t_series, errors='coerce').fillna(0)
    if numeric_t_series.empty:
         max_lag = 0
    else:
         max_lag = int(max(numeric_t_series))

    # X
    for name, lag in t_series.items():
        # Ensure lag is treated as integer for shift
        try:
            lag_int = int(float(lag))
            if lag_int > 0: # Only shift if lag is positive
                 xdeep[name] = xdeep[name].shift(lag_int)
        except ValueError:
            print(f"Warning: Could not convert lag '{lag}' for feature '{name}' to int. Skipping shift.")

    # Drop rows with NaNs introduced by shifting (only drop up to max_lag rows from top)
    xdeep = xdeep.iloc[max_lag:] # More direct way to handle shift NaNs

    # y and date-time alignment
    # Ensure ydeep has enough rows before slicing
    if len(ydeep) >= max_lag:
        ydeep = ydeep.iloc[max_lag:].reset_index(drop=True)
    else:
        # Handle case where ydeep is shorter than max_lag (e.g., return empty DataFrames)
        print(f"Warning: y DataFrame length ({len(ydeep)}) is less than max_lag ({max_lag}). Alignment might be incorrect.")
        return pd.DataFrame(columns=x_df.columns), pd.DataFrame(columns=y_df.columns)

    # Ensure xdeep and ydeep have the same length after alignment
    common_len = min(len(xdeep), len(ydeep))
    xdeep = xdeep.iloc[:common_len].reset_index(drop=True)
    ydeep = ydeep.iloc[:common_len].reset_index(drop=True)

    return xdeep, ydeep

# Training df
X_df = pd.read_excel(file, sheet_name='X_stand')
y_df = pd.read_excel(file, sheet_name='y_nonstand')
t_df = pd.read_excel(file, sheet_name='timelags')
t_series = t_df.iloc[0, :]

X_df, y_df = align_inputs(X_df, y_df, t_df.iloc[0,:])

X_np = X_df.values
y_processed = y_df.y_processed.values
date_time = y_df.date_time.values

# Convert data to torch tensors
floating_point = torch.float64
X = torch.tensor(X_np, dtype=floating_point)
N, D = np.shape(X)

""" 2. Load trained experts """

expert_path = os.path.join(EXPERT_PATH, f'expert{expert_index}.pth')
scaler_path = os.path.join(EXPERT_PATH, f'scaler{expert_index}.pth')

# Load train expert
gp0 = torch.load(expert_path, weights_only=False)
scaler = torch.load(scaler_path, weights_only=False)

# likelihood = VariationalELBO()
likelihood = GaussianLikelihood()

# Predictions
gp0.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(gp0(X))

    # Unormalise predictions
    pred_mean = observed_pred.mean
    mu0 = scaler.inverse_transform(pred_mean.unsqueeze(1))[:,0]


""" 3. Fine tuning """

eval_perc = 0.7
end_train = N - int(N*eval_perc)

X_train = X[0:end_train]
date_train = date_time[0:end_train]
N_train = len(X_train)
y_train_nonstand = y_processed[0:end_train]

# Define X_test_np and y_test_nonstand correctly for evaluation metric
X_test = X[end_train:N]
y_test_nonstand = y_processed[end_train:N]

# Standardise outputs
y_train_reshape = y_train_nonstand.reshape(-1,1)
scaler = ss()
scaler.fit(y_train_reshape)
y_stand_np = scaler.transform(y_train_reshape)

y_test_reshape = y_test_nonstand.reshape(-1,1)
y_test_stand = scaler.transform(y_test_reshape)

# Convert data to torch tensors
floating_point = torch.float64
gp0 = gp0.double()
likelihood = likelihood.double()

y_train = torch.tensor(y_stand_np, dtype=floating_point).squeeze()
y_test = torch.tensor(y_test_stand, dtype=floating_point).squeeze()

auto_trainer = GPTraining(gp0, X_train, y_train,
                          X_eval=X_test, y_eval=y_test)

def generate_stds(lengthscales, base_std_dev):
    """ Penalise lengthscales that are high using a greater std """
    if not torch.is_tensor(lengthscales):
        lengthscales = torch.tensor(lengthscales)

    # Handle both scalar and ARD cases
    if lengthscales.numel() == 1:
        # Scalar lengthscale case
        return base_std_dev
    else:
        # ARD case - return list with correct length
        min_lengthscale = torch.min(lengthscales)
        std_devs = base_std_dev * torch.exp((lengthscales - min_lengthscale)/6)
        std_devs = torch.tensor([300 if std == torch.inf else std for std in std_devs])
        return std_devs.tolist()

# Account for ARD
try:
    if hasattr(gp0.covar_module.base_kernel, 'lengthscale'):
        lengthscales = gp0.covar_module.base_kernel.lengthscale.squeeze()
    else:
        lengthscales = gp0.covar_module.base_kernel.kernels[0].lengthscale.squeeze()
    
    ls_stds = generate_stds(lengthscales, base_std_dev=1e-1)
except Exception as e:
    print(f"Warning: Could not extract lengthscales for tuning: {e}")
    ls_stds = [1e-14] * D

stds = {'outputscale': 1e-14,
        'se_lengthscale': ls_stds,
        'rq_lengthscale': ls_stds,
        'rq_alpha': 1e-3,
        'noise_variance': 1e-14}

# Update the trainer's stds dictionary
auto_trainer.param_stds = stds

print(f"\n🔧 Starting hyperparameter tuning")
tuned_gp = auto_trainer.tune(
    gp_to_tune=gp0,
    N_sim=20,
    mse_stop=0.01,
    lr=0.0001,
    training_iterations=200,
    batch_size=256, track_mse='eval')
print("✓ Hyperparameter tuning completed successfully")

# Make predictions with tuned model
tuned_gp.eval()
tuned_gp.likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred_tuned = tuned_gp.likelihood(tuned_gp(X))

# Unormalise predictions
pred_mean_tuned = observed_pred_tuned.mean
mu_tuned = scaler.inverse_transform(pred_mean_tuned.unsqueeze(1))[:,0]
stds_tuned = scaler.inverse_transform(observed_pred_tuned.stddev.unsqueeze(1))[:,0]
lower_stand_tuned, upper_stand_tuned = observed_pred_tuned.confidence_region()
lower_tuned = scaler.inverse_transform(lower_stand_tuned.unsqueeze(1))[:,0]
upper_tuned = scaler.inverse_transform(upper_stand_tuned.unsqueeze(1))[:,0]

mse0 = mean_squared_error(mu0, y_processed)
mse_tuning = mean_squared_error(mu_tuned, y_processed)

print(f'MSE-AWS: {mse0:.6f}')
print(f'MSE-tuned: {mse_tuning:.6f}')


""" SAVE TRAINED EXPERT """
model_path = EXPERT_PATH / f'expert{expert_index}0.pth'
scaler_path = EXPERT_PATH / f'scaler{expert_index}0.pth'

torch.save(tuned_gp, model_path)
torch.save(scaler, scaler_path)

#-----------------------------------------------------------------------------
# PLOT TRAINING DATA
#-----------------------------------------------------------------------------
end_indx = int(len(X)*0.3)
fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
fig.autofmt_xdate()

plt.title(f'Expert {expert_index} - Data {data_index}')
ax.plot(date_time, y_processed, '*', color='green', label='Val')
# ax.plot(date_time, y_filtered, color='blue', label='Filtered')
ax.plot(date_time, mu0, color='blue', label='GP-AWS')
ax.plot(date_time, mu_tuned, color='red', label='GP-Tuned')
ax.vlines(x=date_time[end_indx], ymin=0, ymax=max(y_processed),
        colors='black', ls='--', label='Test-data')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

plt.show()