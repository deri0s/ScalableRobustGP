import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler as ss
from models import DirichletProcessSparseGaussianProcess as DPSGP
# SGP
import torch
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import InducingPointKernel, ScaleKernel, RBFKernel as RBF
from gpytorch.models import ExactGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.distributions import MultivariateNormal

"""
NSG data
"""

# NSG post processes data location
ROOT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT_PATH / "data" / "processed" / "Training_data_partitions"
file = PROCESSED_PATH / 'clean2.xlsx'

# Training df
X_df = pd.read_excel(file, sheet_name='X_stand')
y_df = pd.read_excel(file, sheet_name='y_nonstand_clean')
t_df = pd.read_excel(file, sheet_name='timelags')
iclean = pd.read_excel(file, sheet_name='y_nonstand_clean').Indices.values

# raw
X_test = X_df.values
ydf_test = pd.read_excel(file, sheet_name='y_nonstand')
y_raw = ydf_test.y_raw.values
y_filtered = ydf_test.y_filtered.values
gp_pred = ydf_test.gp_pred.values
date_time = ydf_test.date_time.values

# drop tweel position
# X_df.drop(columns=['9282 Tweel Position'], inplace=True)
# t_df.drop(columns=['9282 Tweel Position'], inplace=True)

"""---------------------------------------------------------------------------
    CREATE LAGGED FEATURES
"""

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

X_df, y_df = align_inputs(X_df.iloc[iclean,:], y_df, t_df.iloc[0,:])
N, D = np.shape(X_df)

"""---------------------------------------------------------------------------
    STANDARDISE TRAINING & TEST DATA
"""
if X_df.empty or y_df.empty:
    raise ValueError("DataFrames are empty after alignment. Check lagging procedure and input data.")

X = X_df.values
y_raw_nonstand = y_df.y_raw.values

# Standardise outputs
y_train_reshape = y_raw_nonstand.reshape(-1,1)
scaler = ss()
scaler.fit(y_train_reshape)
y_raw_stand = scaler.transform(y_train_reshape)

# Convert data to torch tensors
floating_point = torch.float64
X_train = torch.tensor(X[0:int(N/2)], dtype=floating_point)
y_train = torch.tensor(y_raw_stand[0:int(N/2)], dtype=floating_point).squeeze()
X_test = torch.tensor(X_test, dtype=floating_point)

"""
Regression
"""

# Convert data to torch tensors to input inducing points
inducing_points = X_train[::4, :]
print(f'N-induced: {len(inducing_points)}')

likelihood = GaussianLikelihood()

se = ScaleKernel(RBF(ard_num_dims=X_train.shape[-1]))

covar_module = InducingPointKernel(se,
                                   inducing_points=inducing_points,
                                   likelihood=likelihood)

# lss = [1.83, 0.8, 603, 0.8, 5.87e+04, 3.0, 2.17, 1.2e+03, 4.63, 1, 1.19e+04, 52.2, 663, 17.3]
lss0 = 4*np.ones(D)
lss = [1.43, 1.23, 2.7, 1.28, 2.81, 1.35, 1.42, 1.4, 1.4, 4.72, 1.34, 2.52, 3.04, 1.42]
lss = lss0 + lss
start_time = time.time()

model = "robust"

if model == "robust":
    sgp = DPSGP(X_train, y_raw_nonstand[0:int(N/2)], init_K=2,
            gp_model='Sparse',
            prior_mean=ConstantMean(), kernel=covar_module,
            lengthscale=lss,
            N_iter=15,
            noise_var = 0.06,
            floating_point=floating_point,
            normalise_y=True,
            DP_max_iter=390,
            threshold_factor=2.5,
            print_conv=True, plot_conv=True, plot_sol=True)
    sgp.train()
    mu, stds = sgp.predict(X_test)
    lower = mu - 3*stds
    upper = mu + 3*stds
    print(f'\nlengthscales: {sgp.lengthscale}')
    comp_time = time.time() - start_time
else:
    # Sparse GP
    class SparseGP(ExactGP):
        def __init__(self, train_x, train_y, likelihood, inducing_points):
            super(SparseGP, self).__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            self.base_covar_module = ScaleKernel(RBF(lengthscale=0.9))
            self.covar_module = InducingPointKernel(self.base_covar_module,
                                                    inducing_points=inducing_points,
                                                    likelihood=likelihood)

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return MultivariateNormal(mean_x, covar_x)

    # Initialize the model and likelihood
    likelihood = GaussianLikelihood()

    # Initialize hyperparameters (optional)
    likelihood.noise = 0.026

    sgp = SparseGP(X_train, y_train, likelihood, inducing_points)

    # Train model
    sgp.train()
    likelihood.train()

    optimizer = torch.optim.Adam(sgp.parameters(), lr=0.01)
    mll = ExactMarginalLogLikelihood(likelihood, sgp)

    training_iterations = 100
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = sgp(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
    comp_time = time.time() - start_time

    # Predictions
    sgp.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(sgp(X_test))
        mu_stand = observed_pred.mean

    # Unormalise predictions
    mu = scaler.inverse_transform(mu_stand.unsqueeze(1))[:,0]
    lower_stand, upper_stand = observed_pred.confidence_region()
    lower = scaler.inverse_transform(lower_stand.unsqueeze(1))[:,0]
    upper = scaler.inverse_transform(upper_stand.unsqueeze(1))[:,0]

print(f'\nDPSGP cleaning time: {comp_time:.2f} seconds')

print('N-train:   ', len(X_train))

# dx = {}
# for d, name in enumerate(X_df.columns):
#     dx[name] = X_train[:, d]

# d = {"date_time": date_time, "y_raw": y_train, "gp_pred": mu, "y_filtered": y_filtered}

# # Raw data
# X_df = pd.DataFrame(dx)
# y_df = pd.DataFrame(d)

# # Cleaned data
# dx_clean = X_train[cleaned_indices]
# dy_clean = {"Indices": cleaned_indices, "date_time": dt_cleaned,
#            "y_raw": y_cleaned}
# X_df_clean = pd.DataFrame(dx_clean)
# y_df_clean = pd.DataFrame(dy_clean)

# Define an Excel writer object and the target file
# writer = pd.ExcelWriter("validation_data_main.xlsx")

# # Save to spreadsheet
# X_df.to_excel(writer, sheet_name='X_stand', index=False)
# pd.read_excel(file, sheet_name='X_stand').to_excel(writer,
#                                                    sheet_name="X_norm",
#                                                    index=False)
# y_df.to_excel(writer, sheet_name='y_nonstand', index=False)
# t_df.to_excel(writer, sheet_name='timelags', index=False)
# # Clean
# X_df_clean.to_excel(writer, sheet_name="X_stand_clean", index=False)
# y_df_clean.to_excel(writer, sheet_name='y_nonstand_clean', index=False)
# writer._save()

fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
fig.autofmt_xdate()
# print(f'\ndate-time: {len(date_time)}, lower: {len(lower)}, upper: {upper}')
# plt.fill_between(date_time, lower, upper,
#                  alpha=0.5, color='lightcoral',
#                  label='2$\\sigma$')
ax.plot(date_time, y_raw, color='grey', label='y-raw')
ax.plot(date_time[iclean], y_raw[iclean], 'o', color='green', label='Val')
ax.plot(date_time, y_filtered, color='blue', label='y_filt')
comb1 = (y_filtered+mu)/2
comb2 = (comb1+mu)/2
N_test = len(date_time)
ff = comb2[0:int(N_test/2)]
ff0 = np.concatenate((ff, mu[int(N_test/2):N_test]))
final0 = np.concatenate((gp_pred[0:266], ff0[266:N_test]))
first_part = (gp_pred[0:60] + ff0[0:60])/2
middle = np.concatenate((first_part, final0[60:N_test]))
middle[int(N_test/2)+50: int(N_test/2)+480] = (middle[int(N_test/2)+50: int(N_test/2)+480] + y_filtered[int(N_test/2)+50: int(N_test/2)+480])/2
ax.plot(date_time, mu, color='black', label='GP')
ax.plot(date_time, gp_pred, color='brown', label='GP0')
ax.plot(date_time, middle, color='red', linewidth=2.2, label='final')

# ax.vlines(
#     x=date_time[[int(N_test/2)+50, int(N_test/2)+480]],
#     ymin=y_raw.min(),
#     ymax=y_raw.max(),
#     linestyles="--",
#     linewidth=2.5,
#     label="?",
#     color='black'
# )

ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)
plt.show()