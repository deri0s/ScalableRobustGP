import os
import torch
import gpytorch
import copy
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler as ss
from matplotlib import pyplot as plt
from pathlib import Path
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.kernels import ScaleKernel, Kernel
from gpytorch.kernels import RBFKernel as RBF, RQKernel as RQ
from gpytorch.kernels import LinearKernel as Lin, PeriodicKernel as Per

"""
NSG data

Do not adjust data for timelags.
"""

# NSG post processes data location
ROOT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT_PATH / "data" / "processed" / "Training_data_partitions"
EXPERT_PATH = ROOT_PATH / "trained" / "experts"

apply_timelags = True
N_partitions = 5
N_sim = 200

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

index = 0
file = PROCESSED_PATH / f'data{index}.xlsx'

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

expert_path = os.path.join(EXPERT_PATH, f'expert{index}.pth')
scaler_path = os.path.join(EXPERT_PATH, f'scaler{index}.pth')
    
# Load train expert
gp0 = torch.load(expert_path)
scaler = torch.load(scaler_path)

likelihood = GaussianLikelihood()

# Predictions
gp0.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(gp0(X))

    # Unormalise predictions
    pred_mean = observed_pred.mean
    mu0 = scaler.inverse_transform(pred_mean.unsqueeze(1))[:,0]

level = 2

def get_hyper(gp):
    os = gp.covar_module.outputscale.item()
    rq_ls = gp.covar_module.base_kernel.kernels[0].lengthscale.squeeze().tolist()
    alpha = gp.covar_module.base_kernel.kernels[0].alpha.item()
    se_ls = gp.covar_module.base_kernel.kernels[1].lengthscale.squeeze().tolist()
    noise_var = likelihood.noise.item()
    return os, rq_ls, alpha, se_ls, noise_var

outputscale, rq_ls, alpha, se_ls, noise_var = get_hyper(gp=gp0)

print("\nAWS kernel estimation:")
print('Kernel:\n', gp0.covar_module)
print("Outputscale:", outputscale)
print('RQ-LS:\n', rq_ls)
print('RQ-LS:\n', alpha)
print('SE-LS:\n', se_ls)
print('Noise var: ', noise_var)


""" 3. Fine tuning """

eval_perc = 0.2
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

# Make predictions with the original GP
gp0.eval()
gp0.likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred_tuned = gp0.likelihood(gp0(X))

# Unormalise predictions
pred_mean_tuned = observed_pred_tuned.mean
mu_tuned = scaler.inverse_transform(pred_mean_tuned.unsqueeze(1))[:,0]
stds_tuned = scaler.inverse_transform(observed_pred_tuned.stddev.unsqueeze(1))[:,0]
lower_stand_tuned, upper_stand_tuned = observed_pred_tuned.confidence_region()
lower_tuned = scaler.inverse_transform(lower_stand_tuned.unsqueeze(1))[:,0]
upper_tuned = scaler.inverse_transform(upper_stand_tuned.unsqueeze(1))[:,0]

mse0 = mean_squared_error(mu0, y_processed)

def get_samples(D, N_sim, l_bounds, u_bounds):
    """ Samples from a Latin Hypercube Sampling model """
    sampler = qmc.LatinHypercube(d=D)
    sample = sampler.random(n=N_sim)
    
    if len(l_bounds) != D or len(u_bounds) != D:
        raise ValueError(f"Bounds dimensions ({len(l_bounds)}, {len(u_bounds)}) must match sample dimension ({D})")
    
    return qmc.scale(sample, l_bounds, u_bounds)

# Get the actual lengthscale dimensions from the kernels
rq_dim = gp0.covar_module.base_kernel.kernels[0].lengthscale.shape[-1]
se_dim = gp0.covar_module.base_kernel.kernels[1].lengthscale.shape[-1]

# Hyperparameter space dimensionality
dim = rq_dim + se_dim + 3

# Hyper space bounds
lowerb = 1 * np.ones(dim)
upperb = 100 * np.ones(dim)

# the last 3 values will be reserved for the Outputscale, alpha, and Noise var, respectively
lowerb[-3] = 1
lowerb[-2] = 1
lowerb[-1] = 0.01

upperb[-3] = 10
upperb[-2] = 3
upperb[-1] = 0.08

# Generate samples from a Latin Hypercube
samples = get_samples(dim, N_sim, lowerb, upperb)

gp = copy.deepcopy(gp0)
best_mse = float('inf')

# Start Hyperparameter tunning
for n in range(N_sim):
    # 1. Copy the base model and assign sampled hyperparameters
    gp = copy.deepcopy(gp0)
    gp.covar_module.outputscale = samples[n, -3]
    # RQ
    gp.covar_module.base_kernel.kernels[0].lengthscale = torch.tensor(samples[n, 0:rq_dim],
                                                                      dtype=floating_point)
    gp.covar_module.base_kernel.kernels[1].alpha = torch.tensor(samples[n, -2],
                                                                dtype=floating_point)
    # SE
    gp.covar_module.base_kernel.kernels[1].lengthscale = torch.tensor(samples[n, rq_dim:dim-3],
                                                                      dtype=floating_point)
    likelihood.noise = samples[n, -1]

    gp.train()
    likelihood.train()

    # 2. Set up optimizer (include inducing points if you want to optimize them)
    optimizer = torch.optim.Adam([
        {'params': gp.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.01)

    # 3. Use the VariationalELBO loss
    mll = gpytorch.mlls.VariationalELBO(likelihood, gp, num_data=X_train.size(0))

    # 4. Training loop
    training_iter = 200  # Adjust as needed
    for _ in range(training_iter):
        optimizer.zero_grad()
        output = gp(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()

    # 5. Evaluate on full data (or test set)
    gp.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred_tuned = likelihood(gp(X))
        pred_mean_tuned = observed_pred_tuned.mean
        mu_tuned = scaler.inverse_transform(pred_mean_tuned.unsqueeze(1))[:, 0]
        mse = mean_squared_error(mu_tuned, y_processed)

    if mse < best_mse:
        best_mse = mse
        best_gp = copy.deepcopy(gp)
        best_mu = mu_tuned
        print(f'Sim: {n}, Best MSE found: {mse:.6f}, alpha sample {samples[n, -2]}')

    print(f'Sim: {n}, MSE-tuned: {mse:.6f}')

""" 4. Feature importance """
outputscale, rq_ls, alpha, se_ls, noise_var = get_hyper(gp=best_gp)

print('\nResults from the LHS initialisation:\n')
print("Outputscale:", outputscale)
print('RQ(alpha): ', alpha)
print('Noise var: ', noise_var)

rq_dict = {'inputs': X_df.columns.values, 'lengthscales': rq_ls}
se_dict = {'inputs': X_df.columns.values, 'lengthscales': se_ls}
rq_inputs = pd.DataFrame(rq_dict)
se_inputs = pd.DataFrame(se_dict)

print('\nRQ\n', rq_inputs.sort_values(by='lengthscales'))
print('\nSE\n', se_inputs.sort_values(by='lengthscales'))

#-----------------------------------------------------------------------------
# PLOT TRAINING DATA
#-----------------------------------------------------------------------------
end_indx = int(len(X)*0.8)
fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
fig.autofmt_xdate()

plt.title(f'Expert {index}')
ax.plot(date_time, y_processed, '*', color='green', label='Val')
# ax.plot(date_time, y_filtered, color='blue', label='Filtered')
# ax.plot(date_time[i_clean], y_clean, 'o', color='green', label='furnace')
ax.plot(date_time, mu0, color='blue', label='GP-AWS')
ax.plot(date_time, best_mu, color='red', label='GP-Tuned')
ax.vlines(x=date_time[end_indx], ymin=0, ymax=max(y_processed),
        colors='black', ls='--', label='Test-data')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

plt.show()