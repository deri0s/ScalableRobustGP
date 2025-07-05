import os
import torch
import gpytorch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler as ss
from gpytorch.likelihoods import GaussianLikelihood
from sklearn.cluster import KMeans
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import ScaleKernel
from gpytorch.kernels import RBFKernel as RBF, RQKernel as RQ, MaternKernel, LinearKernel
from models.svgp_auto_model_construction import SVGP

"""
NSG data

Do not adjust data for timelags.
"""
M = 168
training_iter = 200  # Increased iterations
learning_rate = 0.009  # Reduced for stability

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

def get_hyper(gp):
    """Extract hyperparameters based on kernel type"""
    results = {}
    results[f'outputscale'] = gp.covar_module.outputscale.item()
    if hasattr(gp.covar_module.base_kernel, 'kernels'):  # Additive kernel
        results['kernel_type'] = 'additive'
        for i, k in enumerate(gp.covar_module.base_kernel.kernels):
            results[f'kernel_{i}_name'] = k.__class__.__name__
            if hasattr(k, 'lengthscale'):
                results[f'kernel_{i}_lengthscales'] = k.lengthscale.squeeze().tolist()
            if hasattr(k, 'alpha'):
                results[f'kernel_{i}_alpha'] = k.alpha.item()
    else:
        results['kernel_type'] = 'single'
        results['kernel_name'] = gp.covar_module.base_kernel.__class__.__name__
        results['lengthscales'] = gp.covar_module.base_kernel.lengthscale.squeeze().tolist()
        if results['kernel_name'] == 'RQKernel':
            results['alpha'] = gp.covar_module.base_kernel.alpha.item()
    
    results['noise'] = gp.likelihood.noise.item()
    return results


file = PROCESSED_PATH / f'data{data_index}.xlsx'

""" 3. Load second trained expert """
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
expert_path = os.path.join(EXPERT_PATH, f'expert{data_index}.pth')
scaler_path = os.path.join(EXPERT_PATH, f'scaler{data_index}.pth')

# Load train expert
gp = torch.load(expert_path, weights_only=False)
scaler = torch.load(scaler_path, weights_only=False)

likelihood = GaussianLikelihood()

print(f'\nMain Kernel:\n {gp.covar_module.base_kernel}')

hyperparams = get_hyper(gp)

print('\nHyperparameters:')
for key, value in hyperparams.items():
    if not isinstance(value, list):
        print(f"{key}: {value}")

# feature importance
print('\nFeature Importance (sorted by lengthscale):')
if hyperparams['kernel_type'] == 'additive':
    for i in range(2):
        print(f'Kernel: {hyperparams[f'kernel_{i}_name']}')
        if i == 0:
            feature_importance0 = pd.DataFrame({
                'inputs': X_df.columns.values, 
                'lengthscales': hyperparams[f'kernel_{i}_lengthscales']
            })
            print(feature_importance0.sort_values(by='lengthscales'))
        else:
            feature_importance = pd.DataFrame({
                'inputs': X_df.columns.values, 
                'lengthscales': hyperparams[f'kernel_{i}_lengthscales']
            })
            print(feature_importance.sort_values(by='lengthscales'))

else:
    feature_importance = pd.DataFrame({
        'inputs': X_df.columns.values, 
        'lengthscales': hyperparams['lengthscales']
    })
    print('\nFeature Importance (sorted by lengthscale):')
    print(feature_importance.sort_values(by='lengthscales'))


""" 
Train GP
"""

# Enhanced inducing point initialization
import warnings
""" 2. Standardise outputs """

eval_perc = 0.2
N, D = np.shape(X)
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

y_train = torch.tensor(y_stand_np, dtype=floating_point).squeeze()
y_test = torch.tensor(y_test_stand, dtype=floating_point).squeeze()

init_ip_method = 'kmeans++'

if init_ip_method == 'random':
    indices = np.random.choice(N_train, min(M, N_train), replace=False)
    inducing_points = X_train[indices, :]
else:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Try multiple K-means initializations
        best_inertia = float('inf')
        best_centers = None
        for _ in range(5):  # Multiple attempts
            kmeans = KMeans(n_clusters=M, init='k-means++', n_init=10, random_state=None)
            kmeans.fit(X_train)
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_centers = kmeans.cluster_centers_
        inducing_points = torch.tensor(best_centers, dtype=floating_point)

print(f"Inducing points shape: {inducing_points.shape}")
print(f"K-means inertia: {best_inertia:.4f}")

# kernel
kernel = ScaleKernel(RQ(ard_num_dims=D, dtype=floating_point) + RBF(ard_num_dims=D, dtype=floating_point))
gp = SVGP(inducing_points, D, kernel)
likelihood = GaussianLikelihood(noise_constraint=Interval(1e-6, 0.1))  # Stricter noise constraint
gp.likelihood = likelihood
gp.to(floating_point)

# set initial hyperparameter
gp.covar_module.outputscale = torch.tensor(9.40, dtype=floating_point)
gp.covar_module.base_kernel.kernels[1].alpha = torch.tensor(6)
gp.covar_module.base_kernel.kernels[0].lengthscale = torch.tensor(feature_importance0['lengthscales'],
                                                                  dtype=floating_point)
gp.covar_module.base_kernel.kernels[1].lengthscale = torch.tensor(feature_importance['lengthscales'],
                                                                  dtype=floating_point)
likelihood.noise = torch.tensor(0.005, dtype=floating_point)

# Train model
gp.train()
gp.likelihood.train()

optimizer = torch.optim.Adam(gp.parameters(), lr=learning_rate)

# ELBO loss
mll = gpytorch.mlls.VariationalELBO(likelihood, gp,
                                    num_data=X_train.size(0))

for _ in range(training_iter):
    optimizer.zero_grad()
    output = gp(X_train)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()

# Predictions
gp.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(gp(X))

    # Unormalise predictions
    pred_mean = observed_pred.mean
    mu = scaler.inverse_transform(pred_mean.unsqueeze(1))[:,0]
    stds = scaler.inverse_transform(observed_pred.stddev.unsqueeze(1))[:,0]
    lower_stand, upper_stand = observed_pred.confidence_region()
    lower = scaler.inverse_transform(lower_stand.unsqueeze(1))[:,0]
    upper = scaler.inverse_transform(upper_stand.unsqueeze(1))[:,0]

    print(f'MSE(train-test): {mean_squared_error(y_processed,
                                                    mu)}')
    
""" Print Estimated Hyperparameter"""
hyperparams = get_hyper(gp)

print('\nHyperparameters:')
for key, value in hyperparams.items():
    if not isinstance(value, list):
        print(f"{key}: {value}")

# feature importance
print('\nFeature Importance (sorted by lengthscale):')
if hyperparams['kernel_type'] == 'additive':
    for i in range(2):
        print(f'Kernel: {hyperparams[f'kernel_{i}_name']}')
        if i == 0:
            feature_importance0 = pd.DataFrame({
                'inputs': X_df.columns.values, 
                'lengthscales': hyperparams[f'kernel_{i}_lengthscales']
            })
            print(feature_importance0.sort_values(by='lengthscales'))
        else:
            feature_importance = pd.DataFrame({
                'inputs': X_df.columns.values, 
                'lengthscales': hyperparams[f'kernel_{i}_lengthscales']
            })
            print(feature_importance.sort_values(by='lengthscales'))

else:
    feature_importance = pd.DataFrame({
        'inputs': X_df.columns.values, 
        'lengthscales': hyperparams['lengthscales']
    })
    print('\nFeature Importance (sorted by lengthscale):')
    print(feature_importance.sort_values(by='lengthscales'))

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

plt.title(f'Expert {data_index}')
ax.plot(date_time, y_processed, '*', color='green', label='Val')
# ax.plot(date_time, y_filtered, color='blue', label='Filtered')
# ax.plot(date_time[i_clean], y_clean, 'o', color='green', label='furnace')
ax.plot(date_time, mu, color='red', label='GP')
ax.vlines(x=date_time[end_indx], ymin=0, ymax=max(y_processed),
        colors='black', ls='--', label='Test-data')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

plt.show()