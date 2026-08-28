import os
import copy
import numpy as np
import pandas as pd
import torch
import gpytorch

# Configure device (defaults to CUDA if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

from scipy.stats import qmc
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler as ss

import matplotlib
matplotlib.use('QtAgg')
from matplotlib import pyplot as plt

from pathlib import Path
from sklearn.cluster import KMeans
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import ScaleKernel, AdditiveKernel
from gpytorch.kernels import RBFKernel as RBF

"""
Enhanced User Configuration
"""

# Enhanced configuration
data_index = 1  # expert3: Not working RQ
M = 168  # Number of inducing points -- UNUSED with exact GP; kept for a future SVGP toggle
N_sim = 200  # Number of Monte-Carlo posterior samples drawn in section 9
kernel = 'RBF'  # Options: 'RBF', 'RQ', 'Matern52', 'RBF+RQ'
use_log_space = True  # NOTE: not yet wired into preprocessing below (see section 2) -- reserved flag
use_early_stopping = True
training_iter = 300
learning_rate = 0.001  # Reduced for stability

# Target-based early stopping parameters
mse_training_target = 0.009  # 0.006
mse_test_target = 0.008  # 0.008
use_target_early_stopping = True  # Set to False to disable target-based early stopping, uses patience instead
patience = 20  # Used only when use_target_early_stopping is False
eval_every = 5  # How often (in iterations) to compute train/test MSE

# NSG post processes data location
ROOT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT_PATH / "data" / "processed" / "Training_data_partitions"
EXPERT_PATH = ROOT_PATH / "trained" / "experts"

file = PROCESSED_PATH / f'data{data_index}.xlsx'


""" 1. Apply the corresponding timelags """
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
y_raw = y_df.y_raw.values
date_time = y_df.date_time.values

# Convert data to torch tensors and move to GPU
floating_point = torch.float64
X = torch.tensor(X_np, dtype=floating_point, device=device)
N, D = np.shape(X_np)
print('\nTraining data size\n', N, D)


""" 2. Standardise outputs """
eval_perc = 0.2
end_train = N - int(N * eval_perc)

X_train = X[0:end_train]
date_train = date_time[0:end_train]
N_train = len(X_train)
y_train_nonstand = y_processed[0:end_train]

# Define X_test and y_test_nonstand correctly for evaluation metric
X_test = X[end_train:N]
y_test_nonstand = y_processed[end_train:N]

# Standardise outputs (done on CPU with Scikit-Learn)
y_train_reshape = y_train_nonstand.reshape(-1, 1)
scaler = ss()
scaler.fit(y_train_reshape)
y_stand_np = scaler.transform(y_train_reshape)

y_test_reshape = y_test_nonstand.reshape(-1, 1)
y_test_stand = scaler.transform(y_test_reshape)

# Convert standardized outputs to PyTorch tensors on GPU
y_train = torch.tensor(y_stand_np, dtype=floating_point, device=device).squeeze()
y_test = torch.tensor(y_test_stand, dtype=floating_point, device=device).squeeze()

date_test = date_time[end_train:N]


def compute_multi_scale_ls(X_train, subsample_size=1000):
    """Compute lengthscales for different scales using proper distance analysis"""
    X_np = X_train.numpy() if isinstance(X_train, torch.Tensor) else X_train
    n_samples, n_dims = X_np.shape
    
    if n_samples > subsample_size:
        indices = np.random.choice(n_samples, subsample_size, replace=False)
        X_subset = X_np[indices]
    else:
        X_subset = X_np
    
    # Compute all pairwise distances
    all_distances = []
    for dim in range(n_dims):
        dim_data = X_subset[:, dim:dim+1]
        pairwise_dists = pdist(dim_data, metric='euclidean')
        non_zero_dists = pairwise_dists[pairwise_dists > 1e-10]
        all_distances.extend(non_zero_dists)
    
    all_distances = np.array(all_distances)
    
    # Use percentiles of ALL distances, not median distances
    short_scale = np.percentile(all_distances, 25)  # For RBF (short-range)
    long_scale = np.percentile(all_distances, 75)   # For RQ (long-range)
    
    return short_scale, long_scale


""" 3. Define the Exact GP Regression Model (RBF kernel, no sparse approximation) """
class ExactGPModel(gpytorch.models.ExactGP):
    """
    Exact GP regression with an RBF (squared-exponential) kernel.

    We use gpytorch.models.ExactGP rather than an SVGP/inducing-point
    model: with N_train ~ 3.7k and a GPU available, exact inference via
    GPyTorch's GPU-accelerated conjugate-gradient solves (BBMM) is both
    tractable and gives the true (non-approximate) posterior, so there's
    no accuracy/speed reason to introduce inducing points here.

    An ARD lengthscale (one per input dimension) is used since the D=14
    input features are unlikely to share a single characteristic scale.
    """
    def __init__(self, train_x, train_y, likelihood, ard_num_dims=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_kernel = RBF(ard_num_dims=ard_num_dims)
        self.covar_module = ScaleKernel(base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


""" 4. Instantiate model & likelihood, initialise hyperparameters """
# Add .double() to both to match your float64 tensors
likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-4)).to(device).double()
model = ExactGPModel(X_train, y_train, likelihood, ard_num_dims=D).to(device).double()

# Data-driven hyperparameter initialisation
short_ls, long_ls = compute_multi_scale_ls(X_train.cpu())
with torch.no_grad():
    model.covar_module.base_kernel.lengthscale = float(short_ls)
    model.covar_module.outputscale = y_train.var().item()
    likelihood.noise = 0.05

print(f"\nInitial hyperparameters -> lengthscale: {short_ls:.4f}, "
      f"outputscale: {y_train.var().item():.4f}, noise: 0.05")


""" 5. Train the GP (marginal log-likelihood optimisation, with early stopping) """
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

history = {'iter': [], 'loss': [], 'train_mse': [], 'test_mse': []}
best_test_mse = float('inf')
best_state = copy.deepcopy(model.state_dict())
patience_counter = 0

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(X_train)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()

    is_last_iter = (i == training_iter - 1)
    if (i + 1) % eval_every == 0 or is_last_iter:
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            train_pred = likelihood(model(X_train)).mean
            test_pred = likelihood(model(X_test)).mean
            train_mse = torch.mean((train_pred - y_train) ** 2).item()
            test_mse = torch.mean((test_pred - y_test) ** 2).item()
        model.train()
        likelihood.train()

        history['iter'].append(i + 1)
        history['loss'].append(loss.item())
        history['train_mse'].append(train_mse)
        history['test_mse'].append(test_mse)

        print(f"Iter {i + 1:>4}/{training_iter} | Loss: {loss.item():.4f} | "
              f"Train MSE: {train_mse:.5f} | Test MSE: {test_mse:.5f} | "
              f"Noise: {likelihood.noise.item():.5f}")

        if use_early_stopping:
            if use_target_early_stopping:
                # Stop as soon as both targets are simultaneously satisfied
                if train_mse <= mse_training_target and test_mse <= mse_test_target:
                    print(f"\nTarget-based early stopping at iter {i + 1}: "
                          f"train_mse={train_mse:.5f} <= {mse_training_target}, "
                          f"test_mse={test_mse:.5f} <= {mse_test_target}")
                    break
            else:
                # Fallback: patience on test MSE, restore best checkpoint on stop
                if test_mse < best_test_mse - 1e-6:
                    best_test_mse = test_mse
                    best_state = copy.deepcopy(model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"\nPatience-based early stopping at iter {i + 1} "
                              f"(best test MSE {best_test_mse:.5f}); restoring best checkpoint.")
                        model.load_state_dict(best_state)
                        break


""" 6. Evaluate on held-out test data """
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_dist = likelihood(model(X_test))
    y_pred_stand = test_dist.mean
    lower_stand, upper_stand = test_dist.confidence_region()  # 95% CI

y_test_np = y_test.cpu().numpy()
y_pred_np = y_pred_stand.cpu().numpy()

# Metrics on the standardized scale (this is the space the loss/targets operate in)
mse_stand = mean_squared_error(y_test_np, y_pred_np)
mae_stand = mean_absolute_error(y_test_np, y_pred_np)
r2_stand = r2_score(y_test_np, y_pred_np)

# Metrics on the original (de-standardized) fault-density scale
y_pred_orig = scaler.inverse_transform(y_pred_np.reshape(-1, 1)).ravel()
y_test_orig = scaler.inverse_transform(y_test_np.reshape(-1, 1)).ravel()
lower_orig = scaler.inverse_transform(lower_stand.cpu().numpy().reshape(-1, 1)).ravel()
upper_orig = scaler.inverse_transform(upper_stand.cpu().numpy().reshape(-1, 1)).ravel()

mse_orig = mean_squared_error(y_test_orig, y_pred_orig)
mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
r2_orig = r2_score(y_test_orig, y_pred_orig)

print("\n=== Test-set performance ===")
print(f"Standardized space -> MSE: {mse_stand:.5f} | MAE: {mae_stand:.5f} | R2: {r2_stand:.4f}")
print(f"Original scale     -> MSE: {mse_orig:.5f} | MAE: {mae_orig:.5f} | R2: {r2_orig:.4f}")

print("\n=== Learned hyperparameters ===")
print(f"Outputscale: {model.covar_module.outputscale.item():.4f}")
print(f"Noise: {likelihood.noise.item():.6f}")
learned_ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
feature_names = list(X_df.columns)
for name, l in zip(feature_names, learned_ls):
    print(f"  Lengthscale [{name}]: {l:.4f}")


""" 7. Save the trained model """
EXPERT_PATH.mkdir(parents=True, exist_ok=True)
model_save_path = EXPERT_PATH / f"expert_{data_index}_gp_rbf.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'likelihood_state_dict': likelihood.state_dict(),
    'scaler': scaler,
    'config': {
        'kernel': kernel,
        'D': D,
        'N_train': N_train,
        'ard_num_dims': D,
    }
}, model_save_path)
print(f"\nModel saved to {model_save_path}")


""" 8. Posterior sampling (Monte-Carlo simulation draws) """
# Draws N_sim joint samples from the test-set posterior -- useful downstream
# if this expert model feeds into a larger Monte-Carlo / mixture-of-experts
# simulation pipeline.
with torch.no_grad(), gpytorch.settings.fast_pred_samples():
    posterior_samples_stand = test_dist.rsample(torch.Size([N_sim]))  # [N_sim, N_test]

posterior_samples_orig = scaler.inverse_transform(
    posterior_samples_stand.cpu().numpy().reshape(-1, 1)
).reshape(N_sim, -1)
print(f"\nDrew {N_sim} posterior simulation samples over the {posterior_samples_orig.shape[1]} test points.")


""" 9. Plots """
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

# 9a. Raw vs processed training signal
fig1, ax1 = plt.subplots(figsize=(12, 6))
fig1.autofmt_xdate()
plt.title(f'Expert {data_index}', fontsize=16)
ax1.plot(date_time, y_raw, color='grey', label='Raw', markersize=4)
ax1.plot(date_time, y_processed, '*', color='green', label='Actual', markersize=4)
ax1.set_xlabel("Date-time", fontsize=14)
ax1.set_ylabel("Fault density", fontsize=14)
plt.legend(loc='best', prop={"size": 12}, facecolor="white", framealpha=1.0)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 9b. GP predictions vs actual on the held-out test set, with 95% CI
fig2, ax2 = plt.subplots(figsize=(12, 6))
fig2.autofmt_xdate()
plt.title(f'Expert {data_index} - GP Regression (RBF) Test Predictions', fontsize=16)
ax2.plot(date_test, y_test_orig, '*', color='green', label='Actual', markersize=4)
ax2.plot(date_test, y_pred_orig, color='blue', label='GP mean prediction')
ax2.fill_between(date_test, lower_orig, upper_orig, alpha=0.3, color='blue', label='95% CI')
ax2.set_xlabel("Date-time", fontsize=14)
ax2.set_ylabel("Fault density", fontsize=14)
plt.legend(loc='best', prop={"size": 12}, facecolor="white", framealpha=1.0)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 9c. Training curve: MLL loss and train/test MSE vs iteration
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))
ax3a.plot(history['iter'], history['loss'], color='black')
ax3a.set_xlabel("Iteration", fontsize=14)
ax3a.set_ylabel("Negative MLL Loss", fontsize=14)
ax3a.set_title("Training loss", fontsize=14)
ax3a.grid(True, alpha=0.3)

ax3b.plot(history['iter'], history['train_mse'], label='Train MSE', color='tab:blue')
ax3b.plot(history['iter'], history['test_mse'], label='Test MSE', color='tab:orange')
ax3b.axhline(mse_training_target, color='tab:blue', linestyle='--', alpha=0.5, label='Train target')
ax3b.axhline(mse_test_target, color='tab:orange', linestyle='--', alpha=0.5, label='Test target')
ax3b.set_xlabel("Iteration", fontsize=14)
ax3b.set_ylabel("MSE (standardized)", fontsize=14)
ax3b.set_title("Train / test MSE", fontsize=14)
ax3b.legend(loc='best', fontsize=10)
ax3b.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()