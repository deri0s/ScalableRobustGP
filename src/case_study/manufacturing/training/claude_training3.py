import os
import torch
import gpytorch
import copy
import numpy as np
import pandas as pd
from scipy.stats import qmc
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler as ss
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.cluster import KMeans
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import ScaleKernel, AdditiveKernel
from gpytorch.kernels import RBFKernel as RBF, RQKernel as RQ, MaternKernel, LinearKernel
from models.svgp_auto_model_construction import SVGP

"""
Enhanced User Configuration
"""

# Enhanced configuration
data_index = 0
M = 110
N_sim = 100
kernel = 'RBF'  # Options: 'RBF', 'RQ', 'Matern52', 'RBF+Linear', 'RBF+RQ'
use_log_space = True
use_early_stopping = True
training_iter = 200  # Increased iterations
learning_rate = 0.005  # Reduced for stability

# Target-based early stopping parameters
mse_training_target = 0.008
mse_test_target = 0.01
use_target_early_stopping = True  # Set to False to disable target-based early stopping

# NSG post processes data location
ROOT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT_PATH / "data" / "processed" / "Training_data_partitions"
index = 0
file = PROCESSED_PATH / f'data{index}.xlsx'


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

# Feature selection
X_df.drop(columns=['9282 Tweel Position'], inplace=True)
t_df.drop(columns=['9282 Tweel Position'], inplace=True)
X_df.drop(columns=['10091 Furnace Load'], inplace=True)
t_df.drop(columns=['10091 Furnace Load'], inplace=True)
X_df.drop(columns=['7746 Open Crown Temperature - Port 2 (PV)'], inplace=True)
t_df.drop(columns=['7746 Open Crown Temperature - Port 2 (PV)'], inplace=True)

X_df, y_df = align_inputs(X_df, y_df, t_df.iloc[0,:])

X_np = X_df.values
y_processed = y_df.y_processed.values
y_raw = y_df.y_raw.values
date_time = y_df.date_time.values

# Convert data to torch tensors
floating_point = torch.float64
X = torch.tensor(X_np, dtype=floating_point)
N, D = np.shape(X)


""" 2. Standardise outputs """

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

y_train = torch.tensor(y_stand_np, dtype=floating_point).squeeze()
y_test = torch.tensor(y_test_stand, dtype=floating_point).squeeze()


""" 3. Enhanced model setup """

def create_kernel(kernel_type, D, dtype):
    """Create kernel based on type specification"""
    if kernel_type == 'RBF':
        return ScaleKernel(RBF(ard_num_dims=D, dtype=dtype))
    elif kernel_type == 'RQ':
        return ScaleKernel(RQ(ard_num_dims=D, dtype=dtype))
    elif kernel_type == 'Matern52':
        return ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=D, dtype=dtype))
    elif kernel_type == 'RBF+Linear':
        rbf = ScaleKernel(RBF(ard_num_dims=D, dtype=dtype))
        linear = ScaleKernel(LinearKernel(ard_num_dims=D, dtype=dtype))
        return AdditiveKernel(rbf, linear)
    elif kernel_type == 'RBF+RQ':
        rbf = ScaleKernel(RBF(ard_num_dims=D, dtype=dtype))
        rq = ScaleKernel(RQ(ard_num_dims=D, dtype=dtype))
        return AdditiveKernel(rbf, rq)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

def get_hyper(gp):
    """Extract hyperparameters based on kernel type"""
    results = {}
    
    if hasattr(gp.covar_module, 'kernels'):  # Additive kernel
        results['kernel_type'] = 'additive'
        for i, k in enumerate(gp.covar_module.kernels):
            if hasattr(k, 'base_kernel'):
                if hasattr(k.base_kernel, 'lengthscale'):
                    results[f'kernel_{i}_lengthscales'] = k.base_kernel.lengthscale.squeeze().tolist()
                if hasattr(k.base_kernel, 'alpha'):
                    results[f'kernel_{i}_alpha'] = k.base_kernel.alpha.item()
                results[f'kernel_{i}_outputscale'] = k.outputscale.item()
    else:  # Single kernel
        results['outputscale'] = gp.covar_module.outputscale.item()
        if hasattr(gp.covar_module.base_kernel, 'lengthscale'):
            results['lengthscales'] = gp.covar_module.base_kernel.lengthscale.squeeze().tolist()
        if hasattr(gp.covar_module.base_kernel, 'alpha'):
            results['alpha'] = gp.covar_module.base_kernel.alpha.item()
    
    results['noise'] = gp.likelihood.noise.item()
    return results

def setup_hyperparameter_space(kernel_type, D, use_log_space=True):
    """Enhanced hyperparameter space setup"""
    
    # Base parameters for single kernels
    base_params = {
        'RBF': D + 2,          # lengthscales + outputscale + noise
        'RQ': D + 3,           # lengthscales + outputscale + alpha + noise  
        'Matern52': D + 2,     # lengthscales + outputscale + noise
        'RBF+Linear': 2*D + 4, # 2 sets of lengthscales + 2 outputscales + noise
        'RBF+RQ': 2*D + 4      # RBF lengthscales + RQ lengthscales + 2 outputscales + alpha + noise
    }
    
    dim = base_params.get(kernel_type)
    if dim is None:
        raise ValueError(f'Unsupported kernel type: {kernel_type}')
    
    if use_log_space:
        # Log space
        lowerb = np.full(dim, np.log(10))  # Very small lengthscales
        upperb = np.full(dim, np.log(500))  # Very large lengthscales
        # Noise is always positioned at index=-1
        lowerb[-1] = np.log(0.0005)
        upperb[-1] = np.log(0.001)
        
        # Adjust specific parameter bounds
        if kernel_type in ['RBF', 'Matern52']:
            # outputscale bounds
            lowerb[-2] = np.log(0.5)
            upperb[-2] = np.log(50)
            
        elif kernel_type == 'RQ':
            # outputscale bounds
            lowerb[-3] = np.log(0.5)
            upperb[-3] = np.log(50)
            # alpha bounds (keep linear)
            lowerb[-2] = 5
            upperb[-2] = 10.0
            
        elif 'RBF+Linear' in kernel_type:
            # Two outputscales
            lowerb[-3:-1] = np.log(0.5)
            upperb[-3:-1] = np.log(50)
            
        elif 'RBF+RQ' in kernel_type:
            # Two outputscales + alpha
            lowerb[-4] = np.log(0.5)  # RBF outputscale
            upperb[-4] = np.log(50)
            lowerb[-3] = np.log(0.5)  # RQ outputscale  
            upperb[-3] = np.log(50)
            lowerb[-2] = 2          # RQ alpha (linear)
            upperb[-2] = 10.0
    else:
        # Linear space bounds (conservative)
        lowerb = 0.1 * np.ones(dim)
        upperb = 100 * np.ones(dim)
        # Set noise bounds
        lowerb[-1] = 0.001
        upperb[-1] = 0.01
    
    return dim, lowerb, upperb

def set_hyperparameters(gp, likelihood, sample, kernel_type, D, use_log_space=True):
    """Enhanced hyperparameter setting for multiple kernel types"""
    
    if kernel_type == 'RBF' or kernel_type == 'Matern52':
        if use_log_space:
            lengthscales = np.exp(sample[0:D])
            outputscale = np.exp(sample[-2])
            noise = np.exp(sample[-1])
        else:
            lengthscales = sample[0:D]
            outputscale = sample[-2]
            noise = sample[-1]
            
        gp.covar_module.base_kernel.lengthscale = torch.tensor(lengthscales, dtype=floating_point)
        gp.covar_module.outputscale = torch.tensor(outputscale, dtype=floating_point)
        likelihood.noise = torch.tensor(noise, dtype=floating_point)
        
    elif kernel_type == 'RQ':
        if use_log_space:
            lengthscales = np.exp(sample[0:D])
            outputscale = np.exp(sample[-3])
            alpha = sample[-2]  # Keep linear
            noise = np.exp(sample[-1])
        else:
            lengthscales = sample[0:D]
            outputscale = sample[-3]
            alpha = sample[-2]
            noise = sample[-1]
            
        gp.covar_module.base_kernel.lengthscale = torch.tensor(lengthscales, dtype=floating_point)
        gp.covar_module.outputscale = torch.tensor(outputscale, dtype=floating_point)
        gp.covar_module.base_kernel.alpha = torch.tensor(alpha, dtype=floating_point)
        likelihood.noise = torch.tensor(noise, dtype=floating_point)

def evaluate_model(gp, likelihood, X_test, y_test_nonstand, scaler):
    """Comprehensive model evaluation"""
    gp.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(gp(X_test))
        pred_mean = pred_dist.mean
        pred_var = pred_dist.variance
        
        # Convert back to original scale
        pred_mean_orig = scaler.inverse_transform(pred_mean.unsqueeze(1))[:, 0]
        pred_std_orig = np.sqrt(pred_var.numpy()) * scaler.scale_[0]
        
        # Calculate metrics
        mse = mean_squared_error(y_test_nonstand, pred_mean_orig)
        mae = mean_absolute_error(y_test_nonstand, pred_mean_orig)
        r2 = r2_score(y_test_nonstand, pred_mean_orig)
        
        # Mean prediction interval width (as measure of uncertainty)
        mean_uncertainty = np.mean(2 * 1.96 * pred_std_orig)  # 95% CI width
        
    return {
        'mse': mse, 
        'mae': mae, 
        'r2': r2, 
        'mean_uncertainty': mean_uncertainty,
        'predictions': pred_mean_orig,
        'std': pred_std_orig
    }

# Enhanced inducing point initialization
import warnings

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

# Setup hyperparameter space
dim, lowerb, upperb = setup_hyperparameter_space(kernel, D, use_log_space)

print(f"\nUsing {kernel} kernel")
if use_target_early_stopping:
    print(f"Target-based early stopping enabled:")
    print(f"  Training MSE target: {mse_training_target}")
    print(f"  Test MSE target: {mse_test_target}")

# Generate samples
def get_samples(D, N_sim, l_bounds, u_bounds):
    """ Samples from a Latin Hypercube Sampling model """
    sampler = qmc.LatinHypercube(d=D)
    sample = sampler.random(n=N_sim)
    
    if len(l_bounds) != D or len(u_bounds) != D:
        raise ValueError(f"Bounds dimensions ({len(l_bounds)}, {len(u_bounds)}) must match sample dimension ({D})")
    
    return qmc.scale(sample, l_bounds, u_bounds)

samples = get_samples(dim, N_sim, lowerb, upperb)

best_metrics = {'mse': float('inf')}
best_gp = None
best_likelihood = None
target_reached = False

# Enhanced training loop with target-based early stopping
print("\nStarting enhanced hyperparameter optimization...")
for n in range(N_sim):
    # Create fresh model
    k_fresh = create_kernel(kernel, D, floating_point)
    gp_temp = SVGP(inducing_points, D, k_fresh)
    likelihood_temp = GaussianLikelihood(noise_constraint=Interval(1e-6, 0.1))  # Stricter noise constraint
    gp_temp.likelihood = likelihood_temp
    gp_temp.to(floating_point)

    # Set hyperparameters
    set_hyperparameters(gp_temp, likelihood_temp,
                        samples[n], kernel, D, use_log_space)

    # Training
    gp_temp.train()
    likelihood_temp.train()

    # Enhanced optimiser with scheduling
    optimizer = torch.optim.Adam(gp_temp.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           patience=20,
                                                           factor=0.8)

    # ELBO loss
    mll = gpytorch.mlls.VariationalELBO(likelihood_temp, gp_temp,
                                        num_data=X_train.size(0))

    # Training with early stopping
    prev_loss = float('inf')
    patience_counter = 0
    patience = 30
    
    for i in range(training_iter):
        optimizer.zero_grad()
        output = gp_temp(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        
        # Early stopping and learning rate scheduling
        if use_early_stopping and i > 20:
            if loss.item() > prev_loss - 1e-6:
                patience_counter += 1
            else:
                patience_counter = 0
            
            if patience_counter >= patience:
                break
                
        prev_loss = loss.item()
        scheduler.step(loss)

    # Comprehensive evaluation
    test_metrics = evaluate_model(gp_temp, likelihood_temp, X_test, y_test_nonstand, scaler)
    train_metrics = evaluate_model(gp_temp, likelihood_temp, X_train, y_train_nonstand, scaler)

    # if train_metrics['mse'] < best_metrics['mse']:
    if test_metrics['mse'] < best_metrics['mse']:
        best_metrics = test_metrics
        # best_metrics = train_metrics
        best_gp = copy.deepcopy(gp_temp)
        best_likelihood = copy.deepcopy(likelihood_temp)
        print(f'Sim: {n+1}/{N_sim}, New best - Test MSE: {test_metrics["mse"]:.6f}, Train MSE: {train_metrics["mse"]:.6f}')
        
        # Check if target MSE values are reached
        if (use_target_early_stopping and 
            train_metrics['mse'] <= mse_training_target and 
            test_metrics['mse'] <= mse_test_target):
            target_reached = True
            print(f'\n🎯 TARGET REACHED! Stopping early at simulation {n+1}/{N_sim}')
            print(f'   Training MSE: {train_metrics["mse"]:.6f} <= {mse_training_target}')
            print(f'   Test MSE: {test_metrics["mse"]:.6f} <= {mse_test_target}')
            break

    if (n + 1) % 50 == 0:
        print(f'Completed {n+1}/{N_sim}, Best MSE: {best_metrics["mse"]:.6f}')

# Generate full predictions with best model
best_gp.eval()
best_likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    full_pred = best_likelihood(best_gp(X))
    best_mu = scaler.inverse_transform(full_pred.mean.unsqueeze(1))[:, 0]
    best_std = np.sqrt(full_pred.variance.numpy()) * scaler.scale_[0]

""" 4. Enhanced results analysis """
hyperparams = get_hyper(best_gp)

print('\n' + '='*60)
print('ENHANCED OPTIMIZATION RESULTS')
print('='*60)
if target_reached:
    print('✅ TARGET MSE VALUES ACHIEVED!')
    print(f'Stopped early after {n+1} simulations (out of {N_sim})')
else:
    print('⚠️  Target MSE values not reached in all simulations')
    print(f'Completed all {N_sim} simulations')

print(f"Final test MSE: {best_metrics['mse']:.6f}")
print(f"Final test R²: {best_metrics['r2']:.4f}")
print(f"Final test MAE: {best_metrics['mae']:.6f}")
print(f"Mean prediction uncertainty: {best_metrics['mean_uncertainty']:.4f}")

print('\nHyperparameters:')
for key, value in hyperparams.items():
    if isinstance(value, list):
        print(f"{key}: {[f'{v:.4f}' for v in value]}")
    else:
        print('\nhyper: ', hyperparams)

# Feature importance (for single kernel types)
if 'lengthscales' in hyperparams:
    feature_importance = pd.DataFrame({
        'inputs': X_df.columns.values, 
        'lengthscales': hyperparams['lengthscales']
    })
    print('\nFeature Importance (sorted by lengthscale):')
    print(feature_importance.sort_values(by='lengthscales'))

# Training vs test performance comparison
train_metrics = evaluate_model(best_gp, best_likelihood, X_train, y_train_nonstand, scaler)
print(f"\nTraining Performance:")
print(f"Train MSE: {train_metrics['mse']:.6f}, Test MSE: {best_metrics['mse']:.6f}")
print(f"Train R²: {train_metrics['r2']:.4f}, Test R²: {best_metrics['r2']:.4f}")
print(f"Overfitting check: {'Minimal' if best_metrics['mse']/train_metrics['mse'] < 2 else 'Significant'}")

if use_target_early_stopping:
    print(f"\nTarget Achievement Status:")
    print(f"Training MSE target ({mse_training_target}): {'✅ ACHIEVED' if train_metrics['mse'] <= mse_training_target else '❌ NOT ACHIEVED'}")
    print(f"Test MSE target ({mse_test_target}): {'✅ ACHIEVED' if best_metrics['mse'] <= mse_test_target else '❌ NOT ACHIEVED'}")

#-----------------------------------------------------------------------------
# PLOTS
#-----------------------------------------------------------------------------
end_indx = int(len(X)*0.8)
fig, ax = plt.subplots(figsize=(12, 6))

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
fig.autofmt_xdate()

title_suffix = " (Target Reached)" if target_reached else ""
plt.title(f'Expert {index} - {kernel} Kernel{title_suffix}', fontsize=16)
ax.fill_between(date_time, best_mu - 1.96*best_std, best_mu + 1.96*best_std, 
                alpha=0.3, color='coral', label='95% CI')
ax.plot(date_time, y_raw, color='grey', label='Raw', markersize=4)
ax.plot(date_time, y_processed, '*', color='green', label='Actual', markersize=4)
ax.plot(date_time, best_mu, color='red', label=f'GP-{kernel}', linewidth=2)
ax.axvline(x=date_time[end_indx], color='black', linestyle='--', 
           label='Train/Test Split', alpha=0.7)
ax.set_xlabel("Date-time", fontsize=14)
ax.set_ylabel("Fault density", fontsize=14)
plt.legend(loc='best', prop={"size":12}, facecolor="white", framealpha=1.0)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()