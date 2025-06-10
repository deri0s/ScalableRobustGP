# Set before importing any libraries
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import gpytorch
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler as ss
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist # _z plot
# GPyTorch imports
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel
from gpytorch.kernels import RBFKernel as RBF, RQKernel as RQ, PeriodicKernel as Per, LinearKernel as Lin
from gpytorch.constraints import GreaterThan # For noise constraint
import traceback # For detailed error printing
# Automatic Model Construction
from models.svgp_auto_model_construction import GPTraining, SVGP

"""
NSG data
"""

# NSG post processes data location
ROOT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT_PATH / "data" / "processed" / "Training_data_partitions"
file_name = 'data1.xlsx'
file_path = PROCESSED_PATH / file_name

# Training df
X_df = pd.read_excel(file_path, sheet_name='X_stand')
y_df = pd.read_excel(file_path, sheet_name='y_nonstand')
t_df = pd.read_excel(file_path, sheet_name='timelags')

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

X_df, y_df = align_inputs(X_df, y_df, t_df.iloc[0,:])

"""---------------------------------------------------------------------------
    STANDARDISE TRAINING & TEST DATA
"""
data_ratio = {'training': 0.70, 'eval': 0.15, 'test': 0.15}

if X_df.empty or y_df.empty:
    raise ValueError("DataFrames are empty after alignment. Check lagging procedure and input data.")

X = X_df.values
y_all_nonstand, date_time = y_df.y_processed.values, y_df.date_time.values

N, D = np.shape(X)

# Training data
end_train = int(data_ratio['training']*N)
X_train_np = X[0:end_train]
date_train = date_time[0:end_train]
N_train = len(X_train_np)
y_train_nonstand = y_all_nonstand[0:end_train]

# Validation data
end_val = end_train + int(data_ratio['eval']*N)
X_eval_np = X[end_train: end_val]
y_test_nonstand = y_all_nonstand[end_train: end_val]

# Test data
X_test_np = X[end_val: N]
y_test_nonstand = y_all_nonstand[end_val: N]

dt_all = date_time[0:N]

# # --- Input Validation: Check if train/test splits are valid ---
# if N_train <= 0 or len(X_test_np) <= 0:
#     raise ValueError(f"Training size ({N_train}) or Test size ({len(X_test_np)}) is zero or negative. Check data splitting.")
# if len(y_test_nonstand) != len(X_test_np):
#      raise ValueError(f"Test features ({len(X_test_np)}) and test targets ({len(y_test_nonstand)}) lengths differ.")

# # Standardise outputs
# y_train_reshape = y_train_nonstand.reshape(-1,1)
# scaler = ss()
# scaler.fit(y_train_reshape)
# y_stand_np = scaler.transform(y_train_reshape)

# y_test_reshape = y_test_nonstand.reshape(-1,1)
# scaler.fit(y_test_reshape)
# y_test_stand = scaler.transform(y_test_reshape)

# # Convert data to torch tensors
# floating_point = torch.float64
# X_train = torch.tensor(X_train_np, dtype=floating_point)
# y_train = torch.tensor(y_stand_np, dtype=floating_point).squeeze()
# X_test = torch.tensor(X_test_np, dtype=floating_point)
# y_test = torch.tensor(y_test_stand, dtype=floating_point).squeeze()
# X_all = torch.tensor(X, dtype=floating_point) # Full X for final prediction/plot

# # ============================================================================
# # --- Application Code ---
# # ============================================================================
# import warnings
# M = 120 # Number of inducing points
# init_ip_method = 'kmeans++'

# if init_ip_method == 'random':
#     indices = np.random.choice(N_train, min(M, N_train), replace=False)
#     inducing_points =  X_train[indices, :]
#     inducing_points = X_train[np.random.choice(N_train, M, replace=False), :]
# else:
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore")
#         kmeans = KMeans(n_clusters=M, init='k-means++', n_init=10)
#         kmeans.fit(X_train_np)
#         inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=floating_point)

# # Create Initial ApproximateGP Model
# likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-5))

# # choose any kernel to specify ARD and floating-point
# RQ(ard_num_dims=D, dtype=floating_point)
# k = ScaleKernel(RQ(ard_num_dims=D, dtype=floating_point))

# gp0 = SVGP(inducing_points, D, k)
# gp0.likelihood = likelihood
# gp0.likelihood.noise = torch.tensor(0.028, dtype=floating_point)
# gp0.to(floating_point)

# # --- Automatic Model Construction ---
# print("\n--- Starting Automatic Model Construction ---")
# start_time = time.time()
# auto_trainer = GPTraining(gp0, X_train, y_train, X_test, y_test)

# # Define parameter limits for the grid_search - Updated for level=3 complexity
# limits = {
#     'outputscale': [0.1, 10.0],       # Limits for ScaleKernel outputscale
#     'se_lengthscale': [0.05, 100.0],  # Limits for RBF lengthscale (same range for all dims)
#     'rq_lengthscale': [0.05, 100.0],  # Limits for RQ lengthscale (same range for all dims)
#     'rq_alpha': [0.05, 5],            # Limits for RQ alpha
#     'per_period_length': [6, 10],      # Limits for Per Period
#     'per_lengthscale': [0.05, 100.0],  # Limits for Per lengthscale (same range for all dims)
#     'lin_variance': [0.1*0.025, 0.025], # Limits for Lin variance: the variability of the input features in their relationship to the target.
#     'noise_variance': [0.025, 0.028]  # Limits for Likelihood noise
# }

# # Automatic Model Construction: Grid search parameters
# try:
#     gp_gs = auto_trainer.auto_model_cons(
#         levels=1,
#         N_sim=5,
#         param_limits=limits,
#         mse_stop=0.045,
#         lr=0.009,
#         training_iterations=80,
#         batch_size=512)
#     print("✓ Automatic Model Construction completed successfully")
# except Exception as e:
#     print(f"❌ Error during Automatic Model Construction: {e}")
#     print("Using initial kernel")
#     traceback.print_exc()
#     # Fallback to level=2 if level=3 fails
#     try:
#         gp_gs = auto_trainer.auto_model_cons(
#             levels=2,
#             N_sim=5,
#             param_limits=limits,
#             mse_stop=0.055,
#             lr=0.01,
#             training_iterations=50,
#             batch_size=512)
#     except Exception as e2:
#         print(f"❌ Fallback also failed: {e2}")
#         gp_gs = gp0  # Use original model if all fails
#         print("Using original model (gp0)")

# gp_gs.eval()
# gp_gs.likelihood.eval()

# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     observed_pred = likelihood(gp_gs(X_all))
# ctime = time.time() - start_time

# print(f'computational time: {ctime:.6}')

# # Enhanced kernel parameter extraction for complex models
# def extract_kernel_info(model):
#     kernel_info = {}
#     base_kernel = model.covar_module.base_kernel
    
#     print('\n--Estimated Model and kernel parameters--')
#     print(f'Outputscale: {model.covar_module.outputscale.item():.4f}')
    
#     # Handle different kernel types and combinations
#     if hasattr(base_kernel, 'kernels'):  # Additive or Multiplicative kernel
#         print(f"Complex kernel with {len(base_kernel.kernels)} components:")
#         for i, kernel in enumerate(base_kernel.kernels):
#             print(f"  Component {i+1}: {type(kernel).__name__}")
#             if isinstance(kernel, (RBF, RQ)):
#                 lengthscales = kernel.lengthscale.squeeze()
#                 kernel_info[f'component_{i}_lengthscales'] = lengthscales
#                 print(f"    Lengthscales: {lengthscales.tolist()}")
#                 if isinstance(kernel, RQ):
#                     print(f"    Alpha: {kernel.alpha.item():.4f}")
#             elif isinstance(kernel, Per):
#                 print(f"    Period: {kernel.period_length.squeeze().tolist()}")
#                 print(f"    Lengthscales: {kernel.lengthscale.squeeze().tolist()}")
#             elif isinstance(kernel, Lin):
#                 print(f"    Variance: {kernel.variance.squeeze().tolist()}")
#     else:  # Single kernel
#         if isinstance(base_kernel, RBF):
#             lengthscales = base_kernel.lengthscale.squeeze()
#             kernel_info['lengthscales'] = lengthscales
#             print(f'RBF Lengthscales: {lengthscales.tolist()}')
#         elif isinstance(base_kernel, RQ):
#             lengthscales = base_kernel.lengthscale.squeeze()
#             kernel_info['lengthscales'] = lengthscales
#             print(f'RQ Lengthscales: {lengthscales.tolist()}')
#             print(f'RQ Alpha: {base_kernel.alpha.item():.4f}')
#         elif isinstance(base_kernel, Per):
#             print(f'Periodic Period: {base_kernel.period_length.squeeze().tolist()}')
#             print(f'Periodic Lengthscales: {base_kernel.lengthscale.squeeze().tolist()}')
#         elif isinstance(base_kernel, Lin):
#             print(f'Linear Variance: {base_kernel.variance.squeeze().tolist()}')
    
#     print(f'Noise variance: {model.likelihood.noise.item():.6f}')
#     return kernel_info

# kernel_info = extract_kernel_info(gp_gs)

# # Unormalise predictions
# pred_mean = observed_pred.mean
# mu0 = scaler.inverse_transform(pred_mean.unsqueeze(1))[:,0]
# stds = scaler.inverse_transform(observed_pred.stddev.unsqueeze(1))[:,0]
# lower_stand, upper_stand = observed_pred.confidence_region()
# lower = scaler.inverse_transform(lower_stand.unsqueeze(1))[:,0]
# upper = scaler.inverse_transform(upper_stand.unsqueeze(1))[:,0]

# print(f'\nMSE (all): {mean_squared_error(mu0, y_all_nonstand):.6f}')

# # estimated inducing points
# _z = gp_gs.variational_strategy.inducing_points.detach().numpy()

# if N_train > 0 and M > 0:
#     print("Calculating distances between inducing points and training data...")
#     # Calculate pairwise Euclidean distances (M x N_train matrix)
#     dist_matrix = cdist(_z, X_train_np, metric='euclidean')
#     print("Finding nearest training point indices...")
#     # Find the index of the minimum distance for each inducing point (row)
#     _z_indices = np.argmin(dist_matrix, axis=1) # Shape (M,)
#     print(f"Found indices for {len(_z_indices)} inducing points.")

#     # Get the timestamps corresponding to these nearest training points
#     _z_times = date_train[_z_indices]
# else:
#     print("Warning: Cannot find nearest neighbors with no training data or inducing points.")
#     _z_indices = []
#     inducing_point_times = []

# """------------------------------------------------------------------------
#     Feature Selection: Remove features with lengthscale > 50
# """

# def identify_relevant_features(kernel_info, feature_names=None, lengthscale_threshold=50.0):
#     """
#     Identify features with lengthscales <= threshold for feature selection
    
#     Args:
#         kernel_info: Dictionary containing kernel lengthscale information
#         feature_names: List of feature names (optional)
#         lengthscale_threshold: Threshold for lengthscale filtering
    
#     Returns:
#         relevant_indices: List of feature indices to keep
#         removed_indices: List of feature indices to remove
#     """
#     relevant_indices = []
#     removed_indices = []
    
#     # Handle complex kernels with multiple components
#     all_lengthscales = []
    
#     for key, lengthscales in kernel_info.items():
#         if 'lengthscales' in key and torch.is_tensor(lengthscales):
#             if lengthscales.dim() == 0:  # Scalar
#                 all_lengthscales.append(lengthscales.item())
#             else:  # Vector
#                 all_lengthscales.extend(lengthscales.tolist())
    
#     # If no lengthscales found in complex kernel, try to extract from simple kernel
#     if not all_lengthscales and 'lengthscales' in kernel_info:
#         ls = kernel_info['lengthscales']
#         if torch.is_tensor(ls):
#             if ls.dim() == 0:
#                 all_lengthscales = [ls.item()]
#             else:
#                 all_lengthscales = ls.tolist()
    
#     # If still no lengthscales, extract directly from model
#     if not all_lengthscales:
#         print("Warning: Could not extract lengthscales from kernel_info. Extracting directly from model...")
#         try:
#             base_kernel = gp_gs.covar_module.base_kernel
#             if hasattr(base_kernel, 'lengthscale'):
#                 ls = base_kernel.lengthscale.squeeze()
#                 all_lengthscales = ls.tolist() if ls.dim() > 0 else [ls.item()]
#         except:
#             print("Could not extract lengthscales. Keeping all features.")
#             return list(range(D)), []
    
#     # Ensure we have the right number of lengthscales
#     if len(all_lengthscales) < D:
#         print(f"Warning: Found {len(all_lengthscales)} lengthscales but have {D} features.")
#         print("Using available lengthscales and keeping remaining features.")
#         all_lengthscales.extend([1.0] * (D - len(all_lengthscales)))  # Default to 1.0 for missing
#     elif len(all_lengthscales) > D:
#         print(f"Warning: Found {len(all_lengthscales)} lengthscales but only have {D} features.")
#         all_lengthscales = all_lengthscales[:D]  # Truncate
    
#     # Identify relevant and irrelevant features
#     for i, ls in enumerate(all_lengthscales):
#         if ls <= lengthscale_threshold:
#             relevant_indices.append(i)
#         else:
#             removed_indices.append(i)
    
#     print(f"\nFeature Selection Results:")
#     print(f"  Features to keep: {len(relevant_indices)}/{D}")
#     print(f"  Features to remove: {len(removed_indices)}/{D}")
#     print(f"  Lengthscale threshold: {lengthscale_threshold}")
    
#     if feature_names:
#         print(f"  Removed features: {[feature_names[i] for i in removed_indices]}")
#     else:
#         print(f"  Removed feature indices: {removed_indices}")
    
#     return relevant_indices, removed_indices

# # Apply feature selection
# feature_names = X_df.columns.tolist() if hasattr(X_df, 'columns') else None
# relevant_indices, removed_indices = identify_relevant_features(
#     kernel_info, 
#     feature_names=feature_names, 
#     lengthscale_threshold=50.0
# )

# # Create filtered datasets if features were removed
# if removed_indices:
#     print(f"\n🔥 Removing {len(removed_indices)} features with high lengthscales...")
    
#     # Filter training and test data
#     X_train_filtered = X_train[:, relevant_indices]
#     X_test_filtered = X_test[:, relevant_indices] 
#     X_all_filtered = X_all[:, relevant_indices]
    
#     # Update dimensions
#     D_filtered = len(relevant_indices)
#     print(f"Reduced feature dimensions: {D} -> {D_filtered}")
    
#     # Re-initialize inducing points with filtered data
#     if init_ip_method == 'random':
#         indices = np.random.choice(N_train, min(M, N_train), replace=False)
#         inducing_points_filtered = X_train_filtered[indices, :]
#     else:
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             kmeans = KMeans(n_clusters=M, init='k-means++', n_init=10)
#             kmeans.fit(X_train_filtered.numpy())
#             inducing_points_filtered = torch.tensor(kmeans.cluster_centers_, dtype=floating_point)
    
#     # Create new kernel with filtered dimensions
#     k_filtered = ScaleKernel(RQ(ard_num_dims=D_filtered, dtype=floating_point))
    
#     # Create filtered model for tuning
#     gp_filtered = SVGP(inducing_points_filtered, D_filtered, k_filtered)
#     gp_filtered.likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-5))
#     gp_filtered.likelihood.noise = torch.tensor(0.028, dtype=floating_point)
#     gp_filtered.to(floating_point)
    
#     print("✓ Feature filtering completed. Ready for hyperparameter tuning.")
    
# else:
#     print("\n✓ No features removed (all lengthscales <= 50). Using original data for tuning.")
#     X_train_filtered = X_train
#     X_test_filtered = X_test
#     X_all_filtered = X_all
#     D_filtered = D
#     gp_filtered = gp_gs

# """------------------------------------------------------------------------
#     Fine Tuning with Filtered Features
# """
# def generate_stds(lengthscales, base_std_dev):
#     """ Penalise lengthscales that are high using a greater std """

#     # Ensure lengthscales is a torch tensor
#     if not torch.is_tensor(lengthscales):
#         lengthscales = torch.tensor(lengthscales)
    
#     # Find the minimum lengthscale
#     min_lengthscale = torch.min(lengthscales)
    
#     # Calculate the standard deviations for each Gaussian distribution
#     std_devs = base_std_dev * torch.exp((lengthscales - min_lengthscale)/6)
#     std_devs = torch.tensor([300 if std == torch.inf else std for std in std_devs])
#     return std_devs.tolist()

# # Extract lengthscales for tuning (from filtered model if applicable)
# try:
#     if hasattr(gp_filtered.covar_module.base_kernel, 'lengthscale'):
#         tuning_lengthscales = gp_filtered.covar_module.base_kernel.lengthscale.squeeze()
#     else:
#         tuning_lengthscales = gp_filtered.covar_module.base_kernel.kernels[0].lengthscale.squeeze()
    
#     ls_stds = generate_stds(tuning_lengthscales, base_std_dev=1e-4)
# except Exception as e:
#     print(f"Warning: Could not extract lengthscales for tuning: {e}")
#     ls_stds = [1e-4] * D_filtered  # Default std devs

# if gp_filtered is not gp0: # Only tune if we have a model to tune
#     stds = {'outputscale': 1e-3,
#             'se_lengthscale': ls_stds,
#             'rq_lengthscale': ls_stds,
#             'per_lengthscale': ls_stds,
#             'rq_alpha': 1e-2,
#             'lin_variance': 1e-3,
#             'noise_variance': 1e-4}
    
#     # Create new trainer with filtered data
#     if removed_indices:
#         auto_trainer_filtered = GPTraining(gp_filtered, X_train_filtered, y_train, X_test_filtered, y_test)
#         auto_trainer_filtered.param_stds = stds
#         trainer_to_use = auto_trainer_filtered
#         model_to_tune = gp_filtered
#     else:
#         auto_trainer.param_stds = stds
#         trainer_to_use = auto_trainer
#         model_to_tune = gp_gs

#     try:
#         print(f"\n🔧 Starting hyperparameter tuning with {D_filtered} features...")
#         tuned_gp = trainer_to_use.tune(
#             gp_to_tune=model_to_tune,
#             N_sim=10,
#             mse_stop=0.003,
#             lr=0.005,
#             training_iterations=100,
#             batch_size=256)
#         print("✓ Hyperparameter tuning completed successfully")
        
#         # Make predictions with tuned model
#         tuned_gp.eval()
#         tuned_gp.likelihood.eval()

#         with torch.no_grad(), gpytorch.settings.fast_pred_var():
#             observed_pred_tuned = tuned_gp.likelihood(tuned_gp(X_all_filtered))

#         # Unormalise predictions
#         pred_mean_tuned = observed_pred_tuned.mean
#         mu_tuned = scaler.inverse_transform(pred_mean_tuned.unsqueeze(1))[:,0]
#         stds_tuned = scaler.inverse_transform(observed_pred_tuned.stddev.unsqueeze(1))[:,0]
#         lower_stand_tuned, upper_stand_tuned = observed_pred_tuned.confidence_region()
#         lower_tuned = scaler.inverse_transform(lower_stand_tuned.unsqueeze(1))[:,0]
#         upper_tuned = scaler.inverse_transform(upper_stand_tuned.unsqueeze(1))[:,0]

#         print(f'MSE-tuned: {mean_squared_error(mu_tuned, y_all_nonstand):.6f}')
        
#     except Exception as e:
#         print(f"❌ Error during tuning: {e}")
#         traceback.print_exc()
#         # Fallback to non-tuned model
#         tuned_gp = model_to_tune
#         mu_tuned = mu0
#         lower_tuned = lower
#         upper_tuned = upper
#         print("Using non-tuned model for final predictions")
# else:
#     print("No tuning performed (using original model)")
#     tuned_gp = gp_filtered
#     mu_tuned = mu0
#     lower_tuned = lower
#     upper_tuned = upper

# """-------------------------------------------------------------------------
# PLOT
# """
# fig, ax = plt.subplots()

# # Increase the size of the axis numbers
# plt.rcdefaults()
# plt.rc('xtick', labelsize=14)
# plt.rc('ytick', labelsize=14)
# fig.autofmt_xdate()

# plt.fill_between(date_time, lower, upper,
#                 alpha=0.5, color='lightcoral',
#                 label='2$\\sigma$')
# ax.plot(date_time, y_all_nonstand, '*', color='green', label='Val')
# ax.plot(date_time, mu0, color='black', label='GP(GS)')
# ax.plot(date_time, mu_tuned, color='red', label='GP(tuned)')
# plt.axvline(date_time[end_train-1], linestyle='--', linewidth=3,
#             color='black')
# # Estimated _z (inducing points)
# ax.vlines(
#     x=_z_times,
#     ymin=y_all_nonstand.min(),
#     ymax=y_all_nonstand.max(),
#     alpha=0.3,
#     linewidth=1.5,
#     label="z*",
#     color='orange'
# )
# ax.set_xlabel(" Date-time", fontsize=14)
# ax.set_ylabel(" Fault density", fontsize=14)
# plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)
# ax.set_xlabel(" Date-time", fontsize=14)
# ax.set_ylabel(" Fault density", fontsize=14)
# plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)
# plt.show()