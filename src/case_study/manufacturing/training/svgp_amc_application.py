import torch
import gpytorch
import time
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler as ss
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist # _z plot
# GPyTorch imports
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel
from gpytorch.kernels import RBFKernel as RBF, RQKernel as RQ
from gpytorch.constraints import GreaterThan # For noise constraint
import traceback # For detailed error printing
# Automatic Model Construction
from models.svgp_auto_model_construction import GPTraining, SVGP

"""
NSG data
"""

# NSG post processes data location
ROOT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT_PATH / "data" / "processed" / "Training_data_partitions" / "CSV"
file_name = 'data0.xlsx'
format = Path(file_name).suffix.lstrip('.')

file_path = PROCESSED_PATH / file_name

# Training df
if format == 'xlsx':
    X_df = pd.read_excel(file_path, sheet_name='X_stand')
    y_df = pd.read_excel(file_path, sheet_name='y_nonstand')
    t_df = pd.read_excel(file_path, sheet_name='timelags')

    # drop tweel position
    # X_df.drop(columns=['9282 Tweel Position'], inplace=True)
    # t_df.drop(columns=['9282 Tweel Position'], inplace=True)
else:
    df = pd.read_csv(file_path)
    X_df = df.iloc[:,:-1]
    y_df = df.iloc[:,-1]

# """---------------------------------------------------------------------------
#     CREATE LAGGED FEATURES
# """

# def align_inputs(x_df, y_df, t_series):
#     xdeep = x_df.copy()
#     ydeep = y_df.copy()
#     # Ensure t_series values are numeric before finding max
#     numeric_t_series = pd.to_numeric(t_series, errors='coerce').fillna(0)
#     if numeric_t_series.empty:
#          max_lag = 0
#     else:
#          max_lag = int(max(numeric_t_series))

#     # X
#     for name, lag in t_series.items():
#         # Ensure lag is treated as integer for shift
#         try:
#             lag_int = int(float(lag))
#             if lag_int > 0: # Only shift if lag is positive
#                  xdeep[name] = xdeep[name].shift(lag_int)
#         except ValueError:
#             print(f"Warning: Could not convert lag '{lag}' for feature '{name}' to int. Skipping shift.")

#     # Drop rows with NaNs introduced by shifting (only drop up to max_lag rows from top)
#     xdeep = xdeep.iloc[max_lag:] # More direct way to handle shift NaNs

#     # y and date-time alignment
#     # Ensure ydeep has enough rows before slicing
#     if len(ydeep) >= max_lag:
#         ydeep = ydeep.iloc[max_lag:].reset_index(drop=True)
#     else:
#         # Handle case where ydeep is shorter than max_lag (e.g., return empty DataFrames)
#         print(f"Warning: y DataFrame length ({len(ydeep)}) is less than max_lag ({max_lag}). Alignment might be incorrect.")
#         return pd.DataFrame(columns=x_df.columns), pd.DataFrame(columns=y_df.columns)

#     # Ensure xdeep and ydeep have the same length after alignment
#     common_len = min(len(xdeep), len(ydeep))
#     xdeep = xdeep.iloc[:common_len].reset_index(drop=True)
#     ydeep = ydeep.iloc[:common_len].reset_index(drop=True)

#     return xdeep, ydeep

# X_df, y_df = align_inputs(X_df, y_df, t_df.iloc[0,:])

# """---------------------------------------------------------------------------
#     STANDARDISE TRAINING & TEST DATA
# """
# if X_df.empty or y_df.empty:
#     raise ValueError("DataFrames are empty after alignment. Check lagging procedure and input data.")

# X = X_df.values
# y_all_nonstand, date_time = y_df.gp_pred.values, y_df.date_time.values

# N, D = np.shape(X)
# test_perc = 0.2
# end_train = N - int(N*test_perc)

# X_train_np = X[0:end_train]
# date_train = date_time[0:end_train]
# N_train = len(X_train_np)
# y_train_nonstand = y_all_nonstand[0:end_train]

# # Define X_test_np and y_test_nonstand correctly for evaluation metric
# X_test_np = X[0:N] # Test features
# y_test_nonstand = y_all_nonstand[0:N] # Test targets (non-standardized)
# date_test = date_time[0:N] # Test dates

# # --- Input Validation: Check if train/test splits are valid ---
# if N_train <= 0 or len(X_test_np) <= 0:
#     raise ValueError(f"Training size ({N_train}) or Test size ({len(X_test_np)}) is zero or negative. Check data splitting.")
# if len(y_test_nonstand) != len(X_test_np):
#      raise ValueError(f"Test features ({len(X_test_np)}) and test targets ({len(y_test_nonstand)}) lengths differ.")

# # Standardise outputs
# y_train_reshape = y_train_nonstand.reshape(-1,1)
# scaler = ss()
# scaler.fit(y_train_reshape)
# y_norm_np = scaler.transform(y_train_reshape)

# y_test_reshape = y_test_nonstand.reshape(-1,1)
# scaler.fit(y_test_reshape)
# y_test_stand = scaler.transform(y_test_reshape)

# # Convert data to torch tensors
# floating_point = torch.float64
# X_train = torch.tensor(X_train_np, dtype=floating_point)
# y_train = torch.tensor(y_norm_np, dtype=floating_point).squeeze()
# X_test = torch.tensor(X_test_np, dtype=floating_point)
# y_test = torch.tensor(y_test_stand, dtype=floating_point).squeeze()
# X_all = torch.tensor(X, dtype=floating_point) # Full X for final prediction/plot


# # ============================================================================
# # --- Application Code ---
# # ============================================================================

# M = 120 # Number of inducing points
# inducing_points = X_train[np.random.choice(N_train, M, replace=False), :]

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

# # Define parameter limits for the grid_search
# # Keys should match those used in _apply_sampling_to_module
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
# gp_gs = auto_trainer.auto_model_cons(
#     levels=1,                  # Number of levels (e.g., 1: RBF, RQ; 2: RBF+RQ, RBF*RBF etc.)
#     N_sim=10,                 # Reduced simulations per structure for speed
#     param_limits=limits,       # Pass the limits dictionary
#     mse_stop=0.055,            # Target MSE for early stopping
#     lr=0.01,                   # Learning rate for training within AMC
#     training_iterations=50,    # Training iterations per evaluation
#     batch_size=512             # Batch size for training
# )

# gp_gs.eval()
# gp_gs.likelihood.eval()

# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     observed_pred = likelihood(gp_gs(X_all))
# ctime = time.time() - start_time

# print(f'computational time: {ctime:.6}')

# print(f'\nos: {gp_gs.covar_module.outputscale.item()}')
# print(f'ls: {gp_gs.covar_module.base_kernel.lengthscale.squeeze().tolist()}')
# print(f'al: {gp_gs.covar_module.base_kernel.alpha.item()}')
# print(f'nv: {gp_gs.likelihood.noise.item()}')

# # Unormalise predictions
# pred_mean = observed_pred.mean
# mu0 = scaler.inverse_transform(pred_mean.unsqueeze(1))[:,0]
# stds = scaler.inverse_transform(observed_pred.stddev.unsqueeze(1))[:,0]
# lower_stand, upper_stand = observed_pred.confidence_region()
# lower = scaler.inverse_transform(lower_stand.unsqueeze(1))[:,0]
# upper = scaler.inverse_transform(upper_stand.unsqueeze(1))[:,0]

# print('\nMSE (all):',mean_squared_error(mu0, y_all_nonstand))

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
#     Fine Tuning
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

# ls_stds = generate_stds(gp_gs.covar_module.base_kernel.lengthscale.squeeze(),
#                         base_std_dev=1e-4)

# if gp_gs is not gp0: # Only tune if AMC produced a model
#     stds = {
#         'outputscale': 1e-3,
#         'se_lengthscale': ls_stds,
#         'rq_lengthscale': ls_stds,
#         'rq_alpha': 1e-2,
#         'noise_variance': 1e-4
#     }
#     # Update the trainer's stds dictionary
#     auto_trainer.param_stds = stds

#     try:
#         tuned_gp = auto_trainer.tune(
#             gp_to_tune=gp_gs,
#             N_sim=5,
#             mse_stop=0.003,
#             lr=0.005,
#             training_iterations=80,
#             batch_size=256
#         )
#     except Exception as e:
#         print(f"Error during tuning: {e}")
#         traceback.print_exc()

# tuned_gp.eval()
# tuned_gp.likelihood.eval()

# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     observed_pred = likelihood(tuned_gp(X_all))

# # Unormalise predictions
# pred_mean = observed_pred.mean
# mu_tuned = scaler.inverse_transform(pred_mean.unsqueeze(1))[:,0]
# stds = scaler.inverse_transform(observed_pred.stddev.unsqueeze(1))[:,0]
# lower_stand, upper_stand = observed_pred.confidence_region()
# lower = scaler.inverse_transform(lower_stand.unsqueeze(1))[:,0]
# upper = scaler.inverse_transform(upper_stand.unsqueeze(1))[:,0]

# print('\nMSE-tuned:',mean_squared_error(mu_tuned, y_all_nonstand))

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