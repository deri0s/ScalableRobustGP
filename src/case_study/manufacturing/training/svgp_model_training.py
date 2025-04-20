import torch
import copy
import gpytorch
import time
import pandas as pd
import numpy as np
from numpy.random import uniform
import random
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler as ss
from matplotlib import pyplot as plt
import traceback # For detailed error printing

# GPyTorch imports
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy # New imports
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.kernels import ScaleKernel, Kernel
from gpytorch.kernels import RBFKernel as RBF, RQKernel as RQ
from gpytorch.kernels import LinearKernel as Lin, PeriodicKernel as Per
from gpytorch.constraints import GreaterThan # For noise constraint

# DataLoader
from torch.utils.data import TensorDataset, DataLoader

"""
NSG data
"""

file = 'validation_data_main.xlsx'

# Training df
X_df = pd.read_excel(file, sheet_name='X_stand')
y_df = pd.read_excel(file, sheet_name='y_nonstand')
t_df = pd.read_excel(file, sheet_name='timelags')

# drop tweel position
X_df.drop(columns=['9282 Tweel Position'], inplace=True)
t_df.drop(columns=['9282 Tweel Position'], inplace=True)

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
if X_df.empty or y_df.empty:
    raise ValueError("DataFrames are empty after alignment. Check lagging procedure and input data.")

X = X_df.values
y_nonstand, date_time = y_df.gp_pred.values, y_df.date_time.values

N, D = np.shape(X)
test_perc = 0.12
end_train = N - int(N*test_perc)

X_train_np = X[0:end_train]
date_train = date_time[0:end_train]
N_train = len(X_train_np)
y_train_nonstand = y_nonstand[0:end_train]

# Define X_test_np and y_test_nonstand correctly for evaluation metric
X_test_np = X[end_train:N] # Test features
y_test_nonstand = y_nonstand[end_train:N] # Test targets (non-standardized)
date_test = date_time[end_train:N] # Test dates

# --- Input Validation: Check if train/test splits are valid ---
if N_train <= 0 or len(X_test_np) <= 0:
    raise ValueError(f"Training size ({N_train}) or Test size ({len(X_test_np)}) is zero or negative. Check data splitting.")
if len(y_test_nonstand) != len(X_test_np):
     raise ValueError(f"Test features ({len(X_test_np)}) and test targets ({len(y_test_nonstand)}) lengths differ.")

# Standardise outputs
y_train_reshape = y_train_nonstand.reshape(-1,1)
scaler = ss()
scaler.fit(y_train_reshape)
y_norm_np = scaler.transform(y_train_reshape)

# Convert data to torch tensors
floating_point = torch.float64
X_train = torch.tensor(X_train_np, dtype=floating_point)
y_train = torch.tensor(y_norm_np, dtype=floating_point).squeeze()
X_test = torch.tensor(X_test_np, dtype=floating_point) # Use only test features here
X_all = torch.tensor(X, dtype=floating_point) # Full X for final prediction/plot
y_all_nonstand = y_nonstand # Keep full non-standardized y for plotting/final eval


# --- Model Definition ---
class SVGP(ApproximateGP):
    def __init__(self, inducing_points, D, kernel):
        num_inducing = inducing_points.size(0)
        ard_dims = D if getattr(kernel, 'ard_num_dims', None) == D else None

        # Define variational distribution
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            # Handle ARD vs non-ARD batch shape for variational parameters
            batch_shape=torch.Size([D]) if ard_dims else torch.Size([])
        )

        # Define variational strategy
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(SVGP, self).__init__(variational_strategy)

        # Mean and Covariance Modules
        self.mean_module = ConstantMean(batch_shape=torch.Size([D]) if ard_dims else torch.Size([]))
        # Kernel module passed here should NOT be InducingPointKernel
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

# --- GPTraining Class ---
class GPTraining():
    # Added X_train, y_train, y_test_nonstand
    def __init__(self, gp0: SVGP, X_train: torch.Tensor, y_train: torch.Tensor, y_test_nonstand: np.ndarray):
        super(GPTraining, self).__init__()

        # --- Input Validation ---
        if not isinstance(gp0, ApproximateGP):
             raise TypeError("gp0 must be an instance of gpytorch.models.ApproximateGP or its subclass.")
        if not isinstance(X_train, torch.Tensor) or not isinstance(y_train, torch.Tensor):
             raise TypeError("X_train and y_train must be torch Tensors.")
        if not isinstance(y_test_nonstand, np.ndarray):
             raise TypeError("y_test_nonstand must be a numpy array.")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"X_train ({X_train.shape[0]}) and y_train ({y_train.shape[0]}) must have the same number of samples.")
        if len(y_test_nonstand) == 0:
            print("Warning: y_test_nonstand is empty.")

        self.gp0 = gp0 # Keep the initial model structure
        self.X_train = X_train
        self.y_train = y_train
        # Store non-standardised test targets for evaluation (make sure it corresponds to X_test from outside)
        self.y_test_nonstand = y_test_nonstand
        self.N, self.D = self.X_train.shape
        self.nv0 = self.gp0.likelihood.noise.item() # Initial noise
        self.M = len(self.gp0.variational_strategy.inducing_points)

        # Use float type from gp0 or default
        self.dtype = gp0.covar_module.dtype if hasattr(gp0.covar_module, 'dtype') else X_train.dtype

        # Covariance functions building blocks
        self.base_kernels = {
            # 'RBF': lambda: RBF(ard_num_dims=self.D, dtype=self.dtype),
            'RQ': lambda: RQ(ard_num_dims=self.D, dtype=self.dtype),
            # 'Lin': lambda: Lin(ard_num_dims=self.D, dtype=self.dtype), # Uncomment if needed
            # 'Per': lambda: Per(ard_num_dims=self.D, dtype=self.dtype), # Uncomment if needed
            }
        self._validate_base_kernels() # Check if factories produce Kernels

        # Parameter limits and stds (set later by methods)
        self.param_limits = {}
        self.param_stds = {}
        # Store centers for Gaussian sampling
        self.param_centers = {}
        # Store other params like N_sim, mse_stop
        self.N_sim = None
        self.mse_stop = None

    def _validate_base_kernels(self):
         """ Check that base kernel factories produce valid Kernel objects """
         for name, factory in self.base_kernels.items():
              try:
                   kernel_instance = factory()
                   if not isinstance(kernel_instance, Kernel):
                        raise TypeError(f"Factory for '{name}' does not produce a gpytorch.kernels.Kernel.")
                   # Check ARD setting consistency (optional but good)
                   if self.D > 1 and not getattr(kernel_instance, 'ard_num_dims', None) == self.D:
                        print(f"Warning: Kernel '{name}' might not be configured for ARD (D={self.D})")
              except Exception as e:
                   raise ValueError(f"Error creating kernel from factory '{name}': {e}")


    # --- Validation Helpers ---
    def _validate_limits(self, limits, param_name):
        """ Validates [min, max] limits """
        if limits is None:
             raise ValueError(f"Limits for '{param_name}' are not set.")
        if not isinstance(limits, (list, tuple)) or len(limits) != 2:
            raise ValueError(f"{param_name} limits must be a list/tuple of length 2 [min, max], got {limits}")
        if not all(isinstance(x, (int, float)) for x in limits):
            raise ValueError(f"Elements of {param_name} limits must be numeric, got {limits}")
        if limits[0] >= limits[1]:
            raise ValueError(f"Min must be less than max in {param_name} limits, got {limits}")
        return limits # Return validated/formatted limits

    def _validate_std(self, std, param_name):
        """ Validates a single standard deviation value """
        if std is None:
            raise ValueError(f"Standard deviation for '{param_name}' is not set.")
        if not isinstance(std, (int, float)):
             raise TypeError(f"{param_name} standard deviation must be numeric, got {std}")
        if std < 0:
             raise ValueError(f"{param_name} standard deviation cannot be negative, got {std}")
        return std

    def _validate_ard_param(self, param, param_name, is_std=False):
        """ Validates ARD limits [[min,max],...] or stds [std1, std2,...]
            Allows scalar input
        """
        if param is None:
             raise ValueError(f"Parameter '{param_name}' (ARD={self.D}) is not set.")

        if self.D == 1: # Treat as non-ARD case
            if is_std:
                return [self._validate_std(param[0] if isinstance(param, (list, tuple)) else param, f"{param_name}[0]")]
            else:
                return [self._validate_limits(param[0] if isinstance(param, (list, tuple)) else param, f"{param_name}[0]")]


        if isinstance(param, (int, float)):
            # Scalar provided: Broadcast/repeat
            if is_std:
                validated_scalar = self._validate_std(param, param_name)
                return [validated_scalar] * self.D
            else:
                validated_limits = self._validate_limits(param, param_name) # This will fail for scalar limits, expect list
                # This path shouldn't be hit if scalar means same limits/std for all dims
                # Re-think: Maybe scalar limit means [scalar_min, scalar_max] repeated? Assume scalar means std for stds.
                raise ValueError(f"Scalar input for ARD limits '{param_name}' is ambiguous. Provide list.")

        elif isinstance(param, (list, tuple)):
            if len(param) != self.D:
                raise ValueError(f"ARD parameter '{param_name}' must have {self.D} elements for {self.D} dimensions, got {len(param)}")
            # Validate each element
            validated_params = []
            for d in range(self.D):
                element_name = f"{param_name}[{d}]"
                if is_std:
                    validated_params.append(self._validate_std(param[d], element_name))
                else:
                    validated_params.append(self._validate_limits(param[d], element_name))
            return validated_params
        else:
            raise TypeError(f"Unsupported type for ARD parameter '{param_name}': {type(param)}")


    # --- Refactored Kernel Sampling ---

    def _sample_param_uniform(self, limits):
        """ Samples from uniform distribution given [min, max] limits """
        min_val, max_val = limits # Assumes already validated
        return uniform(low=min_val, high=max_val)

    def _sample_param_gauss(self, center, std):
         """ Samples from Gaussian, ensuring result > 1e-6 """
         # Assumes center and std are validated numeric types
         if std < 0: std = 1e-6 # Handle potential negative std after validation bug
         sample = random.gauss(center, sigma=std)
         return max(sample, 1e-6) # Avoid non-positive values

    def _apply_sampling_to_module(self, module, module_name_prefix, sample_type):
        """ Helper to apply sampling/centering logic based on module type and name """
        param_applied = False
        module_type_name = type(module).__name__

        # --- ScaleKernel ---
        if isinstance(module, ScaleKernel):
            param_base_name = f"{module_name_prefix}.outputscale"
            limits_or_std_key = 'outputscale'
            if sample_type == 'uniform':
                limits = self._validate_limits(self.param_limits.get(limits_or_std_key), param_base_name)
                val = self._sample_param_uniform(limits)
                module.outputscale = torch.tensor(max(val, 1e-6), dtype=self.dtype) # Ensure positive
            elif sample_type == 'gaussian':
                center = self.param_centers.get(param_base_name)
                std = self._validate_std(self.param_stds.get(limits_or_std_key), param_base_name)
                if center is None: raise ValueError(f"Center not set for {param_base_name}")
                val = self._sample_param_gauss(center, std)
                module.outputscale = torch.tensor(val, dtype=self.dtype)
            elif sample_type == 'center':
                 self.param_centers[param_base_name] = module.outputscale.item()
            param_applied = True

        # --- RBF Kernel ---
        elif isinstance(module, RBF):
            param_base_name = f"{module_name_prefix}.lengthscale"
            limits_or_std_key = 'se_lengthscale' # Unique key for RBF lengthscale
            ard = getattr(module, 'ard_num_dims', None) == self.D

            if sample_type == 'uniform':
                limits = self._validate_limits(self.param_limits.get(limits_or_std_key), param_base_name) # Expects single [min,max]
                ls = np.array([self._sample_param_uniform(limits) for _ in range(self.D)] if ard else [self._sample_param_uniform(limits)])
                module.lengthscale = torch.tensor(np.maximum(ls, 1e-6), dtype=self.dtype)
            elif sample_type == 'gaussian':
                center_list = self.param_centers.get(param_base_name) # Expects list [c1, c2...] or [c]
                std_list = self._validate_ard_param(self.param_stds.get(limits_or_std_key), param_base_name, is_std=True) # Returns validated list
                if center_list is None: raise ValueError(f"Center not set for {param_base_name}")
                ls = np.array([self._sample_param_gauss(center_list[d], std_list[d]) for d in range(self.D if ard else 1)])
                module.lengthscale = torch.tensor(ls, dtype=self.dtype)
            elif sample_type == 'center':
                 # Store as list, even if not ARD
                 ls_val = module.lengthscale.detach().cpu().squeeze().numpy()
                 self.param_centers[param_base_name] = ls_val.tolist() if isinstance(ls_val, np.ndarray) else [ls_val.item()]
            param_applied = True

        # --- RQ Kernel ---
        elif isinstance(module, RQ):
            ard = getattr(module, 'ard_num_dims', None) == self.D
            # Alpha
            alpha_param_name = f"{module_name_prefix}.alpha"
            alpha_key = 'rq_alpha'
            if sample_type == 'uniform':
                limits = self._validate_limits(self.param_limits.get(alpha_key), alpha_param_name)
                val = self._sample_param_uniform(limits)
                module.alpha = torch.tensor(max(val, 1e-6), dtype=self.dtype)
            elif sample_type == 'gaussian':
                center = self.param_centers.get(alpha_param_name)
                std = self._validate_std(self.param_stds.get(alpha_key), alpha_param_name)
                if center is None: raise ValueError(f"Center not set for {alpha_param_name}")
                val = self._sample_param_gauss(center, std)
                module.alpha = torch.tensor(val, dtype=self.dtype)
            elif sample_type == 'center':
                self.param_centers[alpha_param_name] = module.alpha.item()
            # Lengthscale
            ls_param_name = f"{module_name_prefix}.lengthscale"
            ls_key = 'rq_lengthscale' # Unique key for RQ lengthscale
            if sample_type == 'uniform':
                limits = self._validate_limits(self.param_limits.get(ls_key), ls_param_name)
                ls = np.array([self._sample_param_uniform(limits) for _ in range(self.D)] if ard else [self._sample_param_uniform(limits)])
                module.lengthscale = torch.tensor(np.maximum(ls, 1e-6), dtype=self.dtype)
            elif sample_type == 'gaussian':
                 center_list = self.param_centers.get(ls_param_name)
                 std_list = self._validate_ard_param(self.param_stds.get(ls_key), ls_param_name, is_std=True)
                 if center_list is None: raise ValueError(f"Center not set for {ls_param_name}")
                 ls = np.array([self._sample_param_gauss(center_list[d], std_list[d]) for d in range(self.D if ard else 1)])
                 module.lengthscale = torch.tensor(ls, dtype=self.dtype)
            elif sample_type == 'center':
                 ls_val = module.lengthscale.detach().cpu().squeeze().numpy()
                 self.param_centers[ls_param_name] = ls_val.tolist() if isinstance(ls_val, np.ndarray) else [ls_val.item()]
            param_applied = True

        # --- Add elif blocks for Lin, Per if used ---
        # elif isinstance(module, Per): ... handle period_length and lengthscale ...

        # Return True if any parameter was applied to this specific module instance
        return param_applied


    def initialise_params(self, gp_model, sample_type):
        """ Traverses the kernel and likelihood, applies sampling or stores centers. """
        if sample_type not in ['uniform', 'gaussian', 'center']:
            raise ValueError("sample_type must be 'uniform', 'gaussian', or 'center'")

        # --- Need centers for Gaussian sampling ---
        if sample_type == 'gaussian' and not self.param_centers:
             print("Warning: Gaussian sampling called before centers were set. Setting centers first.")
             self.initialise_params(gp_model, 'center') # Auto-set centers if needed

        # --- Sample likelihood noise ---
        likelihood = gp_model.likelihood
        noise_param_name = "likelihood.noise"
        noise_key = 'noise_variance'
        try:
            if sample_type == 'uniform':
                limits = self._validate_limits(self.param_limits.get(noise_key), noise_param_name)
                nv = self._sample_param_uniform(limits)
                # Use noise setter for constraints
                likelihood.noise = torch.tensor(max(nv, 1e-6), dtype=self.dtype)
            elif sample_type == 'gaussian':
                center = self.param_centers.get(noise_param_name)
                std = self._validate_std(self.param_stds.get(noise_key), noise_param_name)
                if center is None: raise ValueError(f"Center not set for {noise_param_name}")
                nv = self._sample_param_gauss(center, std)
                likelihood.noise = torch.tensor(nv, dtype=self.dtype)
            elif sample_type == 'center':
                 # Store the actual noise value, respecting constraints if applied
                 current_noise = likelihood.noise.item() if hasattr(likelihood.noise, 'item') else likelihood.raw_noise.exp().item() # More robust
                 self.param_centers[noise_param_name] = current_noise
        except Exception as e:
             print(f"Error processing likelihood noise: {e}")
             traceback.print_exc()


        # --- Sample kernel parameters ---
        try:
            # Iterate through all modules in the covariance module
            for name, module in gp_model.covar_module.named_modules():
                # Pass full name prefix for unique parameter identification
                self._apply_sampling_to_module(module, name, sample_type)
        except Exception as e:
             print(f"Error during kernel parameter initialization ({sample_type}): {e}")
             traceback.print_exc()
             raise # Re-raise error to stop problematic simulation


    # Function to train and evaluate the model - MODIFIED for VariationalELBO and Batching
    def train_and_evaluate(self, gp: ApproximateGP, lr: float, training_iterations: int, batch_size: int):
        # --- Validation ---
        if not isinstance(gp, ApproximateGP):
             raise TypeError("gp must be an instance of ApproximateGP for train_and_evaluate.")
        if not isinstance(lr, (int, float)) or lr <= 0:
             raise ValueError(f"Learning rate (lr={lr}) must be positive.")
        if not isinstance(training_iterations, int) or training_iterations <= 0:
             raise ValueError(f"Training iterations ({training_iterations}) must be a positive integer.")
        if not isinstance(batch_size, int) or batch_size <= 0:
             raise ValueError(f"Batch size ({batch_size}) must be a positive integer.")
        if self.N == 0: # Check if training data exists
            print("Warning: No training data (N=0). Skipping training and evaluation.")
            return float('inf')
        if batch_size > self.N:
            batch_size = self.N

        gp.train()
        gp.likelihood.train()

        optimizer = torch.optim.Adam(gp.parameters(), lr=lr)
        # Use VariationalELBO - ensure likelihood and strategy are correctly linked
        mll = VariationalELBO(gp.likelihood, gp, num_data=self.N)

        # DataLoader for batching
        dataset = TensorDataset(self.X_train, self.y_train)
        # Ensure shuffle=True for stochastic optimization
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # --- Training Loop ---
        for i in range(training_iterations):
            epoch_loss = 0.0
            for x_batch, y_batch in dataloader:
                try:
                    optimizer.zero_grad()
                    output = gp(x_batch) # Forward pass on batch
                    # Ensure y_batch has the correct shape if needed by MLL
                    loss = -mll(output, y_batch.squeeze()) # Calculate loss on batch
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                except Exception as e:
                     print(f"Error in training step {i+1}, batch: {e}")
                     # Optionally: break epoch, return inf, etc.
                     return float('inf') # Indicate failure

            # Optional: print epoch loss (can be verbose)
            # if (i + 1) % 20 == 0:
            #     print(f'Iter {i+1}/{training_iterations} - Avg. Loss: {epoch_loss / len(dataloader):.4f}')


        # --- Predictions & Evaluation on Held-Out Test Set ---
        gp.eval()
        gp.likelihood.eval()

        if len(X_test) == 0: # Check if test data exists
            print("Warning: No test data (X_test is empty). Skipping evaluation.")
            return 0.0 # Or Inf? Depends on desired behavior. Let's return 0 if no test data.

        # Use the specific X_test data prepared earlier
        test_dataset = TensorDataset(X_test) # Use the held-out X_test
        # Use a reasonable batch size for prediction, doesn't have to match training
        pred_batch_size = min(batch_size * 2, len(X_test)) if len(X_test) > 0 else 1
        test_dataloader = DataLoader(test_dataset, batch_size=pred_batch_size, shuffle=False)

        all_pred_means = []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
             try:
                 for (x_batch,) in test_dataloader: # Need comma for unpacking
                     # Ensure x_batch is on the correct device if using GPU
                     # x_batch = x_batch.to(self.X_train.device)
                     observed_pred_batch = gp.likelihood(gp(x_batch))
                     all_pred_means.append(observed_pred_batch.mean.cpu()) # Collect means
             except Exception as e:
                  print(f"Error during prediction: {e}")
                  return float('inf') # Indicate prediction failure

        # Handle case where prediction might have failed silently
        if not all_pred_means:
             print("Prediction resulted in no collected means.")
             return float('inf')

        pred_mean_tensor = torch.cat(all_pred_means)

        # Unnormalize predictions using the scaler fitted on y_train
        try:
            # Ensure pred_mean_tensor is correctly shaped for inverse_transform
            mu = scaler.inverse_transform(pred_mean_tensor.unsqueeze(1).numpy())[:, 0]
        except Exception as e:
             print(f"Error during inverse transform: {e}")
             print(f"Shape of pred_mean_tensor: {pred_mean_tensor.shape}")
             return float('inf')


        # Calculate error against the non-standardized held-out test data
        if len(mu) != len(self.y_test_nonstand):
             print(f"Warning: Length mismatch between predictions ({len(mu)}) and test targets ({len(self.y_test_nonstand)}). Cannot calculate MSE.")
             # This might happen if prediction failed partway through
             return float('inf')

        test_error = mean_squared_error(mu, self.y_test_nonstand)

        return test_error


    def combine_kernels(self, operands_1: dict, operation: str, operands_2: dict) -> dict:
        """ Creates new kernel factories by combining factories from two dictionaries. """
        new = {}
        created_keys = set() # Track keys like "(A + B)" to handle commutativity

        if operation not in ["+", "*"]:
            raise ValueError("Operation must be '+' or '*'")

        for name1, kernel_factory1 in operands_1.items():
            for name2, kernel_factory2 in operands_2.items():
                # Create sorted key for commutative operations '+' and '*'
                # to treat "RBF + RQ" the same as "RQ + RBF"
                sorted_names = sorted([name1, name2])
                new_name = f"({sorted_names[0]} {operation} {sorted_names[1]})"

                # Skip if this combination already created (handles commutativity and self-combination)
                if new_name in created_keys:
                    continue

                # Use default args in lambda to capture current kernel_factory correctly
                if operation == "+":
                     new[new_name] = lambda k1=kernel_factory1, k2=kernel_factory2: k1() + k2()
                elif operation == "*":
                     new[new_name] = lambda k1=kernel_factory1, k2=kernel_factory2: k1() * k2()

                created_keys.add(new_name)
        return new

    # Modified to accept training params and handle new model/eval
    def get_best_kernel(self, kernels_to_evaluate: dict, lr: float, training_iterations: int, batch_size: int) -> tuple:
        """ Evaluates a set of kernel structures and returns the best state_dict and error. """
        best_kernel_state_dict = None
        best_error = float('inf')
        best_name = "None"

        if not kernels_to_evaluate:
             print("Warning: No kernels provided to get_best_kernel.")
             return None, float('inf')

        for name, kernel_factory in kernels_to_evaluate.items():
            print(f"\n--- Evaluating Kernel Structure: {name} ---")
            current_error = float('inf')
            temp_gp = None # Define before try block

            try:
                # 1. Create the base kernel from the factory
                base_kernel_instance = kernel_factory()

                # 2. Wrap in ScaleKernel (common practice)
                candidate_kernel_structure = ScaleKernel(base_kernel_instance)
                candidate_kernel_structure.to(self.dtype) # Ensure correct dtype

                # 3. Create a temporary GP model to inherit var strategy etc.
                temp_gp = copy.deepcopy(self.gp0)
                temp_gp.covar_module = candidate_kernel_structure # Replace kernel

                # 4. Perform grid search (random search) for this kernel structure
                #    Pass the temporary model as a template to inherit structure/inducing points
                best_state_dict_for_kernel, mse_list = self.grid_search(
                    gp_template=temp_gp,
                    N_sim=self.N_sim, # N_sim set by auto_model_cons/grid_search call
                    lr=lr,
                    training_iterations=training_iterations,
                    batch_size=batch_size
                    # Limits/mse_stop are accessed via self.param_limits / self.mse_stop
                )
                current_error = min(mse_list) if mse_list else float('inf')

                print(f"Grid search for '{name}' finished: Best Error = {current_error:.5f}")

            except Exception as e:
                 print(f"ERROR evaluating kernel structure '{name}': {e}")
                 traceback.print_exc()
                 current_error = float('inf')
                 best_state_dict_for_kernel = None

            # 5. Update overall best for this level if improvement found
            if current_error < best_error:
                print(f"--- New best structure: '{name}' | Error: {current_error:.5f} < Prev error: {best_error:.5f} ---")
                best_error = current_error
                best_kernel_state_dict = best_state_dict_for_kernel # Store the state_dict
                best_name = name

        print(f"\n--- Best structure in this level: '{best_name}' (Error: {best_error:.5f}) ---")
        # Return the best state dict found among the evaluated structures and its error
        return best_kernel_state_dict, best_error, best_name


    # Modified to accept training params and manage parameter dictionaries
    def auto_model_cons(self, levels, N_sim=100,
                        param_limits={}, param_stds={}, # Pass limits/stds as dicts
                        mse_stop=1e-3,
                        lr=0.01, training_iterations=100, batch_size=64):

        # --- Validation ---
        if not isinstance(levels, int) or levels <= 0:
             raise ValueError("Levels must be a positive integer.")
        if not isinstance(param_limits, dict) or not isinstance(param_stds, dict):
             raise TypeError("param_limits and param_stds must be dictionaries.")

        # --- Store parameters for child methods ---
        self.N_sim = N_sim
        self.param_limits = param_limits # Used by grid_search via initialize_params('uniform')
        self.param_stds = param_stds     # Used by tune via initialize_params('gaussian')
        self.mse_stop = mse_stop         # Used by grid_search/tune

        # --- Initialization ---
        final_best_error = float('inf')
        final_best_state_dict = None # Store the best state dict found
        final_best_name = "None"

        # Start with base kernels factories
        current_level_factories = self.base_kernels
        all_evaluated_factories = {} # Track factories across levels by name

        # --- Main Loop ---
        for level in range(levels):
            print(f"\n{'='*15} Exploring Level {level + 1} Kernels {'='*15}")

            # Combine kernel factories for the next level
            if level > 0:
                 sum_kernels = self.combine_kernels(self.base_kernels, "+", all_evaluated_factories)
                 prod_kernels = {}
                 if level < 2: # Limit complexity of products
                     prod_kernels = self.combine_kernels(self.base_kernels, "*", self.base_kernels) # Base * Base
                     # Optionally: Combine base with previous level bests
                     # prod_kernels.update(self.combine_kernels(self.base_kernels, "*", all_evaluated_factories))

                 current_level_factories = sum_kernels
                 current_level_factories.update(prod_kernels)

            # Identify only new kernel structures to evaluate this level
            factories_to_evaluate = {k: v for k, v in current_level_factories.items() if k not in all_evaluated_factories}
            print(f"Kernels to evaluate at Level {level+1}: {list(factories_to_evaluate.keys())}")

            if not factories_to_evaluate:
                print("No new kernel structures to evaluate at this level.")
                continue

            # Get current level's best kernel (returns state_dict, error) by evaluating the new factories
            best_state_dict, error_level, best_name = self.get_best_kernel(
                factories_to_evaluate,
                lr=lr,
                training_iterations=training_iterations,
                batch_size=batch_size
            )

            # Add the factories evaluated in this level to the master dictionary
            all_evaluated_factories.update(factories_to_evaluate)

            if error_level < final_best_error:
                final_best_error = error_level
                final_best_state_dict = best_state_dict
                final_best_name = best_name
                print(f'*** New Overall Best Found! Error: {final_best_error:.5f} ***')

        # --- Finish ---
        print(f"\n{'='*15} Auto Model Construction Finished {'='*15}")
        if final_best_state_dict is None or final_best_name == "None":
             print("No successful model evaluation completed or best kernel name not found.")
             return None

        print(f"Overall Best Kernel: '{final_best_name}'")
        print(f"Overall Best Error Found: {final_best_error:.5f}")

        # --- Reconstruct the winning model structure ---
        try:
            print(f"Reconstructing final model with kernel: {final_best_name}")
            # Retrieve the winning factory
            if final_best_name not in all_evaluated_factories:
                 raise KeyError(f"Winning kernel name '{final_best_name}' not found in evaluated factories.")
            winning_factory = all_evaluated_factories[final_best_name]

            # Create the winning kernel structure (assuming ScaleKernel wrap)
            winning_kernel = ScaleKernel(winning_factory())

            # Create the final model instance by copying gp0 and replacing the kernel
            final_best_gp = copy.deepcopy(self.gp0)
            final_best_gp.covar_module = winning_kernel.to(self.dtype)
            final_best_gp.to(self.dtype) # Ensure the model is correct dtype

            # Load the state dict into the correctly structured model
            final_best_gp.load_state_dict(final_best_state_dict, strict=True)

        except Exception as e:
             print(f"Error reconstructing or loading final best model: {e}")
             traceback.print_exc() # Print detailed traceback for debugging
             print("Returning the initial model structure instead.")
             return copy.deepcopy(self.gp0)

        return final_best_gp


    def grid_search(self, gp_template: ApproximateGP, N_sim,
                    lr=0.01, training_iterations=100, batch_size=256):
        """ Performs random search over hyperparameters defined in self.param_limits. """

        # Access limits/stop condition stored in self
        # Validation of limits happens during initialize_params

        mse_list = []
        best_mse = float('inf')
        best_state_dict = None
        random_start = True # Always use random start for grid search as per original logic? Yes.

        # Make a working copy of the template model for modification
        gp = copy.deepcopy(gp_template)

        for i in range(N_sim):
            # Fresh copy each time to ensure random init starts clean
            current_sim_gp = copy.deepcopy(gp)

            # Initialise parameters using uniform sampling based on self.param_limits
            try:
                self.initialise_params(current_sim_gp, 'uniform')
            except Exception as e:
                 print(f"Skipping simulation {i+1} due to parameter initialization error: {e}")
                 continue

            # Train and evaluate the model with current parameters
            try:
                mse = self.train_and_evaluate(current_sim_gp, lr, training_iterations, batch_size)
                # Handle potential inf return from train_and_evaluate
                if mse == float('inf'):
                     print(f"Simulation {i+1} failed during training/evaluation.")
                     mse_list.append(mse)
                     continue # Skip failed simulation update

                mse_list.append(mse)
                print(f'Sim {i+1}/{N_sim} | Error: {mse:.5f} | Current Best: {best_mse:.5f}')

                # Update best MSE and model state dict
                if mse < best_mse:
                    print(f"\n Found better parameters! MSE: {mse:.5f} < {best_mse:.5f}\n")
                    best_mse = mse
                    best_state_dict = copy.deepcopy(current_sim_gp.state_dict())

                    # Check for early stopping
                    if self.mse_stop is not None and mse < self.mse_stop:
                        print(f'!! Target MSE ({self.mse_stop}) reached. Stopping Grid Search early. !!')
                        break # Stop simulation loop

            except Exception as e:
                 print(f"ERROR during grid search simulation {i+1}: {e}")
                 traceback.print_exc()
                 mse_list.append(float('inf')) # Record failure

        # Return the best state found and the list of errors
        return best_state_dict, mse_list


    def tune(self, gp_to_tune: ApproximateGP, N_sim, mse_stop=1e-3,
             lr=0.01, training_iterations=100, batch_size=64):
        """ Fine-tunes a GP by sampling params from Gaussian distribution """

        # Access stds/stop condition stored in self
        # Validation of stds happens during initialise_params
        mse_list = []
        best_mse = float('inf')

        # Set Gaussian centers based on the provided model's current parameters
        print("--- Starting Fine Tuning ---")
        print("Setting Gaussian centers from input model...")
        try:
            self.initialise_params(gp_to_tune, 'center')
            # print("Centers set:", self.param_centers) # Debug
        except Exception as e:
             print(f"Error setting initial centers for tuning: {e}")
             traceback.print_exc()
             return copy.deepcopy(gp_to_tune) # Original if centering fails

        best_gp = copy.deepcopy(gp_to_tune) # Start with the input GP as best

        # --- Tuning Loop ---
        for i in range(N_sim):
            print(f'Fine-tuning Simulation {i+1}/{N_sim}')
            # Create a working copy for this simulation's sampling
            # Start from the current best found during tuning for stability
            current_sim_gp = copy.deepcopy(best_gp)

            try:
                # Sample parameters from Gaussian around the centers using self.param_stds
                self.initialise_params(current_sim_gp, 'gaussian')
                # print("Parameters sampled from Gaussian.") # Debug

                # Train and evaluate
                mse = self.train_and_evaluate(current_sim_gp, lr, training_iterations, batch_size)
                 # Handle potential inf return
                if mse == float('inf'):
                     print(f"Tuning simulation {i+1} failed during training/evaluation.")
                     mse_list.append(mse)
                     continue

                mse_list.append(mse)
                print(f'Tune Sim {i+1} | Error: {mse:.5f} | Current Best Tune: {best_mse:.5f}')

                # Update best model if improvement
                if mse < best_mse:
                    print(f"--- Found better parameters during tuning! MSE: {mse:.5f} < {best_mse:.5f} ---")
                    best_mse = mse
                    best_gp = copy.deepcopy(current_sim_gp) # Keep the whole model
                    # Option: Re-center Gaussian around the new best? (Can sometimes lock in too early)
                    # self.initialize_params(best_gp, 'center')

                    # Check early stopping
                    if self.mse_stop is not None and mse < self.mse_stop:
                        print(f'!! Target MSE ({self.mse_stop}) reached. Stopping Tuning early. !!')
                        break # Stop simulation loop

            except Exception as e:
                 print(f"ERROR during tuning simulation {i+1}: {e}")
                 traceback.print_exc()
                 mse_list.append(float('inf'))

        print(f"\n--- Tuning Finished. Best MSE: {best_mse:.5f} ---")
        return best_gp


# ============================================================================
# --- Application Code ---
# ============================================================================

# --- Load or Initialize Initial Model ---
init_state_dict = torch.load('expert_main0.pth', weights_only=False)
inducing_points = init_state_dict['covar_module.inducing_points']
init_noise_var = 0.028

# --- Default Initialization if needed ---
if inducing_points is None:
    N_inducing_points = min(50, N_train) # Ensure not more than N_train
    if N_inducing_points <= 0:
         raise ValueError("Cannot initialize inducing points: N_train is zero.")
    print(f"Initialising {N_inducing_points} inducing points randomly from training data.")
    inducing_points = X_train[np.random.choice(N_train, N_inducing_points, replace=False), :]


# --- Define Initial Kernel Structure ---
# This should match the structure intended by the loaded state_dict if used,
# otherwise it defines the starting point for AMC.
kernel0 = ScaleKernel(RQ(ard_num_dims=D, dtype=floating_point))
print(f"Using initial kernel structure: {kernel0}")

# --- Create Initial ApproximateGP Model ---
# Use a noise constraint for stability
likelihood = GaussianLikelihood(noise_constraint=GreaterThan(1e-5))
gp0 = SVGP(inducing_points, D, kernel0)
gp0.likelihood = likelihood

gp0.likelihood.noise = torch.tensor(0.028, dtype=floating_point)

gp0.to(floating_point)

# --- Automatic Model Construction ---
print("\n--- Starting Automatic Model Construction ---")
# Instantiate GPTraining with data and the prepared initial model gp0
auto_trainer = GPTraining(gp0, X_train, y_train, y_test_nonstand)

# Define parameter limits for the random search (grid_search)
# Keys should match those used in _apply_sampling_to_module
limits = {
    'outputscale': [0.1, 10.0],       # Limits for ScaleKernel outputscale
    'se_lengthscale': [0.05, 100.0],  # Limits for RBF lengthscale (same range for all dims)
    'rq_lengthscale': [0.05, 100.0],  # Limits for RQ lengthscale (same range for all dims)
    'rq_alpha': [0.05, 5],            # Limits for RQ alpha
    'noise_variance': [0.025, 0.028]  # Limits for Likelihood noise
}

# Automatic Model Construction: Grid search parameters
gp_gs = auto_trainer.auto_model_cons(
    levels=1,                  # Number of levels (e.g., 1: RBF, RQ; 2: RBF+RQ, RBF*RBF etc.)
    N_sim=10,                 # Reduced simulations per structure for speed
    param_limits=limits,       # Pass the limits dictionary
    mse_stop=0.005,            # Target MSE for early stopping
    lr=0.01,                   # Learning rate for training within AMC
    training_iterations=50,    # Training iterations per evaluation
    batch_size=256             # Batch size for training
)

gp_gs.eval()
gp_gs.likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(gp_gs(X_all))

# Unormalise predictions
pred_mean = observed_pred.mean
mu0 = scaler.inverse_transform(pred_mean.unsqueeze(1))[:,0]
stds = scaler.inverse_transform(observed_pred.stddev.unsqueeze(1))[:,0]
lower_stand, upper_stand = observed_pred.confidence_region()
lower = scaler.inverse_transform(lower_stand.unsqueeze(1))[:,0]
upper = scaler.inverse_transform(upper_stand.unsqueeze(1))[:,0]

# print('MSE (test):',mean_squared_error(mu, y_test_nonstand))


"""------------------------------------------------------------------------
    Fine Tuning
"""
def generate_stds(lengthscales, base_std_dev):
    """ Penalise lengthscales that are high using a greater std """

    # Ensure lengthscales is a torch tensor
    if not torch.is_tensor(lengthscales):
        lengthscales = torch.tensor(lengthscales)
    
    # Find the minimum lengthscale
    min_lengthscale = torch.min(lengthscales)
    
    # Calculate the standard deviations for each Gaussian distribution
    std_devs = base_std_dev * torch.exp((lengthscales - min_lengthscale)/6)
    std_devs = torch.tensor([300 if std == torch.inf else std for std in std_devs])
    
    return std_devs

ls_stds = generate_stds(gp_gs.covar_module.base_kernel.lengthscale.squeeze(),
                        base_std_dev=1e-4)

if gp_gs is not gp0: # Only tune if AMC produced a model
    print("\n--- Starting Fine Tuning ---")

    # Define parameter stds for Gaussian sampling during tuning
    stds = {
        'outputscale': 1e-3,
        'se_lengthscale': ls_stds,  # Example: List for ARD stds
        'rq_lengthscale': ls_stds,
        'rq_alpha': 1e-2,
        'noise_variance': 1e-4
    }
    # Update the trainer's stds dictionary
    auto_trainer.param_stds = stds

    try:
        tuned_gp = auto_trainer.tune(
            gp_to_tune=gp_gs, # Tune the best model found so far
            N_sim=20,              # Number of tuning simulations
            mse_stop=0.003,        # Tuning target MSE
            lr=0.005,              # Tuning learning rate
            training_iterations=80, # Tuning iterations
            batch_size=256
        )
    except Exception as e:
        print(f"Error during tuning: {e}")
        traceback.print_exc()

tuned_gp.eval()
tuned_gp.likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(tuned_gp(X_all))

# Unormalise predictions
pred_mean = observed_pred.mean
mu_tuned = scaler.inverse_transform(pred_mean.unsqueeze(1))[:,0]
stds = scaler.inverse_transform(observed_pred.stddev.unsqueeze(1))[:,0]
lower_stand, upper_stand = observed_pred.confidence_region()
lower = scaler.inverse_transform(lower_stand.unsqueeze(1))[:,0]
upper = scaler.inverse_transform(upper_stand.unsqueeze(1))[:,0]

"""-------------------------------------------------------------------------
PLOT
"""
fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
fig.autofmt_xdate()

plt.fill_between(date_time, lower, upper,
                alpha=0.5, color='lightcoral',
                label='2$\\sigma$')
ax.plot(date_time, y_all_nonstand, '*', color='green', label='Val')
ax.plot(date_time, mu0, color='black', label='GP(GS)')
ax.plot(date_time, mu_tuned, color='red', label='GP(tuned)')
plt.axvline(date_time[end_train-1], linestyle='--', linewidth=3,
            color='black')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)
plt.show()