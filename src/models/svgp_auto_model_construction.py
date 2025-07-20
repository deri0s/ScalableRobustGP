import torch
import copy
import gpytorch
import numpy as np
from scipy.stats import qmc
from numpy.random import uniform
import random
from sklearn.metrics import mean_squared_error
import traceback # For detailed error printing
import matplotlib.pyplot as plt

# GPyTorch imports
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.mlls import VariationalELBO
from gpytorch.kernels import ScaleKernel, Kernel
from gpytorch.kernels import RBFKernel as RBF, RQKernel as RQ
from gpytorch.kernels import LinearKernel as Lin, PeriodicKernel as Per

# DataLoader
from torch.utils.data import TensorDataset, DataLoader

class SVGP(ApproximateGP):
    def __init__(self, inducing_points, D, kernel):
        num_inducing = inducing_points.size(0)
        ard_dims = D if getattr(kernel, 'ard_num_dims', None) == D else None

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            # ARD vs non-ARD batch shape for variational parameters
            batch_shape=torch.Size([D]) if ard_dims else torch.Size([])
        )

        # Define variational strategy
        variational_strategy = VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(SVGP, self).__init__(variational_strategy)

        # Mean and Covariance Modules
        self.mean_module = ConstantMean(batch_shape=torch.Size([D]) if ard_dims else torch.Size([]))
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class GPTraining():
    def __init__(self, gp0: SVGP,
                 X_train: torch.Tensor, y_train: torch.Tensor,
                 X_eval: torch.Tensor, y_eval: np.ndarray):
        super(GPTraining, self).__init__()

        # --- Input Validation ---
        if not isinstance(gp0, ApproximateGP):
             raise TypeError("gp0 must be an instance of gpytorch.models.ApproximateGP or its subclass.")
        if not isinstance(X_train, torch.Tensor) or not isinstance(y_train, torch.Tensor):
             raise TypeError("X_train and y_train must be torch Tensors.")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(f"X_train ({X_train.shape[0]}) and y_train ({y_train.shape[0]}) must have the same number of samples.")
        if len(y_eval) == 0:
            print("Warning: y_eval is empty.")

        self.gp0 = gp0 # Keep the initial model structure
        self.X_train = X_train
        self.y_train = y_train
        self.X_eval = X_eval
        self.y_eval = y_eval
        self.N, self.D = self.X_train.shape
        self.nv0 = self.gp0.likelihood.noise.item() # Initial noise
        self.M = len(self.gp0.variational_strategy.inducing_points)

        # Use float type from gp0 or default
        self.dtype = gp0.covar_module.dtype if hasattr(gp0.covar_module, 'dtype') else X_train.dtype

        # Covariance functions building blocks
        self.base_kernels = {
            'RBF': lambda: RBF(ard_num_dims=self.D, dtype=self.dtype),
            'RQ': lambda: RQ(ard_num_dims=self.D, dtype=self.dtype),
            'Lin': lambda: Lin(ard_num_dims=self.D, dtype=self.dtype),
            'Per': lambda: Per(ard_num_dims=self.D, dtype=self.dtype),
            }
        self._validate_base_kernels() # Check if factories produce Kernels

        # Parameter limits and stds (set later by methods)
        self.param_limits = {}
        self.param_stds = {}
        self.param_centers = {}
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

        # --- Periodic Kernel ---
        elif isinstance(module, Per):
            ard = getattr(module, 'ard_num_dims', None) == self.D
            # Period Length (usually scalar)
            period_param_name = f"{module_name_prefix}.period_length"
            period_key = 'per_period_length'
            if sample_type == 'uniform':
                limits = self._validate_limits(self.param_limits.get(period_key), period_param_name)
                val = self._sample_param_uniform(limits)
                module.period_length = torch.tensor(max(val, 1e-6), dtype=self.dtype)
            elif sample_type == 'gaussian':
                center = self.param_centers.get(period_param_name)
                std = self._validate_std(self.param_stds.get(period_key), period_param_name)
                if center is None: raise ValueError(f"Center not set for {period_param_name}")
                val = self._sample_param_gauss(center, std)
                module.period_length = torch.tensor(val, dtype=self.dtype)
            elif sample_type == 'center':
                self.param_centers[period_param_name] = module.period_length.item()

            # Lengthscale (can be ARD)
            ls_param_name = f"{module_name_prefix}.lengthscale"
            ls_key = 'per_lengthscale'
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

        # --- Linear Kernel ---
        elif isinstance(module, Lin):
            # Variance (usually scalar)
            var_param_name = f"{module_name_prefix}.variance"
            var_key = 'lin_variance'
            if sample_type == 'uniform':
                limits = self._validate_limits(self.param_limits.get(var_key), var_param_name)
                val = self._sample_param_uniform(limits)
                module.variance = torch.tensor(max(val, 1e-6), dtype=self.dtype)
            elif sample_type == 'gaussian':
                center = self.param_centers.get(var_param_name)
                std = self._validate_std(self.param_stds.get(var_key), var_param_name)
                if center is None: raise ValueError(f"Center not set for {var_param_name}")
                val = self._sample_param_gauss(center, std)
                module.variance = torch.tensor(val, dtype=self.dtype)
            elif sample_type == 'center':
                 self.param_centers[var_param_name] = module.variance.item()
            param_applied = True

        # Return True if any parameter was applied to this specific module instance
        return param_applied


    def initialise_params(self, gp_model, sample_type):
        """ Traverses the kernel and likelihood, applies sampling or stores centers """
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
                 current_noise = likelihood.noise.item() if hasattr(likelihood.noise, 'item') else likelihood.raw_noise.exp().item()
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
             print(f"Error during kernel parameter initialisation ({sample_type}): {e}")
             traceback.print_exc()
             raise # Re-raise error to stop problematic simulation


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
        mll = VariationalELBO(gp.likelihood, gp, num_data=self.N)

        # DataLoader for batching
        dataset = TensorDataset(self.X_train, self.y_train)
        # Ensure shuffle=True for stochastic optimization
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

        # Evaluation on validation set (for model selection)
        gp.eval()
        gp.likelihood.eval()
                
        if len(self.X_eval) == 0:
            print("Warning: No evaluation data available.")
            return 0.0
        
        # Prediction
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            try:
                observed_pred = gp.likelihood(gp(self.X_eval))
                pred_means = observed_pred.mean.cpu()
            except Exception as e:
                print(f"Error during prediction: {e}")
                return float('inf')
        
        # Calculate validation MSE for model selection
        return mean_squared_error(pred_means, self.y_eval)


    def combine_kernels(self, best_kernel_name: str, best_kernel_factory, remaining_kernels: dict, operation: str) -> dict:
        """ Creates new kernel factories by combining the best kernel with remaining kernels. """
        new = {}
        
        if operation not in ["+", "*"]:
            raise ValueError("Operation must be '+' or '*'")
        
        for name, kernel_factory in remaining_kernels.items():
            if name == best_kernel_name:
                continue  # Skip combining with itself
                
            # Create combination name (keeping consistent ordering)
            sorted_names = sorted([best_kernel_name, name])
            new_name = f"({sorted_names[0]} {operation} {sorted_names[1]})"
            
            # Create factory - use default args to capture current factories correctly
            if operation == "+":
                new[new_name] = lambda k1=best_kernel_factory, k2=kernel_factory: k1() + k2()
            elif operation == "*":
                new[new_name] = lambda k1=best_kernel_factory, k2=kernel_factory: k1() * k2()
        
        return new

    # Modified to accept training params and handle new model/eval
    def get_best_kernel(self, kernels_to_evaluate: dict, lr: float, training_iterations: int, batch_size: int) -> tuple:
        """ Evaluates a set of kernel structures and returns the best state_dict, error, and name. """
        best_kernel_state_dict = None
        best_error = float('inf')
        best_name = "None"

        if not kernels_to_evaluate:
             print("Warning: No kernels provided to get_best_kernel.")
             return None, float('inf'), "None"

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

                # 3. Create a temp GP model to inherit var strategy etc.
                temp_gp = copy.deepcopy(self.gp0)
                temp_gp.covar_module = candidate_kernel_structure # Replace kernel

                # 4. Perform grid search (random search) for this kernel structure
                #    Pass the temp model as a template to inherit structure/inducing points
                best_state_dict_for_kernel, mse_list = self.grid_search(
                    gp_template=temp_gp,
                    N_sim=self.N_sim,
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


    def auto_model_cons(self, levels, N_sim=100,
                        param_limits={}, param_stds={}, 
                        mse_stop=1e-3,
                        lr=0.01, training_iterations=100, batch_size=64):

        # --- Validation ---
        if not isinstance(levels, int) or levels <= 0:
            raise ValueError("Levels must be a positive integer.")
        if not isinstance(param_limits, dict) or not isinstance(param_stds, dict):
            raise TypeError("param_limits and param_stds must be dictionaries.")

        # --- Store parameters for child methods ---
        self.N_sim = N_sim
        self.param_limits = param_limits
        self.param_stds = param_stds
        self.mse_stop = mse_stop

        # --- Initialisation ---
        final_best_error = float('inf')
        final_best_state_dict = None
        final_best_name = "None"
        
        # Track the best kernel from each level for tree search
        current_best_kernel_name = None
        current_best_kernel_factory = None
        all_evaluated_factories = {}  # Track all evaluated factories by name

        # --- Main Loop ---
        for level in range(levels):
            print(f"\n{'='*15} Exploring Level {level + 1} Kernels {'='*15}")

            if level == 0:
                # Level 1: Evaluate base kernels
                factories_to_evaluate = self.base_kernels.copy()
                print(f"Level 1 - Evaluating base kernels: {list(factories_to_evaluate.keys())}")
                
            else:
                # Level 2+: Tree search - only combine best from previous level with remaining kernels
                if current_best_kernel_name is None or current_best_kernel_factory is None:
                    print("No best kernel found from previous level. Stopping exploration.")
                    break
                    
                print(f"Level {level+1} - Building tree from best kernel: '{current_best_kernel_name}'")
                
                # Get remaining kernels (exclude the best one to avoid self-combination)
                remaining_kernels = {k: v for k, v in self.base_kernels.items() if k != current_best_kernel_name}
                
                # Generate combinations: best + remaining and best * remaining
                sum_kernels = self.combine_kernels(
                    current_best_kernel_name, current_best_kernel_factory, 
                    remaining_kernels, "+"
                )
                
                prod_kernels = self.combine_kernels(
                    current_best_kernel_name, current_best_kernel_factory, 
                    remaining_kernels, "*"
                )
                
                factories_to_evaluate = {}
                factories_to_evaluate.update(sum_kernels)
                factories_to_evaluate.update(prod_kernels)
                
                print(f"Level {level+1} - Kernels to evaluate: {list(factories_to_evaluate.keys())}")

            # Skip if no new kernels to evaluate
            if not factories_to_evaluate:
                print("No new kernel structures to evaluate at this level.")
                continue

            # Evaluate current level's kernels
            best_state_dict, error_level, best_name = self.get_best_kernel(
                factories_to_evaluate,
                lr=lr,
                training_iterations=training_iterations,
                batch_size=batch_size
            )

            # Add evaluated factories to master dictionary
            all_evaluated_factories.update(factories_to_evaluate)

            # Update overall best if improvement found
            if error_level < final_best_error:
                final_best_error = error_level
                final_best_state_dict = best_state_dict
                final_best_name = best_name
                print(f'*** New Overall Best Found! Error: {final_best_error:.5f} ***')

            # Update current best for next level's tree search
            if best_name != "None" and best_name in factories_to_evaluate:
                current_best_kernel_name = best_name
                current_best_kernel_factory = factories_to_evaluate[best_name]
                print(f"Best kernel for Level {level+1}: '{current_best_kernel_name}' (Error: {error_level:.5f})")
            else:
                print(f"Warning: Could not update best kernel for next level. Best name: '{best_name}'")
                # Continue with previous best if current level failed

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
            final_best_gp.to(self.dtype)

            # Load the state dict into the correctly structured model
            final_best_gp.load_state_dict(final_best_state_dict, strict=True)

        except Exception as e:
            print(f"Error reconstructing or loading final best model: {e}")
            traceback.print_exc()
            print("Returning the initial model structure instead.")
            return copy.deepcopy(self.gp0)

        return final_best_gp


    def grid_search(self, gp_template: ApproximateGP, N_sim,
                    lr=0.01, training_iterations=100, batch_size=256):
        """ Sample random hyperparameters (param_limits) to initialise
            the neg-log-Margilag-Likelihood for the optimisation step (training)
        """

        # Access limits/stop condition stored in self
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
                 print(f"Skipping simulation {i+1} due to parameter initialisation error: {e}")
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
                print(f'Sim {i+1}/{N_sim} | Error (Eval): {mse:.5f} | Current Best: {best_mse:.5f}')

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

        return best_state_dict, mse_list

    def tune(self, gp_to_tune: ApproximateGP, N_sim, mse_stop=1e-3,
            lr=0.01, training_iterations=100, batch_size=256, plot_mse=True,
            track_mse="eval"):
        """ Fine-tunes a GP by sampling params from Gaussian distribution """

        # Validation of stds happens during initialise_params
        self.mse_stop = mse_stop
        mse_list = []
        training_mse_list = []  # Track training MSE
        validation_mse_list = []  # Track validation MSE

        # initialise MSE
        gp_to_tune.eval()
        gp_to_tune.likelihood.eval()

        if track_mse == 'eval':
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = gp_to_tune.likelihood(gp_to_tune(self.X_eval))
            # best_mse = float('inf')
            best_mse = mean_squared_error(observed_pred.mean, self.y_eval)
        else:
            # on training data
            # with torch.no_grad(), gpytorch.settings.fast_pred_var():
            #     observed_pred = gp_to_tune.likelihood(gp_to_tune(self.X_train))
            # best_mse = mean_squared_error(observed_pred.mean, self.y_train)
            best_mse = float('inf')

        try:
            self.initialise_params(gp_to_tune, 'center')
        except Exception as e:
            print(f"Error setting initial centers for tuning: {e}")
            traceback.print_exc()
            return copy.deepcopy(gp_to_tune) # Original if centering fails

        best_gp = copy.deepcopy(gp_to_tune) # Start with the input GP as best

        for i in range(N_sim):
            current_sim_gp = copy.deepcopy(best_gp)

            try:
                self.initialise_params(current_sim_gp, 'gaussian')

                # Get validation MSE from train_and_evaluate
                validation_mse = self.train_and_evaluate(current_sim_gp, lr, training_iterations, batch_size)

                if validation_mse == float('inf'):
                    print(f"Tuning simulation {i+1} failed during training/evaluation.")
                    mse_list.append(validation_mse)
                    training_mse_list.append(float('inf'))
                    validation_mse_list.append(float('inf'))
                    continue

                # Calculate training MSE
                current_sim_gp.eval()
                current_sim_gp.likelihood.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    train_pred = current_sim_gp.likelihood(current_sim_gp(self.X_train))
                    training_mse = mean_squared_error(train_pred.mean.cpu(), self.y_train.cpu())

                # Store MSE values for plotting
                mse_list.append(validation_mse)
                training_mse_list.append(training_mse)
                validation_mse_list.append(validation_mse)

                print(f'Tune Sim {i+1}/{N_sim} | Train MSE: {training_mse:.5f} | Val MSE: {validation_mse:.5f} | Current Best: {best_mse:.5f}')
                
                # Update best model if improvement
                if track_mse == "eval":
                    if validation_mse < best_mse:
                        print(f"\nFound better parameters during tuning! Val MSE: {validation_mse:.5f} < {best_mse:.5f}\n")
                        best_mse = validation_mse
                        best_gp = copy.deepcopy(current_sim_gp) # Keep the whole model
                        # Option: Re-center Gaussian around the new best? (Can sometimes lock in too early)
                        # self.initialise_params(best_gp, 'center')

                        # Check early stopping
                        if self.mse_stop is not None and validation_mse < self.mse_stop:
                            print(f'!! Target MSE ({self.mse_stop}) reached. Stopping Tuning early. !!')
                            break # Stop simulation loop
                else:
                    if training_mse < best_mse:
                        print(f"\nFound better parameters during tuning! Val MSE: {training_mse:.5f} < {best_mse:.5f}\n")
                        best_mse = training_mse
                        best_gp = copy.deepcopy(current_sim_gp)
                        # Check early stopping
                        if self.mse_stop is not None and training_mse < self.mse_stop:
                            print(f'!! Target MSE ({self.mse_stop}) reached. Stopping Tuning early. !!')
                            break # Stop simulation loop


            except Exception as e:
                print(f"ERROR during tuning simulation {i+1}: {e}")
                traceback.print_exc()
                mse_list.append(float('inf'))
                training_mse_list.append(float('inf'))
                validation_mse_list.append(float('inf'))

        # print(f"\n--- Tuning Finished. Best MSE: {best_mse:.5f} ---")
        
        # Plot MSE evolution if requested
        if plot_mse and len(training_mse_list) > 0:
            self._plot_tuning_mse(training_mse_list, validation_mse_list, N_sim)
        
        return best_gp

    def _plot_tuning_mse(self, training_mse_list, validation_mse_list, N_sim):
        """ Helper method to plot training and validation MSE over tuning iterations """
        
        # Filter out infinite values for plotting
        valid_indices = [i for i, (train_mse, val_mse) in enumerate(zip(training_mse_list, validation_mse_list)) 
                        if train_mse != float('inf') and val_mse != float('inf')]
        
        if not valid_indices:
            print("Warning: No valid MSE values to plot.")
            return
        
        valid_train_mse = [training_mse_list[i] for i in valid_indices]
        valid_val_mse = [validation_mse_list[i] for i in valid_indices]
        valid_iterations = [i + 1 for i in valid_indices]  # 1-indexed for display
        
        plt.figure(figsize=(10, 6))
        plt.plot(valid_iterations, valid_train_mse, 'b-', label='Training MSE', alpha=0.7, marker='o', markersize=4)
        plt.plot(valid_iterations, valid_val_mse, 'r-', label='Validation MSE', alpha=0.7, marker='s', markersize=4)
        
        # Highlight the best validation MSE
        best_val_idx = valid_val_mse.index(min(valid_val_mse))
        best_iteration = valid_iterations[best_val_idx]
        best_val_mse = valid_val_mse[best_val_idx]
        
        plt.scatter(best_iteration, best_val_mse, color='red', s=100, marker='*', 
                    label=f'Best Val MSE: {best_val_mse:.5f}', zorder=5)
        
        plt.xlabel('Tuning Iteration')
        plt.ylabel('Mean Squared Error')
        plt.title('Training and Validation MSE During Hyperparameter Tuning')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale often better for MSE visualization
        
        # Add text box with summary statistics
        textstr = f'Total Iterations: {len(valid_iterations)}/{N_sim}\n'
        textstr += f'Best Train MSE: {min(valid_train_mse):.5f}\n'
        textstr += f'Best Val MSE: {min(valid_val_mse):.5f}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()