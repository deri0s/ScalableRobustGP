import os
import torch
import numpy as np
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import yaml
import matplotlib.pyplot as plt
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ConstantMean
from gpytorch.distributions import MultivariateNormal
import pickle

"""
A distributed robust GP class using GPyTorch SVGP experts with gPoE approach.
Based on the generalised Product of Experts (gPoE) approach (Deisenroth and Wei Ng, 2015).
Enhanced with unstandardization capabilities.
"""

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


class DistributedSVGP:
    
    def __init__(self, expert_dir, N_GPs, device='cpu', plot_expert_pred=False, 
                 scaler_dir=None, global_scaler_path=None):
        """
        Initialize the Distributed SVGP with gPoE approach
        
        Parameters
        ----------
        expert_dir : str or Path
            Directory containing the saved SVGP expert models
        N_GPs : int, optional
            Number of GP experts to load. If None, loads all available experts
        device : str
            Device to run computations on ('cpu' or 'cuda')
        plot_expert_pred : bool
            Whether to plot individual expert predictions
        scaler_dir : str or Path, optional
            Directory containing the scaler files for each expert
        global_scaler_path : str or Path, optional
            Path to a global scaler file (if all experts use the same scaler)
        """
        
        self.expert_dir = Path(expert_dir)
        self.device = device
        self.plot_expert_pred = plot_expert_pred
        
        # Scaler information
        self.scaler_dir = Path(scaler_dir) if scaler_dir else None
        self.global_scaler_path = Path(global_scaler_path) if global_scaler_path else None
        
        # Load expert models and scalers
        self.experts = []
        self.likelihoods = []
        self.scalers = []  # Store scalers for each expert
        self.global_scaler = None
        
        self._load_experts(N_GPs)
        self._load_scalers(N_GPs)
        
        self.N_GPs = len(self.experts)
        self.is_loaded = True
        
        # Load colors for plotting
        self._load_colors()
        
        # Plot option only available for N-GPs <= 200
        if self.plot_expert_pred and self.N_GPs > 200:
            raise AssertionError('Expert predictions can only be plotted for N_GP <= 200')
    
    def _load_experts(self, N_GPs=None):
        """Load SVGP experts from directory"""
        
        # Find all model files in the directory (try both .pt and .pth extensions)
        model_files = sorted(list(self.expert_dir.glob('expert*.pth')))
        
        # Limit number of experts if specified
        if N_GPs is not None:
            model_files = model_files[:N_GPs]
        
        print(f"Loading {len(model_files)} expert models...")
        
        for i, model_file in enumerate(model_files):
            try:
                # Load the saved model state
                checkpoint = torch.load(model_file, map_location=self.device)
                
                # Reconstruct the model
                expert = self._reconstruct_expert(checkpoint)
                likelihood = self._reconstruct_likelihood(checkpoint)
                
                expert.to(self.device)
                likelihood.to(self.device)
                
                # Set to evaluation mode
                expert.eval()
                likelihood.eval()
                
                self.experts.append(expert)
                self.likelihoods.append(likelihood)
                
                print(f"Loaded expert {i+1}/{len(model_files)}")
                
            except Exception as e:
                print(f"Error loading expert {i}: {e}")
                continue
    
    def _load_scalers(self, N_GPs):
        """Load scalers for unstandardization"""
        
        # Option 1: Load global scaler if provided
        if self.global_scaler_path and self.global_scaler_path.exists():
            try:
                if self.global_scaler_path.suffix == '.pkl':
                    with open(self.global_scaler_path, 'rb') as f:
                        self.global_scaler = pickle.load(f)
                elif self.global_scaler_path.suffix == '.pth':
                    scaler_data = torch.load(self.global_scaler_path, map_location='cpu')
                    self.global_scaler = self._reconstruct_scaler(scaler_data)
                
                print(f"Loaded global scaler from {self.global_scaler_path}")
                
                # Use global scaler for all experts
                self.scalers = [self.global_scaler] * self.N_GPs
                return
                
            except Exception as e:
                print(f"Error loading global scaler: {e}")
        
        # Option 2: Load individual scalers for each expert
        if self.scaler_dir and self.scaler_dir.exists():
            scaler_files = sorted(list(self.scaler_dir.glob('scaler*.pkl')))
            scaler_files.extend(sorted(list(self.scaler_dir.glob('scaler*.pth'))))
            
            for i in range(N_GPs):
                scaler = None
                
                # Try to find corresponding scaler file
                for scaler_file in scaler_files:
                    if f'scaler{i}' in str(scaler_file) or f'scaler_{i}' in str(scaler_file):
                        try:
                            if scaler_file.suffix == '.pkl':
                                with open(scaler_file, 'rb') as f:
                                    scaler = pickle.load(f)
                            elif scaler_file.suffix == '.pth':
                                scaler_data = torch.load(scaler_file, map_location='cpu')
                                scaler = self._reconstruct_scaler(scaler_data)
                            break
                        except Exception as e:
                            print(f"Error loading scaler {i}: {e}")
                
                self.scalers.append(scaler)
                if scaler is not None:
                    print(f"Loaded scaler for expert {i}")
                else:
                    print(f"No scaler found for expert {i}")
        
        # Option 3: Try to extract scalers from expert checkpoints
        if not self.scalers or all(s is None for s in self.scalers):
            print("Attempting to extract scalers from expert checkpoints...")
            self._extract_scalers_from_checkpoints()
    
    def _extract_scalers_from_checkpoints(self):
        """Extract scalers from expert checkpoint files if available"""
        model_files = sorted(list(self.expert_dir.glob('expert*.pth')))
        
        for i, model_file in enumerate(model_files):
            if i >= self.N_GPs:
                break
                
            try:
                checkpoint = torch.load(model_file, map_location='cpu')
                scaler = None
                
                # Look for scaler in various possible keys
                scaler_keys = ['scaler', 'y_scaler', 'target_scaler', 'output_scaler']
                for key in scaler_keys:
                    if key in checkpoint:
                        scaler = checkpoint[key]
                        break
                
                if scaler is not None:
                    print(f"Found scaler in checkpoint for expert {i}")
                
                if len(self.scalers) <= i:
                    self.scalers.append(scaler)
                else:
                    self.scalers[i] = scaler
                    
            except Exception as e:
                print(f"Error extracting scaler from expert {i} checkpoint: {e}")
                if len(self.scalers) <= i:
                    self.scalers.append(None)
    
    def _reconstruct_scaler(self, scaler_data):
        """Reconstruct scaler from saved data"""
        if hasattr(scaler_data, 'transform'):
            # Already a scaler object
            return scaler_data
        elif isinstance(scaler_data, dict):
            # Try to reconstruct from dict
            if 'scale_' in scaler_data and 'min_' in scaler_data:
                # MinMaxScaler
                scaler = MinMaxScaler()
                scaler.scale_ = scaler_data['scale_']
                scaler.min_ = scaler_data['min_']
                if 'data_min_' in scaler_data:
                    scaler.data_min_ = scaler_data['data_min_']
                if 'data_max_' in scaler_data:
                    scaler.data_max_ = scaler_data['data_max_']
                return scaler
            elif 'mean_' in scaler_data and 'scale_' in scaler_data:
                # StandardScaler
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaler.mean_ = scaler_data['mean_']
                scaler.scale_ = scaler_data['scale_']
                return scaler
        
        return None
    
    def _reconstruct_expert(self, checkpoint):
        """
        Reconstruct SVGP expert from checkpoint
        Handles common PyTorch saving patterns
        """
        # Check if checkpoint is a dict or the model itself
        if hasattr(checkpoint, '__dict__') and hasattr(checkpoint, 'forward'):
            # This is likely the model itself (SVGP instance)
            expert = checkpoint
        elif isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                # Complete model saved
                expert = checkpoint['model']
            elif 'model_state_dict' in checkpoint:
                # State dict saved - need to reconstruct architecture
                raise NotImplementedError("State dict only saving detected. Please provide model architecture details or save the complete model.")
            elif 'state_dict' in checkpoint:
                # Alternative state dict naming
                raise NotImplementedError("State dict only saving detected. Please provide model architecture details or save the complete model.")
            else:
                # Try to find the model in the dict
                # Look for objects that might be the model
                for key, value in checkpoint.items():
                    if hasattr(value, '__dict__') and hasattr(value, 'forward'):
                        expert = value
                        break
                else:
                    # Assume the entire dict is the model
                    expert = checkpoint
        else:
            # Assume the loaded object is the model itself
            expert = checkpoint
        
        return expert
    
    def _reconstruct_likelihood(self, checkpoint):
        """
        Reconstruct likelihood from checkpoint
        Handles common PyTorch saving patterns
        """
        if hasattr(checkpoint, '__dict__') and hasattr(checkpoint, 'forward'):
            # This is the model itself, create default likelihood
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            print("Warning: Model saved without likelihood, using default GaussianLikelihood")
        elif isinstance(checkpoint, dict):
            if 'likelihood' in checkpoint:
                likelihood = checkpoint['likelihood']
            elif 'likelihood_state_dict' in checkpoint:
                likelihood = gpytorch.likelihoods.GaussianLikelihood()
                likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
            else:
                # Look for likelihood in the dict
                for key, value in checkpoint.items():
                    if hasattr(value, 'noise') or 'likelihood' in key.lower():
                        likelihood = value
                        break
                else:
                    # Default likelihood if not found
                    likelihood = gpytorch.likelihoods.GaussianLikelihood()
                    print("Warning: No likelihood found in checkpoint, using default GaussianLikelihood")
        else:
            # Default likelihood
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            print("Warning: Using default GaussianLikelihood")
        
        return likelihood
    
    def _load_colors(self):
        """Load colors for plotting"""
        try:
            FILE = Path(__file__).resolve()
            colors_path_name = FILE.parents[0] / 'colors.yml'
            
            with open(colors_path_name, 'r') as f:
                colors_dict = yaml.safe_load(f)
                self.c = colors_dict['color']
        except FileNotFoundError:
            # Default colors if file not found
            self.c = plt.cm.tab10(np.linspace(0, 1, 10)).tolist()
            # Extend with more colors if needed
            while len(self.c) < self.N_GPs:
                self.c.extend(plt.cm.tab20(np.linspace(0, 1, 20)).tolist())
    
    def set_scaler(self, scaler, expert_idx=None):
        """
        Manually set a scaler for unstandardization
        
        Parameters
        ----------
        scaler : sklearn scaler object
            The scaler to use for unstandardization
        expert_idx : int, optional
            Index of the expert to set the scaler for. If None, sets as global scaler
        """
        if expert_idx is None:
            # Set as global scaler
            self.global_scaler = scaler
            self.scalers = [scaler] * self.N_GPs
            print("Set global scaler for all experts")
        else:
            # Set scaler for specific expert
            if expert_idx < len(self.scalers):
                self.scalers[expert_idx] = scaler
                print(f"Set scaler for expert {expert_idx}")
            else:
                raise IndexError(f"Expert index {expert_idx} out of range")
    
    def _unstandardize_predictions(self, mu, sigma, expert_idx=None):
        """
        Unstandardize predictions using the appropriate scaler
        
        Parameters
        ----------
        mu : numpy.ndarray
            Standardized mean predictions
        sigma : numpy.ndarray
            Standardized standard deviation predictions
        expert_idx : int, optional
            Index of the expert (for individual scaler). If None, uses global scaler
            
        Returns
        -------
        mu_unstd : numpy.ndarray
            Unstandardized mean predictions
        sigma_unstd : numpy.ndarray
            Unstandardized standard deviation predictions
        """
        
        # Determine which scaler to use
        if expert_idx is not None and expert_idx < len(self.scalers):
            scaler = self.scalers[expert_idx]
        elif self.global_scaler is not None:
            scaler = self.global_scaler
        else:
            print("Warning: No scaler available for unstandardization")
            return mu, sigma
        
        if scaler is None:
            print("Warning: Scaler is None, returning standardized predictions")
            return mu, sigma
        
        try:
            # Unstandardize mean
            mu_unstd = mu.copy()
            
            # Handle different scaler types
            if hasattr(scaler, 'inverse_transform'):
                # For sklearn scalers
                mu_unstd = scaler.inverse_transform(mu_unstd.reshape(-1, 1)).flatten()
            elif hasattr(scaler, 'scale_') and hasattr(scaler, 'min_'):
                # Manual MinMaxScaler unstandardization
                mu_unstd = mu_unstd / scaler.scale_ - scaler.min_
            elif hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                # Manual StandardScaler unstandardization
                mu_unstd = mu_unstd * scaler.scale_ + scaler.mean_
            
            # Unstandardize standard deviation
            sigma_unstd = sigma.copy()
            
            # For standard deviation, we only need to scale (not shift)
            if hasattr(scaler, 'scale_'):
                if hasattr(scaler, 'min_'):
                    # MinMaxScaler: scale by the scaling factor
                    sigma_unstd = sigma_unstd / scaler.scale_
                else:
                    # StandardScaler: scale by the scaling factor
                    sigma_unstd = sigma_unstd * scaler.scale_
            
            return mu_unstd, sigma_unstd
            
        except Exception as e:
            print(f"Error during unstandardization: {e}")
            return mu, sigma
    
    def plot_expert(self, X_test, mu_all):
        """Plot the predictions of each expert"""
        plt.figure(figsize=(12, 8))
        plt.title('Expert predictions at each region')
        
        for i in range(self.N_GPs):
            color = self.c[i] if i < len(self.c) else plt.cm.tab10(i % 10)
            plt.plot(mu_all[:, i], color=color, label=f'SVGP({i})', alpha=0.7)
        
        plt.xlabel('Test Points')
        plt.ylabel('Predictions')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def add_expert(self, expert_path, scaler_path=None):
        """
        Add a new expert to the ensemble
        
        Parameters
        ----------
        expert_path : str or Path
            Path to the expert model file
        scaler_path : str or Path, optional
            Path to the scaler file for this expert
        """
        if not self.is_loaded:
            raise RuntimeError("Load experts first before adding new ones")
        
        try:
            checkpoint = torch.load(expert_path, map_location=self.device)
            expert = self._reconstruct_expert(checkpoint)
            likelihood = self._reconstruct_likelihood(checkpoint)
            
            expert.to(self.device)
            likelihood.to(self.device)
            expert.eval()
            likelihood.eval()
            
            self.experts.append(expert)
            self.likelihoods.append(likelihood)
            
            # Load scaler if provided
            scaler = None
            if scaler_path:
                try:
                    if Path(scaler_path).suffix == '.pkl':
                        with open(scaler_path, 'rb') as f:
                            scaler = pickle.load(f)
                    elif Path(scaler_path).suffix == '.pth':
                        scaler_data = torch.load(scaler_path, map_location='cpu')
                        scaler = self._reconstruct_scaler(scaler_data)
                except Exception as e:
                    print(f"Error loading scaler for new expert: {e}")
            
            self.scalers.append(scaler)
            self.N_GPs += 1
            
            print(f"Added expert. Total experts: {self.N_GPs}")
            
        except Exception as e:
            print(f"Error adding expert: {e}")
    
    def remove_expert(self, index):
        """Remove an expert from the ensemble"""
        if 0 <= index < self.N_GPs:
            del self.experts[index]
            del self.likelihoods[index]
            if index < len(self.scalers):
                del self.scalers[index]
            self.N_GPs -= 1
            print(f"Removed expert {index}. Total experts: {self.N_GPs}")
        else:
            raise IndexError(f"Expert index {index} out of range")
    
    def predict(self, X_star, unstandardize=True):
        """
        Make predictions using the gPoE approach
        
        Parameters
        ----------
        X_star : torch.Tensor or numpy.ndarray
            Test input points (should be standardized if experts were trained on standardized data)
        unstandardize : bool
            Whether to unstandardize the final predictions
        
        Returns
        -------
        mu_star : numpy.ndarray
            gPoE predictive mean (unstandardized if unstandardize=True)
        std_star : numpy.ndarray
            gPoE predictive standard deviation (unstandardized if unstandardize=True)
        betas : numpy.ndarray
            Expert weights/powers [N_star x N_GPs]
        """
        
        if not self.is_loaded:
            raise RuntimeError("No experts loaded")
        
        # Convert to tensor if necessary
        if isinstance(X_star, np.ndarray):
            X_star = torch.from_numpy(X_star).float().to(self.device)
        else:
            X_star = X_star.to(self.device)
        
        N_star = X_star.shape[0]
        mu_all = np.zeros([N_star, self.N_GPs])
        sigma_all = np.zeros([N_star, self.N_GPs])
        
        # Get predictions from all experts
        with torch.no_grad():
            for i in range(self.N_GPs):
                try:
                    # Get predictions from expert
                    expert_output = self.experts[i](X_star)
                    observed_pred = self.likelihoods[i](expert_output)
                    
                    # Extract mean and variance
                    mu_expert = observed_pred.mean.cpu().numpy()
                    sigma_expert = observed_pred.stddev.cpu().numpy()
                    
                    # Unstandardize individual expert predictions if needed
                    if unstandardize:
                        mu_expert, sigma_expert = self._unstandardize_predictions(
                            mu_expert, sigma_expert, expert_idx=i
                        )
                    
                    mu_all[:, i] = mu_expert
                    sigma_all[:, i] = sigma_expert
                    
                except Exception as e:
                    print(f"Error getting predictions from expert {i}: {e}")
                    # Set to default values if expert fails
                    mu_all[:, i] = np.zeros(N_star)
                    sigma_all[:, i] = np.ones(N_star)
        
        # Calculate the normalised predictive power (betas)
        betas = np.zeros([N_star, self.N_GPs])
        prior_std = 1 + 1e-6  # Add jitter term to prevent numeric error
        
        for i in range(self.N_GPs):
            # Ensure sigma values are positive
            sigma_all[:, i] = np.maximum(sigma_all[:, i], 1e-6)
            betas[:, i] = 0.5 * (np.log(prior_std) - np.log(sigma_all[:, i]**2))
        
        # Check if we have any valid experts
        if self.N_GPs == 0:
            raise RuntimeError("No valid experts available for prediction")
        
        # Normalise betas only if we have valid data
        if np.any(np.isfinite(betas)):
            scaler = MinMaxScaler(feature_range=(0, 1))
            betas = scaler.fit_transform(betas)
        else:
            print("Warning: All betas are invalid, using uniform weights")
            betas = np.ones_like(betas) / self.N_GPs
        
        # Eliminate beta values <= 0.5 (threshold for expert reliability)
        betas[betas <= 0.4] = 0
        
        # Compute the gPoE precision
        prec_star = np.zeros(N_star)
        for i in range(self.N_GPs):
            prec_star += betas[:, i] * sigma_all[:, i]**-2
        
        # Add small epsilon to avoid division by zero
        prec_star = np.maximum(prec_star, 1e-8)
        
        # Compute the gPoE predictive variance and standard deviation
        var_star = prec_star**-1
        std_star = var_star**0.5
        
        # Compute the gPoE predictive mean
        mu_star = np.zeros(N_star)
        for i in range(self.N_GPs):
            mu_star += betas[:, i] * sigma_all[:, i]**-2 * mu_all[:, i]
        mu_star *= var_star
        
        # Plot if specified
        if self.plot_expert_pred:
            self.plot_expert(X_star.cpu().numpy(), mu_all)
        
        return mu_star, std_star, betas
    
    def predict_with_uncertainty(self, X_star, return_individual=False, unstandardize=True):
        """
        Make predictions with detailed uncertainty information
        
        Parameters
        ----------
        X_star : torch.Tensor or numpy.ndarray
            Test input points (should be standardized if experts were trained on standardized data)
        return_individual : bool
            Whether to return individual expert predictions
        unstandardize : bool
            Whether to unstandardize the predictions
            
        Returns
        -------
        results : dict
            Dictionary containing:
            - 'mean': gPoE predictive mean
            - 'std': gPoE predictive standard deviation
            - 'betas': Expert weights
            - 'individual_means': Individual expert means (if return_individual=True)
            - 'individual_stds': Individual expert stds (if return_individual=True)
        """
        
        mu_star, std_star, betas = self.predict(X_star, unstandardize=unstandardize)
        
        results = {
            'mean': mu_star,
            'std': std_star,
            'betas': betas
        }
        
        if return_individual:
            # Convert to tensor if necessary
            if isinstance(X_star, np.ndarray):
                X_star = torch.from_numpy(X_star).float().to(self.device)
            else:
                X_star = X_star.to(self.device)
            
            N_star = X_star.shape[0]
            individual_means = np.zeros([N_star, self.N_GPs])
            individual_stds = np.zeros([N_star, self.N_GPs])
            
            with torch.no_grad():
                for i in range(self.N_GPs):
                    expert_output = self.experts[i](X_star)
                    observed_pred = self.likelihoods[i](expert_output)
                    
                    mu_expert = observed_pred.mean.cpu().numpy()
                    sigma_expert = observed_pred.stddev.cpu().numpy()
                    
                    # Unstandardize individual expert predictions if needed
                    if unstandardize:
                        mu_expert, sigma_expert = self._unstandardize_predictions(
                            mu_expert, sigma_expert, expert_idx=i
                        )
                    
                    individual_means[:, i] = mu_expert
                    individual_stds[:, i] = sigma_expert
            
            results['individual_means'] = individual_means
            results['individual_stds'] = individual_stds
        
        return results
    
    def get_scaler_info(self):
        """Get information about loaded scalers"""
        info = {
            'global_scaler': self.global_scaler is not None,
            'individual_scalers': [s is not None for s in self.scalers],
            'scaler_types': []
        }
        
        for i, scaler in enumerate(self.scalers):
            if scaler is not None:
                scaler_type = type(scaler).__name__
                info['scaler_types'].append(f"Expert {i}: {scaler_type}")
            else:
                info['scaler_types'].append(f"Expert {i}: None")
        
        return info