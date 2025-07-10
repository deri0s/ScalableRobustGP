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

"""
A distributed robust GP class using GPyTorch SVGP experts with gPoE approach.
Based on the generalised Product of Experts (gPoE) approach (Deisenroth and Wei Ng, 2015).
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
    
    def __init__(self, expert_dir, N_GPs=None, device='cpu', plot_expert_pred=False):
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
        """
        
        self.expert_dir = Path(expert_dir)
        self.device = device
        self.plot_expert_pred = plot_expert_pred
        
        # Load expert models
        self.experts = []
        self.likelihoods = []
        self._load_experts(N_GPs)
        
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
        if not model_files:
            model_files = sorted(list(self.expert_dir.glob('expert*.pt')))
        if not model_files:
            model_files = sorted(list(self.expert_dir.glob('expert_*.pt')))
        if not model_files:
            model_files = sorted(list(self.expert_dir.glob('expert_*.pth')))
        
        if not model_files:
            raise FileNotFoundError(f"No expert models found in {self.expert_dir}")
        
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
    
    def add_expert(self, expert_path):
        """Add a new expert to the ensemble"""
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
            self.N_GPs += 1
            
            print(f"Added expert. Total experts: {self.N_GPs}")
            
        except Exception as e:
            print(f"Error adding expert: {e}")
    
    def remove_expert(self, index):
        """Remove an expert from the ensemble"""
        if 0 <= index < self.N_GPs:
            del self.experts[index]
            del self.likelihoods[index]
            self.N_GPs -= 1
            print(f"Removed expert {index}. Total experts: {self.N_GPs}")
        else:
            raise IndexError(f"Expert index {index} out of range")
    
    def predict(self, X_star):
        """
        Make predictions using the gPoE approach
        
        Parameters
        ----------
        X_star : torch.Tensor or numpy.ndarray
            Test input points
        
        Returns
        -------
        mu_star : numpy.ndarray
            gPoE predictive mean
        std_star : numpy.ndarray
            gPoE predictive standard deviation
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
                    mu_all[:, i] = observed_pred.mean.cpu().numpy()
                    sigma_all[:, i] = observed_pred.stddev.cpu().numpy()
                    
                    print(f"Expert {i} - Mean shape: {observed_pred.mean.shape}, Std shape: {observed_pred.stddev.shape}")
                    
                except Exception as e:
                    print(f"Error getting predictions from expert {i}: {e}")
                    # Set to default values if expert fails
                    mu_all[:, i] = np.zeros(N_star)
                    sigma_all[:, i] = np.ones(N_star)
        
        print(f"mu_all shape: {mu_all.shape}, sigma_all shape: {sigma_all.shape}")
        
        # Calculate the normalised predictive power (betas)
        betas = np.zeros([N_star, self.N_GPs])
        prior_std = 1 + 1e-6  # Add jitter term to prevent numeric error
        
        for i in range(self.N_GPs):
            # Ensure sigma values are positive
            sigma_all[:, i] = np.maximum(sigma_all[:, i], 1e-6)
            betas[:, i] = 0.5 * (np.log(prior_std) - np.log(sigma_all[:, i]**2))
        
        print(f"betas shape before normalization: {betas.shape}")
        print(f"betas range: [{np.min(betas)}, {np.max(betas)}]")
        
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
        betas[betas <= 0.5] = 0
        
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
    
    def predict_with_uncertainty(self, X_star, return_individual=False):
        """
        Make predictions with detailed uncertainty information
        
        Parameters
        ----------
        X_star : torch.Tensor or numpy.ndarray
            Test input points
        return_individual : bool
            Whether to return individual expert predictions
            
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
        
        mu_star, std_star, betas = self.predict(X_star)
        
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
                    
                    individual_means[:, i] = observed_pred.mean.cpu().numpy()
                    individual_stds[:, i] = observed_pred.stddev.cpu().numpy()
            
            results['individual_means'] = individual_means
            results['individual_stds'] = individual_stds
        
        return results