import numpy as np
from sklearn import mixture as m
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel as RBF
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

"""
A Robust Sparse Gaussian Process regression approach based on Dirichlet Process
clustering and Sparse Gaussian Process regression for scenarios where
the measurement noise is assumed to be generated from a mixture of Gaussian
distributions. The proposed class inherits attributes and methods from the
GPyTorch classes.

- Normalise the features when doing Sparse GP regression. !Not working otherwise

Diego Echeverria Rios (Derios) & P.L.Green
"""
# Define the GP model
class SparseGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, mu0, kernel, noise_var):
        super(SparseGP, self).__init__(train_x, train_y, likelihood)
        likelihood.noise = noise_var
        self.mean_module = mu0
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

class DirichletProcessSparseGaussianProcess():
    def __init__(self, X, Y, init_K,
                 gp_model='Sparse',
                 prior_mean=gpytorch.means.ConstantMean(),
                 kernel=RBF(),
                 likelihood=GaussianLikelihood(),
                 lengthscale = 1.0,
                 noise_var = 0.05,
                 floating_point = torch.float32,
                 normalise_y=False, N_iter=8, DP_max_iter=70,
                 print_conv=False,
                 plot_conv=False, plot_sol=False):
        
        """
            Initialising variables and parameters
        """
        self.floating_point = floating_point
        
        # Convert data only once to minimize conversions
        if torch.is_tensor(X):
            self.X = X
            self.X_org = X.detach().numpy()
        else:
            self.X_org = np.asarray(X)
            self.X = torch.tensor(self.X_org, dtype=self.floating_point)

        # Handle Y inputs - avoid unnecessary conversions
        if torch.is_tensor(Y):
            self.Y = Y
            self.Y_org = Y.detach().numpy()
            if len(self.Y_org.shape) == 1:
                self.Y_org = self.Y_org.reshape(-1, 1)
        else:
            self.Y_org = np.asarray(Y)
            if len(self.Y_org.shape) == 1:
                self.Y_org = self.Y_org.reshape(-1, 1)
            self.Y = torch.tensor(self.Y_org, dtype=self.floating_point)
            if len(self.Y.shape) > 1 and self.Y.shape[1] == 1:
                self.Y = self.Y.squeeze(1)

        self.N = len(Y)                 # No. training points
        self.D = self.X.shape[-1]       # No. Dimensions
        self.normalise_y = normalise_y  # Normalise data
        self.mu0 = prior_mean
        self.kernel = kernel
        self.lengthscale = lengthscale
        self.N_iter = N_iter            # Max number of iterations (DPSGP)
        self.DP_max_iter = DP_max_iter  # Max number of iterations (DP)
        self.print_conv = print_conv    # Print hyperparameter estimation at each step
        self.plot_conv = plot_conv      # Plot neg-margigal-log-likelihood
        self.plot_sol = plot_sol        # Plot clustering at each step
        self.gp_model = gp_model        # Standard or Sparse GP regression for now
        
        # The upper bound of the number of Gaussian noise sources
        self.init_K = init_K

        # Standardise data if specified
        if self.normalise_y is True:
            self.Y_mu = float(np.mean(self.Y_org))
            self.Y_std = float(np.std(self.Y_org))
            self.Y_org = (self.Y_org - self.Y_mu) / self.Y_std
            # Update tensor version too
            self.Y = torch.tensor((Y - self.Y_mu) / self.Y_std, dtype=self.floating_point)
            if len(self.Y.shape) > 1 and self.Y.shape[1] == 1:
                self.Y = self.Y.squeeze(1)

        self.likelihood = likelihood

        self.model = SparseGP(self.X, self.Y,
                              self.likelihood,
                              self.mu0, self.kernel, noise_var)

        # Initialise kernel parameters
        if self.gp_model == 'Sparse':
            if np.isscalar(self.lengthscale):
                self.model.covar_module.base_kernel.base_kernel.lengthscale = self.lengthscale 
            else:
                assert len(self.lengthscale) == self.D, "Input dimension different from lengthscale vector size"
                self.model.covar_module.base_kernel.base_kernel.lengthscale = torch.tensor(np.array(self.lengthscale))
        else:
            if np.isscalar(self.lengthscale):
                self.model.covar_module.base_kernel.lengthscale = self.lengthscale 
            else:
                assert len(self.lengthscale) == self.D, "Input dimension different from lengthscale vector size"
                self.model.covar_module.base_kernel.lengthscale = torch.tensor(np.array(self.lengthscale))

        # Train model with early stopping
        self.model.train()
        self.likelihood.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        training_iterations = 100
        previous_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for count in range(training_iterations):
            optimizer.zero_grad()
            output = self.model(self.X)
            loss = -mll(output, self.Y)
            loss.backward()
            optimizer.step()
            
            # Check for convergence
            current_loss = loss.item()
            if abs(previous_loss - current_loss) < 1e-5:
                patience_counter += 1
                if patience_counter >= patience:
                    if self.print_conv:
                        print(f"Early stopping at iteration {count} due to convergence")
                    break
            else:
                patience_counter = 0
            
            previous_loss = current_loss

        # Save estimated hyperparameters
        self.update_ls(self.model)

        # Print the estimated hyperparameters?
        if self.print_conv:
            print('\nThe very first estimated hyperparameters')
            self.print_hyper(self.model)

        # model evaluation
        self.mll_eval = loss.detach().numpy()

        # Predictions - reuse tensor data to avoid conversion
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(self.X))
            mu = observed_pred.mean

        # Initialise the residuals and initial GP hyperparameters
        self.init_errors = mu.detach().numpy().reshape(-1, 1) - self.Y_org
        
        # Plot solution
        self.x_axis = np.linspace(0, len(Y), len(Y))

        if self.plot_sol:
            fig, ax = plt.subplots()
            plt.rcdefaults()
            plt.rc('xtick', labelsize=14)
            plt.rc('ytick', labelsize=14)
            if gp_model == 'Sparse':
                _z = self.model.covar_module.inducing_points.detach().numpy()
                ax.vlines(
                    x=self.X[::10],
                    ymin=self.Y.min().item(),
                    ymax=self.Y.max().item(),
                    alpha=0.3,
                    linewidth=1.5,
                    ls='--',
                    label="z0",
                    color='grey'
                )
                ax.vlines(
                    x=_z,
                    ymin=self.Y.min().item(),
                    ymax=self.Y.max().item(),
                    alpha=0.3,
                    linewidth=1.5,
                    label="z*",
                    color='orange'
                )
            plt.plot(self.X, self.Y, 'o', color='black')
            plt.plot(self.X, mu.numpy(), color='lightgreen', linewidth = 2)
            plt.title('First GP approximation')
            ax.set_xlabel(" Date-time", fontsize=14)
            ax.set_ylabel(" Fault density", fontsize=14)
            plt.legend(loc=0, prop={"size":18}, facecolor="white",
                        framealpha=1.0)
                    
    def update_ls(self, gp):
        """Update lengthscale parameters from the GP model"""
        if self.gp_model == 'Sparse':
            if np.isscalar(self.lengthscale):
                self.lengthscale = gp.covar_module.base_kernel.base_kernel.lengthscale.item()
            else:
                self.lengthscale = gp.covar_module.base_kernel.base_kernel.lengthscale.tolist()
        else:
            if np.isscalar(self.lengthscale):
                self.lengthscale = gp.covar_module.base_kernel.lengthscale.item()
            else:
                self.lengthscale = gp.covar_module.base_kernel.lengthscale.tolist()

    def print_hyper(self, gp):
        if self.gp_model == 'Sparse':
            print("Outputscale:", gp.covar_module.base_kernel.outputscale.item())
            if np.isscalar(self.lengthscale):
                print("Lengthscale:", gp.covar_module.base_kernel.base_kernel.lengthscale.item())
            else:
                print("Lengthscale:", gp.covar_module.base_kernel.base_kernel.lengthscale.tolist())
        else:
            print("Outputscale:", gp.covar_module.outputscale.item())
            if np.isscalar(self.lengthscale):
                print("Lengthscale:", gp.covar_module.base_kernel.lengthscale.item())
            else:
                print("Lengthscale:", gp.covar_module.base_kernel.lengthscale.tolist())
        print("Noise:", self.likelihood.noise.item(), '\n')

    def plot_convergence(self, lnP, title):
        plt.figure()
        # Fix the dimension issue - lnP is 1D so we don't need axis parameter
        mask = lnP != 0.0
        ll = lnP[mask]
        plt.plot(ll, color='blue')
        plt.title(title, fontsize=17)
        plt.xlabel('Iterations', fontsize=17)
        plt.ylabel('- Marg-log-likelihood', fontsize=17)
        self.convergence = ll
        
    def plot_solution(self, K, indices, mu, iter):
        color_iter = ['lightgreen', 'orange','red', 'brown','black']

        enumerate_K = [i for i in range(K)]

        fig, ax = plt.subplots()
        # Increase the size of the axis numbers
        plt.rcdefaults()
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)
        
        fig.autofmt_xdate()
        ax.set_title("DPSGP: Clustering performance, Iteration "+str(iter),
                     fontsize=18)
        if K != 1:
            for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
                ax.plot(self.x_axis[indices[k]], self.Y[indices[k]],
                        'o',color=c, markersize = 8,
                        label='Noise Level '+str(k))
        ax.plot(self.x_axis, mu, color="green", linewidth = 2, label=" DPGP")
        ax.set_xlabel(" Date-time", fontsize=14)
        ax.set_ylabel(" Fault density", fontsize=14)
        plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

    def get_z_indices(self, x, inducing_inputs):
        """Optimized version of the inducing point index finder"""
        # Use numpy's searchsorted for faster lookup
        indices = np.zeros(len(inducing_inputs), dtype=int)
        for i, val in enumerate(inducing_inputs):
            indices[i] = np.argmin(np.abs(x - val))
        
        # Get unique indices more efficiently
        unique_indices = np.unique(indices)
        return unique_indices
        
    def gmm_loglikelihood(self, y, f, sigmas, pies, K):
        """
        The log-likelihood of a finite mixture model that is
        evaluated once the f (mus), pies, and sigmas has been estimated.
        This is the function that we evaluate for the model convergence.
        
        Improved to handle numerical stability.
        """
        # More numerically stable approach using log probabilities
        log_probs = np.zeros((len(y), K))
        
        for k in range(K):
            # Calculate log probabilities to avoid underflow
            log_probs[:, k] = np.log(pies[k]) + mvn.logpdf(y, f, sigmas[k]**2)
            
        # Use logsumexp for numerical stability
        from scipy.special import logsumexp
        loglikelihood = logsumexp(log_probs, axis=1)
        return np.exp(loglikelihood)  # Convert back to probabilities

    def DP(self, X, Y, errors, T):
        """
            Dirichlet Process mixture model for clustering.
            
            Inputs
            ------
            - T: The upper limit of the number of noise sources (clusters)
            
            Returns
            -------
            - Indices: The indices of the clustered observations.
            
            - X0, Y0: Pair of inputs and outputs associated with the
                        Gaussian of narrowest width.
                        
            - resp[0]: The responsibility vector of the Gaussian with the
                        narrowest width
                        
            - pies: The mixture proportionalities.
            
            - K_opt: The number of components identified in the mixture.
        """
        # Cache common calculations
        gmm = m.BayesianGaussianMixture(
            n_components=T,
            covariance_type='spherical',
            max_iter=self.DP_max_iter,
            weight_concentration_prior_type='dirichlet_process',
            init_params="random",
            random_state=42,
            n_init=1  # Reduce redundant initializations
        )
                
        # The data labels correspond to the position of the mix parameters
        labels = gmm.fit_predict(errors)
        
        # Capture the pies: It is a tuple with not ordered elements
        pies_no = np.sort(gmm.weights_)
        
        # Capture the sigmas
        covs = gmm.covariances_.reshape(1, gmm.n_components)[0]
        stds_no = np.sqrt(covs)
        
        # Get the width of each Gaussian 
        not_ordered = np.sqrt(gmm.covariances_)
        
        # Initialise the ordered pies, sigmas and responsibilities
        pies = np.zeros(gmm.n_components)
        stds = np.zeros(gmm.n_components)
        resp_no = gmm.predict_proba(errors)
        resp = []
        
        # Order the Gaussian components by their width
        order = np.argsort(not_ordered)
        
        indx = []    
        # The 0 position or first element of the 'order' vector corresponds
        # to the Gaussian with the min(std0, std1, std2, ..., stdk)
        for new_order in range(gmm.n_components):
            pies[new_order] = pies_no[order[new_order]]
            stds[new_order] = stds_no[order[new_order]]
            resp.append(resp_no[:, order[new_order]])
            indx.append(np.where(labels == order[new_order])[0])
        
        # The ensemble task has to account for empty subsets - more efficient check
        indices = [x for x in indx if len(x) > 0]
        K_opt = len(indices)         # The optimum number of components
        
        # Only extract data for non-empty clusters
        if K_opt > 0:
            X0 = X[indices[0]]
            Y0 = Y[indices[0]]
        else:
            X0 = X
            Y0 = Y
            indices = [np.arange(len(X))]
            K_opt = 1
    
        return indices, X0, Y0, resp[0], pies, stds, K_opt
    
    def predict(self, X_test):
        """ X_test: Standardised features at test locations """
        # Avoid unnecessary conversion
        if not torch.is_tensor(X_test):
            X_test = torch.tensor(X_test, dtype=self.floating_point)

        self.gp.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.gp(X_test))
            mu_norm = observed_pred.mean.numpy()
            std_norm = observed_pred.stddev.numpy()

        # Return the un-standardised calculations if required
        if self.normalise_y is True:
            mu = self.Y_std * mu_norm + self.Y_mu
            var = self.Y_std * std_norm
            return mu, var
        else:
            return mu_norm, std_norm

    def train(self, tol=12):
        """
            The present algorithm first performs clustering with a 
            Dirichlet Process mixture model (DP method).
            Then, it uses the inferred noise structure to train a standard GP
            
            Estimates
            ---------
            
            - indices: The indices of the clustered observations, equivalent
                        to estimate the latent variables Z (Paper Diego).
                        
            - pies: Mixture proportionalities.
            
            - K_opt: The number of components in the mixture.
            
            - hyperparameters: The optimum GP kernel hyperparameters
        """
        
        # Initialise variables and parameters
        errors = self.init_errors  # The residuals 
        K0 = self.init_K           # K upper bound
        max_iter = self.N_iter     # Prevent infinite loop
        i = 0                      # Count the number of iterations

        noise_var = self.likelihood.noise.item()
        # Only allocate what we need
        lnP = np.zeros(max_iter)
        
        # The log-likelihood(s) with the initial hyperparameters
        lnP[i] = self.mll_eval
        
        # Calculate tolerance based on initial log-likelihood
        tolerance = abs(lnP[0] * tol) / 700
        
        while i < max_iter:
            """ CLUSTERING """
            index, X0, Y0, resp0, pies, stds, K = self.DP(self.X_org, self.Y_org,
                                                          errors, K0)
            
            # In case I want to know the initial mixture parameters
            if i == 1:
                self.init_sigmas = stds
                self.init_pies = pies
                
            K0 = self.init_K
            self.resp = resp0

            """ REGRESSION """
            # Assemble training data - efficiently convert to tensors
            X0_tensor = torch.tensor(X0, dtype=self.floating_point)
            if Y0.shape[1] == 1:
                Y0_tensor = torch.tensor(Y0.reshape(-1), dtype=self.floating_point)
            else:
                Y0_tensor = torch.tensor(Y0, dtype=self.floating_point)

            self.gp = SparseGP(X0_tensor, Y0_tensor,
                               self.likelihood,
                               self.mu0, self.kernel, noise_var)
            
            # Initialize kernel parameters consistently
            if self.gp_model == 'Sparse':
                self.gp.covar_module.base_kernel.base_kernel.outputscale = 1
                self.gp.covar_module.base_kernel.base_kernel.lengthscale = self.lengthscale
            else:
                self.gp.covar_module.base_kernel.outputscale = 1
                self.gp.covar_module.base_kernel.lengthscale = self.lengthscale
            
            # Train model with early stopping
            self.gp.train()
            self.likelihood.train()

            optimizer = torch.optim.Adam(self.gp.parameters(), lr=0.01)
            mll = ExactMarginalLogLikelihood(self.likelihood, self.gp)

            previous_loss = float('inf')
            patience = 5
            patience_counter = 0
            
            for conteo in range(100):
                optimizer.zero_grad()
                output = self.gp(X0_tensor)
                loss = -mll(output, Y0_tensor)
                loss.backward()
                optimizer.step()
                
                # Check for convergence
                current_loss = loss.item()
                if abs(previous_loss - current_loss) < 1e-5:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if self.print_conv:
                            print(f"Early stopping GP training at iteration {conteo}")
                        break
                else:
                    patience_counter = 0
                
                previous_loss = current_loss

            # Update kernel parameters after training - THIS IS CRITICAL
            self.update_ls(self.gp)

            # Predictions - avoid unnecessary tensor conversions
            self.gp.eval()
            self.likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.likelihood(self.gp(self.X))
                pred_mean = observed_pred.mean.detach().numpy()
                
                # Reshape prediction to match error dimensions
                mu = pred_mean.reshape(-1, 1)

            # Calculate residuals
            errors = self.Y_org - mu
            
            # Update noise variance
            noise_var = self.likelihood.noise.item()
            
            # Model convergence is controlled with the standard GP likelihood
            lnP[i+1] = loss.detach().numpy()

            if self.print_conv:
                print('\nTraining...\n Iteration: ', i, ' tolerance: ', tolerance,
                      ' calculated(GP): ', abs(lnP[i+1] - lnP[i]), '\n')
                self.print_hyper(self.gp)

            if self.plot_sol:
                self.plot_solution(K, index, mu, i)
                
            # Check convergence
            if abs(lnP[i+1] - lnP[i]) < tolerance:
                print('\n Model trained')
                break
                
            i += 1
            
            if i == max_iter:
                print('\n The model did not converge after ', max_iter,
                      ' iterations')
                        
        # If specified, plot model convergence
        if self.plot_conv:
            # Only use the elements of lnP that were actually filled
            valid_indices = min(i+2, len(lnP))
            self.plot_convergence(lnP[:valid_indices], 'DPSGP: Regression step convergence')
            
        # Capture and save the estimated parameters
        index, X0, Y0, resp0, pies, stds, K = self.DP(self.X_org, self.Y_org,
                                                      errors, K)
        self.indices = index
        self.resp = resp0
        self.pies = pies
        self.stds = stds
        self.K_opt = K

        # get estimated hyperparameters
        if self.gp_model == 'Sparse':
            self._z_normalised = self.gp.covar_module.inducing_points.detach().numpy()
            self._z_indices = self.get_z_indices(X0[:, 0], self._z_normalised[:, 0])

        # Return the unornalised values
        if self.normalise_y is True:
            for k in range(self.K_opt):
                self.stds[k] = self.stds[k] * self.Y_std