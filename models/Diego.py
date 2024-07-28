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
A Robust Sparse Gaussian Process regression aprroach based on Dirichlet Process
clustering and Sparse Gaussian Process regression for scenarios where
the measurement noise is assumed to be generated from a mixture of Gaussian
distributions. The proposed class inherits attributes and methods from the
GPyTorch classes.

!Note:
- Sometimes normalising the inputs produces worst results compared to using the
un-normalised inputs when doing standard GP regression

- Normalise the features when doing Sparse GP regression. Not working otherwise

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
                 noise_var = 0.05,
                 normalise_y=False, N_iter=8, DP_max_iter=70,
                 print_conv=False,
                 plot_conv=False, plot_sol=False):
        
                """ 
                    Initialising variables and parameters
                """
                self.X_org = np.vstack(X)       # Required for DP clustering
                self.X = torch.tensor(X, dtype=torch.float64)
                self.Y = Y                      # Targets never vstacked
                self.N = len(Y)                 # No. training points
                self.D = self.X.dim()           # No. Dimensions
                self.normalise_y = normalise_y  # Normalise data
                self.mu0 = prior_mean
                self.kernel = kernel
                self.N_iter = N_iter            # Max number of iterations (DPGP)
                self.DP_max_iter = DP_max_iter  # Max number of iterations (DP)
                self.print_conv = print_conv    # Print hyperparameter estimation at each step
                self.plot_conv = plot_conv      # Plot neg-margigal-log-likelihood
                self.plot_sol = plot_sol        # Plot clustering at each step
                self.gp_model = gp_model        # Standard or Sparse GP regression for now
                
                # The upper bound of the number of Gaussian noise sources
                self.init_K = init_K

                # Standardise data if specified
                if self.normalise_y is True:
                    self.Y_mu = np.mean(self.Y)
                    self.Y_std = np.std(self.Y)
                    self.Y = (self.Y - self.Y_mu) / self.Y_std

                # Required for DP clustering
                self.Y_org = np.vstack(self.Y)

                # convert y to tensor for torch
                self.Y = torch.tensor(self.Y, dtype=torch.float64)

                self.likelihood = likelihood

                self.model = SparseGP(self.X, self.Y,
                                      self.likelihood,
                                      self.mu0, self.kernel, noise_var)

                # Train model
                self.model.train()
                self.likelihood.train()

                optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
                mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

                training_iterations = 100
                for count in range(training_iterations):
                    optimizer.zero_grad()
                    output = self.model(self.X)
                    loss = -mll(output, self.Y)
                    loss.backward()
                    optimizer.step()

                # Print the estimated hyperparameters?
                if self.print_conv:
                    print('\nThe very first estimated hyperparameters')
                    if self.gp_model == 'Sparse':
                        print("Outputscale:", self.model.covar_module.base_kernel.outputscale.item())
                        print("Lengthscale:", self.model.covar_module.base_kernel.base_kernel.lengthscale.item())
                    else:
                        print("Outputscale:", self.model.covar_module.outputscale.item())
                        print("Lengthscale:", self.model.covar_module.base_kernel.lengthscale.item())
                    print("Noise:", self.likelihood.noise.item(), '\n')

                # model evaluation
                self.mll_eval = loss.detach().numpy()

                # Predictions
                self.model.eval()
                self.likelihood.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    observed_pred = self.likelihood(self.model(self.X))
                    mu = observed_pred.mean

                # Initialise the residuals and initial GP hyperparameters
                self.init_errors = np.vstack(mu.numpy()) - self.Y_org
                
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
                                    
    def plot_convergence(self, lnP, title):
        plt.figure()
        ll = lnP[~np.all(lnP== 0.0, axis=1)]
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
        ax.set_title("DPSGP: Clustering performance, Iteration "+str(iter), fontsize=18)
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
        closest_values = np.zeros_like(inducing_inputs)
        indices = np.zeros_like(x, dtype=int)

        for i, val1 in enumerate(inducing_inputs):
            closest_idx = np.argmin(np.abs(x - val1))
            closest_values[i] = x[closest_idx]
            indices[i] = closest_idx

        unique_indices = np.unique(closest_values, return_index=True)[1]
        return indices[unique_indices][:,0]
        
        
    def gmm_loglikelihood(self, y, f, sigmas, pies, K):
        """
        The log-likelihood of a finite mixture model that is
        evaluated once the f, pies, and sigmas has been estimated.
        This is the function that we evaluate for the model convergence.
        """

        temp_sum = 0
        for k in range(K):
            temp_sum += pies[k] * mvn.pdf(y, f, sigmas[k]**2)
        loglikelihood = temp_sum
        return loglikelihood

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

        gmm =m.BayesianGaussianMixture(n_components=T,
                                       covariance_type='spherical',
                                       max_iter=self.DP_max_iter,            # original 70
                                       weight_concentration_prior_type='dirichlet_process',
                                       init_params="random",
                                       random_state=42)
                
        # The data labels correspond to the position of the mix parameters
        labels = gmm.fit_predict(errors)
        
        # Capture the pies: It is a tuple with not ordered elements
        pies_no = np.sort(gmm.weights_)
        
        # Capture the sigmas
        covs = np.reshape(gmm.covariances_, (1, gmm.n_components))
        covariances = covs[0]
        stds_no = np.sqrt(covariances)
        
        # Get the width of each Gaussian 
        not_ordered = np.array(np.sqrt(gmm.covariances_))
        
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
            indx.append([i for (i, val) in enumerate(labels) if val == order[new_order]])
        
        # The ensemble task has to account for empty subsets.                
        indices = [x for x in indx if x != []]
        K_opt = len(indices)         # The optimum number of components
        X0 = X[indices[0]]
        Y0 = Y[indices[0]]
    
        return indices, X0, Y0, resp[0], pies, stds, K_opt
    
    def predict(self, X_test):
        """
        X_test:     Normalised features at test locations
        """

        X_test = torch.tensor(X_test, dtype=torch.float64)

        self.gp.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.gp(X_test))
            # Dont vstack
            pred_mean = observed_pred.mean
            lower_norm, upper_norm  = observed_pred.confidence_region()
            mu = pred_mean.numpy()

        # Return the un-standardised calculations if required
        if self.normalise_y is True:
            mu = self.Y_std * mu + self.Y_mu
            lower = self.Y_std * lower_norm.numpy() + self.Y_mu
            upper = self.Y_std * upper_norm.numpy() + self.Y_mu

        return mu, lower, upper

    def train(self, tol=12):
        """
            The present algorithm first performs clustering with a 
            Dirichlet Process mixture model (DP method).
            Then, it uses the inferred noise structure to train a standard
            GP. The OMGP predictive distribution is used to incorporate the
            responsibilities in realise new estimates of the latent function.
            
            Estimates
            ---------
            
            - indices: The indices of the clustered observations, equivalent
                        to estimate the latent variables Z (Paper Diego).
                        
            - pies: Mixture proportionalities.
            
            - K_opt: The number of components in the mixture.
            
            - hyperparameters: The optimum GP kernel hyperparameters
        """
        
        # Initialise variables and parameters
        errors = self.init_errors   # The residuals 
        K0 = self.init_K            # K upper bound
        max_iter = self.N_iter      # Prevent infinite loop
        i = 0                       # Count the number of iterations

        noise_var = self.likelihood.noise.item()
        lnP = np.zeros((3*max_iter,1))
        lnP[1] = float('inf')
        
        # The log-likelihood(s) with the initial hyperparameters
        lnP[i] = self.mll_eval
        
        # Stop if the change in the log-likelihood is no > than 10% of the 
        # log-likelihood evaluated with the initial hyperparameters
        tolerance = abs(lnP[0]*tol)/100
        
        while i < max_iter:
            """
            CLUSTERING
            """
            index, X0, Y0, resp0, pies, stds, K = self.DP(self.X_org, self.Y_org,
                                                          errors, K0)
            
            # In case I want to know the initial mixure parameters
            if i == 1:
                self.init_sigmas = stds
                self.init_pies = pies
                
            K0 = self.init_K
            self.resp = resp0

            """
            REGRESSION
            """
            # Assemble training data
            X0 = torch.tensor(X0, dtype=torch.float64)
            Y0 = torch.tensor(Y0[:,0], dtype=torch.float64)

            self.gp = SparseGP(X0, Y0,
                               self.likelihood,
                               self.mu0, self.kernel, noise_var)
            
            # Train model
            self.gp.train()
            self.likelihood.train()

            optimizer = torch.optim.Adam(self.gp.parameters(), lr=0.01)
            mll = ExactMarginalLogLikelihood(self.likelihood, self.gp)

            for conteo in range(100):
                optimizer.zero_grad()
                output = self.gp(X0)
                loss = -mll(output, Y0)
                loss.backward()
                optimizer.step()

            # Predictions
            self.gp.eval()
            self.likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.likelihood(self.gp(self.X))

                # ! Always vstack mu and not the errors (issues otherwise)
                pred_mean = observed_pred.mean
                mu = np.vstack(pred_mean.numpy())

            errors = self.Y_org - mu
            # Update hyperparameters
            noise_var = self.likelihood.noise.item()
            
            # Model convergence is controlled with the standard GP likelihood
            lnP[i+1] = loss.detach().numpy()

            if self.print_conv:
                print('\nTraining...\n Iteration: ', i, ' tolerance: ', tolerance,
                      ' calculated(GP): ', abs(lnP[i+1] - lnP[i]), '\n')
                if self.gp_model=='Sparse':
                    print("Outputscale:", self.gp.covar_module.base_kernel.outputscale.item())
                    print("Lengthscale:", self.gp.covar_module.base_kernel.base_kernel.lengthscale.item())
                else:
                    print("Outputscale:", self.gp.covar_module.outputscale.item())
                    print("Lengthscale:", self.gp.covar_module.base_kernel.lengthscale.item())
                print("Noise:", self.likelihood.noise.item())


            if self.plot_sol:
                self.plot_solution(K, index, mu, i)
                
            if abs(lnP[i+1] - lnP[i]) < tolerance:
                print('\n Model trained')
                break
                
            i += 1
            
            if i == max_iter:
                print('\n The model did not converge after ', max_iter,
                      ' iterations')
                        
        # If specified, plot model convergence
        if self.plot_conv:
            self.plot_convergence(lnP, 'DPSGP: Regression step convergence')
            
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
            self._z_indices = self.get_z_indices(X0, self._z_normalised)

        # Return the unornalised values
        if self.normalise_y is True:
            for k in range(self.K_opt):
                self.stds[k] = self.stds[k] * self.Y_std