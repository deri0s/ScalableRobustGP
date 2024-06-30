import numpy as np
from sklearn import mixture as m
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
import torch
import gpytorch
from gpytorch.models import ExactGP, ApproximateGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.kernels import RBFKernel as RBF, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood, VariationalELBO
import types

"""
A Robust Sparse Gaussian Process regression aprroach based on Dirichlet Process
clustering and Gaussian Process Variational Sparse regression for scenarios where
the measurement noise is assumed to be generated from a mixture of Gaussian
distributions. The proposed class inherits attributes and methods from the
GPJAX classes.

Diego Echeverria Rios & P.L.Green
"""


class DirichletProcessSparseGaussianProcess():
    def __init__(self, X, Y, init_K,
                 gp_model=ExactGP,
                 mu0 = gpytorch.means.ConstantMean(),
                 kernel=RBF(),
                 likelihood=GaussianLikelihood(),
                 n_inducing = 15,
                 normalise_y=False, N_iter=8, DP_max_iter=70,
                 plot_conv=False, plot_sol=False):
        
                """ 
                    Initialising variables and parameters
                """
                
                # Initialisation of the variables related to the training data
                self.X_org = np.vstack(X)       # Required for DP clustering
                self.X = torch.tensor(X, dtype=torch.float32)
                self.Y = Y                      # Targets never vstacked
                self.N = len(Y)                 # No. training points
                self.D = self.X.dim()           # No. Dimensions
                self.n_inducing = n_inducing
                self.normalise_y = normalise_y  # Normalise data
                self.N_iter = N_iter            # Max number of iterations (DPGP)
                self.DP_max_iter = DP_max_iter  # Max number of iterations (DP)
                self.plot_conv = plot_conv
                self.plot_sol = plot_sol
                
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
                self.Y = torch.tensor(self.Y, dtype=torch.float32)

                self.likelihood = likelihood

                # Initialise GP model
                if gp_model.isinstance(ApproximateGP):
                    inducing_points = self.X[::n_inducing]
                    variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
                    model = VariationalStrategy(ApproximateGP, inducing_points=inducing_points,
                                                            variational_distribution=variational_distribution,
                                                            learn_inducing_locations=True)
                else:
                    self.gp_model = gp_model
                    self.model = gp_model(self.X, self.Y, self.likelihood)

                # Assign mean and covariance modules
                self.model.mean_module = mu0
                self.model.covar_module = kernel

                # Define the forward method
                def forward(self, x):
                    mean_x = self.mean_module(x)
                    covar_x = self.covar_module(x)
                    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

                # Dynamically add the forward method to the model
                self.model.forward = types.MethodType(forward, self.model)

                # Initialize hyperparameters (optional)
                self.likelihood.noise = 0.05

                # Train model
                self.model.train()
                self.likelihood.train()

                optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

                if gp_model.isinstance(ApproximateGP):
                    mll = VariationalELBO(self.likelihood, model, self.Y.numel())
                else:
                    mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

                for i in range(100):
                    optimizer.zero_grad()
                    if i == 1: print('en init: ', self.model(self.X))
                    output = self.model(self.X)
                    loss = -mll(output, self.Y)
                    loss.backward()
                    optimizer.step()

                self.mll_eval = loss.detach().numpy()

                # Print the estimated hyperparameters
                print("Lengthscale:", self.model.covar_module.base_kernel.lengthscale.item())
                print("Outputscale:", self.model.covar_module.outputscale.item())
                print("Noise:", self.likelihood.noise.item())

                # Predictions
                self.model.eval()
                self.likelihood.eval()
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    observed_pred = likelihood(self.model(self.X))
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
                    plt.plot(self.Y, 'o', color='black')
                    plt.plot(mu.numpy(), color='lightgreen', linewidth = 2)
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
        X_test = torch.tensor(X_test, dtype=torch.float32)

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
            X0 = torch.tensor(X0[:,0], dtype=torch.float32)
            Y0 = torch.tensor(Y0[:,0], dtype=torch.float32)

            self.gp = self.gp_model(X0, Y0, self.likelihood)
            self.gp.mean_module  = self.model.mean_module
            self.gp.covar_module = self.model.covar_module

            def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

            # Dynamically add the forward method to the model
            self.gp.forward = types.MethodType(forward, self.gp)

            # Initialize hyperparameters
            ls = self.model.covar_module.base_kernel.lengthscale.item()
            self.gp.covar_module.base_kernel.lengthscale = ls
            self.likelihood.noise = self.likelihood.noise.item()

            # Train model
            self.gp.train()
            self.likelihood.train()

            optimizer = torch.optim.Adam(self.gp.parameters(), lr=0.1)
            mll = ExactMarginalLogLikelihood(self.likelihood, self.gp)

            for conteo in range(100):
                optimizer.zero_grad()
                output = self.gp(X0)
                loss = -mll(output, Y0)
                loss.backward()
                optimizer.step()

            self.mll_eval = loss

            # Predictions
            self.gp.eval()
            self.likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.likelihood(self.gp(self.X))

                # ! Always vstack mu and not the errors (issues otherwise)
                pred_mean = observed_pred.mean
                mu = np.vstack(pred_mean.numpy())

            errors = self.Y_org - mu
            
            # Model convergence is controlled with the standard GP likelihood
            lnP[i+1] = loss.detach().numpy()

            print('Training...\n Iteration: ', i, ' tolerance: ', tolerance,
                  ' calculated(GP): ', abs(lnP[i+1] - lnP[i]), '\n')
            # Print the estimated hyperparameters
            print("Lengthscale:", self.model.covar_module.base_kernel.lengthscale.item())
            print("Outputscale:", self.model.covar_module.outputscale.item())
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
        
        # Return the unornalised values
        if self.normalise_y is True:
            for k in range(self.K_opt):
                self.stds[k] = self.stds[k] * self.Y_std