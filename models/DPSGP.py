import numpy as np
from sklearn import mixture as m
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel as RBF, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

"""
A Robust Sparse Gaussian Process regression aprroach based on Dirichlet Process
clustering and Gaussian Process Variational Sparse regression for scenarios where
the measurement noise is assumed to be generated from a mixture of Gaussian
distributions. The proposed class inherits attributes and methods from the
GPJAX classes.

Diego Echeverria Rios & P.L.Green
"""


class DirichletProcessSparseGaussianProcess(ExactGP):
    def __init__(self, X, Y, init_K,
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
                self.X = torch.tensor(np.vstack(X), dtype=torch.float32)
                self.Y = np.vstack(Y)           # Targets always vstacked
                self.N = len(Y)                 # No. training points
                self.D = self.X.dim()           # No. Dimensions
                self.N_iter = N_iter            # Max number of iterations (DPGP)
                self.DP_max_iter = DP_max_iter  # Max number of iterations (DP)
                self.plot_conv = plot_conv
                self.plot_sol = plot_sol
                
                # The upper bound of the number of Gaussian noise sources
                self.init_K = init_K            
                
                # Initialisation of the GPR attributes
                self.mu0 = mu0
                self.kernel = ScaleKernel(RBF())

                def forward(self, x):
                    mean_x = self.mu0(x)
                    covar_x = self.kernel(x)
                    return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
                
                self.normalise_y = normalise_y
                # Standardise data if specified
                if self.normalise_y is True:
                    self.Y_mu = np.mean(self.Y)
                    self.Y_std = np.std(self.Y)
                    self.Y = (self.Y - self.Y_mu) / self.Y_std

                # convert to tensor for torch
                self.Y = torch.tensor(self.Y, dtype=torch.float32)
                
                # Initialise a GPR class
                self.likelihood = likelihood
                model = super(DirichletProcessSparseGaussianProcess, self).__init__(self.X, self.Y, likelihood)

                # Initialize hyperparameters (optional)
                model.kernel.base_kernel.lengthscale = 0.9
                likelihood.noise = 0.05

                # Training
                super.train()
                likelihood.train()
                optimizer = torch.optim.Adam(super.parameters(), lr=0.1)
                mll = ExactMarginalLogLikelihood(likelihood, model)

                for i in range(100):
                    optimizer.zero_grad()
                    output = model(X)
                    loss = -mll(output, y_normalised)
                    loss.backward()
                    optimizer.step()

                self.prior = gpx.gps.Prior(mean_function=self.mu0, kernel=self.kernel)

                # Likelihood
                likelihood = gpx.likelihoods.Gaussian(num_datapoints=self.N)

                # posterior
                self.posterior = self.prior * likelihood

                # Define the training objective
                self.negative_mll = gpx.objectives.ConjugateMLL(negative=True)

                print('Before training: ', self.posterior.prior.kernel)

                self.opt_posterior, _ = gpx.fit_scipy(
                                model=self.posterior,
                                objective=self.negative_mll,
                                train_data=self.data
                            )
                
                print('After training: ', self.opt_posterior.prior.kernel)

                # estimated hyperparameters
                print('am: ', self.opt_posterior.prior.kernel.kernels[0].variance)
                print('ls: ', self.opt_posterior.prior.kernel.kernels[0].lengthscale)
                print('vr: ', self.opt_posterior.prior.kernel.kernels[1].variance)

                # If n_inducing points is close to N, the model will not return
                # accurate solutions.
                # z = np.linspace(self.X.min(), self.X.max(), n_inducing).reshape(-1, 1)

                # Predictions
                latent_dist = self.opt_posterior(X, train_data=self.data)
                predictive_dist = self.opt_posterior.likelihood(latent_dist)

                # ! Always vstack the predictions and not the errors
                # (critical error otherwise)
                mu = np.vstack(predictive_dist.mean())
                
                # Initialise the residuals and initial GP hyperparameters
                self.init_errors = mu - self.Y
                
                # Plot solution
                self.x_axis = np.linspace(0, len(Y), len(Y))
                
                if self.plot_sol:
                    fig, ax = plt.subplots()
                    plt.rcdefaults()
                    plt.rc('xtick', labelsize=14)
                    plt.rc('ytick', labelsize=14)
                    plt.plot(self.Y, 'o', color='black')
                    plt.plot(predictive_dist.mean(), color='lightgreen', linewidth = 2)
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
        plt.ylabel('log-likelihood', fontsize=17)
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
        latent_dist = self.opt_posterior(X_test, self.data)
        predictive_dist = self.opt_posterior.likelihood(latent_dist)
        
        muf = predictive_dist.mean()
        stdf = predictive_dist.stddev()

        # Return the un-standardised calculations if required
        if self.normalise_y is True:
            mu = self.Y_std * muf + self.Y_mu
            std = self.Y_std * stdf

        return mu, std

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
        opt_posterior = self.opt_posterior
        errors = self.init_errors   # The residuals 
        K0 = self.init_K            # K upper bound
        max_iter = self.N_iter      # Prevent infinite loop
        i = 0                       # Count the number of iterations
        lnP = np.zeros((3*max_iter,1))
        lnP[1] = float('inf')
        
        # The log-likelihood(s) with the initial hyperparameters
        lnP[i] = -self.negative_mll(opt_posterior, self.data)
        # lnP[i] = 0
        
        # Stop if the change in the log-likelihood is no > than 10% of the 
        # log-likelihood evaluated with the initial hyperparameters
        tolerance = abs(lnP[0]*tol)/1000
        
        while i < max_iter:
            # The clustering step
            index, X0, Y0, resp0, pies, stds, K = self.DP(self.X, self.Y,
                                                          errors, K0)
            
            # In case I want to know the initial mixure parameters
            if i == 1:
                self.init_sigmas = stds
                self.init_pies = pies
                
            K0 = self.init_K
            self.resp = resp0

            # Assemble training dataset
            D0 = gpx.Dataset(X=X0, y=Y0)
            
            # The regression step
            likelihood = gpx.likelihoods.Gaussian(num_datapoints=D0.n)

            # posterior
            self.posterior = self.prior * likelihood

            # Initialise hyperparameters

            negative_mll = gpx.objectives.ConjugateMLL(negative=True)
            negative_mll(self.posterior, train_data=D0)

            opt_posterior, self.history = gpx.fit_scipy(
                model=self.posterior,
                objective=negative_mll,
                train_data=D0
            )

            print('am: ', opt_posterior.prior.kernel.kernels[0].variance)
            print('ls: ', opt_posterior.prior.kernel.kernels[0].lengthscale)
            print('vr: ', opt_posterior.prior.kernel.kernels[1].variance)
            
            # Update the estimates of the latent function values
            # ! Always vstack mu and not the errors (issues otherwise)
            # Predictions
            latent_dist = opt_posterior(self.X, train_data=self.data)
            predictive_dist = opt_posterior.likelihood(latent_dist)
            # self.inducing_points = opt_posterior.inducing_inputs

            mu = np.vstack(predictive_dist.mean())
            errors = self.Y - mu
            
            # Compute log-likelihood(s):
            # Model convergence is controlled with the standard GP likelihood
            lnP[i+1] = -negative_mll(opt_posterior, D0)
            # lnP[i+1] = -self.elbo(opt_posterior, D0)
            print('Training...\n Iteration: ', i, ' tolerance: ', tolerance,
                  ' calculated(GP): ', abs(lnP[i+1] - lnP[i]), '\n')
            
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
        index, X0, Y0, resp0, pies, stds, K = self.DP(self.X, self.Y,
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