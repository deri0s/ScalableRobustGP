from pathlib import Path
import yaml
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor as GPR

"""
The Distributed Gaussian Process model implemented with the rBCM approach
introduced in the paper "Distributed Gaussian Processes", M. Deisenroth and
J. 2015
Diego Echeverria
"""


class DistributedGP(GPR):

    def __init__(self, X, Y, N_GPs, kernel, plot_expert_pred=False):
        """
            Initialise objects, variables and parameters
        """
        
        # Initialise a GP class to access the kernel object when inheriting
        # from the DGP class
        super().__init__(kernel=kernel, alpha=0,
                         normalize_y = False, n_restarts_optimizer = 0)

        self.X = np.vstack(X)  # Inputs always vstacked
        self.Y = np.vstack(Y)  # Targets always vstacked
        self.N = len(Y)        # No. training points
        self.N_GPs = N_GPs     # No. GPs
        self.kernel = kernel
        self.ARD = isinstance(self.kernel, list)
        self.D = np.shape(self.X)[1]  # Dimension of inputs

        # Divide up data evenly between GPs
        self.X_split = np.array_split(self.X, N_GPs)
        self.Y_split = np.array_split(self.Y, N_GPs)

                # Array of colors to use in the plots (max 200 colors)
        FILE = Path(__file__).resolve()
        colors_path_name = FILE.parents[0] / 'colors.yml'

        with open(colors_path_name, 'r') as f:
            colors_dict = yaml.safe_load(f)
            self.c = colors_dict['color']
        
        # Plot option only available for N-GPs <= 200
        self.plot_expert_pred = plot_expert_pred
        if self.plot_expert_pred:
            if self.N_GPs > 200:
                assert False, 'Experts predictions can only be plotted for N_GP <= 200'

    def plot_expert(self, X_test, mu_all):
        # Plot the predictions of each expert
        plt.figure()
        plt.title('Expert predictions at each region')
        advance = 0
        step = int(len(X_test)/self.N_GPs)
        # draw a line dividing training and test data
        # plt.axvline(self.N + step, linestyle='--', linewidth=3, color='red',
        #             label='-> test data')
        for i in range(self.N_GPs):
            plt.plot(mu_all[:, i], color=self.c[i], label='DPGP('+str(i)+')')
            plt.axvline(int(advance), linestyle='--', linewidth=3,
                        color='black')
            advance += step
        plt.legend()


    def predict(self, X_star):
        """
        Description
        -----------
            Training and prediction tasks for the rBCM model
        Parameters
        ----------
            X_star : numpy array of new inputs
            N_star : no. new inputs
        Returns
        -------
            mu_star : vector of the rBCM predicitve mean values
            sigma_star : vector of the rBCM predicitve std values
            beta : [N_star x N_GPs] predictive power of each expert
        """
        
        X_star = np.vstack(X_star)

        # Train GP experts
        gps = []
        for m in range(self.N_GPs):
            # Check if the kernel uses ARD
            if self.ARD:
                gp = GPR(kernel=self.kernel[m], alpha=self.alpha,
                         normalize_y = False, n_restarts_optimizer = 0)
                gp.fit(self.X_split[m], self.Y_split[m])
                gps.append(gp)
            else:
                gp = GPR(kernel=self.kernel, alpha=self.alpha,
                         normalize_y = False, n_restarts_optimizer = 0)
                gp.fit(self.X_split[m], self.Y_split[m])
                gps.append(gp)
                
        self.gps = gps

        # Collect the experts predictive mean and standard deviation
        N_star = len(X_star)
        mu_all = np.zeros([N_star, self.N_GPs])
        sigma_all = np.zeros([N_star, self.N_GPs])
        
        # Compute local predictions if X_test is > 4000
        if len(X_star) > 4000:
            
            # Divide the test input space into 10 sub-spaces
            N_split = (lambda x : 10 if x < 10000 else 100)(len(X_star))
            X_star_split = np.array_split(X_star, N_split)
            N_local = len(X_star_split)
            
            # Outer loop: Move only on the GPs
            for i in range(self.N_GPs):
                full_mean_exp = []
                full_std_exp = []                
            
                # Inner loop: Move only on X_star_split
                for k in range(N_local):
                    mu, sigma = self.gps[i].predict(X_star_split[k],
                                                    return_std=True)
                    full_mean_exp.extend(mu[:, 0])
                    full_std_exp.extend(sigma)
                    
                mu_all[:, i] = np.asarray(full_mean_exp)
                sigma_all[:, i] = np.asarray(full_std_exp)
        else:
            for i in range(self.N_GPs):
                mu, sigma = self.gps[i].predict(X_star, return_std=True)
                mu_all[:, i] = mu
                sigma_all[:, i] = sigma

        # Calculate the normalised predictive power of the predictions made
        # by each GP. Note that, we are assuming that k(x_star, x_star)=1
        betas = np.zeros([N_star, self.N_GPs])
        # Add Jitter term to prevent numeric error
        prior_std = 1 + 1e-6
        # betas
        for i in range(self.N_GPs):
            betas[:, i] = 0.5*(np.log(prior_std) - np.log(sigma_all[:, i]**2))

        # Normalise betas
        scaler = MinMaxScaler(feature_range=(0,1))
        betas = scaler.fit_transform(betas)

        # Compute the gPoE precision
        prec_star = np.zeros(N_star)
        for i in range(self.N_GPs):
            prec_star += betas[:, i] * sigma_all[:, i]**-2

        # Compute the gPoE predictive variance and standard deviation
        var_star = prec_star**-1
        std_star = var_star**0.5

        # Compute the gPoE predictive mean
        mu_star = np.zeros(N_star)
        for i in range(self.N_GPs):
            mu_star += betas[:, i] * sigma_all[:, i]**-2 * mu_all[:, i]
        mu_star *= var_star

        # plot if specified
        if self.plot_expert_pred:
            self.plot_expert(X_star, mu_all)

        return mu_star, std_star, np.vstack(betas)