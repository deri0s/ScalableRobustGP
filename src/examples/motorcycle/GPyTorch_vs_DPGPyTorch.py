import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io as sio
import mat73

plt.close('all')

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------

file_name = 'motorcycle.mat'
motorcycle_data = sio.loadmat(file_name)
X_org = motorcycle_data['X']
Y_org = motorcycle_data['y']
N = len(X_org)

# Load MVHGP predictive mean and variance
vhgp_path= 'mvhgp.mat'
mvhgp = mat73.loadmat(vhgp_path)
std_var = mvhgp['fnm']
x = mvhgp['xm']
mu_var = mvhgp['ym']

#-----------------------------------------------------------------------------
# STANDARDISE OR NORMALISE DATA
#-----------------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler as minmax
from sklearn.preprocessing import StandardScaler as ss
preprocess = "ss"

if preprocess == "ss":
    train_scaler = ss()
    output_scaler = ss()
elif preprocess == "mm":
    train_scaler = minmax()
    output_scaler = minmax()
else:
    print('Not a valid prerpocesing method')

x_norm = train_scaler.fit_transform(X_org.reshape(-1,1))
y_norm = output_scaler.fit_transform(Y_org.reshape(-1,1))

# Convert data to torch tensors
floating_point = torch.float64

X = torch.tensor(x_norm, dtype=floating_point)
Y = torch.tensor(np.hstack(y_norm), dtype=floating_point)

import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel, LinearKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

"""
Standard GP (GPytorch)
"""

class GPRegressionModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        # Define the kernel
        rbf_kernel = RBFKernel()
        rbf_kernel.lengthscale = 0.9
        rbf_kernel.lengthscale_constraint = gpytorch.constraints.Interval(0.07, 2.9)
        
        scale_kernel = ScaleKernel(rbf_kernel)
        scale_kernel.outputscale = 1.0
        scale_kernel.outputscale_constraint = gpytorch.constraints.Interval(0.9, 1.1)
        
        white_noise_kernel = LinearKernel()
        white_noise_kernel.variance = 0.05
        white_noise_kernel.varaince_constraint = gpytorch.constraints.Interval(1e-6, 0.7)
        
        self.covar_module = scale_kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Model and likelihood
likelihood = GaussianLikelihood(noise_constraint=gpytorch.constraints.LessThan(0.7))
likelihood.noise = 0.005
gp = GPRegressionModel(X, Y, likelihood)

# Training
gp.train()
likelihood.train()
optimizer = torch.optim.Adam(gp.parameters(), lr=0.01)
mll = ExactMarginalLogLikelihood(likelihood, gp)

for i in range(100):
    optimizer.zero_grad()
    output = gp(X)
    loss = -mll(output, Y)
    loss.backward()
    optimizer.step()

# Evaluation
gp.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(gp(X))
    mu_norm = observed_pred.mean
    std_norm = observed_pred.stddev

# Unormalise predictions
mu = np.hstack(output_scaler.inverse_transform(np.vstack(mu_norm)))
std = np.hstack(output_scaler.inverse_transform(np.vstack(std_norm)))

"""
DPGP-torch
"""
from gpytorch.means import ConstantMean
from models.dpsgp_gpytorch import DirichletProcessSparseGaussianProcess as DPSGP

# Define the kernel
rbf_kernel = RBFKernel()
rbf_kernel.lengthscale = 0.9
rbf_kernel.lengthscale_constraint = gpytorch.constraints.Interval(0.07, 2.9)

scale_kernel = ScaleKernel(rbf_kernel)
scale_kernel.outputscale = 1.0
scale_kernel.outputscale_constraint = gpytorch.constraints.Interval(0.9, 1.1)

covar_module = scale_kernel

dpgp = DPSGP(X, np.hstack(Y_org), init_K=7,
            gp_model='Standard',
            prior_mean=ConstantMean(), kernel=covar_module,
            noise_var = 0.005,
            floating_point=floating_point,
            normalise_y=True,
            print_conv=False, plot_conv=True, plot_sol=False)
dpgp.train()
mus, stds = dpgp.predict(X)

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------
print("x: ", np.shape(mu), " std: ", np.shape(std), "x: ", np.shape(x))
plt.figure()
plt.fill_between(x, mu_var + 3*std_var, mu_var - 3*std_var,
                 alpha=0.5,color='pink',label='3$\\sigma$ (VHGP)')
plt.fill_between(x, mu + 3*std, mu - 3*std,
                 alpha=0.4,color='lightblue',label='3$\\sigma$ (GP-torch)')
plt.fill_between(x, mus + 3*stds, mus - 3*stds,
                 alpha=0.4,color='limegreen',label='3$\\sigma$ (DPGP-torch)')
plt.plot(X_org, Y_org, 'o', color='black')
plt.plot(x, mu_var, 'red', label='VHGP')
plt.plot(x, mu, 'blue', label='GP-torch')
plt.plot(x, mus, 'green', label='DPGP')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration')
plt.legend(loc=4, prop={"size":20})

plt.show()