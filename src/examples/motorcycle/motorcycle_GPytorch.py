import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io as sio
import mat73
from sklearn.gaussian_process import GaussianProcessRegressor as GP
from models.dpgp import DirichletProcessGaussianProcess as DPGP
from models.dpsgp_gpytorch import DirichletProcessSparseGaussianProcess as DPSGP
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

plt.close('all')

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------

file_name = 'motorcycle.mat'
motorcycle_data = sio.loadmat(file_name)
X = motorcycle_data['X']
Y = motorcycle_data['y']
N = len(X)

#-----------------------------------------------------------------------------
# STANDARDISE DATA
#-----------------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler as minmax

train_scaler = minmax()
x_norm = train_scaler.fit_transform(X.reshape(-1,1))

outout_scaler = minmax()
y_norm = outout_scaler.fit_transform(Y.reshape(-1,1))

# Convert data to torch tensors
floating_point = torch.float32

X = torch.tensor(x_norm, dtype=floating_point)
Y = torch.tensor(y_norm, dtype=floating_point)

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
        rbf_kernel.lengthscale_constraint = gpytorch.constraints.Interval(0.07, 0.9)
        
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
# likelihood = GaussianLikelihood()
model = GPRegressionModel(X, Y, likelihood)

# Training
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = ExactMarginalLogLikelihood(likelihood, model)

for i in range(100):
    optimizer.zero_grad()
    output = model(X)
    loss = -mll(output, Y)
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(X))
    mu_norm = observed_pred.mean
    lower_norm, upper_norm = observed_pred.confidence_region()

# Unormalise predictions
mus = minmax.inverse_transform(mu_norm)