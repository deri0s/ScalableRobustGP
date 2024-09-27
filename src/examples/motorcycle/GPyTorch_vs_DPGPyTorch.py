import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io as sio
import mat73
from sklearn.metrics import mean_squared_error

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

"""
Best configuration:
    ?
    I am changing manually the train_scaler
"""

if preprocess == "ss":
    train_scaler = minmax()
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
        rbf_kernel.lengthscale = 0.38
        rbf_kernel.lengthscale_constraint = gpytorch.constraints.Interval(1e-5, 10)
        
        scale_kernel = ScaleKernel(rbf_kernel)
        scale_kernel.outputscale = 1.0
        scale_kernel.outputscale_constraint = gpytorch.constraints.Interval(0.9, 1.1)
        
        white_noise_kernel = LinearKernel()
        white_noise_kernel.variance = 0.05
        # white_noise_kernel.varaince_constraint = gpytorch.constraints.Interval(1e-6, 0.7)
        
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
rbf_kernel.lengthscale_constraint = gpytorch.constraints.Interval(1e-5, 10)

scale_kernel = ScaleKernel(rbf_kernel)
scale_kernel.outputscale = 1
scale_kernel.outputscale_constraint = gpytorch.constraints.Interval(0.9, 1.5)

covar_module = scale_kernel

dpgp = DPSGP(X, np.hstack(Y_org), init_K=7,
            gp_model='Standard',
            prior_mean=ConstantMean(), kernel=covar_module,
            noise_var = 0.05,
            floating_point=floating_point,
            normalise_y=True,
            print_conv=False, plot_conv=True, plot_sol=False)
dpgp.train()
mus, stds = dpgp.predict(X)

# Access the hyperparameters
output_scale = dpgp.gp.covar_module.outputscale.item()
length_scale = dpgp.gp.covar_module.base_kernel.lengthscale.item()
noise_var = dpgp.gp.likelihood.noise.item()

# Format the hyperparameters into an equation-like string
kernel_eq = f"{output_scale:.2f} * SE(ls={length_scale:.2f}) + {noise_var:.2f}^2"

print("\nEstimated hyperparameters:")
print(f"\nKernel Equation: {kernel_eq}\n")

print("Mean Squared Error")
print("GP-torch: ", mean_squared_error(mu_var, mu))
print("DPGP:     ", mean_squared_error(mu_var, mus))

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------
plt.figure()
plt.fill_between(x, mu_var + 3*std_var, mu_var - 3*std_var,
                 alpha=0.5,color='pink',label='3$\\sigma$ (VHGP)')
plt.fill_between(x, mu + 3*std, mu - 3*std,
                 alpha=0.4,color='lightblue',label='3$\\sigma$ (GP-torch)')
plt.fill_between(x, mus + 3*stds, mus - 3*stds,
                 alpha=0.4,color='limegreen',label='3$\\sigma$ (DPGP-torch)')
plt.plot(X_org, Y_org, 'o', color='black')
plt.plot(x, mu_var, 'red', linewidth=3, label='VHGP')
plt.plot(x, mu, 'blue', linewidth=3, label='GP-torch')
plt.plot(x, mus, 'green', linewidth=3, label='DPGP')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration')
plt.legend(loc=4, prop={"size":20})

# ----------------------------------------------------------------------------
# CLUSTERING
# ----------------------------------------------------------------------------

# color_iter = ['lightgreen', 'red', 'black']
# nl = ['Noise level 0', 'Noise level 1']
# enumerate_K = [i for i in range(dpgp.K_opt)]

# plt.figure()
# if dpgp.K_opt != 1:
#     for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
#         plt.plot(X_org[dpgp.indices[k]], Y_org[dpgp.indices[k]], 'o',
#                   color=c, markersize = 8, label = nl[k])
# plt.plot(x, mus, color="green", linewidth = 4, label="DPGP-torch")
# plt.xlabel('Time (s)', fontsize=16)
# plt.ylabel('Acceleration', fontsize=16)
# plt.legend(loc=0, prop={"size":20})

plt.show()