import torch
import time
import gpytorch
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler as ss
from sklearn.metrics import mean_squared_error as mse
from matplotlib import pyplot as plt
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import InducingPointKernel, ScaleKernel
from gpytorch.kernels import RBFKernel as RBF, RQKernel as RQ

"""
NSG data
"""
# NSG post processes data location
file = '../validation_data_main.xlsx'

# Training df
X_df = pd.read_excel(file, sheet_name='X_stand')
y_df = pd.read_excel(file, sheet_name='y_nonstand')
t_df = pd.read_excel(file, sheet_name='timelags')

# drop Tewwl position ! only add this for the my-intuition.yaml
X_df.drop(columns=['9282 Tweel Position'], inplace=True)
t_df.drop(columns=['9282 Tweel Position'], inplace=True)

# Pre-Process training data
N, D = np.shape(X_df.values)

# Create tag inputs
X = np.zeros([N, D])

"""---------------------------------------------------------------------------
    CREATE LAGGED FEATURES
"""

def align_inputs(x_df, y_df, t_series):
    # ! Always close/Deep copy
    xdeep = x_df.copy()
    ydeep = y_df.copy()
    max_lag = int(max(t_series))

    # X
    for name, lag in t_series.items():
        xdeep[name] = xdeep[name].shift(int(lag))
    xdeep.dropna(inplace=True)

    # y and date-time
    ydeep = ydeep.iloc[max_lag:].reset_index(drop=True)

    return xdeep.reset_index(drop=True), ydeep
 
X_df, y_df = align_inputs(X_df, y_df, t_df.iloc[0,:])

"""---------------------------------------------------------------------------
    STANDARDISE TRAINING & TEST DATA
"""
# Read best hyperparameters and initialisation values from the yml file
with open('config_RBF_step40.yaml', 'r') as f:
    config = yaml.safe_load(f)
    
test_perc = config['test_percentage']

X = X_df.values
y_nonstand, date_time = y_df.gp_pred.values, y_df.date_time.values

N, D = np.shape(X)
end_train = N - int(len(y_nonstand)*test_perc)

# make sure X and y are the same size
assert N - int(len(X)*test_perc) == N - int(len(y_nonstand)*test_perc), 'Size of X and y are not the same'

X_train_np = X[0:end_train]
date_train = date_time[0:end_train]
N_train = len(X_train_np)
y_train_nonstand = y_nonstand[0:end_train]

X_test = X[0:N]
date_time = date_time[0:N]

# Standardise outputs
y_train = y_train_nonstand.reshape(-1,1)
scaler = ss()
scaler.fit(y_train)
y_norm_np = scaler.transform(y_train)

# Convert data to torch tensors
floating_point = torch.float64
X_train = torch.tensor(X_train_np, dtype=floating_point)
y_train = torch.tensor(y_norm_np, dtype=floating_point).squeeze()
X_test = torch.tensor(X_test, dtype=floating_point)

"""----------------------------------------------------------------------------
Sparse GP
"""
step = config['step']
inducing_points = X_train[::step, :].clone()

# ! Ensure data is of shape [N, D]
print(X_train.shape)            # Should be [N_train, D]
print(inducing_points.shape)    # Should be [N_train/step, D]
print(X_test.shape)             # Should be [N_test, D]
print(y_train.shape)            # Should be [N_train]

assert inducing_points[2, 0] == X_train[step+step, 0], 'Init induced not the same as in X_train'

# Model
likelihood = GaussianLikelihood()

k = ScaleKernel(RBF(ard_num_dims=D))
covar_module = InducingPointKernel(k,
                                   inducing_points=inducing_points,
                                   likelihood=likelihood)

# read initial hyperparameters
init_os = config['outputscale']['initial']
init_ls = config['RBF']['lengthscale']['initial']
init_noise_var = config['WN']['var']['initial']

class SparseGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, noise_var):
        super(SparseGP, self).__init__(train_x, train_y, likelihood)
        likelihood.noise = noise_var
        self.mean_module = ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

gp = SparseGP(X_train, y_train, likelihood, covar_module, init_noise_var)
# initialise kernel parameters
gp.covar_module.base_kernel.outputscale = init_os
gp.covar_module.base_kernel.base_kernel.lengthscale = init_ls

# Print initial kernel parameters
print("\nInitial kernel parameters:")
print("Outputscale:", gp.covar_module.base_kernel.outputscale.item())

# Train model
start_time = time.time()
gp.train()
gp.likelihood.train()

optimizer = torch.optim.Adam(gp.parameters(), lr=0.05)
mll = ExactMarginalLogLikelihood(likelihood, gp)

training_iterations = 100
for count in range(training_iterations):
    optimizer.zero_grad()
    output = gp(X_train)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()
end_time = time.time() - start_time

print(f'\nTraining time: {end_time} ms')

print("\nEstimated kernel parameters")
print("Outputscale:", gp.covar_module.base_kernel.outputscale.item())

# *Induced points
init_z_indices = np.arange(0, len(X_train.numpy()), step)

# Make sure the _z (induced inputs) are a subset of the X_train dataset
_z = gp.covar_module.inducing_points.detach()

_z_indices = []
for z in _z:
    distances = torch.norm(X_train - z, dim=1)
    closest_index = torch.argmin(distances).item()
    _z_indices.append(closest_index)

# check the z0 and z* are not the same
print('\nInputs induced? ',
      ~np.all(list(init_z_indices == _z_indices)))

# Predictions
gp.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(gp(X_test))

    # Unormalise predictions
    pred_mean = observed_pred.mean
    mu = scaler.inverse_transform(pred_mean.unsqueeze(1))[:,0]
    stds = scaler.inverse_transform(observed_pred.stddev.unsqueeze(1))[:,0]
    lower_stand, upper_stand = observed_pred.confidence_region()
    lower = scaler.inverse_transform(lower_stand.unsqueeze(1))[:,0]
    upper = scaler.inverse_transform(upper_stand.unsqueeze(1))[:,0]

print('MSE (train - test): ', mse(mu, y_nonstand))
print('MSE (test):         ', mse(mu[end_train:-1], y_nonstand[end_train:-1]))

"""--------------------------------------------------------------------------
PLOT
"""
#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------
fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
fig.autofmt_xdate()

plt.fill_between(date_time, lower, upper,
                 alpha=0.5, color='lightcoral',
                 label='2$\\sigma$')
ax.plot(date_time, y_nonstand, '*', color='green', label='Val')
ax.plot(date_time, mu, color='red', label='GP')
plt.axvline(date_time[end_train-1], linestyle='--', linewidth=3,
            color='black')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

ax.vlines(
    x=date_time[::step],
    ymin=-2*stds.min(),
    ymax=y_train.max().item(),
    alpha=0.3,
    linewidth=1.5,
    ls='--',
    label="z0",
    color='grey'
)

# Induced points
ax.vlines(
    x=date_time[_z_indices],
    ymin=-2*stds.min(),
    ymax=y_train.max().item(),
    alpha=0.4,
    linewidth=1.5,
    label="z*",
    color='orange'
)
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)
plt.show()