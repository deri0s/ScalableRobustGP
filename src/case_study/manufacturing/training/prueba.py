import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import yaml
from matplotlib import pyplot as plt
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import InducingPointKernel, ScaleKernel
from gpytorch.kernels import RBFKernel as RBF, RQKernel as RQ
from gpytorch.kernels import LinearKernel as Lin, PeriodicKernel as Per

# Load data
file = 'validation_data_main.xlsx'

# Training df
X_df = pd.read_excel(file, sheet_name='X_stand')
y_df = pd.read_excel(file, sheet_name='y_nonstand')
t_df = pd.read_excel(file, sheet_name='timelags')

# Drop Tweel position
X_df.drop(columns=['9282 Tweel Position'], inplace=True)
t_df.drop(columns=['9282 Tweel Position'], inplace=True)

# Pre-Process training data
N, D = np.shape(X_df.values)

# Create tag inputs
X = np.zeros([N, D])

# Create lagged features
def align_inputs(x_df, y_df, t_series):
    xdeep = x_df.copy()
    ydeep = y_df.copy()
    max_lag = int(max(t_series))

    for name, lag in t_series.items():
        xdeep[name] = xdeep[name].shift(int(lag))
    xdeep.dropna(inplace=True)

    ydeep = ydeep.iloc[max_lag:].reset_index(drop=True)

    return xdeep.reset_index(drop=True), ydeep

X_df, y_df = align_inputs(X_df, y_df, t_df.iloc[0,:])

# Standardize training & test data
X = X_df.values
y_nonstand, date_time = y_df.gp_pred.values, y_df.date_time.values

N, D = np.shape(X)
test_perc = 0.12
end_train = N - int(len(y_nonstand) * test_perc)

X_train_np = X[0:end_train]
date_train = date_time[0:end_train]
N_train = len(X_train_np)
y_train_nonstand = y_nonstand[0:end_train]

X_test = X[0:N]
date_time = date_time[0:N]

# Standardize outputs
y_train = y_train_nonstand.reshape(-1, 1)
scaler = StandardScaler()
scaler.fit(y_train)
y_norm_np = scaler.transform(y_train)

# Convert data to torch tensors
floating_point = torch.float64
X_train = torch.tensor(X_train_np, dtype=floating_point)
y_train = torch.tensor(y_norm_np, dtype=floating_point).squeeze()
X_test = torch.tensor(X_test, dtype=floating_point)

# Always clone
step = 60
inducing_points = X_train[::step, :].clone()

# Read best hyperparameters and initialization values from the yml file
with open('config_RBF_plus_RQ_0_step40.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Model
likelihood = GaussianLikelihood()

k = ScaleKernel(RBF(ard_num_dims=D) + RQ(ard_num_dims=D))
covar_module = InducingPointKernel(k,
                                   inducing_points=inducing_points,
                                   likelihood=likelihood)

# read initial hyperparameters
os0 = torch.tensor(config['outputscale']['optimal'], dtype=floating_point)
ls_se0 = torch.tensor(config['RBF']['lengthscale']['optimal'],
                      dtype=floating_point)
ls_rq0 = torch.tensor(config['RQ']['lengthscale']['optimal'],
                      dtype=floating_point)
alpha0 = torch.tensor(config['RQ']['alpha']['optimal'],
                      dtype=floating_point)
nv0 = torch.tensor(config['WN']['var']['optimal'], dtype=floating_point)

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

gp = SparseGP(X_train, y_train, likelihood, covar_module, nv0)

print(f'que? \n {gp.covar_module.base_kernel.base_kernel.kernels[0]}')

print(torch.rand(5) + 1, '\n')

import random

# init_ls_se = np.zeros(shape=D)

# for d in range(D):
#     init_ls_se[d] = random.gauss(ls0[d], sigma=1)
print(f'ls0:\n {ls_se0}')

var = 2.5
