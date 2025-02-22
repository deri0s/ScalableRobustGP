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
file = 'validation_data_main.xlsx'

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
test_perc = 0.12

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

# outputscale manual estimation
y_mean = np.mean(y_norm_np)
y_max = np.max(y_norm_np)
outputscale_manual = y_max - y_mean
print(f'\noutputscale standardise: {outputscale_manual} \ny-train max: {y_max}')
print('outputscale-nonstand: ',
      scaler.inverse_transform(outputscale_manual.reshape(1,-1))[0])

"""
READ SIMULATION OUTCOMES
"""
list_k  = []
list_os = []
list_ls = []
list_var= []
list_mse= []
list_mse_test = []

# RBF kernel
for i in range(2):
    with open(f'config_RBF_{int(i)}_step40.yaml', 'r') as f:
        config = yaml.safe_load(f)

    list_k.append('RBF')
    list_os.append(config['outputscale']['optimal'])
    list_ls.append(config['RBF']['lengthscale']['optimal'])
    list_var.append(config['WN']['var']['optimal'])
    list_mse.append(config['mse']['full'])
    list_mse_test.append(config['mse']['test'])

# RQ kernel
for i in range(3):
    with open(f'config_RQ_{int(i)}_step40.yaml', 'r') as f:
        config = yaml.safe_load(f)

    list_k.append('RQ')
    list_os.append(config['outputscale']['optimal'])
    list_ls.append(config['RQ']['lengthscale']['optimal'])
    list_var.append(config['WN']['var']['optimal'])
    list_mse.append(config['mse']['full'])
    list_mse_test.append(config['mse']['test'])
    
# RBF + RQ kernel
for i in range(3):
    with open(f'config_RBF_plus_RQ_{int(i)}_step40.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    list_k.append('RBF+RQ')
    list_os.append(config['outputscale']['optimal'])
    list_ls.append(config['RBF']['lengthscale']['optimal'])
    list_var.append(config['WN']['var']['optimal'])
    list_mse.append(config['mse']['full'])
    list_mse_test.append(config['mse']['test'])

d = {'kernel': list_k,
     'outputscale': list_os,
     'noise_var': list_var,
     'mse_full': list_mse,
     'mse_test': list_mse_test}

# add input lengthscales
print(list_ls[0][0])
print(np.shape(list_ls))
for col, name in enumerate(t_df.columns):
    temp_ls = []
    for row in range(np.shape(list_ls)[0]):
        print(list_ls[row][col])
        temp_ls.append(list_ls[row][col])
    d[name] = temp_ls

df_results = pd.DataFrame(d)
df_results.to_excel('hyper_results.xlsx')
print(df_results.head(9))