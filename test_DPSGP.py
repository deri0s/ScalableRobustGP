from models.Diego import DirichletProcessSparseGaussianProcess as Diego
import numpy as np
import pandas as pd
import torch
import paths
import matplotlib.pyplot as plt

file_name = paths.get_synthetic_path('Synthetic.xlsx')
df = pd.read_excel(file_name, sheet_name='Training')

# Read data
x_test_df = pd.read_excel(file_name, sheet_name='Testing')
labels_df = pd.read_excel(file_name, sheet_name='Real labels')

# Get training data
X_org = df['X'].values
X = df['X'].values
y = df['Y'].values
N = len(y)

# Normalize features
x_mean = np.mean(X)
x_std = np.std(X)
X_norm = (X - x_mean) / x_std

# Convert data to torch tensors
X_temp = torch.tensor(X_norm, dtype=torch.float32)

# Inducing points
inducing_points = X_temp[::10].clone()
# print('length z0: ', np.shape(inducing_points))

from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import InducingPointKernel, ScaleKernel, RBFKernel as RBF


"""
Sparse GP with InducingPointKernel
"""
base_covar_module = ScaleKernel(RBF(lengthscale=0.9))
likelihood = GaussianLikelihood()
covar_module = InducingPointKernel(base_covar_module,
                                   inducing_points=inducing_points,
                                   likelihood=likelihood)

gp = Diego(X_norm, y, init_K=8,
           gp_model='Sparse',
           prior_mean=ConstantMean(), kernel=covar_module,
           noise_var = 0.05,
           normalise_y=True,
           plot_conv=True, plot_sol=True)
plt.show()
# gp.train()
# mut, lower, upper = gp.predict(X)

# # Real function
# F = 150 * X * np.sin(X)

# from sklearn.metrics import mean_squared_error

# print("\nMean Squared Error (DPSGP)   : ", mean_squared_error(mut, F))
# #-----------------------------------------------------------------------------
# # REGRESSION PLOT
# #-----------------------------------------------------------------------------
# plt.figure()
    
# plt.plot(X, F, color='black', linewidth = 4, label='Sine function')
# plt.plot(X, mut, color='red', linewidth = 4,
#          label='DPSGP-torch')
# plt.title('Regression Performance', fontsize=20)
# plt.xlabel('x', fontsize=16)
# plt.ylabel('f(x)', fontsize=16)
# plt.legend(prop={"size":20})

# # ----------------------------------------------------------------------------
# # CONFIDENCE BOUNDS
# # ----------------------------------------------------------------------------

# color_iter = ['green', 'orange', 'red']
# enumerate_K = [i for i in range(gp.K_opt)]

# plt.figure()
# plt.plot(X, F, color='black', linestyle='-', linewidth = 4,
#          label='$f(x)$')
# plt.fill_between(X, lower, upper, color='lightcoral', alpha=0.5,
#                  label='Confidence Interval')

# nl = ['Noise level 0', 'Noise level 1', 'Noise level 2']
# for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
#     plt.plot(X[gp.indices[k]], Y[gp.indices[k]], 'o',color=c,
#              markersize = 9, label=nl[k])
    
# plt.plot(X, mut, linewidth=4, color='green', label='DDPSGP')
# plt.xlabel('x', fontsize=16)
# plt.ylabel('f(x)', fontsize=16)
# plt.legend(prop={"size":20})

# plt.show()