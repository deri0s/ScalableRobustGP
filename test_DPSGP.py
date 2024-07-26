from models.Diego import DirichletProcessSparseGaussianProcess as Diego
from sklearn.preprocessing import StandardScaler as ss
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

# get test data
X_test = x_test_df['X_star'].values

# Normalize features
train_scaler = ss()
X_norm = train_scaler.fit_transform(X.reshape(-1,1))

test_scaler = ss()
X_test_norm = test_scaler.fit_transform(X_test.reshape(-1,1))

# Convert data to torch tensors
X_temp = torch.tensor(X_norm, dtype=torch.float32)

# Inducing points
inducing_points = X_temp[::10].clone()

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
gp.train()
mu, lower, upper = gp.predict(X_test_norm)

# get the indices of the estimated inducing inputs/points
z_indices = gp._z_indices

# Real function
F = 150 * X_test * np.sin(X_test)

from sklearn.metrics import mean_squared_error

print("\nMean Squared Error (DPSGP)   : ", mean_squared_error(mu, F))
#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------
X0, y0 = X[gp.indices[0]], y[gp.indices[0]]

fig, ax = plt.subplots()
# inducing inputs
plt.plot(X0[z_indices], y0[z_indices], '*', color='lightgreen', markersize=8)
plt.plot(X_test, F, color='black', linewidth = 4, label='Sine function')
plt.plot(X_test, mu, color='red', linewidth = 4,
         label='DPSGP-torch')
plt.title('Regression Performance', fontsize=20)
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
ax.vlines(
    x=X0[z_indices],
    ymin=y.min().item(),
    ymax=y.max().item(),
    alpha=0.3,
    linewidth=1.5,
    label="z*",
    color='green'
)
plt.legend(prop={"size":20})

# ----------------------------------------------------------------------------
# CONFIDENCE BOUNDS
# ----------------------------------------------------------------------------

color_iter = ['green', 'orange', 'red']
enumerate_K = [i for i in range(gp.K_opt)]

plt.figure()
plt.plot(X_test, F, color='black', linestyle='-', linewidth = 4,
         label='$f(x)$')
plt.fill_between(X_test, lower, upper, color='lightcoral', alpha=0.5,
                 label='Confidence Interval')

nl = ['Noise level 0', 'Noise level 1', 'Noise level 2']
for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
    plt.plot(X[gp.indices[k]], y[gp.indices[k]], 'o',color=c,
             markersize = 10, label=nl[k])

plt.plot(X_test, mu, linewidth=4, color='green', label='DDPSGP')
plt.plot(X0[z_indices], y0[z_indices], '*', color='lightgreen', markersize=8, label='z*')
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(prop={"size":20})

plt.show()