from models.DPSGP import DirichletProcessSparseGaussianProcess as DPSGP
import numpy as np
import pandas as pd
import paths
import matplotlib.pyplot as plt

file_name = paths.get_synthetic_path('Synthetic.xlsx')
df = pd.read_excel(file_name, sheet_name='Training')

# Read data
x_test_df = pd.read_excel(file_name, sheet_name='Testing')
labels_df = pd.read_excel(file_name, sheet_name='Real labels')

# Get training data
X = df['X'].values
Y = df['Y'].values
N = len(Y)

# Inducing points
inducing_points = X[::14]

# Covariance function
from gpytorch.models import ExactGP, ApproximateGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel as RBF

# Initialise sparse GP
prior_cov = ScaleKernel(RBF(lengthscale=0.9))

gp = DPSGP(X, Y, init_K=8,
           gp_model=ApproximateGP, kernel=prior_cov,
           n_inducing=14, normalise_y=True,
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