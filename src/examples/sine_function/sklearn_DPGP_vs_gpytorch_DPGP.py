from models.dpsgp_gpytorch import DirichletProcessSparseGaussianProcess as DPSGP_gpytorch
from sklearn.preprocessing import MinMaxScaler as minmax
import numpy as np
import pandas as pd
import torch
import time
import matplotlib.pyplot as plt

# file_name = paths.get_synthetic_path('Synthetic.xlsx')
file_name = 'Synthetic.xlsx'
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
# train_scaler = ss()
train_scaler = minmax()
X_norm = train_scaler.fit_transform(X.reshape(-1,1))

# test_scaler = ss()
test_scaler = minmax()
X_test_norm = test_scaler.fit_transform(X_test.reshape(-1,1))

# Convert data to torch tensors
floating_point = torch.float64
X_temp = torch.tensor(X_norm, dtype=floating_point)

# Inducing points
inducing_points = X_temp[::10].clone()

from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel as RBF


"""
GPyTorch

Best results
- Standarise inputs: MinMaxScaler [0,1]
- Standardise outputs: StandardScaler
- torch.float32 or torch.float64 (not big difference)
        ^                ^
-     faster            slower   (around 1 sec difference) 
"""

covar_module = ScaleKernel(RBF(lengthscale=0.9))
likelihood = GaussianLikelihood()

start_time = time.time()
gp = DPSGP_gpytorch(X, y, init_K=8,
                    gp_model='Standard',
                    prior_mean=ConstantMean(), kernel=covar_module,
                    noise_var = 0.05,
                    floating_point=floating_point,
                    normalise_y=True,
                    print_conv=False, plot_conv=True, plot_sol=False)
gp.train()
mu, std = gp.predict(X_test)
comp_time = time.time() - start_time

print("\nEstimated hyperparameters")
print("Outputscale:", gp.gp.covar_module.outputscale.item())
print("Lengthscale:", gp.gp.covar_module.base_kernel.lengthscale.item())
print("Noise:", gp.gp.likelihood.noise.item())

"""
SKLearn DPGP
"""
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from models.dpgp import DirichletProcessGaussianProcess as DPGP
from sklearn.metrics import mean_squared_error as mse

# Covariance functions
se = 1**2 * RBF(length_scale=0.94**2, length_scale_bounds=(0.07, 0.9))
#                variance
wn = WhiteKernel(noise_level=0.0025, noise_level_bounds=(1e-6,0.7))

kernel = se + wn

start_time = time.time()
rgp = DPGP(X_norm, y, init_K=7, kernel=kernel, normalise_y=True, 
           plot_sol=False, plot_conv=True)
rgp.train(pseudo_sparse=False)
mus, stds = rgp.predict(X_test_norm)
comp_time = time.time() - start_time

print('\nkernel: \n', rgp.kernel_)
print('\nhyper: \n', rgp.hyperparameters)

# Real function
F = 150 * X_test * np.sin(X_test)

print(f"\nComputational time: {comp_time:.2f} seconds")
print("\nMean Squared Error (DPGP)")
print(f"DPGP-SKLearn:  {mse(mus, F):.2f}")
print(f'DPGP-GPyTorch: {mse(mu, F):.2f}')

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------
X0, y0 = X[gp.indices[0]], y[gp.indices[0]]

fig, ax = plt.subplots()
# inducing inputs
plt.plot(X_test, F, color='black', linewidth = 4, label='Sine function')
plt.plot(X_test, mu, color='red', linewidth = 4, label='DPSGP-torch')
plt.plot(X_test, mus, color='blue', linewidth = 4, label='DPSGP-sklearn')
plt.title('Regression Performance', fontsize=20)
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(prop={"size":20})

# ----------------------------------------------------------------------------
# CONFIDENCE BOUNDS
# ----------------------------------------------------------------------------

color_iter = ['green', 'orange', 'red']
enumerate_K = [i for i in range(gp.K_opt)]

plt.figure()
plt.fill_between(X_test,
                 mu + 3*std, mu - 3*std,
                 alpha=0.5,color='lightcoral',
                 label='3$\\sigma$ (DPGP-torch)')

plt.fill_between(X_test,
                 mus + 3*std, mu - 3*std,
                 alpha=0.5,color='lightblue',
                 label='3$\\sigma$ (DPGP-sklearn)')

plt.plot(X_test, F, color='black', linestyle='-', linewidth = 4,
         label='$f(x)$')

nl = ['Noise level 0', 'Noise level 1', 'Noise level 2']
for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
    plt.plot(X[gp.indices[k]], y[gp.indices[k]], 'o',color=c,
             markersize = 10, label=nl[k])

plt.plot(X_test, mu, linewidth=4, color='red', label='DPGP-torch')
plt.plot(X_test, mu, linewidth=4, color='blue', label='DPGP-sklearn')
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(prop={"size":20})

plt.show()