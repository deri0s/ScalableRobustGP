import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import paths

# Read data
file_name = paths.get_synthetic_path('Synthetic.xlsx')
df = pd.read_excel(file_name, sheet_name='Training')

# Get training data
X = df['X'].values.reshape(-1, 1)
y = df['Y'].values
N = len(y)

# Get test data
df = pd.read_excel(file_name, sheet_name='Testing')
X_test = np.vstack(df['X_star'].values)

"""
GP Sklearn
"""
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from models.DPGP import DirichletProcessGaussianProcess as DPGP

# Covariance functions
se = 1**2 * RBF(length_scale=0.94**2, length_scale_bounds=(0.07, 0.9))
#                variance
wn = WhiteKernel(noise_level=0.0025, noise_level_bounds=(1e-6,0.7))

kernel = se + wn

# DPGP
rgp = DPGP(X, y, init_K=7, kernel=kernel, normalise_y=True, 
           plot_sol=False, plot_conv=True)
rgp.train(pseudo_sparse=False)
muMix, stdMix = rgp.predict(X_test)

print('\nkernel: \n', rgp.kernel_)
print('\nhyper: \n', rgp.hyperparameters)

"""
GPyTorch
"""
from models.DPSGP import DirichletProcessSparseGaussianProcess as DPSGP
# Covariance function
from gpytorch.models import ExactGP, ApproximateGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel as RBF

#                           lengthscale= l
prior_cov = ScaleKernel(RBF(lengthscale=0.9))

gp = DPSGP(X, y, init_K=8,
           gp_model=ApproximateGP, kernel=prior_cov,
           n_inducing=2, normalise_y=True,
           plot_conv=True, plot_sol=True)
gp.train()
mut, lower, upper = gp.predict(X_test)

# CALCULATING THE OVERALL MSE
from sklearn.metrics import mean_squared_error as mse

F = 150 * X_test * np.sin(X_test)
print("\nMean Squared Error (DPGP)")
print("SKlearn:", mse(muMix, F))
print("Torch:  ", mse(mut, F))

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------
plt.figure()
    
plt.plot(X_test, F, color='black', linewidth = 4, label='Sine function')
plt.plot(X_test, muMix, color='blue', linewidth = 4,
         label='DPGP-SKlearn')
plt.plot(X_test, mut, color='red', linewidth = 4,
         label='DPSGP-torch')
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
plt.plot(X_test, F, color='black', linestyle='-', linewidth = 4,
         label='$f(x)$')
plt.fill_between(X_test.squeeze(), lower, upper, color='lightcoral', alpha=0.5,
                 label='Confidence Interval')

nl = ['Noise level 0', 'Noise level 1', 'Noise level 2']
for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
    plt.plot(X[gp.indices[k]], y[gp.indices[k]], 'o',color=c,
             markersize = 9, label=nl[k])
    
plt.plot(X_test, mut, linewidth=4, color='green', label='DPSGP')
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(prop={"size":20})

plt.show()