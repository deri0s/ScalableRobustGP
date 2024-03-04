import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor as GP
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# Make relatively high amplitude 1D data
def f(x):
    return 10 * np.sin(x)

N = 100    # No. training data
X = np.vstack(np.linspace(0, 10, N))   # Training inputs
F = f(X)    # Truth training data
N_star = 200    # No. test data
X_star = np.vstack(np.linspace(0.1, 9.9, N_star))
F_star = f(X_star)   # Truth test data
sigma = 0.05   # Noise std

# Normalised truth data
F_star_norm = (F_star - np.mean(F_star)) / np.std(F_star)

# Generate noisy observations
np.random.seed(42)
noise = np.random.randn(100)

Y = F + np.vstack(sigma * noise)

# Standardise data
Ys = (Y - np.mean(Y))/np.std(Y)

# The chosen kernels
se = 1**2 * RBF(length_scale=1, length_scale_bounds=(1e-3, 1e3))
wn = WhiteKernel(noise_level=0.1**2, noise_level_bounds=(1e-6, 1e6))
kernel = se + wn

# Create and train standard GP
gp = GP(kernel, alpha=0, normalize_y=True)
gp.fit(X, Ys)
mu, std = gp.predict(X_star, return_std=True)

print('Estimated hyper:\n', gp.kernel_, '\nWith theta: ',
      np.exp(gp.kernel_.theta))