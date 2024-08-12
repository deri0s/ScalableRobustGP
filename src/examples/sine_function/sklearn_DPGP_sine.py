import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler as minmax
import time

"""Best results
- Standardise inputs: MinMaxScaler [0,1]
- Standardise outputs: StandardScaler
"""

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
train_scaler = minmax()
X_norm = train_scaler.fit_transform(X.reshape(-1,1))

test_scaler = minmax()
X_test_norm = test_scaler.fit_transform(X_test.reshape(-1,1))

"""
GP Sklearn
"""
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from models.dpgp import DirichletProcessGaussianProcess as DPGP
from sklearn.metrics import mean_squared_error as mse

# Covariance functions
se = 1**2 * RBF(length_scale=0.94**2, length_scale_bounds=(0.07, 0.9))
#                variance
wn = WhiteKernel(noise_level=0.0025, noise_level_bounds=(1e-6,0.7))

kernel = se + wn

# DPGP
start_time = time.time()
rgp = DPGP(X_norm, y, init_K=7, kernel=kernel, normalise_y=True, 
           plot_sol=False, plot_conv=True)
rgp.train(pseudo_sparse=False)
mu, std = rgp.predict(X_test_norm)
comp_time = time.time() - start_time

print('\nkernel: \n', rgp.kernel_)
print('\nhyper: \n', rgp.hyperparameters)

"""
Standard GP
"""
from sklearn.gaussian_process import GaussianProcessRegressor as GP
from sklearn.preprocessing import MinMaxScaler as minmax
from sklearn.preprocessing import StandardScaler as ss

# standardise outputs
y_scaler = ss()
y_norm = y_scaler.fit_transform(y.reshape(-1,1))

gp = GP(kernel, alpha=0, normalize_y=True)
gp.fit(np.vstack(X_norm), np.vstack(y))
mu_gp, std_gp = gp.predict(np.vstack(X_test_norm), return_std=True)

"""
Performance
"""
F = 150 * X_test * np.sin(X_test)

print(f"\nComputational time: {comp_time:.2f} seconds")
print("\nMean Squared Error (DPGP)")
print(f"DPGP: {mse(mu, F):.2f}")
print(f'GP:   {mse(mu_gp, F):.2f}')

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------
plt.figure()

plt.plot(X_test, F, color='black', linewidth = 4, label='Sine function')
plt.plot(X_test, mu_gp, color='blue', linewidth = 4, label='GP')
plt.plot(X_test, mu, color='red', linewidth = 4,
         label='DPGP')
plt.title('Regression Performance - SkLearn models', fontsize=20)
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(prop={"size":20})

# ----------------------------------------------------------------------------
# CONFIDENCE BOUNDS
# ----------------------------------------------------------------------------

color_iter = ['green', 'orange', 'red']
enumerate_K = [i for i in range(rgp.K_opt)]

plt.figure()
plt.fill_between(X_test,
                 mu_gp + 3*std_gp, mu_gp - 3*std_gp,
                 alpha=0.5,color='lightblue',
                 label='3$\sigma$ (GP)')

plt.fill_between(X_test,
                 mu + 3*std, mu - 3*std,
                 alpha=0.5,color='lime',
                 label='3$\sigma$ (DPGP)')
plt.plot(X_test, F, color='black', linestyle='-', linewidth = 4,
         label='$f(x)$')

nl = ['Noise level 0', 'Noise level 1', 'Noise level 2']
for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
    plt.plot(X[rgp.indices[k]], y[rgp.indices[k]], 'o',color=c,
             markersize = 9, label=nl[k])
    
plt.plot(X_test, mu, linewidth=4, color='green', label='DPSGP')
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(prop={"size":20})

plt.show()