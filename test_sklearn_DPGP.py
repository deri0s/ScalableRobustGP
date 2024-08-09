import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler as ss
from sklearn.preprocessing import MinMaxScaler as minmax
import time
import paths

"""Best results
- Standardise inputs: MinMaxScaler [0,1]
- Standardise outputs: StandardScaler
"""

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
# train_scaler = ss()
train_scaler = minmax()
X_norm = train_scaler.fit_transform(X.reshape(-1,1))

# test_scaler = ss()
test_scaler = minmax()
X_test_norm = test_scaler.fit_transform(X_test.reshape(-1,1))

"""
GP Sklearn
"""
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from models.DPGP import DirichletProcessGaussianProcess as DPGP
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
muMix, stdMix = rgp.predict(X_test_norm)
comp_time = time.time() - start_time

print('\nkernel: \n', rgp.kernel_)
print('\nhyper: \n', rgp.hyperparameters)

F = 150 * X_test * np.sin(X_test)

print(f"\nComputational time: {comp_time:.2f} seconds")
print("\nMean Squared Error (DPGP)")
print(f"SKlearn: {mse(muMix, F):.2f}")

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------
plt.figure()
    
plt.plot(X_test, F, color='black', linewidth = 4, label='Sine function')
plt.plot(X_test, muMix, color='blue', linewidth = 4,
         label='DPGP-SKlearn')
plt.title('Regression Performance', fontsize=20)
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(prop={"size":20})

# ----------------------------------------------------------------------------
# CONFIDENCE BOUNDS
# ----------------------------------------------------------------------------

color_iter = ['green', 'orange', 'red']
enumerate_K = [i for i in range(rgp.K_opt)]

plt.figure()
plt.plot(X_test, F, color='black', linestyle='-', linewidth = 4,
         label='$f(x)$')

nl = ['Noise level 0', 'Noise level 1', 'Noise level 2']
for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
    plt.plot(X[rgp.indices[k]], y[rgp.indices[k]], 'o',color=c,
             markersize = 9, label=nl[k])
    
plt.plot(X_test, muMix, linewidth=4, color='green', label='DPSGP')
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(prop={"size":20})

plt.show()