import paths
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor as GP
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from models.DPGP import DirichletProcessGaussianProcess as DPGP


plt.close('all')

# Read excel file
file_name = paths.get_synthetic_path('Synthetic.xlsx')
df = pd.read_excel(file_name, sheet_name='Training')
x_test_df = pd.read_excel(file_name, sheet_name='Testing')
labels_df = pd.read_excel(file_name, sheet_name='Real labels')

# Get training data
X = df['X'].values
Y = df['Y'].values
N = len(Y)
x_test = x_test_df['X_star'].values

# Get real labels
c0 = labels_df['Noise0'].values
c1 = labels_df['Noise1']
c2 = labels_df['Noise2']
not_nan = ~np.isnan(labels_df['Noise1'].values)
c1 = c1[not_nan]
c1 = [int(i) for i in c1]
not_nan = ~np.isnan(labels_df['Noise2'].values)
c2 = c2[not_nan]
c2 = [int(i) for i in c2]
indices = [c0, c1, c2]
    

# Covariance functions
se = 1**2 * RBF(length_scale=0.5, length_scale_bounds=(0.07, 0.9))
wn = WhiteKernel(noise_level=0.5**2, noise_level_bounds=(1e-6,0.7))

kernel = se + wn

# DPGP
rgp = DPGP(X, Y, init_K=7, kernel=kernel, normalise_y=True, plot_conv=True)
rgp.train(pseudo_sparse=True)
muMix, stdMix = rgp.predict(x_test)
print('DPGP init stds: ', rgp.init_pies)
print('DPGP init pies: ', rgp.init_sigmas)

# GP
gp = GP(kernel, alpha=0, normalize_y=True)
gp.fit(np.vstack(X), np.vstack(Y))
mu, std = gp.predict(np.vstack(x_test), return_std=True)

### CALCULATING THE OVERALL MSE
F = 150 * x_test * np.sin(x_test)
print("Mean Squared Error (GP)     : ", mean_squared_error(mu, F))
print("Mean Squared Error (DPGP)  : ", mean_squared_error(muMix, F))


## Print results for the DP-GP model
print('\n MODEL PARAMETERS EM-GP (with normalisation): \n')
print('Number of components identified, K = ', rgp.K_opt)
print('Proportionalities: ', rgp.pies)
print('Noise Stds: ', rgp.stds)
print('Hyperparameters: ', rgp.hyperparameters)


#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------
plt.figure()    
plt.plot(x_test, F, color='black', linewidth = 4, label='Sine function')
plt.plot(x_test, mu, color='blue', linewidth = 4, label='GP')
plt.plot(x_test, muMix, color='red', linestyle='-', linewidth = 4,
         label='DPGP')
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
plt.fill_between(x_test,
                 mu + 3*std, mu - 3*std,
                 alpha=0.5,color='green',
                 label='Confidence \nBounds (GP)')
plt.fill_between(x_test,
                 muMix + 3*stdMix, muMix - 3*stdMix,
                 alpha=0.5,color='lightgreen',
                 label='Confidence \nBounds (DPGP)')

nl = ['Noise level 0', 'Noise level 1', 'Noise level 2']
for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
    plt.plot(X[rgp.indices[k]], Y[rgp.indices[k]], 'o',color=c,
             markersize = 9, label=nl[k])
    
plt.plot(x_test, muMix, linewidth=2.5, color='green', label='DPGP')
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(prop={"size":20})

plt.show()