import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.insert(0, 'C:\Diego\PhD\Code\phdCode')
from GP import *
from EMGP import ExpectationMaximisationGaussianProcess as EMGP
from DPGP import DirichletProcessGaussianProcess as DPGP
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error

plt.close('all')

# Read excel file
file_name = 'Synthetic.xlsx'
df = pd.read_excel(file_name, sheet_name='Training')
xNewdf = pd.read_excel(file_name, sheet_name='Testing')
labels_df = pd.read_excel(file_name, sheet_name='Real labels')

# Get training data
X = df['X'].values
Y = df['Y'].values
N = len(Y)
xNew = xNewdf['X_star']

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

# ----------------------------------------------------------------------------
# GENERAL KERNEL AND INITIAL GP HYPERPARAMETERS
# ----------------------------------------------------------------------------
se = 1**2 * RBF(length_scale=1, length_scale_bounds=(1e-6,1e3))
wn = WhiteKernel(noise_level=0.5**2, noise_level_bounds=(1e-6,1e3))

kernel = se + wn
del se, wn

# ----------------------------------------------------------------------------
# GP
# ----------------------------------------------------------------------------
gp = GP(X, Y, N, initial_hyperparameters=[1, 0.5],
        hyperparameter_bounds=((1e-6,None),(1e-6,2)), normalise_y=True)
gp.train()
mu, std = gp.predict(xNew, 2*N)

# ----------------------------------------------------------------------------
# ROBUST GP (EMGP)
# ----------------------------------------------------------------------------
mixGP = EMGP(X, Y, init_K=3, kernel=kernel, normalise_y=True,
             N_iter=10, plot_conv=True)
mixGP.train()
mumix, std = mixGP.predict(np.vstack(xNew), return_std=True)

# Unormalised mean and std
muMix = mumix*mixGP.Y_std + mixGP.Y_mu

# ----------------------------------------------------------------------------
# DPGP
# ----------------------------------------------------------------------------
dpgp = DPGP(X, Y, init_K=7, kernel=kernel, normalise_y=True, plot_conv=True)
dpgp.train()
muo, stdo = dpgp.predict(xNew, 2*N)
print('DPGP init stds: ', dpgp.init_pies)
print('DPGP init pies: ', dpgp.init_sigmas)


### CALCULATING THE OVERALL MSE
F = 150 * xNew * np.sin(xNew)
print("Mean Squared Error (GP)  : ", mean_squared_error(mu, F))
print("Mean Squared Error (RGP)  : ", mean_squared_error(muMix, F))
print("Mean Squared Error (DPGP): ", mean_squared_error(muo, F))


### Print results for the EM-GP model
print('\n MODEL PARAMETERS RGP (Stegle):')
print('Proportionalities: ', mixGP.pies)
print('Noise Stds: ', mixGP.stds)
print('Hyperparameters: ', mixGP.hyperparameters)

### Print results for the DPGP model
print('\n MODEL PARAMETERS DPGP (with normalisation):')
print('Proportionalities: ', dpgp.pies)
print('Noise Stds: ', dpgp.stds)
print('Hyperparameters: ', dpgp.hyperparameters)


############################## PLOT THE RESULST ##############################
color_iter = ['lightgreen', 'orange', 'red']
enumerate_real = [i for i in range(3)]
enumerate_K = [i for i in range(mixGP.init_K)]

############################ REAL NOISE LABELS ###############################

plt.figure()
plt.title(" Data corrupted with non-Gaussian noise ", fontsize=20)
plt.plot(xNew, F, color="black", label="Sine function")
for i, (k, c) in enumerate(zip(enumerate_real, color_iter)):
    plt.plot(X[indices[k]], Y[indices[k]], 'o',color=c, markersize = 9,
             label='Gaussian noise '+str(k+1))
plt.xlabel(' X ', fontsize=20)
plt.ylabel(' f(X) ', fontsize=20)
plt.legend(loc=3, prop={"size":25})

# ----------------------------------------------------------------------------
# REGRESSION
# ----------------------------------------------------------------------------
plt.figure()
plt.plot(xNew, F, color='black', linewidth = 5, label='Sine function')
plt.plot(xNew, mu, color='orange', linestyle = '-.', linewidth = 5,
         label='Standard GP')
plt.plot(xNew, muMix, color='blue', linestyle = '-.', linewidth = 5,
         label='RGP(Stegle)')
plt.plot(xNew, muo, color='red', linestyle=':', linewidth = 5,
         label='DPGP')
plt.title('Regression Performance', fontsize=20)
plt.xlabel('x', fontsize=20)
plt.ylabel('f(x)', fontsize=20)
plt.legend(prop={"size":20})

# ----------------------------------------------------------------------------
# CLUSTERING
# ----------------------------------------------------------------------------
plt.figure()
plt.title(" Clustering Performance ", fontsize=20)
color_iter = ['lightgreen', 'orange', 'red']
nl = ['Noise level 0', 'Noise level 1', 'Noise level 2']
plt.plot(xNew, mu, color="green", label="DPGP")
for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
    plt.plot(X[dpgp.indices[k]], Y[dpgp.indices[k]], 'o',color=c,
             markersize = 9, label=nl[k])
plt.xlabel(' x ', fontsize=20)
plt.ylabel(' f(x) ', fontsize=20)
plt.legend(loc=0, prop={"size":20})

plt.show()