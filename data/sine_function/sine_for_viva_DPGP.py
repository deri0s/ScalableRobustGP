import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.insert(0, 'C:\Diego\PhD\Code\phdCode')
from HeteroscedasticGP import*
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
xNew = np.vstack(xNewdf['X_star'])

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
    

### Create and train standard Gaussian Processes (to compare to)
gp = GP(X, Y, N, initial_hyperparameters=[1, 0.5],
        hyperparameter_bounds=((1e-6,None),(1e-6,2)),
        normalise_y=True)
gp.train()
muGP, stdGP = gp.predict(xNew, 2*N)

# ----------------------------------------------------------------------------
# Robust GPs: EMGP and DPGP regression
# ----------------------------------------------------------------------------
# se = 1**2 * RBF(length_scale=1, length_scale_bounds=(0.9, 1.9))
# wn = WhiteKernel(noise_level=0.5**2, noise_level_bounds=(1e-6,0.7))

se = 1**2 * RBF(length_scale=1, length_scale_bounds=(1e-6,1e3))
wn = WhiteKernel(noise_level=0.5**2, noise_level_bounds=(1e-6,1e3))

kernel = se + wn
del se, wn

# DPGP
dpgp = DPGP(X, Y, init_K=7, kernel=kernel, normalise_y=True, plot_conv=True)
dpgp.train()
mu, std = dpgp.predict(xNew, report_normalised=True)


### CALCULATING THE OVERALL MSE
F = 150 * xNew * np.sin(xNew)
print("Mean Squared Error (GP)   : ", mean_squared_error(muGP, F))
print("Mean Squared Error (DPGP)  : ", mean_squared_error(mu, F))

### Print results for the EM-GP model
print('\n MODEL PARAMETERS DPGP (with normalisation): \n')
print(' Number of components identified, K = ', dpgp.K_opt)
print('Proportionalities: ', dpgp.pies)
print('Noise Stds: ', dpgp.stds)
print('Hyperparameters: ', dpgp.kernel_)


############################## PLOT THE RESULST ##############################
color_iter = ['lightgreen', 'orange', 'red']
enumerate_real = [i for i in range(3)]
enumerate_K = [i for i in range(dpgp.K_opt)]

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

############################ Clustering miss-classified #######################
if dpgp.K_opt == 3:
    missing = [np.setdiff1d(indices[0], dpgp.indices[0]),
               np.setdiff1d(indices[1], dpgp.indices[1]),
               np.setdiff1d(indices[2], dpgp.indices[2])]
    plt.figure()
    plt.title(" Miss-classified data ", fontsize=20)
    plt.plot(xNew, F, color="black", label="Sine function")
    plt.plot(X, Y, 'o', color="black", label="Observations")
    for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
        plt.plot(X[missing[k]], Y[missing[k]], 'kx',color=c,
                 markersize = 10, label='Noise Level '+str(k+1))
    plt.xlabel(' X ', fontsize=16)
    plt.ylabel(' f(X) ', fontsize=16)
    plt.legend(loc=3, prop={"size":20})
    plt.show()
    missed = max(len(missing[0]), len(missing[1]), len(missing[2]))
    print('\n The number of miss-clustered observations is: ', missed,
          '\n Correclty identified: ', N - missed,
          '\n with a clustering accuracy of: ', ((N - missed)*100)/150)
else:
    print("Estimated K different from the noise source")

# ----------------------------------------------------------------------------
# CLUSTERING
# ----------------------------------------------------------------------------
plt.figure()
plt.title(" Clustering performance ", fontsize=20)
color_iter = ['lightgreen', 'orange', 'red']
nl = ['Noise level 0', 'Noise level 1', 'Noise level 2']
plt.plot(xNew, mu, color="green", label="DPGP")
for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
    plt.plot(X[dpgp.indices[k]], Y[dpgp.indices[k]], 'o',color=c,
             markersize = 9, label=nl[k])
plt.xlabel(' X ', fontsize=16)
plt.ylabel(' f(X) ', fontsize=16)
plt.legend(loc=0, prop={"size":20})
plt.show()


# ----------------------------------------------------------------------------
# REGRESSION PLOTS
# ----------------------------------------------------------------------------
plt.figure()
plt.plot(xNew, F, color='red', linewidth = 4, label='Sine function')
plt.plot(xNew, muGP, color='blue', linestyle = '-.', linewidth = 4,
         label='GP')
plt.plot(xNew, mu, color='lightgreen', linestyle=':', linewidth = 4,
         label='DPGP')
plt.title('Regression Performance', fontsize=20)
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(prop={"size":20})

# ----------------------------------------------------------------------------
# CONFIDENCE BOUNDS
# ----------------------------------------------------------------------------

plt.figure()
plt.plot(xNew, F, color='red', linewidth = 4, label='Sine function')
plt.fill_between(xNew[:,0], mu[:,0] + 3*std[:,0],
                 mu[:,0] - 3*std[:,0],
                 alpha=0.5,color='lightgreen',
                 label='Confidence \nBounds (DPGP)')
plt.plot(X, Y, 'o', color='black')
plt.plot(xNew, mu, linewidth=4, color='green', label='DPGP')
plt.title('Regression Performance', fontsize=20)
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(prop={"size":18})

plt.show()