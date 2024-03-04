import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import mat73
import sys
sys.path.insert(0, 'C:\Diego\PhD\Code\phdCode')
from HeteroscedasticGP import *
from EMGP import ExpectationMaximisationGaussianProcess as EMGP
from DPGP import DirichletProcessGaussianProcess as DPGP
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

plt.close('all')

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------

motorcycle_data = sio.loadmat('motorcycle.mat')
X = motorcycle_data['X']
Y = motorcycle_data['y']
N = len(X)

# Load MVHGP predictive mean and variance
mvhgp = mat73.loadmat('mvhgp.mat')
mu_var = mvhgp['fnm']
x = mvhgp['xm']
mu = mvhgp['ym']

# ----------------------------------------------------------------------------
# Fit stanard GP using the incomplete data
gp = GP(X, Y, N, initial_hyperparameters=[1, 0.1],
        hyperparameter_bounds=((1e-6, None), (1e-6, 2)),
        normalise_y=True)
gp.train()
Y_star_mean, gp_std = gp.predict(x, len(x))

# ----------------------------------------------------------------------------
# Train Heteroscedastic GP
# hgp = HeteroscedasticGP(X, Y, N, initial_hyperparameters=[1, 0.1],
#                         hyperparameter_bounds=((3.0, None), (0.01, 1)),
#                         gpz_hyperparameter_bounds=((3.0, None), (0.01, 1)),
#                         normalise_y=True)
# hgp.train()
# muhgp, Y_star_std = hgp.predict(x, len(x))

# ----------------------------------------------------------------------------
# DPGP regression
# ----------------------------------------------------------------------------
se = 1**2 * RBF(length_scale=1, length_scale_bounds=(1e-6,1e3))
wn = WhiteKernel(noise_level=2**2, noise_level_bounds=(1e-6,1e3))

kernel = se + wn
del se, wn

# ----------------------------------------------------------------------------
# EM-GP
# ----------------------------------------------------------------------------
mixGP = EMGP(X, Y, init_K=3, kernel=kernel, normalise_y=False,
             N_iter=10, plot_conv=True, plot_sol=True)
mixGP.train()
mew, stdNew = mixGP.predict(X, return_std=True)

print('\n MODEL PARAMETERS EMGP (with normalisation): \n')
print('Number of components identified, K = ', len(mixGP.indices))
print('Proportionalities: ', mixGP.pies)
print('Noise Stds: ', mixGP.stds)
print('Hyperparameters: ', mixGP.kernel_)

# ----------------------------------------------------------------------------
# DPGP
#-----------------------------------------------------------------------------
dpgp = DPGP(X, Y, init_K=6, kernel=kernel, normalise_y=True, plot_conv=False)
dpgp.train()
mu_dpgp, std_dpgp = dpgp.predict(x)

print('\n MODEL PARAMETERS DPGP (with normalisation): \n')
print('Number of components identified, K = ', dpgp.K_opt)
print('Proportionalities: ', dpgp.pies)
print('Noise Stds: ', dpgp.stds)
print('Hyperparameters: ', dpgp.kernel_, '\n')

# ----------------------------------------------------------------------------
# VHGP
# ----------------------------------------------------------------------------

# plt.figure()
# plt.fill_between(x, mu + 3*mu_var, mu - 3*mu_var,
#                   alpha=0.5,color='pink',label='Confidence \nBounds (VHGP)')
# plt.plot(X, Y, 'o', color='black')
# plt.plot(x, Y_star_mean, 'blue', label='GP')
# plt.plot(x, mu, 'red', label='VHGP')
# plt.xlabel('Time (s)')
# plt.ylabel('Acceleration')
# plt.legend(loc=4, prop={"size":20})
# plt.show()

# ----------------------------------------------------------------------------
# STANDARD GP
# ----------------------------------------------------------------------------

plt.figure()
plt.fill_between(x, Y_star_mean[:,0] + 3*gp_std[:,0],
                  Y_star_mean[:,0] - 3*gp_std[:,0],
                  alpha=0.5,color='lightblue',label='Confidence \nBounds (GP)')
plt.plot(X, Y, 'o', color='black')
plt.plot(x, Y_star_mean, 'blue', label='GP')
plt.plot(x, mu, 'red', label='VHGP')
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('Acceleration', fontsize=16)
plt.legend(loc=4, prop={"size":20})
plt.show()

# ----------------------------------------------------------------------------
#                                     DPGP
# ----------------------------------------------------------------------------

# Comparisons
plt.figure()
plt.plot(X, Y, 'o', color='black')
plt.plot(x, Y_star_mean, linewidth = 2.5, color='blue', label='GP')
plt.plot(x, mu, linewidth = 4, color='red', linestyle = '-.', label='VHGP')
plt.plot(x, mew, linewidth = 4, color='orange', label='EM-GP')
plt.plot(x, mu_dpgp, linewidth = 4, color='green', label='DPGP')
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('Acceleration', fontsize=16)
plt.legend(loc=4, prop={"size":20})
plt.show()

# ----------------------------------------------------------------------------
# CONFIDENCE BOUNDS
# ----------------------------------------------------------------------------

plt.figure()
plt.fill_between(x, mu_dpgp[:,0] + 3*std_dpgp[:,0],
                 mu_dpgp[:,0] - 3*std_dpgp[:,0],
                 alpha=0.5,color='lightgreen',
                 label='Confidence \nBounds (DPGP)')
plt.plot(X, Y, 'o', color='black')
plt.plot(x, mu_dpgp, linewidth=2.5, color='green', label='DPGP')
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('Acceleration', fontsize=16)
plt.legend(loc=4, prop={"size":20})

# ----------------------------------------------------------------------------
# CLUSTERING
# ----------------------------------------------------------------------------

color_iter = ['lightgreen', 'red', 'black']
nl = ['Noise level 0', 'Noise level 1']
enumerate_K = [i for i in range(dpgp.K_opt)]

plt.figure()
if dpgp.K_opt != 1:
    for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
        plt.plot(x[dpgp.indices[k]], Y[dpgp.indices[k]], 'o',
                  color=c, markersize = 8, label = nl[k])
plt.plot(x, mu_dpgp, color="green", linewidth = 4, label="DPGP")
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('Acceleration', fontsize=16)
plt.legend(loc=0, prop={"size":20})
plt.show()