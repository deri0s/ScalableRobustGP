import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import mat73
import sys
sys.path.append('../..')   # noqa
from HeteroscedasticGP import *
from EMGP import RobustGaussianProcess as EM_GP

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
hgp = HeteroscedasticGP(X, Y, N, initial_hyperparameters=[1, 0.1],
                        hyperparameter_bounds=((3.0, None), (0.01, 1)),
                        gpz_hyperparameter_bounds=((3.0, None), (0.01, 1)),
                        normalise_y=True)
hgp.train()
muhgp, Y_star_std = hgp.predict(x, len(x))

# ----------------------------------------------------------------------------
# Create and train EM-GP
mixGP = EM_GP(X, Y, N, 3,
              initial_hyperparameters=[1, 0.5],
              hyperparameter_bounds=((1e-6, None), (1e-6, 2)),
              normalise_y=True)

# mixGP = DPGP(X, Y, N, 9,
#              initial_hyperparameters=[1, 2],
#              hyperparameter_bounds=((1e-6, None), (1e-6, 2)),
#              normalise_y=True)

mixGP.train()
mew, stdNew = mixGP.predict(x, len(x))

# Plot results
plt.figure()
plt.fill_between(x, mu + 3*mu_var, mu - 3*mu_var,
                  alpha=0.5,color='pink',label='Confidence \nBounds (VHGP)')
plt.plot(X, Y, 'o', color='black')
plt.plot(x, Y_star_mean, 'blue', label='GP')
plt.plot(x, mu, 'red', label='VHGP')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration')
plt.legend(loc=4, prop={"size":20})
plt.show()

# Plot results
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

#---------------------------------- EM-GP ------------------------------------

# EM-GP predictive mean
plt.figure()
plt.plot(X, Y, 'o', color='black')
plt.plot(x, Y_star_mean, linewidth = 2.5, color='blue', label='GP')
plt.plot(x, mu, linewidth = 4, color='red', label='VHGP')
plt.plot(x, mew, linewidth = 4, color='green', label='EM-GP')
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('Acceleration', fontsize=16)
plt.legend(loc=4, prop={"size":20})
plt.show()

# CONFIDENCE BOUNDS
plt.figure()
plt.fill_between(x, mew[:,0] + 3*stdNew[:,0],
                  mew[:,0] - 3*stdNew[:,0],
                  alpha=0.5,color='lightgreen',
                  label='Confidence \nBounds (EM-GP)')
plt.plot(X, Y, 'o', color='black')
plt.plot(x, mew, linewidth=2.5, color='green', label='EM-GP')
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('Acceleration', fontsize=16)
plt.legend(loc=4, prop={"size":20})


# Clustering performance
color_iter = ['black', 'red', 'lightgreen']
nl = ['Noise level 2', 'Noise level 1', 'Noise level 0']
enumerate_K = [i for i in range(mixGP.noise_sources)]

plt.figure()
if mixGP.noise_sources != 1:
    for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
        plt.plot(x[mixGP.indices[k]], Y[mixGP.indices[k]], 'o',
                  color=c, markersize = 8, label = nl[k])
plt.plot(x, mew, color="green", linewidth = 4, label="EM-GP")
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('Acceleration', fontsize=16)
plt.legend(loc=0, prop={"size":20})
plt.show()