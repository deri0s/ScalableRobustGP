import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, 'C:\Diego\PhD\Code\phdCode')
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from DDPGP import DistributedDPGP as DDPGP

"""
Test DDPGP with synthetic data
"""

# Define the domain of the function and the number of observations.
N = 500
X = np.linspace(0, 10, N)

# Corrupt observations with noise generated from a mixture of 2 Gaussians
F = np.sin(X)
Y = np.zeros(N)
for i in range(N):
    u = np.random.rand()
    if u < 0.95:
        Y[i] = F[i] + 0.05*np.random.randn()
    else:
        Y[i] = F[i] + 90*np.random.randn()

# Contruct array for predictions
N_new = 2*N
xNew = np.vstack(np.linspace(X[0], X[N-1], N_new))

# The chosen kernels
se = 1**2 * RBF(length_scale=5, length_scale_bounds=(1, 2))
wn = WhiteKernel(noise_level=0.5**2, noise_level_bounds=(0.05,100))
kernel = se + wn

# The DDPGP model
N_GPs = 3
rsgp = DDPGP(X, Y, N_GPs, 7, kernel, normalise_y=True)
rsgp.train()
mu, std, betas = rsgp.predict(xNew)

plt.figure()
plt.plot(X, F, color="black", label="Sine function")
plt.plot(xNew, mu, 'red', label='DDPGP')
plt.plot(X, Y, '.', label='Training data')
plt.legend()
plt.title('Robust regression performance')

##############################################################################
# Plot beta
##############################################################################
c = ['red', 'orange', 'blue', 'black', 'green', 'cyan', 'darkred', 'pink',
     'gray', 'magenta','lightgreen', 'darkblue', 'yellow']

fig, ax = plt.subplots()
fig.autofmt_xdate()
step = int(len(xNew)/N_GPs)
advance = 0
for k in range(N_GPs):
    plt.axvline(xNew[int(advance)], linestyle='--', linewidth=3,
                color='black')
    ax.plot(xNew, betas[:,k], color=c[k], linewidth=2,
            label='Beta: '+str(k))
    advance += step

ax.set_xlabel('Date-time')
ax.set_ylabel('Predictive contribution')

plt.show()