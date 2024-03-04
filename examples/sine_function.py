import numpy as np
import matplotlib.pyplot as plt
from models.DGP import DistributedGP as DGP
from models.DDPGP import DistributedDPGP as DDPGP

# Generate always the same random numbers
from numpy.random import RandomState
prob = RandomState(123)

# Define the domain of the function and the number of observations
N = 2000
x = np.linspace(0, 10, N)

# Corrupt samples with noise generated from 2 independent sources
sine = lambda x: np.sin(x)
f = sine(x)
y = np.zeros(N)

# Noise structure
std1 = 0.1
std2 = 1
pi1 = 0.9
pi2 = 1 - pi1

# proportionalities vector
u = prob.uniform(0,1,N)

for i in range(N):
    if u[i] < pi1:
        y[i] = f[i] + std1*prob.randn()
    else:
        y[i] = f[i] + std2*prob.randn()

# define the covariance function
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

se = 1**2 * RBF(length_scale=0.5, length_scale_bounds=(0.07, 0.9))
wn = WhiteKernel(noise_level=0.5**2, noise_level_bounds=(1e-6,0.7))

kernel = se + wn

# training
N_GPs = 5
model = DDPGP(x, y, N_GPs=N_GPs, init_K=7, kernel=kernel, normalise_y=True)
model.train(pseudo_sparse=True)

# predictions
mu, std, betas = model.predict(x)

"""
Compare with DGP
"""
dgp = DGP(x, y, N_GPs=N_GPs, kernel=kernel)
muGP, stdGP, betasGP = dgp.predict(x)

### CALCULATING THE OVERALL MSE
from sklearn.metrics import mean_squared_error
print("Mean Squared Error (DGP)     : ", mean_squared_error(muGP, f))
print("Mean Squared Error (DDPGP)   : ", mean_squared_error(mu, f))

#-----------------------------------------------------------------------------
# EXPERTS PREDICTIVE CONTRIBUTION
#-----------------------------------------------------------------------------

fig, ax = plt.subplots()
fig.autofmt_xdate()
step = int(len(x)/N_GPs)
advance = 0
for k in range(N_GPs):
    plt.axvline(x[int(advance)], linestyle='--', linewidth=3,
                color='grey')
    ax.plot(x, betas[:,k], color=model.c[k], linewidth=2,
            label='Beta: '+str(k))
    advance += step

ax.set_xlabel('Date-time')
ax.set_ylabel('Predictive contribution')

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------

plt.figure()
advance = 0
for k in range(N_GPs):
    plt.axvline(x[int(advance)], linestyle='--', linewidth=3,
                color='lightgrey')
    advance += step

plt.plot(x, y, '*', color='grey', linewidth = 4,
         label='Noisy data')    
plt.plot(x, f, color='black', linewidth = 4,    label='Sine function')
plt.plot(x, muGP, color='red', linewidth = 4,label='DGP')
plt.plot(x, mu, color='limegreen', linewidth = 4,     label='DDPGP')
plt.title('Regression Performance', fontsize=20)
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(prop={"size":20})

plt.show()