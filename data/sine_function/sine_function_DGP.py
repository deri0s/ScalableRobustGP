import numpy as np
import matplotlib.pyplot as plt
from time import process_time
import sys
sys.path.append('../..')
import numpy as np
from sklearn.metrics import mean_squared_error as mse
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from DGP import DistributedGP as DGP

# The true 1D function
def f(x):
    return x * np.sin(x)
    
# Define the domain of the function and the number of observations.
N = 1000
X = np.linspace(0, 10, N)

# Corrupt observations with noise generated from a mixture of 2 Gaussians
F = f(X)
Y = np.zeros(N)

for i in range(N):
    Y[i] = F[i] + 0.5*np.random.randn()
                
# Create test data
N_star = 30000    # No. test data
X_star = np.linspace(0, 10, N_star)   # Test inputs
F_star = f(X_star)   # Truth test data

# Initialise the kernel and the model to be tested
se = 1**2 * RBF(length_scale=2, length_scale_bounds=(1, 1e3))
wn = WhiteKernel(noise_level=0.8**2,noise_level_bounds=(1e-5, 0.05))

kernel = se + wn
    
# DPGP
N_GPs = 3
dgp = DGP(X, Y, 3, kernel=kernel)
mu_star, std_star, betas = dgp.predict(np.vstack(X_star))

# Test mean prediction
print('Error (RMSE): ', mse(F_star,mu_star))
# assert np.allclose(mu_star, np.vstack(F_star), atol=0.3)

plt.plot(X, Y, 'o', color='black')
plt.plot(X_star, F_star, color='darkgreen', linewidth=3)
plt.plot(X_star, mu_star, color='r', linewidth=3)

##############################################################################
# Plot beta
##############################################################################
c = ['red', 'orange', 'blue', 'black', 'green', 'cyan', 'darkred', 'pink',
     'gray', 'magenta','lightgreen', 'darkblue', 'yellow']

fig, ax = plt.subplots()
fig.autofmt_xdate()
for k in range(N_GPs):
    ax.plot(X_star, betas[:,k], color=c[k], linewidth=2,
            label='Beta: '+str(k))

ax.set_xlabel('Date-time')
ax.set_ylabel('Predictive contribution')
