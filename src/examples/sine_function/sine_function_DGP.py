import paths
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from models.DGP import DistributedGP as DGP
from models.DDPGP import DistributedDPGP as DDPGP


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
xNew = x_test_df['X_star'].values

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
    

# ! The following covariance function only works for K=2. Use a
# ! single kernel or add the K=k number of kernels, otherwise
# Covariance functions
se = 1**2 * RBF(length_scale=0.5, length_scale_bounds=(0.07, 0.9))
wn = WhiteKernel(noise_level=0.5**2, noise_level_bounds=(1e-6,0.7))

kernel = se + wn

# Initialise the 2nd exp with the following hyper to obtain better results 
se = 1**2 * RBF(length_scale=1.7, length_scale_bounds=(1e-3,1e3))
kernels = []
kernels.append(kernel)
kernels.append(se + wn)

"""
TRAINING AND PREDICTIONS
"""
# Common variables
N_GPs = 2

gp = DGP(X, Y, N_GPs=N_GPs, kernel=kernel)
gp.train()
muGP, stdGP, betasGP = gp.predict(xNew)

# The DDPGP model
# Uncomment if all kernel hyper can be initialised with the same values
# ! Not the case for the sine function
# kernels = []
# for k in range(N_GPs):
#     kernels.append(kernel)

dgp = DDPGP(X, Y, N_GPs, 7, kernels, normalise_y=True,
            plot_expert_pred=True)
dgp.train()
muMix, stdMix, betas = dgp.predict(xNew)

### CALCULATING THE OVERALL MSE
F = 150 * xNew * np.sin(xNew)
print("Mean Squared Error (Distributed GP)    : ", mean_squared_error(muGP, F))
print("Mean Squared Error (Distributed DPGP)  : ",
      mean_squared_error(muMix, F))
print('mu: ', np.shape(muMix))
print('std: ', np.shape(stdMix))

#-----------------------------------------------------------------------------
# Plot beta
#-----------------------------------------------------------------------------

fig, ax = plt.subplots()
fig.autofmt_xdate()
step = int(len(xNew)/N_GPs)
advance = 0
for k in range(N_GPs):
    plt.axvline(xNew[int(advance)], linestyle='--', linewidth=3,
                color='black')
    ax.plot(xNew, betasGP[:,k], color=dgp.c[k], linewidth=2,
            label='Beta: '+str(k))
    plt.legend()
    advance += step

ax.set_xlabel('Date-time')
ax.set_ylabel('Predictive contribution')

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------
plt.figure()
advance = 0
for k in range(N_GPs):
    plt.axvline(xNew[int(advance)], linestyle='--', linewidth=3,
                color='lime')
    advance += step
    
plt.plot(xNew, F, color='black', linewidth = 4, label='Sine function')
plt.plot(xNew, muGP, color='blue', linewidth = 4,
          label='DPGP')
plt.plot(xNew, muMix, color='red', linestyle='-', linewidth = 4,
          label='DDPGP')
plt.title('Regression Performance', fontsize=20)
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(prop={"size":20})

# ----------------------------------------------------------------------------
# CONFIDENCE BOUNDS
# ----------------------------------------------------------------------------

plt.figure()
plt.fill_between(xNew,
                 muMix + 3*stdMix, muMix - 3*stdMix,
                 alpha=0.5,color='lightgreen',
                 label='Confidence \nBounds (DDPGP)')

plt.fill_between(xNew,
                 muGP + 3*stdGP, muGP - 3*stdGP,
                 alpha=0.5,color='green',
                 label='Confidence \nBounds (DGP)')
    
plt.plot(xNew, muGP, linewidth=2.5, color='green', label='DGP')
plt.plot(xNew, muMix, linewidth=2.5, color='green', label='DDPGP')
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(prop={"size":20})

plt.show()