import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler as minmax
from sklearn.metrics import mean_squared_error as mse
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from models.dgp import DistributedGP as DGP
from models.ddpgp import DistributedDPGP as DDPGP


plt.close('all')

# Read excel file
file_name = 'Synthetic.xlsx'

# Read data
df = pd.read_excel(file_name, sheet_name='Training')
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
train_scaler = minmax()
X_norm = train_scaler.fit_transform(X.reshape(-1,1))

test_scaler = minmax()
X_test_norm = test_scaler.fit_transform(X_test.reshape(-1,1))

labels_df = pd.read_excel(file_name, sheet_name='Real labels')

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
se = 1**2 * RBF(length_scale=0.94**2, length_scale_bounds=(0.07, 0.9))
#                variance
wn = WhiteKernel(noise_level=0.0025, noise_level_bounds=(1e-6,0.7))

kernel = se + wn

"""
TRAINING AND PREDICTIONS
"""
# Common variables
N_GPs = 2

gp = DGP(X_norm, y, N_GPs=N_GPs, kernel=kernel)
gp.train()
muGP, stdGP, betasGP = gp.predict(X_test_norm)

# The DDPGP model
kernels = []
for k in range(N_GPs):
    kernels.append(kernel)

dgp = DDPGP(X_norm, y, N_GPs, 7, kernels, normalise_y=True,
            plot_expert_pred=True)
dgp.train()
muMix, stdMix, betas = dgp.predict(X_test_norm)

### CALCULATING THE OVERALL MSE
F = 150 * X_test * np.sin(X_test)

print('\nMean Squared Error')
print("Distributed GP    : ", mse(muGP, F))
print("Distributed DPGP  : ", mse(muMix, F))

#-----------------------------------------------------------------------------
# Plot beta
#-----------------------------------------------------------------------------

fig, ax = plt.subplots()
fig.autofmt_xdate()
step = int(len(X_test)/N_GPs)
advance = 0
for k in range(N_GPs):
    plt.axvline(X_test[int(advance)], linestyle='--', linewidth=3,
                color='black')
    ax.plot(X_test, betasGP[:,k], color=dgp.c[k], linewidth=2,
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
    plt.axvline(X_test[int(advance)], linestyle='--', linewidth=3,
                color='lime')
    advance += step
    
plt.plot(X_test, F, color='black', linewidth = 4, label='Sine function')
plt.plot(X_test, muGP, color='blue', linewidth = 4,
          label='DPGP')
plt.plot(X_test, muMix, color='red', linestyle='-', linewidth = 4,
          label='DDPGP')
plt.title('Regression Performance', fontsize=20)
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(prop={"size":20})

# ----------------------------------------------------------------------------
# CONFIDENCE BOUNDS
# ----------------------------------------------------------------------------

plt.figure()
plt.fill_between(X_test,
                 muMix + 3*stdMix, muMix - 3*stdMix,
                 alpha=0.5,color='lightcoral',
                 label='3$\\sigma$ (DDPGP)')

plt.fill_between(X_test,
                 muGP + 3*stdGP, muGP - 3*stdGP,
                 alpha=0.5,color='lightblue',
                 label='3$\\sigma$ (DGP)')

plt.plot(X, y, 'k*', label='Training Data')
plt.plot(X_test, muGP, linewidth=2.5, color='blue', label='DGP')
plt.plot(X_test, muMix, linewidth=2.5, color='red', label='DDPGP')
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(prop={"size":20})

plt.show()