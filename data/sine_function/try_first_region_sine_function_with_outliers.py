import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.insert(0, 'C:\Diego\PhD\Code\phdCode')
# sys.path.append('../..')
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from DPGP import DirichletProcessGaussianProcess as DPGP
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
xNew = xNewdf['X_star'].values

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
    

### The covariance function
se = 1**2 * RBF(length_scale=0.6, length_scale_bounds=(1e-6,1e3))
wn = WhiteKernel(noise_level=1**2, noise_level_bounds=(1e-6,1e3))

kernel = se + wn
del se, wn

# The DPGP model
N_GPs = 2
step = int(len(X)/N_GPs)

gp = GPR(kernel, alpha=0, normalize_y = False, n_restarts_optimizer = 2)
gp.fit(X.reshape(-1,1)[:step], Y.reshape(-1,1)[:step])
mu, std = gp.predict(xNew.reshape(-1,1), return_std=True)

# The DPGP model
rgp = DPGP(X[:step], Y[:step], init_K=7, kernel=kernel, normalise_y=False)
rgp.train()
mur, stdr = rgp.predict(xNew, 2*N)

print('Optimal hyperparameters: ', rgp.kernel_)


##############################################################################
# Plot beta
##############################################################################
c = ['red', 'orange', 'blue', 'black', 'green', 'cyan', 'darkred', 'pink',
     'gray', 'magenta','lightgreen', 'darkblue', 'yellow']

# ### Regression performance: Comparison with a HGP
plt.figure()
advance = 0
for k in range(N_GPs):
    plt.axvline(xNew[int(advance)], linestyle='--', linewidth=3,
                color='lime')
    advance += step
    
plt.plot(X, Y, 'o', color='black')
plt.plot(xNew, mu, color='blue', linewidth = 4, label='GP')
plt.plot(xNew, mur, color='red', linewidth = 4, label='DPGP')
plt.title('Regression Performance', fontsize=20)
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(prop={"size":20})
plt.show()
