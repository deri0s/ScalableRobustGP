import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.insert(0, 'C:\Diego\PhD\Code\phdCode')
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn import mixture as m
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.metrics import mean_squared_error

plt.close('all')

# Read excel file
file_name = 'Synthetic.xlsx'
df = pd.read_excel(file_name, sheet_name='Training')
xNewdf = pd.read_excel(file_name, sheet_name='Testing')
labels_df = pd.read_excel(file_name, sheet_name='Real labels')

# Get training data
X = np.vstack(df['X'].values)
Y = np.vstack(df['Y'].values)
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

# ----------------------------------------------------------------------------
# GP REGRESSION
# ----------------------------------------------------------------------------
se = 1**2 * RBF(length_scale=1, length_scale_bounds=(1e-6,1e3))
wn = WhiteKernel(noise_level=0.5**2, noise_level_bounds=(1e-6,1e3))

kernel = se + wn
del se, wn

gp = GPR(kernel=kernel, alpha=0,n_restarts_optimizer=1,normalize_y=True)
gp.fit(X,Y)
mu = gp.predict(X, return_std=False)
mu = np.vstack(mu)
errors = mu - Y

print('Hyperparameters: ', gp.kernel_)

# ----------------------------------------------------------------------------
# DPGP CLUSTERING
# ----------------------------------------------------------------------------
gmm =m.BayesianGaussianMixture(n_components=7,
                               covariance_type='spherical',
                               max_iter=70,
                               weight_concentration_prior_type='dirichlet_process',
                               init_params="random")

# The data labels correspond to the position of the mix parameters
labels = gmm.fit_predict(errors)

# Capture the pies: It is a tuple with not ordered elements
pies_no = np.sort(gmm.weights_)

# Capture the sigmas
covs = np.reshape(gmm.covariances_, (1, gmm.n_components))
covariances = covs[0]
stds_no = np.sqrt(covariances)

# Get the width of each Gaussian 
not_ordered = np.array(np.sqrt(gmm.covariances_))

# Initialise the ordered pies, sigmas and responsibilities
pies = np.zeros(gmm.n_components)
stds = np.zeros(gmm.n_components)
resp_no = gmm.predict_proba(errors)
resp = []

# Order the Gaussian components by their width
order = np.argsort(not_ordered)

indx = []    
# The 0 position or first element of the 'order' vector corresponds
# to the Gaussian with the min(std0, std1, std2, ..., stdk)
for new_order in range(gmm.n_components):
    pies[new_order] = pies_no[order[new_order]]
    stds[new_order] = stds_no[order[new_order]]
    resp.append(resp_no[:, order[new_order]])
    indx.append([i for (i, val) in enumerate(labels) if val == order[new_order]])

# The ensemble task has to account for empty subsets.                
indices = [x for x in indx if x != []]
K_opt = len(indices)         # The optimum number of components
X0 = X[indices[0]]
Y0 = Y[indices[0]]

### CALCULATING THE OVERALL MSE
F = 150 * X * np.sin(X)

### Print results for the EM-GP model
print('\n MODEL PARAMETERS DPGP (with normalisation): \n')
print(' Number of components identified, K = ', K_opt)
print('Proportionalities: ', pies)
print('Noise Stds: ', stds)

############################## PLOT THE RESULST ##############################
color_iter = ['lightgreen', 'orange', 'red']
enumerate_real = [i for i in range(3)]
enumerate_K = [i for i in range(K_opt)]

############################ REAL NOISE LABELS ###############################

# plt.figure()
# plt.title(" Data corrupted with non-Gaussian noise ", fontsize=20)
# plt.plot(X, F, color="black", label="Sine function")
# for i, (k, c) in enumerate(zip(enumerate_real, color_iter)):
#     plt.plot(X[indices[k]], Y[indices[k]], 'o',color=c, markersize = 9,
#              label='Gaussian noise '+str(k+1))
# plt.xlabel(' X ', fontsize=20)
# plt.ylabel(' f(X) ', fontsize=20)
# plt.legend(loc=3, prop={"size":25})

# ----------------------------------------------------------------------------
# CLUSTERING
# ----------------------------------------------------------------------------
plt.figure()
plt.title(" Clustering performance ", fontsize=20)
color_iter = ['lightgreen', 'orange', 'red']
nl = ['Noise level 0', 'Noise level 1', 'Noise level 2']
plt.plot(X, mu, color="green", label="DPGP")
plt.plot(X, Y, 'o', color="black", label="Observations")
plt.plot(X[indices[0]], Y[indices[0]], 'o',color='lightgreen', markersize = 9, label='Gaussian noise 0')
plt.plot(X[indices[1]], Y[indices[1]], 'o',color='orange', markersize = 9, label='Gaussian noise 1')
plt.plot(X[indices[2]], Y[indices[2]], 'o',color='red', markersize = 9, label='Gaussian noise 2')
# for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
#     plt.plot(X[indices[k]], Y[indices[k]], 'o',color=c,
#              markersize = 9, label=nl[k])
plt.xlabel(' X ', fontsize=16)
plt.ylabel(' f(X) ', fontsize=16)
plt.legend(loc=0, prop={"size":20})
plt.show()

plt.show()