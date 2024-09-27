import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io as sio
import mat73
from sklearn.preprocessing import MinMaxScaler as minmax
from sklearn.preprocessing import StandardScaler as ss
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from models.dpsgp_gpytorch import DirichletProcessSparseGaussianProcess as DPSGP

"""
No sirve
"""

plt.close('all')

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------

file_name = 'motorcycle.mat'
motorcycle_data = sio.loadmat(file_name)
X_org = motorcycle_data['X']
Y_org = motorcycle_data['y']
N = len(X_org)

# Load MVHGP predictive mean and variance
vhgp_path = 'mvhgp.mat'
mvhgp = mat73.loadmat(vhgp_path)
std_var = mvhgp['fnm']
x = mvhgp['xm']
mu_var = mvhgp['ym']

#-----------------------------------------------------------------------------
# STANDARDISE OR NORMALISE DATA
#-----------------------------------------------------------------------------
preprocess = "ss"

if preprocess == "ss":
    train_scaler = ss()
elif preprocess == "mm":
    train_scaler = minmax()
else:
    print('Not a valid preprocessing method')

output_scaler = ss()
X = train_scaler.fit_transform(X_org.reshape(-1, 1))

# Convert data to torch tensors
floating_point = torch.float64

# Cross-validation
N_ls = 10
N_std= 10
param_dist = {"ls": np.linspace(1e-5, 2, N_ls),
              "std": np.linspace(1e-3, 0.02, N_std)}

# Perform randomized search cross-validation
best_score = float('inf')
best_params = None
scores = np.zeros((N_ls, N_std))

for i, ls in enumerate(param_dist['ls']):
    for j, std in enumerate(param_dist['std']):
        # Define the kernel
        rbf_kernel = RBFKernel()
        rbf_kernel.lengthscale = ls
        rbf_kernel.lengthscale_constraint = gpytorch.constraints.Interval(1e-5, 10)

        scale_kernel = ScaleKernel(rbf_kernel)
        scale_kernel.outputscale = 1.0
        scale_kernel.outputscale_constraint = gpytorch.constraints.Interval(0.9, 1.1)

        covar_module = scale_kernel

        dpgp = DPSGP(X, np.hstack(Y_org), init_K=7,
                    gp_model='Standard',
                    prior_mean=ConstantMean(), kernel=covar_module,
                    noise_var=std,
                    floating_point=floating_point,
                    normalise_y=True,
                    print_conv=False, plot_conv=False, plot_sol=False)
        dpgp.train()
        mu, pred_std = dpgp.predict(X)

        score = mean_squared_error(mu_var, mu)

        scores[i, j] = round(score, 2)

        if score < best_score:
            best_score = score
            best_params = {'length_scale': ls,
                           'noise_variance': std}
            mus = np.copy(mu)

print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")

param_dist['score(MSE)'] = scores

import pandas as pd
cv_df = pd.DataFrame(scores)
print('MSE\n', cv_df.head(5))
print('ls: ', param_dist['ls'])
print('std: ', param_dist['std'])

#-----------------------------------------------------------------------------
# CROSS-VALIDATION 
#-----------------------------------------------------------------------------

# Plot the scores
fig, ax = plt.subplots()
plt.imshow(scores, label='Grid search')
for (j,i),label in np.ndenumerate(scores):
    ax.text(i,j,label,ha='center',va='center',
            bbox=dict(facecolor='white', alpha=0.9))
plt.colorbar()
plt.xlabel('Standard Deviation')
plt.ylabel('Length scale')

plt.show()