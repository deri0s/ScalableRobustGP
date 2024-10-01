import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler as minmax
from case_study.manufacturing.data_and_preprocessing.raw import data_processing_methods as dpm
from sklearn.decomposition import PCA

"""
NSG data
"""
# NSG post processes data location
file = 'data_and_preprocessing/processed/NSG_processed_data.xlsx'

# Training df
X_df = pd.read_excel(file, sheet_name='X_stand')
y_df = pd.read_excel(file, sheet_name='y')
y_raw_df = pd.read_excel(file, sheet_name='y_raw')
t_df = pd.read_excel(file, sheet_name='timelags')

# Pre-Process training data
X, y0, N0, D, max_lag, time_lags = dpm.align_arrays(X_df, y_df, t_df)

# Replace zero values with interpolation
zeros = y_raw_df.loc[y_raw_df['raw_furnace_faults'] <= 1e-1]
y_raw_df.loc[zeros.index, 'raw_furnace_faults'] = None
y_raw_df.interpolate(inplace=True)

# Remove the first max_lag points (the same as align_arrays)
y_raw = dpm.adjust_time_lag(y_raw_df['raw_furnace_faults'].values,
                            shift=0,
                            to_remove=max_lag)

date_time = dpm.adjust_time_lag(y_df['Time stamp'].values,
                                shift=0,
                                to_remove=max_lag)

# Train and test data
N, D = np.shape(X)
start_train = y_df[y_df['Time stamp'] == '2020-08-16 00:00:00'].index[0]
end_train1 = y_df[y_df['Time stamp'] == '2020-08-23 00:00:00'].index[0]
model_N = 1

step1 = int(end_train1 - start_train)
end_train = start_train + step1

X_train1, y_train1 = X[start_train:end_train], y_raw[start_train:end_train]
N_train = len(y_train1)

start_train2 = end_train + 150
end_train2 = start_train2 + 282

X_train2, y_train2 = X[start_train2:end_train2], y_raw[start_train2:end_train2]

# concatenate
X_train = np.concatenate((X_train1, X_train2))
y_train = np.concatenate((y_train1, y_train2))

end_test = end_train2
X_test, y_test = X[start_train:end_test], y_raw[start_train:end_test]

date_time = date_time[start_train:end_test]
y_raw = y_raw[start_train:end_test]
y_rect = y0[start_train:end_test]

# Save memory
del X_df, y_df, dpm

"""
DPSGP
"""
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import InducingPointKernel, ScaleKernel, RBFKernel as RBF
from models.dpsgp_gpytorch import DirichletProcessSparseGaussianProcess as DPSGP
from sklearn.metrics import mean_squared_error

# Convert data to torch tensors to input inducing points
floating_point = torch.float64
X_temp = torch.tensor(X_train, dtype=floating_point)
inducing_points = X_temp[::10, :]

likelihood = GaussianLikelihood()

# Cross-validation
N_ls = 10
N_std= 10
param_dist = {"ls": np.linspace(1e-5, 3, N_ls),
              "std": np.linspace(1e-3, 0.1, N_std)}

# Perform randomized search cross-validation
best_score = float('inf')
best_params = None
scores = np.zeros((N_ls, N_std))

# read validation data
mu_val_df = pd.read_csv("validation_data.csv")
mu_val = mu_val_df["mu_val"]

for i, ls in enumerate(param_dist['ls']):
    for j, std in enumerate(param_dist['std']):

        se = ScaleKernel(RBF(ard_num_dims=X_train.shape[-1],
                        lengthscale=ls))
        covar_module = InducingPointKernel(se,
                                        inducing_points=inducing_points,
                                        likelihood=likelihood)

        gp = DPSGP(X_train, y_train, init_K=7,
                gp_model='Sparse',
                prior_mean=ConstantMean(), kernel=covar_module,
                noise_var = std,
                floating_point=floating_point,
                normalise_y=True,
                DP_max_iter=400,
                print_conv=False, plot_conv=False, plot_sol=False)
        gp.train()
        mu, stds = gp.predict(X_test)

        score = mean_squared_error(mu_val, mu)

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

# get inducing points indices
_z_indices = gp._z_indices

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