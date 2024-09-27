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
X_df = pd.read_excel(file, sheet_name='X_norm')
y_df = pd.read_excel(file, sheet_name='y')
y_raw_df = pd.read_excel(file, sheet_name='y_raw')
t_df = pd.read_excel(file, sheet_name='timelags')

# Pre-Process training data
X, y0, N0, D, max_lag, time_lags = dpm.align_arrays(X_df, y_df, t_df)

# Replace zero values with interpolation
zeros = y_raw_df.loc[y_raw_df['raw_furnace_faults'] <= 1e-1]
y_raw_df['raw_furnace_faults'][zeros.index] = None
y_raw_df.interpolate(inplace=True)

# Remove the first max_lag points from the date_time array.
y_raw = dpm.adjust_time_lag(y_raw_df['raw_furnace_faults'].values,
                            shift=0,
                            to_remove=max_lag)

date_time = dpm.adjust_time_lag(y_df['Time stamp'].values,
                                shift=0,
                                to_remove=max_lag)

# Train and test data
N, D = np.shape(X)
start_train = y_df[y_df['Time stamp'] == '2021-04-01 00:00:00'].index[0]
end_train = y_df[y_df['Time stamp'] == '2021-04-08 00:00:00'].index[0]
model_N = 1

step = int(end_train - start_train)
end_train = start_train + int(model_N*step)

X_train, y_train = X[start_train:end_train], y_raw[start_train:end_train]
N_train = len(y_train)

end_test = end_train
X_test, y_test = X[start_train:end_test], y_raw[start_train:end_test]

date_time = date_time[start_train:end_test]
y_raw = y_raw[start_train:end_test]
y_rect = y0[start_train:end_test]

# Save memory
del X_df, y_df, dpm

"""
Normalise outputs
See if normalising the outputs provides better predictions compared
to standardising them
"""

scaler = minmax(feature_range=(0,1))
y_norm = scaler.fit_transform(np.vstack(y_train))

"""
DPSGP
"""
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import InducingPointKernel, ScaleKernel, RBFKernel as RBF
from models.dpsgp_gpytorch import DirichletProcessSparseGaussianProcess as DPSGP

# Convert data to torch tensors to input inducing points
floating_point = torch.float64
X_temp = torch.tensor(X_train, dtype=floating_point)
inducing_points = X_temp[::10, :]

likelihood = GaussianLikelihood()

se = ScaleKernel(RBF(ard_num_dims=X_train.shape[-1],
                     lengthscale=0.9))
covar_module = InducingPointKernel(se,
                                   inducing_points=inducing_points,
                                   likelihood=likelihood)

start_time = time.time()
gp = DPSGP(X_train, y_train, init_K=7,
           gp_model='Sparse',
           prior_mean=ConstantMean(), kernel=covar_module,
           noise_var = 0.05,
           floating_point=floating_point,
           normalise_y=True,
           print_conv=False, plot_conv=True, plot_sol=False)
gp.train()
mus, stds = gp.predict(X_train)
comp_time = time.time() - start_time

# unormalised predictions
# mu = scaler.fit_transform(mus)
# std = scaler.fit_transform(stds)

print(f'DPSGP training time: {comp_time:.2f} seconds')

# get inducing points indices
_z_indices = gp._z_indices

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------

fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
fig.autofmt_xdate()

print('date-time: ', np.shape(date_time))
ax.fill_between(date_time,
                mus + 3*stds, mus - 3*stds,
                alpha=0.5, color='lightcoral',
                label='3$\\sigma$')
ax.plot(date_time, y_raw, color='grey', label='Raw')
ax.plot(date_time, y_rect, color='blue', label='Filtered')
ax.plot(date_time, mus, color="red", linewidth = 2.5, label="DPSGP")
plt.axvline(date_time[N_train-1], linestyle='--', linewidth=3,
            color='black')

ax.vlines(
    x=date_time[::10],
    ymin=-0.5,
    ymax=y_train.max().item(),
    alpha=0.3,
    linewidth=1.5,
    ls='--',
    label="z0",
    color='grey'
)
dt0 = date_time[gp.indices[0]]
ax.vlines(
    # Sparse clean data
    x=dt0[_z_indices],
    ymin=-0.5,
    ymax=y_train.max().item(),
    alpha=0.3,
    linewidth=1.5,
    label="z*",
    color='orange'
)
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

# ----------------------------------------------------------------------------
# PCA and PLOTS
# ----------------------------------------------------------------------------
pca = PCA(n_components=2)
pca.fit(X)
Xt = pca.transform(X)

# PCA on training data
Xt_train = pca.transform(X_train)

# PCA on clean data
X0_train = X_train[gp.indices[0], :]
Xt_train_clean = pca.transform(X0_train)

# PCA on sparse clean data
X0_sparse = X0_train[_z_indices]
Xt_train_sparse_clean = pca.transform(X0_sparse)

# PCA on test data
Xt_test = pca.transform(X_test)
    
# Plot at each 1000 points
fig, ax = plt.subplots()
ax.plot(Xt[:, 0], Xt[:, 1], 'o', markersize=0.9, c='grey',
        label='Available training data', alpha=0.9)
ax.plot(Xt_train[:, 0], Xt_train[:, 1], 'o', markersize=8.9, c='orange',
        label='Used Training data', alpha=0.6)
ax.plot(Xt_train_sparse_clean[:, 0], Xt_train_sparse_clean[:, 1],
        'o', markersize=8.9, c='green',
        label='Sparse-clean', alpha=0.6)
ax.set_xlim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 0])))
ax.set_ylim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 1])))
plt.legend(loc=0, prop={"size":16}, facecolor="white", framealpha=1.0)

#-----------------------------------------------------------------------------
# CLUSTERING PLOT
#-----------------------------------------------------------------------------

color_iter = ['lightgreen', 'orange','red', 'brown','black']

# DP-GP
enumerate_K = [i for i in range(gp.K_opt)]

fig, ax = plt.subplots()
# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

fig.autofmt_xdate()
ax.set_title(" Clustering performance", fontsize=18)
if gp.K_opt != 1:
    for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
        ax.plot(date_time[gp.indices[k]], y_raw[gp.indices[k]],
                'o',color=c, markersize = 8, label='Noise Level '+str(k))
ax.plot(date_time, mus, color="green", linewidth = 2, label=" DPGP")
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)
plt.show()