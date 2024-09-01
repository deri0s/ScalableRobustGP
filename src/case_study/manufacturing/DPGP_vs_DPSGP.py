import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from models.dpgp import DirichletProcessGaussianProcess as DPGP
from case_study.manufacturing.pre_processing import data_processing_methods as dpm
from sklearn.decomposition import PCA

"""
NSG data
"""
# NSG post processes data location
file = 'data/processed/NSG_data.xlsx'

# Training df
X_df = pd.read_excel(file, sheet_name='X_training_stand')
y_df = pd.read_excel(file, sheet_name='y_training')
y_raw_df = pd.read_excel(file, sheet_name='y_raw_training')
t_df = pd.read_excel(file, sheet_name='time')

# Pre-Process training data
X, y0, N0, D, max_lag, time_lags = dpm.align_arrays(X_df, y_df, t_df)

# Replace zero values with interpolation
zeros = y_raw_df.loc[y_raw_df['raw_furnace_faults'] <= 1e-1]
y_raw_df['raw_furnace_faults'][zeros.index] = None
y_raw_df.interpolate(inplace=True)

# Process raw targets
# Just removes the first max_lag points from the date_time array.
y_raw = dpm.adjust_time_lag(y_raw_df['raw_furnace_faults'].values,
                            shift=0,
                            to_remove=max_lag)

# Extract corresponding time stamps. Note this essentially just
# removes the first max_lag points from the date_time array.
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

print('N-train: ', len(X_train))

"""
DPGP regression
"""
# Save memory
del X_df, y_df, dpm

# Length scales
# ls = [0.0612, 3.72, 200, 200, 200, 200, 4.35, 0.691, 200, 200]
# ls = [7, 64, 7, 7.60, 10, 7, 7, 123, 76, 78]
ls = 1e5*np.ones(10)

# Kernels
se = 1**2 * RBF(length_scale=ls, length_scale_bounds=(0.25, 1e5))
wn = WhiteKernel(noise_level=0.61**2, noise_level_bounds=(1e-5, 1))

kernel = se + wn

start_time = time.time()
dpgp = DPGP(X_train, y_train, init_K=7, kernel=kernel, DP_max_iter=300)
dpgp.train(pseudo_sparse=True)

mu, std = dpgp.predict(X_train)
comp_time = time.time() - start_time

print(f'DPGP training time: {comp_time:.2f} seconds')

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

print(f'DPSGP training time: {comp_time:.2f} seconds')

# get inducing points indices
_z = gp._z_indices

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
                label='3$\sigma$')
ax.plot(date_time, y_raw, color='grey', label='Raw')
ax.plot(date_time, y_rect, color='blue', label='Filtered')
ax.plot(date_time, mu, color="green", linewidth = 2.5, label="DPGP")
ax.plot(date_time, mus, color="red", linewidth = 2.5, label="DPSGP")
plt.axvline(date_time[N_train-1], linestyle='--', linewidth=3,
            color='black')
# ax.vlines(
#     x=_z,
#     ymin=y_train.min().item(),
#     ymax=y_train.max().item(),
#     alpha=0.3,
#     linewidth=1.5,
#     label="z*",
#     color='orange'
# )
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)
plt.show()
# # ----------------------------------------------------------------------------
# # PCA and PLOTS
# # ----------------------------------------------------------------------------
# pca = PCA(n_components=2)
# pca.fit(X)
# Xt = pca.transform(X)

# # PCA on training data
# Xt_train = pca.transform(X_train)

# # PCA on test data
# Xt_test = pca.transform(X_test)
    
# # Plot at each 1000 points
# fig, ax = plt.subplots()
# ax.plot(Xt[:, 0], Xt[:, 1], 'o', markersize=0.9, c='grey',
#         label='Available training data', alpha=0.9)
# ax.plot(Xt_train[:, 0], Xt_train[:, 1], 'o', markersize=8.9, c='orange',
#         label='Used Training data', alpha=0.6)
# ax.plot(Xt_test[:,0], Xt_test[:,1], '*', markersize=5.5,
#         c='purple', label='test data', alpha=0.6)
# ax.set_xlim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 0])))
# ax.set_ylim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 1])))

# #-----------------------------------------------------------------------------
# # CLUSTERING PLOT
# #-----------------------------------------------------------------------------

# color_iter = ['lightgreen', 'orange','red', 'brown','black']

# # DP-GP
# enumerate_K = [i for i in range(dpgp.K_opt)]

# fig, ax = plt.subplots()
# # Increase the size of the axis numbers
# plt.rcdefaults()
# plt.rc('xtick', labelsize=14)
# plt.rc('ytick', labelsize=14)

# fig.autofmt_xdate()
# ax.set_title(" Clustering performance", fontsize=18)
# if dpgp.K_opt != 1:
#     for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
#         ax.plot(date_time[dpgp.indices[k]], y_raw[dpgp.indices[k]],
#                 'o',color=c, markersize = 8, label='Noise Level '+str(k))
# ax.plot(date_time, mu, color="green", linewidth = 2, label=" DPGP")
# ax.set_xlabel(" Date-time", fontsize=14)
# ax.set_ylabel(" Fault density", fontsize=14)
# plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)
# plt.show()