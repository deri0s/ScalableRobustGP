import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler as ss
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
end_train1 = y_df[y_df['Time stamp'] == '2020-08-23 23:00:00'].index[0]
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

"""
DPSGP cleaning
"""
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import SpectralMixtureKernel as SM
from gpytorch.kernels import InducingPointKernel, ScaleKernel, RBFKernel as RBF
from models.dpsgp_gpytorch import DirichletProcessSparseGaussianProcess as DPSGP

# Convert data to torch tensors to input inducing points
floating_point = torch.float64
X_tensor = torch.tensor(X_train, dtype=floating_point)
inducing_points = X_tensor[::10, :]

likelihood = GaussianLikelihood()

se = ScaleKernel(RBF(ard_num_dims=X_train.shape[-1],
                     lengthscale=1.0))

covar_module = InducingPointKernel(se,
                                   inducing_points=inducing_points,
                                   likelihood=likelihood)

start_time = time.time()
sgp = DPSGP(X_train, y_train, init_K=7,
           gp_model='Sparse',
           prior_mean=ConstantMean(), kernel=covar_module,
           noise_var = 0.02132,
           floating_point=floating_point,
           normalise_y=True,
           DP_max_iter=400,
           print_conv=False, plot_conv=True, plot_sol=True)
sgp.train()
mus, stds = sgp.predict(X_test)
comp_time = time.time() - start_time

print(f'DPSGP training time: {comp_time:.2f} seconds')

# get inducing points indices
_z_indices = sgp._z_indices

# save predictions to use it in another scipt as the `true` fault_density
d = {"date_time": date_time, "mu_val": mus}

df = pd.DataFrame(d)
df.to_csv("validation_data.csv")

"""
GP extrapolation
"""
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import ExactMarginalLogLikelihood

# nomrlise targets
scaler = ss()
y_stand = scaler.fit_transform(y_train.reshape(-1,1))
y_torch = torch.tensor(y_stand, dtype=floating_point)
y_processed = np.hstack(y_torch[sgp.indices[0]])

X_processed = X_tensor[sgp.indices[0]]

class GP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, mu0, kernel, noise_var):
        super(GP, self).__init__(train_x, train_y, likelihood)
        likelihood.noise = noise_var
        self.mean_module = mu0
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

sm = SM(num_mixtures=2, ard_num_dims=X_processed.shape[-1])
sm_kernel = ScaleKernel(sm)

gp = GP(X_processed, y_processed,
        likelihood=likelihood, mu0=ConstantMean(),
        kernel=sm_kernel, noise_var=0.02)

# Train model
gp.train()
likelihood.train()

optimizer = torch.optim.Adam(gp.parameters(), lr=0.01)
mll = ExactMarginalLogLikelihood(likelihood, gp)

for i in range(100):
    optimizer.zero_grad()
    output = gp(X_processed)
    loss = -mll(output, torch.tensor(y_processed, dtype=floating_point))
    loss.backward()
    optimizer.step()

# Predictions
gp.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(gp(X_test))
        mu = observed_pred.mean
        stds = observed_pred.stddev

# #-----------------------------------------------------------------------------
# # REGRESSION PLOT
# #-----------------------------------------------------------------------------

# fig, ax = plt.subplots()

# # Increase the size of the axis numbers
# plt.rcdefaults()
# plt.rc('xtick', labelsize=14)
# plt.rc('ytick', labelsize=14)
# fig.autofmt_xdate()

# ax.fill_between(date_time,
#                 mus + 3*stds, mus - 3*stds,
#                 alpha=0.5, color='lightcoral',
#                 label='3$\\sigma$')
# ax.plot(date_time, y_raw, color='grey', label='Raw')
# ax.plot(date_time, y_rect, color='blue', label='Filtered')
# ax.plot(date_time, mus, color="red", linewidth = 2.5, label="DPSGP")
# plt.axvline(date_time[N_train-1], linestyle='--', linewidth=3,
#             color='black')
# plt.axvline(date_time[N_train+150-1], linestyle='--', linewidth=3,
#             color='black')

# ax.vlines(
#     x=date_time[::10],
#     ymin=-0.5,
#     ymax=y_train.max().item(),
#     alpha=0.3,
#     linewidth=1.5,
#     ls='--',
#     label="z0",
#     color='grey'
# )
# dt0 = date_time[gp.indices[0]]
# ax.vlines(
#     # Sparse clean data
#     x=dt0[_z_indices],
#     ymin=-0.5,
#     ymax=y_train.max().item(),
#     alpha=0.3,
#     linewidth=1.5,
#     label="z*",
#     color='orange'
# )
# ax.set_xlabel(" Date-time", fontsize=14)
# ax.set_ylabel(" Fault density", fontsize=14)
# plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

# # ----------------------------------------------------------------------------
# # PCA and PLOTS
# # ----------------------------------------------------------------------------
# pca = PCA(n_components=2)
# pca.fit(X)
# Xt = pca.transform(X)

# # PCA on training data
# Xt_train = pca.transform(X_train)

# # PCA on clean data
# X0_train = X_train[gp.indices[0], :]
# Xt_train_clean = pca.transform(X0_train)

# # PCA on sparse clean data
# X0_sparse = X0_train[_z_indices]
# Xt_train_sparse_clean = pca.transform(X0_sparse)

# # PCA on test data
# Xt_test = pca.transform(X_test)
    
# # Plot at each 1000 points
# fig, ax = plt.subplots()
# ax.plot(Xt[:, 0], Xt[:, 1], 'o', markersize=0.9, c='grey',
#         label='Available training data', alpha=0.9)
# ax.plot(Xt_train[:, 0], Xt_train[:, 1], 'o', markersize=8.9, c='orange',
#         label='Used Training data', alpha=0.6)
# ax.plot(Xt_train_sparse_clean[:, 0], Xt_train_sparse_clean[:, 1],
#         'o', markersize=8.9, c='green',
#         label='Sparse-clean', alpha=0.6)
# ax.set_xlim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 0])))
# ax.set_ylim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 1])))
# plt.legend(loc=0, prop={"size":16}, facecolor="white", framealpha=1.0)

# #-----------------------------------------------------------------------------
# # CLUSTERING PLOT
# #-----------------------------------------------------------------------------

# color_iter = ['lightgreen', 'orange','red', 'brown','black']

# # DP-GP
# enumerate_K = [i for i in range(gp.K_opt)]

# fig, ax = plt.subplots()
# # Increase the size of the axis numbers
# plt.rcdefaults()
# plt.rc('xtick', labelsize=14)
# plt.rc('ytick', labelsize=14)

# fig.autofmt_xdate()
# ax.set_title(" Clustering performance", fontsize=18)
# if gp.K_opt != 1:
#     for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
#         ax.plot(date_time[gp.indices[k]], y_raw[gp.indices[k]],
#                 'o',color=c, markersize = 8, label='Noise Level '+str(k))
# ax.plot(date_time, mus, color="green", linewidth = 2, label=" DPGP")
# ax.set_xlabel(" Date-time", fontsize=14)
# ax.set_ylabel(" Fault density", fontsize=14)
# plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)
# plt.show()