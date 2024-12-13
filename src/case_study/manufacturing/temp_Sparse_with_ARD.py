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
file = 'data_and_preprocessing/processed/Spearman_corr_timelags.xlsx'

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
start_train = y_df[y_df['Time stamp'] == '2020-08-15'].index[0]
end_train = y_df[y_df['Time stamp'] == '2020-08-30'].index[0]
model_N = 1

X_train, y_train = X[start_train:end_train], y_raw[start_train:end_train]
N_train = len(X_train)

end_test = end_train + 200
X_test = X[start_train:end_test]
date_time = date_time[start_train:end_test]
y_raw = y_raw[start_train:end_test]
y_rect = y0[start_train:end_test]

print('N-train: ', N_train)

"""
DPSGP cleaning

GPytotch is very sensitive to the initial hyperparameters.
I first used the DPGP sklearn version to estimate the initial
lengthscales for this script.
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

se = ScaleKernel(RBF(ard_num_dims=X_train.shape[-1]))

covar_module = InducingPointKernel(se,
                                   inducing_points=inducing_points,
                                   likelihood=likelihood)

# lss = [0.284, 1.54e+04, 0.48, 0.662, 2.79e+04, 337, 4.86e+04, 3.71e+04, 1.13, 0.25]
lss = [1e+05, 342, 516, 0.468, 0.25, 6.57e+04, 1.33e+03, 0.878, 1.07, 4.71e+03]
start_time = time.time()
sgp = DPSGP(X_train, y_train, init_K=7,
            gp_model='Sparse',
            prior_mean=ConstantMean(), kernel=covar_module,
            lengthscale=lss, #0.05*np.ones(X.shape[-1]),
            N_iter=15,
            noise_var = 0.06,
            floating_point=floating_point,
            normalise_y=True,
            DP_max_iter=390,
            print_conv=True, plot_conv=True, plot_sol=True)
sgp.train()
mus, stds = sgp.predict(X_test)
comp_time = time.time() - start_time

print(f'DPSGP cleaning time: {comp_time:.2f} seconds')

print('\n Furnace parameters relevance')
d = {i: ls for i, ls in zip(X_df.columns, sgp.lengthscale[0])}
print(d)

# get inducing points indices
_z_indices = sgp._z_indices

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------
fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
fig.autofmt_xdate()

ax.fill_between(date_time,
                mus + 3*stds, mus - 3*stds,
                alpha=0.5, color='lightcoral',
                label='3$\\sigma$')
ax.plot(date_time, y_raw, color='grey', label='Raw')
ax.plot(date_time, y_rect, color='blue', label='Filtered')
ax.plot(date_time, mus, color="red", linewidth = 2.5, label="Cleaned")
plt.axvline(date_time[N_train-1], linestyle='--', linewidth=3,
            color='black')
plt.axvline(date_time[N_train+150-1], linestyle='--', linewidth=3,
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
dt0 = date_time[sgp.indices[0]]
print('N-clean: ', len(dt0))

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
plt.show()