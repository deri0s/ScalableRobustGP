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

print('In the main script: ', se.base_kernel.lengthscale.tolist(), '\n')
covar_module = InducingPointKernel(se,
                                   inducing_points=inducing_points,
                                   likelihood=likelihood)

# start_time = time.time()
sgp = DPSGP(X_train, y_train, init_K=7,
            gp_model='Sparse',
            prior_mean=ConstantMean(), kernel=covar_module,
            lengthscale=100*np.ones(X.shape[-1]),
            noise_var = 0.36,
            floating_point=floating_point,
            normalise_y=True,
            DP_max_iter=380,
            print_conv=True, plot_conv=True, plot_sol=True)
# # sgp.train()
# # mus, stds = sgp.predict(X_test)
# # # comp_time = time.time() - start_time

# # sgp.gp.covar_module.kernel.lengthscale

# # print(f'DPSGP cleaning time: {comp_time:.2f} seconds')

# # # get inducing points indices
# # _z_indices = sgp._z_indices