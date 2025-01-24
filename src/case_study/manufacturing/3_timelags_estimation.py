import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
from case_study.manufacturing.data_and_preprocessing.raw import data_processing_methods as dpm

"""
NSG data
"""
# NSG post processes data location
file_val = 'validation_data.xlsx'
file = 'data_and_preprocessing/processed/NSG_processed_data_14_inputs.xlsx'
# file = 'data_and_preprocessing/processed/NSG_processed_data.xlsx'

# Training df
X_df = pd.read_excel(file, sheet_name='X_stand')
y_df = pd.read_excel(file, sheet_name='y')
y_raw_df = pd.read_excel(file, sheet_name='y_raw')
t_df = pd.read_excel(file, sheet_name='timelags')

# validation df
val_df = pd.read_excel(file_val)
date_timev = val_df.date_time.values
val_mu = val_df.gp_pred.values

# get data on the region of interest (valitation data)
first_date = val_df.date_time[0]
last_date = val_df.date_time[len(val_df.date_time)-1]

first = y_df[y_df['Time stamp'] == first_date].index[0]
last  = y_df[y_df['Time stamp'] == last_date].index[0]

X_df = X_df.iloc[first:last, :]
y_df = y_df.iloc[first:last, :]
y_raw_df = y_raw_df.iloc[first:last, :]

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
end_train_X = N - int(len(y0)*0.12)
end_train_y = N - int(len(val_mu)*0.12)

X_train_norm = X[0:end_train_X]
date_train = date_time[0:end_train_X]
N_train = len(X_train_norm)
y_train_unorm = val_mu[0:end_train_y]

X_test = X[0:N]
date_time = date_time[0:N]
y_raw = y_raw[0:N]
y_rect = y0[0:N]

"""
PROCESS
"""
import torch
from sklearn.preprocessing import StandardScaler as ss

y_train = y_train_unorm.reshape(-1,1)
scaler = ss()
scaler.fit(y_train)
y_norm = scaler.transform(y_train)

# Convert data to torch tensors
floating_point = torch.float64
X_train = torch.tensor(X_train_norm, dtype=floating_point)
y_train = torch.tensor(y_norm, dtype=floating_point).squeeze()
X_test = torch.tensor(X_test, dtype=floating_point)


"""
Sparse GP
"""
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import InducingPointKernel, ScaleKernel, RBFKernel as RBF

# Convert data to torch tensors to input inducing points
inducing_points = X_train[::10, :]

likelihood = GaussianLikelihood()

se = ScaleKernel(RBF(ard_num_dims=X_train.shape[-1]))

covar_module = InducingPointKernel(se,
                                   inducing_points=inducing_points,
                                   likelihood=likelihood)
lss = [1.83, 0.318, 603, 0.651, 5.87e+04, 3.0, 1.17, 1.2e+03, 4.63, 0.25, 1.19e+04, 52.2, 663, 17.3]
start_time = time.time()

class SparseGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, noise_var):
        super(SparseGP, self).__init__(train_x, train_y, likelihood)
        likelihood.noise = noise_var
        self.mean_module = ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    

# Ensure data is of shape [N, D]
print(X_train.shape)  # Should be [N_train, D]
print(y_train.shape)  # Should be [N_train]
print(X_test.shape)   # Should be [N_test, D]

gp = SparseGP(X_train, y_train, likelihood, covar_module, 0.06)
gp.covar_module.base_kernel.base_kernel.lengthscale = torch.tensor(lss)

# Train model
gp.train()
gp.likelihood.train()

optimizer = torch.optim.Adam(gp.parameters(), lr=0.01)
mll = ExactMarginalLogLikelihood(likelihood, gp)

training_iterations = 100
for count in range(training_iterations):
    optimizer.zero_grad()
    output = gp(X_train)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()

# Predictions
gp.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(gp(X_test))

    # Unormalise predictions
    pred_mean = observed_pred.mean
    mu = pred_mean
    stds = observed_pred.stddev

print('mu-shape:  ', np.shape(mu))
print('std-shape: ', np.shape(stds))
"""
PLOT
"""

def get_z_indices(x, inducing_inputs):
    indices = np.zeros(len(inducing_inputs), dtype=int)
    for i, induced in enumerate(inducing_inputs):
        closest_idx = np.argmin(np.linalg.norm(x - induced, axis=1))
        indices[i] = closest_idx
        if i < 5:
            print(f"i: {i}, closest-indx: {closest_idx}")
    return indices

_z_induced = gp.covar_module.inducing_points.detach().numpy()
_z_indices = get_z_indices(X_train.numpy(), _z_induced)

# common = set(X_train.numpy()).intersection(set(_z_induced))
print(np.all(inducing_points.numpy()==_z_induced))
print('Size X-induced: ', np.shape(_z_induced))
# print('indices-induced: ', np.shape(_z_indices))
# print(_z_indices)
# print('caca: ', np.argmin(np.abs(X_train.numpy() - _z_induced[6])))


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
                mu + 3*stds, mu - 3*stds,
                alpha=0.5, color='lightcoral',
                label='3$\\sigma$')
ax.plot(date_time, y_raw, color='grey', label='Raw')
ax.plot(date_time, y_rect, color='blue', label='Filtered')
ax.plot(date_timev, val_mu, color='green', label='Val')
ax.plot(date_time, mu, color='red', label='GP')
plt.axvline(date_time[end_train-1], linestyle='--', linewidth=3,
            color='black')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

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

ax.vlines(
    # Sparse clean data
    x=date_time[_z_indices],
    ymin=-0.5,
    ymax=y_train.max().item(),
    alpha=0.4,
    linewidth=1.5,
    label="z*",
    color='orange'
)
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)
plt.show()