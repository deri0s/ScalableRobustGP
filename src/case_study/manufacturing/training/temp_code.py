import pandas as pd
import numpy as np
import torch
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import RQKernel as RQ
from gpytorch.kernels import InducingPointKernel, ScaleKernel
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from sklearn.preprocessing import StandardScaler as ss
from matplotlib import pyplot as plt
from pathlib import Path

"""
NSG data

Do not adjust data for timelags.
"""

# NSG post processes data location
ROOT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT_PATH / "data" / "processed"
file = PROCESSED_PATH / 'NSG_processed_data.xlsx'

# Training df
X_df = pd.read_excel(file, sheet_name='X_stand')
y_df = pd.read_excel(file, sheet_name='y')
y_raw_df = pd.read_excel(file, sheet_name='y_raw')
t_df = pd.read_excel(file, sheet_name='timelags')

# drop Tewwl position
X_df.drop(columns=['9282 Tweel Position'], inplace=True)
t_df.drop(columns=['9282 Tweel Position'], inplace=True)

# Pre-Process training data
X    = X_df.values
y0   = y_df.furnace_faults.values
N, D = np.shape(X)

# Replace zero values with interpolation
zeros = y_raw_df.loc[y_raw_df['raw_furnace_faults'] <= 1e-1]
y_raw_df.loc[zeros.index, 'raw_furnace_faults'] = None
y_raw_df.interpolate(inplace=True)

y_raw = y_raw_df['raw_furnace_faults'].values
date_time = y_df['Time stamp'].values

# Train and test data
# Clean area:
# Start: '2020-07-25-01', Ends: '2020-09-01-05'
start_train = y_df[y_df['Time stamp'] == '2020-07-25-01'].index[0]
end_train = y_df[y_df['Time stamp'] == '2020-09-01-05'].index[0]

X_train, y_train = X[start_train:end_train], y0[start_train:end_train]
N_train = len(X_train)

print('X-train: ', y_df[y_df['Time stamp'] == '2020-09-01-05'].index[0])

end_test = 7010
X_test = X[0:end_test]
date_time = date_time[0:end_test]
y_raw = y_raw[0:end_test]
y_rect= y0[0: end_test] 

# Standardise outputs
y_train = y_train.reshape(-1,1)
scaler = ss()
scaler.fit(y_train)
y_norm_np = scaler.transform(y_train)

# Convert data to torch tensors
floating_point = torch.float64
X_train = torch.tensor(X_train, dtype=floating_point)
y_train = torch.tensor(y_norm_np, dtype=floating_point).squeeze()
X_test = torch.tensor(X_test, dtype=floating_point)

#-----------------------------------------------------------------------------
# PLOT TRAINING DATA
#-----------------------------------------------------------------------------

fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
fig.autofmt_xdate()

ax.plot(date_time, y_raw, color='grey', label='Raw')
ax.plot(date_time, y_rect, color='blue', label='Filtered')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

"""
DPSGP cleaning
"""

state_dict = torch.load('expert_main0.pth',
                        weights_only=False)

# Access the inducing points
inducing_points = state_dict['covar_module.inducing_points']

# # Model
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
    
likelihood = GaussianLikelihood()

k = ScaleKernel(RQ(ard_num_dims=D))
covar_module = InducingPointKernel(k,
                                   inducing_points=inducing_points,
                                   likelihood=likelihood)
gp0 = SparseGP(X_train, y_train, likelihood, covar_module, 0.028)
gp0.load_state_dict(state_dict)

# Predictions
gp0.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(gp0(X_test))

    # Unormalise predictions
    pred_mean = observed_pred.mean
    mu = scaler.inverse_transform(pred_mean.unsqueeze(1))[:,0]
    stds = scaler.inverse_transform(observed_pred.stddev.unsqueeze(1))[:,0]
    lower_stand, upper_stand = observed_pred.confidence_region()
    lower = scaler.inverse_transform(lower_stand.unsqueeze(1))[:,0]
    upper = scaler.inverse_transform(upper_stand.unsqueeze(1))[:,0]

# Make sure the _z (induced inputs) are a subset of the X_train dataset
_z = gp0.covar_module.inducing_points.detach()

_z_indices = []
for z in _z:
    distances = torch.norm(X_train - z, dim=1)
    closest_index = torch.argmin(distances).item()
    _z_indices.append(closest_index)

print('N-induced: ', len(_z_indices))

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
ax.plot(date_time, mu, color="red", linewidth = 2.5, label="Cleaned")

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

ax.vlines(
    # Sparse clean data
    x=date_time[_z_indices],
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