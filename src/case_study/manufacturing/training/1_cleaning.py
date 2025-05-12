import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
from pathlib import Path
# DPSGP
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import InducingPointKernel, ScaleKernel, RBFKernel as RBF
from models.dpsgp_torch_ama import DirichletProcessSparseGaussianProcess as DPSGP

"""
NSG data

Do not adjust data for timelags.
"""

# NSG post processes data location
ROOT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT_PATH / "data" / "processed" / "Training_data_partitions"
file = PROCESSED_PATH / 'data1.xlsx'

# Training df
X_df = pd.read_excel(file, sheet_name='X_stand')
y_df = pd.read_excel(file, sheet_name='y_nonstand')
t_df = pd.read_excel(file, sheet_name='timelags')

# Pre-Process training data
X_train = X_df.values
y_filtered = y_df.y_filtered.values
N, D = np.shape(X_train)

# Replace zero values with interpolation
zeros = y_df.loc[y_df['y_raw'] <= 1e-1]
y_df.loc[zeros.index, 'y_raw'] = None
y_df.interpolate(inplace=True)

y_train = y_df['y_raw'].values
date_time = y_df['date_time'].values

N_train = len(X_train)

#-----------------------------------------------------------------------------
# PLOT TRAINING DATA
#-----------------------------------------------------------------------------

fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
fig.autofmt_xdate()

ax.plot(date_time, y_train, color='grey', label='Raw')
ax.plot(date_time, y_filtered, color='blue', label='Filtered')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

"""
DPSGP cleaning

GPytotch is very sensitive to the initial hyperparameters.
I first used the DPGP sklearn version to estimate the initial
lengthscales for this script.
"""

# Convert data to torch tensors to input inducing points
floating_point = torch.float64
X_tensor = torch.tensor(X_train, dtype=floating_point)
inducing_points = X_tensor[::8, :]

likelihood = GaussianLikelihood()

se = ScaleKernel(RBF(ard_num_dims=X_train.shape[-1]))

covar_module = InducingPointKernel(se,
                                   inducing_points=inducing_points,
                                   likelihood=likelihood)

lss = [1.83, 0.8, 603, 0.3, 5.87e+04, 3.0, 2.17, 1.2e+03, 4.63, 1, 1.19e+04, 52.2, 663, 17.3]
start_time = time.time()
sgp = DPSGP(X_train, y_train, init_K=7,
            gp_model='Sparse',
            prior_mean=ConstantMean(), kernel=covar_module,
            lengthscale=lss,
            N_iter=15,
            noise_var = 0.06,
            floating_point=floating_point,
            normalise_y=True,
            DP_max_iter=390,
            # window_size=150,
            threshold_factor=1.5,
            print_conv=True, plot_conv=True, plot_sol=True)
sgp.train()
mu, stds = sgp.predict(X_train)
comp_time = time.time() - start_time

print(f'\nDPSGP cleaning time: {comp_time:.2f} seconds')

print('\n Furnace parameters relevance')
d = {'Features': X_df.columns, 'Importance': sgp.lengthscale[0]}
fidf = pd.DataFrame.from_dict(d)
fidf = fidf.sort_values(by='Importance')
print(fidf.head(14))

# get inducing points indices
_z_indices = sgp._z_indices

print('N-train: \t', N_train)
print('N-induced: ', len(_z_indices))

# save predictions to use it in another scipt as the `true` fault_density
cleaned_indices = sgp.indices[0]

print('N-clean: ', len(cleaned_indices))

dx = {}
for d, name in enumerate(X_df.columns):
    dx[name] = X_train[:, d]

d_clean = {"date_time": date_time[cleaned_indices],
           "y_raw": y_train[sgp.indices[0]]}
d = {"date_time": date_time, "gp_pred": mu}

X_df = pd.DataFrame(dx)
y_df = pd.DataFrame(d)

# Define an Excel writer object and the target file
# writer = pd.ExcelWriter("validation_data_main.xlsx")

# # Save to spreadsheet
# X_df.to_excel(writer, sheet_name='X_stand', index=False)
# y_df.to_excel(writer, sheet_name='y_nonstand', index=False)
# t_df.to_excel(writer, sheet_name='timelags', index=False)
# writer._save()

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
ax.plot(date_time, y_train, color='grey', label='Raw')
ax.plot(date_time, y_filtered, color='blue', label='Filtered')
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