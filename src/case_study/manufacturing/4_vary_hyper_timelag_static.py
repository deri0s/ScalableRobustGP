import torch
import time
import gpytorch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler as ss
from matplotlib import pyplot as plt
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import InducingPointKernel, ScaleKernel, RBFKernel as RBF

"""
NSG data
"""
# NSG post processes data location
file = 'validation_data.xlsx'

# Training df
X_df  = pd.read_excel(file, sheet_name='X_stand')
y_df  = pd.read_excel(file, sheet_name='y_nonstand')
t0_df = pd.read_excel('data_and_preprocessing/processed/timelags_RandomForest.xlsx')

# Pre-Process training data
N, D = np.shape(X_df.values)

# Create tag inputs
X = np.zeros([N, D])

"""---------------------------------------------------------------------------
    CREATE LAGGED FEATURES
"""

def align_inputs(x_df, y_df, t_series):
    max_lag = max(t_series)
    # X
    for name, lag in t_series.items():
        x_df[name] = x_df[name].shift(lag)

    x_df.dropna(inplace=True)
    # y and date-time
    y_df = y_df.iloc[max_lag:].reset_index(drop=True)

    return x_df.reset_index(drop=True), y_df

# create lagged features
# get the timelags estimated by the RF when max-lag = 3 days
X_df, y_df = align_inputs(X_df, y_df, t0_df.iloc[1,:])

"""---------------------------------------------------------------------------
    STANDARDISE TRAINING & TEST DATA
"""

X = X_df.values
y_nonstand, date_time = y_df.gp_pred.values, y_df.date_time.values

N, D = np.shape(X)
end_train = N - int(len(y_nonstand)*0.12)

# make sure X and y are the same size
assert N - int(len(X)*0.12) == N - int(len(y_nonstand)*0.12), 'Size of X and y are not the same'

X_train_np = X[0:end_train]
date_train = date_time[0:end_train]
N_train = len(X_train_np)
y_train_nonstand = y_nonstand[0:end_train]

X_test = X[0:N]
date_time = date_time[0:N]

assert len(X_train_np) == len(y_train_nonstand), 'X-train and y-train length are not the same'
assert len(X_test) == len(y_nonstand), 'X-test and y-test length are not the same'

# Standardise outputs
y_train = y_train_nonstand.reshape(-1,1)
scaler = ss()
scaler.fit(y_train)
y_norm_np = scaler.transform(y_train)

# Convert data to torch tensors
floating_point = torch.float64
X_train = torch.tensor(X_train_np, dtype=floating_point)
y_train = torch.tensor(y_norm_np, dtype=floating_point).squeeze()
X_test = torch.tensor(X_test, dtype=floating_point)

"""----------------------------------------------------------------------------
Sparse GP
"""

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
    
# ! Always clone
step = 40
inducing_points = X_train[::step, :].clone()

# ! Ensure data is of shape [N, D]
# print(X_train.shape)            # Should be [N_train, D]
# print(inducing_points.shape)    # Should be [N_train/step, D]
# print(X_test.shape)             # Should be [N_test, D]
# print(y_train.shape)            # Should be [N_train]

assert inducing_points[2, 0] == X_train[step+step, 0], 'Init induced not the same as in X_train'

# Model
likelihood = GaussianLikelihood()

se = ScaleKernel(RBF(ard_num_dims=X_train.shape[-1]))

covar_module = InducingPointKernel(se,
                                   inducing_points=inducing_points,
                                   likelihood=likelihood)

N_sim = 50
init_ls = []
init_nv = []

os_list = []
ls_list = []
nv_list = []
mse_list = []
random = True

for i in range(N_sim):
    print(f'Hyperparameter simulation: {i}/{N_sim}')

    if random:
        ls = np.random.uniform(low=0.1, high=100, size=D)
        nv = np.random.uniform(low=0.01, high=0.1)
        # save initial hyperparameters
        init_ls.append(ls)
        init_nv.append(nv)
    else:
        # save initial hyperparameters
        init_ls.append(ls.squeeze(0).detach().numpy())
        init_nv.append(nv)

    # GP object
    gp = SparseGP(X_train, y_train, likelihood, covar_module, nv)
    gp.covar_module.base_kernel.base_kernel.outputscale = 1
    gp.covar_module.base_kernel.base_kernel.lengthscale = ls

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

    # get the estimated hyperparameters
    os = gp.covar_module.base_kernel.outputscale.item()
    ls = gp.covar_module.base_kernel.base_kernel.lengthscale
    nv = likelihood.noise.item()

    # Predictions
    gp.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(gp(X_test))

        # Unormalise predictions
        pred_mean = observed_pred.mean
        mu = scaler.inverse_transform(pred_mean.unsqueeze(1))[:,0]
        mse = mean_squared_error(mu, y_nonstand)

    # collect results
    os_list.append(os)
    ls_list.append(ls.squeeze(0).detach().numpy())
    nv_list.append(nv)
    mse_list.append(mse)

    # check error
    if i > 1:
        if mse_list[i] < mse_list[i-1]:
            random = False
        else:
            random = True

"""--------------------------------------------------------------------------
BEST HYPERPARAMETER CONFIGURATION
"""

d = {'step': step,
     'init_os': np.ones(N_sim), 'init_ls': init_ls, 'init_nv': init_nv,
     'outputscale': os_list,
     'lengthscale': ls_list,
     'noise_var': nv_list,
     'mse': mse_list}

df_sim = pd.DataFrame(d)

# save into spreadsheet
df_best5 = df_sim.sort_values(by='mse').iloc[0:5, :]
df_best5.to_excel('4_RF_timelags_best_hyper.xlsx')

print('lowest errors \n', df_sim.mse.sort_values()[0:5])

indx = df_sim[df_sim.mse == df_sim.mse.min()].index

init_opt_ls = df_sim.init_ls[indx].values[0]
init_opt_nv = df_sim.init_nv[indx].values[0]
opt_ls = df_sim.lengthscale[indx].values[0]
opt_nv = df_sim.noise_var[indx].values

print(f'\ninit_ls: \n {init_opt_ls}\ninit_nv: {init_opt_nv}')
print('\nmse: ', df_sim.mse[indx].values)

# GP object
gp = SparseGP(X_train, y_train, likelihood, covar_module, init_opt_nv)
gp.covar_module.base_kernel.base_kernel.outputscale = 1
gp.covar_module.base_kernel.base_kernel.lengthscale = init_opt_ls

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

# *Induced points
init_z_indices = np.arange(0, len(X_train.numpy()), step)

# Make sure the _z (induced inputs) are a subset of the X_train dataset
_z = gp.covar_module.inducing_points.detach()

_z_indices = []
for z in _z:
    distances = torch.norm(X_train - z, dim=1)
    closest_index = torch.argmin(distances).item()
    _z_indices.append(closest_index)

# check the z0 and z* are not the same
assert ~np.all(list(init_z_indices == _z_indices)), 'induced not trained'

# Predictions
gp.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(gp(X_test))

# Unormalise predictions
pred_mean = observed_pred.mean
mu = scaler.inverse_transform(pred_mean.unsqueeze(1))[:,0]
stds = scaler.inverse_transform(observed_pred.stddev.unsqueeze(1))[:,0]
lower_stand, upper_stand = observed_pred.confidence_region()
lower = scaler.inverse_transform(lower_stand.unsqueeze(1))[:,0]
upper = scaler.inverse_transform(upper_stand.unsqueeze(1))[:,0]

print('MSE: ', mean_squared_error(mu, y_nonstand))

"""--------------------------------------------------------------------------
PLOT
"""

plt.figure()
plt.plot(mse_list)
plt.xlabel('iteration')
plt.xlabel('MSE')

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------
fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
fig.autofmt_xdate()

plt.fill_between(date_time, lower, upper,
                alpha=0.5, color='lightcoral',
                label='2$\\sigma$')
ax.plot(date_time, y_nonstand, '*', color='green', label='Val')
ax.plot(date_time, mu, color='red', label='GP')
plt.axvline(date_time[end_train-1], linestyle='--', linewidth=3,
        color='black')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

ax.vlines(
    x=date_time[::step],
    ymin=-2*stds.min(),
    ymax=y_train.max().item(),
    alpha=0.3,
    linewidth=1.5,
    ls='--',
    label="z0",
    color='grey'
)

# Induced points
ax.vlines(
    x=date_time[_z_indices],
    ymin=-2*stds.min(),
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