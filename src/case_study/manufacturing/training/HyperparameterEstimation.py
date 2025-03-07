import torch
import copy
import gpytorch
import pandas as pd
import numpy as np
import random
import yaml
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler as ss
from matplotlib import pyplot as plt
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import InducingPointKernel, ScaleKernel
from gpytorch.kernels import RBFKernel as RBF, RQKernel as RQ
from gpytorch.kernels import LinearKernel as Lin, PeriodicKernel as Per

"""
NSG data
"""

file = 'validation_data_main.xlsx'

# Training df
X_df = pd.read_excel(file, sheet_name='X_stand')
y_df = pd.read_excel(file, sheet_name='y_nonstand')
t_df = pd.read_excel(file, sheet_name='timelags')

# drop Tewwl position
X_df.drop(columns=['9282 Tweel Position'], inplace=True)
t_df.drop(columns=['9282 Tweel Position'], inplace=True)

"""---------------------------------------------------------------------------
    CREATE LAGGED FEATURES
"""

def align_inputs(x_df, y_df, t_series):
    # ! Always close/Deep copy
    xdeep = x_df.copy()
    ydeep = y_df.copy()
    max_lag = int(max(t_series))

    # X
    for name, lag in t_series.items():
        xdeep[name] = xdeep[name].shift(int(lag))
    xdeep.dropna(inplace=True)

    # y and date-time
    ydeep = ydeep.iloc[max_lag:].reset_index(drop=True)

    return xdeep.reset_index(drop=True), ydeep
 
X_df, y_df = align_inputs(X_df, y_df, t_df.iloc[0,:])

"""---------------------------------------------------------------------------
    STANDARDISE TRAINING & TEST DATA
"""

X = X_df.values
y_nonstand, date_time = y_df.gp_pred.values, y_df.date_time.values

N, D = np.shape(X)
test_perc = 0.12
end_train = N - int(len(y_nonstand)*test_perc)

X_train_np = X[0:end_train]
date_train = date_time[0:end_train]
N_train = len(X_train_np)
y_train_nonstand = y_nonstand[0:end_train]

X_test = X[0:N]
date_time = date_time[0:N]

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

"""
READ ESTIMATED MODEL
"""

# Read best hyperparameters and initialisation values from the yml file
with open('config_RBF_plus_RQ_2_step40.yaml', 'r') as f:
    config = yaml.safe_load(f)

test_perc = config['test_percentage']
init_noise_var = config['WN']['var']['optimal']

X = X_df.values
y_nonstand, date_time = y_df.gp_pred.values, y_df.date_time.values

N, D = np.shape(X)
end_train = N - int(len(y_nonstand)*test_perc)

# make sure X and y are the same size
assert N - int(len(X)*test_perc) == N - int(len(y_nonstand)*test_perc), 'Size of X and y are not the same'

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

# Train and test data
step = config['step']
inducing_points = X_train[::step, :].clone()

# ! Ensure data is of shape [N, D]
print(X_train.shape)            # Should be [N_train, D]
print(inducing_points.shape)    # Should be [N_train/step, D]
print(X_test.shape)             # Should be [N_test, D]
print(y_train.shape)            # Should be [N_train]

assert inducing_points[2, 0] == X_train[step+step, 0], 'Init induced not the same as in X_train'

# Model
likelihood = GaussianLikelihood()

# k = ScaleKernel(RBF(ard_num_dims=D) + RQ(ard_num_dims=D))
k = ScaleKernel(RQ(ard_num_dims=D) * Per(ard_num_dims=D))
covar_module = InducingPointKernel(k,
                                   inducing_points=inducing_points,
                                   likelihood=likelihood)

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

# state_dict = torch.load('gp_state_RBF_plus_RQ_opt_step40.pth', weights_only=False)
gp = SparseGP(X_train, y_train, likelihood, covar_module, init_noise_var)

# gp.load_state_dict(state_dict)


"""
    TRAINING
"""
class Train():
    def __init__(self, gp0):
        super(Train, self).__init__()
        self.gp0 = gp0
        self.os0 = self.gp0.covar_module.base_kernel.outputscale.item()

        # Function to initialise kernel parameters
    def init_hyper(self, gp: ExactGP,
                   os_limits=None,
                   ls_se_limits=None, ls_rq_limits=None,
                   alpha_limits=None, plength_limits=None) -> ExactGP:
        kernel = gp.covar_module
        kernel.base_kernel.outputscale = torch.tensor(self.os0)

        def set_params(k):
            if not isinstance(k, Lin):
                if isinstance(k, RBF):
                    k.lengthscale = np.random.uniform(low=ls_se_limits[0],
                                                      high=ls_se_limits[1])
                if isinstance(k, RQ):
                    k.alpha = torch.tensor(np.random.uniform(low=alpha_limits[0],
                                                             high=alpha_limits[1]))
                    k.lengthscale = np.random.uniform(low=ls_rq_limits[0],
                                                      high=ls_rq_limits[1])
                if isinstance(k, Per):
                    k.period_length = np.random.uniform(low=plength_limits[0],
                                                        high=plength_limits[1])
            return k

        # check number of operand kernels
        if hasattr(kernel.base_kernel.base_kernel, 'kernels'):
            kernel.base_kernel.outputscale = np.random.uniform(low=os_limits[0],
                                                               high=os_limits[1])
            N_kernels = len(kernel.base_kernel.base_kernel.kernels)
            for i in range(N_kernels):
                kernel.base_kernel.base_kernel.kernels[i] = set_params(kernel.base_kernel.base_kernel.kernels[i])
        elif hasattr(kernel.base_kernel, 'base_kernel'):
            kernel.base_kernel.base_kernel = set_params(kernel.base_kernel.base_kernel)
        else:
            kernel.outputscale = np.random.uniform(low=os_limits[0],
                                                   high=os_limits[1])
            kernel.base_kernel = set_params(kernel.base_kernel)

        gp.covar_module = kernel
        return gp
    
    def trainGP(self, gp: ExactGP) -> ExactGP:
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
        
        return gp

    def grid_search(self, N_sim,
                    os_limits=None,
                    ls_se_limits=None, ls_rq_limits=None,
                    alpha_limits=None, plength_limits=None):
        gp = self.gp0
        mse_list = np.zeros(shape=N_sim)
        best_mse = float('inf')
        best_gp = None
        random_start = True

        for i in range(N_sim):
            print(f'Hyperparameter simulation: {i}/{N_sim}')

            if random_start:
                gp = self.init_hyper(gp,
                                     os_limits,
                                     ls_se_limits,
                                     ls_rq_limits,
                                     alpha_limits,
                                     plength_limits)

            # Train model
            gp = self.trainGP(gp)

            # Predictions
            gp.eval()
            likelihood.eval()
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = likelihood(gp(X_test))

                # Unormalise predictions
                pred_mean = observed_pred.mean
                mu = scaler.inverse_transform(pred_mean.unsqueeze(1))[:,0]
                mse = mean_squared_error(mu, y_nonstand)
                mse_list[i] = mse
                print('Error: ', mse, '\n')

            # Check error and update best GP if necessary
            if mse < best_mse:
                best_mse = mse
                best_gp = copy.deepcopy(gp)

            # Adjust random start based on error improvement
            random_start = mse >= best_mse

        return best_gp, mse_list
            

"""
    FINE TUNE
"""

class FineTune():
    def __init__(self, gp0, var):
        super(FineTune, self).__init__()
        self.gp0 = gp0
        self.var = var
        self.nv0 = self.gp0.likelihood.noise.item()
        self.random_start = True

    def get_init_hyper(self):
        kernel = self.gp0.covar_module
        self.os0 = kernel.base_kernel.outputscale.item()

        def set_params(k):
            if not isinstance(k, Lin):
                if isinstance(k, RBF):
                    self.ls_se0 = k.lengthscale
                if isinstance(k, RQ):
                    self.alpha0 = k.alpha
                    self.ls_rq0 = k.lengthscale
                if isinstance(k, Per):
                    self.plength0 = k.period_length
            return k

        # check number of operand kernels
        if hasattr(kernel.base_kernel.base_kernel, 'kernels'):
            N_kernels = len(kernel.base_kernel.base_kernel.kernels)
            for i in range(N_kernels):
                kernel.base_kernel.base_kernel.kernels[i] = set_params(kernel.base_kernel.base_kernel.kernels[i])
        elif hasattr(kernel.base_kernel, 'base_kernel'):
            kernel.base_kernel.base_kernel = set_params(kernel.base_kernel.base_kernel)
        else:
            kernel.base_kernel = set_params(kernel.base_kernel)

        # Function to initialise kernel parameters
    def init_hyper(self, gp: ExactGP) -> ExactGP:
        kernel = gp.covar_module
        kernel.base_kernel.outputscale = torch.tensor(self.os0)

        def set_params(k):
            if not isinstance(k, Lin):
                if isinstance(k, RBF):
                    k.lengthscale = torch.tensor([random.gauss(self.ls_se0[0,d],
                                                               sigma=self.var) for d in range(D)])
                if isinstance(k, RQ):
                    k.alpha = torch.tensor(random.gauss(mu=0.16,
                                                        sigma=1e-3))
                    k.lengthscale = torch.tensor([random.gauss(self.ls_rq0[0,d],
                                                               sigma=0.24) for d in range(D)])
                if isinstance(k, Per):
                    k.period_length = torch.tensor(random.gauss(mu=self.plength0,
                                                   sigma=self.var))
            return k

        # check number of operand kernels
        if hasattr(kernel.base_kernel.base_kernel, 'kernels'):
            N_kernels = len(kernel.base_kernel.base_kernel.kernels)
            for i in range(N_kernels):
                kernel.base_kernel.base_kernel.kernels[i] = set_params(kernel.base_kernel.base_kernel.kernels[i])
        elif hasattr(kernel.base_kernel, 'base_kernel'):
            kernel.base_kernel.base_kernel = set_params(kernel.base_kernel.base_kernel)
        else:
            kernel.base_kernel = set_params(kernel.base_kernel)

        gp.covar_module = kernel
        return gp

    def tune(self, N_sim):
        mse_list = np.zeros(shape=N_sim)
        mse_list[0] = float('inf')
        self.get_init_hyper()

        for i in range(N_sim):
            gp = SparseGP(X_train, y_train, likelihood, covar_module,
                          torch.tensor(self.nv0))
            gp = self.init_hyper(gp)

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
                mu = scaler.inverse_transform(pred_mean.unsqueeze(1))[:,0]
                mse_list[i] = mean_squared_error(mu, y_nonstand)

            print(f'Hyper simulation: {i}/{N_sim}, Error: {round(mse_list[i], 5)}')

            # check error
            if i > 1:
                if mse_list[i] < 0.00145:
                    break
        return gp

"""
    TEST classes
"""
g = Train(gp)
# gp = g.grid_search(N_sim=10, os_limits=[0.5, 10],
#                    ls_se_limits=[10, 100], ls_rq_limits=[0.5, 90],
#                    alpha_limits=[0.01, 2])[0]
gp, mse_list = g.grid_search(N_sim=20, os_limits=[0.5, 10],
                   ls_rq_limits=[0.5, 90],
                   alpha_limits=[0.01, 2],
                   plength_limits=[0.1, 6])
# ft = FineTune(gp, 0.8)
# gp = ft.simulate(N_sim=250)

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

print('MSE (train-test): ',mean_squared_error(mu, y_nonstand))
print('MSE (test):       ',mean_squared_error(mu[end_train:-1],
                                              y_nonstand[end_train:-1]))

"""--------------------------------------------------------------------------
PLOT
"""
plt.figure()
plt.plot(mse_list)
plt.xlabel('iteration')
plt.xlabel('MSE')

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