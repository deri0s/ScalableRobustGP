import torch
import copy
import gpytorch
import pandas as pd
import numpy as np
from numpy.random import uniform
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
# with open('expert_main0.yaml', 'r') as f:
#     config = yaml.safe_load(f)

test_perc = 0.12
# init_noise_var = config['WN']['var']['optimal']
init_noise_var = 0.02825

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
# step = config['step']
# step = config['step']
# inducing_points = X_train[::step, :].clone()

# # ! Ensure data is of shape [N, D]
# print(X_train.shape)            # Should be [N_train, D]
# print(inducing_points.shape)    # Should be [N_train/step, D]
# print(X_test.shape)             # Should be [N_test, D]
# print(y_train.shape)            # Should be [N_train]

# assert inducing_points[2, 0] == X_train[step+step, 0], 'Init induced not the same as in X_train'

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

"""
    FINE TUNE
"""

class GPTraining():
    def __init__(self, gp0, y_test):
        super(GPTraining, self).__init__()
        self.y_test = y_test
        self.gp0 = gp0
        self.N, self.D = np.shape(self.gp0.train_inputs[0])
        self.kernel = self.gp0.covar_module
        self.nv0 = self.gp0.likelihood.noise.item()
        self.random_start = True

        # covariance functions building blocks
        self.base_kernels = {
            'RBF': lambda: RBF(ard_num_dims=D, dtype=floating_point),
            'RQ': lambda: RQ(ard_num_dims=D, dtype=floating_point)
            # 'Lin': lambda: Lin(ard_num_dims=D, dtype=floating_point),
            # 'Per': lambda: Per(ard_num_dims=D, dtype=floating_point)
            }

        # Initialise class attributes tha will be updated in:
        # The grid_search method
        self.os_limits = None
        self.se_ls_limits = None
        self.rq_ls_limits = None
        self.alpha_limits = None
        self.plength_limits = None

        # The tune method
        self.os_std = None
        self.se_ls_std = None
        self.alpha_std = None
        self.rq_ls_std = None
        self.plength_std = None

    def init_kernel(self, set_params, k):
        kernel = self.gp0.covar_module
        # check number of operand kernels
        if hasattr(self.kernel.base_kernel.base_kernel, 'kernels'):
            N_kernels = len(self.kernel.base_kernel.base_kernel.kernels)
            for i in range(N_kernels):
                set_params(kernel.base_kernel.base_kernel.kernels[i])
        elif hasattr(self.kernel.base_kernel, 'base_kernel'):
            set_params(kernel.base_kernel.base_kernel)
        else:
            set_params(kernel.base_kernel)

    def from_uniform(self, k):
        if isinstance(self.kernel.base_kernel, ScaleKernel):
            os = uniform(low=self.os_limits[0],
                         high=self.os_limits[1])
            self.kernel.base_kernel.outputscale = os

        if not isinstance(k, Lin):
            if isinstance(k, RBF):
                k.lengthscale = uniform(low=self.se_ls_limits[0],
                                        high=self.se_ls_limits[1],
                                        size=self.D)
            if isinstance(k, RQ):
                k.alpha = torch.tensor(uniform(low=self.alpha_limits[0],
                                               high=self.alpha_limits[1]))
                k.lengthscale = uniform(low=self.rq_ls_limits[0],
                                        high=self.rq_ls_limits[1],
                                        size=self.D)
            if isinstance(k, Per):
                k.period_length = uniform(low=self.plength_limits[0],
                                          high=self.plength_limits[1],
                                          size=self.D)
        return k

    def set_gauss_centres(self, k):
        if isinstance(self.kernel.base_kernel, ScaleKernel):
            self.os0 = self.kernel.base_kernel.outputscale.item()

        if not isinstance(k, Lin):
            if isinstance(k, RBF):
                self.ls_se0 = k.lengthscale
            if isinstance(k, RQ):
                self.alpha0 = k.alpha.clone().detach().item()
                self.ls_rq0 = k.lengthscale
            if isinstance(k, Per):
                self.plength0 = k.period_length

    def from_gauss(self, k):
        if isinstance(self.kernel.base_kernel, ScaleKernel):
            os = random.gauss(self.os0, sigma=self.os_std)
            os = 1 if os <= 1e-6 else os
            self.kernel.base_kernel.outputscale = os

        if not isinstance(k, Lin):
            if isinstance(k, RBF):
                ls_se_array = np.zeros(shape=D)
                for d in range(D):
                    ls = random.gauss(self.ls_se0[0,d],
                                      sigma=self.se_ls_std)
                    ls_se_array[d] = 1e-3 if ls <= 1e-6 else ls
                k.lengthscale = ls_se_array
                
            if isinstance(k, RQ):
                k.alpha = torch.tensor(random.gauss(mu=self.alpha0,
                                                    sigma=self.alpha_std))
                ls_rq_array = np.zeros(shape=D)
                for d in range(D):
                    ls = random.gauss(self.ls_rq0[0,d],
                                      sigma=self.rq_ls_std)
                    ls_rq_array[d] = 1e-3 if ls <= 1e-6 else ls
                k.lengthscale = ls_rq_array

            if isinstance(k, Per):
                pl_array = np.zeros(shape=D)
                for d in range(D):
                    pl = random.gauss(self.plength0[0,d],
                                      sigma=self.plength_std)
                    pl_array[d] = 1e-3 if ls <= 1e-6 else pl
                k.period_length = pl_array

    # Function to train and evaluate the model
    def train_and_evaluate(self, gp: ExactGP):
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

            # Unnormalize predictions
            pred_mean = observed_pred.mean
            mu = scaler.inverse_transform(pred_mean.unsqueeze(1))[:, 0]

        test_error = mean_squared_error(mu, self.y_test)

        return test_error
    
    def combine_kernels(operands_1, operation: str, operands_2) -> dict:
        new = {}
        combined = []

        for base_name, base_kernel in operands_1.items():
            for name, kernel in operands_2.items():
                if name in combined:
                    pass
                else:
                    new[f"{base_name} {operation} {name}"] = (
                        lambda b=base_kernel, k=kernel: k() + b())
            combined.append(base_name)
        return new

    def get_best_kernel(self, kernels: dict) -> tuple:
        best_kernel = None
        best_name = None
        best_error = float('inf') # Reset per-call for level comparison
        mse = float('inf')

        for name, kernel in kernels.items():
            # update kernel
            basek = InducingPointKernel(ScaleKernel(kernel()),
                                inducing_points=inducing_points,
                                likelihood=likelihood)
            self.gp0.covar_module = basek

            print(f"\nKernel: {name}, Test Error: {mse}")

            best_gp, mse_list = self.grid_search(N_sim=self.N_sim,
                                        os_limits=self.os_limits,
                                        se_ls_limits=self.se_ls_limits,
                                        rq_ls_limits=self.rq_ls_limits,
                                        alpha_limits=self.alpha_limits,
                                        plength_limits=self.plength_limits,
                                        mse_stop=self.mse_stop)
            mse = min(mse_list)
            print('errors: \n', mse_list)

            if mse < best_error:
                print(f'\n Local\nmse: {mse} best-error: {best_error}')
                best_error = copy.deepcopy(mse)
                best_kernel = copy.deepcopy(best_gp.covar_module)
                best_name = name

        return best_kernel, best_name, best_error

    # Function to explore kernel configurations
    def auto_model_learn(self, levels, N_sim=100,
                        os_limits=[0.8, 10], se_ls_limits=[0.1, 100],
                        rq_ls_limits=[0.1, 100], alpha_limits=[0.05, 2],
                        plength_limits=[1e-1, 5], nv_limits=[1e-2, 1e-3],
                        mse_stop=1e-3):
        
        self.N_sim = N_sim
        self.os_limits = os_limits
        self.se_ls_limits = se_ls_limits
        self.rq_ls_limits = rq_ls_limits
        self.alpha_limits = alpha_limits
        self.plength_limits = plength_limits
        self.nv_limits = nv_limits
        self.mse_stop = mse_stop

        final_best_error = float('inf')
        final_best_kernel = None
        
        if levels <= 0:
            raise Exception("Non valid number of levels")
    
        combined = self.base_kernels

        for level in range(levels):
            print(f"\nExploring level {level + 1} kernels...")
            if level <= 0:
                # Get current level's best
                ckernel, cname, cerror = self.get_best_kernel(combined)
                # best_kernel, best_name = self.get_best_kernel(combined)
            else:
                combined = self.combine_kernels(self.base_kernels, "+", combined)
                if level < 2:
                    combined.update(self.combine_kernels(self.base_kernels, "*",
                                                         self.base_kernels))
                    ckernel, cname, cerror = self.get_best_kernel(combined)
                    # best_kernel, best_name = self.get_best_kernel(combined)
                else:
                    ckernel, cname, cerror = self.get_best_kernel(combined)
                    # best_kernel, best_name = self.get_best_kernel(combined)
                
            # Update overall best if improvement found
            print(f'\nGlobal \nmse: {cerror} best-error: {final_best_error}')
            if cerror < final_best_error:
                final_best_error = copy.deepcopy(cerror)
                final_best_kernel = copy.deepcopy(ckernel)

                print('\nActualice: ', ckernel.base_kernel.base_kernel)
                print('os: ', final_best_kernel.base_kernel.outputscale.item())
                print('ls: ', final_best_kernel.base_kernel.base_kernel.lengthscale)

        return final_best_kernel
    
    def grid_search(self, N_sim, os_limits=None, se_ls_limits=None,
                    rq_ls_limits=None, alpha_limits=None,
                    plength_limits=None, mse_stop=1e-3):
        gp = self.gp0

        # Uniform distribution min-max values
        self.os_limits = os_limits
        self.se_ls_limits = se_ls_limits
        self.rq_ls_limits = rq_ls_limits
        self.alpha_limits = alpha_limits
        self.plength_limits = plength_limits

        # avoid 0.0 errors when random_start = False
        mse_list = np.ones(shape=N_sim)
        best_mse = float('inf')
        best_gp = None
        random_start = True

        for i in range(N_sim):
            print(f'Hyperparameter simulation: {i}/{N_sim}')

            if random_start:
                gp.likelihood.noise = uniform(low=self.nv_limits[0],
                                              high=self.nv_limits[1])
                self.init_kernel(self.from_uniform, self.kernel)

                # train and evaluate
                mse = self.train_and_evaluate(gp)

                mse_list[i] = mse
                print('Error: ', mse, '\n')

            # update best MSE
            if i > 1:
                if mse < mse_stop:
                    print('!! MSE successfully met the target MSE !!')
                    best_gp = gp
                    break
                else:
                    if mse < best_mse:
                        best_mse = copy.deepcopy(mse)
                        best_gp = copy.deepcopy(gp)

            # Adjust random start based on error improvement
            random_start = mse >= best_mse

        return best_gp, mse_list

    def tune(self, N_sim,
             os_std=None, se_ls_std=None, alpha_std=None, rq_ls_std=None,
             plength_std=None,
             mse_stop=1e-3):
        # Gaussian stds for each base kernel parameter
        self.os_std = os_std
        self.se_ls_std = se_ls_std
        self.alpha_std = alpha_std
        self.rq_ls_std = rq_ls_std
        self.plength_std = plength_std

        mse_list = np.zeros(shape=N_sim)
        mse_list[0] = float('inf')
        best_mse = float('inf')
        self.init_kernel(self.set_gauss_centres, self.kernel)
        gp = copy.deepcopy(self.gp0)
        best_gp = copy.deepcopy(self.gp0)

        for i in range(N_sim):
            self.init_kernel(self.from_gauss, self.kernel)
            gp.covar_module = self.kernel

            # Train and evaluate
            mse = self.train_and_evaluate(gp)
            mse_list[i] = mse

            print(f'Simulation: {i}/{N_sim}, Error: {round(mse_list[i], 5)}',
                  'Best error: ', round(best_mse, 5))

            # check error
            if i > 1:
                if mse < mse_stop:
                    print('!! MSE successfully met the target MSE !!')
                    best_gp = gp
                    break
                else:
                    if mse < best_mse:
                        best_mse = copy.deepcopy(mse)
                        best_gp = copy.deepcopy(gp)

        return best_gp
    
"""
Trained model from manual training
"""

state_dict = torch.load('expert_main0.pth',
                        weights_only=False)

# Access the inducing points
inducing_points = state_dict['covar_module.inducing_points']

# # Model
likelihood = GaussianLikelihood()

k = ScaleKernel(RQ(ard_num_dims=D))
covar_module = InducingPointKernel(k,
                                   inducing_points=inducing_points,
                                   likelihood=likelihood)
gp0 = SparseGP(X_train, y_train, likelihood, covar_module, init_noise_var)
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

print('\nMSE (train - test): ', mean_squared_error(mu, y_nonstand))
print('MSE (test):         ', mean_squared_error(mu[end_train:-1],
                                                 y_nonstand[end_train:-1]))

"""
    TEST classes
"""

pipeline = GPTraining(gp0, y_nonstand)
k = pipeline.auto_model_learn(levels=1, N_sim=300, os_limits=[1,3.5],
                              plength_limits=[1e-1, 3.5],
                              nv_limits=[0.026, 0.028],
                              mse_stop=0.003)
print('\nque devuelvo? \n', k)
print('ls: ', k.lengthscale)

gp = SparseGP(X_train, y_train, likelihood, k, init_noise_var)

print("\nOptimised kernel parameters:")
print("Outputscale:", gp.covar_module.base_kernel.outputscale.item())
# print("RBF-LS:\n", gp.covar_module.base_kernel.base_kernel.kernels[1].lengthscale)
# print("RQ-LS:\n", gp.covar_module.base_kernel.base_kernel.kernels[0].lengthscale)
# print("RQ-alpha: ", gp.covar_module.base_kernel.base_kernel.kernels[0].alpha.item())
print("Noise-var:", gp.likelihood.noise.item())
# torch.save(gp.state_dict(), 'expert_main.pth')

# Make sure the _z (induced inputs) are a subset of the X_train dataset
_z = gp.covar_module.inducing_points.detach()

_z_indices = []
for z in _z:
    distances = torch.norm(X_train - z, dim=1)
    closest_index = torch.argmin(distances).item()
    _z_indices.append(closest_index)

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
# plt.figure()
# plt.plot(mse_list)
# plt.xlabel('iteration')
# plt.xlabel('MSE')

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