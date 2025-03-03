import torch
import gpytorch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import yaml
from matplotlib import pyplot as plt
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import InducingPointKernel, ScaleKernel
from gpytorch.kernels import RBFKernel as RBF, RQKernel as RQ
from gpytorch.kernels import LinearKernel as Lin, PeriodicKernel as Per

# Load data
file = 'validation_data_main.xlsx'

# Training df
X_df = pd.read_excel(file, sheet_name='X_stand')
y_df = pd.read_excel(file, sheet_name='y_nonstand')
t_df = pd.read_excel(file, sheet_name='timelags')

# Drop Tweel position
X_df.drop(columns=['9282 Tweel Position'], inplace=True)
t_df.drop(columns=['9282 Tweel Position'], inplace=True)

# Pre-Process training data
N, D = np.shape(X_df.values)

# Create tag inputs
X = np.zeros([N, D])

# Create lagged features
def align_inputs(x_df, y_df, t_series):
    xdeep = x_df.copy()
    ydeep = y_df.copy()
    max_lag = int(max(t_series))

    for name, lag in t_series.items():
        xdeep[name] = xdeep[name].shift(int(lag))
    xdeep.dropna(inplace=True)

    ydeep = ydeep.iloc[max_lag:].reset_index(drop=True)

    return xdeep.reset_index(drop=True), ydeep

X_df, y_df = align_inputs(X_df, y_df, t_df.iloc[0,:])

# Standardize training & test data
X = X_df.values
y_nonstand, date_time = y_df.gp_pred.values, y_df.date_time.values

N, D = np.shape(X)
test_perc = 0.12
end_train = N - int(len(y_nonstand) * test_perc)

X_train_np = X[0:end_train]
date_train = date_time[0:end_train]
N_train = len(X_train_np)
y_train_nonstand = y_nonstand[0:end_train]

X_test = X[0:N]
date_time = date_time[0:N]

# Standardize outputs
y_train = y_train_nonstand.reshape(-1, 1)
scaler = StandardScaler()
scaler.fit(y_train)
y_norm_np = scaler.transform(y_train)

# Convert data to torch tensors
floating_point = torch.float64
X_train = torch.tensor(X_train_np, dtype=floating_point)
y_train = torch.tensor(y_norm_np, dtype=floating_point).squeeze()
X_test = torch.tensor(X_test, dtype=floating_point)

# Always clone
step = 19
inducing_points = X_train[::step, :].clone()

# Read best hyperparameters and initialization values from the yml file
with open('config_RBF_plus_RQ_2_step40.yaml', 'r') as f:
    config = yaml.safe_load(f)

init_os = torch.tensor(config['outputscale']['initial'],
                       dtype=floating_point)
init_ls_se = torch.tensor(config['RBF']['lengthscale']['initial'],
                          dtype=floating_point)
init_ls_rq = torch.tensor(config['RQ']['lengthscale']['initial'],
                          dtype=floating_point)
init_alpha = torch.tensor(config['RQ']['alpha']['initial'],
                          dtype=floating_point)
init_noise_var = torch.tensor(config['WN']['var']['initial'],
                              dtype=floating_point)
init_plength = torch.tensor(torch.rand(D), dtype=floating_point)

# Automatic model selection
base_kernels = {
    'RBF': lambda: RBF(ard_num_dims=D, dtype=floating_point),
    'RQ': lambda: RQ(ard_num_dims=D, dtype=floating_point),
    'Lin': lambda: Lin(ard_num_dims=D, dtype=floating_point),
    'Per': lambda: Per(ard_num_dims=D, dtype=floating_point)
}

# Define the model
likelihood = GaussianLikelihood()

class GPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = kernel
        self.likelihood.noise = init_noise_var

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

# Function to initialize kernel parameters
def init_hyper(gp: ExactGP) -> ExactGP:
    kernel = gp.covar_module
    kernel.base_kernel.outputscale = init_os

    def set_params(k):
        if not isinstance(k, Lin):
            if isinstance(k, RBF):
                k.lengthscale = init_ls_se
            if isinstance(k, RQ):
                k.alpha = init_alpha
                k.lengthscale = init_ls_rq
            if isinstance(k, Per):
                k.period_length = init_plength
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

# Function to train and evaluate the model
def train_and_evaluate(kernel):
    kernel = InducingPointKernel(ScaleKernel(kernel),
                                 inducing_points=inducing_points,
                                 likelihood=likelihood)
    model = GPModel(X_train, y_train, likelihood, kernel)
    model = init_hyper(model)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    mll = ExactMarginalLogLikelihood(likelihood, model)

    training_iterations = 100
    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()

    # Predictions
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(X_test))

        # Unnormalize predictions
        pred_mean = observed_pred.mean
        mu = scaler.inverse_transform(pred_mean.unsqueeze(1))[:, 0]

    test_error = mean_squared_error(mu, y_nonstand)

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

def get_best_kernel(kernels: dict) -> gpytorch.kernels:
    best_kernel = None
    best_name = None
    best_error = float('inf')

    for name, kernel in kernels.items():
        test_error = train_and_evaluate(kernel())
        print(f"Kernel: {name}, Test Error: {test_error}")

        if test_error < best_error:
            best_error = test_error
            best_kernel = kernel
            best_name = name
    return best_kernel, best_name

# Function to explore kernel configurations
def explore_kernels(levels):
    if levels == 0:
        raise Exception("The number shouldn't be an odd integer")
   
    combined = base_kernels

    for level in range(levels):
        print(f"\nExploring level {level + 1} kernels...")
        if level <= 0:
            best_kernel, best_name = get_best_kernel(combined)
        else:
            combined = combine_kernels(base_kernels, "+", combined)
            if level < 2:
                combined.update(combine_kernels(base_kernels, "*", base_kernels))
                best_kernel, best_name = get_best_kernel(combined)
            else:
                best_kernel, best_name = get_best_kernel(combined)

    return best_kernel(), best_name

# User-defined level of complexity
level = 3

# Explore kernel configurations
estimated_kernel, estimated_kernel_name = explore_kernels(level)
kernel = InducingPointKernel(ScaleKernel(estimated_kernel),
                             inducing_points=inducing_points,
                             likelihood=likelihood)

# Train and evaluate the final model with the best kernel
likelihood = GaussianLikelihood()
gp = GPModel(X_train, y_train, likelihood, kernel)
gp = init_hyper(gp)

print("\nInitial kernel parameters:")
print("Outputscale:", gp.covar_module.base_kernel.outputscale.item())
print("RBF-LS:\n", gp.covar_module.base_kernel.base_kernel.kernels[1].lengthscale)
print("RQ-LS:\n", gp.covar_module.base_kernel.base_kernel.kernels[0].lengthscale)
# print("RQ-alpha: ", gp.covar_module.base_kernel.base_kernel.kernels[0].alpha.item())
print("Noise-var:", gp.likelihood.noise.item())

gp.train()
likelihood.train()

optimizer = torch.optim.Adam(gp.parameters(), lr=0.01)
mll = ExactMarginalLogLikelihood(likelihood, gp)

training_iterations = 150
for i in range(training_iterations):
    optimizer.zero_grad()
    output = gp(X_train)
    loss = -mll(output, y_train)
    loss.backward()
    optimizer.step()

gp.eval()
likelihood.eval()

with torch.no_grad():
    observed_pred = likelihood(gp(X_test))

    # Unormalise predictions
    pred_mean = observed_pred.mean
    mu = scaler.inverse_transform(pred_mean.unsqueeze(1))[:,0]
    stds = scaler.inverse_transform(observed_pred.stddev.unsqueeze(1))[:,0]
    lower_stand, upper_stand = observed_pred.confidence_region()
    lower = scaler.inverse_transform(lower_stand.unsqueeze(1))[:,0]
    upper = scaler.inverse_transform(upper_stand.unsqueeze(1))[:,0]

print(f"\nEstimated kernel:\n {estimated_kernel_name}")
print('MSE (train-test): ',mean_squared_error(mu, y_nonstand))
print('MSE (test):       ',mean_squared_error(mu[end_train:-1],
                                              y_nonstand[end_train:-1]))

print("\nOpt kernel parameters:")
print("Outputscale:", gp.covar_module.base_kernel.outputscale.item())
"""--------------------------------------------------------------------------
PLOT
"""

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
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)
plt.show()