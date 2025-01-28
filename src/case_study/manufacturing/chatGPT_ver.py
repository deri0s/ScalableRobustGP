import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
import torch
from sklearn.preprocessing import StandardScaler as ss
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import InducingPointKernel, ScaleKernel, RBFKernel as RBF

"""
NSG data
"""
file = 'validation_data.xlsx'

# Training df
X_df = pd.read_excel(file, sheet_name='X_stand')
y_df = pd.read_excel(file, sheet_name='y_nonstand')
t_df = pd.read_excel('data_and_preprocessing/processed/NSG_processed_data_14_inputs.xlsx',
                     sheet_name='timelags')

# Pre-Process training data
N, D = np.shape(X_df.values)
X = np.zeros([N, D])

def align_inputs(x_df, y_df, t_df):
    max_lag = max(t_df.iloc[0,:])
    for name, lag in t_df.items():
        x_df[name] = x_df[name].shift(lag[0])
    y_df = y_df.iloc[max_lag:].reset_index(drop=True)
    return x_df, y_df

X_df, y_df = align_inputs(X_df, y_df, t_df)
X_df.dropna(inplace=True)
X_df = X_df.reset_index(drop=True)

X = X_df.values
y_nonstand, date_time = y_df.gp_pred.values, y_df.date_time.values

N, D = np.shape(X)
end_train = N - int(len(y_nonstand) * 0.12)

assert N - int(len(X) * 0.12) == N - int(len(y_nonstand) * 0.12), 'Size of X and y are not the same'

X_train_np = X[0:end_train]
date_train = date_time[0:end_train]
N_train = len(X_train_np)
y_train_nonstand = y_nonstand[0:end_train]

X_test = X
date_time = date_time

assert len(X_train_np) == len(y_train_nonstand), 'X-train and y-train length are not the same'
assert len(X_test) == len(y_nonstand), 'X-test and y-test length are not the same'

# Standardise outputs
y_train = y_train_nonstand.reshape(-1, 1)
scaler = ss()
scaler.fit(y_train)
y_norm_np = scaler.transform(y_train)

X_train = torch.tensor(X_train_np, dtype=torch.float64)
y_train = torch.tensor(y_norm_np, dtype=torch.float64).squeeze()
X_test = torch.tensor(X_test, dtype=torch.float64)

# Sparse GP
inducing_points = X_train[::10, :].clone()

likelihood = GaussianLikelihood()

class SparseGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, inducing_points):
        super(SparseGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(RBF(ard_num_dims=train_x.shape[-1]))
        self.covar_module = InducingPointKernel(self.base_covar_module,
                                                inducing_points=inducing_points,
                                                likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

gp = SparseGP(X_train, y_train, likelihood, inducing_points)

# Print initial kernel parameters and inducing points
print("Initial kernel parameters:")
print("Lengthscale:", gp.covar_module.base_kernel.base_kernel.lengthscale)
print("Outputscale:", gp.covar_module.base_kernel.outputscale)
print("Initial inducing points:", gp.covar_module.inducing_points)

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

# Print estimated kernel parameters and inducing points after training
print("Estimated kernel parameters:")
print("Lengthscale:", gp.covar_module.base_kernel.base_kernel.lengthscale)
print("Outputscale:", gp.covar_module.base_kernel.outputscale)
print("Estimated inducing points:", gp.covar_module.inducing_points)

# Predictions
gp.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(gp(X_test))

    # Unnormalise predictions
    pred_mean = observed_pred.mean
    mu = scaler.inverse_transform(pred_mean.unsqueeze(1))[:, 0]
    stds = scaler.inverse_transform(observed_pred.stddev.unsqueeze(1))[:, 0]

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

print('\n', inducing_points[0:5])

print(np.all(inducing_points.detach().numpy() == _z_induced))
print('\n', inducing_points[0:5])
print('Size X-induced: ', np.shape(_z_induced))

fig, ax = plt.subplots()

plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
fig.autofmt_xdate()

ax.fill_between(date_time,
                mu + 2 * stds, mu - 2 * stds,
                alpha=0.5, color='lightcoral',
                label='3$\\sigma$')
ax.plot(date_time, y_nonstand, color='green', label='Val')
ax.plot(date_time, mu, color='red', label='GP')
plt.axvline(date_time[end_train - 1], linestyle='--', linewidth=3,
            color='black')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

ax.vlines(
    x=date_time[_z_indices],
    ymin=-2 * stds.min(),
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