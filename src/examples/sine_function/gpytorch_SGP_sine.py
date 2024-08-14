import pandas as pd
import numpy as np
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import InducingPointKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
import matplotlib.pyplot as plt

# Load the Excel file
file_name = 'Synthetic.xlsx'
df = pd.read_excel(file_name, sheet_name='Training')

# Get training data
X_org = df['X'].values
X = df['X'].values
y = df['Y'].values
N = len(y)

# Normalize features
x_mean = np.mean(X)
x_std = np.std(X)
X_norm = (X - x_mean) / x_std

# Normalize targets
y_mean = np.mean(y)
y_std = np.std(y)
y_normalised = (y - y_mean) / y_std

# Convert data to torch tensors
X = torch.tensor(X_norm, dtype=torch.float32)
y_normalised = torch.tensor(y_normalised, dtype=torch.float32)

# Inducing points
inducing_points = X[::10].clone()

# Define the GP model
class SparseGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, inducing_points):
        super(SparseGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.base_covar_module = ScaleKernel(RBFKernel(lengthscale=0.9))
        self.covar_module = InducingPointKernel(self.base_covar_module,
                                                inducing_points=inducing_points,
                                                likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

# Initialize the model and likelihood
likelihood = GaussianLikelihood()

# Initialize hyperparameters (optional)
likelihood.noise = 0.05

model = SparseGP(X, y_normalised, likelihood, inducing_points)

# Train model
model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iterations = 100
for i in range(training_iterations):
    optimizer.zero_grad()
    output = model(X)
    loss = -mll(output, y_normalised)
    loss.backward()
    optimizer.step()

# Print the estimated hyperparameters
print("Lengthscale:", model.covar_module.base_kernel.base_kernel.lengthscale.item())
print("Outputscale:", model.covar_module.base_kernel.outputscale.item())
print("Noise:", likelihood.noise.item())

# get estimated inducing points
_z = model.covar_module.inducing_points.detach().numpy()

# Predictions
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(X))
    mu_norm = observed_pred.mean
    
# Unnormalize predictions
mu = mu_norm * y_std + y_mean
    
# MSE
# Real function
from sklearn.metrics import mean_squared_error as mse

F = 150 * X_org * np.sin(X_org)
print('MSE: ', mse(mu, F))

fig, ax = plt.subplots()
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.plot(X.numpy(), y, 'o', color='black')
plt.plot(X.numpy(), F, color='red', linewidth=3,
         label='f(x)')
plt.plot(X.numpy(), mu.numpy(), color='lightgreen', linewidth=3,
        label = 'SGP')
ax.vlines(
    x=X[::10],
    ymin=y.min().item(),
    ymax=y.max().item(),
    alpha=0.3,
    linewidth=1.5,
    ls='--',
    label="z0",
    color='grey'
)
ax.vlines(
    x=_z,
    ymin=y.min().item(),
    ymax=y.max().item(),
    alpha=0.3,
    linewidth=1.5,
    label="z*",
    color='orange'
)
plt.title('GPyTorch Sparse GP using inducing point kernel')
ax.set_xlabel("$x$", fontsize=14)
ax.set_ylabel("$f(x)$", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)
plt.show()