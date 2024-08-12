import pandas as pd
import numpy as np
import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood
import matplotlib.pyplot as plt

"""
Not working with SKLearn normalisation packages:
The problem seems to be related with the vstack and hstack data
"""

# Load the Excel file
file_path = 'Synthetic.xlsx'
df = pd.read_excel(file_path)

# Read data
x_test_df = pd.read_excel(file_path, sheet_name='Testing')
labels_df = pd.read_excel(file_path, sheet_name='Real labels')

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

class SparseGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(SparseGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(lengthscale=0.9))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Inducing points
inducing_points = X[::10].clone()

# Model and likelihood
model = SparseGPModel(inducing_points)
likelihood = GaussianLikelihood()

# Initialize hyperparameters
likelihood.noise = 0.05

# Training
model.train()
likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
mll = VariationalELBO(likelihood, model, y_normalised.numel())

for i in range(100):
    optimizer.zero_grad()
    output = model(X)
    loss = -mll(output, y_normalised)
    loss.backward()
    optimizer.step()

# Evaluation
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(X))
    mean = observed_pred.mean
    lower, upper = observed_pred.confidence_region()

# Print the estimated hyperparameters
print("Lengthscale:", model.covar_module.base_kernel.lengthscale.item())
print("Outputscale:", model.covar_module.outputscale.item())
print("Noise var:  ", likelihood.noise.item())

estimated_z = model.variational_strategy.inducing_points.detach().numpy()

# Plotting
fig, ax = plt.subplots()
plt.plot(X.numpy(), y_normalised.numpy(), 'k*', label='Training Data')
plt.plot(X.numpy(), mean.numpy(), 'b', label='Predicted Mean')
plt.fill_between(X.numpy(), lower.numpy(), upper.numpy(), alpha=0.5,
                 label='2$\sigma$')
ax.vlines(
    x=inducing_points,
    ymin=y_normalised.min(),
    ymax=y_normalised.max(),
    alpha=0.3,
    linewidth=1.5,
    ls='--',
    label="z0",
    color='grey'
)
ax.vlines(
    x=estimated_z,
    ymin=y_normalised.min(),
    ymax=y_normalised.max(),
    alpha=0.3,
    linewidth=1.5,
    label="z*",
    color='orange'
)
plt.xlabel('Normalised x')
plt.xlabel('Normalised y')
plt.legend()
plt.title('Variational Sparse GP Regression with Inducing Points')
plt.show()