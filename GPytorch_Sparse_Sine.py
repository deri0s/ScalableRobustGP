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
import paths

file_name = paths.get_synthetic_path('Synthetic.xlsx')
df = pd.read_excel(file_name, sheet_name='Training')
steps = 6

# Get training data
X = df['X'].values
y = df['Y'].values
N = len(y)

# Normalize targets
y_mean = np.mean(y)
y_std = np.std(y)
y_normalised = (y - y_mean) / y_std

# Convert data to torch tensors
X = torch.tensor(X, dtype=torch.float32)
y_normalised = torch.tensor(y_normalised, dtype=torch.float32)

class SparseGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points,
                                                   variational_distribution,
                                                   learn_inducing_locations=True)
        super(SparseGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(lengthscale=0.9))
        print('variational strategy: ', variational_strategy)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Inducing points
inducing_points = X[::steps]
print('inducing points: \n', inducing_points)

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

# estimated inducing points
estimated_z = model.variational_strategy.inducing_points.detach().numpy()
print('Inducing inputs:')
print('original:  ', len(inducing_points))
print('estimated: ', len(estimated_z[:,0]))

# Plotting
fig, ax = plt.subplots()
plt.plot(X.numpy(), y_normalised.numpy(), 'k*', label='Training Data')
plt.plot(X.numpy(), mean.numpy(), 'b', label='Predicted Mean')
plt.fill_between(X.numpy(), lower.numpy(), upper.numpy(), alpha=0.5, label='Confidence Interval')
ax.vlines(
    x=estimated_z,
    ymin=y_normalised.min(),
    ymax=y_normalised.max(),
    alpha=0.3,
    linewidth=1.5,
    label="Inducing point",
    color='orange'
)

plt.legend()
plt.title('Sparse GP Regression with Inducing Points')
plt.show()