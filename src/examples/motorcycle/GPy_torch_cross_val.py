import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io as sio
import mat73
from sklearn.metrics import mean_squared_error

plt.close('all')

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------

file_name = 'motorcycle.mat'
motorcycle_data = sio.loadmat(file_name)
X_org = motorcycle_data['X']
Y_org = motorcycle_data['y']
N = len(X_org)

# Load MVHGP predictive mean and variance
vhgp_path= 'mvhgp.mat'
mvhgp = mat73.loadmat(vhgp_path)
std_var = mvhgp['fnm']
x = mvhgp['xm']
mu_var = mvhgp['ym']

#-----------------------------------------------------------------------------
# STANDARDISE OR NORMALISE DATA
#-----------------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler as minmax
from sklearn.preprocessing import StandardScaler as ss
preprocess = "ss"

"""
Best configuration:
    ?
    I am changing manually the train_scaler
"""

if preprocess == "ss":
    train_scaler = minmax()
    output_scaler = ss()
elif preprocess == "mm":
    train_scaler = minmax()
    output_scaler = minmax()
else:
    print('Not a valid prerpocesing method')

x_norm = train_scaler.fit_transform(X_org.reshape(-1,1))
y_norm = output_scaler.fit_transform(Y_org.reshape(-1,1))

# Convert data to torch tensors
floating_point = torch.float64

X = torch.tensor(x_norm, dtype=floating_point)
Y = torch.tensor(np.hstack(y_norm), dtype=floating_point)

import torch
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import numpy as np

# Define the Gaussian Process model using GPyTorch
class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Define a function to train and evaluate the model
def train_and_evaluate(length_scale, noise_variance):
    likelihood = GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(noise_variance))
    model = GPRegressionModel(X, Y, likelihood)
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for i in range(50):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, Y)
        loss.backward()
        optimizer.step()
    
    model.eval()
    likelihood.eval()
    
    with torch.no_grad():
        preds = model(torch.tensor(X).float())
        mean = preds.mean
        mse = mean_squared_error(mu_var, mean.numpy())
    
    return mse

# Define the parameter grid
param_grid = {
    'length_scale': np.logspace(-2, 2, 10),
    'noise_variance': np.logspace(-4, 0, 10)
}

# Perform randomized search cross-validation
best_score = float('inf')
best_params = None

for length_scale in param_grid['length_scale']:
    for noise_variance in param_grid['noise_variance']:
        score = train_and_evaluate(length_scale, noise_variance)
        if score < best_score:
            best_score = score
            best_params = {'length_scale': length_scale, 'noise_variance': noise_variance}

print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")
