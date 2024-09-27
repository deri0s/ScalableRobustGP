import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io as sio
import mat73
from sklearn.preprocessing import MinMaxScaler as minmax
from sklearn.preprocessing import StandardScaler as ss
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.means import ConstantMean
from models.dpsgp_gpytorch import DirichletProcessSparseGaussianProcess as DPSGP

"""
No sirve
"""

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
vhgp_path = 'mvhgp.mat'
mvhgp = mat73.loadmat(vhgp_path)
std_var = mvhgp['fnm']
x = mvhgp['xm']
mu_var = mvhgp['ym']

#-----------------------------------------------------------------------------
# STANDARDISE OR NORMALISE DATA
#-----------------------------------------------------------------------------
preprocess = "ss"

if preprocess == "ss":
    train_scaler = minmax()
    output_scaler = ss()
elif preprocess == "mm":
    train_scaler = minmax()
    output_scaler = minmax()
else:
    print('Not a valid preprocessing method')

x_norm = train_scaler.fit_transform(X_org.reshape(-1, 1))
y_norm = output_scaler.fit_transform(Y_org.reshape(-1, 1))

# Convert data to torch tensors
floating_point = torch.float64

X = torch.tensor(x_norm, dtype=floating_point)
Y = torch.tensor(np.hstack(y_norm), dtype=floating_point)

# Define the kernel
rbf_kernel = RBFKernel()
rbf_kernel.lengthscale = 0.9
rbf_kernel.lengthscale_constraint = gpytorch.constraints.Interval(1e-5, 10)

scale_kernel = ScaleKernel(rbf_kernel)
scale_kernel.outputscale = 1.0
scale_kernel.outputscale_constraint = gpytorch.constraints.Interval(0.9, 1.1)

covar_module = scale_kernel

dpgp = DPSGP(X, np.hstack(Y_org), init_K=7,
             gp_model='Standard',
             prior_mean=ConstantMean(), kernel=covar_module,
             noise_var=0.005,
             floating_point=floating_point,
             normalise_y=True,
             print_conv=False, plot_conv=True, plot_sol=False)

# Create a class that sklearn RandomizedSearchCV can use
class DPGP_cv:
    def __init__(self, model):
        self.model = model

    def fit(self, X, Y):
        self.model.train()

    def predict(self, X):
        mus, stds = self.model.predict(X)
        return mus

    def get_params(self, deep=True):
        return {"model": self.model}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

# Cross-validation
param_dist = {"ls": np.linspace(1e-5, 1, 10),
              "std": np.linspace(1e-5, 1, 10)}

dpgp_cv = DPGP_cv(dpgp)

randomized_search = RandomizedSearchCV(
    estimator=dpgp_cv,
    param_distributions=param_dist,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

randomized_search.fit(X, np.hstack(Y_org))

# print search
cv_results = randomized_search.cv_results_

# Print the parameter combinations and their scores
for mean_score, params in zip(cv_results['mean_test_score'], cv_results['params']):
    print(f"Score: {mean_score}, Parameters: {params}")


# Best model
best_model = randomized_search.best_estimator_

# Access the hyperparameters
length_scale = best_model.model.kernel.base_kernel.lengthscale.item()
output_scale = best_model.model.kernel.outputscale.item()
noise_var = best_model.model.likelihood.noise.item()

# Format the hyperparameters into an equation-like string
kernel_eq = f"{output_scale:.2f} * SE(ls={length_scale:.2f}) + {noise_var:.2f}^2"

print("\nTuned hyperparameters: \n")
print(f"\nKernel Equation: {kernel_eq}")

mus = best_model.predict(X)
mse = mean_squared_error(mu_var, mus)
print(f"\nMean Squared Error: {mse}")

#-----------------------------------------------------------------------------
# CROSS-VALIDATION 
#-----------------------------------------------------------------------------

results = randomized_search.cv_results_
scores = results['mean_test_score']
params = results['params']

# Extract parameter values
ls_values = [param['ls'] for param in params]
std_values = [param['std'] for param in params]

# Plot the scores
plt.figure()
plt.scatter(ls_values, scores, label='Length Scale')
plt.scatter(std_values, scores, label='Standard Deviation')
plt.xlabel('Hyperparameter Value')
plt.ylabel('Score')
plt.legend()

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------
plt.figure()
plt.fill_between(x, mu_var + 3*std_var, mu_var - 3*std_var,
                 alpha=0.5,color='pink',label='3$\\sigma$ (VHGP)')
plt.plot(X_org, Y_org, 'o', color='black')
plt.plot(x, mu_var, 'red', linewidth=3, label='VHGP')
plt.plot(x, mus, 'green', linewidth=3, label='DPGP')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration')
plt.legend(loc=4, prop={"size":20})

plt.show()