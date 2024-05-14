import pandas as pd
import paths

file_name = paths.get_synthetic_path('Synthetic.xlsx')
df = pd.read_excel(file_name, sheet_name='Training')

import numpy as np
from jax import jit
from jaxtyping import install_import_hook
import matplotlib.pyplot as plt

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

plt.close('all')

# Read data
x_test_df = pd.read_excel(file_name, sheet_name='Testing')
labels_df = pd.read_excel(file_name, sheet_name='Real labels')

# Get training data
X = df['X'].values
Y = df['Y'].values
N = len(Y)
xNew = np.vstack(x_test_df['X_star'].values)

# Get real labels
c0 = labels_df['Noise0'].values
c1 = labels_df['Noise1']
c2 = labels_df['Noise2']
not_nan = ~np.isnan(labels_df['Noise1'].values)
c1 = c1[not_nan]
c1 = [int(i) for i in c1]
not_nan = ~np.isnan(labels_df['Noise2'].values)
c2 = c2[not_nan]
c2 = [int(i) for i in c2]
indices = [c0, c1, c2]

# standardise data
X = np.vstack(X)
y_mu = np.mean(Y)
y_std = np.std(Y)
Y = np.vstack((Y - y_mu) / y_std)
D = gpx.Dataset(X=X, y=Y)

# Covariance functions
kernel = gpx.kernels.RBF()
meanf = gpx.mean_functions.Zero()
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

print('prior: ', prior)

# Likelihood
likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)

posterior = prior * likelihood

"""
    SPARSE APPROXIMATION
"""
import optax as ox
import jax.random as jr
from sklearn.metrics import mean_squared_error

# If n_inducing points is close to N, the model will not return
# accurate solutions.
n_inducing = 20
z = np.linspace(X.min(), X.max(), n_inducing).reshape(-1, 1)

q = gpx.variational_families.CollapsedVariationalGaussian(posterior=posterior,
                                                          inducing_inputs=z)

elbo = gpx.objectives.CollapsedELBO(negative=True)
elbo = jit(elbo)

# Training
opt_posterior, history = gpx.fit(
    model=q,
    objective=elbo,
    train_data=D,
    optim=ox.adamw(learning_rate=1e-2),
    num_iters=500,
    key=jr.key(123)
)

# Model convergence
fig, ax = plt.subplots()
ax.plot(history, color='red')
ax.set(xlabel="Training iterate", ylabel="ELBO")

print('prior.kernel.__call__:\n', prior.kernel.gram.__dict__)
# print('prior.kernel.lengthscale:\n', prior.kernel.lengthscale)
# print('prior.kernel.variance:\n', prior.kernel.variance)
# print('prior.kernel.lengthscale:\n', prior.kernel.)
# # Predictions
# latent_dist = opt_posterior(xNew, train_data=D)
# predictive_dist = opt_posterior.posterior.likelihood(latent_dist)
# inducing_points = opt_posterior.inducing_inputs

# mu = predictive_dist.mean()
# std = predictive_dist.stddev()