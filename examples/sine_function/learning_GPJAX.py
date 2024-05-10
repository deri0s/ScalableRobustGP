from pathlib import Path

import sys
import os

FILE = Path(__file__).resolve()
root = FILE.parents[2]
os.chdir(root)

from models.DPSGP import DirichletProcessSparseGaussianProcess as DPSGP
import pandas as pd

# # Import data
# file_name = 'Synthetic.xlsx'
# FILE = Path(__file__).resolve()
# root = FILE.parents[2]
# path = root / 'examples/sine_function' / file_name
# df = pd.read_excel(file_name, sheet_name='Training')

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# from jax import jit
# from jaxtyping import install_import_hook
# import matplotlib.pyplot as plt

# with install_import_hook("gpjax", "beartype.beartype"):
#     import gpjax as gpx

# plt.close('all')

# from pathlib import Path
# import pandas as pd

# # Read data
# x_test_df = pd.read_excel(file_name, sheet_name='Testing')
# labels_df = pd.read_excel(file_name, sheet_name='Real labels')

# # Get training data
# X = df['X'].values
# Y = df['Y'].values
# N = len(Y)
# xNew = np.vstack(x_test_df['X_star'].values)

# # Get real labels
# c0 = labels_df['Noise0'].values
# c1 = labels_df['Noise1']
# c2 = labels_df['Noise2']
# not_nan = ~np.isnan(labels_df['Noise1'].values)
# c1 = c1[not_nan]
# c1 = [int(i) for i in c1]
# not_nan = ~np.isnan(labels_df['Noise2'].values)
# c2 = c2[not_nan]
# c2 = [int(i) for i in c2]
# indices = [c0, c1, c2]

# # standardise data
# X = np.vstack(X)
# y_mu = np.mean(Y)
# y_std = np.std(Y)
# Y = np.vstack((Y - y_mu) / y_std)
# D = gpx.Dataset(X=X, y=Y)

# # Covariance functions
# kernel = gpx.kernels.RBF()
# meanf = gpx.mean_functions.Zero()
# prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

# # Likelihood
# likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)

# posterior = prior * likelihood

# """
#     SPARSE APPROXIMATION
# """
# import optax as ox
# import jax.random as jr
# from jax import vmap
# from sklearn.metrics import mean_squared_error

# # If n_inducing points is close to N, the model will not return
# # accurate solutions.
# n_inducing = 20
# z = np.linspace(X.min(), X.max(), n_inducing).reshape(-1, 1)

# q = gpx.variational_families.CollapsedVariationalGaussian(posterior=posterior,
#                                                           inducing_inputs=z)

# elbo = gpx.objectives.CollapsedELBO(negative=True)
# elbo = jit(elbo)

# # Training
# opt_posterior, history = gpx.fit(
#     model=q,
#     objective=elbo,
#     train_data=D,
#     optim=ox.adamw(learning_rate=1e-2),
#     num_iters=500,
#     key=jr.key(123)
# )

# # def q_moments(x):
# #     qx = q(x)
# #     return qx.mean().squeeze(), qx.covariance().squeeze()

# # mean, variance = vmap(q_moments(), in_axes= (0, None)) #(X[:, None])

# # # ≈ ∫[log(p(y|f(x))) q(f(x))] df(x)
# # expectation = q.posterior.likelihood.expected_log_likelihood(
# #     Y, mean[:, None], variance[:, None]
# # )
# # import jax.numpy as jnp

# # print('pri: ', jnp.sum(expectation))

# print('second: ', np.sum(history))
# print('second: ', np.shape(history))
# print('second: ', history[-1])

# # print('log-likelihood: ', jnp.sum(likelihood.expected_log_likelihood(y=Y,
# #                                                              mean=0,
# #                                                              variance=variance)))
# # # Model convergence
# # fig, ax = plt.subplots()
# # ax.plot(history, color='red')
# # ax.set(xlabel="Training iterate", ylabel="ELBO")

# # # Predictions
# # latent_dist = opt_posterior(xNew, train_data=D)
# # predictive_dist = opt_posterior.posterior.likelihood(latent_dist)
# # inducing_points = opt_posterior.inducing_inputs

# # mu = predictive_dist.mean()
# # std = predictive_dist.stddev()

# # # CALCULATING THE OVERALL MSE
# # F = 150 * xNew * np.sin(xNew)
# # F = np.vstack((F - np.mean(F)) / np.std(F))
# # print("Mean Squared Error (Sparse GP)    : ", mean_squared_error(mu, F))

# # #-----------------------------------------------------------------------------
# # # REGRESSION PLOT
# # #-----------------------------------------------------------------------------
# # fig, ax = plt.subplots()

# # ax.vlines(
# #     x=inducing_points,
# #     ymin=Y.min(),
# #     ymax=Y.max(),
# #     alpha=0.3,
# #     linewidth=1.5,
# #     label="Inducing point",
# #     color='orange',
# # )

# # plt.plot(xNew, F, color='black', linewidth = 4,
# #          label='Sine function')

# # plt.plot(X, Y, "x", label="Observations", color='black', alpha=0.5)

# # plt.fill_between(
# #     xNew.squeeze(),
# #     mu - 2 * std,
# #     mu + 2 * std,
# #     alpha=0.2,
# #     label="Two sigma",
# #     color='lightcoral',
# # )

# # plt.plot(xNew, mu, color='red', linestyle='-', linewidth = 4,
# #          label='Sparse GP')
# # plt.title('Regression Performance', fontsize=20)
# # plt.xlabel('x', fontsize=16)
# # plt.ylabel('f(x)', fontsize=16)
# # plt.legend(prop={"size":20})

# # plt.show()