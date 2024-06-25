from models.DPSGP import DirichletProcessSparseGaussianProcess as DPSGP
import pandas as pd
import paths

file_name = paths.get_synthetic_path('Synthetic.xlsx')
df = pd.read_excel(file_name, sheet_name='Training')

import numpy as np
from jax import jit
import jax.numpy as jnp
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

# Covariance functions
se = gpx.kernels.RBF(variance=1.0, lengthscale=7.9)
se = se.replace_trainable(variance=False)
# se = se.replace_trainable(lengthscale=False)

# Initialize the White kernel with initial values
white = gpx.kernels.White(variance=0.05)

# Combine the RBF and White kernels
kernel = se + white

"""
Standard GPJAX
"""
# Define the GP model
mu0 = gpx.mean_functions.Zero()
prior = gpx.gps.Prior(mean_function=mu0, kernel=kernel)

likelihood = gpx.likelihoods.Gaussian(num_datapoints=N)

posterior = prior * likelihood

# Define the training dataset
# Normalize targets
y_mean = np.mean(Y)
y_std = np.std(Y)
y_normalised = (Y - y_mean) / y_std

train_data = gpx.Dataset(X=np.vstack(X), y=np.vstack(y_normalised))

# Define the training objective
negative_mll = gpx.objectives.ConjugateMLL(negative=True)

# Train the model
opt_posterior, _ = gpx.fit_scipy(
    model=posterior,
    objective=negative_mll,
    train_data=train_data
)

# estimated hyperparameters
print('am: ', opt_posterior.prior.kernel.kernels[0].variance)
print('ls: ', opt_posterior.prior.kernel.kernels[0].lengthscale)
print('vr: ', opt_posterior.prior.kernel.kernels[1].variance)

# Make predictions
latent_dist = opt_posterior(X, train_data)
predictive_dist = opt_posterior.likelihood(latent_dist)
y_pred_gpjax_normalized = predictive_dist.mean()
sigma_gpjax_normalized = predictive_dist.stddev()

# Unnormalize predictions
y_pred_gpjax = y_pred_gpjax_normalized * y_std + y_mean
sigma_gpjax = sigma_gpjax_normalized * y_std

# Plot GPJAX predictions
plt.figure()
plt.title("GPJAX Gaussian Process Regressor")
plt.plot(X, Y, 'b.', markersize=10, label='Training data')
plt.plot(X, y_pred_gpjax, 'k-', color='red', label='Prediction')
plt.fill_between(X.ravel(), 
                 y_pred_gpjax - 1.96 * sigma_gpjax, 
                 y_pred_gpjax + 1.96 * sigma_gpjax, 
                 alpha=0.2, color='k')
plt.legend()

# The DPGP model
rgp = DPSGP(X, Y, init_K=7, kernel=kernel, n_inducing=30, normalise_y=True,
            plot_sol=True, plot_conv=True)
plt.show()
rgp.train()
mu, std = rgp.predict(xNew)

# print('DPGP init stds: ', rgp.init_pies)
# print('DPGP init pies: ', rgp.init_sigmas)

"""
DPGP
"""
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from models.DPGP import DirichletProcessGaussianProcess as DPGP

# Covariance functions
se = 1**2 * RBF(length_scale=0.70**2, length_scale_bounds=(0.07, 0.9))
#                variance
wn = WhiteKernel(noise_level=0.025, noise_level_bounds=(1e-6,0.7))

kernel = se + wn

# DPGP
rgp = DPGP(X, Y, init_K=7, kernel=kernel, normalise_y=True, 
           plot_sol=False, plot_conv=True)
rgp.train(pseudo_sparse=True)
muMix, stdMix = rgp.predict(xNew)

print('\nkernel: \n', rgp.kernel_)
print('\nhyper: \n', rgp.hyperparameters)

print('\nDPGP init stds: ', rgp.init_pies)
print('DPGP init pies: ', rgp.init_sigmas)

### CALCULATING THE OVERALL MSE
from sklearn.metrics import mean_squared_error

F = 150 * xNew * np.sin(xNew)
print("\nMean Squared Error (DPSGP)   : ", mean_squared_error(mu, F))
print("Mean Squared Error (DPGP)   : ", mean_squared_error(muMix, F))

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------
plt.figure()
    
plt.plot(xNew, F, color='black', linewidth = 4, label='Sine function')
plt.plot(xNew, mu, color='red', linewidth = 4,
         label='DDPSGP')
plt.plot(xNew, muMix, color='brown', linewidth = 4,
         label='DDPGP')
plt.title('Regression Performance', fontsize=20)
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(prop={"size":20})

# ----------------------------------------------------------------------------
# CONFIDENCE BOUNDS
# ----------------------------------------------------------------------------

color_iter = ['green', 'orange', 'red']
enumerate_K = [i for i in range(rgp.K_opt)]

plt.figure()
plt.plot(xNew, F, color='black', linestyle='-', linewidth = 4,
         label='$f(x)$')
plt.fill_between(
    xNew.squeeze(),
    mu - 2 * std,
    mu + 2 * std,
    alpha=0.2,
    label="Two sigma",
    color='lightcoral',
)

nl = ['Noise level 0', 'Noise level 1', 'Noise level 2']
for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
    plt.plot(X[rgp.indices[k]], Y[rgp.indices[k]], 'o',color=c,
             markersize = 9, label=nl[k])
    
plt.plot(xNew, mu, linewidth=4, color='green', label='DDPSGP')
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(prop={"size":20})

plt.show()