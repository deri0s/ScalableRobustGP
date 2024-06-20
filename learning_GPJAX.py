from models.DPSGP import DirichletProcessSparseGaussianProcess as DPSGP
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

"""
Radial Basis Function
"""
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from gpjax.base import static_field
import jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb
from gpjax.base import Module, param_field

@dataclass
class RBF(Module):
    lengthscale: float = param_field(
        init=False, bijector=tfb.Softplus(), trainable=True
    )
    variance: float = param_field(init=False, bijector=tfb.Softplus(), trainable=True)
    key: jax.Array = static_field(default_factory=lambda: jr.key(42))

    def __post_init__(self):
        # Split key into two keys
        key1, key2 = jr.split(self.key)

        # Sample from Gamma distribution to initialise lengthscale and variance
        self.lengthscale = tfd.Gamma(0.07, 0.9).sample(seed=key1)
        self.variance = tfd.Gamma(1e-6,0.7).sample(seed=key2)

    def covariance(self, x: jax.Array, y: jax.Array) -> jax.Array:
        return self.variance * jnp.exp(-0.5 * ((x - y) / self.lengthscale) ** 2)

# Covariance functions
se = gpx.kernels.RBF(variance=1, lengthscale=0.9)

# Initialize the White kernel with initial values
white = gpx.kernels.White(variance=0.05)

# Combine the RBF and White kernels
K = se + white

# Replace the bijector to enforce bounds on the lengthscale and variance
# kernel = kernel.replace_bijector(
#     lengthscale=tfb.Chain([tfb.Softplus(), tfb.Scale(scale=upper - lower), tfb.Shift(lower)])
# )

# kernel = kernel.replace_trainable(lengthscale=False)
# kernel = kernel.replace_trainable(variance=False)

# The DPGP model
rgp = DPSGP(X, Y, init_K=7, kernel=K, n_inducing=30, normalise_y=True,
            plot_sol=True, plot_conv=True)
rgp.train()
mu, std = rgp.predict(xNew)

print('DPGP init stds: ', rgp.init_pies)
print('DPGP init pies: ', rgp.init_sigmas)

### CALCULATING THE OVERALL MSE
from sklearn.metrics import mean_squared_error

F = 150 * xNew * np.sin(xNew)
print("Mean Squared Error (DPSGP)   : ", mean_squared_error(mu, F))

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------
plt.figure()
    
plt.plot(xNew, F, color='black', linewidth = 4, label='Sine function')
plt.plot(xNew, mu, color='red', linewidth = 4,
         label='DDPSGP')
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