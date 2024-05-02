import paths
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from models.DGP import DistributedGP as DGP
from models.DDPGP import DistributedDPGP as DDPGP

from jax import jit
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import install_import_hook
import matplotlib as mpl
import matplotlib.pyplot as plt
import optax as ox

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

plt.close('all')

# Read excel file
file_name = paths.get_synthetic_path('Synthetic.xlsx')
df = pd.read_excel(file_name, sheet_name='Training')
x_test_df = pd.read_excel(file_name, sheet_name='Testing')
labels_df = pd.read_excel(file_name, sheet_name='Real labels')

# Get training data
X = df['X'].values
Y = df['Y'].values
N = len(Y)
xNew = x_test_df['X_star'].values

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
    
D = gpx.Dataset(X=X, y=Y)

# ! The following covariance function only works for K=2. Use a
# ! single kernel or add the K=k number of kernels, otherwise
# Covariance functions
kernel = gpx.kernels.RBF()
meanf = gpx.mean_functions.Zero()
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

# Likelihood
likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)

posterior = prior * likelihood

negative_mll = gpx.objectives.ConjugateMLL(negative=True)
negative_mll(posterior, train_data=D)

negative_mll = jit(negative_mll)

# Optimiser
opt_posterior, history = gpx.fit_scipy(model=posterior,
                                       objective=negative_mll,
                                       train_data=D)

# Predictions
latent_dist = opt_posterior.predict(xNew, train_data=D)
predictive_dist = opt_posterior.likelihood(latent_dist)

mu = predictive_dist.mean()
std = predictive_dist.stddev()

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------
plt.figure()

plt.plot(xNew, Y, "x", label="Observations", color='black', alpha=0.5)
# plt.plot(xNew, F, color='black', linewidth = 4, label='Sine function')
plt.plot(xNew, mu, color='red', linestyle='-', linewidth = 4,
         label='DDPGP')
plt.title('Regression Performance', fontsize=20)
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(prop={"size":20})

plt.show()