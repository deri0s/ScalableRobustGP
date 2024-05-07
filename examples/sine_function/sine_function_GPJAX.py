import os
import sys
from pathlib import Path
import pandas as pd

# Import data
file_name = 'Synthetic.xlsx'
FILE = Path(__file__).resolve()
root = FILE.parents[2]
path = root / 'examples/sine_function' / file_name
df = pd.read_excel(file_name, sheet_name='Training')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from jax import jit
from jaxtyping import install_import_hook
import matplotlib.pyplot as plt

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

plt.close('all')

from pathlib import Path
import pandas as pd

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
from sklearn.preprocessing import StandardScaler as ss

# Y = ss.fit_transform(np.vstack(Y))
X=np.vstack(X)
y_mu = np.mean(Y)
y_std = np.std(Y)
Y = np.vstack((Y - y_mu) / y_std)
D = gpx.Dataset(X=X, y=Y)

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

plt.plot(X, Y, "x", label="Observations", color='black', alpha=0.5)
plt.fill_between(
    xNew.squeeze(),
    mu - 2 * std,
    mu + 2 * std,
    alpha=0.2,
    label="Two sigma",
    color='lightcoral',
)
# plt.plot(xNew, F, color='black', linewidth = 4, label='Sine function')
plt.plot(xNew, mu, color='red', linestyle='-', linewidth = 4,
         label='DDPGP')
plt.title('Regression Performance', fontsize=20)
plt.xlabel('x', fontsize=16)
plt.ylabel('f(x)', fontsize=16)
plt.legend(prop={"size":20})

plt.show()