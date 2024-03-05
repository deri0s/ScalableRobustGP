import paths
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import mat73
from sklearn.gaussian_process import GaussianProcessRegressor as GP
from models.DPGP import DirichletProcessGaussianProcess as DPGP
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

plt.close('all')

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------

mot_path = paths.get_motorcycle_path('motorcycle.mat')
motorcycle_data = sio.loadmat(mot_path)
X = motorcycle_data['X']
Y = motorcycle_data['y']
N = len(X)

# Load MVHGP predictive mean and variance
vhgp_path= paths.get_motorcycle_path('mvhgp.mat')
mvhgp = mat73.loadmat(vhgp_path)
std_var = mvhgp['fnm']
x = mvhgp['xm']
mu_var = mvhgp['ym']

# ----------------------------------------------------------------------------
# Covariance Function
# ----------------------------------------------------------------------------
se = 1**2 * RBF(length_scale=1, length_scale_bounds=(1e-3,1e3))
wn = WhiteKernel(noise_level=2**2, noise_level_bounds=(1e-6,1e3))

kernel = se + wn

# ----------------------------------------------------------------------------
# GP
# ----------------------------------------------------------------------------
gp = GP(kernel, alpha=0, normalize_y=True)
gp.fit(np.vstack(X), np.vstack(Y))
mu, std = gp.predict(X, return_std=True)

# ----------------------------------------------------------------------------
# DPGP
#-----------------------------------------------------------------------------
dpgp = DPGP(X, Y, init_K=6, DP_max_iter=70, kernel=kernel, normalise_y=True, plot_conv=True)
dpgp.train()
mu_dpgp, std_dpgp = dpgp.predict(X)

# ----------------------------------------------------------------------------
# VHGP
# ----------------------------------------------------------------------------

plt.figure()
plt.fill_between(x, mu_var + 3*std_var, mu_var - 3*std_var,
                  alpha=0.5,color='pink',label='Confidence \nBounds (VHGP)')
plt.fill_between(x, mu_dpgp + 3*std_dpgp, mu_dpgp - 3*std_dpgp,
                  alpha=0.4,color='limegreen',label='Confidence \nBounds (DPGP)')
plt.plot(X, Y, 'o', color='black')
plt.plot(x, mu, 'blue', label='GP')
plt.plot(x, mu_var, 'red', label='VHGP')
plt.plot(x, mu_dpgp, 'purple', label='DPGP')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration')
plt.legend(loc=4, prop={"size":20})

# ----------------------------------------------------------------------------
# CLUSTERING
# ----------------------------------------------------------------------------

color_iter = ['lightgreen', 'red', 'black']
nl = ['Noise level 0', 'Noise level 1']
enumerate_K = [i for i in range(dpgp.K_opt)]

plt.figure()
if dpgp.K_opt != 1:
    for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
        plt.plot(x[dpgp.indices[k]], Y[dpgp.indices[k]], 'o',
                  color=c, markersize = 8, label = nl[k])
plt.plot(x, mu_dpgp, color="green", linewidth = 4, label="DPGP")
plt.xlabel('Time (s)', fontsize=16)
plt.ylabel('Acceleration', fontsize=16)
plt.legend(loc=0, prop={"size":20})
plt.show()