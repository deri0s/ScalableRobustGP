import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.io as sio
import mat73
from sklearn.gaussian_process import GaussianProcessRegressor as GP
from models.dpgp import DirichletProcessGaussianProcess as DPGP
from models.dpsgp_gpytorch import DirichletProcessSparseGaussianProcess as DPSGP
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

plt.close('all')

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------

file_name = 'motorcycle.mat'
motorcycle_data = sio.loadmat(file_name)
X = motorcycle_data['X']
Y = motorcycle_data['y']
N = len(X)

# Load MVHGP predictive mean and variance
# vhgp_path= 'mvhgp.mat'
# mvhgp = mat73.loadmat(vhgp_path)
# std_var = mvhgp['fnm']
# x = mvhgp['xm']
# mu_var = mvhgp['ym']

#-----------------------------------------------------------------------------
# STANDARDISE DATA
#-----------------------------------------------------------------------------
from sklearn.preprocessing import MinMaxScaler as minmax

train_scaler = minmax()
X_norm = train_scaler.fit_transform(X.reshape(-1,1))

# Convert data to torch tensors
floating_point = torch.float32

# # ----------------------------------------------------------------------------
# # Covariance Function
# # ----------------------------------------------------------------------------
# se = 1**2 * RBF(length_scale=1, length_scale_bounds=(1e-3,1e3))
# wn = WhiteKernel(noise_level=2**2, noise_level_bounds=(1e-6,1e3))

# kernel = se + wn

# # ----------------------------------------------------------------------------
# # GP
# # ----------------------------------------------------------------------------
# gp = GP(kernel, alpha=0, normalize_y=True)
# gp.fit(np.vstack(X_norm), np.vstack(Y))
# mu, std = gp.predict(X, return_std=True)

# # ----------------------------------------------------------------------------
# # DPGP
# #-----------------------------------------------------------------------------
# dpgp = DPGP(X_norm, Y, init_K=6, DP_max_iter=70, kernel=kernel, normalise_y=True,
#             plot_conv=True)
# dpgp.train()
# mu_dpgp, std_dpgp = dpgp.predict(X_norm)
# print(type(X_norm))
#-------------------------------------------------------------------------------
# DDPGP torch
#-------------------------------------------------------------------------------
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel as RBF

covar_module = ScaleKernel(RBF(lengthscale=0.9))
likelihood = GaussianLikelihood()

gp = DPSGP(X_norm, Y, init_K=6,
           gp_model='Standard',
           prior_mean=ConstantMean(), kernel=covar_module,
           noise_var = 0.05,
           floating_point=floating_point,
           normalise_y=True,
           print_conv=False, plot_conv=True, plot_sol=False)
gp.train()
mus, stds = gp.predict(X_norm)

# # ----------------------------------------------------------------------------
# # VHGP
# # ----------------------------------------------------------------------------

# plt.figure()
# plt.fill_between(x, mu_var + 3*std_var, mu_var - 3*std_var,
#                   alpha=0.5,color='pink',label='3$\\sigma$ (VHGP)')
# plt.fill_between(x, mu_dpgp + 3*std_dpgp, mu_dpgp - 3*std_dpgp,
#                   alpha=0.4,color='limegreen',label='3$\\sigma$ (DPGP-sklearn)')
# plt.plot(X, Y, 'o', color='black')
# plt.plot(x, mu, 'blue', label='GP')
# plt.plot(x, mu_var, 'red', label='VHGP')
# plt.plot(x, mu_dpgp, 'green', label='DPGP-sklearn')
# plt.plot(x, mus, 'purple', label='DPGP-torch')
# plt.xlabel('Time (s)')
# plt.ylabel('Acceleration')
# plt.legend(loc=4, prop={"size":20})

# # ----------------------------------------------------------------------------
# # CLUSTERING
# # ----------------------------------------------------------------------------

# color_iter = ['lightgreen', 'red', 'black']
# nl = ['Noise level 0', 'Noise level 1']
# enumerate_K = [i for i in range(dpgp.K_opt)]

# plt.figure()
# if dpgp.K_opt != 1:
#     for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
#         plt.plot(x[dpgp.indices[k]], Y[dpgp.indices[k]], 'o',
#                   color=c, markersize = 8, label = nl[k])
# plt.plot(x, mus, color="green", linewidth = 4, label="DPGP-torch")
# plt.xlabel('Time (s)', fontsize=16)
# plt.ylabel('Acceleration', fontsize=16)
# plt.legend(loc=0, prop={"size":20})
# plt.show()