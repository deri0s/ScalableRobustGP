from models.DPSGP import DirichletProcessSparseGaussianProcess as DPSGP
import pandas as pd
import paths
import matplotlib.pyplot as plt

file_name = paths.get_synthetic_path('Synthetic.xlsx')
df = pd.read_excel(file_name, sheet_name='Training')

# Read data
x_test_df = pd.read_excel(file_name, sheet_name='Testing')
labels_df = pd.read_excel(file_name, sheet_name='Real labels')

# Get training data
X = df['X'].values
Y = df['Y'].values
N = len(Y)

# Covariance function
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel as RBF

prior_cov = ScaleKernel(RBF(lengthscale=0.9))

gp = DPSGP(X, Y, init_K=8,
           gp_model=ExactGP, kernel=prior_cov,
           n_inducing=30, normalise_y=True,
           plot_conv=True, plot_sol=True)
plt.show()
# # Covariance functions
# se = gpx.kernels.RBF(variance=1.0, lengthscale=7.9)
# se = se.replace_trainable(variance=False)
# # se = se.replace_trainable(lengthscale=False)

# # Initialize the White kernel with initial values
# white = gpx.kernels.White(variance=0.05)

# # Combine the RBF and White kernels
# kernel = se + white

# """
# Standard GPytorch
# """

# F = 150 * xNew * np.sin(xNew)
# print("\nMean Squared Error (DPSGP)   : ", mean_squared_error(mu, F))
# print("Mean Squared Error (DPGP)   : ", mean_squared_error(muMix, F))

# #-----------------------------------------------------------------------------
# # REGRESSION PLOT
# #-----------------------------------------------------------------------------
# plt.figure()
    
# plt.plot(xNew, F, color='black', linewidth = 4, label='Sine function')
# plt.plot(xNew, mu, color='red', linewidth = 4,
#          label='DDPSGP')
# plt.plot(xNew, muMix, color='brown', linewidth = 4,
#          label='DDPGP')
# plt.title('Regression Performance', fontsize=20)
# plt.xlabel('x', fontsize=16)
# plt.ylabel('f(x)', fontsize=16)
# plt.legend(prop={"size":20})

# # ----------------------------------------------------------------------------
# # CONFIDENCE BOUNDS
# # ----------------------------------------------------------------------------

# color_iter = ['green', 'orange', 'red']
# enumerate_K = [i for i in range(rgp.K_opt)]

# plt.figure()
# plt.plot(xNew, F, color='black', linestyle='-', linewidth = 4,
#          label='$f(x)$')
# plt.fill_between(
#     xNew.squeeze(),
#     mu - 2 * std,
#     mu + 2 * std,
#     alpha=0.2,
#     label="Two sigma",
#     color='lightcoral',
# )

# nl = ['Noise level 0', 'Noise level 1', 'Noise level 2']
# for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
#     plt.plot(X[rgp.indices[k]], Y[rgp.indices[k]], 'o',color=c,
#              markersize = 9, label=nl[k])
    
# plt.plot(xNew, mu, linewidth=4, color='green', label='DDPSGP')
# plt.xlabel('x', fontsize=16)
# plt.ylabel('f(x)', fontsize=16)
# plt.legend(prop={"size":20})

# plt.show()