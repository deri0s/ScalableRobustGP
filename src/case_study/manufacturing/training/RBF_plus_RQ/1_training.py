import torch
import gpytorch
import pandas as pd
import numpy as np
import yaml
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler as ss
from matplotlib import pyplot as plt
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import InducingPointKernel, ScaleKernel, RBFKernel as RBF, RQKernel as RQ

"""
NSG data
"""

file = '../validation_data_main.xlsx'

# Training df
X_df = pd.read_excel(file, sheet_name='X_stand')
y_df = pd.read_excel(file, sheet_name='y_nonstand')
t_df = pd.read_excel(file, sheet_name='timelags')

# drop Tewwl position
X_df.drop(columns=['9282 Tweel Position'], inplace=True)
t_df.drop(columns=['9282 Tweel Position'], inplace=True)

# Pre-Process training data
N, D = np.shape(X_df.values)

# Create tag inputs
X = np.zeros([N, D])

"""---------------------------------------------------------------------------
    CREATE LAGGED FEATURES
"""

def align_inputs(x_df, y_df, t_series):
    # ! Always close/Deep copy
    xdeep = x_df.copy()
    ydeep = y_df.copy()
    max_lag = int(max(t_series))

    # X
    for name, lag in t_series.items():
        xdeep[name] = xdeep[name].shift(int(lag))
    xdeep.dropna(inplace=True)

    # y and date-time
    ydeep = ydeep.iloc[max_lag:].reset_index(drop=True)

    return xdeep.reset_index(drop=True), ydeep
 
X_df, y_df = align_inputs(X_df, y_df, t_df.iloc[0,:])

"""---------------------------------------------------------------------------
    STANDARDISE TRAINING & TEST DATA
"""

X = X_df.values
y_nonstand, date_time = y_df.gp_pred.values, y_df.date_time.values

N, D = np.shape(X)
test_perc = 0.12
end_train = N - int(len(y_nonstand)*test_perc)

X_train_np = X[0:end_train]
date_train = date_time[0:end_train]
N_train = len(X_train_np)
y_train_nonstand = y_nonstand[0:end_train]

X_test = X[0:N]
date_time = date_time[0:N]

# Standardise outputs
y_train = y_train_nonstand.reshape(-1,1)
scaler = ss()
scaler.fit(y_train)
y_norm_np = scaler.transform(y_train)

# Convert data to torch tensors
floating_point = torch.float64
X_train = torch.tensor(X_train_np, dtype=floating_point)
y_train = torch.tensor(y_norm_np, dtype=floating_point).squeeze()
X_test = torch.tensor(X_test, dtype=floating_point)

"""----------------------------------------------------------------------------
Sparse GP
"""

class SparseGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel, noise_var):
        super(SparseGP, self).__init__(train_x, train_y, likelihood)
        likelihood.noise = noise_var
        self.mean_module = ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
# ! Always clone
step = 40
inducing_points = X_train[::step, :].clone()

# Model
likelihood = GaussianLikelihood()
se = ScaleKernel(RBF(ard_num_dims=D) + RQ(ard_num_dims=D))
covar_module = InducingPointKernel(se,
                                   inducing_points=inducing_points,
                                   likelihood=likelihood)
N_sim = 6000
init_os = []
init_ls = []
init_nv = []
init_ls_rq = []
init_alpha = []
os_list = []
ls_list = []
nv_list = []
ls_rq_list = []
alpha_list = []
mse_list = []
random = True
kconfig = '_RBF_plus_RQ_FT'

# read initial hyperparameters
df_hyper = pd.read_excel('../hyper_results.xlsx')

ls_temp_list = []
df_hyper.iloc[5, int(np.shape(df_hyper)[1]-1)]
for d in range(D):
    lower = df_hyper[t_df.columns[0]][7]
    upper = df_hyper[t_df.columns[0]][8]
    ls_temp_list.append(np.random.uniform(low=lower, high=upper))

print(ls_temp_list)

# for i in range(N_sim):
#     print(f'Hyperparameter simulation: {i}/{N_sim}')

#     if random:
#         os = np.random.uniform(low=100, high=700)
#         ls = np.random.uniform(low=0.5, high=95, size=D)
#         ls_rq= np.random.uniform(low=0.1, high=95, size=D)
#         alpha = np.random.uniform(low=0.01, high=2)
#         nv = np.random.uniform(low=0.04, high=0.085)
#         # save initial hyperparameters
#         init_os.append(os)
#         init_ls.append(ls)
#         init_nv.append(nv)
#         init_ls_rq.append(ls_rq)
#         init_alpha.append(alpha)
#     else:
#         # save initial hyperparameters
#         init_os.append(os)
#         init_ls.append(ls.squeeze(0).detach().numpy())
#         init_ls_rq.append(ls_rq.squeeze(0).detach().numpy())
#         init_alpha.append(alpha)
#         init_nv.append(nv)

#     # GP object ! check if the outputscale in base_kernel.base_kernel?
#     gp = SparseGP(X_train, y_train, likelihood, covar_module, nv)
#     gp.covar_module.base_kernel.outputscale = os
#     gp.covar_module.base_kernel.base_kernel.kernels[0].lengthscale = ls
#     gp.covar_module.base_kernel.base_kernel.kernels[1].lengthscale = ls_rq
#     gp.covar_module.base_kernel.base_kernel.kernels[1].alpha = alpha

#     # Train model
#     gp.train()
#     gp.likelihood.train()

#     optimizer = torch.optim.Adam(gp.parameters(), lr=0.01)
#     mll = ExactMarginalLogLikelihood(likelihood, gp)

#     training_iterations = 100
#     for count in range(training_iterations):
#         optimizer.zero_grad()
#         output = gp(X_train)
#         loss = -mll(output, y_train)
#         loss.backward()
#         optimizer.step()

#     # get the estimated hyperparameters
#     os   = gp.covar_module.base_kernel.outputscale.item()
#     ls   = gp.covar_module.base_kernel.base_kernel.kernels[0].lengthscale
#     ls_rq= gp.covar_module.base_kernel.base_kernel.kernels[1].lengthscale
#     alpha= gp.covar_module.base_kernel.base_kernel.kernels[1].alpha.item()
#     nv = likelihood.noise.item()

#     # Predictions
#     gp.eval()
#     likelihood.eval()
#     with torch.no_grad(), gpytorch.settings.fast_pred_var():
#         observed_pred = likelihood(gp(X_test))

#         # Unormalise predictions
#         pred_mean = observed_pred.mean
#         mu = scaler.inverse_transform(pred_mean.unsqueeze(1))[:,0]
#         mse = mean_squared_error(mu, y_nonstand)
#         # mse = mean_squared_error(mu[end_train:-1], y_nonstand[end_train:-1])

#     # collect results
#     os_list.append(os)
#     ls_list.append(ls.squeeze(0).detach().numpy())
#     ls_rq_list.append(ls_rq.squeeze(0).detach().numpy())
#     nv_list.append(nv)
#     alpha_list.append(alpha)
#     mse_list.append(mse)

#     # check error
#     if i > 1:
#         if mse_list[i] < mse_list[i-1]:
#             random = False
#         else:
#             random = True

# d = {'step': step,
#      'init_os': init_os, 'init_ls': init_ls, 'init_nv': init_nv,
#      'init_ls_rq': init_ls_rq, 'init_alpha': init_alpha,
#      'outputscale': os_list,
#      'lengthscale': ls_list,
#      'noise_var': nv_list,
#      'lengthscale_RQ': ls_rq_list,
#      'alpha': alpha_list,
#      'mse': mse_list}

# df_sim = pd.DataFrame(d)

# print('lowest errors \n', df_sim.mse.sort_values()[0:3], '\n')

# indx = df_sim[df_sim.mse == df_sim.mse.min()].index

# init_opt_os = df_sim.init_os[indx].values[0]
# init_opt_ls = df_sim.init_ls[indx].values[0]
# init_opt_nv = df_sim.init_nv[indx].values[0]
# init_opt_ls_rq = df_sim.init_ls_rq[indx].values[0]
# init_opt_alpha = df_sim.init_alpha[indx].values[0]
# opt_os = df_sim.outputscale[indx].values[0]
# opt_ls = df_sim.lengthscale[indx].values[0]
# opt_nv = df_sim.noise_var[indx].values[0]
# opt_ls_rq = df_sim.lengthscale_RQ[indx].values[0]
# opt_alpha = df_sim.alpha[indx].values[0]
# mse_test = df_sim.mse[indx].values[0]

# # GP object
# gp = SparseGP(X_train, y_train, likelihood, covar_module, init_opt_nv)
# gp.covar_module.base_kernel.outputscale = init_opt_os
# gp.covar_module.base_kernel.base_kernel.kernels[0].lengthscale = init_opt_ls
# gp.covar_module.base_kernel.base_kernel.kernels[1].lengthscale = init_opt_ls_rq
# gp.covar_module.base_kernel.base_kernel.kernels[1].alpha = init_opt_alpha

# # Train model
# gp.train()
# gp.likelihood.train()

# optimizer = torch.optim.Adam(gp.parameters(), lr=0.01)
# mll = ExactMarginalLogLikelihood(likelihood, gp)

# training_iterations = 100
# for count in range(training_iterations):
#     optimizer.zero_grad()
#     output = gp(X_train)
#     loss = -mll(output, y_train)
#     loss.backward()
#     optimizer.step()

# # *Induced points
# init_z_indices = np.arange(0, len(X_train.numpy()), step)

# # Make sure the _z (induced inputs) are a subset of the X_train dataset
# _z = gp.covar_module.inducing_points.detach()

# _z_indices = []
# for z in _z:
#     distances = torch.norm(X_train - z, dim=1)
#     closest_index = torch.argmin(distances).item()
#     _z_indices.append(closest_index)

# # check the z0 and z* are not the same
# assert ~np.all(list(init_z_indices == _z_indices)), 'induced not trained'

# # Predictions
# gp.eval()
# likelihood.eval()
# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     observed_pred = likelihood(gp(X_test))

# # Unormalise predictions
# pred_mean = observed_pred.mean
# mu = scaler.inverse_transform(pred_mean.unsqueeze(1))[:,0]
# stds = scaler.inverse_transform(observed_pred.stddev.unsqueeze(1))[:,0]
# lower_stand, upper_stand = observed_pred.confidence_region()
# lower = scaler.inverse_transform(lower_stand.unsqueeze(1))[:,0]
# upper = scaler.inverse_transform(upper_stand.unsqueeze(1))[:,0]

# mse_full = mean_squared_error(mu, y_nonstand)
# print('MSE (test) in loop: ', df_sim.mse[indx].values)
# print('MSE: (train-test):', mse_full)

# """--------------------------------------------------------------------------
# SAVE OPTIMAL CONFIGURATION
# """
# torch.save(gp.state_dict(), 'gp_state'+kconfig+'step'+str(step)+'.pth')

# # Save hyperparameters in a YAML file just in case
# def convert_numpy(obj):
#     """Recursively convert NumPy arrays and scalars to Python-native types."""
#     if isinstance(obj, (np.generic, np.number)):  
#         return obj.item()  # Convert NumPy scalar (float32, float64, int32, int64) to Python scalar
#     elif isinstance(obj, np.ndarray):  
#         return obj.astype(float).tolist()  # Convert array to list of Python floats
#     elif isinstance(obj, dict):  
#         return {k: convert_numpy(v) for k, v in obj.items()}  # Recursively process dict
#     elif isinstance(obj, list):  
#         return [convert_numpy(i) for i in obj]  # Recursively process list
#     return obj  # Return unchanged if not a NumPy type

# d = {
#     'step': step,
#     'N': N,
#     'N_train': N_train,
#     'test_percentage': test_perc,
#     'date': {'start': str(date_time[0]), 'end': str(date_time[-1])},
#     'kernel_equation': 'InducingPoint( Scale(RBF + RQ) ) + WN(in likelihood)',
#     'outputscale': {'initial': init_opt_os, 'optimal': opt_os},
#     'RBF': {
#         'lengthscale': {'initial': init_opt_ls, 'optimal': opt_ls}
#     },
#     'RQ': {
#         'lengthscale': {'initial': init_opt_ls_rq, 'optimal': opt_ls_rq},
#         'alpha': {'initial': init_opt_alpha, 'optimal': opt_alpha}
#     },
#     'WN': {
#         'var': {'initial': init_opt_nv, 'optimal': opt_nv}
#     },
#     'mse': {'full': mse_full, 'test': mean_squared_error(mu[end_train:-1],
#                                                          y_nonstand[end_train:-1])}
# }

# # Convert NumPy objects to Python-native types
# d_converted = convert_numpy(d)

# # Dump to YAML
# with open('config'+kconfig+'step'+str(step)+'.yaml', 'w') as file:
#     yaml.safe_dump(d_converted, file, default_flow_style=False)

# print("\nData successfully written")

# """--------------------------------------------------------------------------
# PLOT
# """

# plt.figure()
# plt.plot(mse_list)
# plt.xlabel('iteration')
# plt.xlabel('MSE')

# #-----------------------------------------------------------------------------
# # REGRESSION PLOT
# #-----------------------------------------------------------------------------
# fig, ax = plt.subplots()

# # Increase the size of the axis numbers
# plt.rcdefaults()
# plt.rc('xtick', labelsize=14)
# plt.rc('ytick', labelsize=14)
# fig.autofmt_xdate()

# plt.fill_between(date_time, lower, upper,
#                 alpha=0.5, color='lightcoral',
#                 label='2$\\sigma$')
# ax.plot(date_time, y_nonstand, '*', color='green', label='Val')
# ax.plot(date_time, mu, color='red', label='GP')
# plt.axvline(date_time[end_train-1], linestyle='--', linewidth=3,
#         color='black')
# ax.set_xlabel(" Date-time", fontsize=14)
# ax.set_ylabel(" Fault density", fontsize=14)
# plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

# ax.vlines(
#     x=date_time[::step],
#     ymin=-2*stds.min(),
#     ymax=y_train.max().item(),
#     alpha=0.3,
#     linewidth=1.5,
#     ls='--',
#     label="z0",
#     color='grey'
# )

# # Induced points
# ax.vlines(
#     x=date_time[_z_indices],
#     ymin=-2*stds.min(),
#     ymax=y_train.max().item(),
#     alpha=0.4,
#     linewidth=1.5,
#     label="z*",
#     color='orange'
# )
# ax.set_xlabel(" Date-time", fontsize=14)
# ax.set_ylabel(" Fault density", fontsize=14)
# plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)
# plt.show()