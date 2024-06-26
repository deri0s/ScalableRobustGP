import paths
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from models.DDPGP import DistributedDPGP as DDPGP
from real_applications.manufacturing.pre_processing import data_processing_methods as dpm
from sklearn.decomposition import PCA

"""
NSG data
"""
# NSG post processes data location
file = paths.get_nsg_path('processed/NSG_data.xlsx')

# Training df
X_df = pd.read_excel(file, sheet_name='X_training_stand')
y_df = pd.read_excel(file, sheet_name='y_training')
y_raw_df = pd.read_excel(file, sheet_name='y_raw_training')
timelags_df = pd.read_excel(file, sheet_name='time')

# Pre-Process training data
X, y0, N0, D, max_lag, time_lags = dpm.align_arrays(X_df, y_df, timelags_df)

# Replace zero values with interpolation
zeros = y_raw_df.loc[y_raw_df['raw_furnace_faults'] < 1e-2]
y_raw_df['raw_furnace_faults'][zeros.index] = None
y_raw_df.interpolate(inplace=True)

# Process raw targets
# Just removes the first max_lag points from the date_time array.
y_raw = dpm.adjust_time_lag(y_raw_df['raw_furnace_faults'].values,
                            shift=0,
                            to_remove=max_lag)

# Extract corresponding time stamps. Note this essentially just
# removes the first max_lag points from the date_time array.
date_time = dpm.adjust_time_lag(y_df['Time stamp'].values,
                                shift=0,
                                to_remove=max_lag)

# temp drop noisy region
start_drop = y_df[y_df['Time stamp'] == '2020-08-30 00:00:00'].index[0]
end_drop = y_df[y_df['Time stamp'] == '2020-09-07 00:20:00'].index[0]
remove = np.arange(start_drop, end_drop)

X = np.delete(X, remove, 0)
y_raw = np.delete(y_raw, remove, 0)
date_time = np.delete(date_time, remove, 0)
y0 = np.delete(y0, remove, 0)

"""
READ

Model parameters and initialisation values from the config.yml file
"""

with open(paths.get_config_path('config.yml'), 'r') as f:
    config = yaml.safe_load(f)

# Train and test data
step = config['N_local']
N_gps = config['NGPs']
N, D = np.shape(X)

start_train = 0
end_train = start_train + int(N_gps * step)
end_test = end_train + 400

X_train, y_train = X[start_train:end_train], y_raw[start_train:end_train]
X_test, y_test = X[start_train:end_test], y_raw[start_train:end_test]
N_train = len(y_train)

date_time = date_time[start_train:end_test]
y_raw = y_raw[start_train:end_test]
y_rect = y0[start_train:end_test]


# read initial hyperparameters
length_scale = config['length_scale']
noise = config['noise']

# array of kernel with the predetermined initialisation values
kernels = []
for k in range(N_gps):
    se = 1**2 * RBF(length_scale=length_scale['initial'][k],
                    length_scale_bounds=(length_scale['bounds'][k][0],
                                         length_scale['bounds'][k][1]))
    wn = WhiteKernel(noise_level=noise['level'][k]**2,
                     noise_level_bounds=(noise['bounds'][k][0],
                                         noise['bounds'][k][1]))
    kernels.append(se + wn)
    # print('ls-bounds: ', noise['bounds'][k][0], noise['bounds'][k][1])

"""
DPGP regression
"""
# Save memory
del X_df, y_df, dpm

dpgp = DDPGP(X_train, y_train, N_GPs=N_gps,
             init_K=config['DP_hyper']['init_K'],
             DP_max_iter=config['DP_hyper']['DP_max_iter'],
             kernel=kernels,
             normalise_y=True, plot_expert_pred=True)
dpgp.train(pseudo_sparse=True)

# # save trained model
# import pickle

# with open('DDPGP_NGPs1_config1.pkl','wb') as f:
#     pickle.dump(dpgp,f)

# predictions
mu, std, betas = dpgp.predict(X_test)

#-----------------------------------------------------------------------------
# Plot beta
#-----------------------------------------------------------------------------

step = int(len(X_train)/N_gps)
fig, ax = plt.subplots()
fig.autofmt_xdate()
for k in range(N_gps):
    ax.plot(date_time, betas[:,k], color=dpgp.c[k], linewidth=2,
            label='Beta: '+str(k))
    plt.axvline(date_time[int(k*step)], linestyle='--', linewidth=2,
                color='black')

plt.axvline(date_time[N_train-1], linestyle='--', linewidth=3,
            color='limegreen', label='<- train \n-> test')
ax.set_title('Predictive contribution of robust GP experts')
ax.set_xlabel('Date-time')
ax.set_ylabel('Predictive contribution')
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------

fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
fig.autofmt_xdate()
ax.fill_between(date_time,
                mu + 3*std, mu - 3*std,
                alpha=0.5, color='pink',
                label='Confidence \nBounds (DRGPs)')
ax.plot(date_time, y_raw[0:N], color='grey', label='Raw')
ax.plot(date_time, mu, color="red", linewidth = 2.5, label="DRGPs")

# Plot the limits of each expert
for s in range(N_gps):
    plt.axvline(date_time[int(s*step)], linestyle='--', linewidth=2,
                color='black')
    
plt.axvline(date_time[-1], linestyle='--', linewidth=3,
            color='black')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

# ----------------------------------------------------------------------------
# PCA and PLOTS
# ----------------------------------------------------------------------------
pca = PCA(n_components=2)
pca.fit(X)
Xt = pca.transform(X)

# PCA on training data
Xt_train = pca.transform(X_train)

# PCA on test data
Xt_test = pca.transform(X_test)
    
# Plot at each 1000 points
fig, ax = plt.subplots()
ax.plot(Xt[:, 0], Xt[:, 1], 'o', markersize=0.9, c='grey',
        label='Available training data', alpha=0.9)
ax.plot(Xt_train[:, 0], Xt_train[:, 1], 'o', markersize=8.9, c='orange',
        label='Used Training data', alpha=0.6)
ax.plot(Xt_test[:,0], Xt_test[:,1], '*', markersize=5.5,
        c='purple', label='test data', alpha=0.6)
ax.set_xlim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 0])))
ax.set_ylim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 1])))
plt.show()