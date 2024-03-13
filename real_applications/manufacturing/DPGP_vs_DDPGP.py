import paths
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from models.DPGP import DirichletProcessGaussianProcess as DPGP
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
t_df = pd.read_excel(file, sheet_name='time')

# Pre-Process training data
X, y0, N0, D, max_lag, time_lags = dpm.align_arrays(X_df, y_df, t_df)

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

# Train and test data
N, D = np.shape(X)
start_train = y_df[y_df['Time stamp'] == '2020-08-15'].index[0]
end_train = y_df[y_df['Time stamp'] == '2020-08-30'].index[0]

X_train, y_train = X[start_train:end_train], y_raw[start_train:end_train]
X_test, y_test = X[start_train:end_train], y_raw[start_train:end_train]
N_train = len(y_train)

date_time = date_time[start_train:end_train]
y_raw = y_raw[start_train:end_train]
y_rect = y0[start_train:end_train]

"""
DPGP regression
"""
# Save memory
del X_df, y_df, dpm

# Length scales
ls = [7, 64, 7, 7.60, 7, 7, 7, 123, 76, 78]
# ls = [0.0612, 3.72, 200, 200, 200, 200, 4.35, 0.691, 200, 200]

# Kernels
se = 1**2 * RBF(length_scale=ls, length_scale_bounds=(0.5, 200))
wn = WhiteKernel(noise_level=0.61**2, noise_level_bounds=(1e-5, 1))

kernel = se + wn

dpgp = DPGP(X_train, y_train, init_K=7, kernel=kernel, plot_conv=True)
dpgp.train(pseudo_sparse=True)
# predictions
mu, std = dpgp.predict(X_test)

"""
DDPGP regression
"""
N_gps = 2
dgp = DDPGP(X_train, y_train, N_GPs=N_gps, init_K=7, kernel=kernel)
dgp.train(pseudo_sparse=True)
# predictions
mud, stdd, betas = dgp.predict(X_test)

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------
step = int(N_train/N_gps)
fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
fig.autofmt_xdate()

ax.fill_between(date_time,
                mu + 3*std, mu - 3*std,
                alpha=0.5, color='lightcoral',
                label='Confidence \nBounds (DPGP)')
ax.fill_between(date_time,
                mud + 3*stdd, mud - 3*stdd,
                alpha=0.5, color='grey',
                label='Confidence \nBounds (DPGP)')
ax.plot(date_time, y_raw, color='black', label='Raw')
ax.plot(date_time, y_rect, color='blue', label='Filtered')
ax.plot(date_time, mu, color="red", linewidth = 2.5, label="DPGP")
ax.plot(date_time, mud, color="orange", linewidth = 2.5, label="DDPGP")
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