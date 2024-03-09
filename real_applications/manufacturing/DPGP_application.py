import paths
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from models.DPGP import DirichletProcessGaussianProcess as DPGP
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

# Replace zero values with interpolation
zeros = y_raw_df.loc[y_raw_df['raw_furnace_faults'] <= 1e-1]
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

# Train and test data
N, D = np.shape(X)
# start_train = y_df[y_df['Time stamp'] == '2020-08-15'].index[0]
# end_train = y_df[y_df['Time stamp'] == '2020-08-30'].index[0]
model_N = 1
print('\n\n Model : ', model_N)
step = 2257
start_train = int(model_N - 1)*step
end_train = int(model_N*step)
end_test = end_train + 600
N_train = abs(end_train - start_train)

X_train, y_train = X[start_train:end_train], y_raw[start_train:end_train]
X_test, y_test = X[start_train:end_test], y_raw[start_train:end_test]

date_time = date_time[start_train:end_test]
y_raw = y_raw[start_train:end_test]
y_rect = y0[start_train:end_test]

"""
DPGP regression
"""
# Save memory
del X_df, y_df, dpm

# Length scales
# ls = [0.0612, 3.72, 200, 200, 200, 200, 4.35, 0.691, 200, 200]
# ls = [7, 64, 7, 7.60, 10, 7, 7, 123, 76, 78]
# ls = 1e4*np.ones(10)
ls = 800*np.ones(10)

# Kernels
se = 1**2 * RBF(length_scale=ls, length_scale_bounds=(0.1, 1e5))
wn = WhiteKernel(noise_level=0.61**2, noise_level_bounds=(1e-5, 1))

kernel = se + wn

dpgp = DPGP(X_train, y_train, init_K=7, kernel=kernel, DP_max_iter=300)
dpgp.train(pseudo_sparse=True)

# save trained model
import pickle

with open('expert'+str(model_N)+'.pkl','wb') as f:
    pickle.dump(dpgp,f)

# predictions
mu, std = dpgp.predict(X_test)

# The estimated GP hyperparameters
print('\nEstimated hyper DRGP: ', dpgp.kernel_)

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
                label='Confidence \nBounds (DPGP)')
ax.plot(date_time, y_raw, color='grey', label='Raw')
ax.plot(date_time, y_rect, color='blue', label='Filtered')
ax.plot(date_time, mu, color="red", linewidth = 2.5, label="DPGP")
plt.axvline(date_time[N_train-1], linestyle='--', linewidth=3,
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

#-----------------------------------------------------------------------------
# CLUSTERING PLOT
#-----------------------------------------------------------------------------

color_iter = ['lightgreen', 'orange','red', 'brown','black']

# DP-GP
enumerate_K = [i for i in range(dpgp.K_opt)]

fig, ax = plt.subplots()
# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

fig.autofmt_xdate()
ax.set_title(" Clustering performance", fontsize=18)
if dpgp.K_opt != 1:
    for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
        ax.plot(date_time[dpgp.indices[k]], y_raw[dpgp.indices[k]],
                'o',color=c, markersize = 8, label='Noise Level '+str(k))
ax.plot(date_time, mu, color="green", linewidth = 2, label=" DPGP")
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)
plt.show()