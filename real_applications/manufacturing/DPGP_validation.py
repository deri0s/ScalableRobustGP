import paths
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from models.DPGP import DirichletProcessGaussianProcess as DPGP
from real_applications.manufacturing.pre_processing import data_processing_methods as dpm

"""
MODEL VALIDATION
----------------
Glass experts have identified a region containing data that have been 
genrated by the melting process.
From the glass experts, we know that:

- Increases in fault density that last between 2 hours and 4 days are most
  likely generated by the melting process.
- Area of interest ranges from: 2020-08-15 to: 2020-08-30
- Furnace faults increses identified at:
    1) From 2020-08-20 to 2020-08-23
    2) From 2020-08-24 to 2020-08-27
"""

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

# Train and test data
N, D = np.shape(X)
start_train = y_df[y_df['Time stamp'] == '2020-08-15'].index[0]
end_train = y_df[y_df['Time stamp'] == '2020-08-30'].index[0]
N_train = abs(end_train - start_train)

X_train, y_train = X[start_train:end_train], y_raw[start_train:end_train]
X_test, y_test = X[start_train:end_train], y_raw[start_train:end_train]

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

# Kernels
se = 1**2 * RBF(length_scale=ls, length_scale_bounds=(0.5, 300))
wn = WhiteKernel(noise_level=0.61**2, noise_level_bounds=(1e-5, 1))

kernel = se + wn

dpgp = DPGP(X_train, y_train, init_K=7, kernel=kernel)
dpgp.train(pseudo_sparse=True)

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
                alpha=0.5, color='lightcoral',
                label='Confidence \nBounds (DPGP)')
ax.plot(date_time, y_raw, color='grey', label='Raw')
ax.plot(date_time, y_rect, color='blue', label='Filtered')
ax.plot(date_time, mu, color="red", linewidth = 2.5, label="DPGP")
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

#-----------------------------------------------------------------------------
# CLUSTERING PLOT
#-----------------------------------------------------------------------------

# processes colors
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
                'o',color=c, markersize = 8, label='Noise level '+str(k))
ax.plot(date_time, mu, color="green", linewidth = 2, label="DPGP")
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)
plt.show()