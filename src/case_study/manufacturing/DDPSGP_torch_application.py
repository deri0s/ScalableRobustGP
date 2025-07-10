import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from models.gPoE_torch import DistributedSVGP as DSGP
from case_study.manufacturing.preprocessing import data_processing_methods as dpm
from sklearn.decomposition import PCA

"""
NSG data
"""
NGPs = 5
ROOT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT_PATH / "manufacturing" / "data" / "processed"
EXPERT_PATH = ROOT_PATH / "manufacturing" / "trained" / "experts"

file = os.path.join(PROCESSED_PATH / 'NSG_processed_data.xlsx')

# Training df
X_df = pd.read_excel(file, sheet_name='X_stand')
y_df = pd.read_excel(file, sheet_name='y')
y_raw_df = pd.read_excel(file, sheet_name='y_raw')
timelags_df = pd.read_excel(file, sheet_name='timelags')

# Pre-Process training data
X, y0, N0, D, max_lag, time_lags = dpm.align_arrays(X_df, y_df, timelags_df)

# Convert data to torch tensors
floating_point = torch.float64
X = torch.tensor(X, dtype=floating_point)

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
end_indx = int(len(X)*0.8)
end_train = N - end_indx

X_train = X[0:end_train]

dgp = DSGP(EXPERT_PATH, N_GPs=5, device='cpu', plot_expert_pred=True)

dgp.predict_with_uncertainty(X)

# predictions
mu, std, betas = dgp.predict(X)

#-----------------------------------------------------------------------------
# Plot beta
#-----------------------------------------------------------------------------

step = int(len(X_train)/NGPs)
fig, ax = plt.subplots()
fig.autofmt_xdate()
for k in range(NGPs):
    ax.plot(date_time, betas[:,k], color=dgp.c[k], linewidth=2,
            label='Beta: '+str(k))
    plt.axvline(date_time[int(k*step)], linestyle='--', linewidth=2,
                color='black')

plt.axvline(date_time[end_train-1], linestyle='--', linewidth=3,
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
for s in range(NGPs):
    plt.axvline(date_time[int(s*step)], linestyle='--', linewidth=2,
                color='black')
    
plt.axvline(date_time[-1], linestyle='--', linewidth=3,
            color='black')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

# # ----------------------------------------------------------------------------
# # PCA and PLOTS
# # ----------------------------------------------------------------------------
# pca = PCA(n_components=2)
# pca.fit(X)
# Xt = pca.transform(X)

# # PCA on training data
# Xt_train = pca.transform(X_train)

# # PCA on test data
# Xt_test = pca.transform(X_test)
    
# # Plot at each 1000 points
# fig, ax = plt.subplots()
# ax.plot(Xt[:, 0], Xt[:, 1], 'o', markersize=0.9, c='grey',
#         label='Available training data', alpha=0.9)
# ax.plot(Xt_train[:, 0], Xt_train[:, 1], 'o', markersize=8.9, c='orange',
#         label='Used Training data', alpha=0.6)
# ax.plot(Xt_test[:,0], Xt_test[:,1], '*', markersize=5.5,
#         c='purple', label='test data', alpha=0.6)
# ax.set_xlim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 0])))
# ax.set_ylim(np.min(Xt[:, 0]), np.max(np.max(Xt[:, 1])))
plt.show()