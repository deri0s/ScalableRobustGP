import paths
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
from real_applications.manufacturing.pre_processing import data_processing_methods as dpm

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

"""
TRAINING AND TESTING DATA
"""
current_path = Path(__file__).resolve()
trained_path = current_path.parents[1] / 'trained'

# Train and test data
N, D = np.shape(X)
N_train = int(N*0.84)

test_range = range(0, N)
X_test = X[test_range]
dt_test, y_test = date_time[test_range], y0[test_range]

"""
Scalable Robust GP
"""

gp_config1_path = trained_path / 'results/DDPGP_NGPs6_config1_predictions.csv'
df = pd.read_csv(gp_config1_path)

# identify regions where the DDPGP model is not confident
uncertain = df.mu[df.mu.isna()].index

# # set nan values for mu and std to zero and 100, respectively 
df.interpolate(method='zero', inplace=True)
mu = df.mu
std = df['std']
N_gps = 6
betas = np.zeros((len(df), N_gps))

for k in range(3, len(df.columns)):
    betas[:, k-N_gps] = df.iloc[:, k]

# print('mean:', np.mean(std), ' max: ', np.max(std), ' min: ', np.min(std))

#-----------------------------------------------------------------------------
# Plot beta
#-----------------------------------------------------------------------------

step = int(len(mu)/N_gps)
fig, ax = plt.subplots()
fig.autofmt_xdate()
for k in range(N_gps):
    ax.plot(date_time, betas[:,k], linewidth=2,
            label='Beta: '+str(k))
    plt.axvline(date_time[int(k*step)], linestyle='--', linewidth=2,
                color='black')

ax.set_title('Predictive contribution of robust GP experts - NGPs-4')
ax.set_xlabel('Date-time')
ax.set_ylabel('Predictive contribution')
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------

# Region where test data is similar to the training data
similar = range(21000,21500)

fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

fig.autofmt_xdate()
ax.axvline(dt_test[N_train], linestyle='--', linewidth=3,
           color='red', label='-> test data')
ax.fill_between(dt_test,
                mu + 3*std, mu - 3*std,
                alpha=0.5, color='lightcoral',
                label='Confidence \nBounds (DRGPs)')
ax.plot(date_time, y_raw, color="grey", linewidth = 2.5, label="Raw")
ax.plot(dt_test, y_test, color="blue", linewidth = 2.5, label="Processed")
ax.plot(dt_test, mu, color="red", linewidth = 2.5, label="DDPGP")
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":12}, facecolor="white", framealpha=1.0)

plt.show()