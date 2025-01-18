import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler as ss
from case_study.manufacturing.data_and_preprocessing.raw import data_processing_methods as dpm

"""
NSG data
"""
# NSG post processes data location
file_val = 'validation_data.xlsx'
file = 'data_and_preprocessing/processed/NSG_processed_data_14_inputs.xlsx'

# Training df
X_df = pd.read_excel(file, sheet_name='X_stand')
y_df = pd.read_excel(file, sheet_name='y')
y_raw_df = pd.read_excel(file, sheet_name='y_raw')
t_df = pd.read_excel(file, sheet_name='timelags')

# validation df
val_df = pd.read_excel(file_val)
date_timev = val_df.date_time.values
val_mu = val_df.gp_pred.values

# get data on the region of interest (valitation data)
first_date = val_df.date_time[0]
last_date = val_df.date_time[len(val_df.date_time)-1]

first = y_df[y_df['Time stamp'] == first_date].index[0]
last  = y_df[y_df['Time stamp'] == last_date].index[0]

X_df = X_df.iloc[first:last, :]
y_df = y_df.iloc[first:last, :]

# Pre-Process training data
X, y0, N0, D, max_lag, time_lags = dpm.align_arrays(X_df, y_df, t_df)

print('max-lag: ', max_lag)
print('y0: ', len(y0), ' 20%: ', int(len(y0)*0.25), ' 3 days: 216')

# Replace zero values with interpolation
zeros = y_raw_df.loc[y_raw_df['raw_furnace_faults'] <= 1e-1]
y_raw_df.loc[zeros.index, 'raw_furnace_faults'] = None
y_raw_df.interpolate(inplace=True)

# Remove the first max_lag points (the same as align_arrays)
y_raw = dpm.adjust_time_lag(y_raw_df['raw_furnace_faults'].values,
                            shift=0,
                            to_remove=max_lag)

date_time = dpm.adjust_time_lag(y_df['Time stamp'].values,
                                shift=0,
                                to_remove=max_lag)

# Train and test data
N, D = np.shape(X)
end_train = N - int(len(y0)*0.25)

X_train, y_train = X[0:end_train], y_raw[0:end_train]
date_train = date_time[0:end_train]
N_train = len(X_train)

X_test = X[0:N]
date_time = date_time[0:N]
y_raw = y_raw[0:N]
y_rect = y0[0:N]

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------

fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
fig.autofmt_xdate()

print(type(date_timev[0]))
print(type(date_time[0]))

ax.plot(date_timev, val_mu, color='green', label='Val')
ax.plot(date_train, y_train, color='grey', label='Raw')
ax.plot(date_time, y_rect, color='blue', label='Filtered')
plt.axvline(date_time[end_train-1], linestyle='--', linewidth=3,
            color='black')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

plt.show()