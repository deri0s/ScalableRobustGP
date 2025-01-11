import pandas as pd
import numpy as np
import time
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler as ss
from case_study.manufacturing.data_and_preprocessing.raw import data_processing_methods as dpm
from sklearn.decomposition import PCA

"""
NSG data
"""
# NSG post processes data location
# file = 'data_and_preprocessing/processed/NSG_processed_data.xlsx'
file = 'data_and_preprocessing/processed/NSG_processed_data_14_inputs.xlsx'

# Training df
X_df = pd.read_excel(file, sheet_name='X_stand')
y_df = pd.read_excel(file, sheet_name='y')
y_raw_df = pd.read_excel(file, sheet_name='y_raw')
t_df = pd.read_excel(file, sheet_name='timelags')

# Pre-Process training data
X, y0, N0, D, max_lag, time_lags = dpm.align_arrays(X_df, y_df, t_df)

# Replace zero values with interpolation
zeros = y_raw_df.loc[y_raw_df['raw_furnace_faults'] <= 1e-1]
y_raw_df.loc[zeros.index, 'raw_furnace_faults'] = None
y_raw_df.interpolate(inplace=True)

# Remove the first max_lag points (the same as align_arrays)
y_raw = dpm.adjust_time_lag(y_raw_df['raw_furnace_faults'].values,
                            shift=0,
                            to_remove=max_lag)

date_time0 = y_df['Time stamp'].values

date_time = dpm.adjust_time_lag(y_df['Time stamp'].values,
                                shift=0,
                                to_remove=max_lag)

# Train and test data
N, D = np.shape(X)
# '2020-08-14
start_train = y_df[y_df['Time stamp'] == '2020-07-25-10'].index[0]
end_train = y_df[y_df['Time stamp'] == '2020-08-27-14'].index[0]

X_train, y_train = X[start_train:end_train], y_raw[start_train:end_train]
N_train = len(X_train)

caca = y_df[y_df['Time stamp'] == '2020-08-29'].index[0] - start_train
print('start-train: ', start_train, 'end-train: ', end_train,
      ' caca: ', caca, ' caca + start-train: ', start_train + caca,
      ' length: ', len(date_time))

end_test1 = start_train + caca
date_time1 = date_time[start_train:start_train + caca]
X_test = X[start_train:end_test1]
date_time = date_time[start_train:end_test1]
y_raw = y_raw[start_train:end_test1]
y_rect = y0[start_train:end_test1]

# end_test = end_train + 200
# X_test = X[start_train:end_test]
# date_time = date_time[start_train:end_test]
# y_raw = y_raw[start_train:end_test]
# y_rect = y0[start_train:end_test]

# #-----------------------------------------------------------------------------
# # REGRESSION PLOT
# #-----------------------------------------------------------------------------

fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
fig.autofmt_xdate()

ax.plot(date_time, y_raw, color='grey', label='Raw')
ax.plot(date_time, y_rect, color='blue', label='Filtered')
plt.axvline(date_time[N_train-1], linestyle='--', linewidth=3,
            color='black')
# plt.axvline(date_time[N_train+150-1], linestyle='--', linewidth=3,
#             color='black')
# plt.axvline(date_time[start_train + caca], linestyle='--', linewidth=3,
#             color='orange', label='end-date')

ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

plt.show()