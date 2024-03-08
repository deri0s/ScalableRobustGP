import paths
import numpy as np
import pandas as pd
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

model_N = 16
print('\n\n Model : ', model_N)
step = 1128
start_train = 0
end_train = int(model_N*step)
end_test = N

X_train, y_train = X[start_train:end_train], y_raw[start_train:end_train]
X_test, y_test = X[start_train:end_test], y_raw[start_train:end_test]
N_train = len(y_train)

date_time = date_time[start_train:end_test]
y_raw = y_raw[start_train:end_test]
y_rect = y0[start_train:end_test]

"""
LOAD DDPGP
"""
import pickle

with open('ddpgp_NGPs'+str(model_N)+'.pkl','rb') as f:
    ddpgp = pickle.load(f)

"""
DELETE AND SAVE
"""
# delete sixth expert
ddpgp.delete(6)

with open('ddpgp_NGPs'+str(model_N)+'_deleted6th.pkl','wb') as f:
    pickle.dump(ddpgp,f)

N_gps = model_N

# predictions
mu, std, betas = ddpgp.predict(X_test)

#-----------------------------------------------------------------------------
# Plot beta
#-----------------------------------------------------------------------------

fig, ax = plt.subplots()
fig.autofmt_xdate()
for k in range(N_gps-2):
    ax.plot(date_time, betas[:,k], color=ddpgp.c[k], linewidth=2,
            label='Beta: '+str(k))
    plt.axvline(date_time[int(k*step)], linestyle='--', linewidth=2,
                color='black')

plt.axvline(date_time[-1], linestyle='--', linewidth=3,
            color='black', label='<- train \n-> test')
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
                alpha=0.5, color='lightcoral',
                label='Confidence \nBounds (DPGP)')
ax.plot(date_time, y_raw, color='grey', label='Raw')
ax.plot(date_time, y_rect, color='blue', label='Filtered')
ax.plot(date_time, mu, color="red", linewidth = 2.5, label="DDPGP")
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

# Plot the limits of each expert
for s in range(N_gps):
    plt.axvline(date_time[int(s*step)], linestyle='--', linewidth=2,
                color='black')

plt.axvline(date_time[N_train-1], linestyle='--', color='black',
            label='<- train \n-> test')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

plt.show()