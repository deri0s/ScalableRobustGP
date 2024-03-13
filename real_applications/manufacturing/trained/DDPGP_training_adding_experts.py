import paths
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.DDPGP import DistributedDPGP as DDPGP
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
y_raw0 = dpm.adjust_time_lag(y_raw_df['raw_furnace_faults'].values,
                            shift=0,
                            to_remove=max_lag)

# Extract corresponding time stamps. Note this essentially just
# removes the first max_lag points from the date_time array.
date_time = dpm.adjust_time_lag(y_df['Time stamp'].values,
                                shift=0,
                                to_remove=max_lag)

# Train and test data
N, D = np.shape(X)

start_train = y_df[y_df['Time stamp'] == '2020-08-30'].index[0]
end_train = y_df[y_df['Time stamp'] == '2020-09-09'].index[0]
X_train, y_train = X[start_train:end_train], y_raw0[start_train:end_train]
N_train = len(y_train)

X_test, y_test = X[0:N], y_raw0[0:N]
y_raw = y_raw0[0:N]
date_time = date_time[0:N]

"""
LOAD DPGP expert
"""
import glob
import pickle
from pathlib import Path
current_path = Path(__file__).resolve()
trained_path = current_path.parents[1] / 'trained'
dpgp_path = trained_path / 'DPGPs/*.pkl'

# Create a list of all the file names in the DPGPs folder
file_paths = glob.glob(str(dpgp_path))

expert = []
for k in file_paths:
    with open(k,'rb') as f:
        expert.append(pickle.load(f))

# add trained expert
ddpgp = DDPGP(X_train, y_train)
ddpgp.add(expert)
N_gps = ddpgp.N_GPs
print('N-GPs: ', N_gps)

"""
SAVE DDPGP MODEL
"""
ddpgp_path = trained_path / 'DDPGPs'
with open(str(ddpgp_path) + '/ddpgp_NGPs'+str(N_gps)+'_assembled.pkl','wb') as f:
    pickle.dump(ddpgp,f)

# predictions
mu, std, betas = ddpgp.predict(X_test)

# save to CVS file
d = {'date-time': date_time, 'mu': mu, 'std': std}

# save betas
for k in range(N_gps):
    d['beta'+str(k+1)] = betas[:,k]

df = pd.DataFrame(d)
df.to_csv(str(trained_path) + '/results/ddpgp_NGPs'+str(N_gps)+'_assembled.csv',
          index=False)

#-----------------------------------------------------------------------------
# Plot beta
#-----------------------------------------------------------------------------

fig, ax = plt.subplots()
fig.autofmt_xdate()
for k in range(N_gps):
    ax.plot(date_time, betas[:,k], color=ddpgp.c[k], linewidth=2,
            label='Beta: '+str(k))
    
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
ax.plot(date_time, mu, color="red", linewidth = 2.5, label="DDPGP")
plt.axvline(date_time[N_train-1], linestyle='--', color='black',
            label='<- train \n-> test')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

plt.show()