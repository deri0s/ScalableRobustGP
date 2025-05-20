import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path

"""
NSG data

Do not adjust data for timelags.
"""

# NSG post processes data location
ROOT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT_PATH / "data" / "processed" / "Training_data_partitions"
file = PROCESSED_PATH / 'clean2.xlsx'

# Training df
y_df = pd.read_excel(file, sheet_name='y_nonstand')
t_df = pd.read_excel(file, sheet_name='timelags')

# Pre-Process training data
mu = y_df.gp_pred.values
y_filtered = y_df.y_filtered.values

# Replace zero values with interpolation
zeros = y_df.loc[y_df['y_raw'] <= 1e-1]
y_df.loc[zeros.index, 'y_raw'] = None
y_df.interpolate(inplace=True)

y_clean = pd.read_excel(file, sheet_name='y_nonstand_clean').y_raw.values
i_clean = pd.read_excel(file, sheet_name='y_nonstand_clean').Indices.values

y_train = y_df['y_raw'].values
date_time = y_df['date_time'].values

# Train test split ratio
N = len(y_train)
N_test = int(N*0.18)
end_indx = N - N_test

#-----------------------------------------------------------------------------
# PLOT TRAINING DATA
#-----------------------------------------------------------------------------

fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
fig.autofmt_xdate()

ax.plot(date_time, y_train, color='grey', label='Raw')
ax.plot(date_time, y_filtered, color='blue', label='Filtered')
ax.plot(date_time[i_clean], y_clean, 'o', color='green', label='furnace')
ax.plot(date_time, y_train, color='grey', label='Raw')
ax.plot(date_time, mu, color='red', label='GP')
ax.vlines(x=date_time[end_indx], ymin=-2, ymax=max(y_train),
          colors='black', ls='--', label='Test-data')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)
plt.show()