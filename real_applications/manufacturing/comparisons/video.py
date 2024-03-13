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
y_raw0 = dpm.adjust_time_lag(y_raw_df['raw_furnace_faults'].values,
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


"""
Scalable Robust GP
"""

gp_config0_path = trained_path / 'results/DDPGP_NGPs6_config1_predictions.csv'
df = pd.read_csv(gp_config0_path)

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

"""
CREATE GIF
"""
from matplotlib.animation import FuncAnimation, PillowWriter

N_test = 216

start = y_df[y_df['Time stamp'] == '2021-04-29'].index[0]
test_range = range(start, N)

# The number of fault density data points that we can see after the model's
# last prediction
plot_lim = 144

X_test = X[test_range]
dt_test, y_rect, y_raw = date_time[test_range], y0[test_range], y_raw0[test_range]

# Initialise plot for video
fig, ax = plt.subplots()
fig.suptitle('Fault density forecast')
print('frames: \n', np.arange(0, N_test-plot_lim))
print('dt-test: ', np.shape(dt_test))

def animate(i):
    print('Frame', str(i), 'out of', str(len(y_rect) - N_test - plot_lim))

    indx = range(i, i+N_test)

    ax.clear()
    ax.plot(dt_test, y_raw, 'grey', label='Pre-processed')
    ax.plot(dt_test, y_rect,'black', label='Processed')

    # Plot linear regression predictions
    mu_line = ax.plot(dt_test[indx], mu[indx], 'red')

    var_area = ax.fill_between(dt_test[indx],
                               mu[indx] + 3*std[indx], mu[indx] - 3*std[indx],
                               alpha=0.5, color='gold',
                               label='Confidence \nBounds (DRGPs)')

    # Set axes of first subplot
    ax.set_ylim([-0.3, 2])
    ax.set_xlim([dt_test[i - 10], dt_test[i+N_test + plot_lim]])

    return [mu_line, var_area]


# Calling animation function to create a video
ani = FuncAnimation(fig,
                    func=animate,
                    frames=np.arange(0, N_test-plot_lim),
                    interval=1,
                    repeat=False)

# Saving the created video
ani.save('draft_gif.gif', writer=PillowWriter(fps=15))
plt.show()
plt.close()