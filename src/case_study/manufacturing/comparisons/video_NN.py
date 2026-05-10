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
mu = df.mu.values
std = df['std'].values
dt_dpgp = df['date-time']

print('\ndate-time: ', date_time[0], ' dt-dpgp: ', dt_dpgp[0])

"""
CREATE GIF
"""
from matplotlib.animation import FuncAnimation, PillowWriter

N_test = 216 # 3 days into the future
up_to = 9    # 8 = one month, 1 = full test data

start = y_df[y_df['Time stamp'] == '2021-05-04'].index[0]
train_test_line = N_train - start
test_range = range(start, N)

# The number of fault density data points that we can see after the model's
# last prediction
plot_lim = 144

X_test = X[test_range]

"""
Neural Network
"""
import os
from tensorflow import keras
from sklearn.metrics import mean_absolute_error as mae

# Load trained model
NN_path = trained_path / '2HL_64_units_Nonstandardised'
model1 = keras.models.load_model(NN_path)

# Predictions on test data
yNN = model1.predict(X_test)

mu, std = mu[test_range], std[test_range]
dt_test, y_rect, y_raw = date_time[test_range], y0[test_range], y_raw0[test_range]

# # Initialise plot for video
fig, ax = plt.subplots()

indx = range(0, 0+N_test)

def animate(i):
    print('Frame', str(i), 'out of', str(9*N_test - plot_lim))

    indx = range(i, i+N_test)

    ax.clear()
    fig.autofmt_xdate()
    ax.plot(dt_test, y_raw, 'grey', label='Pre-processed')
    ax.plot(dt_test, y_rect,'royalblue', label='Processed')
    plt.text(dt_test[0+train_test_line-110], 2.5,
             r'$\leftarrow$ Training data',
             fontsize = 12,
             color='white',
             bbox = dict(facecolor = 'black'))
    plt.text(dt_test[0+train_test_line+7], 2,
             r'Test data $\rightarrow$',
             fontsize = 12,
             color='white',
             bbox = dict(facecolor = 'black'))
    ax.axvline(dt_test[0+train_test_line], linestyle='--',
               linewidth=3,
               color='red')

    # Plot linear regression predictions
    var_area = ax.fill_between(dt_test[indx],
                               mu[indx] + 3*std[indx], mu[indx] - 3*std[indx],
                               alpha=0.5, color='gold',
                               label='Confidence \nBounds')
    
    nn_line = ax.plot(dt_test[indx], yNN[indx], linewidth=3, c='black',
                      label='NN')
    
    mu_line = ax.plot(dt_test[indx], mu[indx], linewidth=3, c='red',
                    label='DDPGP')

    # Set axes of first subplot
    ax.set_ylim([-0.3, 3])
    ax.set_xlim([dt_test[i - 10], dt_test[i+N_test + plot_lim]])

    ax.set_xlabel(" Date-time", fontsize=13)
    ax.set_ylabel(" Fault density", fontsize=13)
    ax.legend(bbox_to_anchor=(0.5, 1.19), ncol=5, loc='upper center')
            #   facecolor="white", framealpha=1.0)

    return [mu_line, var_area, nn_line]


# Calling animation function to create a video
ani = FuncAnimation(fig,
                    func=animate,
                    frames=np.arange(0, len(test_range) - up_to*N_test - plot_lim),
                    interval=1,
                    repeat=False)

# Saving the created video
ani.save('figures/NSG/DDPGP_vs_NN.gif', writer=PillowWriter(fps=15))
# plt.show()
plt.close()