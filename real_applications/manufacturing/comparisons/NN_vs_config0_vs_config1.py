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
READ config
"""
current_path = Path(__file__).resolve()
trained_path = current_path.parents[1] / 'trained'

with open(trained_path / 'config0.yml', 'r') as f:
    config = yaml.safe_load(f)
    
    val_split = config['test_per']

# Train and test data
N, D = np.shape(X)
N_train = int(N*0.84)

test_range = range(0, N)
X_test = X[test_range]
# print('X: ', np.shape(X), ' N_train: ', N_train, ' N-test: ', N_train + 720)
dt_test, y_test = date_time[test_range], y0[test_range]


"""
Neural Network
"""
from tensorflow import keras
from sklearn.metrics import mean_absolute_error as mae

# Load trained model
NN_path = trained_path / '2HL_64_units_Nonstandardised'
model1 = keras.models.load_model(NN_path)

# Predictions on test data
yNN = model1.predict(X_test)


"""
Distributed GP
"""
import pickle

dgp_path = trained_path / 'DGP_8.pkl'

with open(dgp_path, 'rb') as f:
    dgp = pickle.load(f)

mu, std, betas = dgp.predict(X_test)
mu[np.isnan(mu)] = 0

"""
Scalable Robust GP
"""

# from pickle
# ddpgp_path = trained_path / 'ddpgp_NGPs3_config1'

# with open(ddpgp_path, 'rb') as f:
#     ddpgp = pickle.load(f)

# muMix, stdMix, betasMix = ddpgp.predict(X_test)

gp_config0_path = trained_path / 'results/DDPGP_NGPs16_config0_predictions.csv'
df = pd.read_csv(gp_config0_path)

# # set nan values for mu and std to zero and 100, respectively 
df.interpolate(method='zero', inplace=True)
mu0 = df.mu
std0 = df.std

# config1
gp_config1_path = trained_path / 'results/DDPGP_NGPs6_config1_predictions.csv'
df1 = pd.read_csv(gp_config1_path)

print(' antes: ', df1.isna().sum())
# # set nan values for mu and std to zero and 100, respectively 
df1.interpolate(method='zero', inplace=True)
mu1 = df1.mu
mu1[np.isnan(mu1)] = 0
std1 = df1.std

print('MAE')
print('NN: ', mae(y_test, yNN))
print('DGP: ', mae(y_test, mu))
print('config0: ', mae(y_test, mu0))
print('config1: ', mae(y_test, mu1))


"""
Plots
"""
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
                alpha=0.5, color='gold',
                label='Confidence \nBounds (DRGPs)')
ax.plot(date_time, y_raw, color="grey", linewidth = 2.5, label="Raw")
# ax.plot(dt_test, y_test, color="blue", linewidth = 2.5, label="Conditioned")
# ax.plot(dt_test, yNN, color="red", linewidth = 2.5, label="NN")
# ax.plot(dt_test, mu, color="green", linewidth = 2.5, label="DGP")
ax.plot(dt_test, mu0, color="black", linewidth = 2.5, label="config0")
ax.plot(dt_test, mu1, color="red", linewidth = 2.5, label="config1")
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":12}, facecolor="white", framealpha=1.0)

plt.show()