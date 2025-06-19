import os
import torch
import gpytorch
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal

"""
NSG data

Do not adjust data for timelags.
"""

# NSG post processes data location
ROOT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT_PATH / "data" / "processed" / "Training_data_partitions"
EXPERT_PATH = ROOT_PATH / "trained" / "experts"

apply_timelags = True

N_partitions = 1
for index in range(N_partitions):
    file = PROCESSED_PATH / f'data{index}.xlsx'

    # Training df
    X_df = pd.read_excel(file, sheet_name='X_stand')
    y_df = pd.read_excel(file, sheet_name='y_nonstand')
    t_df = pd.read_excel(file, sheet_name='timelags')
    t_series = t_df.iloc[0, :]

    if apply_timelags:
        """ 1. Align inputs and variables according to their time lags """
        xdeep = X_df.copy()
        ydeep = y_df.copy()
        # Ensure t_series values are numeric before finding max
        numeric_t_series = pd.to_numeric(t_series, errors='coerce').fillna(0)
        if numeric_t_series.empty:
            max_lag = 0
        else:
            max_lag = int(max(numeric_t_series))

            # X
    for name, lag in t_series.items():
        # Ensure lag is treated as integer for shift
        try:
            lag_int = int(float(lag))
            if lag_int > 0: # Only shift if lag is positive
                 xdeep[name] = xdeep[name].shift(lag_int)
        except ValueError:
            print(f"Warning: Could not convert lag '{lag}' for feature '{name}' to int. Skipping shift.")

    # Drop rows with NaNs introduced by shifting (only drop up to max_lag rows from top)
    xdeep = xdeep.iloc[max_lag:] # More direct way to handle shift NaNs

    # Ensure xdeep and ydeep have the same length after alignment
    common_len = min(len(xdeep), len(ydeep))
    xdeep = xdeep.iloc[:common_len].reset_index(drop=True)
    ydeep = ydeep.iloc[:common_len].reset_index(drop=True)

    # rename X and y dataframes
    X_np = xdeep.values

    floating_point = torch.float64
    X = torch.tensor(X_np, dtype=floating_point)

    # Pre-Process training data
    y_train = ydeep.y_processed.values
    y_filtered = ydeep.y_filtered.values
    date_time = ydeep.date_time.values

    """ 2. Load trained experts """

    expert_path = os.path.join(EXPERT_PATH, f'expert{index}.pth')
    scaler_path = os.path.join(EXPERT_PATH, f'scaler{index}.pth')
        
    # Load train expert
    gp = torch.load(expert_path)
    scaler = torch.load(scaler_path)

    likelihood = GaussianLikelihood()

    # Predictions
    gp.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(gp(X))

        # Unormalise predictions
        pred_mean = observed_pred.mean
        mu = scaler.inverse_transform(pred_mean.unsqueeze(1))[:,0]
        stds = scaler.inverse_transform(observed_pred.stddev.unsqueeze(1))[:,0]
        lower_stand, upper_stand = observed_pred.confidence_region()
        lower = scaler.inverse_transform(lower_stand.unsqueeze(1))[:,0]
        upper = scaler.inverse_transform(upper_stand.unsqueeze(1))[:,0]

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
    # ax.plot(date_time, y_filtered, color='blue', label='Filtered')
    # ax.plot(date_time[i_clean], y_clean, 'o', color='green', label='furnace')
    ax.plot(date_time, mu, color='red', label='GP')
    # ax.vlines(x=date_time[end_indx], ymin=-2, ymax=max(y_train),
    #         colors='black', ls='--', label='Test-data')
    # ax.set_xlabel(" Date-time", fontsize=14)
    ax.set_ylabel(" Fault density", fontsize=14)
    plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

plt.show()