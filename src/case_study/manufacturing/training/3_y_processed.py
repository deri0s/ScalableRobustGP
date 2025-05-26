import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d

""" NSG data """

# NSG post processes data location
ROOT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT_PATH / "data" / "processed" / "Training_data_partitions"

# initialise variables
smoothed_full = []

for i in range(5):
    file = PROCESSED_PATH / f'clean{i}.xlsx'

    # Training df
    X_df = pd.read_excel(file, sheet_name='X_stand')
    y_df = pd.read_excel(file, sheet_name='y_nonstand')
    t_df = pd.read_excel(file, sheet_name='timelags')
    iclean = pd.read_excel(file, sheet_name='y_nonstand_clean').Indices.values

    # raw
    X_test = X_df.values
    y_raw = y_df.y_raw.values
    y_filtered = y_df.y_filtered.values
    date_time = y_df.date_time.values

    # Clean
    yff = y_df.y_raw.values[iclean]
    dt_clean = date_time[iclean]

    print(f'X: {len(X_df)}, yff: {len(yff)}, date-time: {len(date_time)}')

    # 1. Eliminate outliers using Z-scores for each data point
    z_scores = np.abs(stats.zscore(yff))

    # Define a threshold for outlier detection (common thresholds are 2, 2.5, or 3)
    if i == 1:
        z_score_threshold = 40.5
    else:
        z_score_threshold = 4.5

    # Identify outliers
    outliers_indices = np.where(z_scores > z_score_threshold)[0]
    data_filtered_zscore = yff[z_scores <= z_score_threshold]

    # Get the indices of the identified outliers
    outlier_mask = z_scores > z_score_threshold
    outlier_indices_found = np.where(outlier_mask)[0]

    # 2. Interpolate the Outliers
    interpolated_signal = np.copy(yff)

    # Separate known (non-outlier) data points and their corresponding time values
    known_indices = np.where(~outlier_mask)[0]
    x_indices = np.linspace(0, 20, len(dt_clean))
    known_time = x_indices[known_indices]
    known_values = yff[known_indices]

    # Ensure there are enough known points to interpolate
    if len(known_indices) < 2:
        print("Not enough non-outlier points to perform interpolation. Exiting.")
    else:
        # Create an interpolation function using the known data
        interp_func = interp1d(known_time, known_values, kind='linear',
                               fill_value="extrapolate")

        # Interpolate values at the outlier locations
        time_to_interpolate = x_indices[outlier_indices_found]
        interpolated_values = interp_func(time_to_interpolate)

        # Replace the outlier values with the interpolated values
        interpolated_signal[outlier_indices_found] = interpolated_values

    # 2. Apply uniform_filter1d for moving average smoothing
    window_size_ndimage = 10 # Moving Average windod size
    smoothed = uniform_filter1d(interpolated_signal, size=window_size_ndimage)
    smoothed = uniform_filter1d(smoothed, size=window_size_ndimage)
    if i in [1,2]:
        pass
    else:
        smoothed = uniform_filter1d(smoothed, size=window_size_ndimage)
    
    # 3. Interpolate at nonfurnace faults locations
    nonff_indices = [x for x in range(len(X_test)) if x not in iclean]

    x_indices = np.linspace(0, 20, len(date_time))
    known_time = x_indices[iclean]

    # Ensure there are enough known points to interpolate
    if len(known_indices) < 2:
        print("Not enough non-outlier points to perform interpolation. Exiting.")
    else:
        # Create an interpolation function using the known data
        interp_func = interp1d(known_time, smoothed, kind='quadratic',
                               fill_value="extrapolate")

        # Interpolate values at the non-furnace faults locations
        time_to_interpolate = x_indices[nonff_indices]
        interpolated_values = interp_func(time_to_interpolate)

        # Replace the outlier values with the interpolated values
        y_raw[nonff_indices] = interpolated_values
        y_raw[iclean] = smoothed
        # eliminate wiggles from the quadratic interpolation
        if i == 1:
            y_processed = uniform_filter1d(np.copy(y_raw), size=40)
            y_processed[int(len(y_raw)/2) + 480: len(y_raw)] = uniform_filter1d(y_raw[int(len(y_raw)/2) + 480: len(y_raw)], size=100)
        else:
            y_processed = uniform_filter1d(np.copy(y_raw), size=40)

    # 4. Assemble full training targets for Deep Learning
    if i == 0:
        y_processed_all = np.copy(y_processed)
    else:
        y_processed_all = np.concatenate((y_processed_all, y_processed))

    """ SAVE PROCESSED DATA """
    with pd.ExcelWriter(f"{PROCESSED_PATH}/data{i}.xlsx", engine='openpyxl') as writer:
        X_df.to_excel(writer, sheet_name='X_stand', index=False)
        X_df_norm = pd.read_excel(file, sheet_name='X_stand')
        X_df_norm.to_excel(writer, sheet_name='X_norm', index=False)
        # y
        y_df.drop(columns=['gp_pred'], inplace=True)
        y_df['y_processed'] = y_processed
        y_df.to_excel(writer, sheet_name='y_nonstand', index=False)
        # time-lags
        t_df.to_excel(writer, sheet_name='timelags', index=False)
        # furnace-faults indices
        indices_df = pd.DataFrame({'indices': iclean})
        indices_df.to_excel(writer, sheet_name='clean_indices', index=False) 

    """ Plots """
    fig, ax = plt.subplots()

    # Increase the size of the axis numbers
    plt.rcdefaults()
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    fig.autofmt_xdate()

    ax.plot(date_time, y_raw, color='grey', label='y-raw')
    ax.plot(dt_clean, yff, 'o', color='green', label='DPSGP')
    ax.plot(date_time, y_filtered, color='blue', label='y_filtered')
    ax.plot(dt_clean, smoothed, color='orange', label='smoothed')
    ax.plot(date_time, y_processed, linewidth=1.5, color='red', label='y_processed')

    ax.vlines(
        x=dt_clean[outliers_indices],
        ymin=y_raw.min(),
        ymax=y_raw.max(),
        alpha=0.3,
        linewidth=1.5,
        label="outliers",
        color='coral'
    )

    plt.title(F"Patition: {i}", fontsize=17)
    ax.set_xlabel(" Date-time", fontsize=14)
    ax.set_ylabel(" Fault density", fontsize=14)
    plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)
    ax.set_xlabel(" Date-time", fontsize=14)
    ax.set_ylabel(" Fault density", fontsize=14)
    plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

# Assemble full training dataset
# PROCESSED_PATH = ROOT_PATH / "data" / "processed"
# file = PROCESSED_PATH / "NSG_processed_data.xlsx"
# # X
# X_df = pd.read_excel(file, sheet_name='X_stand')
# X_df_norm = pd.read_excel(file, sheet_name='X_stand')
# # y
# y_df = pd.read_excel(file, sheet_name='y')
# y_df['y_processed'] = y_processed_all
# y_df['date_time'] = y_df['Time stamp'].values
# y_df['y_raw'] = pd.read_excel(file, sheet_name='y_raw')['raw_furnace_faults'].values

# y_df.drop(columns=['furnace_faults', 'Hig values replaced', 'Low values replaced'], inplace=True)
# with pd.ExcelWriter(f"{PROCESSED_PATH}/training_data.xlsx", engine='openpyxl') as writer:
#     X_df.to_excel(writer, sheet_name='X_stand', index=False)
#     X_df_norm.to_excel(writer, sheet_name='X_norm', index=False)
#     y_df.to_excel(writer, sheet_name='y_nonstand', index=False)
#     # time-lags
#     t_df.to_excel(writer, sheet_name='timelags', index=False)

plt.show()