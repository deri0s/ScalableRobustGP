import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler as ss
from scipy import stats
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d

""" NSG data """

# NSG post processes data location
ROOT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT_PATH / "data" / "processed" / "Training_data_partitions"

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
        interp_func = interp1d(known_time, known_values, kind='cubic',
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

    """ SAVE PROCESSED DATA """
    # save filtered data
    X_test_df = pd.DataFrame(X_test)
    y_test_df = pd.DataFrame(smoothed)
    processed_data = pd.concat([X_test_df, y_test_df], axis=1)

    processed_data.to_csv(f"{PROCESSED_PATH}/data{i}.csv", index=False)

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
    ax.plot(dt_clean, smoothed, color='red', label='smoothed')

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

plt.show()