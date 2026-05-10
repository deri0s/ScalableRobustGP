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

apply_timelags = True

N_partitions = 5
for index in range(N_partitions):
    file = PROCESSED_PATH / f'data{index}.xlsx'

    # Training df
    y_df = pd.read_excel(file, sheet_name='y_nonstand')
    t_df = pd.read_excel(file, sheet_name='timelags')
    t_series = t_df.iloc[0, :]

    if apply_timelags:
        """ 1. Align inputs and variables according to their time lags """
        ydeep = y_df.copy()
        # Ensure t_series values are numeric before finding max
        numeric_t_series = pd.to_numeric(t_series, errors='coerce').fillna(0)
        if numeric_t_series.empty:
            max_lag = 0
        else:
            max_lag = int(max(numeric_t_series))

        # y and date-time alignment
        # Ensure ydeep has enough rows before slicing
        ydeep = ydeep.iloc[max_lag:].reset_index(drop=True)

        # Rename dataframes after alignment
        common_len = len(ydeep)
        y_df = ydeep.iloc[:common_len].reset_index(drop=True)

        # Convert to numpy for saving
        y = y_df.y_processed.values

    # Pre-Process training data
    mu = y_df.y_processed.values
    y_filtered = y_df.y_filtered.values

    # # Replace zero values with interpolation
    # zeros = y_df.loc[y_df['y_raw'] <= 1e-1]
    # y_df.loc[zeros.index, 'y_raw'] = None
    # y_df.interpolate(inplace=True)

    # y_clean = pd.read_excel(file, sheet_name='y_nonstand_clean').y_raw.values
    # i_clean = pd.read_excel(file, sheet_name='y_nonstand_clean').Indices.values

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
    # ax.plot(date_time[i_clean], y_clean, 'o', color='green', label='furnace')
    ax.plot(date_time, mu, color='red', label='GP')
    ax.vlines(x=date_time[end_indx], ymin=-2, ymax=max(y_train),
            colors='black', ls='--', label='Test-data')
    ax.set_xlabel(" Date-time", fontsize=14)
    ax.set_ylabel(" Fault density", fontsize=14)
    plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

plt.show()