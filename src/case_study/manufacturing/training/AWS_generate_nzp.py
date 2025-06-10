import boto3
import numpy as np
from pathlib import Path
import pandas as pd

def generate_partition(partition_index, bucket_name):
    """ Prepare and upload a specific data partition to S3 """
    ROOT_PATH = Path(__file__).resolve().parent.parent
    PROCESSED_PATH = ROOT_PATH / "data" / "processed" / "Training_data_partitions"
    file = PROCESSED_PATH / f'data{partition_index}.xlsx'

    # Training df
    X_df = pd.read_excel(file, sheet_name='X_stand')
    y_df = pd.read_excel(file, sheet_name='y_nonstand')
    t_df = pd.read_excel(file, sheet_name='timelags')
    t_series = t_df.iloc[0,:]

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

    # y and date-time alignment
    # Ensure ydeep has enough rows before slicing
    if len(ydeep) >= max_lag:
        ydeep = ydeep.iloc[max_lag:].reset_index(drop=True)
    else:
        # Handle case where ydeep is shorter than max_lag (e.g., return empty DataFrames)
        print(f"Warning: y DataFrame length ({len(ydeep)}) is less than max_lag ({max_lag}). Alignment might be incorrect.")
        return pd.DataFrame(columns=X_df.columns), pd.DataFrame(columns=y_df.columns)

    # Rename dataframes after alignment
    common_len = min(len(xdeep), len(ydeep))
    X_df = xdeep.iloc[:common_len].reset_index(drop=True)
    y_df = ydeep.iloc[:common_len].reset_index(drop=True)

    # Convert to numpy for saving
    X = X_df.values
    y = y_df.y_processed.values
    
    # Split into train/test
    test_perc = 0.2
    N = len(X)
    end_train = N - int(N*test_perc)
    X_train = X[0:end_train]
    y_train = y[0:end_train]
    X_test = X[end_train:]
    y_test = y[end_train:]

    # Save to temporary files
    # Code for local testing
    np.savez('train_data.npz', X=X_train, y=y_train)
    np.savez('test_data.npz', X=X_test, y=y_test)
    np.savez('all_data.npz', X=)

    # # Upload to S3
    # s3_client = boto3.client('s3')
    # partition_name = f'expert{partition_index}'
    # s3_client.upload_file('train_data.npz', bucket_name,
    #                       f'data/{partition_name}/train/train_data.npz')
    # s3_client.upload_file('test_data.npz', bucket_name,
    #                       f'data/{partition_name}/test/test_data.npz')

# Generate the 5 data partitions
N_partitions = 5
for i in range(1, N_partitions):
    generate_partition(i, 'gpr-amc-bucket')