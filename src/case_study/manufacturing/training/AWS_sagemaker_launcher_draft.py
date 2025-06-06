import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import (IntegerParameter,ContinuousParameter,HyperparameterTuner)
import numpy as np
from pathlib import Path
import pandas as pd

def prepare_data_partition(partition_index, bucket_name='gpr-amc-bucket'):
    """ Prepare and upload a specific data partition to S3 """
    # Load your full dataset here
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
    y = y_df.gp_pred.values
    
    # Split into train/test
    test_perc = 0.2
    N = len(X)
    end_train = N - int(N*test_perc)
    X_train = X[0:end_train]
    y_train = y[0:end_train]
    X_test = X[end_train:]
    y_test = y[end_train:]

    # # Save to temporary files
    np.savez('train_data.npz', X=X_train, y=y_train)
    np.savez('test_data.npz', X=X_test, y=y_test)

    # Upload to S3
    s3_client = boto3.client('s3')
    partition_name = f'expert{partition_index}'
    s3_client.upload_file('train_data.npz', bucket_name,
                          f'data/{partition_name}/train/train_data.npz'
    )
    s3_client.upload_file('test_data.npz', bucket_name,
                          f'data/{partition_name}/test/test_data.npz'
    )
    return (f's3://{bucket_name}/data/{partition_name}/train_data.npz',
            f's3://{bucket_name}/data/{partition_name}/test_data.npz')

def launch_training_job(partition_index, use_tuner=True):
    # Prepare and upload data
    train_data_uri, test_data_uri = prepare_data_partition(partition_index)
    # Define the PyTorch estimator
    pytorch_estimator = PyTorch(
        entry_point='sgpr_training.py',
        role=role,
        instance_count=1, # Start with 1, can be increased for distributed training
        instance_type='ml.p3.2xlarge', # GPU instance, you can choose based on needs
        framework_version='1.12.0', # Choose appropriate PyTorch version
        py_version='py38',
        hyperparameters={'levels': 2,
                        'N_sim': 10000,
                        'training_iterations': 100,
                        'lr': 0.01,
                        'batch_size': 256},
        max_run=72*3600 # 72 hours max runtime
    )

    # If using hyperparameter tuning
    if use_tuner:
        hyperparameter_ranges = {
        'outputscale': ContinuousParameter(0.1, 10.0),
        'se_lengthscale': ContinuousParameter(0.05, 100.0),
        'rq_lengthscale': ContinuousParameter(0.05, 100.0),
        'rq_alpha': ContinuousParameter(0.05, 5.0),
        'noise_variance': ContinuousParameter(0.025, 0.028),
        'batch_size': IntegerParameter(128, 512),
        'lr': ContinuousParameter(0.001, 0.05)
        }
        tuner = HyperparameterTuner(
            estimator=pytorch_estimator,
            objective_metric_name='Final MSE',
            objective_type='Minimize',
            hyperparameter_ranges=hyperparameter_ranges,
            metric_definitions=[{'Name': 'Final MSE', 'Regex': 'Final MSE: ([0-9\\.]+)'}]
            max_jobs=100, # Total number of training jobs
            max_parallel_jobs=5, # Number of concurrent jobs
            strategy='Bayesian' # Can also use 'Random' or 'Grid'
        )
        # Start the hyperparameter tuning job
        tuner.fit({'train': train_data_uri, 'test': test_data_uri})
        return tuner
    else:
        # Start a single training job
        pytorch_estimator.fit({'train': train_data_uri, 'test': test_data_uri})
        return pytorch_estimator
    
# Launch training for all partitions in parallel
def train_all_experts(num_partitions=5, use_tuner=True):
    """Launch training jobs for all data partitions"""
    jobs = []
    for i in range(1, num_partitions+1):
    print(f"Launching training for partition {i}...")
    job = launch_training_job(i, use_tuner)
    jobs.append(job)
    return jobs

# Main execution
if __name__ == "__main__":
    jobs = train_all_experts(num_partitions=5)
    print("All training jobs launched. Check AWS SageMaker console for progress.")