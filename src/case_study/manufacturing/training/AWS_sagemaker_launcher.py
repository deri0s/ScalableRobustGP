import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import (IntegerParameter,ContinuousParameter,HyperparameterTuner)
import numpy as np
from pathlib import Path
import pandas as pd

def prepare_data_partition(partition_index, total_partitions=5):
    """Prepare and upload a specific data partition to S3"""
    # Load your full dataset here
    ROOT_PATH = Path(__file__).resolve().parent.parent
    PROCESSED_PATH = ROOT_PATH / "data" / "processed" / "Training_data_partitions"
    file = PROCESSED_PATH / f'clean{partition_index}.xlsx' # Adjust based on your file

    X_df = pd.read_excel(file, sheet_name='X_stand')
    y_df = pd.read_excel(file, sheet_name='y_nonstand')

    # # Drop tweel position if present
    # if '9282 Tweel Position' in X_df.columns:
    # X_df.drop(columns=['9282 Tweel Position'], inplace=True)

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
    # Save to temporary files
    np.savez('train_data.npz', X=X_train, y=y_train)
    np.savez('test_data.npz', X=X_test, y=y_test)
    # Upload to S3
    bucket = sagemaker.Session().default_bucket()

    # Step 2: Create a SageMaker Launcher Script
    prefix = f'sgpr-partition-{partition_index}'
    s3_client = boto3.client('s3')
    s3_client.upload_file('train_data.npz', bucket, f'{prefix}/train/train_data.npz')
    s3_client.upload_file('test_data.npz', bucket, f'{prefix}/test/test_data.npz')
    return f's3://{bucket}/{prefix}/train', f's3://{bucket}/{prefix}/test'

def launch_training_job(partition_index, use_tuner=True):
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()
    role = sagemaker.get_execution_role()
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