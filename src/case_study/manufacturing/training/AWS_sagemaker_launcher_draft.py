import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.tuner import (IntegerParameter,ContinuousParameter,HyperparameterTuner)
import numpy as np
from pathlib import Path
import pandas as pd

def launch_training_job(index, use_tuner=True):

    train_data_uri = f's3://gpr-amc-bucket/data/partition{index}/train_data.npz',
    test_data_uri = f's3://gpr-amc-bucket/data/partition{index}/test_data.npz'
    
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
                        'lr': 0.009,
                        'batch_size': 256},
        max_run=72*3600 # 72 hours max runtime
    )

    # If using hyperparameter tuning
    if use_tuner:
        hyperparameter_ranges = {
        'outputscale': ContinuousParameter(0.1, 10.0),
        'se_lengthscale': ContinuousParameter(0.05, 100.0),
        'rq_lengthscale': ContinuousParameter(0.05, 100.0),
        'rq_alpha': ContinuousParameter(0.1, 5.0),
        'per_period_length': ContinuousParameter(6, 10),
        'per_lengthscale': ContinuousParameter(0.05, 100.0),
        'lin_variance': ContinuousParameter(0.1*0.01, 0.025),
        'noise_variance': ContinuousParameter(0.01, 0.028),
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
    """ Launch training jobs for all data partitions """
    jobs = []
    for i in range(1, num_partitions+1):
    print(f"\nLaunching training for partition {i}...")
    job = launch_training_job(i, use_tuner)
    jobs.append(job)
    return jobs

# Main execution
if __name__ == "__main__":
    jobs = train_all_experts(num_partitions=5)
    print("All training jobs launched. Check AWS SageMaker console for progress.")