import os
import torch
import gpytorch
import yaml
import pandas as pd
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_squared_error
from gpytorch.likelihoods import GaussianLikelihood

"""
Inputs
"""

N_partitions = 5

# NSG post processes data location
ROOT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT_PATH / "data" / "processed" / "Training_data_partitions"
EXPERT_PATH = ROOT_PATH / "trained" / "experts"

def align_inputs(x_df, y_df, t_series):
    xdeep = x_df.copy()
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
        return pd.DataFrame(columns=x_df.columns), pd.DataFrame(columns=y_df.columns)

    # Ensure xdeep and ydeep have the same length after alignment
    common_len = min(len(xdeep), len(ydeep))
    xdeep = xdeep.iloc[:common_len].reset_index(drop=True)
    ydeep = ydeep.iloc[:common_len].reset_index(drop=True)

    return xdeep, ydeep

def get_hyper(gp):
    """Extract hyperparameters based on kernel type"""
    results = {}

    if hasattr(gp.covar_module.base_kernel, 'kernels'):  # Additive kernel
        results['kernel_type'] = 'additive'
        results[f'outputscale'] = gp.covar_module.outputscale.item()
        for i, k in enumerate(gp.covar_module.base_kernel.kernels):
            results[f'kernel_{i}_name'] = k.__class__.__name__
            if hasattr(k, 'lengthscale'):
                results[f'kernel_{i}_lengthscales'] = k.lengthscale.squeeze().tolist()
            if hasattr(k, 'alpha'):
                results[f'kernel_{i}_alpha'] = k.alpha.item()
    else:  # Single kernel
        results['kernel_type'] = 'single'
        results['outputscale'] = gp.covar_module.outputscale.item()
        if hasattr(gp.covar_module.base_kernel, 'lengthscale'):
            results['lengthscales'] = gp.covar_module.base_kernel.lengthscale.squeeze().tolist()
        if hasattr(gp.covar_module.base_kernel, 'alpha'):
            results['alpha'] = gp.covar_module.base_kernel.alpha.item()

    results['noise'] = gp.likelihood.noise.item()
    return results


def print_est_hyper(hyperparams, X_df):
    print('\nEstimated Hyperparameters:')
    for key, value in hyperparams.items():
        if not isinstance(value, list):
            print(f"{key}: {value}")

    print('\nFeature Importance (sorted by lengthscale):')
    if hyperparams['kernel_type'] == 'additive':
        for i in range(2):
            name = hyperparams[f'kernel_{i}_name']
            print(f'\nKernel: {name}')
            if i == 0:
                feature_importance0 = pd.DataFrame({
                    'inputs': X_df.columns.values, 
                    'lengthscales': hyperparams[f'kernel_{i}_lengthscales']
                })
                print(feature_importance0.sort_values(by='lengthscales'))
            else:
                feature_importance = pd.DataFrame({
                    'inputs': X_df.columns.values, 
                    'lengthscales': hyperparams[f'kernel_{i}_lengthscales']
                })
                print(feature_importance.sort_values(by='lengthscales'))
    else:
        feature_importance = pd.DataFrame({
            'inputs': X_df.columns.values, 
            'lengthscales': hyperparams['lengthscales']
        })
        print(feature_importance.sort_values(by='lengthscales'))


def predict_and_eval(gp, likelihood, scaler, X, X_train, X_test):
    gp.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(gp(X))
        observed_pred_train = likelihood(gp(X_train))
        observed_pred_test = likelihood(gp(X_test))

        # Unormalise predictions
        pred_mean = observed_pred.mean
        pred_mean_train = observed_pred_train.mean
        pred_mean_test = observed_pred_test.mean

    mu = scaler.inverse_transform(pred_mean.unsqueeze(1))[:,0]
    mu_train = scaler.inverse_transform(pred_mean_train.unsqueeze(1))[:,0]
    mu_test = scaler.inverse_transform(pred_mean_test.unsqueeze(1))[:,0]
    stds = scaler.inverse_transform(observed_pred.stddev.unsqueeze(1))[:,0]
    lower_stand, upper_stand = observed_pred.confidence_region()
    lower = scaler.inverse_transform(lower_stand.unsqueeze(1))[:,0]
    upper = scaler.inverse_transform(upper_stand.unsqueeze(1))[:,0]

    mse_all = mean_squared_error(y_processed, mu)
    mse_train = mean_squared_error(y_train, mu_train)
    mse_test = mean_squared_error(y_test, mu_test)

    return mse_all, mse_train, mse_test, mu, lower, upper


for index in range(N_partitions):
    file = PROCESSED_PATH / f'data{index}.xlsx'

    # Training df
    X_df = pd.read_excel(file, sheet_name='X_stand')
    y_df = pd.read_excel(file, sheet_name='y_nonstand')
    t_df = pd.read_excel(file, sheet_name='timelags')
    t_series = t_df.iloc[0, :]

    # if os.path.exists(os.path.join(EXPERT_PATH, f'dropped_inputs{index}.yaml')):
    #     dropped_path = os.path.join(EXPERT_PATH, f'dropped_inputs{index}.yaml')
    #     with open(dropped_path, 'r') as f:
    #         dropped = yaml.load(f, Loader=yaml.SafeLoader)
        
    #     for input in dropped['to_drop']:
    #         X_df.drop(columns=input, inplace=True)
    #         t_df.drop(columns=input, inplace=True)

    X_df, y_df = align_inputs(X_df, y_df, t_df.iloc[0,:])

    N, D = X_df.shape
    X_np = X_df.values
    y_processed = y_df.y_processed.values
    date_time = y_df.date_time.values

    end_indx = int(len(X_np)*0.8)
    end_train = N - end_indx
    X_train = X_np[0:end_train]
    date_train = date_time[0:end_train]
    N_train = len(X_train)
    y_train_nonstand = y_processed[0:end_train]

    # Define X_test_np and y_test_nonstand correctly for evaluation metric
    X_test = X_np[end_train:N]
    y_test_nonstand = y_processed[end_train:N]

    # Convert data to torch tensors
    floating_point = torch.float64
    X = torch.tensor(X_np, dtype=floating_point)
    X_train = torch.tensor(X_train, dtype=floating_point)
    y_train = torch.tensor(y_train_nonstand, dtype=floating_point).squeeze()
    X_test = torch.tensor(X_test, dtype=floating_point)
    y_test = torch.tensor(y_test_nonstand, dtype=floating_point).squeeze()

    """ 2. Load trained experts """
    expert_path = os.path.join(EXPERT_PATH, f'expert{index}.pth')
    scaler_path = os.path.join(EXPERT_PATH, f'scaler{index}.pth')
        
    # Load train expert
    gp = torch.load(expert_path, weights_only=False)
    scaler = torch.load(scaler_path, weights_only=False)

    likelihood = GaussianLikelihood()

    k_name = gp.covar_module.base_kernel.__class__.__name__.replace('Kernel', '')
    print(f'\nEstimated Kernel:\n {k_name}')

    # print estimated hyperparameters
    hyperparams = get_hyper(gp)
    print_est_hyper(hyperparams, X_df)

    (mse_all, mse_train, mse_test,
     mu, lower, upper) = predict_and_eval(gp, likelihood, scaler,
                                          X, X_train, X_test)

    #-----------------------------------------------------------------------------
    # PLOT TRAINING DATA
    #-----------------------------------------------------------------------------
    fig, ax = plt.subplots()

    # Increase the size of the axis numbers
    plt.rcdefaults()
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    fig.autofmt_xdate()

    plt.title(f'Expert: {index}, K: {k_name}, MSE-train: {mse_train:.4f}, MSE-test: {mse_test:.4f}, MSE-all: {mse_all:.4f}')
    ax.fill_between(date_time, lower, upper, 
                    alpha=0.3, color='coral', label='95% CI')
    ax.plot(date_time, y_processed, '*', color='green', label='Val')
    # ax.plot(date_time, y_filtered, color='blue', label='Filtered')
    # ax.plot(date_time[i_clean], y_clean, 'o', color='green', label='furnace')
    ax.plot(date_time, mu, color='red', label='GP')
    ax.vlines(x=date_time[end_indx], ymin=0, ymax=max(y_processed),
            colors='black', ls='--', label='Test-data')
    ax.set_xlabel(" Date-time", fontsize=14)
    ax.set_ylabel(" Fault density", fontsize=14)
    plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

plt.show()