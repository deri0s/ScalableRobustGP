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
NSG data

Do not adjust data for timelags.
"""

# NSG post processes data location
ROOT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT_PATH / "data" / "processed" / "Training_data_partitions"
EXPERT_PATH = ROOT_PATH / "trained" / "experts"

apply_timelags = True
N_partitions = 5

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
    
    if hasattr(gp.covar_module, 'kernels'):  # Additive kernel
        results['kernel_type'] = 'additive'
        for i, k in enumerate(gp.covar_module.kernels):
            if hasattr(k, 'base_kernel'):
                if hasattr(k.base_kernel, 'lengthscale'):
                    results[f'kernel_{i}_lengthscales'] = k.base_kernel.lengthscale.squeeze().tolist()
                if hasattr(k.base_kernel, 'alpha'):
                    results[f'kernel_{i}_alpha'] = k.base_kernel.alpha.item()
                results[f'kernel_{i}_outputscale'] = k.outputscale.item()
    else:  # Single kernel
        results['outputscale'] = gp.covar_module.outputscale.item()
        if hasattr(gp.covar_module.base_kernel, 'lengthscale'):
            results['lengthscales'] = gp.covar_module.base_kernel.lengthscale.squeeze().tolist()
        if hasattr(gp.covar_module.base_kernel, 'alpha'):
            results['alpha'] = gp.covar_module.base_kernel.alpha.item()
    
    results['noise'] = gp.likelihood.noise.item()
    return results


for index in range(N_partitions):
    file = PROCESSED_PATH / f'data{index}.xlsx'

    # Training df
    X_df = pd.read_excel(file, sheet_name='X_stand')
    y_df = pd.read_excel(file, sheet_name='y_nonstand')
    t_df = pd.read_excel(file, sheet_name='timelags')
    t_series = t_df.iloc[0, :]

    if os.path.exists(os.path.join(EXPERT_PATH, f'dropped_inputs{index}.yaml')):
        dropped_path = os.path.join(EXPERT_PATH, f'dropped_inputs{index}.yaml')
        with open(dropped_path, 'r') as f:
            dropped = yaml.load(f, Loader=yaml.SafeLoader)
        
        for input in dropped['to_drop']:
            X_df.drop(columns=input, inplace=True)
            t_df.drop(columns=input, inplace=True)

    X_df, y_df = align_inputs(X_df, y_df, t_df.iloc[0,:])

    X_np = X_df.values
    y_processed = y_df.y_processed.values
    date_time = y_df.date_time.values

    # Convert data to torch tensors
    floating_point = torch.float64
    X = torch.tensor(X_np, dtype=floating_point)

    """ 2. Load trained experts """
    expert_path = os.path.join(EXPERT_PATH, f'expert{index}.pth')
    scaler_path = os.path.join(EXPERT_PATH, f'scaler{index}.pth')
        
    # Load train expert
    gp = torch.load(expert_path, weights_only=False)
    scaler = torch.load(scaler_path, weights_only=False)

    likelihood = GaussianLikelihood()

    print(f'\nEstimated Kerne:\n {gp.covar_module.base_kernel}')

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

        print(f'MSE(train-test): {mean_squared_error(y_processed,
                                                     mu)}')

    #-----------------------------------------------------------------------------
    # PLOT TRAINING DATA
    #-----------------------------------------------------------------------------
    end_indx = int(len(X)*0.8)
    fig, ax = plt.subplots()

    # Increase the size of the axis numbers
    plt.rcdefaults()
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    fig.autofmt_xdate()

    plt.title(f'Expert {index}')
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