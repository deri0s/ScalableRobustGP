import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import matplotlib.dates as mdates # Needed for formatting axis

# --- Your data loading code here ---

# NSG post processes data location
ROOT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT_PATH / "data" / "processed"
file = PROCESSED_PATH / 'NSG_processed_data.xlsx'

# Load excel file
X_df = pd.read_excel(file, sheet_name='X_stand')
y_df = pd.read_excel(file, sheet_name='y')
y_raw_df = pd.read_excel(file, sheet_name='y_raw')
t_df = pd.read_excel(file, sheet_name='timelags')

print(f'at the beginning: {y_raw_df.size}')

# Extract values
X = X_df.values
# Assuming 'furnace_faults' exists in y_df based on original code context
if 'furnace_faults' in y_df.columns:
    y0 = y_df.furnace_faults.values
else:
    y0 = None # Handle case where column might be missing
N, D = np.shape(X)

# Replace zero values with interpolation
zeros = y_raw_df.loc[y_raw_df['raw_furnace_faults'] <= 1e-1]
y_raw_df.loc[zeros.index, 'raw_furnace_faults'] = None
y_raw_df.interpolate(inplace=True)

y_raw = y_raw_df['raw_furnace_faults'].values
date_time = y_df['Time stamp'].values

# Get the sampling rate from the first two data points
step = date_time[1] - date_time[0]

""" Noise Burst Detection """

def ignore_noise_bursts(errors, y_raw, window_size, threshold_factor):
        """
        Identify noise bursts using moving standard deviation and
        penalise residuals at those locations.
        
        Parameters:
        -----------
        errors : ndarray
            Residuals from GP prediction
        y_raw : ndarray
            Raw measurements/observations
        window_size : int
            Size of the moving window for standard deviation calculation
        threshold_factor : float
            Multiple of the median moving std dev to use as threshold
            
        Returns:
        --------
        ndarray
            Modified errors with penalised values at noise burst locations
        """
        # Make a copy to avoid modifying the input
        penalised_errors = errors.copy()
        
        # Create a pandas Series for rolling calculations
        y_raw_series = pd.Series(y_raw.flatten())
        
        # 1. Calculate Moving Standard Deviation
        moving_std = y_raw_series.rolling(window=window_size, center=True,
                                          min_periods=1).std()

        # 2. Determine Threshold
        # Calculate median and std of the non-NaN moving_std values for robustness
        valid_moving_std = moving_std.dropna()
        if not valid_moving_std.empty:
            median_moving_std = valid_moving_std.median()
            std_moving_std = valid_moving_std.std()
            # Avoid threshold being NaN if std_moving_std is 0 (flat line)
            if pd.isna(std_moving_std) or std_moving_std == 0:
                std_moving_std = 1e-6  # Assign small value

            threshold = median_moving_std + threshold_factor * std_moving_std
        else:
            # Handle case where moving_std is all NaN (e.g., window > len(data))
            threshold = np.inf  # Set a threshold that won't be exceeded
            print("Warning: Could not calculate a valid threshold from moving_std.")

        # 3. Apply Threshold to identify bursts
        is_burst = moving_std > threshold

        # Track bursts as indices
        bursts = [i for i, val in enumerate(is_burst.values) if val]

        # 4. Penalize errors at noise burst locations
        # if bursts:
        #     penalised_errors[bursts] = 1e4
        #     print(f"Identified {len(bursts)} points as noise bursts")

        return bursts

bursts = ignore_noise_bursts([0,0], y_raw=y_raw, window_size=60,
                             threshold_factor=2.5)

# --- Visualization (Highly Recommended for Tuning) ---
fig, ax = plt.subplots(figsize=(15, 8))

# Plot Raw Data and Highlight Regions
ax.plot(date_time, y_raw, label='Raw Data (y_raw)', color='lightblue', zorder=1)
ax.plot(date_time[bursts], y_raw[bursts], '.', color='orangered',
        label='Points > Threshold', markersize=4, zorder=2)

ax.set_title('Raw Furnace Faults with Detected Noise Bursts')
ax.set_ylabel('Value')
ax.legend()
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Improve date formatting on x-axis
fig.autofmt_xdate()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

plt.tight_layout()
plt.show()