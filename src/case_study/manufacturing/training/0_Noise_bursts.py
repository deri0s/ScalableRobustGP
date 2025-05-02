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

# 1. Choose Window Size
window_size = 20 # Example: Adjust based on your data and expected burst duration

# 2. Calculate Moving Standard Deviation
y_raw_series = pd.Series(y_raw, index=pd.to_datetime(date_time)) # Ensure index is datetime
moving_std = y_raw_series.rolling(window=window_size, center=True,
                                  min_periods=1).std()

# 3. Determine Threshold (Example: using a multiple of the median moving std dev)
# Calculate median and std of the non-NaN moving_std values for robustness
valid_moving_std = moving_std.dropna()
if not valid_moving_std.empty:
    median_moving_std = valid_moving_std.median()
    std_moving_std = valid_moving_std.std()
    # Avoid threshold being NaN if std_moving_std is 0 (flat line)
    if pd.isna(std_moving_std) or std_moving_std == 0:
        std_moving_std = 1e-6 # Assign small value

    # Set threshold (e.g., median + 3 stds of the moving stds)
    threshold_factor = 2.0
    threshold = median_moving_std + threshold_factor * std_moving_std
else:
    # Handle case where moving_std is all NaN (e.g., window > len(data))
    threshold = np.inf # Set a threshold that won't be exceeded
    print("Warning: Could not calculate a valid threshold from moving_std.")

print(f"Calculated Threshold for Noise Burst Detection: {threshold:.4f}")

# 4. Apply Threshold
is_burst = moving_std > threshold

# --- Find Burst Regions for Visualization ---
burst_int = is_burst.astype(int)
burst_diff = burst_int.diff() # NaNs will be at the start

# Find start times: Point where diff changes to 1, or the very first point if it's a burst
start_mask = (burst_diff == 1)
if not is_burst.empty and burst_int.iloc[0] == 1:
    start_mask.iloc[0] = True # Handle burst starting at the very beginning
start_times = is_burst.index[start_mask]

# Find end times for shading: Point where diff changes to -1 (this is the *first* point AFTER the burst)
end_mask = (burst_diff == -1)
end_times_for_span = is_burst.index[end_mask]

# Handle burst ending at the very last point
if not is_burst.empty and burst_int.iloc[-1] == 1:
    # If the last point is True, we need an end time for the last span.
    # Use the next timestamp if possible, otherwise the last timestamp itself.
    if step > pd.Timedelta(seconds=0):
      last_end_time = is_burst.index[-1] + step
    else:
      # If step unknown, just use the last timestamp. Span might end visually slightly early.
      last_end_time = is_burst.index[-1]
    # Append only if needed (mismatched counts)
    if len(start_times) > len(end_times_for_span):
        end_times_for_span = end_times_for_span.append(pd.DatetimeIndex([last_end_time]))


# Ensure equal number of starts and ends for pairing
min_len = min(len(start_times), len(end_times_for_span))
if len(start_times) != len(end_times_for_span):
    print(f"Warning: Mismatch in burst start ({len(start_times)}) and end ({len(end_times_for_span)}) counts. Truncating to {min_len} pairs.")
    start_times = start_times[:min_len]
    end_times_for_span = end_times_for_span[:min_len]


# --- Visualization (Highly Recommended for Tuning) ---
fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True) # Use subplots axes directly

# Plot Raw Data and Highlight Regions
ax = axes[0]
ax.plot(y_raw_series.index, y_raw_series.values, label='Raw Data (y_raw)', color='lightblue', zorder=1)
ax.plot(y_raw_series.index[is_burst], y_raw_series.values[is_burst], '.', color='orangered', label='Points > Threshold', markersize=4, zorder=2) # Keep points for clarity

print(f'Are these indices? {is_burst.index}')
# bursts as indices
bursts = [i for i, val in enumerate(is_burst.index) if val]

def fill_missing_values(arr):
    arr_filled = arr.copy()
    for i in range(1, len(arr) - 1):
        if np.isnan(arr[i]) and (arr[i+1] - arr[i-1] < 10):
            arr_filled[i] = (arr[i-1] + arr[i+1]) / 2
    return arr_filled

print(f'Son indices? {bursts}')

# Add shaded regions for bursts
label_added = False
for start, end in zip(start_times, end_times_for_span):
    label = 'Detected Noise Burst Region' if not label_added else "_nolegend_"
    ax.axvspan(start, end, color='red', alpha=0.3, zorder=0, label=label)
    label_added = True

ax.set_title('Raw Furnace Faults with Detected Noise Bursts')
ax.set_ylabel('Value')
ax.legend()
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Plot Moving Standard Deviation and Threshold
ax = axes[1]
ax.plot(moving_std.index, moving_std, label=f'Moving Std Dev (window={window_size})', color='orange')
ax.axhline(threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.2f})')
ax.set_title('Moving Standard Deviation and Threshold')
ax.set_xlabel('Time Stamp')
ax.set_ylabel('Standard Deviation')
ax.legend()
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Improve date formatting on x-axis
fig.autofmt_xdate()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
# Optional: Adjust locator frequency if needed
# ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))

plt.tight_layout()
plt.show()

# --- Using the results for preprocessing ---
# The 'is_burst' boolean Series (aligned with date_time) can now be used.
# For example, you might want to exclude these points from training:
y_clean_target_df = y_df.copy() # Use the processed y_df as target base
if 'furnace_faults' in y_clean_target_df.columns:
    # Align is_burst index with y_clean_target_df index if they differ (e.g., due to missing values)
    # Assuming y_df['Time stamp'] is reliable and matches y_raw_df['Time stamp'] initially
    is_burst_aligned, _ = is_burst.align(y_clean_target_df.set_index('Time stamp'), join='right', fill_value=False)
    y_clean_target_df.loc[is_burst_aligned.values, 'furnace_faults'] = np.nan # Set target to NaN during bursts
    print(f"\nSet {int(is_burst_aligned.sum())} target points in 'y_clean_target_df' to NaN based on noise bursts.")
else:
    print("\nSkipping target cleaning: 'furnace_faults' column not found in y_df.")

# Example of accessing clean data for training:
# y_train = y_clean_target_df['furnace_faults'].dropna()