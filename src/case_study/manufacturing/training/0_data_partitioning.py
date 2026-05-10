import copy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

"""
Load NSG data
"""

# NSG post processes data location
ROOT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT_PATH / "data" / "processed"
file = PROCESSED_PATH / 'NSG_processed_data.xlsx'

# Load csvs
X_df = pd.read_excel(file, sheet_name='X_stand')
y_df = pd.read_excel(file, sheet_name='y')
y_raw_df = pd.read_excel(file, sheet_name='y_raw')
t_df = pd.read_excel(file, sheet_name='timelags')

# Extract values
X    = X_df.values
y0   = y_df.furnace_faults.values
N, D = np.shape(X)

# Replace zero values with interpolation
zeros = y_raw_df.loc[y_raw_df['raw_furnace_faults'] <= 1e-1]
y_raw_df.loc[zeros.index, 'raw_furnace_faults'] = None
y_raw_df.interpolate(inplace=True)

y_raw = y_raw_df['raw_furnace_faults'].values
date_time = y_df['Time stamp'].values

# Get the sampling rate from the first two data points
step = date_time[1] - date_time[0]

structured_data = []
start_indices = [0]
end_indices = []
for i, time in enumerate(date_time):
    if i == len(date_time) - 1:
        break
    else:
        if date_time[i+1] - date_time [i] > step:
            # assemble structure data
            start_indices.append(i+1)
            end_indices.append(i)

end_indices.append(N)
dt_chunks = []
X_chunks = []
y0_chunks = []
y_raw_chunks = []
for i in range(len(start_indices)):
    dt = date_time[start_indices[i]: end_indices[i]]
    X_chunks.append(X[start_indices[i]: end_indices[i]])
    y0_chunks.append(y0[start_indices[i]: end_indices[i]])
    y_raw_chunks.append(y_raw[start_indices[i]: end_indices[i]])

    dt_chunks.append(dt)
    print(f'N per chunck: {len(dt)}')

print(f'\n20% of N({N}) = {0.2*N:2}\n')

# split big chunks and ignore noisy regions
start_train = 0
end_train = 3887

N_train_regions = 6

dt_struct = copy.deepcopy(dt_chunks)
X_struct = copy.deepcopy(X_chunks)
y0_struct = copy.deepcopy(y0_chunks)
y_raw_struct = copy.deepcopy(y_raw_chunks)

# 2nd region
dt_struct[1] = dt_chunks[2][start_train: end_train]
X_struct[1] = X_chunks[2][start_train: end_train]
y0_struct[1] = y0_chunks[2][start_train: end_train]
y_raw_struct[1] = y_raw_chunks[2][start_train: end_train]

# # # Noise burst
# # start_train = 2755
# # end_train = start_train + 456
# # # dt_chunks[3] = dt_chunks[2][start_train: end_train]
# # # X_chunks[3] = X_chunks[2][start_train: end_train]
# # # y0_chunks[3] = y0_chunks[2][start_train: end_train]
# # # y_raw_chuncks[3] = y_raw_chuncks[2][start_train: end_train]

# 3th region
start_train = 3887
end_train = 11662 - 3887
dt_struct[2] = dt_chunks[2][start_train: end_train]
X_struct[2] = X_chunks[2][start_train: end_train]
y0_struct[2] = y0_chunks[2][start_train: end_train]
y_raw_struct[2] = y_raw_chunks[2][start_train: end_train]

# 4th region
start_train = end_train
end_train = 11662
dt_struct[3] = dt_chunks[2][start_train: end_train]
X_struct[3] = X_chunks[2][start_train: end_train]
y0_struct[3] = y0_chunks[2][start_train: end_train]
y_raw_struct[3] = y_raw_chunks[2][start_train: end_train]

# 5th region
start_train = 0
end_train = int(len(dt_chunks[-1])/2) + 173
dt_struct.append(dt_chunks[3][start_train: end_train])
X_struct.append(X_chunks[3][start_train: end_train])
y0_struct.append(y0_chunks[3][start_train: end_train])
y_raw_struct.append(y_raw_chunks[3][start_train: end_train])

# final regions length
sum = 0
for i in range(len(dt_struct)):
    size = len(dt_struct[i])
    sum += size
    print(f'region {i}, size: {size}')
    
print('N-train: ', sum)
print(f'N-Train: {(sum*100)/N:.2f} %')

""" Save data partitions in individual xlsx files """

# --- Best Practice: Ensure parent directory exists before the loop ---
partitions_dir = PROCESSED_PATH / "Training_data_partitions"
partitions_dir.mkdir(parents=True, exist_ok=True)
print(f"Ensured directory exists: {partitions_dir}") # Optional: confirmation

# --- Adjust loop range if you need files 0 through 5 ---
# If you need data0 to data5, use range(N_train_regions)
loop_range = range(N_train_regions)

# for k in loop_range:
#     file_path = partitions_dir / f"data{k}.xlsx"
#     print(f"Processing k={k}, file path: {file_path}")

#     try:
#         # The 'with' statement creates and manages the writer object
#         with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
#             # --- DO NOT ADD 'writer = pd.ExcelWriter(file_path)' HERE ---

#             # Convert data for partition k to DataFrames
#             # Add checks in case list indices are out of bounds, though your setup seems okay for k=0..5
#             if k < len(X_struct) and k < len(dt_struct):
#                  dx = {}
#                  for d, name in enumerate(X_df.columns):
#                     dx[name] = X_struct[k][:, d]

#                  X_df = pd.DataFrame(dx)

#                  d = {"date_time": dt_struct[k], "y_raw": y_raw_struct[k],
#                       "y_filtered": y0_struct[k]}
#                  y_df = pd.DataFrame(d)

#                  # Write DataFrames to the 'writer' managed by the 'with' statement
#                  print(f"  Writing sheets for k={k}...")
#                  X_df.to_excel(writer, sheet_name='X_stand', index=False)
#                  y_df.to_excel(writer, sheet_name='y_nonstand', index=False)
#                  # This saves the globally loaded t_df to every file
#                  t_df.to_excel(writer, sheet_name='timelags', index=False)
#             else:
#                  print(f"  WARNING: k={k} is out of bounds for data structures (len(X_struct)={len(X_struct)}, len(dt_struct)={len(dt_struct)}). Skipping file.")
#                  continue # Skip to the next iteration

#         # When the 'with' block exits, the writer is automatically saved and closed.
#         print(f"  Successfully saved file: {file_path}")

#     except Exception as e:
#         print(f"  ERROR processing or writing file {file_path}: {e}")
#         # You might want to add more specific error handling here

# --- Remember to adjust the plotting loop range too if you changed the saving loop range ---
print("\nStarting plotting...")
plot_loop_range = loop_range # Use the same range for consistency

fig, ax = plt.subplots()
# ... (rest of your plotting setup) ...

# Plot full raw data
ax.plot(date_time, y_raw, color='grey', label='Raw')

# Define colors for the number of regions you are actually plotting
# Make sure 'c' has enough colors for your loop_range
colors = ['red', 'green', 'coral', 'orange', 'black', 'purple', 'brown'] # Add more if needed
if len(colors) < len(plot_loop_range):
    print("Warning: Not enough colors defined for all plot regions.")

# Use the same adjusted loop range for plotting
for k in plot_loop_range:
     if k < len(dt_struct) and k < len(y_raw_struct): # Check index bounds for plotting data too
        # Ensure you have enough colors in 'c' for the range
        plot_color = colors[k % len(colors)] # Cycle through colors if needed
        ax.plot(dt_struct[k], y_raw_struct[k], color=plot_color, label=f'Region {k}') # Modified label
     else:
        print(f"Skipping plot for region k={k}: Index out of bounds.")


ax.plot(date_time, y0, color='blue', label='Filtered')
# Check if start_indices is not empty before accessing it
if start_indices:
    ax.vlines(date_time[start_indices], max(y_raw) if len(y_raw)>0 else 0, min(y_raw) if len(y_raw)>0 else 0,
              color='red', label='Jumps')
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
# Adjust legend size dynamically or ensure it fits
ax.legend(loc=0, prop={"size":10}, facecolor="white", framealpha=1.0) # Reduced size slightly
plt.tight_layout() # Helps fit elements like labels
plt.show()