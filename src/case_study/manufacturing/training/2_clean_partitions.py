import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt
from pathlib import Path
# DPSGP
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.kernels import InducingPointKernel, ScaleKernel, RBFKernel as RBF
from models.dpsgp_torch import DirichletProcessSparseGaussianProcess as DPSGP

"""
NSG data

Do not adjust data for timelags.
"""

# NSG post processes data location
ROOT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT_PATH / "data" / "processed" / "Raw_data_partitions"
SAVE_PATH = ROOT_PATH / "data" / "processed" / "Training_data_partitions"

# Adaptive Moving Average parameters
tf_list = [1.5, 1.5, 2, 2, 2.5]

for i in range(5):
    file = PROCESSED_PATH / f'data{i}.xlsx'

    # Training df
    X_df = pd.read_excel(file, sheet_name='X_stand')
    y_df = pd.read_excel(file, sheet_name='y_nonstand')
    t_df = pd.read_excel(file, sheet_name='timelags')

    # Pre-Process training data
    X_train = X_df.values
    y_filtered = y_df.y_filtered.values
    N, D = np.shape(X_train)

    # Replace zero values with interpolation
    zeros = y_df.loc[y_df['y_raw'] <= 1e-1]
    y_df.loc[zeros.index, 'y_raw'] = None
    y_df.interpolate(inplace=True)

    y_train = y_df['y_raw'].values
    date_time = y_df['date_time'].values

    N_train = len(X_train)

    """ DPSGP cleaning """

    # Convert data to torch tensors to input inducing points
    floating_point = torch.float64
    X_tensor = torch.tensor(X_train, dtype=floating_point)
    inducing_points = X_tensor[::8, :]

    likelihood = GaussianLikelihood()

    se = ScaleKernel(RBF(ard_num_dims=X_train.shape[-1]))

    covar_module = InducingPointKernel(se,
                                    inducing_points=inducing_points,
                                    likelihood=likelihood)

    lss = [1.83, 0.8, 603, 0.3, 5.87e+04, 3.0, 2.17, 1.2e+03, 4.63, 1, 1.19e+04, 52.2, 663, 17.3]
    start_time = time.time()
    sgp = DPSGP(X_train, y_train, init_K=7,
                gp_model='Sparse',
                prior_mean=ConstantMean(), kernel=covar_module,
                lengthscale=lss,
                N_iter=15,
                noise_var = 0.06,
                floating_point=floating_point,
                normalise_y=True,
                DP_max_iter=390,
                threshold_factor=tf_list[i],
                print_conv=False, plot_conv=False, plot_sol=False)
    sgp.train()
    mu, stds = sgp.predict(X_train)
    comp_time = time.time() - start_time

    print(f'\nDPSGP cleaning time: {comp_time:.2f} seconds')

    # get inducing points indices
    _z_indices = sgp._z_indices

    print('N-train:   ', N_train)
    print('N-induced: ', len(_z_indices))

    # save predictions to use it in another scipt as the `true` fault_density
    cleaned_indices = sgp.indices[0]
    if i == 4:
        print(f'Shouldt be here, i: {i}')
        cleaned_indices = np.sort(np.append(cleaned_indices, sgp.indices[1]))
    dt_cleaned = date_time[cleaned_indices]
    y_cleaned = y_train[cleaned_indices]
    print('N-clean:   ', len(cleaned_indices))

    dx = {}
    for d, name in enumerate(X_df.columns):
        dx[name] = X_train[:, d]

    d = {"date_time": date_time, "y_raw": y_train, "gp_pred": mu, "y_filtered": y_filtered}

    # Raw data
    X_df = pd.DataFrame(dx)
    y_df = pd.DataFrame(d)

    # Cleaned data
    dx_clean = X_train[cleaned_indices]
    dy_clean = {"Indices": cleaned_indices, "date_time": dt_cleaned,
            "y_raw": y_cleaned}
    X_df_clean = pd.DataFrame(dx_clean)
    y_df_clean = pd.DataFrame(dy_clean)

    """ Save clean data """
    save_file = SAVE_PATH / f'clean{i}.xlsx'

    # Define an Excel writer object and the target file
    writer = pd.ExcelWriter(save_file)

    # Save to spreadsheet
    X_df.to_excel(writer, sheet_name='X_stand', index=False)
    pd.read_excel(file, sheet_name='X_stand').to_excel(writer,
                                                       sheet_name="X_norm",
                                                       index=False)
    y_df.to_excel(writer, sheet_name='y_nonstand', index=False)
    t_df.to_excel(writer, sheet_name='timelags', index=False)
    # Clean
    X_df_clean.to_excel(writer, sheet_name="X_stand_clean", index=False)
    y_df_clean.to_excel(writer, sheet_name='y_nonstand_clean', index=False)
    writer._save()

#-----------------------------------------------------------------------------
# REGRESSION PLOT
#-----------------------------------------------------------------------------
fig, ax = plt.subplots()

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
fig.autofmt_xdate()

ax.fill_between(date_time,
                mu + 3*stds, mu - 3*stds,
                alpha=0.5, color='lightcoral',
                label='3$\\sigma$')
ax.plot(date_time, y_train, color='grey', label='Raw')
ax.plot(date_time, y_filtered, color='blue', label='Filtered')
ax.plot(date_time, mu, color="red", linewidth = 2.5, label="Cleaned")

ax.vlines(
    x=date_time[::10],
    ymin=-0.5,
    ymax=y_train.max().item(),
    alpha=0.3,
    linewidth=1.5,
    ls='--',
    label="z0",
    color='grey'
)

ax.vlines(
    # Sparse clean data
    x=dt_cleaned[_z_indices],
    ymin=-0.5,
    ymax=y_train.max().item(),
    alpha=0.3,
    linewidth=1.5,
    label="z*",
    color='orange'
)
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

#-----------------------------------------------------------------------------
# CLUSTERING PLOT
#-----------------------------------------------------------------------------

# processes colors
color_iter = ['lightgreen', 'orange','red', 'brown', 'blue', 'black']

# DP-GP
enumerate_K = [i for i in range(sgp.K_opt)]

fig, ax = plt.subplots()
# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

fig.autofmt_xdate()
ax.set_title(" Clustering performance", fontsize=18)
if sgp.K_opt != 1:
    for i, (k, c) in enumerate(zip(enumerate_K, color_iter)):
        ax.plot(date_time[sgp.indices[k]], y_train[sgp.indices[k]],
                'x', color=c, markersize = 8, label='Noise level '+str(k))
ax.plot(dt_cleaned, y_train[cleaned_indices], 'o', color="lightgreen",
        linewidth = 2, label="Furnace")        
ax.plot(date_time, y_filtered, color="blue", linewidth = 2, label="y-filtered")
ax.plot(date_time, mu, color="green", linewidth = 2, label="DPSGP")
ax.set_xlabel(" Date-time", fontsize=14)
ax.set_ylabel(" Fault density", fontsize=14)
plt.legend(loc=0, prop={"size":18}, facecolor="white", framealpha=1.0)

plt.show()