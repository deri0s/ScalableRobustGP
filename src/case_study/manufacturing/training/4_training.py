import copy
import os
from pathlib import Path

import gpytorch
import numpy as np
import pandas as pd
import torch
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.kernels import (
    AdditiveKernel,
    LinearKernel,
    MaternKernel,
    ScaleKernel,
    SpectralMixtureKernel,
)
from gpytorch.kernels import RBFKernel as RBF
from gpytorch.kernels import RQKernel as RQ
from gpytorch.likelihoods import GaussianLikelihood
from matplotlib import pyplot as plt
from models.svgp_auto_model_construction import SVGP
from scipy.spatial.distance import pdist, squareform
from scipy.stats import qmc
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler as ss

"""
Enhanced User Configuration
"""

# Enhanced configuration
data_index = 1  # expert3: Not working RQ,
M = 170  # Number of inducing points
N_sim = 100
kernel = "RQ+SM"  # Options: 'RBF', 'RQ', 'Matern52', 'RBF+RQ', 'RBF+SM'
use_log_space = True
use_early_stopping = True
training_iter = 300
learning_rate = 0.001  # Reduced for stability

# --- Spectral Mixture kernel settings ---
# Q: number of mixture components. Keep small (3–6) to avoid overfitting noise.
# Each component learns one dominant frequency from the FFT seed below.
SM_NUM_MIXTURES = 2  # Q — increase cautiously
SM_LBFGSB_MAX_ITER = 300  # inner L-BFGS-B iterations for SM fine-tuning
# SM_TIME_DIM: column index in X that represents the time / sequential dimension.
# The SM kernel is restricted to this single column via active_dims so its
# parameter count stays at 3*Q regardless of D (avoids a 2*Q*D parameter explosion).
# Set to whichever column in your feature matrix is the time index (0-based).
SM_TIME_DIM = 0

# Target-based early stopping parameters
mse_training_target = 0.009  # 0.006
mse_test_target = 0.008  # 0.008
use_target_early_stopping = True  # Set to False to disable target-based early stopping

# NSG post processes data location
ROOT_PATH = Path(__file__).resolve().parent.parent
PROCESSED_PATH = ROOT_PATH / "data" / "processed" / "Training_data_partitions"
EXPERT_PATH = ROOT_PATH / "trained" / "experts"

file = PROCESSED_PATH / f"data{data_index}.xlsx"


""" 1. Apply the corresponding timelags """


def align_inputs(x_df, y_df, t_series):
    xdeep = x_df.copy()
    ydeep = y_df.copy()
    # Ensure t_series values are numeric before finding max
    numeric_t_series = pd.to_numeric(t_series, errors="coerce").fillna(0)
    if numeric_t_series.empty:
        max_lag = 0
    else:
        max_lag = int(max(numeric_t_series))

    # X
    for name, lag in t_series.items():
        # Ensure lag is treated as integer for shift
        try:
            lag_int = int(float(lag))
            if lag_int > 0:  # Only shift if lag is positive
                xdeep[name] = xdeep[name].shift(lag_int)
        except ValueError:
            print(
                f"Warning: Could not convert lag '{lag}' for feature '{name}' to int. Skipping shift."
            )

    # Drop rows with NaNs introduced by shifting (only drop up to max_lag rows from top)
    xdeep = xdeep.iloc[max_lag:]  # More direct way to handle shift NaNs

    # y and date-time alignment
    # Ensure ydeep has enough rows before slicing
    if len(ydeep) >= max_lag:
        ydeep = ydeep.iloc[max_lag:].reset_index(drop=True)
    else:
        # Handle case where ydeep is shorter than max_lag (e.g., return empty DataFrames)
        print(
            f"Warning: y DataFrame length ({len(ydeep)}) is less than max_lag ({max_lag}). Alignment might be incorrect."
        )
        return pd.DataFrame(columns=x_df.columns), pd.DataFrame(columns=y_df.columns)

    # Ensure xdeep and ydeep have the same length after alignment
    common_len = min(len(xdeep), len(ydeep))
    xdeep = xdeep.iloc[:common_len].reset_index(drop=True)
    ydeep = ydeep.iloc[:common_len].reset_index(drop=True)

    return xdeep, ydeep


# Training df
X_df = pd.read_excel(file, sheet_name="X_stand")
y_df = pd.read_excel(file, sheet_name="y_nonstand")
t_df = pd.read_excel(file, sheet_name="timelags")
t_series = t_df.iloc[0, :]

X_df, y_df = align_inputs(X_df, y_df, t_df.iloc[0, :])

X_np = X_df.values
y_processed = y_df.y_processed.values
y_raw = y_df.y_raw.values
date_time = y_df.date_time.values

# Convert data to torch tensors
floating_point = torch.float64
X = torch.tensor(X_np, dtype=floating_point)
N, D = np.shape(X)


""" 2. Standardise outputs """

eval_perc = 0.2
end_train = N - int(N * eval_perc)

X_train = X[0:end_train]
date_train = date_time[0:end_train]
N_train = len(X_train)
y_train_nonstand = y_processed[0:end_train]

# Define X_test_np and y_test_nonstand correctly for evaluation metric
X_test = X[end_train:N]
y_test_nonstand = y_processed[end_train:N]

# Standardise outputs
y_train_reshape = y_train_nonstand.reshape(-1, 1)
scaler = ss()
scaler.fit(y_train_reshape)
y_stand_np = scaler.transform(y_train_reshape)

y_test_reshape = y_test_nonstand.reshape(-1, 1)
y_test_stand = scaler.transform(y_test_reshape)

y_train = torch.tensor(y_stand_np, dtype=floating_point).squeeze()
y_test = torch.tensor(y_test_stand, dtype=floating_point).squeeze()


""" 3. Median Heuristic Implementation """


def compute_median_heuristic(X_train, subsample_size=1000):
    """
    Compute the median heuristic for lengthscale initialization.

    Args:
        X_train: Training data tensor
        subsample_size: Maximum number of points to use for distance computation

    Returns:
        median_distances: Array of median distances for each dimension
        std_distances: Array of standard deviations for each dimension
    """
    X_np = X_train.numpy() if isinstance(X_train, torch.Tensor) else X_train
    n_samples, n_dims = X_np.shape

    # Subsample for computational efficiency if dataset is large
    if n_samples > subsample_size:
        indices = np.random.choice(n_samples, subsample_size, replace=False)
        X_subset = X_np[indices]
    else:
        X_subset = X_np

    print(f"Computing median heuristic using {X_subset.shape[0]} samples...")

    # Compute pairwise distances for each dimension separately
    median_distances = np.zeros(n_dims)
    std_distances = np.zeros(n_dims)

    for dim in range(n_dims):
        # Compute pairwise distances for this dimension
        dim_data = X_subset[:, dim : dim + 1]  # Keep 2D for pdist
        pairwise_dists = pdist(dim_data, metric="euclidean")

        # Remove zero distances (identical points)
        non_zero_dists = pairwise_dists[pairwise_dists > 1e-10]

        if len(non_zero_dists) > 0:
            median_distances[dim] = np.median(non_zero_dists)
            std_distances[dim] = np.std(non_zero_dists)
        else:
            # Fallback if all distances are zero
            median_distances[dim] = 1.0
            std_distances[dim] = 0.5

    return median_distances, std_distances


def compute_multi_scale_ls(X_train, subsample_size=1000):
    """Compute lengthscales for different scales using proper distance analysis"""
    X_np = X_train.numpy() if isinstance(X_train, torch.Tensor) else X_train
    n_samples, n_dims = X_np.shape

    if n_samples > subsample_size:
        indices = np.random.choice(n_samples, subsample_size, replace=False)
        X_subset = X_np[indices]
    else:
        X_subset = X_np

    # Compute all pairwise distances
    all_distances = []
    for dim in range(n_dims):
        dim_data = X_subset[:, dim : dim + 1]
        pairwise_dists = pdist(dim_data, metric="euclidean")
        non_zero_dists = pairwise_dists[pairwise_dists > 1e-10]
        all_distances.extend(non_zero_dists)

    all_distances = np.array(all_distances)

    # Use percentiles of ALL distances, not median distances
    short_scale = np.percentile(all_distances, 25)  # For RBF (short-range)
    long_scale = np.percentile(all_distances, 75)  # For RQ (long-range)

    return short_scale, long_scale


def setup_hyperspace(kernel_type, D, X_train, use_log_space=True):
    """
    Enhanced hyperparameter space setup using median heuristic for lengthscales.
    """
    # Compute median heuristic
    median_distances, std_distances = compute_median_heuristic(X_train)

    print(f"Median heuristic results:")
    print(f"  Median distances: {median_distances}")
    print(f"  Std distances: {std_distances}")

    # Base parameters for single kernels
    base_params = {
        "RBF": D + 2,  # lengthscales + outputscale + noise
        "RQ": D + 3,  # lengthscales + outputscale + alpha + noise
        "Matern52": D + 2,  # lengthscales + outputscale + noise
        "RBF+Lin": 2 * D + 2,  # 2 sets of lengthscales + 1 outputscales + noise
        "RBF+RQ": 2 * D
        + 4,  # RBF lengthscales + RQ lengthscales + 2 outputscales + alpha + noise
        "RBF+SM": D
        + 1
        + 3 * SM_NUM_MIXTURES
        + 1,  # D ls + 1 os + 3Q SM params + 1 noise
        "RQ+SM": D
        + 2
        + 3 * SM_NUM_MIXTURES
        + 1,  # D ls + 1 os + 1 alpha + 3Q SM params + 1 noise
    }

    dim = base_params.get(kernel_type)
    if dim is None:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")

    # RBF+SM has a dedicated setup that uses FFT seeding — delegate immediately
    if kernel_type == "RBF+SM":
        # Seed spectral frequencies from the TARGET signal (y_train), which directly
        # expresses the periodicities we want the SM kernel to model.
        # The SM kernel will then see the time column of X (SM_TIME_DIM) at eval time.
        y_np = y_train.numpy() if isinstance(y_train, torch.Tensor) else y_train
        dim, lowerb, upperb, _, _ = setup_hyperspace_sm(
            D, SM_NUM_MIXTURES, y_np, use_log_space=use_log_space
        )
        return dim, lowerb, upperb

    # RQ+SM: RQ (ARD, heavy-tail) + SM (temporal, FFT-seeded) — delegate immediately
    if kernel_type == "RQ+SM":
        # Same FFT-seeding strategy as RBF+SM.  The RQ part adds one extra scalar
        # alpha parameter (sampled linearly) on top of the RBF+SM layout.
        y_np = y_train.numpy() if isinstance(y_train, torch.Tensor) else y_train
        dim, lowerb, upperb, _, _ = setup_hyperspace_rq_sm(
            D, SM_NUM_MIXTURES, y_np, use_log_space=use_log_space
        )
        return dim, lowerb, upperb

    # Initialize bounds arrays
    lowerb = np.zeros(dim)
    upperb = np.zeros(dim)

    if use_log_space:
        # single kernel
        if kernel_type in ["RBF", "RQ", "Matern52"]:
            for i in range(D):
                #   lower_lengthscale = median_distances[i] * 2  # 10% of median
                #   upper_lengthscale = median_distances[i] * 100   # 10x median

                #   lower_lengthscale = max(lower_lengthscale, 0.5)
                #   upper_lengthscale = min(upper_lengthscale, 10)

                #   lower_lengthscale = max(median_distances[i] - std_distances[i], 0.01)
                #   upper_lengthscale = min(median_distances[i] + 10*std_distances[i], 100)

                #   lowerb[i] = np.log(lower_lengthscale)
                #   upperb[i] = np.log(upper_lengthscale)

                lowerb[i] = np.log(2)
                upperb[i] = np.log(10)

            # 2922 Closed Bottom Temperature - Downstream Working ...
            lowerb[0] = np.log(1e14)
            upperb[0] = np.log(8 * 1e14)

            # Furnace Load
            lowerb[4] = np.log(10)
            upperb[4] = np.log(100)
            # Tweel position
            lowerb[12] = np.log(80)
            upperb[12] = np.log(100)

            # Handle different kernel types
            if kernel_type in ["RBF", "Matern52"]:
                # outputscale bounds
                lowerb[-2] = np.log(1)
                upperb[-2] = np.log(15)
                # noise bounds
                lowerb[-1] = np.log(0.001)
                upperb[-1] = np.log(0.009)

            elif kernel_type == "RQ":
                # outputscale bounds
                lowerb[-3] = np.log(0.8)
                upperb[-3] = np.log(100)
                # alpha bounds (keep linear)
                lowerb[-2] = 2
                upperb[-2] = 10.0
                # noise bounds
                lowerb[-1] = np.log(0.0009)
                upperb[-1] = np.log(0.009)

        elif kernel_type == "RBF+RQ":
            short_scale, long_scale = compute_multi_scale_ls(X_train)

            # RBF lengthscales (short-range, tighter bounds)
            lowerb[0:D] = np.log(short_scale * 2)
            upperb[0:D] = np.log(short_scale * 100.0)

            # RQ lengthscales (long-range, wider bounds)
            lowerb[D : 2 * D] = np.log(long_scale * 20.0)
            upperb[D : 2 * D] = np.log(long_scale * 200.0)

            # account for zeros
            zero_idx = np.where(lowerb == 0)[0]
            lowerb[zero_idx] = np.log(0.5)
            zero_idx = np.where(upperb == 0)[0]
            upperb[zero_idx] = np.log(10)

            # outputscale
            lowerb[-3] = np.log(0.8)
            upperb[-3] = np.log(60)
            # RQ alpha (linear)
            lowerb[-2] = 2
            upperb[-2] = 20.0
            # noise bounds
            lowerb[-1] = np.log(0.0009)
            upperb[-1] = np.log(0.009)
    else:
        # Linear space bounds using median heuristic
        for i in range(D):
            lower_lengthscale = max(median_distances[i] - std_distances[i], 2.0)
            upper_lengthscale = min(median_distances[i] + 3 * std_distances[i], 500.0)
            lowerb[i] = lower_lengthscale
            upperb[i] = upper_lengthscale

        # Handle other parameters in linear space
        if kernel_type in ["RBF", "Matern52"]:
            lowerb[-2] = 15  # outputscale
            upperb[-2] = 50
            lowerb[-1] = 0.001  # noise
            upperb[-1] = 0.01
        elif kernel_type == "RQ":
            lowerb[-3] = 0.8  # outputscale
            upperb[-3] = 100
            lowerb[-2] = 5  # alpha
            upperb[-2] = 10.0
            lowerb[-1] = 0.001  # noise
            upperb[-1] = 0.01

    print(f"\nLengthscale bounds (log space: {use_log_space}):")
    feature_names = (
        X_df.columns.values
        if "X_df" in globals()
        else [f"Feature_{i}" for i in range(D)]
    )
    for i in range(min(D, len(feature_names))):
        if use_log_space:
            print(
                f"  {feature_names[i]}: [{np.exp(lowerb[i]):.3f}, {np.exp(upperb[i]):.3f}]"
            )
        else:
            print(f"  {feature_names[i]}: [{lowerb[i]:.3f}, {upperb[i]:.3f}]")

    return dim, lowerb, upperb


""" 3a. Spectral Mixture kernel — FFT-based initialisation utilities """


def fft_seed_spectral_means(y_train_np, dt=1.0, Q=4, plot=False):
    """
    Use the FFT of the (standardised) training signal to seed the spectral means
    (frequencies) of a Spectral Mixture kernel.

    The SM kernel has a power-spectral-density interpretation: each mixture
    component i is parameterised by
        μ_i  — the centre frequency  (spectral mean)
        v_i  — the bandwidth         (spectral variance, controls lengthscale)
        w_i  — the mixture weight    (outputscale contribution)

    Seeding μ from the dominant FFT peaks dramatically reduces the chance of
    the optimiser getting trapped in a flat region of the (highly non-convex)
    marginal-likelihood surface.

    Args:
        y_train_np : 1-D numpy array of *standardised* training targets.
        dt         : Sampling interval in whatever units your time index uses
                     (default 1.0 — normalised index).  If your data are hourly
                     set dt=1/24 to express frequencies in cycles-per-day.
        Q          : Number of mixture components (must match SM_NUM_MIXTURES).
        plot       : If True, plots the power spectrum with selected peaks.

    Returns:
        freqs_seed : (Q,) array of dominant frequencies  [cycles / unit time]
        bws_seed   : (Q,) array of bandwidths            [suggested v_i init]
        weights_seed:(Q,) array of normalised powers     [suggested w_i init]
    """
    N = len(y_train_np)
    # One-sided FFT — only positive frequencies carry information
    fft_vals = np.fft.rfft(y_train_np - y_train_np.mean())
    power = np.abs(fft_vals) ** 2
    freqs_all = np.fft.rfftfreq(N, d=dt)  # cycles per dt-unit

    # Exclude the DC component (index 0) — it is absorbed by the mean function
    power_pos = power[1:]
    freqs_pos = freqs_all[1:]

    # Pick the Q highest-power frequencies as seeds
    # Use a minimum-distance guard so we don't pick harmonics of the same peak
    selected_idx = []
    remaining = np.argsort(power_pos)[::-1]  # descending power order
    min_sep = max(1, len(freqs_pos) // (4 * Q))  # coarse spacing guard

    for idx in remaining:
        if all(abs(idx - s) >= min_sep for s in selected_idx):
            selected_idx.append(idx)
        if len(selected_idx) == Q:
            break

    # Fallback: if not enough well-separated peaks found, just take top-Q
    if len(selected_idx) < Q:
        selected_idx = list(np.argsort(power_pos)[::-1][:Q])

    selected_idx = np.array(selected_idx)
    freqs_seed = freqs_pos[selected_idx]
    weights_seed = power_pos[selected_idx] / power_pos[selected_idx].sum()

    # Bandwidth heuristic: set v_i ≈ half the distance to the nearest neighbour
    # in frequency space — gives a component that is neither too sharp nor too broad
    if Q > 1:
        sorted_f = np.sort(freqs_seed)
        diffs = np.diff(sorted_f)
        min_gap = diffs.min() if len(diffs) > 0 else freqs_seed.mean()
        bws_seed = np.full(Q, max(min_gap / 2.0, 1e-4))
    else:
        bws_seed = (
            np.array([freqs_seed[0] / 2.0]) if freqs_seed[0] > 0 else np.array([0.1])
        )

    # Ensure strictly positive
    freqs_seed = np.clip(freqs_seed, 1e-6, None)
    bws_seed = np.clip(bws_seed, 1e-6, None)

    print("\n[FFT Seed] Dominant spectral components for SM initialisation:")
    for q in range(Q):
        print(
            f"  Component {q + 1}: freq={freqs_seed[q]:.6f}, bw={bws_seed[q]:.6f}, "
            f"rel_weight={weights_seed[q]:.3f}"
        )

    if plot:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.semilogy(
            freqs_pos,
            power_pos,
            color="steelblue",
            linewidth=0.8,
            label="Power spectrum",
        )
        ax.scatter(
            freqs_seed,
            power_pos[selected_idx],
            color="red",
            zorder=5,
            label=f"Top-{Q} seeds",
        )
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Power")
        ax.legend()
        ax.set_title("FFT Power Spectrum — SM Seed Frequencies")
        plt.tight_layout()
        plt.show()

    return freqs_seed, bws_seed, weights_seed


def setup_hyperspace_sm(D, Q, y_train_np, use_log_space=True):
    """
    Build LHS bounds for the RBF + SM composite kernel.

    Parameter layout (total = D + 1 + 3*Q + 1 = D + 3*Q + 2):
      [0    : D      ]  — RBF ARD log-lengthscales       (D params, log)
      [D    : D+1    ]  — ScaleKernel outputscale         (1 param,  log)
      [D+1  : D+1+Q  ]  — SM spectral means  μ_q          (Q params, log)
      [D+1+Q: D+1+2Q ]  — SM spectral scales v_q          (Q params, log)
      [D+1+2Q:D+1+3Q ]  — SM mixture weights w_q          (Q params, log)
      [D+1+3Q]           — likelihood noise               (1 param,  log)

    NOTE: The ScaleKernel wraps (RBF + SM), so there is a single shared
    outputscale.  The SM component has its own internal mixture weights w_q
    that determine relative contributions within the SM part; the RBF
    contribution is modulated by its ARD lengthscales and the shared scale.
    """
    freqs_seed, bws_seed, _ = fft_seed_spectral_means(y_train_np, Q=Q)

    dim = D + 1 + 3 * Q + 1  # total: D ls + 1 os + Q freqs + Q bws + Q wts + 1 noise

    lowerb = np.zeros(dim)
    upperb = np.zeros(dim)

    # --- RBF ARD lengthscales (reuse existing heuristic bounds) ---
    lowerb[0:D] = np.log(2)
    upperb[0:D] = np.log(10)
    # Feature-specific overrides matching the existing single-RBF setup
    lowerb[0] = np.log(1e14)
    upperb[0] = np.log(8e14)  # Temp feature
    lowerb[4] = np.log(10)
    upperb[4] = np.log(100)  # Furnace load
    lowerb[12] = np.log(80)
    upperb[12] = np.log(100)  # Tweel position

    # --- Shared ScaleKernel outputscale ---
    idx_os = D
    lowerb[idx_os] = np.log(1.0)
    upperb[idx_os] = np.log(15.0)

    # --- SM: frequencies (seeded from FFT, ±1 octave search band) ---
    # Layout: [D+1 ... D+Q]
    idx_f = D + 1
    for q in range(Q):
        lowerb[idx_f + q] = np.log(max(freqs_seed[q] / 2.0, 1e-6))
        upperb[idx_f + q] = np.log(freqs_seed[q] * 2.0)

    # --- SM: bandwidths (v_q) — layout: [D+1+Q ... D+2Q] ---
    idx_v = D + 1 + Q
    for q in range(Q):
        lowerb[idx_v + q] = np.log(max(bws_seed[q] / 4.0, 1e-6))
        upperb[idx_v + q] = np.log(bws_seed[q] * 4.0)

    # --- SM: mixture weights (w_q) — layout: [D+1+2Q ... D+3Q] ---
    idx_w = D + 1 + 2 * Q
    lowerb[idx_w : idx_w + Q] = np.log(0.01)
    upperb[idx_w : idx_w + Q] = np.log(10.0)

    # --- Noise — layout: [D+1+3Q] = [-1] ---
    lowerb[-1] = np.log(0.001)
    upperb[-1] = np.log(0.009)

    return dim, lowerb, upperb, freqs_seed, bws_seed


def setup_hyperspace_rq_sm(D, Q, y_train_np, use_log_space=True):
    """
    Build LHS bounds for the RQ + SM composite kernel.

    The RQ kernel replaces RBF for the multi-variate part, adding one extra
    scalar parameter alpha (the mixture-scale / heavy-tail exponent).

    Parameter layout (total = D + 2 + 3*Q + 1 = D + 3*Q + 3):
      [0    : D      ]  — RQ ARD log-lengthscales            (D params, log)
      [D    : D+1    ]  — ScaleKernel outputscale             (1 param,  log)
      [D+1  : D+2    ]  — RQ alpha  (heavy-tail exponent)     (1 param,  LINEAR)
      [D+2  : D+2+Q  ]  — SM spectral means  μ_q              (Q params, log)
      [D+2+Q: D+2+2Q ]  — SM spectral scales v_q              (Q params, log)
      [D+2+2Q:D+2+3Q ]  — SM mixture weights w_q              (Q params, log)
      [D+2+3Q]           — likelihood noise                   (1 param,  log)

    Alpha is kept in *linear* space (consistent with how RBF+RQ treats it) so
    the LHS sampler explores it on a human-interpretable scale (typical range
    2–20).  Everything else mirrors setup_hyperspace_sm.
    """
    freqs_seed, bws_seed, _ = fft_seed_spectral_means(y_train_np, Q=Q)

    dim = D + 2 + 3 * Q + 1  # D ls + 1 os + 1 alpha + Q freqs + Q bws + Q wts + 1 noise

    lowerb = np.zeros(dim)
    upperb = np.zeros(dim)

    # --- RQ ARD lengthscales (same heuristic overrides as RBF+SM) ---
    lowerb[0:D] = np.log(2)
    upperb[0:D] = np.log(10)
    lowerb[0]  = np.log(1e14);  upperb[0]  = np.log(8e14)   # Temp feature
    lowerb[4]  = np.log(10);    upperb[4]  = np.log(100)     # Furnace load
    lowerb[12] = np.log(80);    upperb[12] = np.log(100)     # Tweel position

    # --- Shared ScaleKernel outputscale ---
    idx_os = D
    lowerb[idx_os] = np.log(1.0)
    upperb[idx_os] = np.log(15.0)

    # --- RQ alpha (linear space; larger alpha → closer to RBF behaviour) ---
    idx_alpha = D + 1
    lowerb[idx_alpha] = 2.0
    upperb[idx_alpha] = 20.0

    # --- SM: frequencies (FFT-seeded, ±1 octave) ---
    idx_f = D + 2
    for q in range(Q):
        lowerb[idx_f + q] = np.log(max(freqs_seed[q] / 2.0, 1e-6))
        upperb[idx_f + q] = np.log(freqs_seed[q] * 2.0)

    # --- SM: bandwidths ---
    idx_v = D + 2 + Q
    for q in range(Q):
        lowerb[idx_v + q] = np.log(max(bws_seed[q] / 4.0, 1e-6))
        upperb[idx_v + q] = np.log(bws_seed[q] * 4.0)

    # --- SM: mixture weights ---
    idx_w = D + 2 + 2 * Q
    lowerb[idx_w : idx_w + Q] = np.log(0.01)
    upperb[idx_w : idx_w + Q] = np.log(10.0)

    # --- Noise ---
    lowerb[-1] = np.log(0.001)
    upperb[-1] = np.log(0.009)

    return dim, lowerb, upperb, freqs_seed, bws_seed


def create_kernel(kernel_type, D, dtype):
    """Create kernel based on type specification"""
    if kernel_type == "RBF":
        return ScaleKernel(RBF(ard_num_dims=D, dtype=dtype))
    elif kernel_type == "RQ":
        return ScaleKernel(RQ(ard_num_dims=D, dtype=dtype))
    elif kernel_type == "Matern52":
        return ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=D, dtype=dtype))
    elif kernel_type == "RBF+Linear":
        return ScaleKernel(
            RBF(ard_num_dims=D, dtype=dtype) + LinearKernel(ard_num_dims=D, dtype=dtype)
        )
    elif kernel_type == "RBF+RQ":
        return ScaleKernel(
            RBF(ard_num_dims=D, dtype=dtype) + RQ(ard_num_dims=D, dtype=dtype)
        )
    elif kernel_type == "RBF+SM":
        # ------------------------------------------------------------------
        # RBF + Spectral Mixture composite kernel
        # ------------------------------------------------------------------
        # Design rationale:
        #   • The RBF part operates on ALL D input features via ARD lengthscales,
        #     capturing smooth multi-variate covariance.
        #   • The SM part captures periodic / quasi-periodic TEMPORAL structure.
        #     It is restricted to a single time-index dimension via `active_dims`,
        #     so its parameter count stays at Q*(1+1+1) = 3Q regardless of D.
        #     Using ard_num_dims=D would give Q*D means + Q*D scales = 2QD extra
        #     parameters — a completely intractable LHS search space for D=14.
        #
        # SM_TIME_DIM (default 0) selects which column of X is the time index.
        # Adjust this constant at the top of the file if your time feature is
        # at a different column position.
        # ------------------------------------------------------------------
        rbf_part = RBF(ard_num_dims=D)
        sm_part = SpectralMixtureKernel(
            num_mixtures=SM_NUM_MIXTURES,
            ard_num_dims=1,
            active_dims=torch.tensor([SM_TIME_DIM]),  # restrict SM to 1-D time column
        )
        rbf_part = rbf_part.to(dtype)
        sm_part = sm_part.to(dtype)
        return ScaleKernel(rbf_part + sm_part)
    elif kernel_type == "RQ+SM":
        # ------------------------------------------------------------------
        # RQ + Spectral Mixture composite kernel
        # ------------------------------------------------------------------
        # Design rationale:
        #   • The RQ kernel replaces RBF as the multi-variate component.
        #     RQ is an infinite mixture of RBF kernels with different
        #     lengthscales (parameterised by alpha), making it strictly more
        #     expressive and naturally robust to multi-scale variation in the
        #     input features (heavy-tail covariance decay).
        #   • The SM part is identical to the RBF+SM case — restricted to
        #     the single time column (SM_TIME_DIM) via active_dims, keeping
        #     the parameter count at 3*Q regardless of D.
        #   • Together: RQ captures smooth multi-variate covariance with
        #     variable-scale lengthscales; SM captures temporal periodicity.
        # ------------------------------------------------------------------
        rq_part = RQ(ard_num_dims=D)
        sm_part = SpectralMixtureKernel(
            num_mixtures=SM_NUM_MIXTURES,
            ard_num_dims=1,
            active_dims=torch.tensor([SM_TIME_DIM]),  # restrict SM to 1-D time column
        )
        rq_part = rq_part.to(dtype)
        sm_part = sm_part.to(dtype)
        return ScaleKernel(rq_part + sm_part)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")


def get_hyper(gp):
    """Extract hyperparameters based on kernel type"""
    results = {}

    if hasattr(gp.covar_module.base_kernel, "kernels"):  # Additive kernel
        results["kernel_type"] = "additive"
        results[f"outputscale"] = gp.covar_module.outputscale.item()
        for i, k in enumerate(gp.covar_module.base_kernel.kernels):
            results[f"kernel_{i}_name"] = k.__class__.__name__
            # SpectralMixtureKernel has no lengthscale property — guard explicitly
            ls = getattr(k, "lengthscale", None)
            if ls is not None:
                results[f"kernel_{i}_lengthscales"] = ls.squeeze().tolist()
            if hasattr(k, "alpha"):
                results[f"kernel_{i}_alpha"] = k.alpha.item()
    else:  # Single kernel
        results["kernel_type"] = "single"
        results["outputscale"] = gp.covar_module.outputscale.item()
        if hasattr(gp.covar_module.base_kernel, "lengthscale"):
            results["lengthscales"] = (
                gp.covar_module.base_kernel.lengthscale.squeeze().tolist()
            )
        if hasattr(gp.covar_module.base_kernel, "alpha"):
            results["alpha"] = gp.covar_module.base_kernel.alpha.item()

    # --- RBF+SM: extract SM-specific parameters ---
    if hasattr(gp.covar_module.base_kernel, "kernels"):
        for k in gp.covar_module.base_kernel.kernels:
            if isinstance(k, SpectralMixtureKernel):
                results["sm_mixture_means"] = (
                    k.mixture_means.detach().squeeze().tolist()
                )
                results["sm_mixture_scales"] = (
                    k.mixture_scales.detach().squeeze().tolist()
                )
                results["sm_mixture_weights"] = (
                    k.mixture_weights.detach().squeeze().tolist()
                )
                break

    results["noise"] = gp.likelihood.noise.item()
    return results


def set_hyperparameters(gp, likelihood, sample, kernel_type, D, use_log_space=True):
    """Enhanced hyperparameter setting for multiple kernel types"""

    if kernel_type == "RBF" or kernel_type == "Matern52":
        if use_log_space:
            lengthscales = np.exp(sample[0:D])
            outputscale = np.exp(sample[-2])
            noise = np.exp(sample[-1])
        else:
            lengthscales = sample[0:D]
            outputscale = sample[-2]
            noise = sample[-1]

        gp.covar_module.base_kernel.lengthscale = torch.tensor(
            lengthscales, dtype=floating_point
        )
        gp.covar_module.outputscale = torch.tensor(outputscale, dtype=floating_point)
        likelihood.noise = torch.tensor(noise, dtype=floating_point)

    elif kernel_type == "RQ":
        if use_log_space:
            lengthscales = np.exp(sample[0:D])
            outputscale = np.exp(sample[-3])
            alpha = sample[-2]  # Keep linear
            noise = np.exp(sample[-1])
        else:
            lengthscales = sample[0:D]
            outputscale = sample[-3]
            alpha = sample[-2]
            noise = sample[-1]
    elif kernel_type == "RBF+RQ":
        if use_log_space:
            ls_se = np.exp(sample[0:D])
            ls_rq = np.exp(sample[D : 2 * D])
            outputscale = np.exp(sample[-3])
            alpha = sample[-2]  # Keep linear
            noise = np.exp(sample[-1])

            gp.covar_module.outputscale = torch.tensor(
                outputscale, dtype=floating_point
            )
            # RBF
            gp.covar_module.base_kernel.kernels[0].lengthscale = torch.tensor(
                ls_se, dtype=floating_point
            )
            # RQ
            gp.covar_module.base_kernel.kernels[1].alpha = torch.tensor(
                alpha, dtype=floating_point
            )
            gp.covar_module.base_kernel.kernels[1].lengthscale = torch.tensor(
                ls_rq, dtype=floating_point
            )
            # Noise
            likelihood.noise = torch.tensor(noise, dtype=floating_point)
        else:
            print("\n !!! this kernel only works on the log-space for now")

    elif kernel_type == "RBF+SM":
        # ------------------------------------------------------------------
        # Hyperparameter layout (log-space only) — must match setup_hyperspace_sm:
        #   sample[0:D]            → RBF ARD lengthscales
        #   sample[D]              → shared ScaleKernel outputscale
        #   sample[D+1 : D+1+Q]   → SM spectral means   (frequencies μ_q)
        #   sample[D+1+Q : D+1+2Q]→ SM spectral scales  (bandwidths  v_q)
        #   sample[D+1+2Q:D+1+3Q] → SM mixture weights  (w_q)
        #   sample[-1]             → noise
        # ------------------------------------------------------------------
        Q = SM_NUM_MIXTURES
        if not use_log_space:
            print(" !!! RBF+SM only supports log-space sampling")
            return

        ls_rbf = np.exp(sample[0:D])
        outputscale = np.exp(sample[D])
        sm_means = np.exp(sample[D + 1 : D + 1 + Q])  # (Q,)
        sm_scales = np.exp(sample[D + 1 + Q : D + 1 + 2 * Q])  # (Q,)
        sm_weights = np.exp(
            sample[D + 1 + 2 * Q : D + 1 + 3 * Q]
        )  # (Q,) raw; SM normalises
        noise = np.exp(sample[-1])

        # Identify sub-kernels
        rbf_kernel = gp.covar_module.base_kernel.kernels[0]
        sm_kernel = gp.covar_module.base_kernel.kernels[1]

        # RBF lengthscales (ARD, shape [1, D])
        rbf_kernel.lengthscale = torch.tensor(ls_rbf, dtype=floating_point).unsqueeze(0)

        # Shared outputscale
        gp.covar_module.outputscale = torch.tensor(outputscale, dtype=floating_point)

        # SM parameters
        # With ard_num_dims=1 (single time column via active_dims):
        #   mixture_means  shape: [Q, 1]
        #   mixture_scales shape: [Q, 1]
        #   mixture_weights shape: [Q]
        with torch.no_grad():
            sm_kernel.mixture_means.data = torch.tensor(
                sm_means, dtype=floating_point
            ).view(Q, 1)
            sm_kernel.mixture_scales.data = torch.tensor(
                sm_scales, dtype=floating_point
            ).view(Q, 1)
            sm_kernel.mixture_weights.data = torch.tensor(
                sm_weights, dtype=floating_point
            ).view(Q)

        likelihood.noise = torch.tensor(noise, dtype=floating_point)

    elif kernel_type == "RQ+SM":
        # ------------------------------------------------------------------
        # Hyperparameter layout (log-space only) — must match setup_hyperspace_rq_sm:
        #   sample[0:D]              → RQ ARD lengthscales           (log)
        #   sample[D]                → shared ScaleKernel outputscale (log)
        #   sample[D+1]              → RQ alpha                       (LINEAR)
        #   sample[D+2 : D+2+Q]     → SM spectral means   (μ_q)      (log)
        #   sample[D+2+Q : D+2+2Q]  → SM spectral scales  (v_q)      (log)
        #   sample[D+2+2Q:D+2+3Q]   → SM mixture weights  (w_q)      (log)
        #   sample[-1]               → noise                          (log)
        # ------------------------------------------------------------------
        Q = SM_NUM_MIXTURES
        if not use_log_space:
            print(" !!! RQ+SM only supports log-space sampling")
            return

        ls_rq     = np.exp(sample[0:D])
        outputscale = np.exp(sample[D])
        alpha     = sample[D + 1]                               # linear — no exp
        sm_means  = np.exp(sample[D + 2 : D + 2 + Q])          # (Q,)
        sm_scales = np.exp(sample[D + 2 + Q : D + 2 + 2 * Q])  # (Q,)
        sm_weights = np.exp(sample[D + 2 + 2 * Q : D + 2 + 3 * Q])  # (Q,)
        noise = np.exp(sample[-1])

        # Identify sub-kernels (RQ is kernels[0], SM is kernels[1])
        rq_kernel = gp.covar_module.base_kernel.kernels[0]
        sm_kernel = gp.covar_module.base_kernel.kernels[1]

        # RQ lengthscales (ARD, shape [1, D])
        rq_kernel.lengthscale = torch.tensor(ls_rq, dtype=floating_point).unsqueeze(0)

        # RQ alpha (scalar; GPyTorch constrains it to be > 0 via softplus)
        rq_kernel.alpha = torch.tensor(alpha, dtype=floating_point)

        # Shared outputscale
        gp.covar_module.outputscale = torch.tensor(outputscale, dtype=floating_point)

        # SM parameters (ard_num_dims=1, restricted to SM_TIME_DIM via active_dims)
        with torch.no_grad():
            sm_kernel.mixture_means.data = torch.tensor(
                sm_means, dtype=floating_point
            ).view(Q, 1)
            sm_kernel.mixture_scales.data = torch.tensor(
                sm_scales, dtype=floating_point
            ).view(Q, 1)
            sm_kernel.mixture_weights.data = torch.tensor(
                sm_weights, dtype=floating_point
            ).view(Q)

        likelihood.noise = torch.tensor(noise, dtype=floating_point)


def evaluate_model(gp, likelihood, X_test, y_test_nonstand, scaler):
    """Comprehensive model evaluation"""
    gp.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_dist = likelihood(gp(X_test))
        pred_mean = pred_dist.mean
        pred_var = pred_dist.variance

        # Convert back to original scale
        pred_mean_orig = scaler.inverse_transform(pred_mean.unsqueeze(1))[:, 0]
        pred_std_orig = np.sqrt(pred_var.numpy()) * scaler.scale_[0]

        # Calculate metrics
        mse = mean_squared_error(y_test_nonstand, pred_mean_orig)
        mae = mean_absolute_error(y_test_nonstand, pred_mean_orig)
        r2 = r2_score(y_test_nonstand, pred_mean_orig)

        # Mean prediction interval width (as measure of uncertainty)
        mean_uncertainty = np.mean(2 * 1.96 * pred_std_orig)  # 95% CI width

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "mean_uncertainty": mean_uncertainty,
        "predictions": pred_mean_orig,
        "std": pred_std_orig,
    }


# Enhanced inducing point initialization
import warnings

init_ip_method = "kmeans++"

if init_ip_method == "random":
    indices = np.random.choice(N_train, min(M, N_train), replace=False)
    inducing_points = X_train[indices, :]
else:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Try multiple K-means initializations
        best_inertia = float("inf")
        best_centers = None
        for _ in range(5):  # Multiple attempts
            kmeans = KMeans(
                n_clusters=M, init="k-means++", n_init=10, random_state=None
            )
            kmeans.fit(X_train)
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_centers = kmeans.cluster_centers_
        inducing_points = torch.tensor(best_centers, dtype=floating_point)

print(f"Inducing points shape: {inducing_points.shape}")
print(f"K-means inertia: {best_inertia:.4f}")

# Setup hyperparameter space with median heuristic
dim, lowerb, upperb = setup_hyperspace(kernel, D, X_train, use_log_space)
print("\nlower:\n", lowerb)
print("\nupper:\n", upperb)

print(f"\nUsing {kernel} kernel")
if use_target_early_stopping:
    print(f"Target-based early stopping enabled:")
    print(f"  Training MSE target: {mse_training_target}")
    print(f"  Test MSE target: {mse_test_target}")


# Generate samples
def get_samples(D, N_sim, l_bounds, u_bounds):
    """Samples from a Latin Hypercube Sampling model"""
    sampler = qmc.LatinHypercube(d=D)
    sample = sampler.random(n=N_sim)

    if len(l_bounds) != D or len(u_bounds) != D:
        raise ValueError(
            f"Bounds dimensions ({len(l_bounds)}, {len(u_bounds)}) must match sample dimension ({D})"
        )

    return qmc.scale(sample, l_bounds, u_bounds)


samples = get_samples(dim, N_sim, lowerb, upperb)

best_metrics = {"mse": float("inf")}
best_gp = None
best_likelihood = None
target_reached = False

# Enhanced training loop with target-based early stopping
print("\nStarting hyperparameter optimisation with median heuristic...")
for n in range(N_sim):
    # Create fresh model
    k_fresh = create_kernel(kernel, D, floating_point)
    gp_temp = SVGP(inducing_points, D, k_fresh)
    likelihood_temp = GaussianLikelihood(
        noise_constraint=Interval(1e-6, 0.1)
    )  # Stricter noise constraint
    gp_temp.likelihood = likelihood_temp
    gp_temp.to(floating_point)

    # Set hyperparameters
    set_hyperparameters(gp_temp, likelihood_temp, samples[n], kernel, D, use_log_space)

    # Training
    gp_temp.train()
    likelihood_temp.train()

    # Enhanced optimiser with scheduling
    optimizer = torch.optim.Adam(gp_temp.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=20, factor=0.8
    )

    # ELBO loss
    mll = gpytorch.mlls.VariationalELBO(
        likelihood_temp, gp_temp, num_data=X_train.size(0)
    )

    # Training with early stopping
    prev_loss = float("inf")
    patience_counter = 0
    patience = 30

    for i in range(training_iter):
        optimizer.zero_grad()
        output = gp_temp(X_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()

        # Early stopping and learning rate scheduling
        if use_early_stopping and i > 20:
            if loss.item() > prev_loss - 1e-6:
                patience_counter += 1
            else:
                patience_counter = 0

            if patience_counter >= patience:
                break

        prev_loss = loss.item()
        scheduler.step(loss)

    # Comprehensive evaluation
    test_metrics = evaluate_model(
        gp_temp, likelihood_temp, X_test, y_test_nonstand, scaler
    )
    train_metrics = evaluate_model(gp_temp, likelihood_temp, X, y_processed, scaler)

    # if train_metrics['mse'] < best_metrics['mse']:
    if test_metrics["mse"] < best_metrics["mse"]:
        best_metrics = test_metrics
        # best_metrics = train_metrics
        best_gp = copy.deepcopy(gp_temp)
        best_likelihood = copy.deepcopy(likelihood_temp)
        print(
            f"Sim: {n + 1}/{N_sim}, New best - Test MSE: {test_metrics['mse']:.6f}, Train MSE: {train_metrics['mse']:.6f}"
        )

        # Check if target MSE values are reached
        if (
            use_target_early_stopping
            and train_metrics["mse"] <= mse_training_target
            and test_metrics["mse"] <= mse_test_target
        ):
            target_reached = True
            print(f"\n🎯 TARGET REACHED! Stopping early at simulation {n + 1}/{N_sim}")
            print(
                f"   Training MSE: {train_metrics['mse']:.6f} <= {mse_training_target}"
            )
            print(f"   Test MSE: {test_metrics['mse']:.6f} <= {mse_test_target}")
            break

    if (n + 1) % 50 == 0:
        print(f"Completed {n + 1}/{N_sim}, Best MSE: {best_metrics['mse']:.6f}")

# Generate full predictions with best model
best_gp.eval()
best_likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    full_pred = best_likelihood(best_gp(X))
    best_mu = scaler.inverse_transform(full_pred.mean.unsqueeze(1))[:, 0]
    best_std = np.sqrt(full_pred.variance.numpy()) * scaler.scale_[0]


""" 3b. RBF+SM — L-BFGS-B fine-tuning of the best candidate

    Rationale
    ---------
    The marginal-likelihood (ELBO) surface for a Spectral Mixture kernel is
    highly non-convex: it contains many narrow, sharp valleys corresponding to
    different periodic modes.  Adam with a fixed learning rate tends to hover
    around a broad basin found during the LHS sweep but rarely descends into the
    narrow local optima where SM parameters live.

    L-BFGS-B uses second-order curvature information (via limited-memory
    quasi-Newton updates) and a line-search, making it far more effective for
    these sharp, nearly flat regions.  We run it as a *refinement* step on the
    single best model found by the LHS sweep rather than from scratch, so the
    warm start keeps us in the right basin while L-BFGS-B sharpens the fit.

    The SciPy interface requires scalar NumPy values, so we wrap the ELBO in a
    closure that converts back and forth between flat NumPy vectors and the GP's
    named parameters.
"""

if kernel in {"RBF+SM", "RQ+SM"}:
    from scipy.optimize import minimize as scipy_minimize

    print("\n" + "=" * 60)
    print(f"{kernel} — L-BFGS-B fine-tuning of the best candidate")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Pack/unpack in GPyTorch's RAW (unconstrained) parameter space       #
    # ------------------------------------------------------------------ #
    # GPyTorch applies constraint transforms (softplus, exp, etc.) between
    # the raw buffers it optimises and the constrained values you read back
    # via .lengthscale, .noise, etc.  The ELBO and its .backward() work in
    # raw space — so the L-BFGS-B closure must too.  Iterating over
    # .parameters() returns the RAW tensors (the ones that accumulate grads);
    # we pin the exact list once at construction time so pack and unpack
    # always visit parameters in the same order.

    def _get_raw_params(gp, likelihood_obj):
        """Return an ordered list of (name, raw_parameter_tensor) pairs."""
        named = list(gp.named_parameters()) + list(likelihood_obj.named_parameters())
        return named  # each tensor IS the raw/unconstrained buffer

    def _pack_sm_params(gp, likelihood_obj):
        named = _get_raw_params(gp, likelihood_obj)
        return np.concatenate([p.detach().numpy().ravel() for _, p in named])

    def _unpack_sm_params(vec, gp, likelihood_obj):
        named = _get_raw_params(gp, likelihood_obj)
        offset = 0
        for _, p in named:
            numel = p.numel()
            p.data = torch.tensor(
                vec[offset : offset + numel], dtype=floating_point
            ).reshape(p.shape)
            offset += numel

    # ------------------------------------------------------------------ #
    # Deep-copy the best model so a failed refinement doesn't corrupt it  #
    # ------------------------------------------------------------------ #
    refined_gp = copy.deepcopy(best_gp)
    refined_likelihood = copy.deepcopy(best_likelihood)

    refined_gp.train()
    refined_likelihood.train()

    mll_refine = gpytorch.mlls.VariationalELBO(
        refined_likelihood, refined_gp, num_data=X_train.size(0)
    )

    x0 = _pack_sm_params(refined_gp, refined_likelihood)

    # ------------------------------------------------------------------ #
    # Objective + gradient closure for scipy.optimize.minimize            #
    # ------------------------------------------------------------------ #
    _n_calls = [0]
    # Pin the parameter list once — must be same order as _pack_sm_params
    _named_params = _get_raw_params(refined_gp, refined_likelihood)

    def _elbo_and_grad(vec):
        _unpack_sm_params(vec, refined_gp, refined_likelihood)

        # Zero gradients on the pinned list
        for _, p in _named_params:
            if p.grad is not None:
                p.grad.zero_()

        output = refined_gp(X_train)
        loss = -mll_refine(output, y_train)
        loss.backward()

        # Collect in the same order as pack
        grads = []
        for _, p in _named_params:
            g = (
                p.grad.detach().numpy().ravel()
                if p.grad is not None
                else np.zeros(p.numel())
            )
            grads.append(g)
        grad_vec = np.concatenate(grads)

        _n_calls[0] += 1
        if _n_calls[0] % 10 == 0:
            print(f"  L-BFGS-B step {_n_calls[0]:4d}  |  ELBO loss: {loss.item():.6f}")

        return float(loss.item()), grad_vec.astype(np.float64)

    # ------------------------------------------------------------------ #
    # Run L-BFGS-B                                                         #
    # ------------------------------------------------------------------ #
    lbfgsb_result = scipy_minimize(
        fun=_elbo_and_grad,
        x0=x0.astype(np.float64),
        method="L-BFGS-B",
        jac=True,  # gradient is returned as part of fun
        options={
            "maxiter": SM_LBFGSB_MAX_ITER,
            "ftol": 1e-9,  # tight function tolerance for SM landscape
            "gtol": 1e-6,
            "disp": False,
        },
    )

    print(
        f"\n  L-BFGS-B converged: {lbfgsb_result.success}  "
        f"(status={lbfgsb_result.status}, msg='{lbfgsb_result.message}')"
    )
    print(f"  Total function evaluations: {lbfgsb_result.nfev}")

    # Write the optimal parameters back
    _unpack_sm_params(lbfgsb_result.x, refined_gp, refined_likelihood)

    # ------------------------------------------------------------------ #
    # Compare refined vs LHS-best on the test set; keep the winner        #
    # ------------------------------------------------------------------ #
    refined_metrics = evaluate_model(
        refined_gp, refined_likelihood, X_test, y_test_nonstand, scaler
    )

    print(f"\n  LHS-best   test MSE : {best_metrics['mse']:.6f}")
    print(f"  Refined    test MSE : {refined_metrics['mse']:.6f}")

    if refined_metrics["mse"] < best_metrics["mse"]:
        print(
            "  ✅  L-BFGS-B refinement improved performance — adopting refined model."
        )
        best_gp = refined_gp
        best_likelihood = refined_likelihood
        best_metrics = refined_metrics
    else:
        print(
            "  ℹ️   L-BFGS-B refinement did not improve performance — retaining LHS-best model."
        )

    # Recompute full predictions with (possibly updated) best model
    best_gp.eval()
    best_likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        full_pred = best_likelihood(best_gp(X))
        best_mu = scaler.inverse_transform(full_pred.mean.unsqueeze(1))[:, 0]
        best_std = np.sqrt(full_pred.variance.numpy()) * scaler.scale_[0]


""" 4. Results analysis """

print("\n" + "=" * 60)
print("OPTIMISATION RESULTS WITH MEDIAN HEURISTIC")
print("=" * 60)
if target_reached:
    print("✅ TARGET MSE VALUES ACHIEVED!")
    print(f"Stopped early after {n + 1} simulations (out of {N_sim})")
else:
    print("⚠️  Target MSE values not reached in all simulations")
    print(f"Completed all {N_sim} simulations")

print(f"Final test MSE: {best_metrics['mse']:.6f}")
print(f"Final test R²: {best_metrics['r2']:.4f}")
print(f"Final test MAE: {best_metrics['mae']:.6f}")
print(f"Mean prediction uncertainty: {best_metrics['mean_uncertainty']:.4f}")

hyperparams = get_hyper(best_gp)

print("\nHyperparameters:")
for key, value in hyperparams.items():
    if not isinstance(value, list):
        print(f"{key}: {value}")

# --- SM-specific hyperparameter report ---
if kernel in {"RBF+SM", "RQ+SM"} and "sm_mixture_means" in hyperparams:
    print(f"\nSpectral Mixture Components ({kernel}):")
    Q = SM_NUM_MIXTURES
    means = hyperparams["sm_mixture_means"]
    scales = hyperparams["sm_mixture_scales"]
    weights = hyperparams["sm_mixture_weights"]
    means = [means] if not isinstance(means, list) else means
    scales = [scales] if not isinstance(scales, list) else scales
    weights = [weights] if not isinstance(weights, list) else weights
    for q in range(len(means)):
        print(
            f"  Component {q + 1}: freq={means[q]:.6f}, bw={scales[q]:.6f}, "
            f"weight={weights[q]:.4f}"
        )

# --- RQ+SM: also report RQ alpha ---
if kernel == "RQ+SM":
    rq_alpha_key = "kernel_0_alpha"
    if rq_alpha_key in hyperparams:
        print(f"  RQ alpha (heavy-tail exponent): {hyperparams[rq_alpha_key]:.4f}")

# Feature importance
print("\nFeature Importance (sorted by lengthscale):")
if hyperparams["kernel_type"] == "additive":
    # For RBF+SM / RQ+SM the first sub-kernel is always the multi-variate part
    k0_ls_key = "kernel_0_lengthscales"
    k0_name   = hyperparams.get("kernel_0_name", "kernel_0")
    if k0_ls_key in hyperparams:
        feature_importance = pd.DataFrame(
            {"inputs": X_df.columns.values, "lengthscales": hyperparams[k0_ls_key]}
        )
        print(f"\n{k0_name} component lengthscales:")
        print(feature_importance.sort_values(by="lengthscales"))
    for i in range(2):
        name = hyperparams.get(f"kernel_{i}_name", f"kernel_{i}")
        ls_key = f"kernel_{i}_lengthscales"
        if name == "SpectralMixtureKernel":
            continue  # SM lengthscales not ARD; already reported above
        if ls_key in hyperparams and name not in {"RBFKernel", "RQKernel"}:
            fi = pd.DataFrame(
                {"inputs": X_df.columns.values, "lengthscales": hyperparams[ls_key]}
            )
            print(f"\nKernel: {name}")
            print(fi.sort_values(by="lengthscales"))
else:
    feature_importance = pd.DataFrame(
        {"inputs": X_df.columns.values, "lengthscales": hyperparams["lengthscales"]}
    )
    print("\nFeature Importance (sorted by lengthscale):")
    print(feature_importance.sort_values(by="lengthscales"))

# Training vs test performance comparison
train_metrics = evaluate_model(
    best_gp, best_likelihood, X_train, y_train_nonstand, scaler
)
print(f"\nTraining Performance:")
print(f"Train MSE: {train_metrics['mse']:.6f}, Test MSE: {best_metrics['mse']:.6f}")
print(f"Train R²: {train_metrics['r2']:.4f}, Test R²: {best_metrics['r2']:.4f}")
print(
    f"Overfitting check: {'Minimal' if best_metrics['mse'] / train_metrics['mse'] < 2 else 'Significant'}"
)

if use_target_early_stopping:
    print(f"\nTarget Achievement Status:")
    print(
        f"Training MSE target ({mse_training_target}): {'✅ ACHIEVED' if train_metrics['mse'] <= mse_training_target else '❌ NOT ACHIEVED'}"
    )
    print(
        f"Test MSE target ({mse_test_target}): {'✅ ACHIEVED' if best_metrics['mse'] <= mse_test_target else '❌ NOT ACHIEVED'}"
    )


""" SAVE TRAINED EXPERT """
model_path = EXPERT_PATH / f"expert{data_index}0.pth"
scaler_path = EXPERT_PATH / f"scaler{data_index}0.pth"

torch.save(best_gp, model_path)
torch.save(scaler, scaler_path)

# -----------------------------------------------------------------------------
# PLOTS
# -----------------------------------------------------------------------------
end_indx = int(len(X) - eval_perc)
fig, ax = plt.subplots(figsize=(12, 6))

# Increase the size of the axis numbers
plt.rcdefaults()
plt.rc("xtick", labelsize=14)
plt.rc("ytick", labelsize=14)
fig.autofmt_xdate()

title_suffix = " (Target Reached)" if target_reached else ""
plt.title(f"Expert {data_index} - {kernel} Kernel{title_suffix}", fontsize=16)
ax.fill_between(
    date_time,
    best_mu - 1.96 * best_std,
    best_mu + 1.96 * best_std,
    alpha=0.3,
    color="coral",
    label="95% CI",
)
ax.plot(date_time, y_raw, color="grey", label="Raw", markersize=4)
ax.plot(date_time, y_processed, "*", color="green", label="Actual", markersize=4)
ax.plot(date_time, best_mu, color="red", label=f"GP-{kernel}", linewidth=2)
ax.axvline(
    x=date_time[end_indx],
    color="black",
    linestyle="--",
    label="Train/Test Split",
    alpha=0.7,
)
ax.set_xlabel("Date-time", fontsize=14)
ax.set_ylabel("Fault density", fontsize=14)
plt.legend(loc="best", prop={"size": 12}, facecolor="white", framealpha=1.0)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.show()
