# mn_vmv/summary.py
import numpy as np
import pandas as pd
from typing import List, Tuple


def summarize_vmv_per_fish(
    vmv_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    sensory_cols: List[str],
    motor_cols: List[str],
    r2_col: str,
    fish_col: str = "fish_id",
    group_col: str = "group",
    high_S_quantile: float = 0.75,
    high_R2_thresh: float = 0.2,
    sensory_tuned_quantile: float = 0.75,
    epsilon: float = 1e-6,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build per-neuron and per-fish summary metrics from VMVs and metadata.

    Parameters
    ----------
    vmv_df : pd.DataFrame
        Per-neuron data with VMV components and R².
        Index = neuron_id (must match meta_df.index).
        Must contain:
            - sensory_cols : columns with sensory AUCs (e.g. 'sensory_stim_loom', ...)
            - motor_cols   : columns with motor betas
            - r2_col       : column with cross-validated R² for motor regression
    meta_df : pd.DataFrame
        Per-neuron metadata, index aligned with vmv_df.
        Must contain at least:
            - fish_col : fish identifier (e.g. 'WT01')
            - group_col: experimental group (e.g. 'WT_unexp', 'WT_Mn', 'Mut_unexp', 'Mut_Mn')
    sensory_cols : list of str
        Names of columns in vmv_df corresponding to sensory components.
    motor_cols : list of str
        Names of columns in vmv_df corresponding to motor regression coefficients.
    r2_col : str
        Name of the column in vmv_df containing per-neuron R².
    fish_col : str, default 'fish_id'
        Column name in meta_df specifying fish identity.
    group_col : str, default 'group'
        Column name in meta_df specifying experimental group.
    high_S_quantile : float, default 0.75
        Quantile (0–1) of S_strength across ALL neurons used to define "high S_strength".
        E.g. 0.75 → top 25% are labelled high_S.
    high_R2_thresh : float, default 0.2
        Threshold on R² to define "well modelled" neurons.
    sensory_tuned_quantile : float, default 0.75
        Quantile (0–1) per sensory component used to define "tuned" neurons.
        For each sensory column j, threshold_j = quantile_j(sensory_j) across ALL neurons.
    epsilon : float, default 1e-6
        Small constant to avoid division by zero in SM_index.

    Returns
    -------
    neuron_summary_df : pd.DataFrame
        Per-neuron summary table with:
            - all original vmv_df columns
            - columns from meta_df (fish_id, group, etc.)
            - S_strength, M_strength, SM_index
            - high_S (bool), high_R2 (bool)
            - tuned_<sensory_col> flags for each sensory component
    fish_summary_df : pd.DataFrame
        Per-fish summary table (one row per fish) with columns:
            - fish_col, group_col
            - n_neurons
            - median_S_strength, fraction_high_S
            - median_M_strength
            - median_SM_index
            - median_R2, fraction_R2_high
            - median_<sensory_col> for each sensory component
            - frac_tuned_<sensory_col> for each sensory component
    """
    # 0. Basic checks and alignment
    if not vmv_df.index.equals(meta_df.index):
        # Align on intersection, warn if dropping anything
        common = vmv_df.index.intersection(meta_df.index)
        if len(common) == 0:
            raise ValueError("vmv_df and meta_df have no overlapping neuron IDs (index).")
        vmv_df = vmv_df.loc[common].copy()
        meta_df = meta_df.loc[common].copy()

    # Check required columns
    missing_sensory = set(sensory_cols) - set(vmv_df.columns)
    missing_motor = set(motor_cols) - set(vmv_df.columns)
    if missing_sensory:
        raise ValueError(f"vmv_df is missing sensory columns: {missing_sensory}")
    if missing_motor:
        raise ValueError(f"vmv_df is missing motor columns: {missing_motor}")
    if r2_col not in vmv_df.columns:
        raise ValueError(f"vmv_df is missing R² column: '{r2_col}'")
    for col in (fish_col, group_col):
        if col not in meta_df.columns:
            raise ValueError(f"meta_df is missing required column: '{col}'")

    # 1. Per-neuron metrics: S_strength, M_strength, SM_index
    sensory_vals = vmv_df[sensory_cols].to_numpy(dtype=float)
    motor_vals = vmv_df[motor_cols].to_numpy(dtype=float)

    S_strength = np.nanmax(sensory_vals, axis=1)
    M_strength = np.sqrt(np.nansum(motor_vals**2, axis=1))

    SM_index = (S_strength - M_strength) / (S_strength + M_strength + epsilon)

    # 2. Determine thresholds for "high S" and tuned sensory components
    high_S_thresh = np.nanquantile(S_strength, high_S_quantile)
    high_S = S_strength >= high_S_thresh

    R2_vals = vmv_df[r2_col].to_numpy(dtype=float)
    high_R2 = R2_vals >= high_R2_thresh

    # Sensory-specific tuning thresholds (quantile per component)
    sensory_tuned_flags = {}
    sensory_tuned_thresh = {}
    for col in sensory_cols:
        col_vals = vmv_df[col].to_numpy(dtype=float)
        thr = np.nanquantile(col_vals, sensory_tuned_quantile)
        sensory_tuned_thresh[col] = thr
        sensory_tuned_flags[col] = col_vals >= thr

    # 3. Build per-neuron summary DataFrame
    neuron_summary_df = vmv_df.copy()
    # Attach metadata
    for col in meta_df.columns:
        neuron_summary_df[col] = meta_df[col]

    neuron_summary_df["S_strength"] = S_strength
    neuron_summary_df["M_strength"] = M_strength
    neuron_summary_df["SM_index"] = SM_index
    neuron_summary_df["high_S"] = high_S
    neuron_summary_df["high_R2"] = high_R2

    for col in sensory_cols:
        tuned_colname = f"tuned_{col}"
        neuron_summary_df[tuned_colname] = sensory_tuned_flags[col]

    # 4. Aggregate to per-fish metrics
    def _agg_per_fish(df: pd.DataFrame) -> pd.Series:
        out = {}

        n_neurons = df.shape[0]
        out["n_neurons"] = n_neurons

        # Core global metrics
        out["median_S_strength"] = df["S_strength"].median()
        out["fraction_high_S"] = df["high_S"].mean() if n_neurons > 0 else np.nan

        out["median_M_strength"] = df["M_strength"].median()
        out["median_SM_index"] = df["SM_index"].median()

        out["median_R2"] = df[r2_col].median()
        out["fraction_R2_high"] = df["high_R2"].mean() if n_neurons > 0 else np.nan

        # Per-sensory-component medians and tuning fractions
        for col in sensory_cols:
            median_col = f"median_{col}"
            frac_col = f"frac_tuned_{col}"
            tuned_colname = f"tuned_{col}"

            out[median_col] = df[col].median()
            out[frac_col] = df[tuned_colname].mean() if n_neurons > 0 else np.nan

        return pd.Series(out)

    # groupby fish; group_col is copied along
    # to preserve group membership, we can group by [fish_col, group_col]
    group_keys = [fish_col, group_col]
    fish_summary_df = (
        neuron_summary_df
        .groupby(group_keys, dropna=False)
        .apply(_agg_per_fish)
        .reset_index()
    )

    return neuron_summary_df, fish_summary_df


# mn_vmv/clustering.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import Tuple

def kmeans_vmv(
    vmv_norm_matrix: np.ndarray,
    n_clusters: int = 12,
    random_state: int = 0,
) -> np.ndarray:
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(vmv_norm_matrix)
    return labels
# examples/run_one_fish.py

import numpy as np
import pandas as pd
from scipy.io import loadmat

from mn_vmv import (
    extract_F_neurons_from_all_z_cropped,
    compute_sensory_component,
    build_regressors_from_eye_tail,
    fit_motor_regression_elastic_net,
    prepare_vmv_for_clustering,
    summarize_vmv_per_fish,
)

def main(mat_path: str):
    # 1. Load MATLAB data
    data = loadmat(mat_path, simplify_cells=True)  # adjust as needed

    # 2. Extract continuous F for all ROIs across z-planes
    F_neurons, neuron_meta = extract_F_neurons_from_all_z_cropped(
        data,
        gmROIts_key="gmROIts",
        use_zscored=True,
    )
    fs_imaging = 3.6
    min_T = F_neurons.shape[1]

    # 3. Build regressors from eye & tail (you already defined how to get these)
    regressors_df = build_regressors_from_eye_tail(
        data,
        fs_imaging=fs_imaging,
        # ... whatever arguments you need
    )
    regressors_df = regressors_df.iloc[:min_T, :]

    # 4. Motor regression
    motor_coeffs_df, metrics_df = fit_motor_regression_elastic_net(
        F=F_neurons,
        regressors_df=regressors_df,
        fs=fs_imaging,
        lag_frames=3,
        neuron_ids=neuron_meta.index.values,
    )

    # 5. Sensory component (using your trial-wise F and design)
    # Here you plug the trial-based F, stim_ids, onset_frames, include_mask, fs
    sensory_df = compute_sensory_component(
        F_trials,          # (n_trials, n_neurons, n_timepoints_trial)
        stim_ids,          # (n_trials,)
        onset_frames,      # (n_trials,) in frames
        include_mask,      # (n_trials,) bool
        fs=fs_imaging,
        stim_to_window_s={
            "loom": 5.6,
            "okr": 10.0,
            "bars": 12.0,
            # ...
        },
        neuron_ids=neuron_meta.index.values,
    )

    # 6. VMV assembly + normalisation
    motor_cols = [
        "LEye_ipsi_pos", "LEye_ipsi_vel",
        "REye_ipsi_pos", "REye_ipsi_vel",
        "Conv_tail_sym", "Conv_tail_L", "Conv_tail_R",
        "Tail_bout_L", "Tail_bout_R",
        "Tail_vigour", "Motion_error",
    ]

    vmv_df, vmv_norm_df, vmv_norm_matrix = prepare_vmv_for_clustering(
        sensory_df=sensory_df,
        motor_coeffs_df=motor_coeffs_df,
        motor_cols=motor_cols,
    )

    # 7. Attach per-neuron metadata
    meta_df = neuron_meta.copy()
    meta_df["fish_id"] = "fish01"      # fill from file name
    meta_df["group"] = "WT_unexp"      # fill from experimental log
    meta_df["R2"] = metrics_df["R2"]

    # 8. Summarise per-fish (mostly trivial here since 1 fish)
    sensory_cols = [c for c in vmv_df.columns if c.startswith("sensory_stim_")]
    neuron_summary_df, fish_summary_df = summarize_vmv_per_fish(
        vmv_df=vmv_df,
        meta_df=meta_df,
        sensory_cols=sensory_cols,
        motor_cols=motor_cols,
        r2_col="R2",
        fish_col="fish_id",
        group_col="group",
    )

    # Save outputs
    neuron_summary_df.to_csv("fish01_neuron_summary.csv")
    fish_summary_df.to_csv("fish01_fish_summary.csv")
    vmv_df.to_csv("fish01_vmv_raw.csv")
    vmv_norm_df.to_csv("fish01_vmv_norm.csv")

if __name__ == "__main__":
    import sys
    mat_path = sys.argv[1]
    main(mat_path)


