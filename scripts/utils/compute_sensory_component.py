import numpy as np
import pandas as pd
from typing import Optional, Union, Sequence, Dict, Any, Tuple

from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


def compute_sensory_component(
    F: np.ndarray,
    stim_ids: np.ndarray,
    onset_frames: np.ndarray,
    include_mask: np.ndarray,
    fs: float,
    window_s: float = 12.0,
    neuron_ids: Optional[Sequence[Union[int, str]]] = None,
    stim_duration_map_s: Optional[Dict[Any, float]] = None,
) -> pd.DataFrame:
    """
    Compute the sensory component (visual tuning) of the visuomotor vector
    for each neuron, with stimulus-specific integration windows expressed
    in *frames* (derived from durations in seconds).

    Parameters
    ----------
    F : np.ndarray
        Calcium activity array of shape (n_trials, n_neurons, n_timepoints).
        Values should be ΔF/F (or similar) per trial/neuron/timepoint.
    stim_ids : np.ndarray
        Array of shape (n_trials,) giving the stimulus identity for each trial.
        Can be integers or strings (e.g. 0..9, or "loom_left", "grating_20deg", ...).
    onset_frames : np.ndarray
        Array of shape (n_trials,) giving the frame index (0-based) of stimulus onset
        for each trial (already in frames).
    include_mask : np.ndarray
        Boolean array of shape (n_trials,) where True means the trial is included
        in the sensory tuning (e.g. no convergent eye movements / no hunting).
    fs : float
        Sampling frequency of imaging in Hz (frames per second).
    window_s : float, optional
        Default length of the post-stimulus integration window in seconds.
        Used if stim_duration_map_s is None or a given stim is not in the map.
    neuron_ids : sequence of int/str, optional
        Labels for neurons, length n_neurons. If None, uses range(n_neurons).
    stim_duration_map_s : dict, optional
        Mapping from stimulus ID (same type as entries in stim_ids) to
        stimulus duration in seconds, e.g.:
            { "OKR": 10.0, "BARS": 12.0, "LOOM": 5.6 }
        These durations are converted to *frame counts* internally as:
            duration_frames = round(duration_s * fs)

    Returns
    -------
    sensory_df : pd.DataFrame
        DataFrame of shape (n_neurons, n_stimuli) with the mean AUC over
        the post-stimulus window for each neuron and each stimulus.
        Rows: neurons (indexed by neuron_ids)
        Columns: 'sensory_stim_<stim_label>' for each unique stim_ids value.

    Notes
    -----
    - All integration is done over integer frame indices:
        start = onset_frame
        end   = onset_frame + duration_frames
      where duration_frames is derived from the duration in seconds and fs.
    - If no valid trials exist for a given neuron × stimulus, the entry is np.nan.
    - You can pass a uniform duration via window_s only, or per-stimulus via
      stim_duration_map_s.
    """
    # Basic sanity checks
    if F.ndim != 3:
        raise ValueError(f"F must be 3D (n_trials, n_neurons, n_timepoints), got shape {F.shape}")
    n_trials, n_neurons, n_timepoints = F.shape

    stim_ids = np.asarray(stim_ids)
    onset_frames = np.asarray(onset_frames)
    include_mask = np.asarray(include_mask, dtype=bool)

    if stim_ids.shape[0] != n_trials:
        raise ValueError("stim_ids must have length n_trials")
    if onset_frames.shape[0] != n_trials:
        raise ValueError("onset_frames must have length n_trials")
    if include_mask.shape[0] != n_trials:
        raise ValueError("include_mask must have length n_trials")

    # Default integration length in frames (if no per-stim duration is provided)
    default_window_frames = int(round(window_s * fs))
    if default_window_frames <= 0:
        raise ValueError("window_s * fs must be > 0 to define a valid default window")

    # Convert per-stimulus durations (in seconds) -> frame counts
    stim_duration_frames_map = None
    if stim_duration_map_s is not None:
        stim_duration_frames_map = {
            stim: int(round(dur_s * fs))
            for stim, dur_s in stim_duration_map_s.items()
            if dur_s is not None
        }

    # Set up neuron IDs
    if neuron_ids is None:
        neuron_ids = np.arange(n_neurons)
    else:
        if len(neuron_ids) != n_neurons:
            raise ValueError("neuron_ids length must match n_neurons")

    # Identify unique stimulus conditions (and keep a stable order)
    unique_stims = np.unique(stim_ids)
    n_stims = unique_stims.shape[0]

    # Prepare array for results: (n_neurons, n_stims)
    sensory_matrix = np.full((n_neurons, n_stims), np.nan, dtype=float)

    # Main loop: for each stimulus and neuron, compute mean AUC
    for s_idx, stim in enumerate(unique_stims):
        # Determine integration length in frames for this stimulus
        if stim_duration_frames_map is not None and stim in stim_duration_frames_map:
            duration_frames = stim_duration_frames_map[stim]
        else:
            duration_frames = default_window_frames

        if duration_frames <= 0:
            # Skip this stimulus if duration is zero or invalid
            continue

        # Trials with this stimulus AND passing the include_mask
        valid_trials = (stim_ids == stim) & include_mask
        if not np.any(valid_trials):
            # No valid trials for this stimulus; leave entire column as NaN
            continue

        # Loop over neurons
        for n in range(n_neurons):
            auc_values = []

            for trial_idx in np.where(valid_trials)[0]:
                onset = int(onset_frames[trial_idx])
                start = onset
                end = onset + duration_frames

                # Skip if the window would run beyond the recorded timepoints
                if end > n_timepoints:
                    continue

                # Extract ΔF/F in the post-stimulus window (pure frame-based slice)
                trace_segment = F[trial_idx, n, start:end]

                # Compute area under the curve using trapezoidal rule.
                # Here the x-axis is "frames"; if you want AUC in units of
                # (ΔF/F * seconds), multiply by (1/fs) afterwards.
                auc = np.trapz(trace_segment)
                auc_values.append(auc)

            if auc_values:
                sensory_matrix[n, s_idx] = np.mean(auc_values)
            # else: remains NaN if no usable trials for this neuron × stimulus

    # Build column names: one per stimulus condition
    col_names = [f"sensory_stim_{str(stim)}" for stim in unique_stims]

    # Wrap into a DataFrame for easy downstream merging
    sensory_df = pd.DataFrame(
        data=sensory_matrix,
        index=pd.Index(neuron_ids, name="neuron_id"),
        columns=col_names,
    )

    return sensory_df

## =======================================================================================
def fit_motor_regression_elastic_net(
    F: np.ndarray,
    regressors_df: pd.DataFrame,
    fs: float,
    lag_frames: int = 0,
    neuron_ids: Optional[Sequence[Union[int, str]]] = None,
    l1_ratios: Sequence[float] = (0.1, 0.5, 0.9),
    alphas: Optional[Sequence[float]] = None,
    n_splits: int = 10,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit elastic-net motor regression for each neuron, following the spirit of
    Antinucci et al. (eLife 2019), and return per-neuron motor coefficients
    (for the VMV) plus model diagnostics.

    Parameters
    ----------
    F : np.ndarray
        Neural activity array of shape (n_neurons, n_timepoints_imaging).
        Values should be ΔF/F (or similar) per neuron and frame.
    regressors_df : pd.DataFrame
        Time-series of motor regressors, shape (n_timepoints_imaging, n_regressors).
        Columns are regressor names (e.g. LEye_ipsi_pos, Conv_tail_L, Tail_vigour...).
        Index should correspond to imaging frames (length >= n_timepoints_imaging).
    fs : float
        Imaging sampling frequency (Hz). Used for documentation; regression uses
        frame indices internally.
    lag_frames : int, default 0
        Number of frames by which motor regressors *lead* the calcium signal.
        For lag_frames > 0:
          - X (regressors) uses frames [0 .. T - lag_frames - 1]
          - y (calcium)   uses frames [lag_frames .. T - 1]
    neuron_ids : sequence, optional
        Labels for neurons, length n_neurons. If None, uses np.arange(n_neurons).
    l1_ratios : sequence of float, default (0.1, 0.5, 0.9)
        Values of l1_ratio (elastic-net mixing parameter) to cross-validate over.
        l1_ratio = 1 is LASSO-like, 0 is ridge-like.
    alphas : sequence of float, optional
        Grid of alpha values (penalty strengths) to consider. If None, ElasticNetCV
        will choose a default path based on the data.
    n_splits : int, default 10
        Number of CV folds for ElasticNetCV.
    random_state : int, default 0
        Random seed for ElasticNetCV.

    Returns
    -------
    motor_coeffs_df : pd.DataFrame
        Per-neuron motor regression coefficients, shape (n_neurons, n_regressors).
        Index: neuron_ids
        Columns: regressors_df.columns
    metrics_df : pd.DataFrame
        Per-neuron model diagnostics with columns:
            - 'R2'        : cross-validated R^2 on held-out folds
            - 'alpha'     : chosen penalty strength
            - 'l1_ratio'  : chosen L1/L2 mix
            - 'n_samples' : number of timepoints used after lagging
    """
    if F.ndim != 2:
        raise ValueError(f"F must be 2D (n_neurons, n_timepoints), got shape {F.shape}")
    n_neurons, n_timepoints_F = F.shape

    # Align regressors to F in time
    X_full = regressors_df.to_numpy(dtype=float)
    n_timepoints_X, n_regressors = X_full.shape

    # Use the overlapping time window
    T = min(n_timepoints_F, n_timepoints_X)
    if T <= lag_frames:
        raise ValueError(
            f"Not enough timepoints ({T}) for lag_frames={lag_frames}. "
            "T must be > lag_frames."
        )

    # Build lagged design:
    # X uses frames [0 .. T - lag_frames - 1]
    # y uses frames [lag_frames .. T - 1]
    if lag_frames > 0:
        X = X_full[: T - lag_frames, :]
        F_aligned = F[:, lag_frames:T]
    else:
        X = X_full[:T, :]
        F_aligned = F[:, :T]

    n_samples = X.shape[0]
    if F_aligned.shape[1] != n_samples:
        raise RuntimeError("Time alignment bug: X and F_aligned have mismatched length.")

    # Standardise regressors (mean=0, std=1) as is standard before elastic-net
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_std = scaler.fit_transform(X)

    # Set neuron IDs
    if neuron_ids is None:
        neuron_ids = np.arange(n_neurons)
    else:
        if len(neuron_ids) != n_neurons:
            raise ValueError("neuron_ids length must match n_neurons")

    regressor_names = list(regressors_df.columns)

    # Outputs
    coeffs = np.zeros((n_neurons, n_regressors), dtype=float)
    R2_list = np.zeros(n_neurons, dtype=float)
    alpha_list = np.zeros(n_neurons, dtype=float)
    l1_ratio_list = np.zeros(n_neurons, dtype=float)

    # Fit model per neuron
    for i in range(n_neurons):
        y = F_aligned[i, :]

        # Optionally, you can mean-center y, but not strictly necessary
        y = y - np.nanmean(y)

        # Handle neurons with flat or NaN traces
        if np.allclose(y, 0) or np.isnan(y).all():
            coeffs[i, :] = 0.0
            R2_list[i] = np.nan
            alpha_list[i] = np.nan
            l1_ratio_list[i] = np.nan
            continue

        # ElasticNetCV handles alpha path; we supply l1_ratio grid
        model = ElasticNetCV(
            l1_ratio=list(l1_ratios),
            alphas=None if alphas is None else list(alphas),
            cv=n_splits,
            fit_intercept=True,
            n_jobs=-1,
            random_state=random_state,
        )
        model.fit(X_std, y)

        # Store coefficients for this neuron
        coeffs[i, :] = model.coef_
        alpha_list[i] = model.alpha_
        l1_ratio_list[i] = model.l1_ratio_

        # Compute CV-like R^2 on the whole data as a rough summary
        y_pred = model.predict(X_std)
        R2_list[i] = r2_score(y, y_pred)

    # Wrap outputs in DataFrames
    index = pd.Index(neuron_ids, name="neuron_id")

    motor_coeffs_df = pd.DataFrame(
        data=coeffs,
        index=index,
        columns=regressor_names,
    )

    metrics_df = pd.DataFrame(
        {
            "R2": R2_list,
            "alpha": alpha_list,
            "l1_ratio": l1_ratio_list,
            "n_samples": n_samples,
            "fs": fs,
            "lag_frames": lag_frames,
        },
        index=index,
    )

    return motor_coeffs_df, metrics_df

