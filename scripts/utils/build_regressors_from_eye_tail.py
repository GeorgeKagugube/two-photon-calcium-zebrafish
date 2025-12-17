import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Iterable
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

###=================================================================================================
def extract_eye_traces_from_dff(
    dff: dict,
    fs_imaging: float,
    gmb_key: str = "gmb",
    p_key: str = "p",
    L_key: str = "Langles",
    R_key: str = "Rangles",
) -> pd.DataFrame:
    """
    Extract continuous left/right eye angle traces from the dff structure.

    Assumes:
      - dff[gmb_key][p_key] is an indexable sequence of epochs (length = n_epochs)
      - Each epoch entry has keys L_key and R_key with arrays of length n_frames_epoch
      - All epochs share the same fs_imaging (e.g. 3.6 Hz) and are back-to-back.

    Parameters
    ----------
    dff : dict
        Your top-level data structure.
    fs_imaging : float
        Imaging / eye sampling frequency in Hz (e.g. 3.6).
    gmb_key : str
        Key for the group in dff (default "gmb").
    p_key : str
        Key for the per-epoch container (default "p").
    L_key : str
        Key for left eye angles inside each epoch (default "Langles").
    R_key : str
        Key for right eye angles inside each epoch (default "Rangles").

    Returns
    -------
    eye_df : pd.DataFrame
        Columns:
          - "time"      : global time in seconds across all epochs
          - "epoch"     : epoch index (0..n_epochs-1)
          - "frame"     : frame index within epoch (0..n_frames_epoch-1)
          - "left_eye"  : left eye angle (deg)
          - "right_eye" : right eye angle (deg)
    """
    p = dff.get(gmb_key).get(p_key)
    n_epochs = len(p)

    times = []
    epochs = []
    frames = []
    left_all = []
    right_all = []

    dt = 1.0 / fs_imaging
    global_frame = 0

    for epoch_idx in range(n_epochs):
        # Extract per-epoch traces
        L = np.asarray(p[epoch_idx][L_key], dtype=float).ravel()
        R = np.asarray(p[epoch_idx][R_key], dtype=float).ravel()

        if L.shape != R.shape:
            raise ValueError(
                f"Epoch {epoch_idx}: Langles and Rangles have different shapes "
                f"{L.shape} vs {R.shape}"
            )

        n_frames_epoch = L.shape[0]

        # Time for these frames (global)
        t_epoch = np.arange(global_frame, global_frame + n_frames_epoch) * dt

        times.append(t_epoch)
        epochs.append(np.full(n_frames_epoch, epoch_idx, dtype=int))
        frames.append(np.arange(n_frames_epoch, dtype=int))
        left_all.append(L)
        right_all.append(R)

        global_frame += n_frames_epoch

    # Concatenate all epochs
    time_concat = np.concatenate(times)
    epoch_concat = np.concatenate(epochs)
    frame_concat = np.concatenate(frames)
    left_concat = np.concatenate(left_all)
    right_concat = np.concatenate(right_all)

    eye_df = pd.DataFrame(
        {
            "time": time_concat,
            "epoch": epoch_concat,
            "frame": frame_concat,
            "left_eye": left_concat,
            "right_eye": right_concat,
        }
    ).sort_values("time").reset_index(drop=True)

    return eye_df

### ================================================================================================
def extract_tail_traces_from_dff(
    dff: dict,
    fs_imaging: float,
    gmbt_key: str = "gmbt",
    p_key: str = "p",
    tail_key: str = "cumtail",
) -> pd.DataFrame:
    """
    Extract continuous tail angle/curvature trace from the dff structure.

    Assumes:
      - dff[gmbt_key][p_key] is an indexable sequence of epochs (length = n_epochs)
      - Each epoch entry has key tail_key with an array of length n_frames_epoch
        (here 'cumtail' per frame)
      - All epochs share the same fs_imaging (e.g. 3.6 Hz) and are back-to-back.

    Parameters
    ----------
    dff : dict
        Your top-level data structure.
    fs_imaging : float
        Imaging / tail sampling frequency in Hz (e.g. 3.6).
    gmbt_key : str
        Key for the tail group in dff (default "gmbt").
    p_key : str
        Key for the per-epoch container (default "p").
    tail_key : str
        Key for tail angle/curvature inside each epoch (default "cumtail").

    Returns
    -------
    tail_df : pd.DataFrame
        Columns:
          - "time"       : global time in seconds across all epochs
          - "epoch"      : epoch index (0..n_epochs-1)
          - "frame"      : frame index within epoch (0..n_frames_epoch-1)
          - "tail_angle" : signed tail angle/curvature (from cumtail)
    """
    p = dff[gmbt_key][p_key]
    n_epochs = len(p)

    times = []
    epochs = []
    frames = []
    tail_all = []

    dt = 1.0 / fs_imaging
    global_frame = 0

    for epoch_idx in range(n_epochs):
        tail = np.asarray(p[epoch_idx][tail_key], dtype=float).ravel()
        n_frames_epoch = tail.shape[0]

        # Global time for these frames
        t_epoch = np.arange(global_frame, global_frame + n_frames_epoch) * dt

        times.append(t_epoch)
        epochs.append(np.full(n_frames_epoch, epoch_idx, dtype=int))
        frames.append(np.arange(n_frames_epoch, dtype=int))
        tail_all.append(tail)

        global_frame += n_frames_epoch

    time_concat = np.concatenate(times)
    epoch_concat = np.concatenate(epochs)
    frame_concat = np.concatenate(frames)
    tail_concat = np.concatenate(tail_all)

    tail_df = pd.DataFrame(
        {
            "time": time_concat,
            "epoch": epoch_concat,
            "frame": frame_concat,
            "tail_angle": tail_concat,
        }
    ).sort_values("time").reset_index(drop=True)

    return tail_df


###=============================================================================================
def build_regressors_from_eye_tail(
    eye_df: pd.DataFrame,
    tail_df: pd.DataFrame,
    fs: float,
    time_col: str = "time",
    left_eye_col: str = "left_eye",
    right_eye_col: str = "right_eye",
    tail_angle_col: str = "tail_angle",
    ipsi_side: str = "left",
    vergence_thresh: float = 12.0,       # deg; tune on your data
    saccade_vel_thresh: float = 60.0,    # deg/s; tune on your data
    tail_bout_angle_thresh: float = 5.0, # deg; tune on your data
    tail_vigour_smooth_frames: int = 3,
    motion_error_series: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Build the 11 motor regressors required for VMV analysis from eye and tail data.

    Parameters
    ----------
    eye_df : pd.DataFrame
        Must contain columns:
            time_col       (e.g. "time")
            left_eye_col   (e.g. "left_eye")
            right_eye_col  (e.g. "right_eye")
        May also contain 'epoch', 'frame' etc., which are ignored here.
    tail_df : pd.DataFrame
        Must contain columns:
            time_col       (e.g. "time")
            tail_angle_col (e.g. "tail_angle")
    fs : float
        Sampling frequency of the time series (Hz), here 3.6 for your data.
    time_col : str
        Name of the time column in both DataFrames.
    left_eye_col : str
        Name of the left eye angle column (deg).
    right_eye_col : str
        Name of the right eye angle column (deg).
    tail_angle_col : str
        Name of the tail angle / curvature column (signed).
    ipsi_side : str
        "left" if the imaged hemisphere is left (so left eye is ipsilateral),
        "right" if the imaged hemisphere is right.
    vergence_thresh : float
        Threshold (deg) on vergence (L - R) to call convergence.
    saccade_vel_thresh : float
        Threshold (deg/s) on eye velocity magnitude to call saccades.
    tail_bout_angle_thresh : float
        Threshold (deg) on |tail_angle| to call a tail bout and to
        distinguish symmetric vs L/R tail for convergent saccades.
    tail_vigour_smooth_frames : int
        Rolling window (frames) to smooth Tail_vigour.
    motion_error_series : pd.Series, optional
        Optional motion-correction magnitude per frame. If provided, it must
        have an index representing time (same units as time_col); will be
        aligned via merge_asof. If None, Motion_error is set to 0.

    Returns
    -------
    regressors_df : pd.DataFrame
        Indexed by time (time_col) with columns:
            "LEye_ipsi_pos"
            "LEye_ipsi_vel"
            "REye_ipsi_pos"
            "REye_ipsi_vel"
            "Conv_tail_sym"
            "Conv_tail_L"
            "Conv_tail_R"
            "Tail_bout_L"
            "Tail_bout_R"
            "Tail_vigour"
            "Motion_error"
    """
    # --- 1. Align eye and tail on time ---
    eye_df = eye_df.sort_values(time_col).reset_index(drop=True)
    tail_df = tail_df.sort_values(time_col).reset_index(drop=True)

    # Use merge_asof to align tail to eye times (robust to tiny floating jitter)
    merged = pd.merge_asof(
        eye_df[[time_col, left_eye_col, right_eye_col]],
        tail_df[[time_col, tail_angle_col]],
        on=time_col,
        direction="nearest",
    )

    # Drop rows with missing data in critical columns
    merged = merged.dropna(subset=[left_eye_col, right_eye_col, tail_angle_col])

    t = merged[time_col].to_numpy()
    L_raw = merged[left_eye_col].to_numpy().astype(float)
    R_raw = merged[right_eye_col].to_numpy().astype(float)
    tail = merged[tail_angle_col].to_numpy().astype(float)

    n = len(t)
    if n < 3:
        raise ValueError("Not enough aligned timepoints after merging eye and tail data.")

    # --- 2. Eye ipsi/contra positions and velocities ---
    if ipsi_side.lower() == "left":
        ipsi_raw = L_raw
        contra_raw = R_raw
    elif ipsi_side.lower() == "right":
        ipsi_raw = R_raw
        contra_raw = L_raw
    else:
        raise ValueError("ipsi_side must be 'left' or 'right'")

    # Demeaned positions (median as neutral angle)
    ipsi_pos = ipsi_raw - np.nanmedian(ipsi_raw)
    contra_pos = contra_raw - np.nanmedian(contra_raw)

    # Velocities: gradient(y, dt) with dt = 1/fs
    dt = 1.0 / fs
    ipsi_vel = np.gradient(ipsi_raw, dt)
    contra_vel = np.gradient(contra_raw, dt)

    # For convergence detection we also want raw left/right velocities
    L_vel = np.gradient(L_raw, dt)
    R_vel = np.gradient(R_raw, dt)

    # --- 3. Convergent saccades & tail context ---
    vergence = L_raw - R_raw   # positive = converging (L turns nasal, R turns nasal)
    conv_event = (
        (vergence > vergence_thresh) &
        (np.abs(L_vel) > saccade_vel_thresh) &
        (np.abs(R_vel) > saccade_vel_thresh)
    )

    Conv_tail_sym = np.zeros(n, dtype=float)
    Conv_tail_L = np.zeros(n, dtype=float)
    Conv_tail_R = np.zeros(n, dtype=float)

    sym_thresh = tail_bout_angle_thresh

    idx_conv = np.where(conv_event)[0]
    for idx in idx_conv:
        ta = tail[idx]
        if np.abs(ta) < sym_thresh:
            Conv_tail_sym[idx] = 1.0
        elif ta < 0:   # leftward tail bend
            Conv_tail_L[idx] = 1.0
        else:          # rightward tail bend
            Conv_tail_R[idx] = 1.0

    # --- 4. Tail bouts & vigour ---
    Tail_bout_L = np.zeros(n, dtype=float)
    Tail_bout_R = np.zeros(n, dtype=float)

    bout_mask = np.abs(tail) > tail_bout_angle_thresh
    left_mask = bout_mask & (tail < 0)
    right_mask = bout_mask & (tail > 0)

    Tail_bout_L[left_mask] = 1.0
    Tail_bout_R[right_mask] = 1.0

    # Tail vigour: smoothed |tail_angle|
    Tail_vigour_raw = np.abs(tail)
    if tail_vigour_smooth_frames > 1:
        tv_series = pd.Series(Tail_vigour_raw)
        Tail_vigour = tv_series.rolling(
            window=tail_vigour_smooth_frames,
            center=True,
            min_periods=1
        ).mean().to_numpy()
    else:
        Tail_vigour = Tail_vigour_raw

    # --- 5. Motion error regressor ---
    if motion_error_series is not None:
        # Ensure motion_error_series has a monotonically increasing index in time
        mot = motion_error_series.sort_index()
        mot_df = pd.DataFrame({time_col: mot.index.values, "motion": mot.values})

        mot_aligned = pd.merge_asof(
            pd.DataFrame({time_col: t}),
            mot_df,
            on=time_col,
            direction="nearest",
        )
        Motion_error = mot_aligned["motion"].to_numpy().astype(float)
    else:
        Motion_error = np.zeros(n, dtype=float)

    # --- 6. Assemble full regressors DataFrame ---
    regressors_df = pd.DataFrame(
        {
            "LEye_ipsi_pos": ipsi_pos,
            "LEye_ipsi_vel": ipsi_vel,
            "REye_ipsi_pos": contra_pos,
            "REye_ipsi_vel": contra_vel,
            "Conv_tail_sym": Conv_tail_sym,
            "Conv_tail_L": Conv_tail_L,
            "Conv_tail_R": Conv_tail_R,
            "Tail_bout_L": Tail_bout_L,
            "Tail_bout_R": Tail_bout_R,
            "Tail_vigour": Tail_vigour,
            "Motion_error": Motion_error,
        },
        index=pd.Index(t, name=time_col),
    )

    return regressors_df

#### ================================================================================
def build_vmv(
    sensory_df: pd.DataFrame,
    motor_coeffs_df: pd.DataFrame,
    motor_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Assemble the visuomotor vectors (VMVs) from sensory AUCs and motor coefficients.

    Parameters
    ----------
    sensory_df : pd.DataFrame
        Per-neuron sensory components, shape (n_neurons, n_sensory).
        Index must be neuron IDs (e.g. 'neuron_id'), as returned by
        compute_sensory_component().
    motor_coeffs_df : pd.DataFrame
        Per-neuron motor regression coefficients, shape (n_neurons, n_motor).
        Index must be neuron IDs aligned with sensory_df.
    motor_cols : list of str, optional
        Columns from motor_coeffs_df to use (in order). If None, all columns
        in motor_coeffs_df are used in their existing order.

    Returns
    -------
    vmv_df : pd.DataFrame
        Concatenated visuomotor vectors, shape (n_neurons_common, n_sensory + n_motor_used).
        Rows: neuron IDs (intersection of indices of sensory_df and motor_coeffs_df)
        Columns: [sensory columns..., motor columns...]
    """
    # Align neuron IDs by intersection of indices
    common_ids = sensory_df.index.intersection(motor_coeffs_df.index)
    if len(common_ids) == 0:
        raise ValueError("No overlapping neuron IDs between sensory_df and motor_coeffs_df")

    sensory_sub = sensory_df.loc[common_ids].copy()

    if motor_cols is None:
        motor_sub = motor_coeffs_df.loc[common_ids].copy()
    else:
        missing = set(motor_cols) - set(motor_coeffs_df.columns)
        if missing:
            raise ValueError(f"motor_coeffs_df is missing expected columns: {missing}")
        motor_sub = motor_coeffs_df.loc[common_ids, motor_cols].copy()

    # Concatenate along columns: sensory first, then motor
    vmv_df = pd.concat([sensory_sub, motor_sub], axis=1)

    return vmv_df


def normalise_vmv_columns(
    vmv_df: pd.DataFrame,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Normalise each VMV component (column) by its standard deviation across neurons.

    This follows the paper's statement:
      "Each component (column) was normalised across neurones by its standard
       deviation before clustering."

    Parameters
    ----------
    vmv_df : pd.DataFrame
        Raw VMV matrix, shape (n_neurons, n_components).
    eps : float
        Small constant to avoid division by zero if a column has zero SD.

    Returns
    -------
    vmv_norm_df : pd.DataFrame
        Same shape as vmv_df, but each column divided by its SD.
    """
    # Population SD (ddof=0)
    stds = vmv_df.std(axis=0, ddof=0)
    stds_safe = stds.replace(0, eps)

    vmv_norm_df = vmv_df / stds_safe

    return vmv_norm_df


def prepare_vmv_for_clustering(
    sensory_df: pd.DataFrame,
    motor_coeffs_df: pd.DataFrame,
    motor_cols: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    High-level wrapper: sensory + motor → raw VMV → SD-normalised VMV → NumPy matrix.

    Parameters
    ----------
    sensory_df : pd.DataFrame
        Per-neuron sensory AUCs, shape (n_neurons, n_sensory).
    motor_coeffs_df : pd.DataFrame
        Per-neuron motor β-coefficients, shape (n_neurons, n_motor).
    motor_cols : list of str, optional
        Subset / ordering of motor coefficient columns to use.
        If None, all columns of motor_coeffs_df are used.

    Returns
    -------
    vmv_df : pd.DataFrame
        Raw VMV matrix (un-normalised), neurons × components.
    vmv_norm_df : pd.DataFrame
        Column-wise SD-normalised VMV matrix.
    vmv_norm_matrix : np.ndarray
        Same data as vmv_norm_df but as a NumPy array,
        shape (n_neurons_common, n_components), ready for clustering.
    """
    vmv_df = build_vmv(sensory_df, motor_coeffs_df, motor_cols=motor_cols)
    vmv_norm_df = normalise_vmv_columns(vmv_df)
    vmv_norm_matrix = vmv_norm_df.values.astype(float)

    return vmv_df, vmv_norm_df, vmv_norm_matrix

def build_cirf_kernel(
    fs: float,
    tau_rise: float = 0.02,
    tau_decay: float = 0.42,
    duration_s: float = 5.0,
) -> np.ndarray:
    """
    Build a discrete calcium impulse response function (CIRF) kernel:
    fast rise, slow decay, bi-exponential, normalised to max=1.

    Parameters
    ----------
    fs : float
        Imaging sampling frequency (Hz).
    tau_rise : float
        Rise time constant (s).
    tau_decay : float
        Decay time constant (s).
    duration_s : float
        Total duration of the kernel (s).

    Returns
    -------
    kernel : np.ndarray
        1D array of length int(duration_s * fs) representing the CIRF.
    """
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    kernel = (1.0 - np.exp(-t / tau_rise)) * np.exp(-t / tau_decay)
    if kernel.max() > 0:
        kernel = kernel / kernel.max()
    return kernel


def convolve_regressors_with_cirf(
    regressors_df: pd.DataFrame,
    cirf_kernel: np.ndarray,
) -> pd.DataFrame:
    """
    Convolve each regressor with the CIRF kernel (causal) and trim to original length.

    Parameters
    ----------
    regressors_df : pd.DataFrame
        (n_timepoints, n_predictors) behavioural predictors.
    cirf_kernel : np.ndarray
        1D CIRF kernel.

    Returns
    -------
    convolved_df : pd.DataFrame
        Same shape as regressors_df, convolved and trimmed.
    """
    X = regressors_df.values
    n_timepoints, n_pred = X.shape

    convolved = np.zeros_like(X, dtype=float)
    for j in range(n_pred):
        conv_full = np.convolve(X[:, j], cirf_kernel, mode="full")
        convolved[:, j] = conv_full[:n_timepoints]

    return pd.DataFrame(
        convolved,
        index=regressors_df.index,
        columns=regressors_df.columns,
    )


def apply_time_shift(
    X: np.ndarray,
    Y: np.ndarray,
    lag_frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a global time shift between design matrix X and response Y.

    Positive lag_frames means behaviour leads calcium:
    - drop first 'lag_frames' samples from X
    - drop last 'lag_frames' samples from Y

    Parameters
    ----------
    X : np.ndarray
        Design matrix, shape (n_timepoints, n_predictors).
    Y : np.ndarray
        Calcium matrix, shape (n_neurons, n_timepoints).
    lag_frames : int
        Shift in frames (can be negative, zero, or positive).

    Returns
    -------
    X_shifted : np.ndarray
        Time-aligned design matrix, shape (n_samples, n_predictors).
    Y_shifted : np.ndarray
        Time-aligned calcium matrix, shape (n_neurons, n_samples).
    """
    n_neurons, n_t = Y.shape
    n_t_X, _ = X.shape
    if n_t_X != n_t:
        raise ValueError("X and Y must have the same number of timepoints before shifting")

    if lag_frames > 0:
        # behaviour precedes calcium
        X_shifted = X[lag_frames:, :]
        Y_shifted = Y[:, :-lag_frames]
    elif lag_frames < 0:
        lag = -lag_frames
        # calcium precedes behaviour (rarely what you want biologically)
        X_shifted = X[:-lag, :]
        Y_shifted = Y[:, lag:]
    else:
        X_shifted = X
        Y_shifted = Y

    return X_shifted, Y_shifted


def fit_motor_regression_elastic_net(
    F: np.ndarray,
    regressors_df: pd.DataFrame,
    fs: float,
    lag_frames: int = 0,
    neuron_ids: Optional[Sequence] = None,
    use_cirf: bool = True,
    cirf_params: Optional[Dict[str, Any]] = None,
    l1_ratio_grid: Sequence[float] = (0.1, 0.5, 0.9),
    alphas_grid: Sequence[float] = (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    n_splits_cv: int = 10,
    max_iter: int = 5000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit elastic-net regression models for each neuron, using
    (optionally CIRF-convolved) motor regressors.

    This function:
      1) optionally convolves each regressor with a CIRF kernel,
      2) applies a global lag between behaviour and calcium,
      3) standardises all predictors (zero mean, unit variance),
      4) fits ElasticNetCV per neuron,
      5) returns β coefficients and per-neuron metrics.

    Parameters
    ----------
    F : np.ndarray
        Calcium activity, shape (n_neurons, n_timepoints).
        Typically ΔF/F or z-scored traces.
    regressors_df : pd.DataFrame
        Motor regressors at imaging frame times, shape (n_timepoints, n_predictors).
        Column names will be used as coefficient labels.
    fs : float
        Imaging sampling frequency (Hz).
    lag_frames : int, optional
        Global lag between behaviour and calcium (in frames).
        Positive = behaviour leads calcium (most common).
    neuron_ids : sequence, optional
        Labels for neurons; if None, np.arange(n_neurons) is used.
    use_cirf : bool, optional
        If True, convolve regressors with a CIRF kernel before regression.
    cirf_params : dict, optional
        Parameters passed to build_cirf_kernel(fs, **cirf_params),
        e.g. {"tau_rise": 0.02, "tau_decay": 0.42, "duration_s": 5.0}.
    l1_ratio_grid : sequence of float, optional
        Grid of l1_ratio values (0 = ridge-like, 1 = lasso-like).
    alphas_grid : sequence of float, optional
        Grid of alpha values for ElasticNetCV.
    n_splits_cv : int, optional
        Number of cross-validation folds.
    max_iter : int, optional
        Max iterations for ElasticNetCV.

    Returns
    -------
    coeffs_df : pd.DataFrame
        (n_neurons, n_predictors) regression coefficients (βs),
        ready to be used as motor components of the VMV.
    metrics_df : pd.DataFrame
        (n_neurons, 3+2) with columns:
          - 'alpha'     : chosen penalty strength
          - 'l1_ratio'  : chosen L1/L2 mix
          - 'r2'        : model R² on the fitted data
          - 'n_samples' : number of timepoints used after lagging
          - 'lag_frames': lag used
    """
    # ---- sanity checks ----
    if F.ndim != 2:
        raise ValueError("F must be 2D (n_neurons, n_timepoints)")
    n_neurons, n_timepoints = F.shape

    if regressors_df.shape[0] != n_timepoints:
        raise ValueError(
            f"regressors_df must have the same number of timepoints as F "
            f"({regressors_df.shape[0]} vs {n_timepoints})"
        )

    if neuron_ids is None:
        neuron_ids = np.arange(n_neurons)

    # ---- 1) CIRF convolution (optional) ----
    if use_cirf:
        if cirf_params is None:
            cirf_params = {}
        cirf_kernel = build_cirf_kernel(fs, **cirf_params)
        X_df = convolve_regressors_with_cirf(regressors_df, cirf_kernel)
    else:
        X_df = regressors_df.copy()

    # ---- 2) Time shift between X and F ----
    X = X_df.values
    X_shifted, Y_shifted = apply_time_shift(X, F, lag_frames=lag_frames)
    # X_shifted: (n_samples, n_predictors)
    # Y_shifted: (n_neurons, n_samples)
    n_samples, n_pred = X_shifted.shape

    # ---- 3) Standardise predictors (zero mean, unit variance) ----
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_shifted_std = scaler.fit_transform(X_shifted)

    # ---- 4) Fit ElasticNetCV per neuron ----
    coeffs = np.zeros((n_neurons, n_pred), dtype=float)
    alphas_chosen = np.zeros(n_neurons, dtype=float)
    l1_ratios_chosen = np.zeros(n_neurons, dtype=float)
    r2_scores = np.zeros(n_neurons, dtype=float)

    for i in range(n_neurons):
        y = Y_shifted[i, :]
        # Optional: mean-centre y
        y = y - np.nanmean(y)

        # Handle flat or all-NaN traces
        if np.allclose(y, 0) or np.isnan(y).all():
            coeffs[i, :] = 0.0
            alphas_chosen[i] = np.nan
            l1_ratios_chosen[i] = np.nan
            r2_scores[i] = np.nan
            continue

        model = ElasticNetCV(
            l1_ratio=list(l1_ratio_grid),
            alphas=list(alphas_grid),
            cv=n_splits_cv,
            max_iter=max_iter,
            n_jobs=None,
        )
        model.fit(X_shifted_std, y)

        coeffs[i, :] = model.coef_
        alphas_chosen[i] = model.alpha_
        l1_ratios_chosen[i] = (
            model.l1_ratio_ if np.isscalar(model.l1_ratio_) else float(model.l1_ratio_[0])
        )
        r2_scores[i] = model.score(X_shifted_std, y)

    # ---- 5) Wrap in DataFrames ----
    coeffs_df = pd.DataFrame(
        coeffs,
        index=pd.Index(neuron_ids, name="neuron_id"),
        columns=X_df.columns,
    )

    metrics_df = pd.DataFrame(
        {
            "alpha": alphas_chosen,
            "l1_ratio": l1_ratios_chosen,
            "r2": r2_scores,
            "n_samples": n_samples,
            "lag_frames": lag_frames,
        },
        index=pd.Index(neuron_ids, name="neuron_id"),
    )

    return coeffs_df, metrics_df

