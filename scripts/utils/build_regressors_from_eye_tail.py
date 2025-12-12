import numpy as np
import pandas as pd
from typing import Optional

###=================================================================================================
import numpy as np
import pandas as pd


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
