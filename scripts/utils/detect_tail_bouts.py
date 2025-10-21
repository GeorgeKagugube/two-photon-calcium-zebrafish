
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Union, Optional

def _ensure_arrays(x: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(x, pd.DataFrame):
        if x.shape[1] < 2:
            raise ValueError("DataFrame must have two columns: time, angle_deg.")
        t = x.iloc[:, 0].to_numpy(dtype=float)
        a = x.iloc[:, 1].to_numpy(dtype=float)
    elif isinstance(x, np.ndarray):
        if x.ndim != 2 or x.shape[1] < 2:
            raise ValueError("ndarray must be shape (N, 2): time, angle_deg.")
        t = x[:, 0].astype(float)
        a = x[:, 1].astype(float)
    else:
        raise TypeError("Input must be a pandas DataFrame or a 2-column numpy ndarray.")
    return t, a

def _rolling_mean(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return x.copy()
    s = pd.Series(x)
    return s.rolling(window=win, center=True, min_periods=1).mean().to_numpy()

def _mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return 1.4826 * mad + 1e-12

def detect_tail_bouts(
    tail: Union[pd.DataFrame, np.ndarray],
    fs: Optional[float] = 3.6,          # set None if time already in seconds
    smooth_window: int = 5,
    z_high: float = 3.0,
    z_low: float = 1.5,
    min_duration_s: float = 0.08,
    min_inter_bout_s: float = 0.10,
    return_intermediates: bool = True
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Robust bout detector for tail angle. Handles duplicate/unsorted/NaN time stamps.
    Returns bouts DataFrame and intermediates (angle_smooth, velocity, thresholds).
    """
    # 1) arrays + basic cleaning
    t_raw, angle = _ensure_arrays(tail)
    mfin = np.isfinite(t_raw) & np.isfinite(angle)
    t_raw, angle = t_raw[mfin], angle[mfin]

    # 2) sort by time and collapse duplicates by averaging angle
    df = pd.DataFrame({"t": t_raw, "a": angle}).sort_values("t")
    df = df.groupby("t", as_index=False)["a"].mean()
    t_sorted = df["t"].to_numpy()
    angle = df["a"].to_numpy()

    # 3) decide time base and compute seconds
    time_is_frames = False
    if fs is not None:
        # frames-like if median step ≈ 1
        dt_raw = np.diff(t_sorted) if t_sorted.size > 1 else np.array([])
        if dt_raw.size == 0 or np.nanmedian(np.abs(dt_raw - 1.0)) < 1e-6:
            time_is_frames = True

    if time_is_frames:
        time_s = t_sorted / float(fs)
        dt_uniform = 1.0 / float(fs)
    else:
        # assume seconds; ensure strictly increasing
        if np.any(np.diff(t_sorted) <= 0):
            raise ValueError("Time must be strictly increasing after deduplication.")
        time_s = t_sorted
        dt_uniform = None  # not used

    # 4) bail out early if too short
    if time_s.size < 3:
        empty = pd.DataFrame(columns=[
            "onset_idx","offset_idx","onset_time","offset_time","duration_s",
            "peak_speed_deg_s","mean_speed_deg_s","angle_amp_deg","rms_speed_deg_s"
        ])
        info = {"time_s": time_s, "angle_smooth": angle, "vel_deg_s": np.full_like(angle, np.nan),
                "thr_high": np.nan, "thr_low": np.nan, "sigma_vel": np.nan}
        return (empty, info) if return_intermediates else (empty, {})

    # 5) smooth angle and compute velocity safely
    angle_s = _rolling_mean(angle, smooth_window)
    if time_is_frames:
        # uniform spacing → avoid np.gradient(x, time) to sidestep zero-dt issues
        vel = np.gradient(angle_s, dt_uniform)  # deg/s
    else:
        # variable spacing
        vel = np.gradient(angle_s, time_s)      # deg/s

    # 6) thresholds with robust sigma; handle edge-case NaNs
    sigma = _mad_sigma(np.abs(vel))
    if not np.isfinite(sigma) or sigma == 0:
        # fallback to std; if still bad, return empty
        sigma = np.nanstd(vel)
    if not np.isfinite(sigma) or sigma == 0:
        empty = pd.DataFrame(columns=[
            "onset_idx","offset_idx","onset_time","offset_time","duration_s",
            "peak_speed_deg_s","mean_speed_deg_s","angle_amp_deg","rms_speed_deg_s"
        ])
        info = {"time_s": time_s, "angle_smooth": angle_s, "vel_deg_s": vel,
                "thr_high": np.nan, "thr_low": np.nan, "sigma_vel": sigma}
        return (empty, info) if return_intermediates else (empty, {})

    thr_high = z_high * sigma
    thr_low  = z_low  * sigma

    # 7) hysteresis detection
    in_bout = False
    start_idx = None
    last_offset_time = -np.inf
    rows = []
    N = time_s.size
    for i in range(N):
        vabs = abs(vel[i]) if np.isfinite(vel[i]) else 0.0
        if not in_bout:
            if vabs >= thr_high and (time_s[i] - last_offset_time) >= min_inter_bout_s:
                in_bout = True
                start_idx = i
        else:
            if (vabs <= thr_low) or (i == N - 1):
                end_idx = i
                dur = time_s[end_idx] - time_s[start_idx]
                if dur >= min_duration_s:
                    seg = slice(start_idx, end_idx + 1)
                    vseg = vel[seg]
                    aseg = angle_s[seg]
                    peak_speed = float(np.nanmax(np.abs(vseg)))
                    mean_speed = float(np.nanmean(np.abs(vseg)))
                    angle_amp = float(np.nanmax(aseg) - np.nanmin(aseg))
                    rms_speed = float(np.sqrt(np.nanmean(vseg**2)))
                    rows.append({
                        "onset_idx": int(start_idx),
                        "offset_idx": int(end_idx),
                        "onset_time": float(time_s[start_idx]),
                        "offset_time": float(time_s[end_idx]),
                        "duration_s": float(dur),
                        "peak_speed_deg_s": peak_speed,
                        "mean_speed_deg_s": mean_speed,
                        "angle_amp_deg": angle_amp,
                        "rms_speed_deg_s": rms_speed,
                    })
                    last_offset_time = time_s[end_idx]
                in_bout = False
                start_idx = None

    bouts = pd.DataFrame(rows, columns=[
        "onset_idx","offset_idx","onset_time","offset_time","duration_s",
        "peak_speed_deg_s","mean_speed_deg_s","angle_amp_deg","rms_speed_deg_s"
    ])
    info = {
        "time_s": time_s,
        "angle_smooth": angle_s,
        "vel_deg_s": vel,
        "thr_high": thr_high,
        "thr_low": thr_low,
        "sigma_vel": sigma
    }
    return (bouts, info) if return_intermediates else (bouts, {})
