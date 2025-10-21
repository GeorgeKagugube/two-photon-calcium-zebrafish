
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Union, Optional

def _ensure_arrays(x: Union[pd.DataFrame, np.ndarray]):
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

def plot_tail_with_bouts(
    tail: Union[pd.DataFrame, np.ndarray],
    bouts: pd.DataFrame,
    info: Dict[str, np.ndarray],
    fs: Optional[float] = 3.6,
    smooth_window: int = 5,
    stim_frame: Optional[int] = 11,
    save_prefix: Optional[str] = None
):
    t_raw, angle = _ensure_arrays(tail)
    dt = np.diff(t_raw)
    if fs is not None and (dt.size == 0 or np.nanmedian(np.abs(dt - 1.0)) < 1e-6):
        time_s = t_raw / float(fs)
    else:
        time_s = t_raw.astype(float)
    angle_s = info.get("angle_smooth", None)
    if angle_s is None or len(angle_s) != len(angle):
        angle_s = _rolling_mean(angle, smooth_window)
    vel = info.get("vel_deg_s", None)
    if vel is None or len(vel) != len(angle):
        if fs is not None and (dt.size == 0 or np.nanmedian(np.abs(dt - 1.0)) < 1e-6):
            vel = np.gradient(angle_s, 1.0/float(fs))
        else:
            vel = np.gradient(angle_s, time_s)
    thr_high = info.get("thr_high", np.nan); thr_low = info.get("thr_low", np.nan)

    fig1, ax1 = plt.subplots(figsize=(6,3.2))
    ax1.plot(time_s, angle, linewidth=0.8, label="angle (deg)")
    ax1.plot(time_s, angle_s, linewidth=1.2, label="smoothed")
    if bouts is not None and not bouts.empty:
        for _, b in bouts.iterrows():
            ax1.axvspan(b["onset_time"], b["offset_time"], alpha=0.15)
    if stim_frame is not None and fs is not None:
        ax1.axvline(stim_frame/float(fs), linestyle="--")
    ax1.set_xlabel("time (s)"); ax1.set_ylabel("tail angle (deg)"); ax1.legend(loc="best")
    if save_prefix: fig1.savefig(f"{save_prefix}_angle.png", bbox_inches="tight")

    fig2, ax2 = plt.subplots(figsize=(6,3.2))
    ax2.plot(time_s, vel, linewidth=0.9, label="angular velocity (deg/s)")
    if np.isfinite(thr_high): ax2.axhline(thr_high, linestyle="--")
    if np.isfinite(thr_low):  ax2.axhline(thr_low, linestyle=":")
    if bouts is not None and not bouts.empty:
        for _, b in bouts.iterrows():
            ax2.axvspan(b["onset_time"], b["offset_time"], alpha=0.15)
    if stim_frame is not None and fs is not None:
        ax2.axvline(stim_frame/float(fs), linestyle="--")
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("angular velocity (deg/s)"); ax2.legend(loc="best")
    if save_prefix: fig2.savefig(f"{save_prefix}_velocity.png", bbox_inches="tight")

    return {"angle": fig1, "velocity": fig2}

def summarize_bouts_stim_aligned(
    bouts: pd.DataFrame,
    fs: float = 3.6,
    total_frames: int = 108,
    stim_frame: int = 11,
    during_start: int = 12,
    during_end: int = 50,
    post_start: int = 51,
    response_window_s: Tuple[float, float] = (0.0, 2.0)
) -> pd.DataFrame:
    stim_t = stim_frame / float(fs)
    pre_t = (0/float(fs), stim_frame/float(fs))
    dur_t = (during_start/float(fs), (during_end+1)/float(fs))
    post_t = (post_start/float(fs), total_frames/float(fs))
    rw_lo = stim_t + response_window_s[0]; rw_hi = stim_t + response_window_s[1]
    b_on = bouts["onset_time"].to_numpy() if (bouts is not None and "onset_time" in bouts.columns) else np.array([])
    after = b_on[(b_on >= rw_lo) & (b_on <= rw_hi)]
    responded = int(after.size > 0)
    latency = float(np.min(after) - stim_t) if responded else np.nan
    def _in(seg, window): return np.logical_and(seg >= window[0], seg < window[1])
    pre_count   = int(np.sum(_in(b_on, pre_t)))
    during_count= int(np.sum(_in(b_on, dur_t)))
    post_count  = int(np.sum(_in(b_on, post_t)))
    pre_rate    = pre_count / (pre_t[1] - pre_t[0]) if (pre_t[1] - pre_t[0]) > 0 else np.nan
    during_rate = during_count / (dur_t[1] - dur_t[0]) if (dur_t[1] - dur_t[0]) > 0 else np.nan
    post_rate   = post_count / (post_t[1] - post_t[0]) if (post_t[1] - post_t[0]) > 0 else np.nan
    def _median_in(colname, window):
        if bouts is None or bouts.empty or colname not in bouts.columns: return np.nan
        mask = _in(bouts["onset_time"].to_numpy(), window)
        vals = bouts.loc[mask, colname].to_numpy()
        vals = vals[np.isfinite(vals)]
        return float(np.median(vals)) if vals.size else np.nan
    out = pd.DataFrame([{
        "stim_time_s": stim_t, "responded": responded, "latency_s": latency,
        "pre_count": pre_count, "during_count": during_count, "post_count": post_count,
        "pre_rate_hz": pre_rate, "during_rate_hz": during_rate, "post_rate_hz": post_rate,
        "pre_median_peak_speed": _median_in("peak_speed_deg_s", pre_t),
        "during_median_peak_speed": _median_in("peak_speed_deg_s", dur_t),
        "post_median_peak_speed": _median_in("peak_speed_deg_s", post_t),
        "pre_median_angle_amp": _median_in("angle_amp_deg", pre_t),
        "during_median_angle_amp": _median_in("angle_amp_deg", dur_t),
        "post_median_angle_amp": _median_in("angle_amp_deg", post_t),
    }])
    return out
