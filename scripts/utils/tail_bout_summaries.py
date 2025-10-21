
import numpy as np
import pandas as pd
import os
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

def _mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return 1.4826 * mad + 1e-12

def _detect_tail_bouts_internal(
    tail: Union[pd.DataFrame, np.ndarray],
    fs: Optional[float] = 3.6,
    smooth_window: int = 5,
    z_high: float = 3.0,
    z_low: float = 1.5,
    min_duration_s: float = 0.08,
    min_inter_bout_s: float = 0.10
):
    t_raw, angle = _ensure_arrays(tail)
    mfin = np.isfinite(t_raw) & np.isfinite(angle)
    t_raw, angle = t_raw[mfin], angle[mfin]
    df = pd.DataFrame({"t": t_raw, "a": angle}).sort_values("t").groupby("t", as_index=False)["a"].mean()
    t_sorted = df["t"].to_numpy(); angle = df["a"].to_numpy()
    time_is_frames = False
    if fs is not None:
        dt_raw = np.diff(t_sorted) if t_sorted.size > 1 else np.array([])
        if dt_raw.size == 0 or np.nanmedian(np.abs(dt_raw - 1.0)) < 1e-6:
            time_is_frames = True
    if time_is_frames:
        time_s = t_sorted / float(fs); dt_uniform = 1.0/float(fs)
    else:
        if np.any(np.diff(t_sorted) <= 0):
            raise ValueError("Time must be strictly increasing after deduplication.")
        time_s = t_sorted; dt_uniform = None
    if time_s.size < 3:
        empty = pd.DataFrame(columns=[
            "onset_idx","offset_idx","onset_time","offset_time","duration_s",
            "peak_speed_deg_s","mean_speed_deg_s","angle_amp_deg","rms_speed_deg_s"
        ])
        info = {"time_s": time_s, "angle_smooth": angle, "vel_deg_s": np.full_like(angle, np.nan),
                "thr_high": np.nan, "thr_low": np.nan, "sigma_vel": np.nan}
        return empty, info
    angle_s = _rolling_mean(angle, smooth_window)
    if time_is_frames: vel = np.gradient(angle_s, 1.0/float(fs))
    else: vel = np.gradient(angle_s, time_s)
    sigma = _mad_sigma(np.abs(vel))
    if not np.isfinite(sigma) or sigma == 0: sigma = np.nanstd(vel)
    if not np.isfinite(sigma) or sigma == 0:
        empty = pd.DataFrame(columns=[
            "onset_idx","offset_idx","onset_time","offset_time","duration_s",
            "peak_speed_deg_s","mean_speed_deg_s","angle_amp_deg","rms_speed_deg_s"
        ])
        info = {"time_s": time_s, "angle_smooth": angle_s, "vel_deg_s": vel,
                "thr_high": np.nan, "thr_low": np.nan, "sigma_vel": sigma}
        return empty, info
    thr_high =  z_high * sigma; thr_low = z_low * sigma
    in_bout=False; start_idx=None; last_offset_time=-np.inf; rows=[]; N = time_s.size
    for i in range(N):
        vabs = abs(vel[i]) if np.isfinite(vel[i]) else 0.0
        if not in_bout:
            if vabs >= thr_high and (time_s[i] - last_offset_time) >= min_inter_bout_s:
                in_bout = True; start_idx=i
        else:
            if (vabs <= thr_low) or (i==N-1):
                end_idx = i; dur = time_s[end_idx] - time_s[start_idx]
                if dur >= min_duration_s:
                    seg = slice(start_idx, end_idx+1)
                    vseg = vel[seg]; aseg = angle_s[seg]
                    rows.append({
                        "onset_idx": int(start_idx), "offset_idx": int(end_idx),
                        "onset_time": float(time_s[start_idx]), "offset_time": float(time_s[end_idx]),
                        "duration_s": float(dur),
                        "peak_speed_deg_s": float(np.nanmax(np.abs(vseg))),
                        "mean_speed_deg_s": float(np.nanmean(np.abs(vseg))),
                        "angle_amp_deg": float(np.nanmax(aseg) - np.nanmin(aseg)),
                        "rms_speed_deg_s": float(np.sqrt(np.nanmean(vseg**2)))
                    })
                    last_offset_time = time_s[end_idx]
                in_bout=False; start_idx=None
    bouts = pd.DataFrame(rows, columns=[
        "onset_idx","offset_idx","onset_time","offset_time","duration_s",
        "peak_speed_deg_s","mean_speed_deg_s","angle_amp_deg","rms_speed_deg_s"
    ])
    info = {"time_s": time_s, "angle_smooth": angle_s, "vel_deg_s": vel,
            "thr_high": thr_high, "thr_low": thr_low, "sigma_vel": sigma}
    return bouts, info

def _classify_window(times: np.ndarray, fs: float, stim_frame: int, during_start: int, during_end: int, post_start: int):
    stim_t = stim_frame/float(fs)
    pre_lo, pre_hi = 0.0, stim_t
    dur_lo, dur_hi = during_start/float(fs), (during_end+1)/float(fs)
    post_lo, post_hi = post_start/float(fs), np.inf
    labels = np.empty(times.shape, dtype=object); labels[:] = None
    labels[(times >= pre_lo) & (times < pre_hi)] = "pre"
    labels[(times >= dur_lo) & (times < dur_hi)] = "during"
    labels[(times >= post_lo)] = "post"
    return labels

def extract_tail_bout_features_and_windows(
    data: Union[pd.DataFrame, np.ndarray, pd.Series],
    data_is_bouts: bool = False,
    fs: float = 3.6,
    total_frames: int = 108,
    stim_frame: int = 11,
    during_start: int = 12,
    during_end: int = 50,
    post_start: int = 51,
    smooth_window: int = 5,
    z_high: float = 3.0,
    z_low: float = 1.5,
    min_duration_s: float = 0.08,
    min_inter_bout_s: float = 0.10,
    response_window_s: tuple = (0.0, 2.0),
    save_prefix: str = "/mnt/data/tail_bout_summary"
):
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
    if data_is_bouts:
        bouts = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        if bouts.empty or "onset_time" not in bouts.columns:
            raise ValueError("When data_is_bouts=True, provide a DataFrame with 'onset_time' and 'offset_time'.")
        info = {}
    else:
        bouts, info = _detect_tail_bouts_internal(
            data, fs=fs, smooth_window=smooth_window, z_high=z_high, z_low=z_low,
            min_duration_s=min_duration_s, min_inter_bout_s=min_inter_bout_s
        )
    if not bouts.empty:
        bouts = bouts.copy()
        bouts["window_class"] = _classify_window(
            bouts["onset_time"].to_numpy(), fs, stim_frame, during_start, during_end, post_start
        )
        stim_t = stim_frame/float(fs)
        lat = np.where(bouts["onset_time"].to_numpy() >= stim_t,
                       bouts["onset_time"].to_numpy() - stim_t, np.nan)
        bouts["latency_from_stim_s"] = lat
    else:
        bouts = pd.DataFrame(columns=[
            "onset_idx","offset_idx","onset_time","offset_time","duration_s",
            "peak_speed_deg_s","mean_speed_deg_s","angle_amp_deg","rms_speed_deg_s",
            "window_class","latency_from_stim_s"
        ])
    windows = {
        "pre": (0/float(fs), stim_frame/float(fs)),
        "during": (during_start/float(fs), (during_end+1)/float(fs)),
        "post": (post_start/float(fs), total_frames/float(fs))
    }
    stim_t = stim_frame/float(fs)
    rw_lo = stim_t + response_window_s[0]; rw_hi = stim_t + response_window_s[1]
    rows = []
    for cls, (lo, hi) in windows.items():
        in_win = (bouts["onset_time"] >= lo) & (bouts["onset_time"] < hi)
        sub = bouts.loc[in_win]
        n = int(sub.shape[0])
        dur_s = (hi - lo) if np.isfinite(hi) else np.nan
        rate = n / dur_s if (dur_s and dur_s > 0) else np.nan
        if cls == "during":
            after = sub["onset_time"].to_numpy()
            resp_mask = (after >= rw_lo) & (after <= rw_hi)
            responded = int(np.any(resp_mask))
            latency = float(np.min(after[resp_mask]) - stim_t) if np.any(resp_mask) else np.nan
        else:
            responded, latency = np.nan, np.nan
        def med(col):
            vals = sub[col].to_numpy() if (col in sub.columns and not sub.empty) else np.array([])
            vals = vals[np.isfinite(vals)]
            return float(np.median(vals)) if vals.size else np.nan
        rows.append({
            "window_class": cls,
            "bout_count": n, "bout_rate_hz": rate,
            "responded": responded, "latency_s": latency,
            "median_duration_s": med("duration_s"),
            "median_peak_speed_deg_s": med("peak_speed_deg_s"),
            "median_mean_speed_deg_s": med("mean_speed_deg_s"),
            "median_angle_amp_deg": med("angle_amp_deg"),
            "median_rms_speed_deg_s": med("rms_speed_deg_s"),
        })
    window_summary = pd.DataFrame(rows).set_index("window_class").sort_index()
    all_path = f"{save_prefix}_bouts_all.csv"
    pre_path = f"{save_prefix}_bouts_pre.csv"
    during_path = f"{save_prefix}_bouts_during.csv"
    post_path = f"{save_prefix}_bouts_post.csv"
    summary_path = f"{save_prefix}_window_summary.csv"
    bouts.to_csv(all_path, index=False)
    bouts.loc[bouts["window_class"] == "pre"].to_csv(pre_path, index=False)
    bouts.loc[bouts["window_class"] == "during"].to_csv(during_path, index=False)
    bouts.loc[bouts["window_class"] == "post"].to_csv(post_path, index=False)
    window_summary.to_csv(summary_path)
    return {
        "bouts_df": bouts, "window_summary_df": window_summary,
        "bouts_all_csv": all_path, "bouts_pre_csv": pre_path,
        "bouts_during_csv": during_path, "bouts_post_csv": post_path,
        "window_summary_csv": summary_path
    }
