
import numpy as np
import pandas as pd
import os
from typing import Dict, Tuple, Optional, Union, List

def _mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0: return np.nan
    med = np.nanmedian(x); mad = np.nanmedian(np.abs(x - med))
    return 1.4826 * mad + 1e-12

def _detect_events_one_roi(y: np.ndarray, fs: float, k_sigma_height: float, k_sigma_prom: float, min_distance_s: float) -> np.ndarray:
    sigma = _mad_sigma(y)
    if not np.isfinite(sigma) or sigma <= 0: sigma = np.nanstd(y) + 1e-12
    height_thr = k_sigma_height * sigma; prom_thr = k_sigma_prom * sigma
    min_dist = max(1, int(round(min_distance_s * fs)))
    try:
        from scipy.signal import find_peaks
        peaks, props = find_peaks(y, height=height_thr, prominence=prom_thr, distance=min_dist)
    except Exception:
        peaks = []
        for i in range(1, len(y)-1):
            if y[i] > y[i-1] and y[i] > y[i+1] and y[i] >= height_thr:
                if len(peaks)==0 or (i - peaks[-1]) >= min_dist:
                    peaks.append(i)
        peaks = np.array(peaks, dtype=int)
    return peaks.astype(int)

def _compute_tail_velocity(tail_df: pd.DataFrame, fs: Optional[float]):
    arr = np.asarray(tail_df.values, float); t_raw = arr[:,0]; angle = arr[:,1]
    mfin = np.isfinite(t_raw) & np.isfinite(angle); t_raw, angle = t_raw[mfin], angle[mfin]
    df = pd.DataFrame({"t": t_raw, "a": angle}).sort_values("t").groupby("t", as_index=False)["a"].mean()
    t_sorted = df["t"].to_numpy(); angle = df["a"].to_numpy()
    if fs is not None:
        dt = np.diff(t_sorted) if t_sorted.size>1 else np.array([])
        time_is_frames = (dt.size==0) or (np.nanmedian(np.abs(dt-1.0)) < 1e-6)
    else:
        time_is_frames = False
    if time_is_frames: time_s = t_sorted / float(fs)
    else:
        if np.any(np.diff(t_sorted) <= 0): raise ValueError("Tail time must be strictly increasing after deduplication.")
        time_s = t_sorted
    s = pd.Series(angle); angle_s = s.rolling(window=5, center=True, min_periods=1).mean().to_numpy()
    if time_is_frames and fs is not None: vel = np.gradient(angle_s, 1.0/float(fs))
    else: vel = np.gradient(angle_s, time_s)
    return time_s, angle_s, vel

def _resample_to_frames(time_s: np.ndarray, value: np.ndarray, n_frames: int, fs: float) -> np.ndarray:
    t_frames = np.arange(n_frames, dtype=float) / float(fs)
    mask = np.isfinite(time_s) & np.isfinite(value)
    if np.sum(mask) < 2: return np.full(n_frames, np.nan, float)
    ts = time_s[mask]; vs = value[mask]
    v_frames = np.interp(t_frames, ts, vs, left=vs[0], right=vs[-1])
    return v_frames

def _class_windows(n_frames: int, stim_frame: int, during_start: int, during_end: int, post_start: Optional[int]):
    pre = (0, stim_frame); during = (during_start, during_end)
    if post_start is None: post_start = during_end + 1
    post = (post_start, n_frames - 1)
    return {"pre": pre, "during": during, "post": post}

def _events_in_window(peaks: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return peaks[(peaks >= lo) & (peaks <= hi)].astype(int)

def _etb_delta_vigor(vigor_frame: np.ndarray, peak_idx: int, fs: float, pre_s: float, post_s: float) -> float:
    pre_frames = max(1, int(round(pre_s * fs))); post_frames = max(1, int(round(post_s * fs)))
    n = vigor_frame.size; a0 = max(0, peak_idx - pre_frames); a1 = peak_idx; b0 = peak_idx; b1 = min(n, peak_idx + post_frames + 1)
    pre_mean = float(np.nanmean(np.abs(vigor_frame[a0:a1]))) if (a1>a0) else np.nan
    post_mean = float(np.nanmean(np.abs(vigor_frame[b0:b1]))) if (b1>b0) else np.nan
    return post_mean - pre_mean if np.isfinite(pre_mean) and np.isfinite(post_mean) else np.nan

def _p_bout_given_event(peak_time_s: float, bouts_df: Optional[pd.DataFrame], lo_s: float, hi_s: float) -> float:
    if bouts_df is None or bouts_df.empty or "onset_time" not in bouts_df.columns: return np.nan
    ons = bouts_df["onset_time"].to_numpy()
    return float(np.any((ons >= peak_time_s + lo_s) & (ons <= peak_time_s + hi_s)))

def _xcorr_peak(x: np.ndarray, y: np.ndarray, fs: float, max_lag_s: float = 2.0):
    msk = np.isfinite(x) & np.isfinite(y); x = x[msk]; y = y[msk]
    if x.size < 3 or y.size < 3: return (np.nan, np.nan)
    x = (x - np.mean(x)) / (np.std(x) + 1e-12); y = (y - np.mean(y)) / (np.std(y) + 1e-12)
    max_lag = int(round(max_lag_s * fs)); best_r, best_lag = -np.inf, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0: xx = x[-lag:]; yy = y[:len(xx)]
        elif lag > 0: xx = x[:len(x)-lag]; yy = y[lag:]
        else: xx = x; yy = y
        if len(xx) < 3: continue
        r = np.corrcoef(xx, yy)[0,1]
        if np.isfinite(r) and r > best_r: best_r, best_lag = r, lag
    if best_r == -np.inf: return (np.nan, np.nan)
    return float(best_r), float(best_lag/float(fs))

def neural_to_behaviour_alignment_single_roi(
    dff_smooth: Union[pd.DataFrame, np.ndarray],
    fs: float = 3.6,
    stim_frame: int = 11, during_start: int = 12, during_end: int = 50, post_start: int = 51,
    events_df: Optional[pd.DataFrame] = None,
    k_sigma_height: float = 2.5, k_sigma_prom: float = 2.0, min_distance_s: float = 0.5,
    tail_df: Optional[pd.DataFrame] = None,
    bouts_df: Optional[pd.DataFrame] = None,
    etb_pre_s: float = 1.5, etb_post_s: float = 1.5,
    pwin_post_s: float = 1.5, pwin_pre_s: float = 1.5,
    xcorr_max_lag_s: float = 2.0,
    save_prefix: str = "/mnt/data/neuro_behaviour_alignment"
):
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
    df = dff_smooth.copy() if isinstance(dff_smooth, pd.DataFrame) else pd.DataFrame(dff_smooth, columns=[f"ROI_{i+1}" for i in range(dff_smooth.shape[1])])
    n_frames, n_rois = df.shape
    windows = _class_windows(n_frames, stim_frame, during_start, during_end, post_start)
    vigor_frame = None
    if tail_df is not None:
        t_s, angle_s, vel = _compute_tail_velocity(tail_df, fs=fs)
        vigor_frame = _resample_to_frames(t_s, np.abs(vel), n_frames, fs)
    events_by_roi = {}
    if events_df is not None and not events_df.empty:
        if "roi" not in events_df.columns or "peak_idx" not in events_df.columns:
            raise ValueError("events_df must have columns ['roi','peak_idx']")
        for roi in df.columns:
            peaks = events_df.loc[events_df["roi"] == roi, "peak_idx"].to_numpy(dtype=int)
            events_by_roi[roi] = peaks
    else:
        for roi in df.columns:
            y = df[roi].to_numpy(dtype=float)
            peaks = _detect_events_one_roi(y, fs, k_sigma_height, k_sigma_prom, min_distance_s)
            events_by_roi[roi] = peaks
    rows = []
    for roi in df.columns:
        peaks = events_by_roi.get(roi, np.array([], dtype=int))
        for cls, (lo, hi) in windows.items():
            peaks_w = _events_in_window(peaks, lo, hi)
            n_ev = int(peaks_w.size); resp = int(n_ev > 0)
            etb_list = []; p_post_list = []; p_pre_list = []
            for p in peaks_w:
                if vigor_frame is not None and np.isfinite(vigor_frame).any():
                    etb = _etb_delta_vigor(vigor_frame, int(p), fs, etb_pre_s, etb_post_s)
                else:
                    etb = np.nan
                etb_list.append(etb)
                peak_time_s = float(p) / float(fs)
                p_post = _p_bout_given_event(peak_time_s, bouts_df, 0.0, pwin_post_s)
                p_pre  = _p_bout_given_event(peak_time_s, bouts_df, -pwin_pre_s, 0.0)
                p_post_list.append(p_post); p_pre_list.append(p_pre)
            etb_mean = float(np.nanmean(etb_list)) if len(etb_list)>0 else np.nan
            etb_med  = float(np.nanmedian(etb_list)) if len(etb_list)>0 else np.nan
            p_post_mean = float(np.nanmean(p_post_list)) if len(p_post_list)>0 else np.nan
            p_pre_mean  = float(np.nanmean(p_pre_list)) if len(p_pre_list)>0 else np.nan
            p_diff = p_post_mean - p_pre_mean if (np.isfinite(p_post_mean) and np.isfinite(p_pre_mean)) else np.nan
            if vigor_frame is not None:
                x = df[roi].to_numpy(dtype=float)[lo:hi+1]; y = vigor_frame[lo:hi+1]
                r_peak, lag_s = _xcorr_peak(x, y, fs, max_lag_s=xcorr_max_lag_s)
            else:
                r_peak, lag_s = (np.nan, np.nan)
            rows.append({"roi": roi, "class": cls, "n_events": n_ev, "responsive": resp,
                         "etb_delta_vigor_mean": etb_mean, "etb_delta_vigor_median": etb_med,
                         "p_bout_event_post": p_post_mean, "p_bout_event_pre": p_pre_mean, "p_bout_event_diff": p_diff,
                         "xcorr_peak_r": r_peak, "xcorr_peak_lag_s": lag_s})
    roi_window_metrics = pd.DataFrame(rows, columns=["roi","class","n_events","responsive","etb_delta_vigor_mean","etb_delta_vigor_median","p_bout_event_post","p_bout_event_pre","p_bout_event_diff","xcorr_peak_r","xcorr_peak_lag_s"])
    csv_path = f"{save_prefix}_roi_window_alignment.csv"
    roi_window_metrics.to_csv(csv_path, index=False)
    return {"roi_window_metrics": roi_window_metrics, "csv_path": csv_path}
