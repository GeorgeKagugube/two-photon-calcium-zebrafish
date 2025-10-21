
from typing import Dict, Tuple, Union, Optional
import numpy as np
import pandas as pd
try:
    from scipy.signal import find_peaks
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

def _ensure_df(x: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    elif isinstance(x, np.ndarray):
        df = pd.DataFrame(x)
        df.columns = [f"ROI_{i+1}" for i in range(df.shape[1])]
        return df
    else:
        raise TypeError("Input must be a pandas DataFrame or numpy ndarray.")

def _mad(x: np.ndarray) -> float:
    med = np.nanmedian(x)
    return 1.4826 * np.nanmedian(np.abs(x - med)) + 1e-12

def _local_baseline(arr: np.ndarray, idx: int, pre: int) -> float:
    start = max(0, idx - pre)
    if start >= idx:
        return float(np.nanmedian(arr[:idx])) if idx > 0 else float(np.nanmedian(arr))
    pre_seg = arr[start:idx]
    if pre_seg.size == 0:
        return float(np.nanmedian(arr[:idx])) if idx > 0 else float(np.nanmedian(arr))
    return float(np.percentile(pre_seg, 20))

def _onset_offset(arr: np.ndarray, peak_idx: int, baseline: float, half_frac: float, pre: int, post: int):
    n = arr.size
    left = max(0, peak_idx - pre)
    right = min(n - 1, peak_idx + post)
    peak_amp = arr[peak_idx] - baseline
    if not np.isfinite(peak_amp) or peak_amp <= 0:
        return None, None
    thresh = baseline + half_frac * peak_amp
    onset_idx = None
    for i in range(peak_idx, left - 1, -1):
        if arr[i] <= thresh:
            onset_idx = i
            break
    if onset_idx is None:
        onset_idx = left
    offset_idx = None
    for i in range(peak_idx, right + 1):
        if arr[i] <= thresh:
            offset_idx = i
            break
    if offset_idx is None:
        offset_idx = right
    return onset_idx, offset_idx

def _decay_tau(arr: np.ndarray, peak_idx: int, baseline: float, fs: float, max_seconds: float = 6.0) -> float:
    n = arr.size
    peak_amp = arr[peak_idx] - baseline
    if peak_amp <= 0 or not np.isfinite(peak_amp):
        return float("nan")
    target = baseline + peak_amp / np.e
    max_idx = min(n - 1, peak_idx + int(round(max_seconds * fs)))
    for j in range(peak_idx, max_idx + 1):
        if arr[j] <= target:
            return (j - peak_idx) / fs
    return float("nan")

def detect_events(
    dff_smooth: Union[pd.DataFrame, np.ndarray],
    fs: float = 3.6,
    k_sigma_height: float = 2.5,
    k_sigma_prom: float = 2.0,
    min_distance_s: float = 0.5,
    pre_window_s: float = 2.0,
    post_window_s: float = 4.0,
    merge_within_s: float = 0.3
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    df = _ensure_df(dff_smooth)
    n_frames, n_rois = df.shape
    min_dist = max(1, int(round(min_distance_s * fs)))
    pre_frames = max(1, int(round(pre_window_s * fs)))
    post_frames = max(1, int(round(post_window_s * fs)))
    merge_frames = max(1, int(round(merge_within_s * fs)))
    events_by_roi: Dict[str, pd.DataFrame] = {}
    summary_rows = []
    for col in df.columns:
        y = df[col].values.astype(float)
        sigma = _mad(y)
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = np.nanstd(y) + 1e-12
        height_thr = k_sigma_height * sigma
        prom_thr = k_sigma_prom * sigma
        if SCIPY_AVAILABLE:
            peaks, props = find_peaks(y, height=height_thr, prominence=prom_thr, distance=min_dist)
            prominences = props.get("prominences", np.full_like(peaks, np.nan, dtype=float))
        else:
            peaks = []
            for i in range(1, len(y) - 1):
                if y[i] > y[i-1] and y[i] > y[i+1] and y[i] >= height_thr:
                    if len(peaks) == 0 or (i - peaks[-1]) >= min_dist:
                        peaks.append(i)
            peaks = np.array(peaks, dtype=int)
            prominences = np.full_like(peaks, np.nan, dtype=float)
        if peaks.size > 1 and merge_frames > 1:
            keep = [0]
            for i in range(1, len(peaks)):
                if peaks[i] - peaks[keep[-1]] < merge_frames:
                    keep[-1] = i if y[peaks[i]] > y[peaks[keep[-1]]] else keep[-1]
                else:
                    keep.append(i)
            peaks = peaks[keep]
            if prominences.size == len(keep):
                prominences = prominences[keep]
        rows = []
        for p_idx, p in enumerate(peaks):
            base = _local_baseline(y, p, pre_frames)
            onset_idx, offset_idx = _onset_offset(y, p, base, half_frac=0.5, pre=pre_frames, post=post_frames)
            amp = y[p] - base
            prom = float(prominences[p_idx]) if np.isfinite(prominences[p_idx]) else np.nan
            onset_time = (onset_idx / fs) if onset_idx is not None else np.nan
            offset_time = (offset_idx / fs) if offset_idx is not None else np.nan
            peak_time = p / fs
            rise_time_s = (peak_time - onset_time) if np.isfinite(onset_time) else np.nan
            decay_tau_s = _decay_tau(y, p, base, fs=fs, max_seconds=post_window_s)
            rows.append({
                "peak_idx": int(p),
                "peak_time": float(peak_time),
                "amp": float(amp),
                "prominence": float(prom),
                "onset_idx": int(onset_idx) if onset_idx is not None else np.nan,
                "onset_time": float(onset_time),
                "offset_idx": int(offset_idx) if offset_idx is not None else np.nan,
                "offset_time": float(offset_time),
                "rise_time_s": float(rise_time_s),
                "decay_tau_s": float(decay_tau_s),
            })
        events_df = pd.DataFrame(rows, columns=[
            "peak_idx","peak_time","amp","prominence",
            "onset_idx","onset_time","offset_idx","offset_time",
            "rise_time_s","decay_tau_s"
        ])
        events_by_roi[col] = events_df
        n_ev = int(events_df.shape[0])
        duration_s = n_frames / fs
        event_rate = n_ev / duration_s if duration_s > 0 else np.nan
        summary_rows.append({
            "ROI": col,
            "n_events": n_ev,
            "event_rate_hz": event_rate,
            "median_amp": float(events_df["amp"].median()) if n_ev else np.nan,
            "median_prom": float(events_df["prominence"].median()) if n_ev else np.nan,
            "median_rise_s": float(events_df["rise_time_s"].median()) if n_ev else np.nan,
            "median_decay_tau_s": float(events_df["decay_tau_s"].median()) if n_ev else np.nan,
        })
    roi_summary = pd.DataFrame(summary_rows).set_index("ROI")
    return events_by_roi, roi_summary
