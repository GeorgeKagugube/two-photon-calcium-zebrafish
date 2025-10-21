
# Stimulus-aligned event detection (no OASIS)
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
    peak_amp = arr[peak_idx] - baseline
    if peak_amp <= 0 or not np.isfinite(peak_amp):
        return float("nan")
    target = baseline + peak_amp / np.e
    max_idx = min(arr.size - 1, peak_idx + int(round(max_seconds * fs)))
    for j in range(peak_idx, max_idx + 1):
        if arr[j] <= target:
            return (j - peak_idx) / fs
    return float("nan")

def _detect_events_core(
    y: np.ndarray,
    fs: float,
    k_sigma_height: float,
    k_sigma_prom: float,
    min_distance_s: float,
    pre_window_s: float,
    post_window_s: float,
    merge_within_s: float
) -> pd.DataFrame:
    import numpy as np
    import pandas as pd
    try:
        from scipy.signal import find_peaks
        SCIPY_AVAILABLE = True
    except Exception:
        SCIPY_AVAILABLE = False
    def _mad(x):
        med = np.nanmedian(x)
        return 1.4826 * np.nanmedian(np.abs(x - med)) + 1e-12
    sigma = _mad(y)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.nanstd(y) + 1e-12
    height_thr = k_sigma_height * sigma
    prom_thr = k_sigma_prom * sigma
    min_dist = max(1, int(round(min_distance_s * fs)))
    pre_frames = max(1, int(round(pre_window_s * fs)))
    post_frames = max(1, int(round(post_window_s * fs)))
    merge_frames = max(1, int(round(merge_within_s * fs)))
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
        base = np.percentile(y[max(0, p-pre_frames):p] if p>0 else y[:1], 20) if p>0 else np.percentile(y[:1], 20)
        onset_idx = max(0, p-pre_frames)
        offset_idx = min(len(y)-1, p+post_frames)
        amp = y[p] - base
        rise_time_s = (p - onset_idx)/fs
        rows.append({
            "peak_idx": int(p),
            "peak_time": float(p/fs),
            "amp": float(amp),
            "prominence": np.nan,
            "onset_idx": int(onset_idx),
            "onset_time": float(onset_idx/fs),
            "offset_idx": int(offset_idx),
            "offset_time": float(offset_idx/fs),
            "rise_time_s": float(rise_time_s),
            "decay_tau_s": float('nan'),
        })
    return pd.DataFrame(rows, columns=[
        "peak_idx","peak_time","amp","prominence",
        "onset_idx","onset_time","offset_idx","offset_time",
        "rise_time_s","decay_tau_s"
    ])

def _summarize_window(events_df: pd.DataFrame, roi: str, fs: float, n_frames_in_window: int) -> dict:
    n_ev = int(events_df.shape[0])
    duration_s = n_frames_in_window / fs if n_frames_in_window > 0 else np.nan
    return {
        "ROI": roi,
        "responsive": bool(n_ev > 0),
        "n_events": n_ev,
        "event_rate_hz": (n_ev / duration_s) if (duration_s and duration_s > 0) else np.nan,
        "median_amp": float(events_df["amp"].median()) if n_ev else np.nan,
        "median_prom": float(events_df["prominence"].median()) if n_ev else np.nan,
        "median_rise_s": float(events_df["rise_time_s"].median()) if n_ev else np.nan,
        "median_decay_tau_s": float(events_df["decay_tau_s"].median()) if n_ev else np.nan,
        "first_event_time_s": float(events_df["peak_time"].min()) if n_ev else np.nan,
    }

def detect_events_stim_aligned(
    dff_smooth: Union[pd.DataFrame, np.ndarray],
    fs: float = 3.6,
    k_sigma_height: float = 2.5,
    k_sigma_prom: float = 2.0,
    min_distance_s: float = 0.5,
    pre_window_s: float = 2.0,
    post_window_s: float = 4.0,
    merge_within_s: float = 0.3,
    stim_frame: int = 11,
    during_start: int = 12,
    during_end: int = 50,
    post_start: Optional[int] = 51,
    save_prefix: str = "/mnt/data/stim_windows"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _ensure_df(dff_smooth)
    n_frames, _ = df.shape
    per_roi_events: Dict[str, pd.DataFrame] = {}
    for col in df.columns:
        y = df[col].values.astype(float)
        ev = _detect_events_core(y, fs, k_sigma_height, k_sigma_prom, min_distance_s, pre_window_s, post_window_s, merge_within_s)
        per_roi_events[col] = ev
    pre_start, pre_end = 0, stim_frame
    dur_start, dur_end = during_start, during_end
    if post_start is None:
        post_start = dur_end + 1
    post_end = n_frames - 1
    def filter_by_window(ev, lo, hi):
        if ev.empty:
            return ev
        m = (ev["peak_idx"] >= lo) & (ev["peak_idx"] <= hi)
        return ev.loc[m].reset_index(drop=True)
    pre_rows, dur_rows, post_rows = [], [], []
    for roi, ev in per_roi_events.items():
        pre_ev = filter_by_window(ev, pre_start, pre_end)
        dur_ev = filter_by_window(ev, dur_start, dur_end)
        post_ev = filter_by_window(ev, post_start, post_end)
        pre_rows.append(_summarize_window(pre_ev, roi, fs, n_frames_in_window=(pre_end - pre_start + 1)))
        dur_rows.append(_summarize_window(dur_ev, roi, fs, n_frames_in_window=(dur_end - dur_start + 1)))
        post_rows.append(_summarize_window(post_ev, roi, fs, n_frames_in_window=(post_end - post_start + 1)))
    pre_df = pd.DataFrame(pre_rows).set_index("ROI").sort_index()
    during_df = pd.DataFrame(dur_rows).set_index("ROI").sort_index()
    post_df = pd.DataFrame(post_rows).set_index("ROI").sort_index()
    pre_df.to_csv(f"{save_prefix}_pre.csv")
    during_df.to_csv(f"{save_prefix}_during.csv")
    post_df.to_csv(f"{save_prefix}_post.csv")
    print(f"Saved {save_prefix}_pre.csv, {save_prefix}_during.csv, {save_prefix}_post.csv")
    return pre_df, during_df, post_df
