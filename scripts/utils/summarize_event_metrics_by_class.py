
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

def _detect_events_core_for_metrics(
    y: np.ndarray,
    fs: float,
    k_sigma_height: float,
    k_sigma_prom: float,
    min_distance_s: float,
    pre_window_s: float,
    post_window_s: float,
    merge_within_s: float
) -> pd.DataFrame:
    import numpy as np, pandas as pd
    try:
        from scipy.signal import find_peaks
        SCIPY_AVAILABLE = True
    except Exception:
        SCIPY_AVAILABLE = False
    def _mad(x):
        med = np.nanmedian(x)
        return 1.4826 * np.nanmedian(np.abs(x - med)) + 1e-12
    def _local_baseline(arr, idx, pre):
        start = max(0, idx - pre)
        if start >= idx:
            return float(np.nanmedian(arr[:idx])) if idx > 0 else float(np.nanmedian(arr))
        pre_seg = arr[start:idx]
        if pre_seg.size == 0:
            return float(np.nanmedian(arr[:idx])) if idx > 0 else float(np.nanmedian(arr))
        return float(np.percentile(pre_seg, 20))
    def _onset_offset(arr, peak_idx, baseline, half_frac, pre, post):
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
        base = _local_baseline(y, p, pre_frames)
        onset_idx, offset_idx = _onset_offset(y, p, base, half_frac=0.5, pre=pre_frames, post=post_frames)
        amp = y[p] - base
        rows.append({
            "peak_idx": int(p),
            "onset_idx": int(onset_idx) if onset_idx is not None else np.nan,
            "offset_idx": int(offset_idx) if offset_idx is not None else np.nan,
            "amp": float(amp),
            "prominence": float(prominences[p_idx]) if np.isfinite(prominences[p_idx]) else np.nan,
            "baseline": float(base),
        })
    return pd.DataFrame(rows, columns=["peak_idx","onset_idx","offset_idx","amp","prominence","baseline"])

def _event_metrics_for_roi(y: np.ndarray, events: pd.DataFrame, fs: float) -> pd.DataFrame:
    import numpy as np, pandas as pd
    metrics_rows = []
    for _, ev in events.iterrows():
        p = int(ev["peak_idx"])
        onset_idx = ev["onset_idx"]
        offset_idx = ev["offset_idx"]
        base = ev["baseline"]
        amp = ev["amp"]
        if not np.isfinite(onset_idx) or not np.isfinite(offset_idx):
            continue
        onset_idx = int(onset_idx)
        offset_idx = int(offset_idx)
        if offset_idx <= onset_idx:
            continue
        seg = y[onset_idx:offset_idx + 1] - base
        seg = np.where(np.isfinite(seg), seg, 0.0)
        auc = np.trapz(seg, dx=1.0 / fs)
        up_dt = (p - onset_idx) / fs if p > onset_idx else np.nan
        down_dt = (offset_idx - p) / fs if offset_idx > p else np.nan
        up_slope = (amp / up_dt) if (np.isfinite(up_dt) and up_dt > 0) else np.nan
        peak_val = y[p]
        offset_val = y[offset_idx]
        down_slope = ((offset_val - peak_val) / down_dt) if (np.isfinite(down_dt) and down_dt > 0) else np.nan
        duration_s = (offset_idx - onset_idx) / fs
        metrics_rows.append({
            "AUC": float(auc),
            "peak_amp": float(amp),
            "upstroke_slope": float(up_slope) if np.isfinite(up_slope) else np.nan,
            "downstroke_slope": float(down_slope) if np.isfinite(down_slope) else np.nan,
            "duration_s": float(duration_s)
        })
    return pd.DataFrame(metrics_rows, columns=["AUC","peak_amp","upstroke_slope","downstroke_slope","duration_s"])

def summarize_event_metrics_by_class(
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
    pre_df: Optional[pd.DataFrame] = None,
    during_df: Optional[pd.DataFrame] = None,
    post_df: Optional[pd.DataFrame] = None,
    save_path: str = "/mnt/data/event_metrics_by_class.csv"
) -> pd.DataFrame:
    df = _ensure_df(dff_smooth)
    n_frames, _ = df.shape
    pre_start, pre_end = 0, stim_frame
    dur_start, dur_end = during_start, during_end
    if post_start is None:
        post_start = dur_end + 1
    post_end = n_frames - 1
    roi_list = list(df.columns)
    if pre_df is not None:
        roi_list = sorted(set(roi_list).intersection(set(pre_df.index)))
    if during_df is not None:
        roi_list = sorted(set(roi_list).intersection(set(during_df.index)))
    if post_df is not None:
        roi_list = sorted(set(roi_list).intersection(set(post_df.index)))
    def in_window(ev: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
        if ev.empty:
            return ev
        mask = (ev["peak_idx"] >= lo) & (ev["peak_idx"] <= hi)
        return ev.loc[mask].reset_index(drop=True)
    rows = []
    for roi in roi_list:
        y = df[roi].values.astype(float)
        finite_mask = np.isfinite(y)
        if finite_mask.sum() == 0:
            for cls, lo, hi in [("pre", pre_start, pre_end), ("during", dur_start, dur_end), ("post", post_start, post_end)]:
                rows.append({
                    "ROI": roi, "class": cls, "mean_auc": np.nan, "mean_peak_amp": np.nan,
                    "mean_upstroke_slope": np.nan, "mean_downstroke_slope": np.nan,
                    "mean_duration_s": np.nan, "n_events": 0, "responsive": False
                })
            continue
        ev = _detect_events_core_for_metrics(
            y, fs,
            k_sigma_height, k_sigma_prom,
            min_distance_s, pre_window_s, post_window_s, merge_within_s
        )
        for cls, lo, hi in [("pre", pre_start, pre_end), ("during", dur_start, dur_end), ("post", post_start, post_end)]:
            ev_w = in_window(ev, lo, hi)
            if ev_w.empty:
                rows.append({
                    "ROI": roi, "class": cls, "mean_auc": np.nan, "mean_peak_amp": np.nan,
                    "mean_upstroke_slope": np.nan, "mean_downstroke_slope": np.nan,
                    "mean_duration_s": np.nan, "n_events": 0, "responsive": False
                })
                continue
            ev_metrics = _event_metrics_for_roi(y, ev_w, fs)
            if ev_metrics.empty:
                rows.append({
                    "ROI": roi, "class": cls, "mean_auc": np.nan, "mean_peak_amp": np.nan,
                    "mean_upstroke_slope": np.nan, "mean_downstroke_slope": np.nan,
                    "mean_duration_s": np.nan, "n_events": 0, "responsive": False
                })
                continue
            rows.append({
                "ROI": roi,
                "class": cls,
                "mean_auc": float(ev_metrics["AUC"].mean()),
                "mean_peak_amp": float(ev_metrics["peak_amp"].mean()),
                "mean_upstroke_slope": float(ev_metrics["upstroke_slope"].mean()),
                "mean_downstroke_slope": float(ev_metrics["downstroke_slope"].mean()),
                "mean_duration_s": float(ev_metrics["duration_s"].mean()),
                "n_events": int(ev_w.shape[0]),
                "responsive": True
            })
    out_df = pd.DataFrame(rows).set_index(["ROI", "class"]).sort_index()
    out_df.to_csv(save_path)
    print(f"Saved event metrics by class to: {save_path}")
    return out_df
