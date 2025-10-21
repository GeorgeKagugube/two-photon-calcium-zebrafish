
from typing import Union, Optional, Dict, Tuple
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

def _mad_sigma(x: np.ndarray) -> float:
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
    import numpy as np, pandas as pd
    try:
        from scipy.signal import find_peaks
        SCIPY_AVAILABLE = True
    except Exception:
        SCIPY_AVAILABLE = False
    def _mad_sigma(x):
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
    sigma = _mad_sigma(y)
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
        left = max(0, p - pre_frames); right = min(len(y)-1, p + post_frames)
        peak_amp = y[p] - base
        thresh = base + 0.5 * peak_amp if peak_amp > 0 else base
        onset_idx = left
        for i in range(p, left - 1, -1):
            if y[i] <= thresh:
                onset_idx = i
                break
        offset_idx = right
        for i in range(p, right + 1):
            if y[i] <= thresh:
                offset_idx = i
                break
        rows.append({
            "peak_idx": int(p),
            "onset_idx": int(onset_idx),
            "offset_idx": int(offset_idx),
            "amp": float(peak_amp),
            "baseline": float(base),
            "prominence": float(prominences[p_idx]) if np.isfinite(prominences[p_idx]) else np.nan,
        })
    return pd.DataFrame(rows, columns=["peak_idx","onset_idx","offset_idx","amp","baseline","prominence"])

def _decay_tau_1e(y: np.ndarray, p: int, base: float, fs: float, max_seconds: float) -> float:
    amp = y[p] - base
    if not np.isfinite(amp) or amp <= 0:
        return float("nan")
    target = base + amp / np.e
    max_idx = min(len(y) - 1, p + int(round(max_seconds * fs)))
    for j in range(p, max_idx + 1):
        if y[j] <= target:
            return (j - p) / fs
    return float("nan")

def _time_to_fraction(y: np.ndarray, p: int, base: float, fs: float, frac: float, max_seconds: float) -> float:
    amp = y[p] - base
    if not np.isfinite(amp) or amp <= 0:
        return float("nan")
    target = base + frac * amp
    max_idx = min(len(y) - 1, p + int(round(max_seconds * fs)))
    for j in range(p, max_idx + 1):
        if y[j] <= target:
            return (j - p) / fs
    return float("nan")

def _time_to_baseline_z(y: np.ndarray, p: int, base: float, sigma: float, z: float, fs: float, max_seconds: float) -> float:
    if not np.isfinite(sigma) or sigma <= 0:
        return float("nan")
    target = base + z * sigma
    max_idx = min(len(y) - 1, p + int(round(max_seconds * fs)))
    for j in range(p, max_idx + 1):
        if y[j] <= target:
            return (j - p) / fs
    return float("nan")

def _tail_auc(y: np.ndarray, p: int, base: float, fs: float, tail_seconds: float) -> float:
    end_idx = min(len(y) - 1, p + int(round(tail_seconds * fs)))
    seg = y[p:end_idx + 1] - base
    seg = np.where(np.isfinite(seg), seg, 0.0)
    return float(np.trapz(seg, dx=1.0 / fs))

def clearance_metrics_by_class(
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
    tail_seconds: float = 4.0,
    max_decay_seconds: float = 8.0,
    baseline_z: float = 1.0,
    save_path: str = "/mnt/data/clearance_metrics_by_class.csv"
) -> pd.DataFrame:
    df = _ensure_df(dff_smooth)
    n_frames, _ = df.shape
    pre_start, pre_end = 0, stim_frame
    dur_start, dur_end = during_start, during_end
    if post_start is None:
        post_start = dur_end + 1
    post_end = n_frames - 1
    def in_window(ev: pd.DataFrame, lo: int, hi: int) -> pd.DataFrame:
        if ev.empty:
            return ev
        return ev.loc[(ev["peak_idx"] >= lo) & (ev["peak_idx"] <= hi)].reset_index(drop=True)
    rows = []
    for roi in df.columns:
        y = df[roi].values.astype(float)
        med = np.nanmedian(y)
        sigma_global = _mad_sigma(y - med)
        ev = _detect_events_core(y, fs, k_sigma_height, k_sigma_prom, min_distance_s, pre_window_s, post_window_s, merge_within_s)
        for cls, lo, hi in [("pre", pre_start, pre_end), ("during", dur_start, dur_end), ("post", post_start, post_end)]:
            ev_w = in_window(ev, lo, hi)
            metrics = {"decay_tau_1e_s": [], "t90_s": [], "time_to_baseline_s": [], "tail_auc_s": []}
            for _, e in ev_w.iterrows():
                p = int(e["peak_idx"]); base = float(e["baseline"])
                tau1e = _decay_tau_1e(y, p, base, fs, max_seconds=max_decay_seconds)
                t90 = _time_to_fraction(y, p, base, fs, frac=0.1, max_seconds=max_decay_seconds)
                tbase = _time_to_baseline_z(y, p, base, sigma_global, baseline_z, fs, max_seconds=max_decay_seconds)
                auc_tail = _tail_auc(y, p, base, fs, tail_seconds=tail_seconds)
                metrics["decay_tau_1e_s"].append(tau1e)
                metrics["t90_s"].append(t90)
                metrics["time_to_baseline_s"].append(tbase)
                metrics["tail_auc_s"].append(auc_tail)
            n_ev = int(ev_w.shape[0])
            rows.append({
                "ROI": roi,
                "class": cls,
                "mean_decay_tau_1e_s": float(np.nanmean(metrics["decay_tau_1e_s"])) if n_ev else np.nan,
                "mean_t90_s": float(np.nanmean(metrics["t90_s"])) if n_ev else np.nan,
                "mean_time_to_baseline_s": float(np.nanmean(metrics["time_to_baseline_s"])) if n_ev else np.nan,
                "mean_tail_auc_s": float(np.nanmean(metrics["tail_auc_s"])) if n_ev else np.nan,
                "n_events": n_ev,
                "responsive": bool(n_ev > 0)
            })
    out = pd.DataFrame(rows).set_index(["ROI", "class"]).sort_index()
    out.to_csv(save_path)
    return out
