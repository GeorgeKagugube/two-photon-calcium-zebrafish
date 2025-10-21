
from typing import Union, Optional
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

def _detect_events_core(y: np.ndarray, fs: float, k_sigma_height: float, k_sigma_prom: float, min_distance_s: float, pre_window_s: float, post_window_s: float, merge_within_s: float) -> pd.DataFrame:
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
        amp = y[p] - base
        rows.append({"peak_idx": int(p), "amp": float(amp)})
    return pd.DataFrame(rows, columns=["peak_idx","amp"])

def excitability_metrics_by_class(
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
    save_path: str = "/mnt/data/excitability_metrics_by_class.csv"
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
        ev = _detect_events_core(y, fs, k_sigma_height, k_sigma_prom, min_distance_s, pre_window_s, post_window_s, merge_within_s)
        for cls, lo, hi in [("pre", pre_start, pre_end), ("during", dur_start, dur_end), ("post", post_start, post_end)]:
            ev_w = in_window(ev, lo, hi)
            n_ev = int(ev_w.shape[0])
            win_len_s = (hi - lo + 1) / fs if hi >= lo else np.nan
            event_rate = (n_ev / win_len_s) if (win_len_s and win_len_s > 0) else np.nan
            mean_amp = float(ev_w["amp"].mean()) if n_ev else np.nan
            if n_ev >= 2:
                iei = np.diff(ev_w["peak_idx"].values) / fs
                mean_iei = float(np.mean(iei))
                median_iei = float(np.median(iei))
            else:
                mean_iei = np.nan
                median_iei = np.nan
            resp_prob = 1.0 if n_ev > 0 else 0.0
            rows.append({
                "ROI": roi,
                "class": cls,
                "mean_peak_amp": mean_amp,
                "event_rate_hz": event_rate,
                "mean_IEI_s": mean_iei,
                "median_IEI_s": median_iei,
                "response_probability": resp_prob,
                "n_events": n_ev
            })
    out = pd.DataFrame(rows).set_index(["ROI", "class"]).sort_index()
    out.to_csv(save_path)
    return out
