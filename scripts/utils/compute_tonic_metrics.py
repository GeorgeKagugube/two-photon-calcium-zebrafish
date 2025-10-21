
from typing import Union, Optional
import numpy as np
import pandas as pd

def _ensure_df(x: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    elif isinstance(x, np.ndarray):
        df = pd.DataFrame(x)
        df.columns = [f"ROI_{i+1}" for i in range(df.shape[1])]
        return df
    else:
        raise TypeError("Input must be a pandas DataFrame or numpy ndarray.")

def _running_percentile(df: pd.DataFrame, q: float, win_frames: int) -> pd.DataFrame:
    def _perc(a):
        return float(np.percentile(np.asarray(a), q))
    return df.rolling(window=win_frames, center=True, min_periods=1).apply(_perc, raw=False)

def _mad_sigma(x: np.ndarray) -> float:
    med = np.nanmedian(x)
    return 1.4826 * np.nanmedian(np.abs(x - med)) + 1e-12

def compute_tonic_metrics(
    dff: Union[pd.DataFrame, np.ndarray],
    fs: float = 3.6,
    baseline_percentile: float = 20.0,
    window_seconds: float = 20.0,
    activity_z: float = 2.0,
    save_path: Optional[str] = "/mnt/data/tonic_metrics.csv"
) -> pd.DataFrame:
    df = _ensure_df(dff)
    n_frames, n_rois = df.shape
    fs = float(fs)
    win_frames = max(1, int(round(window_seconds * fs)))
    win_frames = min(win_frames, n_frames)
    f0 = _running_percentile(df, q=baseline_percentile, win_frames=win_frames)
    resid = df - f0
    t = np.arange(n_frames, dtype=float) / fs
    results = []
    for col in df.columns:
        y = df[col].values.astype(float)
        b = f0[col].values.astype(float)
        r = resid[col].values.astype(float)
        valid = np.isfinite(y) & np.isfinite(b)
        if valid.sum() == 0:
            results.append({"ROI": col, "median_dff": np.nan, "duty_cycle": np.nan, "baseline_drift_s": np.nan, "baseline_drift_min": np.nan, "noise_sigma": np.nan})
            continue
        median_dff = float(np.nanmedian(y))
        med = np.nanmedian(r[valid])
        sigma = 1.4826 * np.nanmedian(np.abs(r[valid] - med)) + 1e-12
        if np.isfinite(sigma) and sigma > 0:
            active = (r[valid] > activity_z * sigma).astype(float)
            duty_cycle = float(active.mean()) if active.size > 0 else np.nan
        else:
            duty_cycle = np.nan
        tv = t[valid]
        bv = b[valid]
        if tv.size >= 2:
            coeffs = np.polyfit(tv, bv, deg=1)
            slope_s = float(coeffs[0])
        else:
            slope_s = np.nan
        slope_min = slope_s * 60.0 if np.isfinite(slope_s) else np.nan
        results.append({"ROI": col, "median_dff": median_dff, "duty_cycle": duty_cycle, "baseline_drift_s": slope_s, "baseline_drift_min": slope_min, "noise_sigma": float(sigma)})
    metrics = pd.DataFrame(results).set_index("ROI").sort_index()
    if save_path is not None:
        metrics.to_csv(save_path)
    return metrics
