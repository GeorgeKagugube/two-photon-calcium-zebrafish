
from typing import Tuple, Union, Optional
import numpy as np
import pandas as pd

def compute_dff(
    data: Union[pd.DataFrame, np.ndarray],
    fs: float = 3.6,
    baseline_percentile: float = 20.0,
    window_seconds: float = 20.0,
    detrend: Optional[str] = "poly",   # None, "poly"
    polyorder: int = 1,
    eps: float = 1e-9,
    return_f0: bool = False,
):
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
        df.columns = [f"ROI_{i+1}" for i in range(df.shape[1])]
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError("`data` must be a pandas DataFrame or numpy ndarray.")
    
    n_frames, n_rois = df.shape
    if n_frames < n_rois and n_frames != 108:
        if df.shape[0] < df.shape[1]:
            df = df.T
            n_frames, n_rois = df.shape
    
    t = np.arange(n_frames) / fs
    win_frames = max(1, int(round(window_seconds * fs)))
    win_frames = min(win_frames, n_frames)
    
    if detrend == "poly":
        trend = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
        detr = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
        X = np.vander(t, N=polyorder + 1, increasing=True)
        for col in df.columns:
            y = df[col].values.astype(float)
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
            y_trend = X @ coeffs
            trend[col] = y_trend
            detr[col] = y - y_trend
        df_for_baseline = detr
    else:
        trend = pd.DataFrame(np.zeros_like(df.values), index=df.index, columns=df.columns)
        df_for_baseline = df
    
    def _percentile(arr, q):
        return float(np.percentile(np.asarray(arr), q))
    
    f0 = df_for_baseline.rolling(
        window=win_frames, center=True, min_periods=1
    ).apply(lambda x: _percentile(x, baseline_percentile), raw=False)
    
    f0 = f0 + trend
    f0_safe = f0.clip(lower=eps)
    dff = (df - f0_safe) / f0_safe
    
    if return_f0:
        return dff, f0
    return dff
