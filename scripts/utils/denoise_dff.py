
from typing import Union, Optional
import numpy as np
import pandas as pd
try:
    from scipy.signal import butter, filtfilt, savgol_filter
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

def _ensure_dataframe(x: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    elif isinstance(x, np.ndarray):
        df = pd.DataFrame(x)
        df.columns = [f"ROI_{i+1}" for i in range(df.shape[1])]
        return df
    else:
        raise TypeError("Input must be a pandas DataFrame or numpy ndarray.")

def _odd(n: int) -> int:
    n = int(max(3, n))
    return n if n % 2 == 1 else n + 1

def _hampel_1d(y: np.ndarray, k: int, n_sigma: float = 3.0) -> np.ndarray:
    y = y.astype(float).copy()
    n = y.size
    if n == 0:
        return y
    ypad = np.pad(y, pad_width=k, mode='reflect')
    out = y.copy()
    for i in range(n):
        win = ypad[i:i+2*k+1]
        med = np.median(win)
        mad = np.median(np.abs(win - med)) + 1e-12
        if np.abs(y[i] - med) > n_sigma * 1.4826 * mad:
            out[i] = med
    return out

def denoise_dff(
    dff: Union[pd.DataFrame, np.ndarray],
    fs: float = 3.6,
    method: str = "butter",
    cutoff_hz: float = 1.0,
    order: int = 3,
    sg_window_seconds: float = 5.0,
    sg_polyorder: int = 2,
    median_window_seconds: Optional[float] = None,
    hampel_window_seconds: Optional[float] = None,
    hampel_n_sigma: float = 3.0,
    zscore_clip: Optional[float] = None,
    preserve_nans: bool = True
) -> pd.DataFrame:
    df = _ensure_dataframe(dff)
    index, columns = df.index, df.columns
    data = df.values.astype(float)
    nan_mask = np.isnan(data)
    if hampel_window_seconds is not None and hampel_window_seconds > 0:
        k = max(1, int(round((hampel_window_seconds * fs) // 2)))
        data_proc = np.where(nan_mask, np.nanmedian(data, axis=0, keepdims=True), data)
        for j in range(data.shape[1]):
            data_proc[:, j] = _hampel_1d(data_proc[:, j], k=k, n_sigma=hampel_n_sigma)
    else:
        data_proc = np.where(nan_mask, np.nanmedian(data, axis=0, keepdims=True), data)
    m = method.lower()
    if m == "butter":
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for Butterworth filtering. Try method='savgol' or 'median'.")
        nyq = fs / 2.0
        Wn = min(0.999, float(cutoff_hz) / nyq)
        b, a = butter(N=order, Wn=Wn, btype="low", analog=False)
        for j in range(data_proc.shape[1]):
            col = data_proc[:, j]
            med = np.nanmedian(col)
            col = np.where(np.isnan(col), med, col)
            data_proc[:, j] = filtfilt(b, a, col, method="gust")
    elif m == "savgol":
        if not SCIPY_AVAILABLE:
            raise ImportError("scipy is required for Savitzky–Golay filtering. Try method='median'.")
        win = _odd(int(round(sg_window_seconds * fs)))
        for j in range(data_proc.shape[1]):
            col = data_proc[:, j]
            med = np.nanmedian(col)
            col = np.where(np.isnan(col), med, col)
            poly = min(sg_polyorder, max(2, (win - 1) // 2))
            data_proc[:, j] = savgol_filter(col, window_length=win, polyorder=poly, mode="interp")
    elif m == "median":
        win = max(3, int(round(sg_window_seconds * fs)))
        tmp_df = pd.DataFrame(data_proc, columns=columns).rolling(window=win, center=True, min_periods=1).median()
        data_proc = tmp_df.values
    elif m == "none":
        pass
    else:
        raise ValueError("Unknown method. Choose from {'butter','savgol','median','none'}.")
    if median_window_seconds is not None and median_window_seconds > 0:
        win = max(3, int(round(median_window_seconds * fs)))
        tmp_df = pd.DataFrame(data_proc, columns=columns).rolling(window=win, center=True, min_periods=1).median()
        data_proc = tmp_df.values
    if zscore_clip is not None and zscore_clip > 0:
        mu = np.nanmean(data_proc, axis=0, keepdims=True)
        sd = np.nanstd(data_proc, axis=0, keepdims=True) + 1e-12
        z = (data_proc - mu) / sd
        z = np.clip(z, -zscore_clip, zscore_clip)
        data_proc = mu + z * sd
    if preserve_nans:
        data_proc[nan_mask] = np.nan
    return pd.DataFrame(data_proc, index=index, columns=columns)
