import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional

def _window_indices(total_frames: int,
                    windows: Dict[str, Tuple[int, int]]) -> Dict[str, np.ndarray]:
    """
    Build index arrays for each window. end is inclusive.
    """
    idxs = {}
    for name, (start, end) in windows.items():
        start = max(0, int(start))
        end = min(total_frames - 1, int(end))
        if end < start:
            idxs[name] = np.array([], dtype=int)
        else:
            idxs[name] = np.arange(start, end + 1, dtype=int)
    return idxs

def _inter_event_intervals(times_idx: np.ndarray, fs: float) -> np.ndarray:
    """Return IEIs (s) given event indices in frames."""
    if times_idx.size < 2:
        return np.array([], dtype=float)
    return np.diff(times_idx) / fs

def _fano_in_bins(is_spike_row: np.ndarray, fs: float, window_idx: np.ndarray) -> float:
    """
    Fano factor on 1-s bins over the window.
    """
    if window_idx.size == 0:
        return np.nan
    # slice window
    w = is_spike_row[window_idx]
    bin_size = int(max(1, np.round(fs)))  # ~1 s bins
    # pad to multiple of bin_size
    pad = (-w.size) % bin_size
    if pad:
        w = np.pad(w, (0, pad), constant_values=0)
    counts = w.reshape(-1, bin_size).sum(axis=1)
    mean = counts.mean()
    var = counts.var(ddof=1) if counts.size > 1 else 0.0
    return var / mean if mean > 0 else np.nan

def summarise_activity_by_window(
    S_df: pd.DataFrame,
    fs: float = 3.6,
    fish_id: str = "fish001",
    group: str = "WT",
    windows: Optional[Dict[str, Tuple[int, int]]] = None,
    thr: float = 0.0,
    save_prefix: str = "./",
    name = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarise deconvolved spikes per ROI for pre/during/post windows and
    save tidy CSVs for downstream R analysis.

    Parameters
    ----------
    S_df : DataFrame
        Rows = ROIs, Cols = frames (time). Values are deconvolved spikes (continuous or non-negative).
    fs : float
        Sampling rate (Hz), default 3.6.
    fish_id : str
        Identifier for the recording/animal (used in output).
    group : str
        Experimental group label (e.g., 'WT', 'WT + Mn', 'Mutant + Mn').
    windows : dict or None
        Window definition as {'pre': (0,11), 'during': (12,35), 'post': (36,108)}.
        Frame indices are inclusive.
    thr : float
        Threshold on S to count a spike event (0.0 for OASIS is common).
    save_prefix : str
        Path prefix for CSV outputs.

    Returns
    -------
    per_roi_tidy : DataFrame
        One row per ROI × window with metrics.
    per_fish_tidy : DataFrame
        Fish-level robust (median) summaries per window for inference.
    """
    if windows is None:
        windows = {
            "pre":    (0, 11),
            "during": (12, 35),
            "post":   (36, 108),
        }

    n_rois, n_frames = S_df.shape
    idxs = _window_indices(n_frames, windows)

    # Prepare containers
    rows = []
    S = S_df.to_numpy(float)
    rois = S_df.index.to_numpy()

    # Precompute boolean spike matrix for counting & duty cycle
    is_spike = (S > thr)

    for r_i, roi in enumerate(rois):
        s_row = S[r_i, :]
        spike_bool = is_spike[r_i, :]

        for wname, widx in idxs.items():
            duration_s = widx.size / fs if widx.size > 0 else 0.0

            # Spike counts / rate
            spike_count = int(spike_bool[widx].sum()) if widx.size > 0 else 0
            rate_hz = (spike_count / duration_s) if duration_s > 0 else np.nan

            # AUC-based rate proxy (continuous spikes)
            auc_per_s = (s_row[widx].sum() / duration_s) if duration_s > 0 else np.nan

            # Duty cycle
            duty = (spike_bool[widx].mean() if widx.size > 0 else np.nan)

            # IEIs (s) within the window
            event_times = np.where(spike_bool[widx])[0]  # indices within window
            # convert to global frame indices for accurate IEI in time
            event_frames = widx[event_times] if event_times.size else np.array([], dtype=int)
            iei = _inter_event_intervals(event_frames, fs)
            median_iei = np.median(iei) if iei.size else np.nan
            cv_iei = (np.std(iei, ddof=1) / np.mean(iei)) if iei.size >= 2 and np.mean(iei) > 0 else np.nan
            if iei.size and np.isfinite(median_iei) and median_iei > 0:
                burstiness = np.mean(iei < 0.5 * median_iei)
            else:
                burstiness = np.nan

            # Fano factor (1 s bins) in the window
            fano = _fano_in_bins(spike_bool, fs, widx)

            rows.append({
                "ROI": roi,
                "fish_id": fish_id,
                "group": group,
                "window": wname,
                "n_frames": int(widx.size),
                "duration_s": duration_s,
                "spike_count": spike_count,
                "rate_Hz": rate_hz,
                "auc_per_s": auc_per_s,
                "duty_cycle": duty,
                "median_IEI_s": median_iei,
                "cv_IEI": cv_iei,
                "burstiness": burstiness,
                "fano": fano,
                "responsive": bool(spike_count > 0)
            })

    per_roi_tidy = pd.DataFrame(rows)

    # Fish-level robust summaries (median across ROIs), per window
    agg = {
        "rate_Hz": "median",
        "auc_per_s": "median",
        "duty_cycle": "median",
        "median_IEI_s": "median",
        "cv_IEI": "median",
        "burstiness": "median",
        "fano": "median",
        "responsive": "mean",    # proportion responsive ROIs
        "ROI": "count"           # n_ROIs
    }
    per_roi_tidy["ROI"] = per_roi_tidy["ROI"].astype(str)  # for counting
    per_fish = (per_roi_tidy
                .groupby(["fish_id", "group", "window"], as_index=False)
                .agg(agg)
                .rename(columns={"ROI": "n_ROIs",
                                 "responsive": "prop_responsive_ROIs"}))

    # Save CSVs for R
    roi_path = f"{save_prefix.rstrip('/')}/{name}_roi_metrics_by_window.csv"
    fish_path = f"{save_prefix.rstrip('/')}/{name}_fish_summary_by_window.csv"
    per_roi_tidy.to_csv(roi_path, index=False)
    per_fish.to_csv(fish_path, index=False)

    print(f"Saved per-ROI metrics: {roi_path}")
    print(f"Saved fish-level summary: {fish_path}")
    return per_roi_tidy, per_fish
