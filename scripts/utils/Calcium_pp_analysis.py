#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCaMP epoch pipeline (flexible input + OASIS deconvolution with per-ROI auto-tuning)

Accepts ROI data as either:
  - 2D NumPy array or Pandas DataFrame (frames × ROIs) -> provide frames_per_epoch
  - 3D NumPy array (epochs × frames × ROIs)

Optional synchronized behaviors:
  behavior_input = {"eye_left": ..., "eye_right": ..., "tail": ...}
  Each value can be concatenated (frames,) or epochized (epochs, frames), as ndarray or DataFrame.

Computes:
  - ΔF/F0, OASIS deconvolution (optional; returns S_hat and C_hat)
  - Per-ROI auto-tuning of OASIS: noise estimate (sn), λ grid search (BIC), g (AR coeff)
  - Event detection (by default, on deconvolved spikes if OASIS is enabled)
  - Per-ROI metrics (event rate, amplitude, AUC, half-decay, baseline F0)
  - Stimulus-locked metrics (responsive fraction, peak, AUC, latency, reliability)
  - Population metrics (coactivity, |corr|, PC1 variance explained, mean event rate)
  - Per-epoch ROI×ROI correlation matrices
  - Behavior alignment (corr and peak cross-corr/lag vs pop-mean and PC1)
  - Fish-level summaries & cohort aggregation helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Try to import OASIS (optional)
try:
    from oasis.functions import deconvolve as oasis_deconvolve  # type: ignore
    _OASIS_AVAILABLE = True
except Exception:
    _OASIS_AVAILABLE = False


# ======================================================================
# Input normalization
# ======================================================================

ArrayLike2D = Union[np.ndarray, pd.DataFrame]


def _to_numpy_2d(x: ArrayLike2D) -> np.ndarray:
    """Ensure a (frames, rois) NumPy array from ndarray or DataFrame."""
    if isinstance(x, pd.DataFrame):
        arr = x.to_numpy()
    elif isinstance(x, np.ndarray):
        arr = x
    else:
        raise TypeError(
            "ROI matrix must be a numpy array or pandas DataFrame (frames × rois)."
        )
    if arr.ndim != 2:
        raise ValueError("ROI matrix must be 2D with shape (frames, rois).")
    return arr


def _reshape_frames_to_epochs(M: ArrayLike2D, frames_per_epoch: int) -> np.ndarray:
    """Convert (T_total, N) to (E, F, N) by splitting rows every F frames."""
    M = _to_numpy_2d(M)
    t_total, n = M.shape
    if t_total % frames_per_epoch != 0:
        raise ValueError(
            f"T_total={t_total} not divisible by frames_per_epoch={frames_per_epoch}."
        )
    e = t_total // frames_per_epoch
    return M.reshape(e, frames_per_epoch, n)


def normalize_roi_input(
    roi_input: Union[np.ndarray, pd.DataFrame],
    frames_per_epoch: Optional[int] = None,
) -> np.ndarray:
    """
    Return (E, F, N) ndarray from either:
      - (E, F, N) ndarray (returned unchanged), or
      - (T_total, N) 2D array/DataFrame (requires frames_per_epoch to reshape).
    """
    if isinstance(roi_input, np.ndarray) and roi_input.ndim == 3:
        return roi_input
    if frames_per_epoch is None:
        raise ValueError("frames_per_epoch must be provided for 2D ROI input.")
    return _reshape_frames_to_epochs(roi_input, frames_per_epoch)


def _behavior_to_epochs(
    x: Union[np.ndarray, pd.DataFrame],
    frames_per_epoch: int,
    epochs: Optional[int] = None,
) -> np.ndarray:
    """
    Accept 1D/2D array/DF; return (E, F) epochized behavior array.
    """
    if isinstance(x, pd.DataFrame):
        arr = x.to_numpy()
    else:
        arr = np.asarray(x)

    if arr.ndim == 1:
        t_total = arr.size
        if t_total % frames_per_epoch != 0:
            raise ValueError("Behavior length not divisible by frames_per_epoch.")
        e = t_total // frames_per_epoch
        out = arr.reshape(e, frames_per_epoch)

    elif arr.ndim == 2:
        # Already (E, F)
        if arr.shape[1] == frames_per_epoch:
            out = arr
        # Column vector -> reshape
        elif arr.shape[1] == 1:
            t_total = arr.size
            if t_total % frames_per_epoch != 0:
                raise ValueError("Behavior length not divisible by frames_per_epoch.")
            e = t_total // frames_per_epoch
            out = arr.reshape(e, frames_per_epoch)
        else:
            # Assume (T_total, ?) concatenated along rows
            t_total = arr.shape[0]
            if t_total % frames_per_epoch != 0:
                raise ValueError("Behavior length not divisible by frames_per_epoch.")
            e = t_total // frames_per_epoch
            out = arr[:, 0].reshape(e, frames_per_epoch)  # take 1st col if >1

    else:
        raise ValueError("Behavior must be 1D or 2D.")

    if epochs is not None and out.shape[0] != epochs:
        raise ValueError(
            f"Behavior epochs {out.shape[0]} != ROI epochs {epochs}."
        )
    return out


def normalize_behavior(
    behavior_input: Optional[Dict[str, Union[np.ndarray, pd.DataFrame]]],
    frames_per_epoch: int,
    epochs: int,
) -> Optional[Dict[str, np.ndarray]]:
    """Normalize behavior dict to (E, F) arrays for keys: eye_left, eye_right, tail."""
    if behavior_input is None:
        return None
    out: Dict[str, np.ndarray] = {}
    for key in ("eye_left", "eye_right", "tail"):
        val = behavior_input.get(key, None)
        if val is None:
            continue
        out[key] = _behavior_to_epochs(val, frames_per_epoch, epochs)
    return out


# ======================================================================
# Core signal processing
# ======================================================================

def rolling_percentile_baseline(
    x: np.ndarray, percentile: float = 20.0, window: int = 51
) -> np.ndarray:
    """Rolling-percentile baseline for a 1D trace."""
    n = len(x)
    if window > n:
        window = n if n % 2 == 1 else n - 1
    half = window // 2
    baseline = np.zeros_like(x, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        baseline[i] = np.percentile(x[lo:hi], percentile)
    return baseline


def to_dff(x: np.ndarray, baseline: Optional[np.ndarray] = None) -> np.ndarray:
    """Convert fluorescence to ΔF/F0."""
    if baseline is None:
        base = np.percentile(x, 20.0)
        baseline = np.full_like(x, base, dtype=float)
    eps = 1e-9
    return (x - baseline) / (baseline + eps)


def zscore(x: np.ndarray) -> np.ndarray:
    mu = np.nanmean(x)
    sd = np.nanstd(x) + 1e-12
    return (x - mu) / sd


def detect_events_from_signal(
    sig: np.ndarray, z_threshold: float = 2.0, refractory_frames: int = 2
) -> np.ndarray:
    """
    Simple peak-based detector on a 1D signal (deconvolved spikes or ΔF/F0).
    Returns a binary vector with 1 at detected peaks.
    """
    zt = zscore(sig)
    n = len(zt)
    peaks = np.zeros(n, dtype=int)
    i = 1
    while i < n - 1:
        if zt[i] > z_threshold and zt[i] >= zt[i - 1] and zt[i] >= zt[i + 1]:
            peaks[i] = 1
            i += refractory_frames
        else:
            i += 1
    return peaks


def event_amplitudes(
    dff: np.ndarray, peaks: np.ndarray, pre: int = 1, post: int = 3
) -> np.ndarray:
    """Event amplitude as max ΔF/F0 in [i-pre, i+post] around each peak."""
    idxs = np.where(peaks == 1)[0]
    amps: List[float] = []
    n = len(dff)
    for i in idxs:
        lo = max(0, i - pre)
        hi = min(n, i + post + 1)
        amps.append(float(np.nanmax(dff[lo:hi])))
    return np.array(amps, dtype=float)


def event_auc(dff: np.ndarray, peaks: np.ndarray, width: int = 8) -> np.ndarray:
    """Event AUC as integral of positive ΔF/F0 over fixed window after peak."""
    idxs = np.where(peaks == 1)[0]
    aucs: List[float] = []
    n = len(dff)
    for i in idxs:
        hi = min(n, i + width)
        seg = dff[i:hi]
        aucs.append(float(np.trapz(np.clip(seg, 0, None))))
    return np.array(aucs, dtype=float)


def half_decay_time(
    dff: np.ndarray, peaks: np.ndarray, max_window: int = 30
) -> List[float]:
    """Frames to fall to 50% of excursion after each peak."""
    n = len(dff)
    results: List[float] = []
    for i in np.where(peaks == 1)[0]:
        peak_val = dff[i]
        base = np.nanmedian(dff[max(0, i - 5): i + 1])
        target = base + 0.5 * (peak_val - base)
        t_half = None
        for k in range(i + 1, min(n, i + max_window)):
            if dff[k] <= target:
                t_half = k - i
                break
        if t_half is not None:
            results.append(float(t_half))
    return results


def corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    if np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def crosscorr_peak_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> Tuple[float, int]:
    """
    Peak (max |r|) normalized cross-correlation within ±max_lag frames.
    Returns (peak_r, lag); positive lag means b lags a (a leads).
    """
    a = a - np.nanmean(a)
    b = b - np.nanmean(b)
    denom = (np.nanstd(a) + 1e-12) * (np.nanstd(b) + 1e-12)
    best_r, best_lag = np.nan, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            r = np.nansum(a[:lag] * b[-lag:]) / denom
        elif lag > 0:
            r = np.nansum(a[lag:] * b[:-lag]) / denom
        else:
            r = np.nansum(a * b) / denom
        if not np.isnan(r):
            if np.isnan(best_r) or abs(r) > abs(best_r):
                best_r, best_lag = float(r), lag
    return best_r, best_lag


# ======================================================================
# ΔF/F0, OASIS deconvolution, and events (epochized)
# ======================================================================

def compute_dff_epochs(
    f_epochs: np.ndarray, baseline_percentile: float = 20.0, baseline_window: int = 51
) -> np.ndarray:
    """ΔF/F0 per epoch and ROI."""
    e, _, n = f_epochs.shape
    dff = np.zeros_like(f_epochs, dtype=float)
    for i in range(e):
        for j in range(n):
            base = rolling_percentile_baseline(
                f_epochs[i, :, j], percentile=baseline_percentile, window=baseline_window
            )
            dff[i, :, j] = to_dff(f_epochs[i, :, j], baseline=base)
    return dff


# ---------------------------- OASIS helpers ----------------------------

def _estimate_noise_sn(dff_1d: np.ndarray) -> float:
    """
    Robust noise estimate for ΔF/F0 using MAD of first differences.
    sn ≈ 1.4826 * MAD(diff(x)) / sqrt(2)
    """
    x = np.asarray(dff_1d, float)
    if x.size < 3:
        return float(np.nanstd(x))
    dx = np.diff(x)
    mad = np.median(np.abs(dx - np.median(dx)))
    return float(1.4826 * mad / np.sqrt(2.0) + 1e-12)


def deconvolve_oasis_trace(
    dff_1d: np.ndarray,
    penalty: str = "l1",
    method: str = "foopsi",
    g: Optional[Union[float, List[float]]] = None,
    s_min: float = 0.0,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, float, Union[float, np.ndarray], float]:
    """
    Run OASIS deconvolution on a single ΔF/F0 trace.

    Returns (c_hat, s_hat, b, g, lam) matching OASIS' API.
    """
    if not _OASIS_AVAILABLE:
        raise RuntimeError(
            "OASIS is not available. Install with `pip install oasis` "
            "or set use_deconvolution=False."
        )
    # oasis.functions.deconvolve returns (c, s, b, g, lam)
    c_hat, s_hat, b, g_out, lam = oasis_deconvolve(
        dff_1d, penalty=penalty, method=method, g=g, s_min=s_min, **kwargs
    )
    return c_hat, s_hat, b, g_out, lam


def _bic_of_residual(residual: np.ndarray, k_spikes: int) -> float:
    """Compute a simple BIC-like score using residuals and spike count."""
    n = residual.size
    rss = float(np.nansum(residual ** 2))
    # regularize to avoid log(0)
    return n * np.log(rss / max(n, 1) + 1e-12) + k_spikes * np.log(max(n, 2))


def auto_tune_oasis_per_roi(
    dff_epochs_roi: np.ndarray,
    penalty: str = "l1",
    method: str = "foopsi",
    lam_factors: Optional[List[float]] = None,
    s_min: float = 0.0,
    g_init: Optional[Union[float, List[float]]] = None,
    **kwargs,
) -> Dict[str, float]:
    """
    Auto-tune OASIS for a single ROI using concatenated epochs.

    Strategy:
      1) Estimate noise sn via robust MAD on diff(dff).
      2) Build λ grid = lam_factors * sn.
      3) For each λ: run OASIS on concatenated trace (with g=g_init or letting OASIS estimate),
         compute BIC over residual (dff - c_hat) and spike count.
      4) Pick λ with lowest BIC. Record the corresponding g returned by OASIS.

    Returns a dict with {'sn','lam','g','bic','k_spikes'}.
    """
    if not _OASIS_AVAILABLE:
        return {"sn": np.nan, "lam": np.nan, "g": np.nan, "bic": np.nan, "k_spikes": np.nan}

    if lam_factors is None or len(lam_factors) == 0:
        lam_factors = [0.25, 0.5, 1.0, 2.0, 4.0]

    x = dff_epochs_roi.reshape(-1)
    sn = _estimate_noise_sn(x)

    best = {"bic": np.inf, "lam": None, "g": None, "k_spikes": None}

    for lf in lam_factors:
        lam_try = max(lf * sn, 1e-8)
        try:
            c_hat, s_hat, _, g_out, _ = deconvolve_oasis_trace(
                x, penalty=penalty, method=method, g=g_init, s_min=s_min, lam=lam_try, **kwargs
            )
            k_spikes = int(np.count_nonzero(s_hat > 0))
            bic = _bic_of_residual(x - c_hat, k_spikes)
            if bic < best["bic"]:
                best = {"bic": bic, "lam": float(lam_try),
                        "g": (float(g_out) if np.isscalar(g_out) else (float(g_out[0]) if g_out is not None else np.nan)),
                        "k_spikes": k_spikes}
        except Exception:
            # skip failed settings
            continue

    if best["lam"] is None:
        # fall back to OASIS letting it choose lam
        try:
            c_hat, s_hat, _, g_out, lam_out = deconvolve_oasis_trace(
                x, penalty=penalty, method=method, g=g_init, s_min=s_min, **kwargs
            )
            k_spikes = int(np.count_nonzero(s_hat > 0))
            bic = _bic_of_residual(x - c_hat, k_spikes)
            best = {"bic": bic, "lam": float(lam_out),
                    "g": (float(g_out) if np.isscalar(g_out) else (float(g_out[0]) if g_out is not None else np.nan)),
                    "k_spikes": k_spikes}
        except Exception:
            best = {"bic": np.nan, "lam": np.nan, "g": np.nan, "k_spikes": np.nan}

    return {"sn": float(sn),
            "lam": float(best["lam"]) if best["lam"] is not None else np.nan,
            "g": float(best["g"]) if best["g"] is not None else np.nan,
            "bic": float(best["bic"]),
            "k_spikes": float(best["k_spikes"]) if best["k_spikes"] is not None else np.nan}


def deconvolve_epochs_with_autotune(
    dff: np.ndarray,
    penalty: str = "l1",
    method: str = "foopsi",
    lam_factors: Optional[List[float]] = None,
    s_min: float = 0.0,
    g_init: Optional[Union[float, List[float]]] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    OASIS deconvolution with per-ROI auto-tuning.

    Tuning is performed on the concatenated ΔF/F0 of each ROI to find λ (and record g).
    The tuned (λ, g) are then applied to deconvolve each epoch for that ROI.

    Returns:
      C_hat: (E, F, N) deconvolved calcium
      S_hat: (E, F, N) deconvolved spikes
      tuning_table: DataFrame with columns [roi, sn, lam, g, bic, k_spikes]
    """
    e, f, n = dff.shape
    C_hat = np.zeros_like(dff, dtype=float)
    S_hat = np.zeros_like(dff, dtype=float)
    rows: List[Dict] = []

    if not _OASIS_AVAILABLE:
        print(
            "[WARN] OASIS not found. Proceeding without deconvolution. "
            "Install `oasis` to enable OASIS deconvolution."
        )
        return C_hat, S_hat, pd.DataFrame(columns=["roi", "sn", "lam", "g", "bic", "k_spikes"])

    for j in range(n):
        # Auto-tune on concatenated epochs for ROI j
        tune = auto_tune_oasis_per_roi(
            dff_epochs_roi=dff[:, :, j],
            penalty=penalty,
            method=method,
            lam_factors=lam_factors,
            s_min=s_min,
            g_init=g_init,
            **kwargs,
        )
        lam_best = tune["lam"] if not np.isnan(tune["lam"]) else None
        g_best: Optional[Union[float, List[float]]] = None if np.isnan(tune["g"]) else tune["g"]

        # Apply tuned params to every epoch for this ROI
        for i in range(e):
            try:
                c_hat, s_hat, _, _, _ = deconvolve_oasis_trace(
                    dff[i, :, j],
                    penalty=penalty,
                    method=method,
                    g=g_best,
                    s_min=s_min,
                    lam=lam_best,
                    **kwargs,
                )
                C_hat[i, :, j] = c_hat
                S_hat[i, :, j] = s_hat
            except Exception:
                # Fallback: try with None to let OASIS choose
                try:
                    c_hat, s_hat, _, _, _ = deconvolve_oasis_trace(
                        dff[i, :, j],
                        penalty=penalty,
                        method=method,
                        g=None,
                        s_min=s_min,
                        **kwargs,
                    )
                    C_hat[i, :, j] = c_hat
                    S_hat[i, :, j] = s_hat
                except Exception:
                    # leave zeros
                    pass

        rows.append({"roi": j, **tune})

    tuning_table = pd.DataFrame(rows)
    return C_hat, S_hat, tuning_table


def detect_events_epochs(
    dff: np.ndarray,
    use_deconv: bool,
    s_hat: Optional[np.ndarray],
    z_threshold: float,
    refractory_frames: int,
) -> np.ndarray:
    """
    Detect events from either deconvolved spikes (preferred) or ΔF/F0.
      dff: (E, F, N)
      s_hat: (E, F, N) or None
    Returns events (E, F, N) as 0/1.
    """
    e, f, n = dff.shape
    events = np.zeros_like(dff, dtype=int)

    for i in range(e):
        for j in range(n):
            signal = s_hat[i, :, j] if (use_deconv and s_hat is not None and s_hat.any()) else dff[i, :, j]
            peaks = detect_events_from_signal(signal, z_threshold=z_threshold, refractory_frames=refractory_frames)
            events[i, :, j] = peaks
    return events


# ======================================================================
# Metrics (per-ROI, stimulus-locked, population, correlations)
# ======================================================================

def compute_single_roi_metrics_epochs(
    f_epochs: np.ndarray,
    fps: float,
    events: np.ndarray,
) -> pd.DataFrame:
    """Per-ROI spontaneous metrics aggregated across epochs (events provided)."""
    dff = compute_dff_epochs(f_epochs)
    e, _, n = f_epochs.shape

    rows: List[Dict] = []
    for j in range(n):
        dff_flat = dff[:, :, j].reshape(-1)
        ev_flat = events[:, :, j].reshape(-1)
        erate = ev_flat.sum() / (len(dff_flat) / fps)
        amps = event_amplitudes(dff_flat, ev_flat, pre=1, post=3)
        aucs = event_auc(dff_flat, ev_flat, width=int(0.5 * fps) if fps > 1 else 2)
        t_halfs = half_decay_time(
            dff_flat, ev_flat, max_window=int(2.0 * fps) if fps > 1 else 6
        )
        base_f0 = np.percentile(f_epochs[:, :, j], 20.0)
        rows.append(
            {
                "roi": j,
                "event_rate_hz": float(erate),
                "mean_amp": float(np.nanmean(amps)) if len(amps) else np.nan,
                "median_amp": float(np.nanmedian(amps)) if len(amps) else np.nan,
                "mean_auc": float(np.nanmean(aucs)) if len(aucs) else np.nan,
                "median_auc": float(np.nanmedian(aucs)) if len(aucs) else np.nan,
                "median_t_half_frames": float(np.nanmedian(t_halfs))
                if len(t_halfs)
                else np.nan,
                "baseline_f0": float(base_f0),
            }
        )
    return pd.DataFrame(rows)


def compute_stimulus_locked_metrics(
    f_epochs: np.ndarray,
    stim_frame: int,
    fps: float,
    pre_frames: int = 10,
    post_frames: int = 40,
) -> pd.DataFrame:
    """Per-ROI stimulus-locked metrics across epochs."""
    dff = compute_dff_epochs(f_epochs)
    e, t, n = dff.shape
    lo = max(0, stim_frame - pre_frames)
    hi = min(t, stim_frame + post_frames)

    rows: List[Dict] = []
    for j in range(n):
        windows = dff[:, lo:hi, j]
        mean_trace = np.nanmean(windows, axis=0)
        post_seg = windows[:, (stim_frame - lo):, ]
        base_seg = windows[:, : (stim_frame - lo), ]

        peaks = np.nanmax(post_seg, axis=1)
        aucs = np.trapz(np.clip(post_seg, 0, None), axis=1)

        latencies: List[float] = []
        for i in range(e):
            seg = post_seg[i]
            peak_val = np.nanmax(seg)
            idx = np.where(seg >= peak_val)[0]
            lat = float(idx[0]) if len(idx) else np.nan
            latencies.append(lat)

        responsive = []
        for i in range(e):
            mu = float(np.nanmean(base_seg[i])) if base_seg.shape[1] > 0 else 0.0
            sd = float(np.nanstd(base_seg[i])) if base_seg.shape[1] > 0 else 1.0
            responsive.append(float(peaks[i] > (mu + 2.0 * sd)))

        reliab = []
        for i in range(e):
            a = windows[i]
            b = mean_trace
            if np.nanstd(a) < 1e-12 or np.nanstd(b) < 1e-12:
                reliab.append(np.nan)
            else:
                reliab.append(float(np.corrcoef(a, b)[0, 1]))

        rows.append(
            {
                "roi": j,
                "responsive_fraction": float(np.nanmean(responsive)),
                "peak_dff_mean": float(np.nanmean(peaks)),
                "auc_post_stim_mean": float(np.nanmean(aucs)),
                "latency_frames_median": float(np.nanmedian(latencies)),
                "trial_reliability": float(np.nanmean(reliab)),
            }
        )

    return pd.DataFrame(rows)


@dataclass
class PopulationMetrics:
    mean_coactivity: float
    std_coactivity: float
    mean_abs_corr: float
    pc1_variance_explained: float
    mean_event_rate_hz: float


def compute_population_metrics(
    f_epochs: np.ndarray, fps: float, events: np.ndarray
) -> PopulationMetrics:
    """Population metrics aggregated across epochs (events provided)."""
    dff = compute_dff_epochs(f_epochs)

    # Coactivity
    coacts = [np.mean(events[i], axis=1) for i in range(f_epochs.shape[0])]
    coacts = np.concatenate(coacts, axis=0)
    mean_coact = float(np.nanmean(coacts))
    std_coact = float(np.nanstd(coacts))

    # Pairwise correlations (concatenate epochs along time)
    x = dff.reshape(-1, dff.shape[2])  # (E*F, N)
    cmat = np.corrcoef(x.T)
    iu = np.triu_indices_from(cmat, k=1)
    mean_abs_corr = float(np.nanmean(np.abs(cmat[iu])))

    # PCA variance explained
    xz = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-12)
    _, s, _ = np.linalg.svd(xz, full_matrices=False)
    ve = (s ** 2) / (s ** 2).sum()
    pc1 = float(ve[0]) if len(ve) else float("nan")

    # Mean event rate across ROIs (using provided events)
    mean_event_rate = float(events.sum() / x.shape[1] / (x.shape[0] / fps))

    return PopulationMetrics(
        mean_coactivity=mean_coact,
        std_coactivity=std_coact,
        mean_abs_corr=mean_abs_corr,
        pc1_variance_explained=pc1,
        mean_event_rate_hz=mean_event_rate,
    )


def compute_epoch_correlation_matrices(f_epochs: np.ndarray) -> np.ndarray:
    """ROI–ROI Pearson correlation matrices per epoch on ΔF/F0. Returns (E, N, N)."""
    dff = compute_dff_epochs(f_epochs)
    e, _, n = dff.shape
    out = np.zeros((e, n, n), dtype=float)
    for i in range(e):
        out[i] = np.corrcoef(dff[i].T)
    return out


# ======================================================================
# Behavior alignment
# ======================================================================

def compute_behavior_alignment(
    f_epochs: np.ndarray,
    stim_frame: int,
    fps: float,
    behavior: Optional[Dict[str, np.ndarray]] = None,
    max_lag_frames: int = 15,
) -> pd.DataFrame:
    """
    Align population activity with behavior signals per epoch.

    behavior: dict with optional keys 'eye_left', 'eye_right', 'tail' -> arrays (E, T)
    Returns a DataFrame with columns:
        signal, epoch, r_popmean, r_pc1, xcorr_popmean, lag_popmean, xcorr_pc1, lag_pc1
    """
    dff = compute_dff_epochs(f_epochs)
    e, t, _ = dff.shape
    if behavior is None:
        return pd.DataFrame()

    rows: List[Dict] = []
    for i in range(e):
        x = dff[i]  # (T, N)
        pop_mean = np.nanmean(x, axis=1)

        xz = (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-12)
        u, s, _ = np.linalg.svd(xz, full_matrices=False)
        pc1_tc = u[:, 0] * s[0] if s.size > 0 else np.zeros(t)

        for name in ("eye_left", "eye_right", "tail"):
            ymat = behavior.get(name, None)
            if ymat is None:
                continue
            y = ymat[i]
            r_pop = corrcoef_safe(pop_mean, y)
            r_pc1 = corrcoef_safe(pc1_tc, y)
            x_pop, lag_pop = crosscorr_peak_lag(pop_mean, y, max_lag_frames)
            x_pc1, lag_pc1 = crosscorr_peak_lag(pc1_tc, y, max_lag_frames)
            rows.append(
                {
                    "signal": name,
                    "epoch": i,
                    "r_popmean": r_pop,
                    "r_pc1": r_pc1,
                    "xcorr_popmean": x_pop,
                    "lag_popmean": lag_pop,
                    "xcorr_pc1": x_pc1,
                    "lag_pc1": lag_pc1,
                }
            )
    return pd.DataFrame(rows)


# ======================================================================
# Public API (flexible input + OASIS auto-tuning)
# ======================================================================

def build_outputs(
    roi_input: Union[np.ndarray, pd.DataFrame],
    stim_frame: int,
    fps: Optional[float] = None,
    frames_per_epoch: Optional[int] = None,
    epoch_seconds: Optional[float] = 30.0,
    behavior_input: Optional[Dict[str, Union[np.ndarray, pd.DataFrame]]] = None,
    # Deconvolution options
    use_deconvolution: bool = True,
    auto_tune_deconv: bool = True,
    deconv_penalty: str = "l1",
    deconv_method: str = "foopsi",
    deconv_g_init: Optional[Union[float, List[float]]] = None,
    deconv_s_min: float = 0.0,
    lam_factors: Optional[List[float]] = None,  # e.g., [0.25, 0.5, 1, 2, 4]
    # Event detection thresholds
    event_z_threshold: float = 2.0,
    event_refractory_frames: Optional[int] = None,
    # OASIS kwargs passthrough (sn, lam, etc.)
    **oasis_kwargs,
) -> Dict[str, object]:
    """
    Flexible entry point with optional OASIS deconvolution and per-ROI auto-tuning.

    Parameters
    ----------
    roi_input
        Either:
          - (E, F, N) ndarray, or
          - (T_total, N) 2D array / DataFrame (frames × ROIs); set frames_per_epoch.
    stim_frame
        0-based stimulus frame index within each epoch (e.g., 11).
    fps
        Frames per second. If None, computed as F / epoch_seconds.
    frames_per_epoch
        Required only if roi_input is 2D (frames × ROIs).
    epoch_seconds
        Used to infer fps if not provided (default 30 s).
    behavior_input
        Dict of behaviors keyed by {'eye_left','eye_right','tail'}.
    use_deconvolution
        If True and OASIS is installed, deconvolve ΔF/F0 and detect events on spikes.
    auto_tune_deconv
        If True, perform per-ROI auto-tuning (λ grid, BIC) on concatenated epochs.
    deconv_penalty, deconv_method, deconv_g_init, deconv_s_min
        OASIS settings; passed to oasis.functions.deconvolve.
    lam_factors
        Multipliers for noise sn to build λ grid (default: [0.25, 0.5, 1, 2, 4]).
    event_z_threshold
        Z-threshold used for peak picking on the chosen signal (spikes if deconv, else ΔF/F0).
    event_refractory_frames
        Refractory in frames (default: ~0.25 s worth, computed from fps if None).
    **oasis_kwargs
        Extra keyword args passed to OASIS deconvolve (e.g., 'lam', 'sn').

    Returns
    -------
    dict with:
        per_roi_metrics (DataFrame),
        stim_metrics (DataFrame),
        population_metrics (dict),
        dff (E, F, N),
        events (E, F, N),
        epoch_corrs (E, N, N),
        behavior_alignment (DataFrame),
        deconv (dict: 'C_hat', 'S_hat', 'tuning')  # tuning is per-ROI table
        fps, frames_per_epoch, epochs, n_rois
    """
    f_epochs = normalize_roi_input(roi_input, frames_per_epoch=frames_per_epoch)
    e, f, n = f_epochs.shape

    if fps is None:
        fps = f / float(epoch_seconds if epoch_seconds else 30.0)

    behavior = (
        normalize_behavior(behavior_input, frames_per_epoch=f, epochs=e)
        if behavior_input
        else None
    )

    dff = compute_dff_epochs(f_epochs)

    # Deconvolution
    if use_deconvolution and _OASIS_AVAILABLE:
        if auto_tune_deconv:
            C_hat, S_hat, tuning_table = deconvolve_epochs_with_autotune(
                dff,
                penalty=deconv_penalty,
                method=deconv_method,
                lam_factors=lam_factors,
                s_min=deconv_s_min,
                g_init=deconv_g_init,
                **oasis_kwargs,
            )
        else:
            # single-pass deconv letting OASIS choose params per epoch/ROI
            C_hat = np.zeros_like(dff, dtype=float)
            S_hat = np.zeros_like(dff, dtype=float)
            rows: List[Dict] = []
            for j in range(n):
                # estimate sn for reporting
                sn = _estimate_noise_sn(dff[:, :, j].reshape(-1))
                for i in range(e):
                    try:
                        c_hat, s_hat, _, g_out, lam_out = deconvolve_oasis_trace(
                            dff[i, :, j],
                            penalty=deconv_penalty,
                            method=deconv_method,
                            g=deconv_g_init,
                            s_min=deconv_s_min,
                            **oasis_kwargs,
                        )
                        C_hat[i, :, j] = c_hat
                        S_hat[i, :, j] = s_hat
                        rows.append({"roi": j, "sn": sn,
                                     "lam": float(lam_out) if lam_out is not None else np.nan,
                                     "g": float(g_out) if np.isscalar(g_out) else (float(g_out[0]) if g_out is not None else np.nan),
                                     "bic": np.nan, "k_spikes": float(np.count_nonzero(s_hat > 0))})
                    except Exception:
                        pass
            # summarize tuning per ROI (mean across epochs)
            tuning_table = pd.DataFrame(rows)
            if not tuning_table.empty:
                tuning_table = tuning_table.groupby("roi", as_index=False).agg(
                    {"sn": "mean", "lam": "mean", "g": "mean", "bic": "mean", "k_spikes": "mean"}
                )
            else:
                tuning_table = pd.DataFrame(columns=["roi", "sn", "lam", "g", "bic", "k_spikes"])
    else:
        if use_deconvolution and not _OASIS_AVAILABLE:
            print(
                "[WARN] use_deconvolution=True, but OASIS is not available. "
                "Install `oasis` or set use_deconvolution=False."
            )
        C_hat = np.zeros_like(dff)
        S_hat = np.zeros_like(dff)
        tuning_table = pd.DataFrame(columns=["roi", "sn", "lam", "g", "bic", "k_spikes"])

    # Events from preferred signal
    if event_refractory_frames is None:
        event_refractory_frames = max(1, int(round(0.25 * fps)))  # ~250 ms

    events = detect_events_epochs(
        dff=dff,
        use_deconv=(use_deconvolution and _OASIS_AVAILABLE),
        s_hat=S_hat if (use_deconvolution and _OASIS_AVAILABLE) else None,
        z_threshold=event_z_threshold,
        refractory_frames=event_refractory_frames,
    )

    # Metrics
    per_roi_df = compute_single_roi_metrics_epochs(f_epochs, fps=fps, events=events)
    stim_df = compute_stimulus_locked_metrics(f_epochs, stim_frame=stim_frame, fps=fps)
    popm = compute_population_metrics(f_epochs, fps=fps, events=events)
    epoch_corrs = compute_epoch_correlation_matrices(f_epochs)
    beh_df = (
        compute_behavior_alignment(f_epochs, stim_frame=stim_frame, fps=fps, behavior=behavior)
        if behavior
        else pd.DataFrame()
    )

    return {
        "per_roi_metrics": per_roi_df,
        "stim_metrics": stim_df,
        "population_metrics": {
            "mean_coactivity": popm.mean_coactivity,
            "std_coactivity": popm.std_coactivity,
            "mean_abs_corr": popm.mean_abs_corr,
            "pc1_variance_explained": popm.pc1_variance_explained,
            "mean_event_rate_hz": popm.mean_event_rate_hz,
        },
        "dff": dff,
        "events": events,
        "epoch_corrs": epoch_corrs,
        "behavior_alignment": beh_df,
        "deconv": {"C_hat": C_hat, "S_hat": S_hat, "tuning": tuning_table},
        "fps": fps,
        "frames_per_epoch": f,
        "epochs": e,
        "n_rois": n,
    }


# ======================================================================
# Fish-level summaries & cohort aggregation
# ======================================================================

def summarize_fish(results: Dict[str, object], fish_id: str, cohort: str) -> pd.DataFrame:
    """
    Collapse per-ROI into per-fish summary stats for group comparisons.
    Returns a one-row DataFrame with fish_id, cohort and key metrics.
    """
    roi = results["per_roi_metrics"]
    pop = results["population_metrics"]
    stim = results["stim_metrics"]

    row = {
        "fish_id": fish_id,
        "cohort": cohort,
        "event_rate_hz_mean": float(np.nanmean(roi["event_rate_hz"])),
        "event_amp_mean": float(np.nanmean(roi["mean_amp"])),
        "event_auc_mean": float(np.nanmean(roi["mean_auc"])),
        "t_half_median": float(np.nanmedian(roi["median_t_half_frames"])),
        "baseline_f0_mean": float(np.nanmean(roi["baseline_f0"])),
        "responsive_fraction_mean": float(np.nanmean(stim["responsive_fraction"])),
        "peak_dff_mean": float(np.nanmean(stim["peak_dff_mean"])),
        "latency_median": float(np.nanmedian(stim["latency_frames_median"])),
        "reliability_mean": float(np.nanmean(stim["trial_reliability"])),
        "pop_mean_coactivity": pop["mean_coactivity"],
        "pop_mean_abs_corr": pop["mean_abs_corr"],
        "pop_pc1_var": pop["pc1_variance_explained"],
        "pop_mean_event_rate_hz": pop["mean_event_rate_hz"],
    }
    return pd.DataFrame([row])


def aggregate_cohorts(fish_summaries: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cohort-level means, stds, counts, and standard errors for each metric.
    """
    metrics = [c for c in fish_summaries.columns if c not in ("fish_id", "cohort")]
    agg = fish_summaries.groupby("cohort")[metrics].agg(["mean", "std", "count"])
    for m in metrics:
        agg[(m, "sem")] = agg[(m, "std")] / np.sqrt(agg[(m, "count")].clip(lower=1))
    return agg
