import numpy as np
import matplotlib.pyplot as plt


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """
    Simple 1D moving average filter (denoising).
    """
    if window <= 1:
        return x
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def _moving_percentile(x: np.ndarray, window: int, percentile: float) -> np.ndarray:
    """
    1D adaptive baseline using a sliding percentile.

    For each time point i, baseline[i] = percentile(x[start:end])
    over a window centered on i (clipped at the edges).
    """
    n = x.shape[0]
    if window <= 1:
        # Just return a constant baseline = given percentile of whole trace
        b = np.percentile(x, percentile)
        return np.full_like(x, b, dtype=float)

    half = window // 2
    baseline = np.empty_like(x, dtype=float)

    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        baseline[i] = np.percentile(x[start:end], percentile)

    return baseline


def extract_F_from_mat_struct(
    gmrxanat,
    vprofiles_key: str = "Vprofiles",
    profile_field: str = "meanprofile",
    # ΔF/F + denoising parameters
    compute_dff: bool = True,
    smooth_window: int = 3,          # frames for moving-average denoise
    baseline_window: int = 30,       # frames for adaptive baseline window
    baseline_percentile: float = 20, # e.g. 20th percentile baseline
    eps: float = 1e-6,               # avoid division by zero
) -> np.ndarray:
    """
    Extracts a 3D array F_raw (n_stimuli, n_neurons, n_timepoints) from a
    MATLAB-like structure and optionally converts it to ΔF/F with
    an adaptive baseline and temporal denoising.

        gmrxanat['roi'][i][vprofiles_key][profile_field]

    Parameters
    ----------
    gmrxanat : dict-like
        Object containing 'roi', where gmrxanat['roi'] is a list of ROIs.
    vprofiles_key : str
        Key under each ROI that stores visual profiles (default 'Vprofiles').
    profile_field : str
        Field inside Vprofiles that stores the stimulus-averaged traces
        (default 'meanprofile').
    compute_dff : bool
        If True, compute ΔF/F with adaptive baseline for each trace.
        If False, return raw traces (after extraction) as-is.
    smooth_window : int
        Window (in frames) for moving-average denoising. 1 = no smoothing.
    baseline_window : int
        Window (in frames) for the adaptive baseline percentile filter.
        Should be large enough to track slow drift but not transients.
    baseline_percentile : float
        Percentile (0–100) used for the baseline within each window.
        Typical values: 10–30.
    eps : float
        Small constant to avoid division by zero in ΔF/F.

    Returns
    -------
    F_out : np.ndarray
        3D array (n_stimuli, n_neurons, n_timepoints).
        If compute_dff=True, this is ΔF/F; otherwise raw meanprofile values.
    """
    rois = gmrxanat["roi"]
    n_neurons = len(rois)
    if n_neurons == 0:
        raise ValueError("gmrxanat['roi'] is empty")

    # Inspect the first ROI to determine shape & orientation
    example = np.array(rois[0][vprofiles_key][profile_field])
    if example.ndim != 2:
        raise ValueError(f"{profile_field} must be 2D, got shape {example.shape}")

    d0, d1 = example.shape

    # Heuristic: assume smaller dimension is n_stimuli
    # (e.g. 10 stimuli × ~100 timepoints)
    if d0 <= d1:
        n_stimuli, n_timepoints = d0, d1
        stim_first = True
    else:
        n_timepoints, n_stimuli = d0, d1
        stim_first = False

    # Extract raw traces into F_raw
    F_raw = np.zeros((n_stimuli, n_neurons, n_timepoints), dtype=float)

    for n, roi in enumerate(rois):
        mp = np.array(roi[vprofiles_key][profile_field])
        if stim_first:
            if mp.shape != (n_stimuli, n_timepoints):
                raise ValueError(
                    f"ROI {n}: expected shape {(n_stimuli, n_timepoints)}, got {mp.shape}"
                )
            F_raw[:, n, :] = mp
        else:
            if mp.shape != (n_timepoints, n_stimuli):
                raise ValueError(
                    f"ROI {n}: expected shape {(n_timepoints, n_stimuli)}, got {mp.shape}"
                )
            F_raw[:, n, :] = mp.T  # transpose to (n_stimuli, n_timepoints)

    if not compute_dff:
        # Return raw F if ΔF/F not requested
        return F_raw

    # Compute ΔF/F with adaptive baseline and denoising
    F_dff = np.zeros_like(F_raw, dtype=float)

    for s in range(n_stimuli):
        for n in range(n_neurons):
            trace = F_raw[s, n, :].astype(float)

            # 1) Denoise: moving-average smoothing
            trace_smooth = _moving_average(trace, smooth_window)

            # 2) Adaptive baseline using sliding percentile
            baseline = _moving_percentile(trace_smooth, baseline_window, baseline_percentile)

            # 3) Avoid division by zero
            baseline[baseline < eps] = eps

            # 4) ΔF/F
            F_dff[s, n, :] = (trace_smooth - baseline) / baseline

    return F_dff


def debug_plot_dff_examples(
    gmrxanat,
    examples,
    vprofiles_key: str = "Vprofiles",
    profile_field: str = "meanprofile",
    smooth_window: int = 3,
    baseline_window: int = 30,
    baseline_percentile: float = 20,
    eps: float = 1e-6,
    time_axis: np.ndarray = None,
    figsize=(10, 4),
):
    """
    Plot raw, smoothed, baseline, and ΔF/F for multiple (stim, neuron) examples.

    Parameters
    ----------
    gmrxanat : dict-like
        Structure containing 'roi' as in extract_F_from_mat_struct.
    examples : list of (stim_idx, neuron_idx)
        List of (s, n) index pairs to plot.
        s: stimulus index (0..n_stimuli-1)
        n: neuron/ROI index (0..n_neurons-1)
    vprofiles_key : str
        Key under each ROI that stores visual profiles.
    profile_field : str
        Field inside Vprofiles that stores the stimulus-averaged traces.
    smooth_window : int
        Moving-average window (frames) for denoising.
    baseline_window : int
        Window (frames) for adaptive baseline.
    baseline_percentile : float
        Percentile for baseline estimation.
    eps : float
        Small constant for ΔF/F division.
    time_axis : np.ndarray, optional
        1D array of time values (length = n_timepoints).
        If None, frame indices [0..n_timepoints-1] will be used.
    figsize : tuple
        Figure size for each example.

    Returns
    -------
    figs : list of matplotlib.figure.Figure
        Figures for each (stim, neuron) example.
    """
    rois = gmrxanat["roi"]
    n_neurons = len(rois)
    if n_neurons == 0:
        raise ValueError("gmrxanat['roi'] is empty")

    # Determine orientation from first ROI
    example = np.array(rois[0][vprofiles_key][profile_field])
    if example.ndim != 2:
        raise ValueError(f"{profile_field} must be 2D, got shape {example.shape}")

    d0, d1 = example.shape
    if d0 <= d1:
        n_stimuli, n_timepoints = d0, d1
        stim_first = True
    else:
        n_timepoints, n_stimuli = d0, d1
        stim_first = False

    # Time axis
    if time_axis is None:
        t = np.arange(n_timepoints)
    else:
        t = np.asarray(time_axis)
        if t.shape[0] != n_timepoints:
            raise ValueError(
                f"time_axis length {t.shape[0]} does not match n_timepoints {n_timepoints}"
            )

    figs = []

    for (s_idx, n_idx) in examples:
        if s_idx < 0 or s_idx >= n_stimuli:
            raise IndexError(f"stim_idx {s_idx} out of range [0, {n_stimuli-1}]")
        if n_idx < 0 or n_idx >= n_neurons:
            raise IndexError(f"neuron_idx {n_idx} out of range [0, {n_neurons-1}]")

        mp = np.array(rois[n_idx][vprofiles_key][profile_field])
        if stim_first:
            trace = mp[s_idx, :].astype(float)
        else:
            trace = mp[:, s_idx].astype(float)

        # 1) smoothing
        trace_smooth = _moving_average(trace, smooth_window)

        # 2) baseline
        baseline = _moving_percentile(trace_smooth, baseline_window, baseline_percentile)
        baseline[baseline < eps] = eps

        # 3) dff
        dff = (trace_smooth - baseline) / baseline

        # Plot
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        ax0, ax1 = axes

        ax0.plot(t, trace, label="raw", alpha=0.5)
        ax0.plot(t, trace_smooth, label="smoothed", linewidth=1.5)
        ax0.plot(t, baseline, label="baseline", linestyle="--")
        ax0.set_ylabel("Fluorescence (a.u.)")
        ax0.set_title(f"Stim {s_idx}, Neuron {n_idx}")
        ax0.legend(loc="best")

        ax1.plot(t, dff, label="ΔF/F")
        ax1.axhline(0, color="k", linewidth=0.5)
        ax1.set_ylabel("ΔF/F")
        ax1.set_xlabel("Time (frames)" if time_axis is None else "Time")
        ax1.legend(loc="best")

        plt.tight_layout()
        figs.append(fig)

    return figs
