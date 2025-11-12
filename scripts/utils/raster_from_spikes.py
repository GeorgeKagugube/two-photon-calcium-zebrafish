import numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional

# def raster_from_spikes(S_df, fs=3.6, stim_frame=12, windows={"pre":(0,11),"during":(12,35),"post":(36,108)},
#                        thr=0.0, sort_by="latency_during", title="Fish raster"):
#     S = S_df.to_numpy(float)
#     T = S.shape[1]
#     t = np.arange(T)/fs
#     # binary events
#     events = (S > thr)

#     # get indices for "during" to sort by latency
#     w_dur = np.arange(*windows["during"]) if isinstance(windows["during"], tuple) else windows["during"]
#     if isinstance(windows["during"], tuple):
#         w_dur = np.arange(windows["during"][0], windows["during"][1]+1)
#     # latency per ROI
#     lat = np.full(S.shape[0], np.inf)
#     has = events[:, w_dur].any(axis=1)
#     first_idx = np.argmax(events[:, w_dur], axis=1)  # first True index or 0
#     lat[has] = (w_dur[first_idx[has]] - stim_frame) / fs
#     if sort_by == "latency_during":
#         order = np.argsort(lat)
#     else:
#         order = np.arange(S.shape[0])

#     ev_sorted = events[order, :]

#     # Raster plot
#     fig = plt.figure(figsize=(10,6))
#     ax1 = plt.axes([0.1,0.35,0.85,0.6])
#     rows, cols = np.where(ev_sorted)
#     ax1.scatter(cols/fs, rows, s=2, marker='|')
#     # Window boundaries
#     for name,(a,b) in windows.items():
#         ax1.axvline(a/fs, ls='--', lw=0.8)
#         ax1.axvline((b+1)/fs, ls='--', lw=0.8)
#     ax1.axvline(stim_frame/fs, color='k', lw=1.2)
#     ax1.set_ylabel("ROIs (sorted)")
#     ax1.set_xlim(t[0], t[-1])
#     ax1.set_title(title)

#     # Population rate (top)
#     ax0 = plt.axes([0.1,0.12,0.85,0.18], sharex=ax1)
#     bin_w = max(1, int(round(fs*0.25)))  # 250 ms bins
#     ev_sum = ev_sorted.sum(axis=0)
#     # simple binning
#     pad = (-ev_sum.size) % bin_w
#     x_pad = np.pad(ev_sum, (0,pad))
#     rate = x_pad.reshape(-1, bin_w).sum(axis=1) / (bin_w/fs) / S.shape[0]  # spikes/s/neuron
#     t_bin = (np.arange(rate.size)*bin_w + bin_w/2)/fs
#     ax0.plot(t_bin, rate)
#     ax0.set_ylabel("Rate\n(sp/s/neuron)")
#     ax0.set_xlabel("Time (s)")

#     plt.show()
#     # return ordering & latency for record
#     return {"order_idx": order, "latency_during_s": lat[order]}

def raster_with_optional_dff(
    S_df: pd.DataFrame,
    fs: float = 3.6,
    stim_frame: int = 12,
    windows: Dict[str, Tuple[int, int]] = None,   # inclusive frame indices
    thr: float = 0.0,                              # threshold on deconvolved spikes
    sort_by: str = "latency_during",               # or "none"
    dff_df: Optional[pd.DataFrame] = None,         # rows=ROIs (same order as S_df), cols=frames
    heatmap_mode: str = "zscore",                  # {"zscore","minmax","raw"}
    heatmap_clip_percentile: float = 99.0,         # clip high tails for display
    responsive_only: bool = False,                 # drop non-responsive ROIs in the "during" window
    title: str = "Raster (spikes) ± ΔF/F heatmap",
    figsize: Tuple[float, float] = (10, 7),
    raster_marker_size: float = 2.0,
):
    """
    Plot a spike raster (binary from deconvolved S) and optional ΔF/F heatmap aligned in time.

    Parameters
    ----------
    S_df : DataFrame
        rows=ROIs, cols=frames (time). Values are deconvolved spikes (non-negative).
    fs : float
        Sampling rate (Hz).
    stim_frame : int
        Frame index of stimulus onset line.
    windows : dict
        {"pre": (0,11), "during": (12,35), "post": (36,108)} — inclusive indices.
    thr : float
        Threshold on spikes; >thr is treated as an event.
    sort_by : str
        "latency_during" to sort by first event latency in the 'during' window; "none" to keep input order.
    dff_df : DataFrame or None
        Optional ΔF/F matrix, same shape/alignment as S_df.
    heatmap_mode : str
        How to scale ΔF/F per ROI for display: "zscore", "minmax", or "raw".
    heatmap_clip_percentile : float
        Clip ΔF/F display at this percentile to improve contrast.
    responsive_only : bool
        If True, filter to ROIs that have ≥1 event in the 'during' window.
    title : str
        Figure title.
    figsize : tuple
        Figure size in inches.
    raster_marker_size : float
        Marker size for raster scatter.
    """
    if windows is None:
        windows = {"pre": (0, 11), "during": (12, 35), "post": (36, 108)}
    assert "during" in windows, "windows must contain a 'during' entry."

    S = S_df.to_numpy(float)
    T = S.shape[1]
    t = np.arange(T) / fs
    events = (S > thr)

    # Build index arrays for the windows (inclusive)
    def _idx_range(pair):
        a, b = int(pair[0]), int(pair[1])
        a = max(0, a); b = min(T-1, b)
        return np.arange(a, b+1, dtype=int) if b >= a else np.array([], dtype=int)

    w_pre = _idx_range(windows["pre"])
    w_dur = _idx_range(windows["during"])
    w_post = _idx_range(windows["post"])

    # Determine latency within the 'during' window
    latency = np.full(S.shape[0], np.inf, dtype=float)
    has_event_dur = events[:, w_dur].any(axis=1)
    if has_event_dur.any():
        # first True index per row among during-window columns
        first_idx_local = np.argmax(events[:, w_dur], axis=1)  # returns 0 if none, but we'll mask by has_event_dur
        first_frame = np.where(has_event_dur, w_dur[first_idx_local], np.nan)
        latency[has_event_dur] = (first_frame[has_event_dur] - stim_frame) / fs

    # Filter to responsive-only (during) if requested
    keep_mask = np.ones(S.shape[0], dtype=bool)
    if responsive_only:
        keep_mask = has_event_dur

    # Sorting
    if sort_by == "latency_during":
        order = np.argsort(latency)  # inf (non-responders) go to bottom
    else:
        order = np.arange(S.shape[0])

    order = order[keep_mask[order]]
    events_sorted = events[order, :]
    latency_sorted = latency[order]

    # Population rate (simple binned rate per neuron)
    bin_w = max(1, int(round(fs * 0.25)))  # ~250 ms bins
    ev_sum = events_sorted.sum(axis=0)
    pad = (-ev_sum.size) % bin_w
    ev_pad = np.pad(ev_sum, (0, pad))
    rate = ev_pad.reshape(-1, bin_w).sum(axis=1) / (bin_w / fs) / max(1, events_sorted.shape[0])
    t_bin = (np.arange(rate.size) * bin_w + bin_w / 2) / fs

    # Prepare figure layout: top PSTH, middle raster, bottom heatmap (optional)
    if dff_df is None:
        fig = plt.figure(figsize=figsize)
        # PSTH
        ax0 = plt.axes([0.1, 0.1, 0.85, 0.18])
        # Raster
        ax1 = plt.axes([0.1, 0.35, 0.85, 0.6], sharex=ax0)
    else:
        fig = plt.figure(figsize=(figsize[0], figsize[1] + 2))
        # PSTH
        ax0 = plt.axes([0.1, 0.12, 0.85, 0.15])
        # Raster
        ax1 = plt.axes([0.1, 0.35, 0.85, 0.5])
        # Heatmap
        ax2 = plt.axes([0.1, 0.88, 0.85, 0.08])  # we’ll adjust positions below for a cleaner stack
        # adjust stack: top=heatmap, mid=raster, low=PSTH
        ax2.set_position([0.1, 0.70, 0.85, 0.20])
        ax1.set_position([0.1, 0.38, 0.85, 0.28])
        ax0.set_position([0.1, 0.12, 0.85, 0.18])

    # ---- Raster ----
    rows, cols = np.where(events_sorted)
    ax1.scatter(cols / fs, rows, s=raster_marker_size, marker='|')
    # window boundaries + stim
    for name, pair in windows.items():
        a, b = pair
        ax1.axvline(a / fs, ls='--', lw=0.8)
        ax1.axvline((b + 1) / fs, ls='--', lw=0.8)
    ax1.axvline(stim_frame / fs, lw=1.2, color='k')
    ax1.set_ylabel("ROIs (sorted)")
    ax1.set_xlim(t[0], t[-1])
    ax1.set_title(title)

    # ---- PSTH (population rate) ----
    ax0.plot(t_bin, rate)
    ax0.set_ylabel("Rate\n(sp/s/neuron)")
    ax0.set_xlabel("Time (s)")
    # align vertical lines
    for name, pair in windows.items():
        a, b = pair
        ax0.axvline(a / fs, ls='--', lw=0.8)
        ax0.axvline((b + 1) / fs, ls='--', lw=0.8)
    ax0.axvline(stim_frame / fs, lw=1.2, color='k')

    # ---- Optional ΔF/F heatmap ----
    if dff_df is not None:
        assert dff_df.shape == S_df.shape, "dff_df must match S_df shape."
        D = dff_df.to_numpy(float)[order, :]

        # Per-ROI scaling for display
        if heatmap_mode.lower() == "zscore":
            mu = np.nanmean(D, axis=1, keepdims=True)
            sd = np.nanstd(D, axis=1, keepdims=True) + 1e-12
            H = (D - mu) / sd
        elif heatmap_mode.lower() == "minmax":
            mn = np.nanmin(D, axis=1, keepdims=True)
            mx = np.nanmax(D, axis=1, keepdims=True)
            H = (D - mn) / (mx - mn + 1e-12)
        elif heatmap_mode.lower() == "raw":
            H = D.copy()
        else:
            raise ValueError("heatmap_mode must be 'zscore', 'minmax', or 'raw'.")

        # Clip high tails to stabilise contrast
        if np.isfinite(H).any():
            vmax = np.nanpercentile(H, heatmap_clip_percentile)
            vmin = np.nanpercentile(H, 100 - heatmap_clip_percentile)
            H = np.clip(H, vmin, vmax)

        im = ax2.imshow(H, aspect='auto', interpolation='nearest',
                        extent=[t[0], t[-1], 0, H.shape[0]], origin='lower')
        # Lines
        for name, pair in windows.items():
            a, b = pair
            ax2.axvline(a / fs, ls='--', lw=0.8)
            ax2.axvline((b + 1) / fs, ls='--', lw=0.8)
        ax2.axvline(stim_frame / fs, lw=1.2, color='k')
        ax2.set_yticks([]); ax2.set_ylabel("ΔF/F (scaled)")
        cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.01)
        cbar.ax.set_ylabel(heatmap_mode, rotation=270, labelpad=12)

    plt.show()

    return {
        "order_idx": order,                    # indices into original S_df rows
        "latency_during_s": latency_sorted,    # per-ROI latency (inf for non-responders)
        "responsive_mask_during": has_event_dur[order],
    }

