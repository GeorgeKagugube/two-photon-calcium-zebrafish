"""
Responsive ROI analysis with sliding-baseline detrending

Assumptions:
- Input is ΔF/F
- Rows    = ROIs / neurons
- Columns = frames (0-based)
- Stimulus is applied from frame 12 to 35 (inclusive)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import math

# -----------------------
# USER PARAMETERS
# -----------------------
#DATA_PATH = df2_mut1      # <-- change to your file path

# Frame indices (0-based, inclusive ranges)
BASELINE_FRAMES = np.arange(0, 12)  # frames 0–11
STIM_FRAMES     = np.arange(12, 36) # frames 12–35

# Sliding-baseline parameters
BASELINE_WINDOW     = 31   # window length in frames (odd, > stim duration is good)
BASELINE_PERCENTILE = 20   # lower percentile used as baseline estimate

# Responsiveness thresholds (in z-score units)
MEAN_Z_THRESH = 0.5        # mean stim - mean baseline
PEAK_Z_THRESH = 2.0        # max z during stim window

# Clustering / plotting controls
MAX_ROIS_FOR_RAW_HEATMAP = 150   # max # ROIs in ROI-level heatmap
N_CLUSTERS               = 20    # number of functional clusters to summarise
SNAPSHOT_ROI_INDEX       = 0     # which ROI to visualise for baseline snapshot

# Output file
OUTPUT_CSV = "./gcamp_activity/responsive_rois_clusters.csv"


# -----------------------
# HELPER FUNCTIONS
# -----------------------
def sliding_baseline(trace, window=31, percentile=20):
    """
    Estimate a slow-varying baseline for a 1D trace using a running percentile.

    Parameters
    ----------
    trace : 1D array (ΔF/F or similar)
    window : int
        Window size in frames (must be odd).
    percentile : float
        Lower percentile used as baseline (e.g. 20).

    Returns
    -------
    baseline : 1D array, same shape as trace
    """
    trace = np.asarray(trace, float)
    n = trace.shape[0]
    half = window // 2

    padded = np.pad(trace, pad_width=half, mode="edge")
    baseline = np.empty(n, dtype=float)

    for i in range(n):
        segment = padded[i : i + window]
        baseline[i] = np.nanpercentile(segment, percentile)

    return baseline


def detrend_matrix_sliding_baseline(X, window=31, percentile=20):
    """
    Apply sliding baseline correction to each row of X.

    Parameters
    ----------
    X : 2D array (n_rois, n_frames)

    Returns
    -------
    X_detrended : 2D array, same shape as X
    """
    X = np.asarray(X, float)
    n_rois, n_frames = X.shape
    X_detrended = np.empty_like(X)

    for i in range(n_rois):
        bl = sliding_baseline(X[i, :], window=window, percentile=percentile)
        X_detrended[i, :] = X[i, :] - bl

    return X_detrended


def mean_offdiag(mat):
    """Mean of off-diagonal elements of a square matrix."""
    m = mat.shape[0]
    mask = ~np.eye(m, dtype=bool)
    return np.nanmean(mat[mask])


# -----------------------
# 1. LOAD ΔF/F MATRIX
# -----------------------
df = df2_wtExp.transpose() #pd.read_csv(DATA_PATH, index_col=0)  # rows = ROIs, cols = frames
print("Raw data shape (n_rois, n_frames):", df.shape)

X_raw = df.to_numpy(dtype=float)   # (n_rois, n_frames)
n_rois, n_frames = X_raw.shape

# Sanity check for window ranges
if BASELINE_FRAMES.max() >= n_frames or STIM_FRAMES.max() >= n_frames:
    raise ValueError(
        f"Frame ranges exceed data length: n_frames={n_frames}, "
        f"baseline max={BASELINE_FRAMES.max()}, stim max={STIM_FRAMES.max()}"
    )

# -----------------------
# 2. DETREND WITH SLIDING BASELINE
# -----------------------
X = detrend_matrix_sliding_baseline(
    X_raw,
    window=BASELINE_WINDOW,
    percentile=BASELINE_PERCENTILE
)

print("Detrended data shape:", X.shape)

# -----------------------
# 2B. BASELINE SNAPSHOT FOR ONE ROI
# -----------------------
roi_idx = SNAPSHOT_ROI_INDEX
trace_raw = X_raw[roi_idx, :]
baseline_est = sliding_baseline(
    trace_raw,
    window=BASELINE_WINDOW,
    percentile=BASELINE_PERCENTILE
)
trace_detrended = trace_raw - baseline_est

frames = np.arange(n_frames)

plt.figure(figsize=(10, 4))
plt.plot(frames, trace_raw, label="Raw ΔF/F", alpha=0.7)
plt.plot(frames, baseline_est, label="Estimated baseline", linewidth=2)
plt.plot(frames, trace_detrended, label="Detrended (ΔF/F - baseline)", alpha=0.8)

# shade stim window
plt.axvspan(STIM_FRAMES[0], STIM_FRAMES[-1], alpha=0.1, color="red", label="Stimulus")

plt.xlabel("Frame")
plt.ylabel("Signal (ΔF/F)")
plt.title(f"Baseline snapshot for ROI {roi_idx}")
plt.legend()
plt.tight_layout()
plt.show()

# From here on, we work with X (detrended traces)


# -----------------------
# 3. Z-SCORE PER ROI (ROW-WISE)
# -----------------------
mean = np.nanmean(X, axis=1, keepdims=True)
std  = np.nanstd(X, axis=1, ddof=1, keepdims=True)

valid_var = (std.squeeze() > 0)  # ROIs with non-zero variance
print("ROIs with non-zero variance:", valid_var.sum(), "/", n_rois)

Z = np.full_like(X, np.nan, dtype=float)
Z[valid_var] = (X[valid_var] - mean[valid_var]) / std[valid_var]


# -----------------------
# 4. DEFINE RESPONSIVE ROIs
# -----------------------
baseline = Z[:, BASELINE_FRAMES]   # (n_rois, n_baseline_frames)
stimulus = Z[:, STIM_FRAMES]       # (n_rois, n_stim_frames)

baseline_mean = np.nanmean(baseline, axis=1)  # (n_rois,)
stim_mean     = np.nanmean(stimulus, axis=1)  # (n_rois,)
delta_mean    = stim_mean - baseline_mean     # (n_rois,)

peak_stim = np.nanmax(stimulus, axis=1)       # (n_rois,)

responsive_mask = (
    (delta_mean >= MEAN_Z_THRESH) &
    (peak_stim >= PEAK_Z_THRESH) &
    valid_var
)

responsive_indices = np.where(responsive_mask)[0]
responsive_roi_ids = df.index[responsive_mask]

print("Number of responsive ROIs:", responsive_mask.sum())
print("Responsive ROI IDs (first 20):", list(responsive_roi_ids[:20]))

if responsive_mask.sum() == 0:
    raise RuntimeError("No responsive ROIs detected with current thresholds.")

# -----------------------
# 5. CORRELATION MATRIX FOR RESPONSIVE ROIs
# -----------------------
Z_resp = Z[responsive_mask, :]       # (n_resp_rois, n_frames)
n_resp = Z_resp.shape[0]

corr = np.corrcoef(Z_resp, rowvar=True)   # (n_resp_rois, n_resp_rois)

corr_df = pd.DataFrame(
    corr,
    index=responsive_roi_ids,
    columns=responsive_roi_ids
)

print("Correlation matrix shape (responsive only):", corr_df.shape)

# -----------------------
# 6. ROI-LEVEL CLUSTERED HEATMAP (SUBSAMPLED IF MANY ROIs)
# -----------------------
sns.set(style="white")

if n_resp <= MAX_ROIS_FOR_RAW_HEATMAP:
    corr_df_small = corr_df.copy()
    print(f"Using all {n_resp} responsive ROIs in ROI-level heatmap.")
else:
    # Rank responsive ROIs by strength of response (delta_mean)
    resp_delta = delta_mean[responsive_mask]
    order = np.argsort(resp_delta)[::-1]  # strongest first
    keep = order[:MAX_ROIS_FOR_RAW_HEATMAP]
    corr_df_small = corr_df.iloc[keep, keep]
    print(f"Using top {MAX_ROIS_FOR_RAW_HEATMAP} of {n_resp} responsive ROIs for ROI-level heatmap.")

g_roi = sns.clustermap(
    corr_df_small,
    cmap="vlag",
    vmin=-1, vmax=1,
    center=0,
    figsize=(8, 8),
    metric="euclidean",
    method="average",
    dendrogram_ratio=(0.1, 0.1),
    cbar_kws={"label": "Pearson correlation (z-scored, detrended ΔF/F)"}
)

g_roi.ax_heatmap.set_title(
    "Clustered correlation of top responsive ROIs\n"
    f"(stim frames {STIM_FRAMES[0]}–{STIM_FRAMES[-1]})"
)

if corr_df_small.shape[0] > 40:
    g_roi.ax_heatmap.set_xticklabels([])
    g_roi.ax_heatmap.set_yticklabels([])

plt.show()

# -----------------------
# 7. CLUSTER-LEVEL SUMMARY + CLUSTER HEATMAP
# -----------------------
print(f"Building cluster-level summary with up to {N_CLUSTERS} clusters...")

# 1 - correlation → distance, then condensed form
dist = 1 - corr
np.fill_diagonal(dist, 0.0)
dist_condensed = squareform(dist, checks=False)

link = linkage(dist_condensed, method="average")
cluster_labels = fcluster(link, N_CLUSTERS, criterion="maxclust")

unique_clusters = np.unique(cluster_labels)
actual_n_clusters = len(unique_clusters)
print(f"Found {actual_n_clusters} clusters among responsive ROIs.")

# Compute mean z-scored trace per cluster
cluster_traces = []
cluster_names  = []
for cid in unique_clusters:
    idx = cluster_labels == cid
    cluster_traces.append(np.nanmean(Z_resp[idx, :], axis=0))
    cluster_names.append(f"Cluster {cid} (n={idx.sum()})")

cluster_traces = np.vstack(cluster_traces)  # (n_clusters, n_frames)

# Correlation between cluster-average traces
cluster_corr = np.corrcoef(cluster_traces, rowvar=True)
cluster_corr_df = pd.DataFrame(
    cluster_corr,
    index=cluster_names,
    columns=cluster_names
)

g_cluster = sns.clustermap(
    cluster_corr_df,
    cmap="vlag",
    vmin=-1, vmax=1,
    center=0,
    figsize=(8, 8),
    metric="euclidean",
    method="average",
    dendrogram_ratio=(0.2, 0.2),
    cbar_kws={"label": "Correlation between cluster mean traces"}
)

g_cluster.ax_heatmap.set_title(
    "Functional clusters of stimulus-responsive ROIs\n"
    f"(cluster-level correlation, stim {STIM_FRAMES[0]}–{STIM_FRAMES[-1]})"
)

g_cluster.ax_heatmap.set_xticklabels(
    g_cluster.ax_heatmap.get_xticklabels(),
    rotation=90
)
g_cluster.ax_heatmap.set_yticklabels(
    g_cluster.ax_heatmap.get_yticklabels()
)

plt.show()

# -----------------------
# 8. EXPORT RESPONSIVE ROIs + CLUSTER LABELS TO CSV
# -----------------------
# Per-ROI response measures
n_total = n_rois
n_resp  = responsive_mask.sum()
prop_resp = n_resp / n_total

mean_delta_resp = np.nanmean(delta_mean[responsive_mask])
mean_peak_resp  = np.nanmean(peak_stim[responsive_mask])

pop_mean_baseline = np.nanmean(Z_resp[:, BASELINE_FRAMES])
pop_mean_stim     = np.nanmean(Z_resp[:, STIM_FRAMES])
pop_delta         = pop_mean_stim - pop_mean_baseline

# Stimulus-locked synchrony (ROI level: baseline vs stim)
Zb = Z_resp[:, BASELINE_FRAMES]
Zs = Z_resp[:, STIM_FRAMES]

corr_b = np.corrcoef(Zb, rowvar=True)
corr_s = np.corrcoef(Zs, rowvar=True)

mean_corr_baseline = mean_offdiag(corr_b)
mean_corr_stim     = mean_offdiag(corr_s)
delta_corr         = mean_corr_stim - mean_corr_baseline

export_df = pd.DataFrame(
    {
        "roi_id": responsive_roi_ids,
        "roi_index": responsive_indices,
        "cluster": cluster_labels,
        "delta_mean_z": delta_mean[responsive_mask],
        "peak_z_stim": peak_stim[responsive_mask],
    }
)

# Add per-fish summary metrics as constant columns (handy when concatenating across fish)
export_df["n_total"]            = n_total
export_df["n_resp"]             = n_resp
export_df["prop_resp"]          = prop_resp
export_df["mean_delta_resp"]    = mean_delta_resp
export_df["mean_peak_resp"]     = mean_peak_resp
export_df["pop_delta"]          = pop_delta
export_df["mean_corr_baseline"] = mean_corr_baseline
export_df["mean_corr_stim"]     = mean_corr_stim
export_df["delta_corr"]         = delta_corr

export_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved responsive ROIs + cluster labels and summary metrics to: {OUTPUT_CSV}")

# -----------------------
# 9. CLUSTER PHENOTYPES: MEAN ± SEM TRACE PER CLUSTER
# -----------------------
n_resp, n_frames = Z_resp.shape
frame_axis = np.arange(n_frames)

print(f"Plotting average traces for {actual_n_clusters} clusters.")

n_cols = math.ceil(math.sqrt(actual_n_clusters))
n_rows = math.ceil(actual_n_clusters / n_cols)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(3.5 * n_cols, 2.5 * n_rows),
    sharex=True, sharey=True
)
axes = np.asarray(axes).reshape(-1)

for ax, cid in zip(axes, unique_clusters):
    idx = cluster_labels == cid
    traces = Z_resp[idx, :]  # (n_cells_in_cluster, n_frames)

    mean_trace = np.nanmean(traces, axis=0)
    sem_trace  = np.nanstd(traces, axis=0, ddof=1) / np.sqrt(idx.sum())

    ax.plot(frame_axis, mean_trace, linewidth=1.5)
    ax.fill_between(
        frame_axis,
        mean_trace - sem_trace,
        mean_trace + sem_trace,
        alpha=0.3
    )

    ax.axvspan(BASELINE_FRAMES[0], BASELINE_FRAMES[-1],
               alpha=0.1, label="baseline" if cid == unique_clusters[0] else None)
    ax.axvspan(STIM_FRAMES[0], STIM_FRAMES[-1],
               alpha=0.2, label="stimulus" if cid == unique_clusters[0] else None)

    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_title(f"Cluster {cid}\n(n={idx.sum()})", fontsize=9)

# Turn off unused axes if any
for ax in axes[len(unique_clusters):]:
    ax.axis("off")

fig.suptitle(
    "Cluster phenotypes: mean z-scored, detrended ΔF/F per cluster\n"
    f"Baseline frames {BASELINE_FRAMES[0]}–{BASELINE_FRAMES[-1]}, "
    f"stim frames {STIM_FRAMES[0]}–{STIM_FRAMES[-1]}",
    fontsize=12
)

fig.text(0.5, 0.04, "Frame", ha="center")
fig.text(0.04, 0.5, "z-scored ΔF/F (detrended)", va="center", rotation="vertical")

handles, labels = axes[0].get_legend_handles_labels()
if handles:
    fig.legend(handles, labels, loc="upper right")

plt.tight_layout(rect=[0.03, 0.05, 0.97, 0.9])
plt.show()
