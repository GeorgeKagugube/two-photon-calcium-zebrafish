
from typing import Union, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def _ensure_df(x: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    elif isinstance(x, np.ndarray):
        df = pd.DataFrame(x)
        df.columns = [f"ROI_{i+1}" for i in range(df.shape[1])]
        return df
    else:
        raise TypeError("Input must be a pandas DataFrame or numpy ndarray.")

def _window_slice(n_frames: int, stim_frame: int, during_start: int, during_end: int, post_start: Optional[int]):
    pre = (0, stim_frame)
    during = (during_start, during_end)
    if post_start is None:
        post_start = during_end + 1
    post = (post_start, n_frames - 1)
    return {"pre": pre, "during": during, "post": post}

def _nan_corr(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr()

def _pairwise_r_long(corr: pd.DataFrame) -> pd.DataFrame:
    c = corr.copy()
    np.fill_diagonal(c.values, np.nan)
    df_long = c.stack(dropna=True).reset_index()
    df_long.columns = ["ROI_i","ROI_j","r"]
    return df_long

def _rolling_mean(a: np.ndarray, win: int) -> np.ndarray:
    if win <= 1:
        return a.copy()
    s = pd.Series(a)
    return s.rolling(window=win, center=True, min_periods=1).mean().values

def _svd_pca(window_df: pd.DataFrame):
    X = window_df.values.astype(float)
    col_med = np.nanmedian(X, axis=0)
    inds = np.where(~np.isfinite(X))
    if inds[0].size > 0:
        X[inds] = np.take(col_med, inds[1])
    X = X - np.mean(X, axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    var = (s**2) / (X.shape[0] - 1)
    vr = var / np.sum(var) if np.sum(var) > 0 else np.zeros_like(var)
    cum = np.cumsum(vr)
    pr = 1.0 / np.sum(vr**2) if np.sum(vr**2) > 0 else np.nan
    n80 = int(np.searchsorted(cum, 0.8) + 1) if cum.size > 0 else 0
    return {"vr": vr, "cum": cum, "pr": np.array([pr]), "n80": np.array([n80]), "Vt": Vt, "scores": U * s}

def _set_pubstyle():
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 11, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "figure.autolayout": True})

def _plot_corr_heatmap(corr: pd.DataFrame, title: str, outpath: str):
    _set_pubstyle()
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(corr.values, aspect='auto', interpolation='nearest')
    ax.set_title(title); ax.set_xlabel("ROI"); ax.set_ylabel("ROI")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)

def _plot_pairwise_r_hist(r_long: pd.DataFrame, title: str, outpath: str):
    _set_pubstyle()
    fig, ax = plt.subplots(figsize=(5,4))
    ax.hist(r_long["r"].values, bins=30)
    ax.set_title(title); ax.set_xlabel("pairwise r"); ax.set_ylabel("count")
    fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)

def _plot_pca_scree(vr: np.ndarray, title: str, outpath: str):
    _set_pubstyle()
    fig, ax = plt.subplots(figsize=(5,4))
    x = np.arange(1, vr.size + 1)
    ax.plot(x, vr, marker="o", linewidth=1.5)
    ax.set_title(title); ax.set_xlabel("PC"); ax.set_ylabel("variance explained")
    ax2 = ax.twinx(); ax2.plot(x, np.cumsum(vr), linestyle="--")
    ax2.set_ylabel("cumulative variance")
    fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)

def _plot_pc1pc2_loadings(Vt: np.ndarray, roi_names: list, title: str, outpath: str):
    _set_pubstyle()
    fig, ax = plt.subplots(figsize=(5,4))
    if Vt.shape[0] >= 2:
        ax.scatter(Vt[0, :], Vt[1, :], s=16)
        ax.set_xlabel("PC1 loading"); ax.set_ylabel("PC2 loading")
    ax.set_title(title)
    fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)

def _knn_weights(coords: pd.DataFrame, k: int = 6) -> np.ndarray:
    xy = coords[["x","y"]].values.astype(float)
    N = xy.shape[0]; W = np.zeros((N, N), dtype=float)
    for i in range(N):
        d = np.sqrt(np.sum((xy - xy[i])**2, axis=1))
        idx = np.argsort(d)[1:k+1]
        W[i, idx] = 1.0
    W = np.maximum(W, W.T)
    return W

def _moran_I(x: np.ndarray, W: np.ndarray) -> float:
    x = x.astype(float); x_mean = np.mean(x); x0 = x - x_mean
    S0 = np.sum(W); denom = np.sum(x0**2)
    if S0 <= 0 or denom <= 0: return np.nan
    num = 0.0; N = x.size
    for i in range(N):
        for j in range(N):
            if W[i, j] != 0:
                num += W[i, j] * x0[i] * x0[j]
    return float((N / S0) * (num / denom))

def _map_roughness(x: np.ndarray, W: np.ndarray) -> float:
    N = x.size; num = 0.0; cnt = 0
    for i in range(N):
        for j in range(N):
            if W[i, j] != 0:
                num += (x[i] - x[j])**2; cnt += 1
    return float(num / cnt) if cnt > 0 else np.nan

def _plot_retinotopy(coords: pd.DataFrame, values: pd.Series, title: str, outpath: str):
    _set_pubstyle()
    fig, ax = plt.subplots(figsize=(5,4))
    sc = ax.scatter(coords["x"].values, coords["y"].values, s=20, c=values.values)
    ax.set_title(title); ax.set_xlabel("x (ROI)"); ax.set_ylabel("y (ROI)")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04); cb.set_label("response (ΔF/F0)")
    fig.savefig(outpath, bbox_inches="tight"); plt.close(fig)

def network_reorg_metrics_and_plots(
    dff_smooth: Union[pd.DataFrame, np.ndarray],
    fs: float = 3.6,
    stim_frame: int = 11,
    during_start: int = 12,
    during_end: int = 50,
    post_start: Optional[int] = 51,
    lowpass_seconds: float = 6.0,
    roi_coords: Optional[pd.DataFrame] = None,
    save_prefix: str = "/mnt/data/network_reorg"
) -> Dict[str, pd.DataFrame]:
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
    df = _ensure_df(dff_smooth)
    n_frames, n_rois = df.shape
    pre, during, post = (0, stim_frame), (during_start, during_end), ((post_start if post_start is not None else during_end + 1), n_frames - 1)
    windows = {"pre": pre, "during": during, "post": post}
    win_lp = max(1, int(round(lowpass_seconds * fs)))
    df_low = df.apply(lambda col: pd.Series(col.values).rolling(window=win_lp, center=True, min_periods=1).mean(), axis=0)
    summary_rows = []; corr_outputs = {}; rlong_outputs = {}; pca_outputs = {}
    for cls, (lo, hi) in windows.items():
        w_df = df.iloc[lo:hi+1, :]; w_df_low = df_low.iloc[lo:hi+1, :]
        corr = w_df.corr(); corr_low = w_df_low.corr()
        c = corr.copy(); np.fill_diagonal(c.values, np.nan); r_long = c.stack(dropna=True).reset_index(); r_long.columns = ["ROI_i","ROI_j","r"]
        cl = corr_low.copy(); np.fill_diagonal(cl.values, np.nan); r_low_long = cl.stack(dropna=True).reset_index(); r_low_long.columns = ["ROI_i","ROI_j","r"]
        mean_r = float(r_long["r"].mean()) if not r_long.empty else np.nan
        median_r = float(r_long["r"].median()) if not r_long.empty else np.nan
        mean_r_low = float(r_low_long["r"].mean()) if not r_low_long.empty else np.nan
        # PCA
        X = w_df.values.astype(float)
        col_med = np.nanmedian(X, axis=0); inds = np.where(~np.isfinite(X))
        if inds[0].size > 0: X[inds] = np.take(col_med, inds[1])
        X = X - np.mean(X, axis=0, keepdims=True)
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        var = (s**2) / (X.shape[0] - 1); vr = var / np.sum(var) if np.sum(var) > 0 else np.zeros_like(var)
        cum = np.cumsum(vr); pr = 1.0 / np.sum(vr**2) if np.sum(vr**2) > 0 else np.nan
        n80 = int(np.searchsorted(cum, 0.8) + 1) if cum.size > 0 else 0
        pc1_var = float(vr[0]) if vr.size > 0 else np.nan
        summary_rows.append({"class": cls, "n_frames": hi - lo + 1, "n_rois": n_rois, "mean_pairwise_r": mean_r, "median_pairwise_r": median_r, "mean_pairwise_r_lowfreq": mean_r_low, "pc1_variance_ratio": pc1_var, "effective_dimensionality": float(pr), "n_components_80": int(n80)})
        corr_outputs[f"corr_{cls}"] = corr; rlong_outputs[f"pairwise_r_{cls}"] = r_long; pca_outputs[cls] = {"vr": vr, "Vt": Vt}
        # Plots
        fig, ax = plt.subplots(figsize=(5,4)); im = ax.imshow(corr.values, aspect='auto', interpolation='nearest'); ax.set_title(f"Pairwise r ({cls})"); ax.set_xlabel("ROI"); ax.set_ylabel("ROI"); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); fig.savefig(f"{save_prefix}_{cls}_corr_heatmap.png", bbox_inches="tight"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5,4)); ax.hist(r_long["r"].values, bins=30); ax.set_title(f"Pairwise r distribution ({cls})"); ax.set_xlabel("pairwise r"); ax.set_ylabel("count"); fig.savefig(f"{save_prefix}_{cls}_pairwise_r_hist.png", bbox_inches="tight"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5,4)); x = np.arange(1, vr.size + 1); ax.plot(x, vr, marker="o", linewidth=1.5); ax.set_title(f"PCA scree ({cls})"); ax.set_xlabel("PC"); ax.set_ylabel("variance explained"); ax2 = ax.twinx(); ax2.plot(x, np.cumsum(vr), linestyle="--"); ax2.set_ylabel("cumulative variance"); fig.savefig(f"{save_prefix}_{cls}_pca_scree.png", bbox_inches="tight"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(5,4)); 
        if pca_outputs[cls]["Vt"].shape[0] >= 2:
            ax.scatter(pca_outputs[cls]["Vt"][0, :], pca_outputs[cls]["Vt"][1, :], s=16)
            ax.set_xlabel("PC1 loading"); ax.set_ylabel("PC2 loading")
        ax.set_title(f"PC1–PC2 loadings ({cls})"); fig.savefig(f"{save_prefix}_{cls}_pc1pc2_loadings.png", bbox_inches="tight"); plt.close(fig)
    summary_by_class = pd.DataFrame(summary_rows).set_index("class").sort_index(); summary_by_class.to_csv(f"{save_prefix}_summary_by_class.csv")
    for key, mat in corr_outputs.items(): mat.to_csv(f"{save_prefix}_{key}.csv")
    for key, rlong in rlong_outputs.items(): rlong.to_csv(f"{save_prefix}_{key}.csv", index=False)
    # Optional retinotopy
    if roi_coords is not None:
        lo, hi = windows["during"]
        during_mean = df.iloc[lo:hi+1, :].mean(axis=0)
        coords = roi_coords.reindex(df.columns).dropna()
        common = coords.index.intersection(during_mean.index)
        coords = coords.loc[common]; signal = during_mean.loc[common].values.astype(float)
        # kNN weights
        xy = coords[["x","y"]].values.astype(float); N = xy.shape[0]; W = np.zeros((N, N), dtype=float)
        for i in range(N):
            d = np.sqrt(np.sum((xy - xy[i])**2, axis=1))
            idx = np.argsort(d)[1:min(6, N-1)+1]
            W[i, idx] = 1.0
        W = np.maximum(W, W.T); x = signal; x_mean = np.mean(x); x0 = x - x_mean; S0 = np.sum(W); denom = np.sum(x0**2)
        I = (N / S0) * (np.sum(W * (x0[:, None] * x0[None, :])) / denom) if (S0 > 0 and denom > 0) else np.nan
        rough = np.nan
        if N > 1:
            num = 0.0; cnt = 0
            for i in range(N):
                for j in range(N):
                    if W[i, j] != 0:
                        num += (x[i] - x[j])**2; cnt += 1
            rough = (num / cnt) if cnt > 0 else np.nan
        pd.DataFrame([{"metric":"Moran_I","value":I},{"metric":"roughness","value":rough}]).set_index("metric").to_csv(f"{save_prefix}_retinotopy_metrics.csv")
        fig, ax = plt.subplots(figsize=(5,4)); sc = ax.scatter(coords["x"].values, coords["y"].values, s=20, c=signal); ax.set_title("Retinotopy map (during: mean ΔF/F0)"); ax.set_xlabel("x (ROI)"); ax.set_ylabel("y (ROI)"); cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04); cb.set_label("response (ΔF/F0)"); fig.savefig(f"{save_prefix}_retinotopy_scatter.png", bbox_inches="tight"); plt.close(fig)
    print("Saved summary and figures with prefix:", save_prefix)

