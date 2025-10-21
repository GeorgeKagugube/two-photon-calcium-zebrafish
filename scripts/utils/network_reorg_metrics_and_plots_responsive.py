# --- Network re-organisation (responsive-only) -------------------------------
# Computes per-window metrics using ONLY ROIs marked responsive in pre/during/post tables.
# Outputs: summary CSV + per-window figures; returns a dict with DataFrames and file paths.

import numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from typing import Dict, Tuple, Optional, Union, List

def network_reorg_metrics_and_plots_responsive(
    dff: Union[pd.DataFrame, np.ndarray],
    fs: float = 3.6,
    stim_frame: int = 11, during_start: int = 12, during_end: int = 50, post_start: int = 51,
    # Responsiveness tables MUST have columns ['roi','responsive'] (boolean)
    pre_df: Optional[pd.DataFrame] = None,
    during_df: Optional[pd.DataFrame] = None,
    post_df: Optional[pd.DataFrame] = None,
    lowfreq_hz: float = 0.10,                 # fraction of power below this freq
    save_prefix: str = "./network_reorg_resp"
) -> Dict[str, Union[pd.DataFrame, str]]:
    # ---- helpers ----
    def _ensure_df(x):
        if isinstance(x, pd.DataFrame): return x.copy()
        x = np.asarray(x)
        df = pd.DataFrame(x)
        df.columns = [f"ROI_{i+1}" for i in range(df.shape[1])]
        return df

    def _wins(nF, sf, ds, de, ps):
        if ps is None: ps = de + 1
        return {"pre": (0, sf), "during": (ds, de), "post": (ps, nF - 1)}

    def _keep_list(df_all, resp_tbl):
        if resp_tbl is None:
            raise ValueError("Missing responsiveness table. Provide pre_df, during_df, post_df with ['roi','responsive'].")
        need = {"roi", "responsive"}
        if not need.issubset(resp_tbl.columns):
            raise ValueError(f"Responsiveness table must have columns {need}. Got: {list(resp_tbl.columns)}")
        keep = resp_tbl.loc[resp_tbl["responsive"].astype(bool), "roi"].tolist()
        # keep only ROIs that exist in the data matrix
        return [r for r in keep if r in df_all.columns]

    def _zscore_cols(W):
        mu = np.nanmean(W, axis=0, keepdims=True)
        sd = np.nanstd(W, axis=0, keepdims=True) + 1e-12
        return (W - mu) / sd

    def _pca_stats(W):
        """Return PC1 variance fraction, eigenvalues, and PC1 score."""
        if W.size == 0 or W.shape[1] == 0:
            return np.nan, np.array([]), np.full(W.shape[0], np.nan)
        X = W.copy()
        # impute NaNs with column medians
        for j in range(X.shape[1]):
            col = X[:, j]
            med = np.nanmedian(col)
            col[~np.isfinite(col)] = med
            X[:, j] = col
        Xc = X - np.mean(X, axis=0, keepdims=True)
        U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        eig = (s**2) / max(1, (Xc.shape[0] - 1))
        pc1_frac = float(eig[0] / np.sum(eig)) if eig.size and np.sum(eig) > 0 else np.nan
        pc1_score = U[:, 0] * s[0] if s.size else np.full(X.shape[0], np.nan)
        return pc1_frac, eig, pc1_score

    def _effective_dimensionality(eig):
        if eig.size == 0: return np.nan
        num = (np.sum(eig) ** 2)
        den = np.sum(eig ** 2) + 1e-12
        return float(num / den)

    def _mean_pairwise_r(W):
        if W.shape[1] < 2: return np.nan
        Wz = _zscore_cols(W)
        C = np.corrcoef(Wz, rowvar=False)
        iu = np.triu_indices_from(C, k=1)
        return float(np.nanmean(C[iu]))

    def _lowfreq_fraction(W, fs, fcut):
        if W.size == 0: return np.nan
        T = W.shape[0]
        X = W - np.nanmean(W, axis=0, keepdims=True)
        X[~np.isfinite(X)] = 0.0
        F = np.fft.rfftfreq(T, d=1/fs)
        P = np.abs(np.fft.rfft(X, axis=0)) ** 2
        hi = np.searchsorted(F, fcut, side="right")
        lo_frac = np.nanmean(np.sum(P[1:hi, :], axis=0) / (np.sum(P[1:, :], axis=0) + 1e-12))
        return float(lo_frac)

    # ---- setup ----
    os.makedirs(os.path.dirname(save_prefix) or ".", exist_ok=True)
    df = _ensure_df(dff)
    nF, _ = df.shape
    wins = _wins(nF, stim_frame, during_start, during_end, post_start)
    resp_map = {"pre": pre_df, "during": during_df, "post": post_df}

    rows, ts_rows = [], []

    for cls, (lo, hi) in wins.items():
        keep = _keep_list(df, resp_map[cls])  # enforce responsive-only
        W = df.loc[lo:hi, keep].to_numpy(float)  # T x N
        t = np.arange(lo, hi + 1) / fs

        pop_mean = np.nanmean(W, axis=1) if W.size else np.full((hi - lo + 1,), np.nan)
        pc1_frac, eigvals, pc1_score = _pca_stats(W)
        effD = _effective_dimensionality(eigvals)
        mean_r = _mean_pairwise_r(W)
        lowfrac = _lowfreq_fraction(W, fs, lowfreq_hz)

        rows.append({
            "class": cls,
            "n_rois_used": len(keep),
            "mean_pairwise_r": mean_r,
            "pc1_var_fraction": pc1_frac,
            "effective_dimensionality": effD,
            f"lowfreq_fraction_<{lowfreq_hz:.2f}Hz": lowfrac
        })

        ts_rows.append(pd.DataFrame({
            "class": cls, "time_s": t,
            "pop_mean": pop_mean,
            "pc1_score": pc1_score
        }))

        # ---- quick figure per window ----
        fig = plt.figure(figsize=(9, 6))
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1], wspace=0.35, hspace=0.35)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(t, pop_mean)
        if np.isfinite(pc1_score).any(): ax1.plot(t, pc1_score)
        ax1.set_title(f"{cls}: pop mean & PC1"); ax1.set_xlabel("time (s)"); ax1.set_ylabel("ΔF/F₀ (a.u.)")

        ax2 = fig.add_subplot(gs[0, 1])
        if W.shape[1] >= 2:
            Wz = _zscore_cols(W); C = np.corrcoef(Wz, rowvar=False)
            im = ax2.imshow(C, vmin=-1, vmax=1, aspect="auto")
            ax2.set_title(f"{cls}: correlation (responsive)")
        else:
            ax2.text(0.5, 0.5, "N<2", ha="center", va="center"); ax2.set_title(f"{cls}: correlation")

        ax3 = fig.add_subplot(gs[0, 2])
        if eigvals.size > 0:
            ax3.plot(np.arange(1, eigvals.size + 1), eigvals, marker="o")
        ax3.set_title(f"{cls}: eigenvalue scree"); ax3.set_xlabel("PC"); ax3.set_ylabel("eigenvalue")

        ax4 = fig.add_subplot(gs[1, 0])
        if W.shape[1] >= 2:
            iu = np.triu_indices(W.shape[1], 1)
            rvals = (np.corrcoef(_zscore_cols(W), rowvar=False))[iu]
            ax4.hist(rvals, bins=20)
        else:
            ax4.text(0.5, 0.5, "N<2", ha="center", va="center")
        ax4.set_title(f"{cls}: pairwise r (hist)"); ax4.set_xlabel("r"); ax4.set_ylabel("count")

        ax5 = fig.add_subplot(gs[1, 1])
        bars_x = ["PC1 var", f"Low<{lowfreq_hz:.2f} Hz"]
        bars_y = [
            pc1_frac if np.isfinite(pc1_frac) else 0.0,
            lowfrac if np.isfinite(lowfrac) else 0.0
        ]
        ax5.bar(bars_x, bars_y); ax5.set_ylim(0, 1); ax5.set_title(f"{cls}: synchrony & low-freq")

        ax6 = fig.add_subplot(gs[1, 2]); ax6.axis("off")
        txt = (f"n ROIs: {len(keep)}\n"
               f"mean r: {mean_r:.3f}\n"
               f"PC1 var: {pc1_frac:.3f}\n"
               f"Eff dim: {effD:.2f}\n"
               f"Low-freq frac: {lowfrac:.3f}")
        ax6.text(0.05, 0.95, txt, va="top", ha="left")

        fig.savefig(f"{save_prefix}_{cls}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    summary_by_class = pd.DataFrame(rows).set_index("class").sort_index()
    ts_all = pd.concat(ts_rows, ignore_index=True)

    csv_sum = f"{save_prefix}_summary.csv"
    csv_ts  = f"{save_prefix}_timeseries.csv"
    summary_by_class.to_csv(csv_sum)
    ts_all.to_csv(csv_ts, index=False)

    return {
        "summary_by_class": summary_by_class,
        "timeseries_df": ts_all,
        "summary_csv": csv_sum,
        "timeseries_csv": csv_ts,
        "fig_prefix": save_prefix + "_{pre|during|post}.png"
    }
