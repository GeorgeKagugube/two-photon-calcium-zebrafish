
# population_mapping_extended.py
# Self-contained Stage 5 (Population-level mapping) with:
#  - optional consumption of precomputed, stimulus-aligned PC time-series
#  - Moran's I (kNN weights) WITHOUT SciPy (NumPy-only implementation)
#  - journal-safe plots (no explicit colors)
#
# Public API:
#   population_level_mapping_extended(...)
#
from typing import Dict, Tuple, Optional, Union, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

__all__ = ["population_level_mapping_extended"]

# ---------------- utilities ----------------
def _ensure_df(x: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    elif isinstance(x, np.ndarray):
        df = pd.DataFrame(x)
        df.columns = [f"ROI_{i+1}" for i in range(df.shape[1])]
        return df
    else:
        raise TypeError("Input must be a DataFrame or ndarray")

def _class_windows(n_frames: int, stim_frame: int, during_start: int, during_end: int, post_start: Optional[int]):
    pre = (0, stim_frame)                      # inclusive (0..stim_frame)
    during = (during_start, during_end)        # inclusive
    if post_start is None:
        post_start = during_end + 1
    post = (post_start, n_frames - 1)          # inclusive
    return {"pre": pre, "during": during, "post": post}

def _mad_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return 1.4826 * mad + 1e-12

def _detect_events_one_roi(y: np.ndarray, fs: float,
                           k_sigma_height: float, k_sigma_prom: float,
                           min_distance_s: float) -> np.ndarray:
    # Robust thresholds
    sigma = _mad_sigma(y)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.nanstd(y) + 1e-12
    height_thr = k_sigma_height * sigma
    prom_thr = k_sigma_prom * sigma
    min_dist = max(1, int(round(min_distance_s * fs)))
    # Simple peak picker if scipy isn't available
    peaks = []
    last = -10**9
    for i in range(1, len(y)-1):
        if y[i] > y[i-1] and y[i] > y[i+1] and y[i] >= height_thr:
            if (i - last) >= min_dist:
                # approximate prominence: height minus local min in small window
                w = max(1, min_dist//2)
                left_min = np.nanmin(y[max(0,i-w):i]) if i-w >= 0 else y[i-1]
                right_min = np.nanmin(y[i+1:min(len(y), i+1+w)]) if i+1+w <= len(y) else y[i+1]
                prom = y[i] - max(left_min, right_min)
                if prom >= prom_thr:
                    peaks.append(i)
                    last = i
    return np.asarray(peaks, dtype=int)

def _get_responsive_sets(df: pd.DataFrame, fs: float, windows: Dict[str, Tuple[int,int]],
                         pre_df: Optional[pd.DataFrame], during_df: Optional[pd.DataFrame], post_df: Optional[pd.DataFrame],
                         responsive_only: bool,
                         k_sigma_height: float, k_sigma_prom: float, min_distance_s: float) -> Dict[str, List[str]]:
    if not responsive_only:
        return {k: list(df.columns) for k in windows.keys()}
    resp_sets: Dict[str, List[str]] = {}
    def _extract_resp(table):
        if table is None:
            return None
        if "roi" in table.columns and "responsive" in table.columns:
            m = table.set_index("roi")["responsive"].astype(bool).to_dict()
        elif "responsive" in table.columns:
            m = table["responsive"].astype(bool).to_dict()
        else:
            return None
        return [r for r, v in m.items() if v and r in df.columns]
    resp_sets["pre"]    = _extract_resp(pre_df)
    resp_sets["during"] = _extract_resp(during_df)
    resp_sets["post"]   = _extract_resp(post_df)
    resp_sets = {k:v for k,v in resp_sets.items() if v is not None}
    remaining = set(windows.keys()) - set(resp_sets.keys())
    if remaining:
        # infer responsiveness via event detection within each window
        peaks_by_roi = {}
        for roi in df.columns:
            y = df[roi].to_numpy(dtype=float)
            peaks_by_roi[roi] = _detect_events_one_roi(y, fs, k_sigma_height, k_sigma_prom, min_distance_s)
        for cls in remaining:
            lo, hi = windows[cls]
            resp_sets[cls] = [roi for roi in df.columns if np.any((peaks_by_roi[roi] >= lo) & (peaks_by_roi[roi] <= hi))]
    return resp_sets

def _pca_scores_and_loadings(W: np.ndarray, n_components: int):
    # Impute NaNs with column medians, mean-center, SVD
    X = W.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        med = np.nanmedian(col)
        col[~np.isfinite(col)] = med
        X[:, j] = col
    Xc = X - np.mean(X, axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_components, Vt.shape[0]) if Vt.size else 0
    if k == 0:
        return np.zeros((X.shape[0], 0)), np.zeros((X.shape[1], 0)), np.zeros((0,))
    scores = U[:, :k] * s[:k]       # T x k
    loadings = Vt[:k, :].T          # N x k
    var = (s**2) / max(1, (Xc.shape[0] - 1))
    vr = var / np.sum(var) if np.sum(var) > 0 else np.zeros_like(var)
    return scores, loadings, vr

def _build_lagged(X: np.ndarray, lags: List[int]) -> np.ndarray:
    T, K = X.shape
    Phi = np.zeros((T, K*len(lags)), dtype=float)
    for j, lag in enumerate(lags):
        if lag < 0:
            Phi[-lag:, j*K:(j+1)*K] = X[:T+lag, :]
        elif lag > 0:
            Phi[:T-lag, j*K:(j+1)*K] = X[lag:, :]
        else:
            Phi[:, j*K:(j+1)*K] = X
    return Phi

def _block_kfold_indices(T: int, n_folds: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    fold_sizes = np.full(n_folds, T // n_folds, dtype=int)
    fold_sizes[:T % n_folds] += 1
    idxs = np.arange(T)
    splits = []
    start = 0
    for f in range(n_folds):
        stop = start + fold_sizes[f]
        test_idx = idxs[start:stop]
        train_idx = np.concatenate([idxs[:start], idxs[stop:]])
        splits.append((train_idx, test_idx))
        start = stop
    return splits

def _zscore_cols(X: np.ndarray):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0) + 1e-12
    return (X - mu) / sd, mu, sd

def _ols_cv(Phi: np.ndarray, y: np.ndarray, n_folds: int = 5):
    T = len(y)
    splits = _block_kfold_indices(T, n_folds=n_folds)
    R2 = []
    for tr, te in splits:
        Xtr, Xte = Phi[tr], Phi[te]
        ytr, yte = y[tr], y[te]
        Xtr_z, mu, sd = _zscore_cols(Xtr)
        Xte_z = (Xte - mu) / sd
        Xtr_ = np.hstack([np.ones((Xtr_z.shape[0], 1)), Xtr_z])
        Xte_ = np.hstack([np.ones((Xte_z.shape[0], 1)),  Xte_z])
        w = np.linalg.pinv(Xtr_.T @ Xtr_) @ (Xtr_.T @ ytr)
        yhat = Xte_ @ w
        ss_res = np.sum((yte - yhat)**2)
        ss_tot = np.sum((yte - np.mean(yte))**2) + 1e-12
        R2.append(1.0 - ss_res/ss_tot)
    return float(np.mean(R2)), float(np.std(R2))

def _auc_from_scores(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = y_true.astype(int)
    pos = scores[y==1]; neg = scores[y==0]
    if pos.size == 0 or neg.size == 0:
        return np.nan
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, order.size+1)
    R_pos = np.sum(ranks[:pos.size])
    U = R_pos - pos.size*(pos.size+1)/2.0
    auc = U / (pos.size * neg.size)
    return float(auc)

def _logreg_cv(Phi: np.ndarray, y: np.ndarray, n_folds: int = 5, l2: float = 1e-2, max_iter: int = 80):
    T = len(y)
    splits = _block_kfold_indices(T, n_folds=n_folds)
    aucs = []
    for tr, te in splits:
        Xtr, Xte = Phi[tr], Phi[te]
        ytr, yte = y[tr], y[te].astype(int)
        Xtr_z, mu, sd = _zscore_cols(Xtr)
        Xte_z = (Xte - mu) / sd
        Xtr_ = np.hstack([np.ones((Xtr_z.shape[0], 1)), Xtr_z])
        Xte_ = np.hstack([np.ones((Xte_z.shape[0], 1)),  Xte_z])
        w = np.zeros(Xtr_.shape[1])
        for it in range(max_iter):
            z = Xtr_ @ w
            p = 1.0/(1.0 + np.exp(-z))
            W = p*(1.0-p) + 1e-9
            H = (Xtr_.T * W) @ Xtr_ + l2*np.eye(Xtr_.shape[1])
            g = Xtr_.T @ (p - ytr) + l2*w
            try:
                step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                step = np.linalg.pinv(H) @ g
            w -= step
            if np.linalg.norm(step) < 1e-6:
                break
        scores = Xte_ @ w
        auc = _auc_from_scores(yte, scores)
        aucs.append(auc)
    return float(np.nanmean(aucs)), float(np.nanstd(aucs))

def _bout_counts_per_frame(bouts_df: Optional[pd.DataFrame], n_frames: int, fs: float):
    y_bin = np.zeros(n_frames, dtype=int)
    y_cnt = np.zeros(n_frames, dtype=float)
    if bouts_df is None or bouts_df.empty or "onset_time" not in bouts_df.columns:
        return y_bin, y_cnt
    ons = bouts_df["onset_time"].to_numpy()
    fr = np.clip(np.floor(ons * fs).astype(int), 0, n_frames-1)
    for f in fr:
        y_bin[f] = 1
        y_cnt[f] += 1.0
    return y_bin, y_cnt

# ---- Moran's I (global) with kNN weights; NumPy-only ----
def _knn_weights_numpy(xy: np.ndarray, k: int = 8, mode: str = "binary", eps: float = 1e-9) -> np.ndarray:
    \"\"\"Return row-standardized spatial weights matrix W (N x N).
    mode: 'binary' (1 for kNN) or 'invdist' (1/d).\"\"\"
    xy = np.asarray(xy, float)
    N = xy.shape[0]
    # pairwise distances
    diff = xy[:, None, :] - xy[None, :, :]
    D = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(D, np.inf)
    W = np.zeros((N, N), dtype=float)
    k_eff = max(1, min(k, N-1))
    for i in range(N):
        nbr_idx = np.argpartition(D[i], k_eff)[:k_eff]
        if mode == "binary":
            W[i, nbr_idx] = 1.0
        else:
            d = D[i, nbr_idx]
            W[i, nbr_idx] = 1.0 / (d + eps)
    # row-standardize
    row_sum = W.sum(axis=1, keepdims=True) + eps
    W = W / row_sum
    return W

def _moran_global(z: np.ndarray, W: np.ndarray) -> float:
    z = np.asarray(z, float)
    m = np.nanmean(z)
    zm = z - m
    zm[~np.isfinite(zm)] = 0.0
    N = len(z)
    S0 = np.sum(W)
    num = float(zm @ W @ zm)
    den = float(np.sum(zm**2)) + 1e-12
    I = (N / (S0 + 1e-12)) * (num / den)
    return float(I)

def _moran_perm_test(z: np.ndarray, W: np.ndarray, n_perm: int = 500, seed: Optional[int] = 42):
    rng = np.random.default_rng(seed)
    z = np.asarray(z, float)
    I_obs = _moran_global(z, W)
    cnt = 0
    n_perm = max(10, int(n_perm))
    for _ in range(n_perm):
        z_perm = rng.permutation(z)
        I_perm = _moran_global(z_perm, W)
        if I_perm >= I_obs:
            cnt += 1
    p = (cnt + 1) / (n_perm + 1)
    return float(I_obs), float(p)

# --------------- main function ---------------
def population_level_mapping_extended(
    dff_smooth: Union[pd.DataFrame, np.ndarray],
    fs: float = 3.6,
    stim_frame: int = 11, during_start: int = 12, during_end: int = 50, post_start: int = 51,
    # responsive handling
    responsive_only: bool = True,
    pre_df: Optional[pd.DataFrame] = None,
    during_df: Optional[pd.DataFrame] = None,
    post_df: Optional[pd.DataFrame] = None,
    # consume precomputed, stimulus-aligned PC time-series (optional):
    # DataFrame with columns: ['class','time_s','pc1_score',...,'pcK_score']
    precomputed_pcs_ts: Optional[pd.DataFrame] = None,
    n_pcs: int = 5,
    # behaviour
    bouts_df: Optional[pd.DataFrame] = None,    # onset_time (s)
    eta_window_s: Tuple[float, float] = (-2.0, 3.0),
    lags_s: Tuple[float, float] = (2.0, 2.0),
    n_folds: int = 5,
    # retinotopy: ROI positions and per-ROI metric (optional)
    roi_xy: Optional[Union[pd.DataFrame, np.ndarray]] = None,  # either DataFrame with ['roi','x','y'] or ndarray (N x 2) matching df columns
    roi_metric_df: Optional[pd.DataFrame] = None,              # optional per-ROI×class metric table, with columns ['roi','class', <metric_col>]
    metric_col: Optional[str] = None,                          # which column in roi_metric_df to use for Moran's I
    moran_from: str = "pc1_loading",                           # 'pc1_loading' or 'roi_metric'
    k_neighbors: int = 8,
    weight_mode: str = "binary",                               # 'binary' or 'invdist'
    n_perm: int = 500,
    take_abs_loadings: bool = False,                           # use |PC1 loadings| for Moran if sign flips are a concern
    # output
    save_prefix: str = "./population_mapping_ext"
) -> Dict[str, Union[pd.DataFrame, str]]:
    \"\"\"
    INPUTS
    ------
    dff_smooth : (frames × ROIs) ΔF/F₀ DataFrame/ndarray.
    fs, stim_frame, during_start/end, post_start : recording geometry (frames & Hz).
    responsive_only : if True, restrict within each window to responsive ROIs. If
      'pre_df'/'during_df'/'post_df' provided with 'responsive' per ROI, those are used;
      otherwise responsiveness is inferred via event detection (MAD thresholds).
    precomputed_pcs_ts : optional DataFrame with ['class','time_s','pc1_score',...]. If provided,
      these PC scores are used directly (must cover each window). If absent or mismatched,
      PCA is computed per window from dff_smooth.
    bouts_df : DataFrame with 'onset_time' (seconds), used for ETA and for encoding/decoding targets.
    eta_window_s : window around bout onset for ETAs.
    lags_s, n_folds : model settings for encoding/decoding from lagged PCs.
    roi_xy : ROI coordinates. Either ndarray (N×2) in the same order as dff_smooth columns,
      or DataFrame with columns ['roi','x','y'] matching ROI names.
    roi_metric_df : optional per-ROI×class table to compute Moran's I on (e.g., 'p_bout_event_diff').
    metric_col : name of the metric column within roi_metric_df to use.
    moran_from : 'pc1_loading' (compute on PCA loadings) or 'roi_metric' (use roi_metric_df[metric_col]).
    k_neighbors, weight_mode : spatial weight graph settings ('binary' kNN or 'invdist').
    n_perm : number of permutations for Moran's I p-value.
    take_abs_loadings : if True, use |PC1 loadings| (guards against arbitrary PC sign flips).
    save_prefix : output file prefix.

    OUTPUTS
    -------
    dict with:
      - 'timeseries_csv' (population means & PCs if computed),
      - 'eta_csv'       (ETA of pop mean & PC1),
      - 'encoding_csv'  (mean±SD CV R² by window),
      - 'decoding_csv'  (mean±SD CV AUC by window),
      - 'pca_loadings_csv' (per window, per ROI loadings),
      - 'moran_csv'     (Moran's I and permutation p-value by window),
      plus the corresponding DataFrames in-memory.
    \"\"\"
    os.makedirs(os.path.dirname(save_prefix) or ".", exist_ok=True)
    df = _ensure_df(dff_smooth)
    n_frames, n_rois = df.shape
    windows = _class_windows(n_frames, stim_frame, during_start, during_end, post_start)
    # responsive sets
    resp_sets = _get_responsive_sets(df, fs, windows, pre_df, during_df, post_df, responsive_only,
                                     k_sigma_height=2.5, k_sigma_prom=2.0, min_distance_s=0.5)

    ts_rows = []
    loading_rows = []
    encoding_rows = []
    decoding_rows = []
    eta_pre, eta_post = eta_window_s
    eta_len = int(round((eta_post - eta_pre) * fs)) + 1
    eta_times = np.arange(eta_len)/fs + eta_pre
    eta_popmean_acc = {k: [] for k in windows}
    eta_pc1_acc     = {k: [] for k in windows}

    # bout labels
    y_bin, y_cnt = _bout_counts_per_frame(bouts_df, n_frames, fs)

    # Iterate windows
    for cls, (lo, hi) in windows.items():
        keep = resp_sets[cls] if responsive_only else list(df.columns)
        W = df.loc[lo:hi, keep].to_numpy(dtype=float)  # T x Nwin
        T = W.shape[0]
        t_window = np.arange(lo, hi+1)/fs

        # --- Use precomputed PCs if provided ---
        use_precomp = False
        scores = None
        if precomputed_pcs_ts is not None and not precomputed_pcs_ts.empty:
            sub = precomputed_pcs_ts[precomputed_pcs_ts["class"] == cls]
            if not sub.empty and sub.shape[0] == T:
                arr = []
                for i in range(n_pcs):
                    cname = f"pc{i+1}_score"
                    if cname in sub.columns:
                        arr.append(sub[cname].to_numpy(dtype=float))
                    else:
                        arr.append(np.full(T, np.nan))
                scores = np.vstack(arr).T  # T x k
                use_precomp = True

        # --- If no precomputed/mismatched, compute PCA now ---
        pop_mean = np.nanmean(W, axis=1) if W.size else np.full(T, np.nan)
        if not use_precomp:
            if W.size and W.shape[1] > 0:
                S, L, vr = _pca_scores_and_loadings(W, n_components=n_pcs)  # S: T x k, L: N x k
                scores = S
                # Record loadings (ROI weights) for Moran's I
                for j, roi in enumerate(keep):
                    row = {"class": cls, "roi": roi}
                    for p in range(min(n_pcs, L.shape[1])):
                        row[f"pc{p+1}_loading"] = float(L[j, p])
                    loading_rows.append(row)
            else:
                scores = np.zeros((T, 0), dtype=float)
        else:
            # if using precomputed, we don't know loadings; set NaNs in loading table (optional)
            for roi in keep:
                row = {"class": cls, "roi": roi}
                for p in range(n_pcs):
                    row[f"pc{p+1}_loading"] = np.nan
                loading_rows.append(row)

        # Save timeseries rows
        score_cols = {}
        for i in range(n_pcs):
            col = scores[:, i] if (scores is not None and i < scores.shape[1]) else np.full(T, np.nan)
            score_cols[f"pc{i+1}_score"] = col
        ts_rows.append(pd.DataFrame({"class": cls, "time_s": t_window, "pop_mean": pop_mean, **score_cols}))

        # --- Event-triggered averages around bout onsets ---
        if bouts_df is not None and not bouts_df.empty:
            bout_onsets = bouts_df["onset_time"].to_numpy()
            w_lo_s, w_hi_s = lo/fs, (hi+1)/fs
            on_in = bout_onsets[(bout_onsets >= w_lo_s) & (bout_onsets < w_hi_s)]
            for bt in on_in:
                center_idx = int(round(bt*fs)) - lo
                a0 = center_idx + int(round(eta_pre*fs))
                a1 = a0 + eta_len
                if a0 < 0 or a1 > T:
                    continue
                eta_popmean_acc[cls].append(pop_mean[a0:a1])
                if scores is not None and scores.shape[1] >= 1:
                    eta_pc1_acc[cls].append(scores[a0:a1, 0])

        # --- Encoding / Decoding ---
        past, future = lags_s
        lags = list(range(-int(round(past*fs)), int(round(future*fs))+1))
        Phi = _build_lagged(scores if scores is not None else np.zeros((T,0)), lags) if (scores is not None and scores.shape[1] > 0) else np.zeros((T,0))
        stim_reg = np.zeros(T, dtype=float)
        stim_lo = max(lo, during_start); stim_hi = min(hi, during_end)
        if stim_hi >= stim_lo:
            stim_reg[stim_lo-lo:stim_hi-lo+1] = 1.0
        Phi_enc = np.hstack([Phi, stim_reg.reshape(-1,1)])
        y_cnt_win = y_cnt[lo:hi+1]
        if Phi_enc.shape[1] > 0:
            r2_mean, r2_std = _ols_cv(Phi_enc, y_cnt_win, n_folds=n_folds)
        else:
            r2_mean, r2_std = (np.nan, np.nan)
        encoding_rows.append({"class": cls, "n_rois_used": len(keep), "n_pcs": scores.shape[1] if scores is not None else 0,
                              "lags": len(lags), "encoding_r2_mean": r2_mean, "encoding_r2_std": r2_std})

        y_bin_win = y_bin[lo:hi+1]
        if Phi_enc.shape[1] > 0 and np.any(y_bin_win==1) and np.any(y_bin_win==0):
            auc_mean, auc_std = _logreg_cv(Phi_enc, y_bin_win, n_folds=n_folds, l2=1e-2, max_iter=80)
        else:
            auc_mean, auc_std = (np.nan, np.nan)
        decoding_rows.append({"class": cls, "n_rois_used": len(keep), "n_pcs": scores.shape[1] if scores is not None else 0,
                              "lags": len(lags), "decoder_auc_mean": auc_mean, "decoder_auc_std": auc_std})

    pop_timeseries = pd.concat(ts_rows, ignore_index=True) if ts_rows else pd.DataFrame()
    pca_loadings = pd.DataFrame(loading_rows) if loading_rows else pd.DataFrame()
    encoding_df = pd.DataFrame(encoding_rows).set_index("class").sort_index()
    decoding_df = pd.DataFrame(decoding_rows).set_index("class").sort_index()

    # --- Build ETA DataFrame ---
    eta_rows = []
    for cls in windows:
        if len(eta_popmean_acc[cls]) > 0:
            eta_popmean = np.nanmean(np.vstack(eta_popmean_acc[cls]), axis=0)
        else:
            eta_popmean = np.full(eta_len, np.nan)
        if len(eta_pc1_acc[cls]) > 0:
            eta_pc1 = np.nanmean(np.vstack(eta_pc1_acc[cls]), axis=0)
        else:
            eta_pc1 = np.full(eta_len, np.nan)
        eta_rows.append(pd.DataFrame({"class": cls, "time_rel_s": eta_times,
                                      "eta_pop_mean": eta_popmean, "eta_pc1": eta_pc1}))
    eta_df = pd.concat(eta_rows, ignore_index=True) if eta_rows else pd.DataFrame()

    # --- Retinotopic spatial clustering (Moran's I) ---
    moran_rows = []
    if roi_xy is not None:
        # Build XY array and ROI name order
        if isinstance(roi_xy, pd.DataFrame):
            if set(["roi","x","y"]).issubset(roi_xy.columns):
                mapping = roi_xy.set_index("roi")[["x","y"]]
                mapping = mapping.loc[[c for c in df.columns if c in mapping.index]]
                xy = mapping.to_numpy(dtype=float)
                roi_order = list(mapping.index)
            elif roi_xy.shape[1] >= 2:
                xy = roi_xy.iloc[:, :2].to_numpy(dtype=float)
                roi_order = list(df.columns[:xy.shape[0]])
            else:
                xy, roi_order = None, None
        else:
            # ndarray
            xy = np.asarray(roi_xy, float)
            roi_order = list(df.columns[:xy.shape[0]])
        if xy is not None and xy.shape[0] >= 3:
            W = _knn_weights_numpy(xy, k=k_neighbors, mode=weight_mode)
            for cls in windows:
                # choose z per ROI
                if moran_from == "roi_metric" and roi_metric_df is not None and metric_col is not None:
                    sub = roi_metric_df[roi_metric_df["class"] == cls]
                    z_map = sub.set_index("roi")[metric_col].to_dict()
                    z = np.array([z_map.get(r, np.nan) for r in roi_order], dtype=float)
                else:
                    # use PC1 loadings from PCA results for that window (requires we computed them)
                    if pca_loadings.empty:
                        z = np.full(len(roi_order), np.nan)
                    else:
                        Lsub = pca_loadings[pca_loadings["class"] == cls].set_index("roi")
                        z = Lsub.get("pc1_loading", pd.Series(index=roi_order, dtype=float)).reindex(roi_order).to_numpy(dtype=float)
                    if take_abs_loadings:
                        z = np.abs(z)
                if np.isfinite(z).sum() >= 3:
                    I, p = _moran_perm_test(z, W, n_perm=n_perm, seed=42)
                else:
                    I, p = (np.nan, np.nan)
                moran_rows.append({"class": cls, "metric": (metric_col if (moran_from=='roi_metric') else 'pc1_loading' + ('_abs' if take_abs_loadings else '')),
                                   "moran_I": I, "p_perm": p, "N": int(len(z)), "k_neighbors": int(k_neighbors), "weight_mode": weight_mode})
    moran_df = pd.DataFrame(moran_rows) if moran_rows else pd.DataFrame()

    # --- Save CSVs ---
    ts_csv = f"{save_prefix}_timeseries.csv"
    enc_csv = f"{save_prefix}_encoding.csv"
    dec_csv = f"{save_prefix}_decoding.csv"
    eta_csv = f"{save_prefix}_eta.csv"
    loads_csv = f"{save_prefix}_pca_loadings.csv"
    moran_csv = f"{save_prefix}_moran.csv"
    pop_timeseries.to_csv(ts_csv, index=False)
    encoding_df.to_csv(enc_csv)
    decoding_df.to_csv(dec_csv)
    eta_df.to_csv(eta_csv, index=False)
    pca_loadings.to_csv(loads_csv, index=False)
    moran_df.to_csv(moran_csv, index=False)

    # --- Optional plots: Moran's I scatter maps (marker size ∝ |metric|) ---
    if roi_xy is not None and not moran_df.empty:
        if isinstance(roi_xy, pd.DataFrame) and set(["roi","x","y"]).issubset(roi_xy.columns):
            xy_plot = roi_xy.set_index("roi").loc[[c for c in df.columns if c in roi_xy["roi"].values]][["x","y"]].to_numpy()
            roi_order = [c for c in df.columns if c in roi_xy["roi"].values]
        else:
            xy_plot = np.asarray(roi_xy, float)[:len(df.columns), :2]
            roi_order = list(df.columns[:xy_plot.shape[0]])
        for cls in moran_df["class"].unique():
            if (moran_from == "roi_metric") and (roi_metric_df is not None) and (metric_col is not None):
                sub = roi_metric_df[roi_metric_df["class"] == cls]
                z_map = sub.set_index("roi")[metric_col].to_dict()
                z = np.array([z_map.get(r, np.nan) for r in roi_order], dtype=float)
            else:
                Lsub = pca_loadings[pca_loadings["class"] == cls].set_index("roi")
                z = Lsub.get("pc1_loading", pd.Series(index=roi_order, dtype=float)).reindex(roi_order).to_numpy(dtype=float)
                if take_abs_loadings:
                    z = np.abs(z)
            plt.figure(figsize=(4.5,4.0))
            plt.scatter(xy_plot[:,0], xy_plot[:,1])
            s = 10.0 + 40.0 * (np.nan_to_num(np.abs(z)) / (np.nanmax(np.abs(z)) + 1e-12))
            plt.scatter(xy_plot[:,0], xy_plot[:,1], s=s)
            plt.xlabel("x"); plt.ylabel("y")
            plt.title(f"Moran metric ({cls})")
            plt.savefig(f"{save_prefix}_moran_map_{cls}.png", bbox_inches="tight")
            plt.close()

    return {
        "pop_timeseries": pop_timeseries,
        "pca_loadings": pca_loadings,
        "encoding_df": encoding_df,
        "decoding_df": decoding_df,
        "eta_df": eta_df,
        "moran_df": moran_df,
        "timeseries_csv": ts_csv,
        "encoding_csv": enc_csv,
        "decoding_csv": dec_csv,
        "eta_csv": eta_csv,
        "pca_loadings_csv": loads_csv,
        "moran_csv": moran_csv
    }
