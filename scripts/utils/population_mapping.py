
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple, Optional, Union, List

def _ensure_df(x):
    if isinstance(x, pd.DataFrame): return x.copy()
    elif isinstance(x, np.ndarray):
        df = pd.DataFrame(x); df.columns = [f"ROI_{i+1}" for i in range(df.shape[1])]; return df
    else: raise TypeError("dff_smooth must be a DataFrame or ndarray")

def _class_windows(n_frames, stim_frame, during_start, during_end, post_start):
    pre = (0, stim_frame); during = (during_start, during_end)
    if post_start is None: post_start = during_end + 1
    post = (post_start, n_frames - 1); return {"pre": pre, "during": during, "post": post}

def _mad_sigma(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0: return np.nan
    med = np.nanmedian(x); mad = np.nanmedian(np.abs(x - med)); return 1.4826 * mad + 1e-12

def _detect_events_one_roi(y, fs, k_sigma_height, k_sigma_prom, min_distance_s):
    sigma = _mad_sigma(y); 
    if not np.isfinite(sigma) or sigma <= 0: sigma = np.nanstd(y) + 1e-12
    height_thr = k_sigma_height * sigma; prom_thr = k_sigma_prom * sigma
    min_dist = max(1, int(round(min_distance_s * fs)))
    try:
        from scipy.signal import find_peaks
        peaks, props = find_peaks(y, height=height_thr, prominence=prom_thr, distance=min_dist)
    except Exception:
        peaks = []
        for i in range(1, len(y)-1):
            if y[i] > y[i-1] and y[i] > y[i+1] and y[i] >= height_thr:
                if len(peaks)==0 or (i - peaks[-1]) >= min_dist: peaks.append(i)
        peaks = np.array(peaks, dtype=int)
    return peaks.astype(int)

def _get_responsive_sets(df, fs, windows, pre_df, during_df, post_df, responsive_only, k_sigma_height, k_sigma_prom, min_distance_s):
    if not responsive_only: return {k: list(df.columns) for k in windows.keys()}
    resp_sets = {}
    if pre_df is not None and "responsive" in pre_df.columns:
        resp_sets["pre"] = [r for r, v in pre_df["responsive"].items() if bool(v) and r in df.columns]
    if during_df is not None and "responsive" in during_df.columns:
        resp_sets["during"] = [r for r, v in during_df["responsive"].items() if bool(v) and r in df.columns]
    if post_df is not None and "responsive" in post_df.columns:
        resp_sets["post"] = [r for r, v in post_df["responsive"].items() if bool(v) and r in df.columns]
    remaining = set(windows.keys()) - set(resp_sets.keys())
    if remaining:
        peaks_by_roi = {roi: _detect_events_one_roi(df[roi].to_numpy(dtype=float), fs, k_sigma_height, k_sigma_prom, min_distance_s) for roi in df.columns}
        for cls in remaining:
            lo, hi = windows[cls]
            resp_sets[cls] = [roi for roi in df.columns if np.any((peaks_by_roi[roi] >= lo) & (peaks_by_roi[roi] <= hi))]
    return resp_sets

def _zscore_cols(X):
    mu = np.nanmean(X, axis=0); sd = np.nanstd(X, axis=0) + 1e-12; return (X - mu) / sd, mu, sd

def _pca_scores(X, n_components):
    Xc = X - np.nanmean(X, axis=0, keepdims=True)
    med = np.nanmedian(Xc, axis=0); inds = np.where(!np.isfinite(Xc)) if False else None
    U, s, Vt = np.linalg.svd(np.nan_to_num(Xc, nan=np.nanmedian(Xc, axis=0)), full_matrices=False)
    k = min(n_components, Vt.shape[0]); S = U[:, :k] * s[:k]
    var = (s**2) / (Xc.shape[0] - 1); vr = var / np.sum(var) if np.sum(var) > 0 else np.zeros_like(var)
    return S, vr

def _build_lagged(X, lags):
    T, K = X.shape; Phi = np.zeros((T, K*len(lags)), dtype=float)
    for j, lag in enumerate(lags):
        if lag < 0: Phi[-lag:, j*K:(j+1)*K] = X[:T+lag, :]
        elif lag > 0: Phi[:T-lag, j*K:(j+1)*K] = X[lag:, :]
        else: Phi[:, j*K:(j+1)*K] = X
    return Phi

def _block_kfold_indices(T, n_folds=5):
    fold_sizes = np.full(n_folds, T // n_folds, dtype=int); fold_sizes[:T % n_folds] += 1
    idxs = np.arange(T); splits = []; start = 0
    for f in range(n_folds):
        stop = start + fold_sizes[f]; test_idx = idxs[start:stop]
        train_idx = np.concatenate([idxs[:start], idxs[stop:]]); splits.append((train_idx, test_idx)); start = stop
    return splits

def _ols_cv(Phi, y, n_folds=5):
    T = len(y); splits = _block_kfold_indices(T, n_folds=n_folds); R2 = []
    for tr, te in splits:
        Xtr, Xte = Phi[tr], Phi[te]; ytr, yte = y[tr], y[te]
        Xtr_z, mu, sd = _zscore_cols(Xtr); Xte_z = (Xte - mu) / sd
        Xtr_ = np.hstack([np.ones((Xtr_z.shape[0], 1)), Xtr_z]); Xte_  = np.hstack([np.ones((Xte_z.shape[0], 1)),  Xte_z])
        w = np.linalg.pinv(Xtr_.T @ Xtr_) @ (Xtr_.T @ ytr); yhat = Xte_ @ w
        ss_res = np.sum((yte - yhat)**2); ss_tot = np.sum((yte - np.mean(yte))**2) + 1e-12; R2.append(1.0 - ss_res/ss_tot)
    return float(np.mean(R2)), float(np.std(R2))

def _auc_from_scores(y_true, scores):
    y = y_true.astype(int); pos = scores[y==1]; neg = scores[y==0]
    if pos.size == 0 or neg.size == 0: return np.nan
    order = np.argsort(np.concatenate([pos, neg])); ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, order.size+1)
    R_pos = np.sum(ranks[:pos.size]); U = R_pos - pos.size*(pos.size+1)/2.0; auc = U / (pos.size * neg.size); return float(auc)

def _logreg_cv(Phi, y, n_folds=5, l2=1e-2, max_iter=100):
    T = len(y); splits = _block_kfold_indices(T, n_folds=n_folds); aucs = []
    for tr, te in splits:
        Xtr, Xte = Phi[tr], Phi[te]; ytr, yte = y[tr], y[te].astype(int)
        Xtr_z, mu, sd = _zscore_cols(Xtr); Xte_z = (Xte - mu) / sd
        Xtr_ = np.hstack([np.ones((Xtr_z.shape[0], 1)), Xtr_z]); Xte_ = np.hstack([np.ones((Xte_z.shape[0], 1)), Xte_z])
        w = np.zeros(Xtr_.shape[1])
        for it in range(max_iter):
            z = Xtr_ @ w; p = 1.0/(1.0 + np.exp(-z)); W = p*(1.0-p) + 1e-9
            H = (Xtr_.T * W) @ Xtr_ + l2*np.eye(Xtr_.shape[1]); g = Xtr_.T @ (p - ytr) + l2*w
            try: step = np.linalg.solve(H, g)
            except np.linalg.LinAlgError: step = np.linalg.pinv(H) @ g
            w -= step
            if np.linalg.norm(step) < 1e-6: break
        scores = Xte_ @ w; aucs.append(_auc_from_scores(yte, scores))
    return float(np.nanmean(aucs)), float(np.nanstd(aucs))

def _bout_counts_per_frame(bouts_df, n_frames, fs):
    y_bin = np.zeros(n_frames, dtype=int); y_cnt = np.zeros(n_frames, dtype=float)
    if bouts_df is None or bouts_df.empty or "onset_time" not in bouts_df.columns: return y_bin, y_cnt
    ons = bouts_df["onset_time"].to_numpy(); fr = np.clip(np.floor(ons * fs).astype(int), 0, n_frames-1)
    for f in fr: y_bin[f] = 1; y_cnt[f] += 1.0
    return y_bin, y_cnt

def population_level_mapping(
    dff_smooth, fs=3.6, stim_frame=11, during_start=12, during_end=50, post_start=51,
    responsive_only=True, pre_df=None, during_df=None, post_df=None,
    k_sigma_height=2.5, k_sigma_prom=2.0, min_distance_s=0.5,
    n_pcs=5, lags_s=(2.0, 2.0), n_folds=5,
    tail_df=None, bouts_df=None, eta_window_s=(-2.0, 3.0),
    save_prefix="/mnt/data/population_mapping"
):
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
    df = _ensure_df(dff_smooth); n_frames, n_rois = df.shape
    windows = _class_windows(n_frames, stim_frame, during_start, during_end, post_start)
    resp_sets = _get_responsive_sets(df, fs, windows, pre_df, during_df, post_df, responsive_only, k_sigma_height, k_sigma_prom, min_distance_s)
    y_bin, y_cnt = _bout_counts_per_frame(bouts_df, n_frames, fs)
    ts_rows = []; encoding_rows = []; decoding_rows = []
    eta_pre, eta_post = eta_window_s; eta_len = int(round((eta_post - eta_pre) * fs)) + 1
    eta_times = np.arange(eta_len)/fs + eta_pre
    eta_popmean_acc = {k: [] for k in windows}; eta_pc1_acc = {k: [] for k in windows}
    for cls, (lo, hi) in windows.items():
        keep = resp_sets[cls] if responsive_only else list(df.columns)
        W = df.loc[lo:hi, keep].to_numpy(dtype=float); T = W.shape[0]
        pop_mean = np.nanmean(W, axis=1) if W.size else np.full(T, np.nan)
        if W.size and W.shape[1] > 0:
            S, vr = _pca_scores(W, n_components=n_pcs); k = min(n_pcs, S.shape[1]) if S.ndim==2 else 1
            scores = S[:, :k] if S.ndim==2 else S.reshape(-1,1)
        else:
            k = 0; scores = np.zeros((T, 0), dtype=float); vr = np.array([])
        t_window = np.arange(lo, hi+1)/fs
        score_cols = {f"pc{i+1}_score": (scores[:, i] if i < scores.shape[1] else np.full(T, np.nan)) for i in range(n_pcs)}
        ts_rows.append(pd.DataFrame({"class": cls, "time_s": t_window, "pop_mean": pop_mean, **score_cols}))
        if bouts_df is not None and not bouts_df.empty:
            bout_onsets = bouts_df["onset_time"].to_numpy(); w_lo_s, w_hi_s = lo/fs, (hi+1)/fs
            on_in = bout_onsets[(bout_onsets >= w_lo_s) & (bout_onsets < w_hi_s)]
            for bt in on_in:
                center_idx = int(round(bt*fs)) - lo; a0 = center_idx + int(round(eta_pre*fs)); a1 = a0 + eta_len
                if a0 < 0 or a1 > T: continue
                eta_popmean_acc[cls].append(pop_mean[a0:a1])
                if scores.shape[1] >= 1: eta_pc1_acc[cls].append(scores[a0:a1, 0])
        past, future = lags_s; lags = list(range(-int(round(past*fs)), int(round(future*fs))+1))
        Phi = _build_lagged(scores, lags) if scores.shape[1] > 0 else np.zeros((T,0))
        stim_reg = np.zeros(T, dtype=float); stim_lo = max(lo, during_start); stim_hi = min(hi, during_end)
        if stim_hi >= stim_lo: stim_reg[stim_lo-lo:stim_hi-lo+1] = 1.0
        Phi_enc = np.hstack([Phi, stim_reg.reshape(-1,1)]); y_cnt_win = y_cnt[lo:hi+1]
        if np.any(np.isfinite(Phi_enc)) and np.any(y_cnt_win > -np.inf): r2_mean, r2_std = _ols_cv(Phi_enc, y_cnt_win, n_folds=n_folds)
        else: r2_mean, r2_std = (np.nan, np.nan)
        encoding_rows.append({"class": cls, "n_rois_used": len(keep), "n_pcs": scores.shape[1], "lags": len(lags), "encoding_r2_mean": r2_mean, "encoding_r2_std": r2_std})
        y_bin_win = y_bin[lo:hi+1]; Phi_dec = Phi_enc.copy()
        if np.any(y_bin_win==1) and np.any(y_bin_win==0) and Phi_dec.shape[1] > 0: auc_mean, auc_std = _logreg_cv(Phi_dec, y_bin_win, n_folds=n_folds, l2=1e-2, max_iter=50)
        else: auc_mean, auc_std = (np.nan, np.nan)
        decoding_rows.append({"class": cls, "n_rois_used": len(keep), "n_pcs": scores.shape[1], "lags": len(lags), "decoder_auc_mean": auc_mean, "decoder_auc_std": auc_std})
    pop_timeseries = pd.concat(ts_rows, ignore_index=True) if ts_rows else pd.DataFrame()
    encoding_df = pd.DataFrame(encoding_rows).set_index("class").sort_index()
    decoding_df = pd.DataFrame(decoding_rows).set_index("class").sort_index()
    eta_rows = []
    eta_len = int(round((eta_window_s[1] - eta_window_s[0]) * fs)) + 1; eta_times = np.arange(eta_len)/fs + eta_window_s[0]
    for cls in windows:
        if len(eta_popmean_acc[cls]) > 0: eta_popmean = np.nanmean(np.vstack(eta_popmean_acc[cls]), axis=0)
        else: eta_popmean = np.full(eta_len, np.nan)
        if len(eta_pc1_acc[cls]) > 0: eta_pc1 = np.nanmean(np.vstack(eta_pc1_acc[cls]), axis=0)
        else: eta_pc1 = np.full(eta_len, np.nan)
        eta_rows.append(pd.DataFrame({"class": cls, "time_rel_s": eta_times, "eta_pop_mean": eta_popmean, "eta_pc1": eta_pc1}))
    eta_df = pd.concat(eta_rows, ignore_index=True) if eta_rows else pd.DataFrame()
    ts_csv = f"{save_prefix}_timeseries.csv"; enc_csv = f"{save_prefix}_encoding.csv"; dec_csv = f"{save_prefix}_decoding.csv"; eta_csv = f"{save_prefix}_eta.csv"
    pop_timeseries.to_csv(ts_csv, index=False); encoding_df.to_csv(enc_csv); decoding_df.to_csv(dec_csv); eta_df.to_csv(eta_csv, index=False)
    for cls in windows:
        sub = eta_df[eta_df["class"]==cls]
        if sub.empty: continue
        plt.figure(figsize=(6,3.2)); plt.plot(sub["time_rel_s"], sub["eta_pc1"], linewidth=1.5); plt.axvline(0.0, linestyle="--")
        plt.xlabel("time from bout onset (s)"); plt.ylabel("PC1 score (a.u.)"); plt.title(f"ETA PC1 ({cls})")
        plt.savefig(f"{save_prefix}_eta_pc1_{cls}.png", bbox_inches="tight"); plt.close()
    plt.figure(figsize=(5,3.2)); cls_order = list(encoding_df.index); plt.bar(np.arange(len(cls_order)), encoding_df["encoding_r2_mean"].to_numpy())
    plt.xticks(np.arange(len(cls_order)), cls_order); plt.ylabel("Encoding R² (mean CV)"); plt.title("Encoding performance by window")
    plt.savefig(f"{save_prefix}_encoding_bar.png", bbox_inches="tight"); plt.close()
    plt.figure(figsize=(5,3.2)); cls_order = list(decoding_df.index); plt.bar(np.arange(len(cls_order)), decoding_df["decoder_auc_mean"].to_numpy())
    plt.xticks(np.arange(len(cls_order)), cls_order); plt.ylabel("Decoder AUC (mean CV)"); plt.title("Decoding performance by window")
    plt.savefig(f"{save_prefix}_decoding_bar.png", bbox_inches="tight"); plt.close()
    return {"pop_timeseries": pop_timeseries, "encoding_df": encoding_df, "decoding_df": decoding_df, "eta_df": eta_df, "timeseries_csv": ts_csv, "encoding_csv": enc_csv, "decoding_csv": dec_csv, "eta_csv": eta_csv}
