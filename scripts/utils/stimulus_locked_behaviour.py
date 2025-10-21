
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import List, Union, Dict, Optional, Tuple

def _ensure_bouts_list(bouts_or_list: Union[pd.DataFrame, List[pd.DataFrame]]) -> List[pd.DataFrame]:
    if isinstance(bouts_or_list, list):
        return [b for b in bouts_or_list if isinstance(b, pd.DataFrame)]
    elif isinstance(bouts_or_list, pd.DataFrame):
        return [bouts_or_list]
    else:
        raise TypeError("Input must be a bouts DataFrame or a list of bouts DataFrames.")

def _compute_bins(fs: float, total_frames: int, stim_frame: int):
    t0 = stim_frame / float(fs)
    total_time = total_frames / float(fs)
    pre = t0; post = total_time - t0
    edges = np.arange(-pre, post + 1.0/fs + 1e-9, 1.0/fs)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return edges, centers, t0

def _km_survival(latencies: np.ndarray, censored: np.ndarray, grid_s: Optional[np.ndarray] = None) -> pd.DataFrame:
    order = np.argsort(latencies)
    t = latencies[order]; c = censored[order].astype(int)
    if grid_s is None:
        grid = np.unique(t)
    else:
        grid = np.asarray(grid_s, float)
    S = 1.0
    surv_times=[]; surv_vals=[]; at_risk_list=[]; events_list=[]
    for g in grid:
        at_risk = np.sum(t >= g)
        d = np.sum((t == g) & (c == 0))
        if at_risk > 0 and d > 0:
            S *= (1.0 - d / at_risk)
        surv_times.append(g); surv_vals.append(S)
        at_risk_list.append(int(at_risk)); events_list.append(int(d))
    return pd.DataFrame({"time_s": surv_times, "survival": surv_vals, "n_at_risk": at_risk_list, "n_events": events_list})

def stimulus_locked_behaviour(
    bouts_trials: Union[pd.DataFrame, List[pd.DataFrame]],
    fs: float = 3.6,
    total_frames: int = 108,
    stim_frame: int = 11,
    response_window_s: Tuple[float, float] = (0.0, 2.0),
    save_prefix: str = "/mnt/data/stim_locked"
):
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)
    trials = _ensure_bouts_list(bouts_trials)
    n_trials = len(trials)
    edges, centers, t0 = _compute_bins(fs, total_frames, stim_frame)
    bin_dt = 1.0 / float(fs)
    bout_occurrences = np.zeros((n_trials, len(centers)), dtype=int)
    vigor_vals = [[] for _ in centers]
    window_lo, window_hi = response_window_s
    latencies = np.full(n_trials, np.nan, dtype=float)
    censored = np.zeros(n_trials, dtype=int)
    for i, df in enumerate(trials):
        if df is None or df.empty or "onset_time" not in df.columns:
            continue
        rel = df["onset_time"].to_numpy() - t0
        inds = np.digitize(rel, edges) - 1
        valid = (inds >= 0) & (inds < len(centers))
        if np.any(valid):
            bout_occurrences[i, np.unique(inds[valid])] = 1
        if "peak_speed_deg_s" in df.columns:
            for r, idx in zip(rel[valid], inds[valid]):
                vigor_vals[idx].append(float(df.loc[df["onset_time"] == (r + t0), "peak_speed_deg_s"].iloc[0]))
        mask_resp = (rel >= window_lo) & (rel <= window_hi)
        if np.any(mask_resp):
            latencies[i] = float(np.min(rel[mask_resp]))
            censored[i] = 0
        else:
            latencies[i] = float(window_hi); censored[i] = 1
    prob = bout_occurrences.mean(axis=0) if n_trials > 0 else np.zeros(len(centers))
    vigor_mean = np.array([np.nan if len(v)==0 else float(np.mean(v)) for v in vigor_vals], dtype=float)
    psth_df = pd.DataFrame({"time_rel_s": centers, "bout_prob": prob, "n_trials": n_trials})
    vigor_df = pd.DataFrame({"time_rel_s": centers, "mean_peak_speed_deg_s": vigor_mean})
    lat_clean = np.array([x if np.isfinite(x) and x >= 0 else window_hi for x in latencies], dtype=float)
    cen_clean = np.array([int(c) for c in censored], dtype=int)
    km_grid = np.arange(0.0, window_hi + bin_dt/2.0, bin_dt)
    km_df = _km_survival(lat_clean, cen_clean, grid_s=km_grid)
    med_latency = np.nan
    if not km_df.empty:
        below = km_df.loc[km_df["survival"] <= 0.5, "time_s"]
        if not below.empty:
            med_latency = float(below.iloc[0])
    def resp_prop(tcut):
        row = km_df.loc[np.isclose(km_df["time_s"], tcut, atol=bin_dt/2.0)]
        if row.empty:
            idx = int(np.argmin(np.abs(km_df["time_s"].to_numpy() - tcut)))
            s = float(km_df["survival"].iloc[idx])
        else:
            s = float(row["survival"].iloc[0])
        return max(0.0, min(1.0, 1.0 - s))
    prop_1s = resp_prop(1.0); prop_2s = resp_prop(2.0); prop_window = resp_prop(window_hi)
    latency_stats_df = pd.DataFrame([{"n_trials": n_trials, "median_latency_s": med_latency,
                                      "response_prob_1s": prop_1s, "response_prob_2s": prop_2s,
                                      "response_prob_window": prop_window, "censor_window_s": window_hi}])
    psth_csv = f"{save_prefix}_psth_prob.csv"
    vigor_csv = f"{save_prefix}_psth_vigor.csv"
    km_csv = f"{save_prefix}_latency_km.csv"
    lat_csv = f"{save_prefix}_latency_stats.csv"
    psth_df.to_csv(psth_csv, index=False); vigor_df.to_csv(vigor_csv, index=False)
    km_df.to_csv(km_csv, index=False); latency_stats_df.to_csv(lat_csv, index=False)
    plt.figure(figsize=(6,3.2))
    plt.plot(psth_df["time_rel_s"], psth_df["bout_prob"], linewidth=1.5)
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("time from stimulus (s)"); plt.ylabel("bout probability per frame")
    plt.title("Stimulus-locked bout probability")
    psth_png = f"{save_prefix}_psth_prob.png"; plt.savefig(psth_png, bbox_inches="tight"); plt.close()
    plt.figure(figsize=(6,3.2))
    plt.plot(vigor_df["time_rel_s"], vigor_df["mean_peak_speed_deg_s"], linewidth=1.5)
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("time from stimulus (s)"); plt.ylabel("mean peak speed (deg/s)")
    plt.title("Stimulus-locked bout vigor")
    vigor_png = f"{save_prefix}_psth_vigor.png"; plt.savefig(vigor_png, bbox_inches="tight"); plt.close()
    plt.figure(figsize=(5,3.2))
    plt.step(km_df["time_s"], km_df["survival"], where="post", linewidth=1.5)
    plt.axvline(1.0, linestyle=":"); plt.axvline(2.0, linestyle=":")
    plt.xlabel("time to first bout (s)"); plt.ylabel("survival (no bout yet)")
    plt.title("Latency to first bout (KM)")
    km_png = f"{save_prefix}_latency_km.png"; plt.savefig(km_png, bbox_inches="tight"); plt.close()
    return {"psth_df": psth_df, "vigor_df": vigor_df, "km_df": km_df, "latency_stats_df": latency_stats_df,
            "psth_csv": psth_csv, "vigor_csv": vigor_csv, "km_csv": km_csv, "latency_stats_csv": lat_csv,
            "psth_png": psth_png, "vigor_png": vigor_png, "km_png": km_png}
