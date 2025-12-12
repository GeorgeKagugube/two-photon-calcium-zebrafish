## Import the required libraries here
import numpy as np
import pandas as pd
from typing import Optional, Tuple

# optional speedups / progress
try:
    from joblib import Parallel, delayed
except Exception:
    Parallel = None; delayed = None

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

def _ensure_oasis():
    try:
        from oasis.functions import deconvolve, estimate_parameters
    except Exception as e:
        raise ImportError(
            "OASIS not found. Install with:  pip install oasis\n"
            f"Original import error: {e}"
        )
    return deconvolve, estimate_parameters

def _deconv_one(x, method="ar1", g=None, s_min=0.0, penalty=1.0, optimize_g=True):
    """
    Version-robust OASIS wrapper:
    - Handles old/new OASIS signatures (s_min vs smin vs none).
    - Applies a post-hoc threshold if needed.
    """
    deconvolve, estimate_parameters = _ensure_oasis()
    x = np.asarray(x, float)
    if np.isnan(x).any():
        med = np.nanmedian(x); x = np.where(np.isfinite(x), x, med)

    # Estimate AR(1) parameter if requested
    g_est, sn_est = (g, None)
    if g is None and optimize_g and method.lower() == "ar1":
        g_est, sn_est = estimate_parameters(x, p=1, method="logmexp")

    # Call OASIS with robust handling of s_min/smin
    def _call_deconv(**kwargs):
        if method.lower() == "ar1":
            return deconvolve(x, **kwargs)
        elif method.lower() == "ar2":
            return deconvolve(x, method="ar2", **kwargs)
        else:
            raise ValueError("method must be 'ar1' or 'ar2'")

    try:
        # Newer style (what I used first)
        c, s, b, g_fit, lam = _call_deconv(penalty=penalty, g=g_est, s_min=s_min)
    except TypeError:
        try:
            # Alternative OASIS kw name
            c, s, b, g_fit, lam = _call_deconv(penalty=penalty, g=g_est, smin=s_min)
        except TypeError:
            # Oldest style: no s_min kw at all
            c, s, b, g_fit, lam = _call_deconv(penalty=penalty, g=g_est)

    # If we couldn’t pass s_min through, enforce it post hoc
    if s_min and s is not None:
        s = np.asarray(s, float)
        s[s < s_min] = 0.0

    return {"s": s, "c": c, "b": b, "g": g_fit, "sn": sn_est}

def deconvolve_oasis_dataframe(
    dff: pd.DataFrame,
    method: str = "ar1",
    g: Optional[float] = None,
    s_min: float = 0.0,
    penalty: float = 1.0,
    optimize_g: bool = True,
    n_jobs: int = 1,
    show_progress: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    OASIS deconvolution for ΔF/F data with rows=neurons and columns=frames.
    Returns (S_df, C_df, B_df, G_df): spikes, denoised calcium, baseline, AR param.
    """
    if not isinstance(dff, pd.DataFrame):
        raise TypeError("dff must be a pandas DataFrame (rows=neurons, cols=frames).")
    if dff.shape[0] < 1 or dff.shape[1] < 2:
        raise ValueError("Need at least 1 neuron and 2 frames.")

    idx, cols = dff.index, dff.columns
    S = np.zeros_like(dff.values, dtype=float)
    C = np.zeros_like(dff.values, dtype=float)
    B = np.zeros_like(dff.values, dtype=float)
    G = np.zeros((dff.shape[0],), dtype=float)

    traces = [dff.iloc[i, :].to_numpy(float) for i in range(dff.shape[0])]

    def _work(x):
        return _deconv_one(x, method=method, g=g, s_min=s_min, penalty=penalty, optimize_g=optimize_g)

    if (n_jobs != 1) and (Parallel is not None) and (delayed is not None):
        results = Parallel(n_jobs=n_jobs)(
            delayed(_work)(x) for x in (tqdm(traces, desc="OASIS deconvolution") if show_progress else traces)
        )
    else:
        results = []
        it = tqdm(traces, desc="OASIS deconvolution") if show_progress else traces
        for x in it: results.append(_work(x))

    for i, r in enumerate(results):
        S[i, :] = r["s"]
        C[i, :] = r["c"]
        b = r["b"]
        B[i, :] = float(b) if np.ndim(b) == 0 else np.asarray(b, float)
        gfit = r["g"]
        G[i] = float(gfit) if np.ndim(gfit) == 0 else float(np.linalg.norm(np.asarray(gfit).ravel(), ord=2))

    S_df = pd.DataFrame(S, index=idx, columns=cols)
    C_df = pd.DataFrame(C, index=idx, columns=cols)
    B_df = pd.DataFrame(B, index=idx, columns=cols)
    # replicate g per frame for alignment convenience
    G_df = pd.DataFrame(np.repeat(G[:, None], dff.shape[1], axis=1), index=idx, columns=cols)
    return S_df, C_df, B_df, G_df

def save_oasis_outputs(S_df: pd.DataFrame, C_df: pd.DataFrame, B_df: pd.DataFrame, G_df: pd.DataFrame, prefix: str):
    paths = {
        "spikes_csv": f"{prefix}_spikes_oasis.csv",
        "denoisedC_csv": f"{prefix}_denoisedC_oasis.csv",
        "baseline_csv": f"{prefix}_baseline_oasis.csv",
        "gparam_csv": f"{prefix}_gparam_oasis.csv",
    }
    S_df.to_csv(paths["spikes_csv"])
    C_df.to_csv(paths["denoisedC_csv"])
    B_df.to_csv(paths["baseline_csv"])
    G_df.to_csv(paths["gparam_csv"])
    return paths
