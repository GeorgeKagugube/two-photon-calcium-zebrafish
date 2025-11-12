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


# def _deconv_one(trace: np.ndarray,
#                 method: str = "ar1",
#                 g: Optional[float] = None,
#                 s_min: float = 0.0,
#                 penalty: float = 1.0,
#                 optimize_g: bool = True):
#     deconvolve, estimate_parameters = _ensure_oasis()
#     x = np.asarray(trace, float)
#     if np.isnan(x).any():
#         med = np.nanmedian(x)
#         x = np.where(np.isfinite(x), x, med)

#     # estimate AR(1) parameter if requested
#     g_est, sn_est = (g, None)
#     if g is None and optimize_g and method.lower() == "ar1":
#         g_est, sn_est = estimate_parameters(x, p=1, method="logmexp")

#     if method.lower() == "ar1":
#         c, s, b, g_fit, lam = deconvolve(x, penalty=penalty, g=g_est, s_min=s_min)
#     elif method.lower() == "ar2":
#         c, s, b, g_fit, lam = deconvolve(x, penalty=penalty, g=g_est, s_min=s_min, method="ar2")
#     else:
#         raise ValueError("method must be 'ar1' or 'ar2'.")

#     return {"s": s, "c": c, "b": b, "g": g_fit, "sn": sn_est}

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

# import numpy as np, pandas as pd
# from typing import Optional, Tuple
# # optional progress / parallel
# try:
#     from joblib import Parallel, delayed
# except Exception:
#     Parallel = None; delayed = None
# try:
#     from tqdm import tqdm
# except Exception:
#     def tqdm(x, **k): return x

# # ---------- OASIS front-end ----------
# def _ensure_oasis():
#     try:
#         from oasis.functions import deconvolve, estimate_parameters
#         # quick smoke: some broken wheels import but lack compiled symbols
#         _ = deconvolve; _ = estimate_parameters
#     except Exception as e:
#         raise ImportError(f"OASIS unavailable: {e}")
#     return deconvolve, estimate_parameters

# def _oasis_deconv_one(x, method="ar1", g=None, s_min=0.0, penalty=1.0, optimize_g=True):
#     """
#     Version-robust OASIS wrapper:
#     - Handles old/new OASIS signatures (s_min vs smin vs none).
#     - Applies a post-hoc threshold if needed.
#     """
#     deconvolve, estimate_parameters = _ensure_oasis()
#     x = np.asarray(x, float)
#     if np.isnan(x).any():
#         med = np.nanmedian(x); x = np.where(np.isfinite(x), x, med)

#     # Estimate AR(1) parameter if requested
#     g_est, sn_est = (g, None)
#     if g is None and optimize_g and method.lower() == "ar1":
#         g_est, sn_est = estimate_parameters(x, p=1, method="logmexp")

#     # Call OASIS with robust handling of s_min/smin
#     def _call_deconv(**kwargs):
#         if method.lower() == "ar1":
#             return deconvolve(x, **kwargs)
#         elif method.lower() == "ar2":
#             return deconvolve(x, method="ar2", **kwargs)
#         else:
#             raise ValueError("method must be 'ar1' or 'ar2'")

#     try:
#         # Newer style (what I used first)
#         c, s, b, g_fit, lam = _call_deconv(penalty=penalty, g=g_est, s_min=s_min)
#     except TypeError:
#         try:
#             # Alternative OASIS kw name
#             c, s, b, g_fit, lam = _call_deconv(penalty=penalty, g=g_est, smin=s_min)
#         except TypeError:
#             # Oldest style: no s_min kw at all
#             c, s, b, g_fit, lam = _call_deconv(penalty=penalty, g=g_est)

#     # If we couldn’t pass s_min through, enforce it post hoc
#     if s_min and s is not None:
#         s = np.asarray(s, float)
#         s[s < s_min] = 0.0

#     return {"s": s, "c": c, "b": b, "g": g_fit, "sn": sn_est}

# # ---------- Fallback: non-negative kernel-LASSO deconv ----------
# # For your short traces (108 frames), a Toeplitz-convolution design + non-negative LASSO works well.
# # Requires scikit-learn (you already have it).
# from numpy.linalg import lstsq
# from sklearn.linear_model import Lasso

# def _exp_kernel(T: int, fs: float, tau_s: float) -> np.ndarray:
#     # causal AR(1)-like convolution kernel k[t] = exp(-t*dt/tau), t>=0
#     t = np.arange(T) / fs
#     k = np.exp(-t / max(tau_s, 1e-6))
#     k[0] = 1.0
#     return k

# def _design_from_kernel(T: int, k: np.ndarray) -> np.ndarray:
#     # build lower-triangular Toeplitz matrix so that C ≈ K @ s  (s>=0)
#     K = np.zeros((T, T), float)
#     for i in range(T):
#         L = T - i
#         K[i:, i] = k[:L]
#     return K

# def _nn_lasso_deconv_one(x: np.ndarray, fs: float, tau_grid=(0.4, 0.6, 0.8, 1.0, 1.4, 2.0), alpha=0.01):
#     """
#     Non-negative LASSO on convolution design:  min ||x - K s||^2 + alpha ||s||
#     Returns spikes s, denoised C = K s, baseline b (scalar), g ~ exp(-dt/tau*)
#     """
#     x = np.asarray(x, float)
#     if np.isnan(x).any():
#         med = np.nanmedian(x); x = np.where(np.isfinite(x), x, med)
#     T = x.size
#     best = None
#     dt = 1.0 / fs
#     for tau in tau_grid:
#         k = _exp_kernel(T, fs, tau)
#         K = _design_from_kernel(T, k)
#         # center x to absorb baseline; put it back later
#         xb = x - np.median(x)
#         # positive LASSO on s
#         lasso = Lasso(alpha=alpha, positive=True, fit_intercept=False, max_iter=2000)
#         lasso.fit(K, xb)
#         s = lasso.coef_
#         c = K @ s
#         resid = xb - c
#         rss = float((resid**2).sum())
#         if (best is None) or (rss < best[0]):
#             best = (rss, tau, s, c)
#     rss, tau_star, s_star, c_star = best
#     b = float(np.median(x) )  # simple baseline
#     g_equiv = float(np.exp(-dt / max(tau_star, 1e-6)))
#     return {"s": s_star, "c": c_star + b, "b": b, "g": g_equiv, "sn": None, "tau": tau_star}

# # ---------- Unified API ----------
# def deconvolve_dataframe(
#     dff: pd.DataFrame,
#     fs: float,
#     prefer_oasis: bool = True,
#     method: str = "ar1",
#     g: Optional[float] = None,
#     s_min: float = 0.0,
#     penalty: float = 1.0,
#     optimize_g: bool = True,
#     fallback_alpha: float = 0.01,
#     fallback_tau_grid = (0.4, 0.6, 0.8, 1.0, 1.4, 2.0),
#     n_jobs: int = 1,
#     show_progress: bool = True
# ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
#     """
#     Deconvolution for ΔF/F (rows=neurons, cols=frames).
#     Tries OASIS; if unavailable, uses non-negative kernel-LASSO fallback (grid over tau).
#     Returns S, C, B, G dataframes aligned to input.
#     """
#     if not isinstance(dff, pd.DataFrame):
#         raise TypeError("dff must be a DataFrame (rows=neurons, cols=frames).")
#     idx, cols = dff.index, dff.columns
#     X = dff.to_numpy(float)

#     use_oasis = False
#     if prefer_oasis:
#         try:
#             _ensure_oasis()
#             use_oasis = True
#         except Exception:
#             use_oasis = False

#     S = np.zeros_like(X); C = np.zeros_like(X); B = np.zeros_like(X); G = np.zeros((X.shape[0],), float)

#     def _work_row(xrow):
#         if use_oasis:
#             out = _oasis_deconv_one(xrow, method=method, g=g, s_min=s_min, penalty=penalty, optimize_g=optimize_g)
#         else:
#             out = _nn_lasso_deconv_one(xrow, fs=fs, tau_grid=fallback_tau_grid, alpha=fallback_alpha)
#         return out

#     # In case of broken OASIS inside subprocesses, force n_jobs=1
#     parallel_ok = (use_oasis is False) or (Parallel is not None and delayed is not None and n_jobs != 1)
#     if parallel_ok:
#         results = Parallel(n_jobs=n_jobs)(
#             delayed(_work_row)(X[i, :]) for i in (tqdm(range(X.shape[0]), desc="Deconvolution") if show_progress else range(X.shape[0]))
#         )
#     else:
#         results = []
#         it = tqdm(range(X.shape[0]), desc="Deconvolution") if show_progress else range(X.shape[0])
#         for i in it:
#             results.append(_work_row(X[i, :]))

#     for i, r in enumerate(results):
#         S[i, :] = r["s"]
#         C[i, :] = r["c"]
#         b = r["b"]; B[i, :] = b if np.ndim(b)==0 else np.asarray(b, float)
#         gfit = r["g"]; G[i] = float(gfit) if np.ndim(gfit)==0 else float(np.linalg.norm(np.asarray(gfit).ravel(), 2))

#     S_df = pd.DataFrame(S, index=idx, columns=cols)
#     C_df = pd.DataFrame(C, index=idx, columns=cols)
#     B_df = pd.DataFrame(B, index=idx, columns=cols)
#     G_df = pd.DataFrame(np.repeat(G[:, None], X.shape[1], axis=1), index=idx, columns=cols)
#     return S_df, C_df, B_df, G_df

