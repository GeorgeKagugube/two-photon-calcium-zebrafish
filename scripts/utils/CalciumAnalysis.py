import pandas as pd
from gcamp_epoch_flex import build_outputs

roi_matrix = pd.read_csv("my_fluorescence_matrix.csv")   # frames × ROIs
res = build_outputs(
    roi_input=roi_matrix,
    stim_frame=11,
    frames_per_epoch=108,
    fps=108/30.0,                # or omit to infer from epoch_seconds=30
    behavior_input={"tail": tail_vec},   # optional
    use_deconvolution=True,
    auto_tune_deconv=True,       # <- per-ROI auto-tuning enabled
    lam_factors=[0.25, 0.5, 1, 2, 4],  # optional override
)

# Tuned OASIS parameters per ROI:
tuning = res["deconv"]["tuning"]   # columns: roi, sn, lam, g, bic, k_spikes
S_hat  = res["deconv"]["S_hat"]    # (E, F, N) deconvolved spikes
