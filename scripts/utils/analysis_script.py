import os
import mat73
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy.signal import butter, filtfilt, find_peaks
from mat73 import loadmat as ldt

import matplotlib.pyplot as plt
from scipy.stats import zscore
from functools import lru_cache

#from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable

from scipy.signal import welch
from scipy.optimize import curve_fit
from scipy.stats import skew, kurtosis

## Load personal written functions here
# from compute_dff import compute_dff
# from denoise_dff import denoise_dff
# from detect_events import detect_events
# from detect_events_stim_aligned import detect_events_stim_aligned
# from summarize_event_metrics_by_class import summarize_event_metrics_by_class
# from compute_tonic_metrics import compute_tonic_metrics
# from compute_clearance_metrics import clearance_metrics_by_class
# from excitability_metrics_by_class import excitability_metrics_by_class
# from network_reorg import network_reorg_metrics_and_plots
# from extract_rois import extract_signal_by_stimulus
# from network_reorg_metrics_and_plots_responsive import network_reorg_metrics_and_plots_responsive
from extract_F_from_mat_struct import extract_F_from_mat_struct, _moving_percentile, _moving_average, debug_plot_dff_examples
from compute_sensory_component import compute_sensory_component, fit_motor_regression_elastic_net
from build_regressors_from_eye_tail import extract_eye_traces_from_dff, extract_tail_traces_from_dff, build_regressors_from_eye_tail


## Figure formatting starts here
sns.set_theme(style="darkgrid")
plt.rcParams['font.family'] = 'DejaVu Sans'

## Silence the warnings about NAN later in the notebook
## They do not affect the running of the script but continue to overload the 
## notebook with lengthy unhelpful warnings
## I will fix these later
import warnings
warnings.filterwarnings('ignore')

## ============================= Helper functions for later start from here =========================
@lru_cache(maxsize=50)
def import_2p_data(path_to_file_of_interest):
    df = mat73.loadmat(path_to_file_of_interest)
    return df

##-------------------------------------GCaMP7s signal data ----------------------------
## Path to files
wt1 = "/Users/gwk/Desktop/CalciumImaging/RawData/WT_Unexposed/Fish1/gmrxanat.mat"
wt2 = "/Users/gwk/Desktop/CalciumImaging/RawData/WT_Unexposed/Fish2/gmrxanat.mat"
wt3 = "/Users/gwk/Desktop/CalciumImaging/RawData/WT_Unexposed/Fish2/gmrxanat.mat"

# ## WT Exposed
# wte1 = "/Users/gwk/Desktop/CalciumImaging/RawData/WT_exposed/Fish1/gmrxanat.mat"

# ## Mutant Unexposed
# mutUexpo = "/Users/gwk/Desktop/CalciumImaging/RawData/Mut_Unexposed/Fish1/gmrxanat.mat"

# ## Mutant Exposed
# mute1 = "/Users/gwk/Desktop/CalciumImaging/RawData/Mut_Exposed/Fish1/gmrxanat.mat"
# mute2 = "/Users/gwk/Desktop/CalciumImaging/RawData/Mut_Exposed/Fish2/gmrxanat.mat"
# mute3 = "/Users/gwk/Desktop/CalciumImaging/RawData/Mut_Exposed/Fish3/gmrxanat.mat"
# mute4 = "/Users/gwk/Desktop/CalciumImaging/RawData/Mut_Exposed/Fish4/gmrxanat.mat"
# mute5 = "/Users/gwk/Desktop/CalciumImaging/RawData/Mut_Exposed/Fish5/gmrxanat.mat"

#### ---------------------- The eye movement data ---------------------------------------
## Path to files
wt1_eye = "/Users/gwk/Desktop/CalciumImaging/RawData/WT_Unexposed/Fish1/gmb.mat"
wt2_eye = "/Users/gwk/Desktop/CalciumImaging/RawData/WT_Unexposed/Fish2/gmb.mat"
wt3_eye = "/Users/gwk/Desktop/CalciumImaging/RawData/WT_Unexposed/Fish2/gmb.mat"

# ## WT Exposed
# wte1_eye = "/Users/gwk/Desktop/CalciumImaging/RawData/WT_exposed/Fish1/gmb.mat"

# ## Mutant Unexposed
# mutUexpo_eye = "/Users/gwk/Desktop/CalciumImaging/RawData/Mut_Unexposed/Fish1/gmb.mat"

# ## Mutant Exposed
# mute1_eye = "/Users/gwk/Desktop/CalciumImaging/RawData/Mut_Exposed/Fish1/gmb.mat"
# mute2_eye = "/Users/gwk/Desktop/CalciumImaging/RawData/Mut_Exposed/Fish2/gmb.mat"
# mute3_eye = "/Users/gwk/Desktop/CalciumImaging/RawData/Mut_Exposed/Fish3/gmb.mat"
# mute4_eye = "/Users/gwk/Desktop/CalciumImaging/RawData/Mut_Exposed/Fish4/gmb.mat"
# mute5_eye = "/Users/gwk/Desktop/CalciumImaging/RawData/Mut_Exposed/Fish5/gmb.mat"

####--------------------------- The Tail movement data ----------------------------------
## Path to files
wt1_tail = "/Users/gwk/Desktop/CalciumImaging/RawData/WT_Unexposed/Fish1/gmbt.mat"
# wt2_tail = "/Users/gwk/Desktop/CalciumImaging/RawData/WT_Unexposed/Fish2/gmbt.mat"
# wt3_tail = "/Users/gwk/Desktop/CalciumImaging/RawData/WT_Unexposed/Fish2/gmbt.mat"

# ## WT Exposed
# wte1_tail = "/Users/gwk/Desktop/CalciumImaging/RawData/WT_exposed/Fish1/gmbt.mat"

# ## Mutant Unexposed
# mutUexpo_tail = "/Users/gwk/Desktop/CalciumImaging/RawData/Mut_Unexposed/Fish1/gmbt.mat"

# ## Mutant Exposed
# mute1_tail = "/Users/gwk/Desktop/CalciumImaging/RawData/Mut_Exposed/Fish1/gmbt.mat"
# mute2_tail = "/Users/gwk/Desktop/CalciumImaging/RawData/Mut_Exposed/Fish2/gmbt.mat"
# mute3_tail = "/Users/gwk/Desktop/CalciumImaging/RawData/Mut_Exposed/Fish3/gmbt.mat"
# mute4_tail = "/Users/gwk/Desktop/CalciumImaging/RawData/Mut_Exposed/Fish4/gmbt.mat"
# mute5_tail = "/Users/gwk/Desktop/CalciumImaging/RawData/Mut_Exposed/Fish5/gmbt.mat"

##################
## Wild Type Unexposed
wt_unexp1 = import_2p_data(wt1)
# wt_unexp2 = import_2p_data(wt2)
# wt_unexp3 = import_2p_data(wt3)

# Eye data 
gmb_wt1 = import_2p_data(wt1_eye)
# gmb_wt2 = import_2p_data(wt2_eye)
# gmb_wt3 = import_2p_data(wt3_eye)

# Tail data
gmbt_wt1 = import_2p_data(wt1_tail)
# gmbt_wt2 = import_2p_data(wt2_tail)
# gmb_wt3 = import_2p_data(wt3_tail)

# ## Wild Type Exposed
# wt_exp = import_2p_data(wte1)

# # Eye
# gmb_wte = import_2p_data(wte1_eye)

# # Tail 
# gmbt_wte = import_2p_data(wte1_tail)

# # ## Mutant unexposed
# # mut_unexpo = import_2p_data(mutUnexpo)

# # Eye 
# gmb_mt = import_2p_data(mutUexpo_eye)

# # Tail 
# gmbt_mut = import_2p_data(mutUexpo_tail)

# ## mutant exposed
# mut_exp1 = import_2p_data(mute1)
# mut_exp2 = import_2p_data(mute2)
# mut_exp3 = import_2p_data(mute3)
# mut_exp4 = import_2p_data(mute4)
# mut_exp5 = import_2p_data(mute5)

# # Eye 
# gmb_mt1 = import_2p_data(mute1_eye)
# gmb_mt2 = import_2p_data(mute2_eye)
# gmb_mt3 = import_2p_data(mute3_eye)
# gmb_mt4 = import_2p_data(mute4_eye)
# gmb_mt5 = import_2p_data(mute5_eye)

# # Tail 
# gmbt_mt1 = import_2p_data(mute1_tail)
# gmbt_mt2 = import_2p_data(mute2_tail)
# gmbt_mt3 = import_2p_data(mute3_tail)
# gmbt_mt4 = import_2p_data(mute4_tail)
# gmbt_mt5 = import_2p_data(mute5_tail)


######################
## Wild Type Unexposed
F_dff = extract_F_from_mat_struct(
    wt_unexp1['gmrxanat'],
    compute_dff=True,
    smooth_window=3,
    baseline_window=30,
    baseline_percentile=20.0,
)

stim_type_to_duration_s = {
    "OKR": 12.00,
    "BAR": 10.00,
    "LOOM": 5.66,
}

### ==============Perform some sanity check at this stage ==================
examples = [(0, 10), (1, 10), (2, 10), (3, 10),(4, 42)]

figs = debug_plot_dff_examples(
    wt_unexp1['gmrxanat'],
    examples=examples,
    smooth_window=3,
    baseline_window=30,
    baseline_percentile=20.0,
    # time_axis=your_time_vector_if_you_have_it
)

#### ==================== Compute the sensory component ===================================
stim_ids = np.array(['dark_flash','okr_0.4_12.14','okr_0.4_39.68','okr_0.08_12.14','okr_0.2_12.14','okr_0.4_26.14',
                    'lmb_0_4','lmb_2_12','lmb_0_12','lmb_2_4','rmb_0_4','rmb_2_12','rmb_0_12','rmb_2_4','loomf','loomcf', 'bright_flash'])

#include_mask = np.ones(len(stim_id) , dtype=bool)

n_trials = F_dff.shape[0]
onset_frame_scalar = 12  # or whatever the true onset frame is
onset_frames = np.full(n_trials, onset_frame_scalar, dtype=int)
include_mask = np.ones(n_trials, dtype=bool)  # use all trials, if that's what you want

# Map each stim_id → duration_s according to its type
stim_duration_map_s = {}
for stim in np.unique(stim_ids):
    # Example logic: stim IDs start with type prefix
    if str(stim).startswith("okr"):
        stim_duration_map_s[stim] = 10.0
    elif str(stim).startswith("mb"):
        stim_duration_map_s[stim] = 12.0
    elif str(stim).startswith("loom"):
        stim_duration_map_s[stim] = 5.6
    else:
        stim_duration_map_s[stim] = 3.00

sensory_df = compute_sensory_component(
    F=F_dff,
    stim_ids=stim_ids,
    onset_frames=onset_frames,
    include_mask=include_mask,
    fs=3.6,
    stim_duration_map_s=stim_duration_map_s,
)

### ========================================================================================
fs_imaging = 3.6  # Hz

eye_df = extract_eye_traces_from_dff(gmb_wt1, fs_imaging=fs_imaging)
tail_df = extract_tail_traces_from_dff(gmbt_wt1, fs_imaging=fs_imaging)

regressors_df = build_regressors_from_eye_tail(
    eye_df=eye_df,
    tail_df=tail_df,
    fs=fs_imaging,
    ipsi_side="left",  # or "right", depending on which tectum/pretectum you imaged
    vergence_thresh=12.0,        # adjust based on your vergence histogram
    saccade_vel_thresh=60.0,     # adjust based on your eye velocity distribution
    tail_bout_angle_thresh=5.0,  # adjust based on your tail angle distribution
    tail_vigour_smooth_frames=3,
    motion_error_series=None,    # or your per-frame motion metric
)
## ---------------------------------------------------------------------
# fs_imaging = 3.6  # Hz
# lag_frames = 3    # for example; tune later

# neuron_ids = np.arange(F_neurons.shape[0])

# motor_coeffs_df, metrics_df = fit_motor_regression_elastic_net(
#     F=F_neurons,
#     regressors_df=regressors_df,
#     fs=fs_imaging,
#     lag_frames=lag_frames,
#     neuron_ids=neuron_ids,
# )

## =====================================
# motor_cols = [
#     "LEye_ipsi_pos", "LEye_ipsi_vel",
#     "REye_ipsi_pos", "REye_ipsi_vel",
#     "Conv_tail_sym", "Conv_tail_L", "Conv_tail_R",
#     "Tail_bout_L", "Tail_bout_R",
#     "Tail_vigour", "Motion_error",
# ]

# vmv_df, vmv_norm_df, vmv_norm_matrix = prepare_vmv_for_clustering(
#     sensory_df=sensory_df,         # from compute_sensory_component
#     motor_coeffs_df=motor_coeffs_df,
#     motor_cols=motor_cols,
# )

## Inspect the dataframe here
print('Printing the vmv_df 10 rows')
print(F_dff.shape)

print ('Printing the vmv_norm')
print(sensory_df.head(10))

print(regressors_df.head(10))



































