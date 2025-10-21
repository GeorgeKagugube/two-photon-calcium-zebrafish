#!/Users/gwk/anaconda3/envs/gcamp_analysis/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sys import path
#path.append('..')
from oasis.functions import  gen_sinusoidal_data, deconvolve, estimate_parameters
from oasis.plotting import simpleaxis
from oasis.oasis_methods import oasisAR1, oasisAR2
from functools import lru_cache
import mat73


def extract_signal_by_stimulus(data, stim=1):
    '''
        This function extracts the activity by frame for each stimulus presented to each neuron. 
        Inputes: gmrxanat from the Bianco preprocessing pipeline
        output: a 2d matrix of nuerones by stimulus presented across all the frames
    '''
    nuerones = []
    for roi in range(0,len(data['gmrxanat']['roi'])):
            nuerones.append(data['gmrxanat']['roi'][roi]['Vprofiles']['meanprofile'][stim])
    return (nuerones)

# --- Cell 3: ΔF/F computation ---
def compute_dff(trace, fs, baseline_win_s=10.0, baseline_percentile=20):
    """
    Convert raw fluorescence trace to ΔF/F0 using running percentile baseline.
    """
    w = int(max(3, baseline_win_s * fs))
    F0 = np.zeros_like(trace)
    half = w // 2
    for i in range(len(trace)):
        s = max(0, i-half)
        e = min(len(trace), i+half+1)
        F0[i] = np.percentile(trace[s:e], baseline_percentile)
    dff = (trace - F0) / (F0 + 1e-9)
    return dff, F0

# --- Cell 4: Deconvolution function ---
def deconvolve_trace(dff, Fs, tau=0.9):
    """
    Deconvolve ΔF/F using OASIS if available, else AR(1) approximation.
    tau : decay constant in seconds
    Returns:
        c_hat : estimated calcium
        s_hat : estimated spikes
    """
    dt = 1.0 / Fs
    g = np.exp(-dt / tau)

    try:
        from oasis.functions import deconvolve
        c_hat, s_hat, b, g_fit, lam = deconvolve(
            dff, g=[g], smin=0, optimize_g=0, penalty=1
        )
        print("OASIS deconvolution used.")
    except Exception as e:
        print("OASIS not available, using AR(1) surrogate:", e)
        dff_shift = np.r_[0, dff[:-1]]
        s_hat = np.maximum(0.0, dff - g * dff_shift)
        c_hat = np.zeros_like(dff)
        for t in range(1, len(dff)):
            c_hat[t] = g * c_hat[t-1] + s_hat[t]
    return c_hat, s_hat

# --- Load your GCaMP7s DataFrame ---
# Example: df = pd.read_csv('calcium_data.csv', index_col=0)
## Path to files
wt = "/Users/gwk/Desktop/CalciumImaging/Wt_Unexposed/f1/gmrxanat.mat"
wt7dpf = "/Users/gwk/Desktop/CalciumImaging/Wt_Unexposed/f2/gmrxanat.mat"
wte = "/Users/gwk/Desktop/CalciumImaging/Wt_exposed/f2/gmrxanat.mat"
mute = "/Users/gwk/Desktop/CalciumImaging/Mut_exposed/f1/gmrxanat.mat"
mute7dpf1 = "/Users/gwk/Desktop/CalciumImaging/Mut_exposed/7_dpf/f1/gmrxanat.mat"
mute7dpf3 = "/Users/gwk/Desktop/CalciumImaging/Mut_exposed/7_dpf/f3/gmrxanat.mat"
mute7dpf2 = "/Users/gwk/Desktop/CalciumImaging/Mut_exposed/7_dpf/f2/gmrxanat.mat"


@lru_cache(maxsize=50)
def import_2p_data(path_to_file_of_interest):
    df = mat73.loadmat(path_to_file_of_interest)
    return df

wt_unexp = import_2p_data(wt)
#wt_unexp2 = import_2p_data(wt7dpf)
#wt_exp = import_2p_data(wte)
#mut_exp = import_2p_data(mute7dpf2)
#mut_exp2 = import_2p_data(mute7dpf1)
#mut_exp3 = import_2p_data(mute7dpf2)
#mut_exp4 = import_2p_data(mute7dpf3)

## loaded data
unexposed_wt = extract_signal_by_stimulus(wt_unexp, stim=11)
#exposed_wt = extract_signal_by_stimulus(wt_exp, stim=11)
#exposed_mut = extract_signal_by_stimulus(mut_exp2, stim=11)  

roi_1 = pd.DataFrame(list(range(108)))
roi_1['roi_1'] = pd.DataFrame(unexposed_wt[0])
roi_1.rename(columns={0:'frames', 'roi_1':'cell1'})

print(roi_1.head())

frames = roi_1.iloc[:,0].values
raw_trace = roi_1.iloc[:,1].values
Fs = 3.6  # Hz imaging rate
time = frames / Fs

dff, F0 = compute_dff(raw_trace, Fs)

c_hat, s_hat = deconvolve_trace(dff, Fs, tau=0.9)

#c, s, b, g, lam = deconvolve(roi_1['roi_1'], penalty=1)
plot_trace(True)

def plot_trace(groundtruth=False):
    plt.figure(figsize=(20,4))
    plt.subplot(211)
    plt.plot(b+c, lw=2, label='denoised')
    if groundtruth:
        plt.plot(true_b+true_c, c='r', label='truth', zorder=-11)
    plt.plot(y, label='data', zorder=-12, c='y')
    plt.legend(ncol=3, frameon=False, loc=(.02,.85))
    simpleaxis(plt.gca())
    plt.subplot(212)
    plt.plot(s, lw=2, label='deconvolved', c='g')
    if groundtruth:
        for k in np.where(true_s)[0]:
            plt.plot([k,k],[-.1,1], c='r', zorder=-11, clip_on=False)
    plt.ylim(0,1.3)
    plt.legend(ncol=3, frameon=False, loc=(.02,.85));
    simpleaxis(plt.gca())
    print("Correlation of deconvolved activity  with ground truth ('spikes') : %.4f" % np.corrcoef(s,true_s)[0,1])
    print("Correlation of denoised fluorescence with ground truth ('calcium'): %.4f" % np.corrcoef(c,true_c)[0,1])

