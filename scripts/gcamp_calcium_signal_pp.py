import mat73
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from functools import lru_cache

## Figure formatting starts here
sns.set_theme(style="darkgrid")
plt.rcParams['font.family'] = 'DejaVu Sans'

## --- Smooth signals ---
def smooth_signal(row, sigma=1.0):
    return pd.Series(gaussian_filter1d(row.values, sigma=sigma), index=row.index)

df_smoothed = df.apply(smooth_signal, axis=1)

# --- ΔF/F₀ calculation (20th percentile baseline) ---
def compute_dff(row):
    f0 = np.percentile(row.values, 20)
    return pd.Series((row - f0) / f0 if f0 != 0 else np.zeros_like(row), index=row.index)

df_dff = df_smoothed.apply(compute_dff, axis=1)

# --- Z-score normalization ---
df_zscore = df_dff.apply(zscore, axis=1)

# --- Activity threshold ---
activity_threshold = 2.5
active_coords = np.where(df_zscore >= activity_threshold)

# --- Event detection and quantification ---
frame_duration_sec = 30
peak_height = 0.2
min_distance = 5
event_records = []

for neuron_idx, signal_row in df_dff.iterrows():
    signal = signal_row.values
    baseline_f0 = np.percentile(signal, 20)
    
    peaks, _ = find_peaks(signal, height=peak_height, distance=min_distance)
    
    for i, peak in enumerate(peaks):
        peak_value = signal[peak]
        threshold = baseline_f0 + 0.1 * (peak_value - baseline_f0)

        end = peak + 1
        while end < len(signal) and signal[end] > threshold:
            end += 1

        if end < len(signal):
            duration_frames = end - peak
            duration_sec = duration_frames * frame_duration_sec
            delta_signal = signal[end] - signal[peak]
            slope = delta_signal / duration_sec
            auc = np.trapz(signal[peak:end]) if end > peak else np.nan
        else:
            duration_frames = np.nan
            duration_sec = np.nan
            slope = np.nan
            auc = np.nan

        event_records.append({
            'neuron_id': neuron_idx,
            'event_id': i + 1,
            'frame_peak': peak,
            'amplitude': peak_value,
            'baseline': baseline_f0,
            'duration_frames_90': duration_frames,
            'duration_sec_90': duration_sec,
            'AUC': auc,
            'repolarization_slope_per_sec': slope
        })

# --- Combine events into a DataFrame ---
event_df = pd.DataFrame(event_records)

# --- Aggregate data to return mean activity per a neurone 
# --- Per-neuron average metrics ---
per_neuron_summary = (
    event_df
    .groupby('neuron_id')
    .agg({
        'amplitude': 'mean',
        'duration_frames_90': 'mean',
        'duration_sec_90': 'mean',
        'AUC': 'mean',
        'repolarization_slope_per_sec': 'mean',
        'event_id': 'count'
    })
    .rename(columns={'event_id': 'event_count'})
    .reset_index()
)

# --- Summary statistics ---
print("\n🔎 Population Summary:\n")
print(per_neuron_summary[['amplitude', 'duration_sec_90', 'AUC', 'repolarization_slope_per_sec']].describe())

#event_df.to_csv('calcium_event_summary_all_neurons.csv', index=False)
