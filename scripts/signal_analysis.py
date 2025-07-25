#!/Users/gwk/anaconda3/envs/gcamp_analysis/bin/python

## Import the required libraries for the analysis
import pandas as pd
import numpy as np
from oasis.functions import deconvolve
import matplotlib.pyplot as plt
import mat73
from scipy.stats import zscore
from functools import lru_cache
from oasis.functions import deconvolve
from utils.analyze_responsive import full_neuron_event_analysis
from scipy.ndimage import percentile_filter

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

# --- Load your GCaMP7s DataFrame ---
# Example: df = pd.read_csv('calcium_data.csv', index_col=0)
wt = "/Users/gwk/Desktop/CalciumImaging/Wt_Unexposed/f1/gmrxanat.mat"
import numpy as np
import matplotlib.pyplot as plt
from oasis.functions import deconvolve

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

data = extract_signal_by_stimulus(wt_unexp, stim=10)



# Simulated data (replace with your actual data)
event_times = np.array([11])  # Replace with actual event frame indices
pre_event = 10
post_event = 20

# Parameters
spike_threshold = 2.50
window = 10
percentile = 10

responsive_indices = []
delta_f_over_f = []
aligned_responses = []

for i, trace in enumerate(data):
    c, s, _, _, _ = deconvolve(trace, penalty=1)
    if np.sum(s) >= spike_threshold:
        responsive_indices.append(i)
        
        f0 = percentile_filter(trace, percentile=percentile, size=window, mode='reflect')
        dff = (trace - f0) / f0
        delta_f_over_f.append(dff)
        
        # Event alignment
        aligned_trials = []
        for event in event_times:
            start = event - pre_event
            end = event + post_event
            if start >= 0 and end <= len(trace):
                aligned_trials.append(dff[start:end])
        if aligned_trials:
            avg_response = np.mean(aligned_trials, axis=0)
            aligned_responses.append(avg_response)

# Convert to DataFrames
dff_df = pd.DataFrame(delta_f_over_f, index=[f"Neuron_{i}" for i in responsive_indices])
aligned_df = pd.DataFrame(aligned_responses, index=[f"Neuron_{i}" for i in responsive_indices])

# Raster of ΔF/F₀
plt.figure(figsize=(12, 5))
plt.imshow(dff_df, aspect='auto', cmap='hot', interpolation='none')
plt.colorbar(label='ΔF/F₀')
plt.title("Raster of ΔF/F₀ (Responsive Neurons)")
plt.xlabel("Frame")
plt.ylabel("Neuron")
plt.yticks(ticks=np.arange(len(responsive_indices)), labels=[f"N{i}" for i in responsive_indices])
plt.tight_layout()
plt.show()

# Heatmap of event-aligned ΔF/F₀ responses
plt.figure(figsize=(12, 5))
plt.imshow(aligned_df, aspect='auto', cmap='viridis', interpolation='none',
           extent=[-pre_event, post_event, 0, len(responsive_indices)])
plt.colorbar(label='ΔF/F₀ (Aligned Avg)')
plt.title("Event-Aligned ΔF/F₀ Responses")
plt.xlabel("Time (frames from event)")
plt.ylabel("Neuron")
plt.yticks(ticks=np.arange(len(responsive_indices)), labels=[f"N{i}" for i in responsive_indices])
plt.axvline(0, color='white', linestyle='--', linewidth=1)
plt.tight_layout()
plt.show()

'''
# Parameters
spike_threshold = 2.5  # Total spike amplitude to consider a neuron responsive
window = 10  # Rolling window size for baseline estimation (in frames)
percentile = 10  # Percentile for dynamic F0 calculation

# Containers
responsive_indices = []
delta_f_over_f = []

for i, trace in enumerate(data):
    # Deconvolution
    _, s, _, _, _ = deconvolve(trace, penalty=1)
    
    # Check responsiveness
    if np.sum(s) >= spike_threshold:
        responsive_indices.append(i)
        
        # Dynamic F0 using rolling percentile baseline
        f0 = percentile_filter(trace, percentile=percentile, size=window, mode='reflect')
        dff = (trace - f0) / f0
        delta_f_over_f.append(dff)

# Convert to DataFrame (neurons as rows, frames as columns)
dff_df = pd.DataFrame(delta_f_over_f, index=[f"Neuron_{i}" for i in responsive_indices])

# Plot raster of ΔF/F₀ activity
plt.figure(figsize=(12, 6))
plt.imshow(dff_df, aspect='auto', cmap='hot', interpolation='none')
plt.colorbar(label='ΔF/F₀')
plt.xlabel("Frame")
plt.ylabel("Responsive Neurons")
plt.title("Raster Plot of ΔF/F₀ (Responsive Neurons)")
plt.yticks(ticks=np.arange(len(responsive_indices)), labels=[f"N{i}" for i in responsive_indices])
plt.tight_layout()
plt.show()
'''
## Script for the analysis
'''
# Parameters
spike_threshold = 1.0  # Minimum total spike amplitude to consider neuron responsive

# Containers
deconvolved_traces = []
spike_events = []
responsive_indices = []

# Deconvolution and responsiveness check
for i, trace in enumerate(data):
    c, s, _, _, _ = deconvolve(trace, penalty=1)
    deconvolved_traces.append(c)
    spike_events.append(s)
    if np.sum(s) >= spike_threshold:
        responsive_indices.append(i)

deconvolved_traces = np.array(deconvolved_traces)
spike_events = np.array(spike_events)

# Report responsive neurons
print(f"Found {len(responsive_indices)} responsive neurons (threshold = {spike_threshold})")
print("Responsive neuron indices:", responsive_indices)

# Visualization of the first responsive neuron
if responsive_indices:
    neuron_idx = responsive_indices[0]
    raw_trace = data[neuron_idx]
    deconv_trace = deconvolved_traces[neuron_idx]
    spikes = spike_events[neuron_idx]
    spike_times = np.where(spikes > 0)[0]
    spike_magnitudes = spikes[spike_times]

    plt.figure(figsize=(12, 6))
    plt.plot(raw_trace, label="Raw Calcium Signal", linewidth=1.5)
    plt.plot(deconv_trace, label="Deconvolved Signal", linewidth=1.5)
    plt.stem(spike_times, spike_magnitudes, linefmt='r-', markerfmt='ro', basefmt=" ", label="Spike Events")
    plt.title(f"Responsive Neuron {neuron_idx}")
    plt.xlabel("Frame")
    plt.ylabel("Signal Intensity")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No responsive neurons found with the current threshold.")

'''

## Original script
'''
# Deconvolve each neuron's trace
deconvolved_traces = []
spike_events = []

for trace in data:
    c, s, _, _, _ = deconvolve(trace, penalty=1)
    deconvolved_traces.append(c)
    spike_events.append(s)

deconvolved_traces = np.array(deconvolved_traces)
spike_events = np.array(spike_events)

# Visualize one example neuron
neuron_idx = 0  # Change to visualize a different neuron
raw_trace = data[neuron_idx]
deconv_trace = deconvolved_traces[neuron_idx]
spikes = spike_events[neuron_idx]

spike_times = np.where(spikes > 0)[0]
spike_magnitudes = spikes[spike_times]

plt.figure(figsize=(12, 6))
plt.plot(raw_trace, label="Raw Calcium Signal", linewidth=1.5)
plt.plot(deconv_trace, label="Deconvolved Signal", linewidth=1.5)
plt.stem(spike_times, spike_magnitudes, linefmt='r-', markerfmt='ro', basefmt=" ", label="Spike Events")
plt.title(f"Neuron {neuron_idx} Calcium Trace and Deconvolution")
plt.xlabel("Frame")
plt.ylabel("Signal Intensity")
plt.legend()
plt.tight_layout()
plt.show()
'''
