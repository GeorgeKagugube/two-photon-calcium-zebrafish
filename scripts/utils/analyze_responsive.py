import os
import pandas as pd
import numpy as np
import seaborn as sns
import mat73
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from numpy import trapz
from scipy.ndimage import percentile_filter
from collections import defaultdict
from oasis.functions import deconvolve
from scipy.ndimage import percentile_filter
from functools import lru_cache

def full_neuron_event_analysis(
    df_spikes,
    df_deconv,
    df_spike_response,
    save_dir,
    output_csv='responsive_neuron_event_summary.csv',
    frame_duration_sec=30,
    baseline_mode='static',
    dynamic_window=10,
    visualize=True):
    """
    Full spike event analysis for responsive and non-responsive neurons,
    including per-neuron averaged metrics and visualizations.

    Parameters
    ----------
    df_spikes : pd.DataFrame
        Deconvolved spike trains (neurons x frames)
    df_deconv : pd.DataFrame
        Deconvolved calcium traces (neurons x frames)
    df_spike_response : pd.DataFrame
        DataFrame containing 'is_responsive' column and neuron index
    output_csv : str
        Output filename for summary CSV
    frame_duration_sec : float
        Time per frame in seconds
    baseline_mode : str
        'static' or 'dynamic' for ΔF/F₀ estimation
    dynamic_window : int
        Window size for rolling min baseline
    visualize : bool
        Whether to show comparison plots

    Returns
    -------
    pd.DataFrame
        Combined summary of responsive and non-responsive neurons
    """

    def compute_f0(trace, mode='static', window=10):
        if mode == 'dynamic':
            return pd.Series(trace).rolling(window, min_periods=1, center=True).min().values
        else:
            return np.percentile(trace, 20)

    all_summaries = []

    for group, neuron_list in {
        'responsive': df_spike_response[df_spike_response['is_responsive']].index,
        'non_responsive': df_spike_response[~df_spike_response['is_responsive']].index}.items():
        for neuron_id in neuron_list:
            if neuron_id not in df_spikes.index or neuron_id not in df_deconv.index:
                continue

            spikes = df_spikes.loc[neuron_id].values
            trace = df_deconv.loc[neuron_id].values
            peaks, _ = find_peaks(spikes, height=0.1, distance=2)
            f0_arr = compute_f0(trace, mode=baseline_mode, window=dynamic_window)

            event_metrics = []
            for peak in peaks:
                peak_amp = trace[peak]
                f0_val = f0_arr[peak] if isinstance(f0_arr, np.ndarray) else f0_arr
                threshold = f0_val + 0.1 * (peak_amp - f0_val)

                end = peak + 1
                while end < len(trace) and trace[end] > threshold:
                    end += 1
                if end >= len(trace) or end <= peak:
                    continue

                duration_sec = (end - peak) * frame_duration_sec
                auc = trapz(trace[peak:end])
                slope = (trace[end] - trace[peak]) / duration_sec

                event_metrics.append({
                    'amplitude': peak_amp,
                    'duration_sec_90': duration_sec,
                    'AUC': auc,
                    'repolarization_slope_per_sec': slope
                })

            if event_metrics:
                df_events = pd.DataFrame(event_metrics)
                all_summaries.append({
                    'neuron_id': neuron_id,
                    'group': group,
                    'mean_amplitude': df_events['amplitude'].mean(),
                    'mean_duration_sec_90': df_events['duration_sec_90'].mean(),
                    'mean_auc': df_events['AUC'].mean(),
                    'mean_repolarization_slope': df_events['repolarization_slope_per_sec'].mean(),
                    'event_count': len(df_events)})

    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv(output_csv, index=False)
    df_summary.to_csv(os.path.join(save_dir, output_csv), index=False)
    print(f"✅ Saved full event summary to: {output_csv}")

    if visualize and not df_summary.empty:
        features = ['mean_amplitude', 'mean_duration_sec_90', 'mean_auc', 'mean_repolarization_slope']
        for feature in features:
            plt.figure(figsize=(6, 4))
            sns.boxplot(data=df_summary, x='group', y=feature, palette='Set2')
            sns.stripplot(data=df_summary, x='group', y=feature, color='black', alpha=0.4, jitter=True)
            plt.title(f'{feature.replace("_", " ").title()} by Neuron Group')
            plt.tight_layout()
            plt.show()

    return df_summary

def compute_population_activity_stats(results, save_dir):
    grouped_stats = defaultdict(list)

    for res in results:
        df = res['df_deconv']
        df_mean = df.mean(axis=1)
        df_std = df.std(axis=1)
        df_zscore = (df.sub(df_mean, axis=0)).div(df_std, axis=0)

        mean_activity = df_zscore.mean(axis=0)
        std_activity = df_zscore.std(axis=0)
        sem_activity = std_activity / np.sqrt(df_zscore.shape[0])
        frame_indices = np.arange(df_zscore.shape[1])

        stats_df = pd.DataFrame({
            'frame': frame_indices,
            'mean_activity': mean_activity,
            'std_activity': std_activity,
            'sem_activity': sem_activity})
        
        stats_df.to_csv(os.path.join(save_dir, f"{res['name']}_population_activity_stats.csv"), index=False)
        grouped_stats[res['name'].split('_')[0]].append(stats_df)

        plt.figure(figsize=(10, 5))
        plt.plot(frame_indices, mean_activity, label='Mean Activity')
        plt.fill_between(frame_indices, mean_activity - sem_activity, mean_activity + sem_activity, alpha=0.3, label='SEM')
        plt.axvline(x=res.get('stim_frame', 11), color='cyan', linestyle='--', label='Stimulus')
        plt.title(f"{res['name']} – Mean Z-scored Deconvolved Activity ± SEM")
        plt.xlabel("Frame")
        plt.ylabel("Z-scored Signal")
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"{res['name']}_population_activity_plot.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"✅ Population activity plot saved to {plot_path}")

    for group, stat_list in grouped_stats.items():
        all_means = np.array([df['mean_activity'].values for _, df in enumerate(stat_list)])
        group_mean = np.mean(all_means, axis=0)
        group_sem = np.std(all_means, axis=0) / np.sqrt(all_means.shape[0])
        frame_indices = stat_list[0]['frame']

        plt.figure(figsize=(10, 5))
        plt.plot(frame_indices, group_mean, label='Group Mean')
        plt.fill_between(frame_indices, group_mean - group_sem, group_mean + group_sem, alpha=0.3, label='SEM')
        plt.axvline(x=11, color='cyan', linestyle='--', label='Stimulus')
        plt.title(f"{group} – Group Average Z-scored Activity ± SEM")
        plt.xlabel("Frame")
        plt.ylabel("Z-scored Signal")
        plt.legend()
        plt.tight_layout()
        group_plot_path = os.path.join(save_dir, f"{group}_group_avg_activity_plot.png")
        plt.savefig(group_plot_path)
        plt.close()
        print(f"✅ Group average plot saved to {group_plot_path}")

def compute_dff(trace, window=10, perc=10):
    f0 = percentile_filter(trace, percentile=perc, size=window, mode='reflect')
    return (trace - f0) / np.clip(f0, 1e-6, None)

def process_dataframe(df, name, stim_frame, pre_window, post_window,
                      spike_threshold=1.5, dff_window=10, dff_percentile=10):
    spike_trains, dec_traces, dec_params = [], [], []
    for idx, row in df.iterrows():
        c, s, b, g, lam = deconvolve(row.values, g=[0.95, -0.02], penalty=1)
        spike_trains.append(s)
        dec_traces.append(c)
        dec_params.append({'neuron_id': idx, 'baseline': b, 'g1': g[0], 'g2': g[1], 'lambda': lam})

    df_spikes = pd.DataFrame(spike_trains, index=df.index, columns=df.columns)
    df_deconv = pd.DataFrame(dec_traces, index=df.index, columns=df.columns)
    df_params = pd.DataFrame(dec_params)

    df_spikes.to_csv(f'{name}_deconvolved_spikes.csv')
    df_deconv.to_csv(f'{name}_deconvolved_traces.csv')
    df_params.to_csv(f'{name}_deconv_params.csv', index=False)

    pre = range(max(0, stim_frame - pre_window), stim_frame)
    post = range(stim_frame + 1, min(df.shape[1], stim_frame + post_window + 1))
    avg_pre = df_spikes.iloc[:, pre].mean(axis=1)
    avg_post = df_spikes.iloc[:, post].mean(axis=1)

    df_resp = pd.DataFrame({
        'neuron_id': df.index,
        'avg_spikes_pre': avg_pre,
        'avg_spikes_post': avg_post,
        'delta_spike_rate': avg_post - avg_pre})

    delta_mean = df_resp['delta_spike_rate'].mean()
    delta_std = df_resp['delta_spike_rate'].std()
    threshold = 1.5
    df_resp['is_responsive'] = df_resp['delta_spike_rate'] >= threshold
    df_resp.to_csv(f'{name}_spike_response.csv', index=False)

    print(f"{name}: {df_resp['is_responsive'].sum()} responsive neurons out of {len(df_resp)} (threshold = {threshold:.3f})")

    resp_ids = df_resp[df_resp['is_responsive']]['neuron_id']
    df_spikes_resp = df_spikes.loc[resp_ids]

    return {
        'name': name,
        'df_spikes': df_spikes,
        'df_deconv': df_deconv,
        'df_resp': df_resp,
        'df_spikes_resp': df_spikes_resp,
        'stim_frame': stim_frame
    }

def sort_raster_by_responsiveness(df_spikes_resp, df_resp):
    sorted_ids = df_resp[df_resp['is_responsive']].sort_values(
        'delta_spike_rate', ascending=False)['neuron_id']
    return df_spikes_resp.loc[sorted_ids]

def plot_combined_rasters_sorted(results, stim_frame, save_dir):
    vmin, vmax = 0, 200
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 3 * len(results)), sharex=True)
    if len(results) == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        sorted_spikes = sort_raster_by_responsiveness(res['df_spikes_resp'], res['df_resp'])
        im = ax.imshow(sorted_spikes, aspect='auto', cmap='hot', interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.axvline(x=stim_frame, color='cyan', linestyle='--')
        ax.set_ylabel(res['name'])
        plt.colorbar(im, ax=ax, orientation='vertical', label='Estimated Spike Rate')
    axes[-1].set_xlabel("Frame")
    axes[0].set_title("Sorted Raster Plot of Responsive Neurons (Common Scale)")
    plt.tight_layout()
    save_path = os.path.join(save_dir, "combined_sorted_raster_plots.png")
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Sorted raster plot saved to {save_dir}")

def extract_signal_by_stimulus(data, stim=1):
    try:
        neurons = []
        roi_list = data['gmrxanat']['roi']
        for roi in roi_list:
            profiles = roi['Vprofiles']['meanprofile']
            if stim >= len(profiles):
                raise IndexError(f"Stimulus index {stim} out of range for meanprofile with length {len(profiles)}")
            neurons.append(profiles[stim])
        return neurons
    except KeyError as e:
        raise KeyError(f"Missing expected key in data: {e}")
    except IndexError as e:
        raise IndexError(f"Stimulus extraction failed: {e}")

@lru_cache(maxsize=50)
def import_2p_data(path_to_file):
    return mat73.loadmat(path_to_file)