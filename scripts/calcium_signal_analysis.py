#!/Users/gwk/anaconda3/envs/gcamp_analysis/bin/python

import pandas as pd
from utils.analyze_responsive import full_neuron_event_analysis, compute_population_activity_stats, compute_dff, extract_signal_by_stimulus
from utils.analyze_responsive import plot_combined_rasters_sorted,sort_raster_by_responsiveness, process_dataframe, import_2p_data
from pathlib import Path

## Define global variables here 
## This can only take a value of 0 to 16
stim_type = 2

def main():
    save_dir = Path('/Users/gwk/Desktop/Bioinformatics/two-photon-calcium-zebrafish/data/25_07_25')

    file_paths = {
        'wt_unexposed1': Path("/Users/gwk/Desktop/CalciumImaging/Wt_Unexposed/f1/gmrxanat.mat"),
        #'wt_unexposed2': Path("/Users/gwk/Desktop/CalciumImaging/Wt_Unexposed/f2/gmrxanat.mat"),
        'wt_exposed': Path("/Users/gwk/Desktop/CalciumImaging/Wt_exposed/f2/gmrxanat.mat"),
        #'mut_exposed1': Path("/Users/gwk/Desktop/CalciumImaging/Mut_exposed/f1/gmrxanat.mat"),
        'mut_exposed2': Path("/Users/gwk/Desktop/CalciumImaging/Mut_exposed/7_dpf/f1/gmrxanat.mat"),
        'mute_exposed3' : Path("/Users/gwk/Desktop/CalciumImaging/Mut_exposed/7_dpf/f3/gmrxanat.mat"),
        'mute_exposed4' : Path("/Users/gwk/Desktop/CalciumImaging/Mut_exposed/7_dpf/f2/gmrxanat.mat")
    }

    dataframes = []
    names = []

    for label, path in file_paths.items():
        try:
            data = import_2p_data(path)
            neurons = extract_signal_by_stimulus(data, stim=stim_type)
            df = pd.DataFrame(neurons)
            dataframes.append(df)
            names.append(label)
        except Exception as e:
            print(f"❌ Failed to load or process file: {path}\n   Error: {e}")

        finally:
            stim_frame = 11
            pre, post = 10, 20
            results = []

            for name, dfdata in zip(names, dataframes):
                result = process_dataframe(dfdata, name, stim_frame, pre, post)
                results.append(result)
                plot_combined_rasters_sorted(results, stim_frame, save_dir)
                compute_population_activity_stats(results, save_dir)
'''
            # Call after you have: df_spikes, df_deconv, responsive_neurons
            df_summary = full_neuron_event_analysis(
                df_spikes=df_spikes,
                df_deconv=df_deconv,
                df_spike_response=df_spike_response,
                output_csv='neuron_event_summary.csv',
                frame_duration_sec=30,
                baseline_mode='dynamic',  # or 'static'
                dynamic_window=10,
                visualize=True
            )'''

if __name__ == "__main__":
    main()
