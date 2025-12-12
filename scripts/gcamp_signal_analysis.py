import pandas as pd
import sys
sys.path.append("./utils")
from deconvolve_oasis_dataframe import deconvolve_oasis_dataframe, save_oasis_outputs
from summarise_activity_by_window import summarise_activity_by_window
from raster_from_spikes import raster_with_optional_dff

# dff_df: your denoised & smoothed ΔF/F with rows=neurons, cols=frames
# Example: dff_df = pd.read_csv("dff_cleaned.csv", index_col=0)


## Import the dataframe for further analysis here
## Path to raw flourance data files
#  path_to_files = Path('/Users/gwk/Desktop/CalciumImaging/extractedROI/loomROI')
#dff_df = pd.read_csv('../Data/smoothed_roi_mutExposed4.csv', index_col=0)
# Unexposed wILD tyPE 
#dff_df = pd.read_csv('/Users/gwk/Desktop/CalciumImaging/extractedROI/loomROI/unexposed_wt.csv', index_col=0)
#dff_df = pd.read_csv('/Users/gwk/Desktop/CalciumImaging/extractedROI/loomROI/unexposed_wt2.csv', index_col=0)

# ## Exposed wild type
#dff_df = pd.read_csv('/Users/gwk/Desktop/CalciumImaging/extractedROI/loomROI/exposed_wt.csv', index_col=0)

# ## Exposed mutants
#dff_df = pd.read_csv('/Users/gwk/Desktop/CalciumImaging/extractedROI/loomROI/exposed_mut1.csv', index_col=0)
#dff_df = pd.read_csv('/Users/gwk/Desktop/CalciumImaging/extractedROI/loomROI/exposed_mut2.csv', index_col=0)
#dff_df = pd.read_csv('/Users/gwk/Desktop/CalciumImaging/extractedROI/loomROI/exposed_mut3.csv', index_col=0)
dff_df = pd.read_csv('/Users/gwk/Desktop/CalciumImaging/extractedROI/loomROI/exposed_mut4.csv', index_col=0)

dff_df = dff_df.transpose()

experiment = 'dff_mutExposed4'

## Check the head of the dataframe 
print (dff_df.head())
print (dff_df.shape)

## Run the deconcolution analysis here
S_df, C_df, B_df, G_df = deconvolve_oasis_dataframe(
    dff_df,
    method="ar1",     # or "ar2"
    g=None,           # let OASIS estimate AR coefficient per neuron
    s_min=0.0,        # increase for stronger sparsity on spikes
    penalty=1,      # tune based on SNR (higher => sparser)
    optimize_g=True,
    n_jobs=-1,        # parallel (requires joblib); set 1 to disable
    show_progress=True
)

# Optional: save outputs
save_oasis_outputs(S_df, C_df, B_df, G_df, prefix=f"./oasis_out/{experiment}")

# Quick sanity check plot for one neuron:
import matplotlib.pyplot as plt
i0 = dff_df.index[100]
plt.figure(); plt.plot(dff_df.loc[i0].values, label="ΔF/F")
plt.plot(C_df.loc[i0].values, label="C (denoised)")
plt.plot(S_df.loc[i0].values, label="S (deconvolved)")
plt.legend(); plt.xlabel("Frame"); plt.ylabel("a.u.")
plt.show()



## Load the deconvolved sinals here for further analysis
S_df = pd.read_csv(f'./oasis_out/{experiment}_spikes_oasis.csv', index_col=0)
print (f'Printing the deconvulved signal dataframe here: {S_df.head()}')
print (f'Printings its shape for clarity {S_df.shape}')

### Extract metrices for each fish that can be used for further comparisons
# Inputs you already have:
# S_df : DataFrame (rows=ROIs, cols=frames), deconvolved spikes from OASIS or fallback
# fs   : sampling rate (Hz), e.g. 3.6
# fish_id, group : labels you want in the outputs
per_roi, per_fish = summarise_activity_by_window(
    S_df,
    fs=3.6,
    fish_id="FISH_07",
    group="Mutant + Mn",
    # windows can be customised, default uses (0–11, 12–35, 36–108)
    save_prefix="./out_mn_pipeline",
    thr=0.3,   # raise slightly (e.g., 0.01) if you want to suppress tiny OASIS spikes
    name = f'{experiment}'
)

## Generate raster plots here 
# S_df: deconvolved spikes (rows=ROIs, cols=frames)
# dff_df: OPTIONAL ΔF/F matrix, same shape/alignment
out = raster_with_optional_dff(
    S_df,
    fs=3.6,
    stim_frame=12,
    windows={"pre": (0,11), "during": (12,35), "post": (36,108)},
    thr=0.0,
    sort_by="latency_during",
    dff_df=dff_df,                    # or None to skip heatmap
    heatmap_mode="zscore",            # "zscore" | "minmax" | "raw"
    heatmap_clip_percentile=99.0,
    responsive_only=False,
    title="Fish F05 — Raster ± ΔF/F"
)
