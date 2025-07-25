import os
import mat73
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio

## Machine learning libraries start from here 
from sklearn.cluster import KMeans

## Figure formatting starts here
sns.set_theme(style="darkgrid")

def extract_z_slice(cell_info):
    a = np.array(cell_info_data['slice'])
    z_slice = []
    for item in a:
        for inner_item in item:
            z_slice.append(inner_item[0])
    df = pd.DataFrame(z_slice)
    return (df)

'''
This function collects data from the 2 Photon Calcium analysis using Isaac Bianco's 
Inputs: 
1. metaFile = data structure in the form of a nested dictionary
2. traces = either z scores (roi_tsz) or calcium traces (roi_ts)

Outputs:
A list of lists containing the required data from all the z slices (4 slices in the case of our data)
'''
def get_calcium_data(metaFile, traces = 'roi_tsz'):
    zscore_access = metaFile['gmROIts']['z']
    files = []
    for file in zscore_access:
        files.append(file[traces])
    return files

'''
This function collects data from the 2 Photon Calcium analysis using Isaac Bianco's pipeline.

1. metaFile = data structure in the form of a nested dictionary
2. traces = either z scores (roi_tsz) or calcium traces (roi_ts)

Outputs:
A list of lists containing the required data from all the z slices (4 slices in the case of our data)
'''
def get_behaviour_data(behavoir_data, data_of_interest = 'vis_traj_tt'):
    behave = behavoir_data['gmbt']['p']
    for data in behave:
        return data[data_of_interest]

"""
This function returns the average activity of all the neurones in the dataset. 
Inputs: No arguments are needed. It uses the dictionary (Matlab structure) to extract all the zscores per neurone for the detected spike
any spike with a zscore higher than 2.3 is considered a spike. The sum of spikes from all neurones are calculated and divided
by the number of neurones to get the mean activity
Output:
Mean as one value and this can be taken to be compared across the different conditions
"""
def mean_activity(data):
    zslice_data = get_calcium_data(data, traces = 'roi_tsz')
    count = 0 
    total = len(zslice_data[0])
    for Slice in zslice_data:
        for neuron in Slice:
            for amplitude in neuron:
                if amplitude > 3:
                    count += 1
    mean = count/total
    return mean

df_cal = mat73.loadmat("/Users/gwk/Desktop/CalciumImaging/gmROIts.mat", use_attrdict=True)

## Estimate the mean of the calcium traces in a 
print(mean_activity(df_cal))
print()
