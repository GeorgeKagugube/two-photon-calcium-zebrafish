## Shebang
# Date: 09 - December - 2024
# Name: George William Kagugube

#!/Users/gwk/anaconda3/bin/python3

# Load other modules
import os
import mat73
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.io as sio
from pathlib import Path

## Machine learning libraries start from here 
from sklearn.cluster import KMeans

## # Use geopandas for vector data and rasterio for raster data
#import geopandas as gpd
import rasterio as rio
# Plotting extent is used to plot raster & vector data together
from rasterio.plot import plotting_extent

import earthpy as et
import earthpy.plot as ep

## Figure formatting starts here
sns.set_theme(style="darkgrid")

## Load data to analyse here
wt_gm = mat73.loadmat("/Users/gwk/Desktop/CalciumImaging/Wt_Unexposed/f1/gm.mat", use_attrdict=True)
wt_exp_gm = mat73.loadmat("/Users/gwk/Desktop/CalciumImaging/Wt_exposed/f2/gm.mat", use_attrdict=True)
mut_gm = mat73.loadmat("/Users/gwk/Desktop/CalciumImaging/Mut_exposed/f1/gm.mat", use_attrdict=True)

# Visualise the data here (images)
fig = plt.figure(figsize = (25, 15))
ax = fig.add_subplot(231)
ax1 = fig.add_subplot(232)
ax2 = fig.add_subplot(233)
ax.imshow(wt_gm['gm']['zimg'][0]['e'])
ax1.imshow(wt_gm['gm']['zimg'][1]['aligngood'])
ax2.imshow(wt_gm['gm']['zimg'][2]['aligngood2'])
plt.tight_layout()
plt.show()
