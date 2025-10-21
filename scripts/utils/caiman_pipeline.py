## Import the required modules and set up the environment here
import bokeh.plotting as bpl
import cv2
import datetime
import glob
import holoviews as hv
from IPython import get_ipython
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import psutil
from pathlib import Path

try:
    cv2.setNumThreads(0)
except():
    pass

try:
    if __IPYTHON__:
        get_ipython().run_line_magic('load_ext', 'autoreload')
        get_ipython().run_line_magic('autoreload', '2')
except NameError:
    pass

import caiman as cm
from caiman.motion_correction import MotionCorrect
from caiman.source_extraction.cnmf import cnmf, params
from caiman.utils.utils import download_demo
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour
from caiman.utils.visualization import nb_view_quilt

bpl.output_notebook()
hv.notebook_extension('bokeh')

# set up logging
logfile = None # Replace with a path if you want to log to a file
logger = logging.getLogger('caiman')
logger.setLevel(logging.WARNING)
logfmt = logging.Formatter('%(relativeCreated)12d [%(filename)s:%(funcName)20s():%(lineno)s] [%(process)d] %(message)s')
if logfile is not None:
    handler = logging.FileHandler(logfile)
else:
    handler = logging.StreamHandler()
handler.setFormatter(logfmt)
logger.addHandler(handler)

# set env variables 
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

## Path to tiff files
movie_path = download_demo('Sue_2x_3000_40_-46.tif')

# press q to close
movie_orig = cm.load(movie_path) 
downsampling_ratio = 0.2  # subsample 5x
movie_orig.resize(fz=downsampling_ratio).play(gain=1.3,
                                              q_max=99.5, 
                                              fr=30,
                                              plot_text=True,
                                              magnification=2,
                                              do_loop=False,
                                              backend='opencv')

                                             