#%%
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # .../rats_ficlickrwd

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Project root on sys.path:", PROJECT_ROOT)

import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import zscore
from scipy.signal import lfilter
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import argparse

from pathlib import Path
from probeinterface.plotting import plot_probe


import spikeinterface.extractors as se


from ratcode.config.paths import PATH_STORE_PICKLES, DROPBOX_TASK_PATH
from ratcode.common.logging import determine_experiment
from ratcode.common.colorcodes import *
from ratcode.behavior import change_point
from ratcode.photometry.photometry import get_prediction, quantile_regression, signal2eventsnippets, find_poly, segment_and_fit_function, butter_filter, mask_jumps, make_continuous, query_and_compute_snippets, plot_snippets
from ratcode.ephys.neurons import get_psths_across_cells, align_spikes_to_ttl, compute_FR, load_ibl_sorter, determine_cell_type, produce_neuron_fig, produce_mega_neuron_fig, compute_zscore, get_psths_smooth, half_gaussian_kernel, do_PCA, get_PCA_windows
from ratcode.common.dataframe import group_and_listify
from ratcode.common.time import convert_date_bonsai, convert_timestamp
from ratcode.common.math import drop_nans_matrix
from ratcode.common.colorcodes import FI_order, color_FI_blocks, rwd_order, color_rwd_blocks
from ratcode.common.plotting import remove_legend

from ratcode.init import setup    

setup()
#%%

animal = 'Ruthenium'
date = '260303'
#%%
#neurons df
EPHYS_PATH = os.path.join(DROPBOX_TASK_PATH, 'ephys', animal)
SAVE_SYNC_PATH = glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\*')[0]
IBL_SORTER_PATH =  glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\{animal}{date}*\ibl_sorter_results_drift_amplitude')[0]
neuronsdf = pd.read_pickle(fr'{SAVE_SYNC_PATH}\neuronsdf.pkl')
# %%
neuronsdf.query('KSLabel == "good"').SF
# %%
cluster_SF = pd.read_csv(fr'{IBL_SORTER_PATH}\cluster_SF.tsv', sep = '\t')
cluster_info = pd.read_csv(fr'{IBL_SORTER_PATH}\cluster_info.tsv', sep = '\t')

# %%
len(cluster_info)
# %%

cluster_info.query('SF == "good"')

# %%

neuronsdf.SF = cluster_info.SF
#%%
neuronsdf.SF
#%%

neuronsdf.to_pickle(fr'{SAVE_SYNC_PATH}\neuronsdf.pkl')

# %%
