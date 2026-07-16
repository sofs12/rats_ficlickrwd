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
import pickle

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
"""
....###....##....##.####.##.....##....###....##..........########.....###....########.########...
...##.##...###...##..##..###...###...##.##...##..........##.....##...##.##......##....##.........
..##...##..####..##..##..####.####..##...##..##..........##.....##..##...##.....##....##.........
.##.....##.##.##.##..##..##.###.##.##.....##.##..........##.....##.##.....##....##....######.....
.#########.##..####..##..##.....##.#########.##..........##.....##.#########....##....##.........
.##.....##.##...###..##..##.....##.##.....##.##..........##.....##.##.....##....##....##.........
.##.....##.##....##.####.##.....##.##.....##.########....########..##.....##....##....########...
"""

animal = 'Silver'
date = ''
#%%
#neurons df
EPHYS_PATH = os.path.join(DROPBOX_TASK_PATH, 'ephys', animal)
SAVE_SYNC_PATH = glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\*')[0]
IBL_SORTER_PATH =  glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\{animal}{date}*\ibl_sorter_results_drift_amplitude')[0]
neuronsdf = pd.read_pickle(fr'{SAVE_SYNC_PATH}\neuronsdf.pkl')

print(f'loaded neuronsdf with keys animal {neuronsdf.animal.unique()} and date {neuronsdf.date.unique()}')

#%%
move_to_bad = []
move_to_ok = []
move_to_good = []

#%%

neuronsdf.loc[neuronsdf['cluster_id'].isin(move_to_bad), 'SF'] = 'bad'
neuronsdf.loc[neuronsdf['cluster_id'].isin(move_to_ok), 'SF'] = 'ok'
neuronsdf.loc[neuronsdf['cluster_id'].isin(move_to_good), 'SF'] = 'good'


#%%
## save
neuronsdf.to_pickle(fr'{SAVE_SYNC_PATH}\neuronsdf.pkl')
# %%


"""
.##.....##.########..########.....###....########.########....########.####..######...##.....##.########..########..######.
.##.....##.##.....##.##.....##...##.##......##....##..........##........##..##....##..##.....##.##.....##.##.......##....##
.##.....##.##.....##.##.....##..##...##.....##....##..........##........##..##........##.....##.##.....##.##.......##......
.##.....##.########..##.....##.##.....##....##....######......######....##..##...####.##.....##.########..######....######.
.##.....##.##........##.....##.#########....##....##..........##........##..##....##..##.....##.##...##...##.............##
.##.....##.##........##.....##.##.....##....##....##..........##........##..##....##..##.....##.##....##..##.......##....##
..#######..##........########..##.....##....##....########....##.......####..######....#######..##.....##.########..######.

if producing figs again, need to load more stuff
"""
syncdf = pd.read_pickle(fr'{SAVE_SYNC_PATH}\syncdf.pkl')
exp = determine_experiment(syncdf)

DATACLASS_PATH = rf"{DROPBOX_TASK_PATH}\analysis_ephys\{animal}_{date}_sorted_data.pkl"
with open(DATACLASS_PATH, "rb") as f:
    sorted_data = pickle.load(f)


PATH_SAVE_FIGS = os.path.join(DROPBOX_TASK_PATH, 'analysis_ephys', f'{animal}_{date}')
if not os.path.exists(PATH_SAVE_FIGS):
    os.makedirs(PATH_SAVE_FIGS)
#%%
print('multiple alignment figures being produced')

SFgood_path = rf'{PATH_SAVE_FIGS}\SF_good_new'
if not(os.path.exists(SFgood_path)):
    os.makedirs(SFgood_path)
SFok_path = rf'{PATH_SAVE_FIGS}\SF_ok_new'
if not(os.path.exists(SFok_path)):
    os.makedirs(SFok_path)

SFgood = neuronsdf.query('SF == "good"').cluster_id.values
for cluster_id in SFgood:
    produce_mega_neuron_fig(cluster_id, sorted_data, syncdf, neuronsdf, fig_save_path=SFgood_path, bool_click = False, bool_cp_corrected = False)

SFok = neuronsdf.query('SF == "ok"').cluster_id.values
for cluster_id in SFok:
    produce_mega_neuron_fig(cluster_id, sorted_data, syncdf, neuronsdf, fig_save_path=SFok_path, bool_click = False, bool_cp_corrected = False)
print('all done! :)')
print('')


print('ORIGINAL TOTALS IBL SORTER')
print(len(neuronsdf.query('KSLabel == "good"')))
print(len(neuronsdf.query('KSLabel == "mua"')))
# %%



#%%



#%%


#%%

move_to_bad = []
move_to_ok = []
move_to_good = []