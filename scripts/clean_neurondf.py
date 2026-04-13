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
#import argparse
import re 
import pickle
import time
from tqdm import tqdm

from pathlib import Path
from probeinterface.plotting import plot_probe


import spikeinterface.extractors as se


from ratcode.config.paths import PATH_STORE_PICKLES, DROPBOX_TASK_PATH, PATH_DATAFRAMES
from ratcode.common.logging import determine_experiment
from ratcode.common.colorcodes import *
from ratcode.behavior import change_point
from ratcode.photometry.photometry import get_prediction, quantile_regression, signal2eventsnippets, find_poly, segment_and_fit_function, butter_filter, mask_jumps, make_continuous, query_and_compute_snippets, plot_snippets
from ratcode.ephys.neurons import get_psths_across_cells, align_spikes_to_ttl, compute_FR, load_ibl_sorter, determine_cell_type, produce_neuron_fig, produce_mega_neuron_fig, compute_zscore, get_psths_smooth, half_gaussian_kernel, do_PCA, get_PCA_windows, plot_raster, extract_features_cell_type, classify_cell_type_with_features
from ratcode.common.dataframe import group_and_listify
from ratcode.common.time import convert_date_bonsai, convert_timestamp
from ratcode.common.math import drop_nans_matrix
from ratcode.common.colorcodes import FI_order, color_FI_blocks, rwd_order, color_rwd_blocks
from ratcode.common.plotting import remove_legend

from ratcode.init import setup


setup()


#%%
"""
....###....##....##.####.##.....##....###....##..........########.....###....########.########
...##.##...###...##..##..###...###...##.##...##..........##.....##...##.##......##....##......
..##...##..####..##..##..####.####..##...##..##..........##.....##..##...##.....##....##......
.##.....##.##.##.##..##..##.###.##.##.....##.##..........##.....##.##.....##....##....######..
.#########.##..####..##..##.....##.#########.##..........##.....##.#########....##....##......
.##.....##.##...###..##..##.....##.##.....##.##..........##.....##.##.....##....##....##......
.##.....##.##....##.####.##.....##.##.....##.########....########..##.....##....##....########
"""
animal = 'Palladium'
date = '260319'

#bhv_pkl = glob.glob(rf"{DROPBOX_TASK_PATH}\analysis\{animal}_{date}_*.pkl")[0]
#bhvdf = pd.read_pickle(bhv_pkl)

EPHYS_PATH = os.path.join(DROPBOX_TASK_PATH, 'ephys', animal)
SAVE_SYNC_PATH = glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\*')[0]
IBL_SORTER_PATH =  glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\{animal}{date}*\ibl_sorter_results_drift_amplitude')[0]
neuronsdf = pd.read_pickle(fr'{SAVE_SYNC_PATH}\neuronsdf.pkl')

syncdf = pd.read_pickle(fr'{SAVE_SYNC_PATH}\syncdf.pkl')
exp = determine_experiment(syncdf)

DATACLASS_PATH = rf"{DROPBOX_TASK_PATH}\analysis_ephys\{animal}_{date}_sorted_data.pkl"
with open(DATACLASS_PATH, "rb") as f:
    sorted_data = pickle.load(f)

## this takes less time than the dataclass path
#sorted_data_ = load_ibl_sorter(IBL_SORTER_PATH, animal, date, exp)
#print(f"load_ibl_sorter took {time.time() - start_time:.2f} seconds")
#%%
#palladium 260319b
move_to_bad = [4,61,91,93,124,128,144,148,150,158,172,188,189,217,221,250,267,287,288,291,293,305,317,335,340,343,356,358,385,389,
               103,104,298]
move_to_ok = [38,94,179,227,266,290]
move_to_good = [326]


#palladium 260319b
#move_to_bad = [4,61,93,94,124,144,148,150,158,167,172,188,189,211,217,221,250,266,270,282,287,288,335,340,343,356,358,360,389,
#               103,104,298]
#move_to_ok = [73,91,317]
#move_to_good = [175,326]

#%%

neuronsdf.loc[neuronsdf['cluster_id'].isin(move_to_bad), 'SF'] = 'bad'
neuronsdf.loc[neuronsdf['cluster_id'].isin(move_to_ok), 'SF'] = 'ok'
neuronsdf.loc[neuronsdf['cluster_id'].isin(move_to_good), 'SF'] = 'good'

#%%

## run this to identify the cells

trough_to_peak = []
interspike_ratio = []
spike_suppression = []
waveforms_ms = []
waveforms = []
for cluster_id in tqdm(neuronsdf.cluster_id.values, desc=f"Extracting features {animal} {date}"):
    if cluster_id in neuronsdf.query('SF == "good" or SF == "ok"').cluster_id.values:
        try:        
            trough_to_peak_ms, long_interspike_ratio, post_spike_suppression_ms, waveform_ms, mean_waveform = extract_features_cell_type(cluster_id, sorted_data, syncdf)
        except Exception as e:
            print(f'error in cluster_id {cluster_id}')
            trough_to_peak_ms = np.nan
            long_interspike_ratio = np.nan
            post_spike_suppression_ms = np.nan
            waveform_ms = np.nan
            mean_waveform = np.nan

    else:
        trough_to_peak_ms = np.nan
        long_interspike_ratio = np.nan
        post_spike_suppression_ms = np.nan
        waveform_ms = np.nan
        mean_waveform = np.nan

    
    trough_to_peak.append(trough_to_peak_ms)
    interspike_ratio.append(long_interspike_ratio)
    spike_suppression.append(post_spike_suppression_ms)
    waveforms_ms.append(waveform_ms)
    waveforms.append(mean_waveform)
neuronsdf['trough_to_peak_ms'] = trough_to_peak
neuronsdf['long_interspike_ratio'] = interspike_ratio
neuronsdf['post_spike_suppression_ms'] = spike_suppression
neuronsdf['waveform_ms'] = waveforms_ms
neuronsdf['mean_waveform'] = waveforms

neuronsdf['cell_type'] = neuronsdf.apply(lambda x: classify_cell_type_with_features(x.trough_to_peak_ms,
                        x.long_interspike_ratio, x.post_spike_suppression_ms), axis = 1)

print('total good or ok SF labelled clusters')
print(len(neuronsdf.query('SF == "good" or SF == "ok"').cluster_id))

#%%
## extras and save
neuronsdf['animal'] = animal
neuronsdf['date'] = date

neuronsdf.to_pickle(fr'{SAVE_SYNC_PATH}\neuronsdf_new.pkl')

#%%
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
#%%
#neuronsdf = pd.read_pickle(fr'{SAVE_SYNC_PATH}\neuronsdf_new.pkl')

#neuronsdf_original = pd.read_pickle(fr'{SAVE_SYNC_PATH}\neuronsdf.pkl')
#len(neuronsdf_original.query('SF == "good"'))

## bug in the matshow in the produce_mega_neuron_fig. in cp for instance it is showing the color even if all the cp trials are nan
# but psths are correctly computed, so if it's flat it's flat (no events)

# %%
'''






move_to_bad = []
move_to_ok = []
move_to_good = []

move_to_bad = []
move_to_ok = []
move_to_good = []

move_to_bad = []
move_to_ok = []
move_to_good = []
'''