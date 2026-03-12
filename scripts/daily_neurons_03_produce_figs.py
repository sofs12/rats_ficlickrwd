#%%
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # .../rats_ficlickrwd

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Project root on sys.path:", PROJECT_ROOT)

# %%
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import zscore
from scipy.signal import lfilter

from pathlib import Path
from probeinterface.plotting import plot_probe

import spikeinterface.extractors as se

import pickle 

from ratcode.config.paths import PATH_STORE_PICKLES, DROPBOX_TASK_PATH
from ratcode.common.logging import determine_experiment
from ratcode.common.colorcodes import *
from ratcode.behavior import change_point
from ratcode.ephys.neurons import get_psths_across_cells, align_spikes_to_ttl, compute_FR, load_ibl_sorter, determine_cell_type, produce_neuron_fig, produce_mega_neuron_fig
from ratcode.common.dataframe import group_and_listify
from ratcode.common.time import convert_date_bonsai, convert_timestamp
from ratcode.common.math import drop_nans_matrix
from ratcode.common.colorcodes import FI_order, color_FI_blocks, rwd_order, color_rwd_blocks

from ratcode.init import setup
setup()
# %%

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
date = '260311'
# %%


EPHYS_PATH = os.path.join(DROPBOX_TASK_PATH, 'ephys', animal)

PATH_SAVE_FIGS = os.path.join(DROPBOX_TASK_PATH, 'analysis_ephys', f'{animal}_{date}')
if not os.path.exists(PATH_SAVE_FIGS):
    os.makedirs(PATH_SAVE_FIGS)

SAVE_SYNC_PATH = glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\*')[0]

IBL_SORTER_PATH =  glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\{animal}{date}*\ibl_sorter_results_drift_amplitude')[0]

NEURO_PATH =  glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\{animal}{date}*')[0]
#NEURO_PATH = glob.glob(rf"H:\{animal}{date}*\{animal}{date}*")[0]#[1]
#NEURO_PATH = glob.glob(rf"F:\EPHYS\{animal}{date}*\{animal}{date}*")[0]#[1]

raw_rec = se.read_spikeglx(NEURO_PATH, load_sync_channel=False)
sampling_frequency = int(raw_rec.get_sampling_frequency())


if os.path.exists(fr'{SAVE_SYNC_PATH}/syncdf.pkl'):
    syncdf = pd.read_pickle(fr'{SAVE_SYNC_PATH}/syncdf.pkl')
    print(f'syncdf loaded for {animal} {date}')
else:
    print('syncdf not found, run daily_neurons_02_manual_sync.py to align npx to bhv data')


    ## if we're here, kill the script


# %%

"""
..######...#######..########..########..########.##........#######...######...########.....###....##.....##
.##....##.##.....##.##.....##.##.....##.##.......##.......##.....##.##....##..##.....##...##.##...###...###
.##.......##.....##.##.....##.##.....##.##.......##.......##.....##.##........##.....##..##...##..####.####
.##.......##.....##.########..########..######...##.......##.....##.##...####.########..##.....##.##.###.##
.##.......##.....##.##...##...##...##...##.......##.......##.....##.##....##..##...##...#########.##.....##
.##....##.##.....##.##....##..##....##..##.......##.......##.....##.##....##..##....##..##.....##.##.....##
..######...#######..##.....##.##.....##.########.########..#######...######...##.....##.##.....##.##.....##

compute autocorrelogram (spikes_self_aligned_all), and store it in neuronsdf

run this once
"""
print(f'reading from ibl sorter: {IBL_SORTER_PATH}')

spike_times = np.load(rf'{IBL_SORTER_PATH}\spike_times.npy')
spike_clusters = np.load(rf'{IBL_SORTER_PATH}\spike_clusters.npy')

cluster_info = pd.read_csv(rf'{IBL_SORTER_PATH}\cluster_info.tsv', sep = '\t')

print('computing autocorrelogram. this takes time...')
spikes_self_aligned_all = []

# in seconds
window_start = -.2
window_end = .2
binW = .001

## historically I do this for all cells, but it's a bit of a waste of time tbh
#for cluster_id in cluster_info.query('SF == "good" or SF == "ok"').cluster_id:
## do a load bar here
for cluster_id in cluster_info.cluster_id:
    cluster_spikes = spike_times[spike_clusters == cluster_id]/sampling_frequency

    spikes_self_aligned = np.hstack(align_spikes_to_ttl(cluster_spikes,cluster_spikes,(window_start,window_end)))
    spikes_self_aligned = spikes_self_aligned[spikes_self_aligned!=0]

    spikes_self_aligned_all.append(spikes_self_aligned)
# %%

"""
..######...#######..########..########.########.########.....########.....###....########....###......
.##....##.##.....##.##.....##....##....##.......##.....##....##.....##...##.##......##......##.##.....
.##.......##.....##.##.....##....##....##.......##.....##....##.....##..##...##.....##.....##...##....
..######..##.....##.########.....##....######...##.....##....##.....##.##.....##....##....##.....##...
.......##.##.....##.##...##......##....##.......##.....##....##.....##.#########....##....#########...
.##....##.##.....##.##....##.....##....##.......##.....##....##.....##.##.....##....##....##.....##...
..######...#######..##.....##....##....########.########.....########..##.....##....##....##.....##...
"""

exp = determine_experiment(syncdf)
sorted_data = load_ibl_sorter(IBL_SORTER_PATH, animal, date, exp)

DATACLASS_PATH = rf"{DROPBOX_TASK_PATH}\analysis_ephys\{animal}_{date}_sorted_data.pkl"

with open(DATACLASS_PATH, 'wb') as f:
    pickle.dump(sorted_data, f)

#%%

"""
.##....##.########.##.....##.########...#######..##....##..######.....########..########
.###...##.##.......##.....##.##.....##.##.....##.###...##.##....##....##.....##.##......
.####..##.##.......##.....##.##.....##.##.....##.####..##.##..........##.....##.##......
.##.##.##.######...##.....##.########..##.....##.##.##.##..######.....##.....##.######..
.##..####.##.......##.....##.##...##...##.....##.##..####.......##....##.....##.##......
.##...###.##.......##.....##.##....##..##.....##.##...###.##....##....##.....##.##......
.##....##.########..#######..##.....##..#######..##....##..######.....########..##......
"""
neuronsdf = cluster_info #.query('n_spikes > 1000 and KSLabel in ["good","mua"]')

neuronsdf['spike_times'] = neuronsdf.cluster_id.apply(lambda x: sorted_data.spike_times[sorted_data.spike_clusters == x]/sorted_data.sampling_frequency)
neuronsdf['spikes_self_aligned'] = spikes_self_aligned_all
neuronsdf['cell_type'] = neuronsdf.apply(lambda x: determine_cell_type(x.cluster_id,sorted_data,syncdf) if x.group == 'good' else np.nan, axis = 1)

neuronsdf.to_pickle(fr'{SAVE_SYNC_PATH}\neuronsdf.pkl')


#%%

"""
..######....#######...#######..########.
.##....##..##.....##.##.....##.##.....##
.##........##.....##.##.....##.##.....##
.##...####.##.....##.##.....##.##.....##
.##....##..##.....##.##.....##.##.....##
.##....##..##.....##.##.....##.##.....##
..######....#######...#######..########.
"""

print(rf'DATA REFERING TO {animal} on {date}')

KSlabels = pd.read_csv(rf'{IBL_SORTER_PATH}\cluster_KSLabel.tsv', sep='\t')
cluster_info = pd.read_csv(rf'{IBL_SORTER_PATH}\cluster_info.tsv', sep = '\t')

good_clusters = KSlabels.query('KSLabel == "good"').cluster_id.values
mua_clusters = KSlabels.query('KSLabel == "mua"').cluster_id.values

print(f'total clusters: {len(KSlabels)}')
print(f'good clusters: {len(good_clusters)}')
print(f'mua clusters: {len(mua_clusters)}')

#%%
rising_edges = syncdf.query('trial_duration_s > 2').npx_time.values

#%%

print('quick figures being produced')

for cluster in good_clusters:
    produce_neuron_fig(cluster, rising_edges, sorted_data, window = (-10,10), save_fig=True, fig_save_path=PATH_SAVE_FIGS)

    #produce_mega_neuron_fig(cluster_id, sorted_data, syncdf, neuronsdf, PATH_SAVE_FIGS, bool_click = False)

## check if I want to keep both -- SEE BELOW UNDER THE FOLDERS

#for cluster in mua_clusters:
#    produce_neuron_fig(cluster, rising_edges, sorted_data, sorting_label='mua')

#%%

"""
..######..########....##..........###....########..########.##......
.##....##.##..........##.........##.##...##.....##.##.......##......
.##.......##..........##........##...##..##.....##.##.......##......
..######..######......##.......##.....##.########..######...##......
.......##.##..........##.......#########.##.....##.##.......##......
.##....##.##..........##.......##.....##.##.....##.##.......##......
..######..##..........########.##.....##.########..########.########
"""

print('multiple alignment figures being produced')

SFgood_path = rf'{PATH_SAVE_FIGS}\SF_good'
if not(os.path.exists(SFgood_path)):
    os.makedirs(SFgood_path)

SFok_path = rf'{PATH_SAVE_FIGS}\SF_ok'
if not(os.path.exists(SFok_path)):
    os.makedirs(SFok_path)



SFgood = cluster_info.query('SF == "good"').cluster_id.values
for cluster_id in SFgood:
    produce_mega_neuron_fig(cluster_id, sorted_data, syncdf, neuronsdf, fig_save_path=SFgood_path, bool_click = False, bool_cp_corrected = False)

SFok = cluster_info.query('SF == "ok"').cluster_id.values
for cluster_id in SFok:
    produce_mega_neuron_fig(cluster_id, sorted_data, syncdf, neuronsdf, fig_save_path=SFok_path, bool_click = False, bool_cp_corrected = False)

print('all done! :)')
print('')
#%%

## AND THEN THERE'S A BUNCH OF CODE FOR POPULATION STUFF AND PCA; but this should already be somewhere else
