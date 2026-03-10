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

#import spikeinterface.full as si
import spikeinterface.extractors as se
#import spikeinterface.preprocessing as spre
#import spikeinterface.sorters as ss
#import spikeinterface.qualitymetrics as sqm
#import spikeinterface.exporters as sexp
#import spikeinterface.widgets as sw

from ratcode.config.paths import PATH_STORE_PICKLES, DROPBOX_TASK_PATH
from ratcode.common.logging import determine_experiment
from ratcode.common.colorcodes import *
from ratcode.behavior import change_point
#from ratcode.photometry.photometry import get_prediction, quantile_regression, signal2eventsnippets, find_poly, segment_and_fit_function, butter_filter, mask_jumps, make_continuous
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

animal = 'Ruthenium'
date = '260308'
# %%


## previously in a function which is define_all_paths

EPHYS_PATH = os.path.join(DROPBOX_TASK_PATH, 'ephys', animal)
## defined via H:\

PATH_SAVE_FIGS = os.path.join(DROPBOX_TASK_PATH, 'analysis_ephys', f'{animal}_{date}')
if not os.path.exists(PATH_SAVE_FIGS):
    os.makedirs(PATH_SAVE_FIGS)

SAVE_SYNC_PATH = glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\*')[0]


IBL_SORTER_PATH =  glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\{animal}{date}*\ibl_sorter_results_drift_amplitude')[0]

NEURO_PATH =  glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\{animal}{date}*')[0]
#NEURO_PATH = glob.glob(rf"H:\{animal}{date}*\{animal}{date}*")[0]#[1]
#NEURO_PATH = glob.glob(rf"F:\EPHYS\{animal}{date}*\{animal}{date}*")[0]#[1]

#%%

#raw_rec = se.read_spikeglx(NEURO_PATH, load_sync_channel=False)
#
#sync_rec = se.read_spikeglx(NEURO_PATH, load_sync_channel=True)
#sync_data = sync_rec.get_traces(channel_ids=[sync_rec.get_channel_ids()[-1]])
#
#sampling_frequency = int(raw_rec.get_sampling_frequency())
#print(f'Recording time (min): {len(sync_data)/sampling_frequency/60}')

# %%
## run this only once to update the channel positions if they are in the old format (x = 1 or 2 for shank identity instead of actual x position)

#if os.path.exists(Path(fr'{IBL_SORTER_PATH}\channel_positions_original.npy')):
#    print('channel_positions.npy already updated')
#
#else:
#    file_position = Path(fr'{IBL_SORTER_PATH}\channel_positions.npy')
#    xy = np.load(file_position)
#    shank = np.load(file_position.with_name('channel_shanks.npy'))
#    if len(np.unique(xy[:, 0])) == 2:
#        np.save(file_position.with_name('channel_positions_original.npy'), xy)
#        xy_new = xy.copy()
#        xy_new[:, 0] = xy_new[:, 0] + shank.astype(np.float32) * 32 * 3
#        np.save(file_position, xy_new)
#
#%%
#probe = raw_rec.get_probe()
#
#plt.figure(figsize=(4,6))
#plot_probe(probe)
#plt.savefig(f'{PATH_SAVE_FIGS}\probe_geometry.png')


#%%

"""
..######..##....##.##....##..######.....########.....###....########....###...
.##....##..##..##..###...##.##....##....##.....##...##.##......##......##.##..
.##.........####...####..##.##..........##.....##..##...##.....##.....##...##.
..######.....##....##.##.##.##..........##.....##.##.....##....##....##.....##
.......##....##....##..####.##..........##.....##.#########....##....#########
.##....##....##....##...###.##....##....##.....##.##.....##....##....##.....##
..######.....##....##....##..######.....########..##.....##....##....##.....##
"""

## now go to 01_sync_and_clean
#(stuff above comes from 02_daily_neurons.py)


# %%

## detect TTLs rising edge -- this step takes some time

# Parameters
#chunk_duration_minutes = 2  # Duration of each chunk in minutes (adjust based on memory)
#chunk_size = chunk_duration_minutes * 60 * sampling_frequency  # Number of samples per chunk
#
## Load the sync channel from the recording
#ttl_channel = sync_rec.get_channel_ids()[-1]  # Assuming the TTL channel is the last one
#num_samples = sync_rec.get_num_frames()
#
## To store rising edges
#rising_edges = []
#above_thres = []
#
## Process in chunks
#for start in range(0, num_samples, chunk_size):
#    end = min(start + chunk_size, num_samples)
#    sync_data = sync_rec.get_traces(start_frame=start, end_frame=end, channel_ids=[ttl_channel]).ravel()
#    
#    plt.figure()
#    plt.plot(sync_data)
#    plt.title(start//chunk_size+1)
#    plt.show()
#    
#    # Convert to binary TTL (0 or 1) and detect rising edges
#    binary_ttl = (sync_data > 20).astype(np.uint8)
#    chunk_rising_edges = np.where(np.diff(binary_ttl) == 1)[0] + start
#    rising_edges.extend(chunk_rising_edges)
#
#    # new - discard with voltages > 100
#    #above_thres.extend(np.where(sync_data > 100)[0])
#
#    print(f"Processed chunk {start // chunk_size + 1} / {num_samples // chunk_size}")
#%%
# Convert rising_edges list to a numpy array
#rising_edges = np.array(rising_edges)
#rising_edges = rising_edges/sampling_frequency

#%%
# Save or use the rising edges as needed (for example, saving to file)
#np.save(fr'{SAVE_SYNC_PATH}/rising_edges.npy', rising_edges)
#
#print(f"Total rising edges detected: {len(rising_edges)}")
# %%

'''
all this up has been moved to a script to be run from the terminal: daily_neurons_01_extract_sync_correct_geometry.py
'''

#%%

if os.path.exists(fr'{SAVE_SYNC_PATH}/rising_edges.npy'):
    rising_edges = np.load(fr'{SAVE_SYNC_PATH}/rising_edges.npy')
    print(f'rising edges loaded for {animal} {date}')
else:
    print('rising edges not found, run daily_neurons_01_extract_sync_correct_geometry.py to extract them from the sync channel of the neuropixel recording')


"""
.########..##.....##.##.....##
.##.....##.##.....##.##.....##
.##.....##.##.....##.##.....##
.########..#########.##.....##
.##.....##.##.....##..##...##.
.##.....##.##.....##...##.##..
.########..##.....##....###...
"""

bhv_pkl = glob.glob(rf"{DROPBOX_TASK_PATH}\analysis\{animal}_{date}_*.pkl")[0]
bhvdf = pd.read_pickle(bhv_pkl)

bhvdf['cp'] = bhvdf.apply(lambda x: change_point.accepted_cp_Gallistel(x.trialno, 2, bhvdf, 'lever_rel', True)[0] if len(x.lever_rel)> 0 else np.nan, axis = 1)
bhvdf['cp'] = bhvdf.apply(lambda x: change_point.validate_cp(x.cp, x.lever_rel) if len(x.lever_rel) > 0 else np.nan, axis = 1)

bhvdf['bool_cp'] = np.isnan(bhvdf.cp.values) == False

bhvdf.drop(bhvdf.query('trial_duration < 200').index, inplace = True)
bhvdf.reset_index(drop = True, inplace = True)
bhvdf['trialno'] = bhvdf.index + 1

#%%
bhvdf.get(['FI', 'cp', 'bool_cp', 'n_protocols'])
#%%
duration_npx = np.diff(rising_edges)
duration_bhv = bhvdf.trial_duration_s.values

if (len(duration_npx) != len(duration_bhv)):
    print('different TTL lenghts!')
    print(f'true TTLs: {len(duration_bhv)}')
    print(f'extra TTLs in npx: {len(duration_npx)-len(duration_bhv)}')
#%%
"""
.##.....##....###....##....##.##.....##....###....##...........######..########.########.########.
.###...###...##.##...###...##.##.....##...##.##...##..........##....##....##....##.......##.....##
.####.####..##...##..####..##.##.....##..##...##..##..........##..........##....##.......##.....##
.##.###.##.##.....##.##.##.##.##.....##.##.....##.##...........######.....##....######...########.
.##.....##.#########.##..####.##.....##.#########.##................##....##....##.......##.......
.##.....##.##.....##.##...###.##.....##.##.....##.##..........##....##....##....##.......##.......
.##.....##.##.....##.##....##..#######..##.....##.########.....######.....##....########.##.......
"""
plt.plot(duration_bhv,'.-', label = 'bhv')

#duration_npx = np.concatenate([[np.nan],np.diff(rising_edges)])
#duration_npx = np.concatenate([duration_npx, np.ones(len(duration_bhv) - len(duration_npx))*np.nan])

### for the case where the npx doesn't see the full session
#trials = np.concatenate([[np.nan]*8,rising_edges])
#trials = np.delete(trials,[22,44,63])

trials = np.delete(rising_edges,[0,24,44,64,88,109])

#trials =  np.concatenate([trials, [np.nan]*24])
#np.delete(rising_edges,[0,24,55,56,104,102,103,112,117,120,131,137,143])
duration_npx = np.diff(trials)

plt.plot(duration_npx,'.-', label = 'npx')
#plt.plot(np.diff(rising_edges[307:407]))
#plt.xlim(50)

plt.legend(frameon = False)
#%%
plt.plot(trials)
plt.plot(bhvdf.trial_start/1000)
#plt.xlim(80)
# %%

"""
..######..##....##.##....##..######..########..########
.##....##..##..##..###...##.##....##.##.....##.##......
.##.........####...####..##.##.......##.....##.##......
..######.....##....##.##.##.##.......##.....##.######..
.......##....##....##..####.##.......##.....##.##......
.##....##....##....##...###.##....##.##.....##.##......
..######.....##....##....##..######..########..##......
"""
syncdf = bhvdf.get(['trial_duration_s'])
# %%
syncdf['npx_trial_duration'] = duration_npx
#syncdf['npx_trial_duration'] = np.concatenate([duration_npx,np.ones(3)*np.nan])
#%%
plt.plot(syncdf.trial_duration_s)
plt.plot(syncdf.npx_trial_duration,'--')

#%%
plt.plot(syncdf.npx_trial_duration - syncdf.trial_duration_s)

#%%
# dropping the last trial in npx time
# the time correspondence is
# npx_time = trial_start_s

syncdf['trial_start_s'] = bhvdf.trial_start/1000
#%%
npx_time = trials[:-1]

#npx_time = np.concatenate([trials, np.ones(2)*np.nan])
#%%
#syncdf['npx_time'] = np.delete(rising_edges,[0,-1])

syncdf['npx_time'] = npx_time

#syncdf['npx_time'] = np.concatenate([[np.nan],np.delete(rising_edges,[-1])])

#syncdf['npx_time'] = npx_time[:-1] #np.concatenate([npx_time, np.ones(61)*np.nan])
#syncdf.loc[:len(npx_time),'npx_time'] = npx_time

#%%
plt.plot(syncdf.trial_start_s - syncdf.npx_time)
#%%
plt.plot(syncdf.trial_start_s)
plt.plot(syncdf.npx_time+100)

# %%
syncdf
#%%
syncdf['FI'] = (bhvdf.FI/1000).astype(int)
syncdf['n_protocols'] = bhvdf.n_protocols
# %%

exp = determine_experiment(syncdf)

#%%
#this is in npx time
syncdf['lever_npx'] = syncdf.npx_time + bhvdf.lever_rel/1000
syncdf['poke_npx'] = syncdf.npx_time + bhvdf.poke_rel/1000
syncdf['rwd_onset_npx'] = syncdf.npx_time + bhvdf.pump_rel/1000
syncdf['cp'] = syncdf.npx_time + bhvdf.cp
#syncdf['cp_corrected'] = syncdf.npx_time + bhvdf.cp_corrected
#syncdf['click'] = syncdf.npx_time + bhvdf.click_rel/1000

syncdf['len_lvr'] = syncdf.lever_npx.apply(lambda x: len(x))
syncdf['relative_trial_duration'] = syncdf.trial_duration_s/syncdf.FI
syncdf['bool_cp'] = syncdf.cp.apply(lambda x: not(np.isnan(x)))
#%%
"""
..######.....###....##.....##.########.....######..##....##.##....##..######..########..########
.##....##...##.##...##.....##.##..........##....##..##..##..###...##.##....##.##.....##.##......
.##........##...##..##.....##.##..........##.........####...####..##.##.......##.....##.##......
..######..##.....##.##.....##.######.......######.....##....##.##.##.##.......##.....##.######..
.......##.#########..##...##..##................##....##....##..####.##.......##.....##.##......
.##....##.##.....##...##.##...##..........##....##....##....##...###.##....##.##.....##.##......
..######..##.....##....###....########.....######.....##....##....##..######..########..##......
"""
syncdf.to_pickle(fr'{SAVE_SYNC_PATH}/syncdf.pkl')
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

#templates = np.load(rf'{ibl_sorter_path}\templates.npy')
#channel_map = np.load(rf'{ibl_sorter_path}\channel_map.npy')
#channel_positions = np.load(rf'{ibl_sorter_path}\channel_positions.npy')
#spike_templates = np.load(rf'{ibl_sorter_path}\spike_templates.npy')
#template_features = np.load(rf'{ibl_sorter_path}\template_features.npy')
#pc_features = np.load(rf'{ibl_sorter_path}\pc_features.npy')
#amplitudes = np.load(rf'{ibl_sorter_path}\amplitudes.npy')

cluster_info = pd.read_csv(rf'{IBL_SORTER_PATH}\cluster_info.tsv', sep = '\t')
#%%

#### SAMPLING FREQUENCY NOT DEFINED -- AND ALSO, THIS CAN BE MOVED TO A SCRIPT


print('computing autocorrelogram')
spikes_self_aligned_all = []

# in seconds
window_start = -.2
window_end = .2
binW = .001

## historically I do this for all cells, but it's a bit of a waste of time tbh
#for cluster_id in cluster_info.query('SF == "good" or SF == "ok"').cluster_id:
for cluster_id in cluster_info.cluster_id:
    cluster_spikes = spike_times[spike_clusters == cluster_id]/sampling_frequency

    spikes_self_aligned = np.hstack(align_spikes_to_ttl(cluster_spikes,cluster_spikes,(window_start,window_end)))
    spikes_self_aligned = spikes_self_aligned[spikes_self_aligned!=0]

    spikes_self_aligned_all.append(spikes_self_aligned)
# %%

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
#%%

exp = determine_experiment(syncdf)
sorted_data = load_ibl_sorter(IBL_SORTER_PATH, animal, date, exp)
#%%
#added the dividing by the sampling frequency to convert to seconds
neuronsdf['spike_times'] = neuronsdf.cluster_id.apply(lambda x: sorted_data.spike_times[sorted_data.spike_clusters == x]/sorted_data.sampling_frequency)
neuronsdf['spikes_self_aligned'] = spikes_self_aligned_all
#%%
neuronsdf['cell_type'] = neuronsdf.apply(lambda x: determine_cell_type(x.cluster_id,sorted_data,syncdf) if x.group == 'good' else np.nan, axis = 1)

neuronsdf.to_pickle(fr'{SAVE_SYNC_PATH}\neuronsdf.pkl')
# %%
neuronsdf.query('cell_type == "TAN"')
# %%


#####

""""
EVERYTHING UP THERE KIND OF NEEDS TO BE MANUALLY RUN IN STEPS
this is because of the sync; but ideally it becomes just a script that one runs

the bottleneck for that atm is the syncdf --> automatize

besides that, there should be a step in which we evaluate if the geometry has already been updated or not (if original exists, that means it's been updated already)

(upstairs is the old 01_sync_and_clean)
"""

#%%

## if we need to read stuff (independently from above), and this is probably what will become a script that one just runs to produce figs


#syncdf = pd.read_pickle(fr'{SAVE_SYNC_PATH}\syncdf.pkl')
#exp = determine_experiment(syncdf)
#print(exp)
#
#neuronsdf = pd.read_pickle(fr'{SAVE_SYNC_PATH}\neuronsdf.pkl')
#spikes_self_aligned_all = neuronsdf.spikes_self_aligned.values
#
#sorted_data = load_ibl_sorter(IBL_SORTER_PATH, animal, date, exp)

#%%


## this was in 01_daily_neurons.py but it's still saving stuff -- so it can go up I think

import pickle 

DATACLASS_PATH = rf"{DROPBOX_TASK_PATH}\analysis_ephys\{animal}_{date}_sorted_data.pkl"

with open(DATACLASS_PATH, 'wb') as f:
    pickle.dump(sorted_data, f)

#%%

## now what will become the new script

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


#%%


## AND THEN THERE'S A BUNCH OF CODE FOR POPULATION STUFF AND PCA; but this should already be somewhere else
# %%
