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

animal = 'Cadmium'
date = '260414'
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


if os.path.exists(fr'{SAVE_SYNC_PATH}/rising_edges.npy'):
    rising_edges = np.load(fr'{SAVE_SYNC_PATH}/rising_edges.npy')
    print(f'rising edges loaded for {animal} {date}')
else:
    print('ERROR\nrising edges not found, run daily_neurons_01_extract_sync_correct_geometry.py to extract them from the sync channel of the neuropixel recording')
    print('GO RUN daily_neurons_01 first!\n')


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


trials = np.delete(rising_edges,[0,24])

## if npx disconnected, add nans at the end; uncomment this next line and adjust trial totals
trials =  np.concatenate([trials, [np.nan]*51])

duration_npx = np.diff(trials)

plt.plot(duration_npx,'.-', label = 'npx')

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
syncdf['npx_trial_duration'] = duration_npx

plt.plot(syncdf.trial_duration_s)
plt.plot(syncdf.npx_trial_duration,'--')

#%%
plt.plot(syncdf.npx_trial_duration - syncdf.trial_duration_s)

#%%
# dropping the last trial in npx time
# the time correspondence is
# npx_time = trial_start_s

syncdf['trial_start_s'] = bhvdf.trial_start/1000
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
