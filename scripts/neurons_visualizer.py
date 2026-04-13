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
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import zscore

from ratcode.config.paths import PATH_STORE_PICKLES, DROPBOX_TASK_PATH, PATH_STORE_PHOTOMETRY_PICKLES, PATH_DANEURONS_ANALYSIS, PATH_DATAFRAMES
from ratcode.common.logging import determine_experiment
from ratcode.common.colorcodes import *
from ratcode.behavior import change_point
from ratcode.common.time import convert_date_bonsai, convert_timestamp
from ratcode.photometry.photometry import signal2eventsnippets, butter_filter, quantile_regression, get_prediction, segment_and_fit_function, mask_jumps, find_poly
from ratcode.common.dataframe import group_and_listify
from ratcode.ephys.neurons import get_psths_across_cells, align_spikes_to_ttl, compute_FR
from ratcode.common.math import drop_nan_rows_in_matrix

from ratcode.init import setup
setup()

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import HuberRegressor 
#%%
from scipy.signal import find_peaks, correlate


def create_da_template(fs, rise_time=0.1, decay_time=0.4):
    """Creates a synthetic DA transient template."""
    t_temp = np.arange(0, 2, 1/fs)
    # Alpha-function style: (t/tau) * exp(-t/tau)
    # Or simplified: rapid rise then exponential decay
    template = (1 - np.exp(-t_temp / rise_time)) * np.exp(-t_temp / decay_time)
    return template / np.max(template) # Normalize height

from sklearn.decomposition import NMF

def compute_NMF(red_obs, green_obs):

    """
    Returns motion, DA
    """

    # 1. Prepare data (NMF requires strictly POSITIVE data)
    # Shift data so the minimum value is 0 or a small positive constant
    offset_green = np.min(green_obs)
    offset_red = np.min(red_obs)
    X_positive = np.c_[green_obs - offset_green, red_obs - offset_red]

    # 2. Fit NMF
    # 'mu' solver with 'kullback-leibler' is often more robust for biological peaks
    model = NMF(n_components=2, init='random', random_state=0, solver='mu')
    H = model.fit_transform(X_positive)  # These are your separated traces
    W = model.components_              # This is how they are mixed

    # 3. Identify your signals
    # One component will have the transients (DA), one will have the motion.
    # You can identify them by checking which one correlates more with the Red channel.
    corr0 = np.corrcoef(H[:, 0], red_obs)[0, 1]
    corr1 = np.corrcoef(H[:, 1], red_obs)[0, 1]

    DA = H[:, 1] if corr0 > corr1 else H[:, 0]
    motion = H[:, 0] if corr0 > corr1 else H[:, 1]

    return motion, DA

def compute_NMF_masked(red_obs, green_obs):
    # 1. Create a mask of rows that have valid data in BOTH channels
    valid_mask = ~np.isnan(red_obs) & ~np.isnan(green_obs)
    
    # 2. Filter the data to only include valid timepoints
    red_valid = red_obs[valid_mask]
    green_valid = green_obs[valid_mask]
    
    # 3. Shift and Fit (Using nanmin to be safe)
    offset_red = np.min(red_valid)
    offset_green = np.min(green_valid)
    X_pos = np.c_[green_valid - offset_green, red_valid - offset_red]
    
    model = NMF(n_components=2, init='nndsvd', solver='mu', random_state=0)
    H_valid = model.fit_transform(X_pos)
    
    # 4. Identify which column in H_valid is which
    # Correlate columns of H with the original valid red signal
    corr0 = np.corrcoef(H_valid[:, 0], red_valid)[0, 1]
    corr1 = np.corrcoef(H_valid[:, 1], red_valid)[0, 1]

    if corr0 > corr1:
        motion_valid = H_valid[:, 0]
        DA_valid = H_valid[:, 1]
    else:
        motion_valid = H_valid[:, 1]
        DA_valid = H_valid[:, 0]

    # 5. Reconstruct full-length arrays (matching original input shape)
    motion_full = np.full(red_obs.shape, np.nan)
    DA_full = np.full(red_obs.shape, np.nan)
    
    # Map the valid results back to their original temporal positions
    motion_full[valid_mask] = motion_valid
    DA_full[valid_mask] = DA_valid
    
    return motion_full, DA_full

def get_event_indices(timestamps, events, window, fs):
    """
    Returns a single 1D array of all unique indices falling within the windows.
    """
    # 1. Calculate offsets in samples
    pre_samples = int(window[0] * fs)
    post_samples = int(window[1] * fs)
    
    # 2. Find center indices for each event
    center_indices = np.searchsorted(timestamps, events)
    
    # 3. Create start and end bounds for each event window
    starts = center_indices - pre_samples
    ends = center_indices + post_samples
    
    # 4. Generate the full range of indices
    # This creates a 2D array if windows are equal length, 
    # then flattens and removes duplicates/out-of-bounds
    offsets = np.arange(-pre_samples, post_samples)
    all_indices = (center_indices[:, np.newaxis] + offsets).flatten()
    
    # 5. Clean up: Remove out-of-bounds and duplicates (from overlapping windows)
    all_indices = all_indices[(all_indices >= 0) & (all_indices < len(timestamps))]
    return np.unique(all_indices)
# %%

from scipy.stats import zscore

def get_colors_and_windows(exp, psth_bin = 0.01):
    color_palette = color_nprots_blocks if exp == 'c' else color_FI_blocks

    if exp == 'c':
        cond_I = 'rwd7'
        cond_II = 'rwd14'
        cond_III = 'rwd28'

        window_I = (0,int(30/psth_bin))
        window_II = (int(30/psth_bin),int((30*2)/psth_bin))
        window_III = (int((30*2)/psth_bin),int((30*3)/psth_bin))

    else:
        cond_I = 'FI15'
        cond_II = 'FI30'
        cond_III = 'FI60'

        window_I = (0,int(15/psth_bin))
        window_II = (int(15/psth_bin),int((15+30)/psth_bin))
        window_III = (int((15+30)/psth_bin),int((15+30+60)/psth_bin))


    all_windows = [window_I,window_II,window_III]
    all_conds = [cond_I, cond_II, cond_III]

    return all_windows,all_conds,color_palette

def sort_index_order(split_index, index_order, concat_for_PCA):
    index_order_sorted = np.concatenate([index_order[split_index:], index_order[:split_index]])
    plt.imshow(concat_for_PCA[index_order_sorted], aspect = 'auto', origin = 'lower', vmin = -1, vmax = 2)
    return index_order_sorted
#%%
def add_3d_line_w_start_matplotlib(ax, projection, color, legend, smooth_sigma, linestyle = '-', bool_start_dot = True, project_axis = None, location_project_axis = None):
    #sigma = 10  # in samples; adjust based on your sampling rate
    projection = gaussian_filter1d(projection, sigma=smooth_sigma, axis=0, mode="nearest")
    
    x = projection[:,0]
    y = projection[:,1]
    z = projection[:,2]

    ax.plot(x,y,z, color = color, label = legend, ls = linestyle)
    
    if bool_start_dot:
        ax.plot(x[0],y[0],z[0], 'o', color = color, label = legend)

    if 'x' in project_axis:
        ax.plot(location_project_axis[project_axis == 'x']*np.ones_like(x), y, z, color=color, alpha=0.2, ls = linestyle)

    if 'y' in project_axis:
        ax.plot(x, location_project_axis[project_axis == 'y']*np.ones_like(y), z, color=color, alpha=0.2, ls = linestyle)

    if 'z' in project_axis:
        ax.plot(x,y,location_project_axis[project_axis == 'z']*np.ones_like(z), color = color, alpha = 0.2, ls = linestyle)

#%%
Strontium_ephys_dates = [
 '250219',
 '250220',
 '250221',
 '250225',
 '250226',
 '250227',
 '250228',
 '250304',
 '250305',
 '250306',
 '250307',
 '250311',
 '250313',
 '250314',
 '250318',
 '250319',
 #'250325',
 #'250328'
 ]

Technetium_ephys_dates = [
 '250618',
 #'250619', -- 29 mins recording, so not all blocks came
 #'250620', -- it's only 15 mins
 '250623',
 '250624',
 '250625',
 '250626'
 ]


Niobium_ephys_dates = [
 '250618',
 #'250619', only 16 mins
 '250620',
 '250624',
 '250625',
 '250626',
 '250627',
 '250628'
]


Zirconium_ephys_dates = [
 '250321',
 '250325',
 '250326',
 '250327',
 '250328',
 '250401',
 '250418',
 '250419',
 '250422',
 #'250427', unsortable
 '250428',
 '250429',
 #'250430', sortable but terrible
 '250501',
 '250502',
 '250503',
 '250504'
 ]

ephys_dates_dict = {
    'Strontium': Strontium_ephys_dates,
    'Zirconium': Zirconium_ephys_dates,
    'Niobium': Niobium_ephys_dates,
    'Technetium': Technetium_ephys_dates
    }
# %%

"""
.##........#######.....###....########.....########..########
.##.......##.....##...##.##...##.....##....##.....##.##......
.##.......##.....##..##...##..##.....##....##.....##.##......
.##.......##.....##.##.....##.##.....##....##.....##.######..
.##.......##.....##.#########.##.....##....##.....##.##......
.##.......##.....##.##.....##.##.....##....##.....##.##......
.########..#######..##.....##.########.....########..##......
"""

#%%
#unidf = pd.read_pickle(f'{PATH_DATAFRAMES}/unidf.pkl')

#blocksdf = pd.read_pickle(fr'{PATH_DATAFRAMES}/blocksdf_july25_thesis_dataset.pkl')
blocksdf = pd.read_pickle(fr'{PATH_DATAFRAMES}\blocksdf_march26_Ruthenium_Palladium.pkl')
#blocksdf = pd.read_pickle(fr'D:\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\Data\FIClickRwd\analysis_ephys_thesis\dfs\blocksdf.pkl')
#blocksdf = pd.read_pickle(rf'{PATH_DATAFRAMES}/blocksdf.pkl')

## originally from the analysis_ephys_thesis folder
#all_aggregated_neuronsdf = pd.read_pickle(rf"D:\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\Data\FIClickRwd\analysis_ephys_thesis\dfs\all_aggregated_neuronsdf.pkl")
#%%

allneurons_Ruthenium = pd.read_pickle(rf"D:\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\Data\FIClickRwd\analysis_ephys\Ruthenium_animalneurondf.pkl")
#%%
allneurons_Palladium = pd.read_pickle(rf"D:\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\Data\FIClickRwd\analysis_ephys\Palladium_animalneurondf.pkl")
#%%
allneurons_Ruthenium.keys()

#%%
all_aggregated_neuronsdf = pd.concat([allneurons_Palladium, allneurons_Ruthenium])
#%%

all_aggregated_neuronsdf = pd.read_pickle(rf'{PATH_DATAFRAMES}/all_aggregated_neuronsdf.pkl')

#photometrydf = pd.read_pickle(rf'{PATH_DATAFRAMES}/agg_photometry_withICA.pkl')
photometrydf = pd.read_pickle(rf'{PATH_DATAFRAMES}')
#%%
animal = 'Zirconium' ## single animal
bool_multiple_animals = False

dates_to_consider = ephys_dates_dict[animal]
cells_to_use = list(all_aggregated_neuronsdf.query(f'animal == "{animal}" and date in {list(dates_to_consider)} and cell_type == "MSN" and KSLabel == "good"').get(['animal', 'date', 'cluster_id']).itertuples(index=False, name=None))
neuronsdf = all_aggregated_neuronsdf
bhvdf = blocksdf

smooths_15 = get_psths_across_cells(
    neuronsdf, bhvdf.query('experiment != "c"'), cells_to_use, event_name='npx_trial_start',
    query_condition='FI == 15', pre_time = 0, post_time = 15
)[2]

smooths_30 = get_psths_across_cells(
    neuronsdf, bhvdf.query('experiment != "c"'), cells_to_use, event_name='npx_trial_start',
    query_condition='FI == 30', pre_time = 0, post_time = 30
)[2]

smooths_60 = get_psths_across_cells(
    neuronsdf, bhvdf.query('experiment != "c"'), cells_to_use, event_name='npx_trial_start',
    query_condition='FI == 60', pre_time = 0, post_time = 60
)[2]

smooths_for_PCA = drop_nan_rows_in_matrix(
    np.concatenate([zscore(smooths_15, axis = 1),
                    zscore(smooths_30, axis = 1),
                    zscore(smooths_60, axis = 1)], axis = 1))

#plt.imshow(smooths_for_PCA, aspect = 'auto', vmin = -1, vmax = 3)
# %%

plt.imshow(smooths_for_PCA, aspect = 'auto')
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

animal = 'Zirconium'
date = '250429'

PATH_SAVE_DANEURONS_FIGS = os.path.join(PATH_DANEURONS_ANALYSIS, rf'{animal}_{date}/clusters_DA_DLC')
if not os.path.exists(PATH_SAVE_DANEURONS_FIGS):
    os.makedirs(PATH_SAVE_DANEURONS_FIGS)

PATH_SAVE_STA_FIGS = os.path.join(PATH_DANEURONS_ANALYSIS, rf'{animal}_{date}/STA_DA')
if not os.path.exists(PATH_SAVE_STA_FIGS):
    os.makedirs(PATH_SAVE_STA_FIGS)
    os.makedirs(os.path.join(PATH_SAVE_STA_FIGS,'aligned_to_spike'))
    os.makedirs(os.path.join(PATH_SAVE_STA_FIGS,'aligned_to_DApeak'))

#%%
neurons_sessiondf = all_aggregated_neuronsdf.query(f'animal == "{animal}" and date == "{date}" and SF == "good" and cell_type == "MSN" and fr >=.2')

# %%

DAsessiondf = photometrydf.query(rf'animal == "{animal}" and date == "{date}"')
DAsessiondf['cp_abs_harp'] = DAsessiondf.cp_harp + DAsessiondf.trial_start_harp
snipps_DA_cp, timeDA = signal2eventsnippets(np.hstack(DAsessiondf.timestamp_session),
                                np.hstack(DAsessiondf.DA_session),
                                DAsessiondf.cp_abs_harp.values, [-4,4], .01)
snipps_DA_last_lever, _ = signal2eventsnippets(np.hstack(DAsessiondf.timestamp_session),
                                np.hstack(DAsessiondf.DA_session),
                                DAsessiondf.last_lever_abs_harp.values, [-4,4], .01)

motion, DA_NMF = compute_NMF_masked(np.hstack(DAsessiondf.deltaF_tdtomato.values), np.hstack(DAsessiondf.deltaF_gfp.values))

snipps_DA_NMF_cp, _ = signal2eventsnippets(np.hstack(DAsessiondf.timestamp_session),
                                DA_NMF,
                                DAsessiondf.cp_abs_harp.values, [-4,4], .01)
snipps_DA_NMF_last_lever, _ = signal2eventsnippets(np.hstack(DAsessiondf.timestamp_session),
                                DA_NMF,
                                DAsessiondf.last_lever_abs_harp.values, [-4,4], .01)

snipps_NMF = zscore(np.concatenate([np.nanmean(snipps_DA_NMF_cp, axis = 0), np.nanmean(snipps_DA_NMF_last_lever, axis = 0)]))
snipps = zscore(np.concatenate([np.nanmean(snipps_DA_cp, axis = 0), np.nanmean(snipps_DA_last_lever, axis = 0)]))

# %%

DLC_PATH = rf'{PATH_STORE_PHOTOMETRY_PICKLES}/{animal}_{date}_dlcDAdf.pkl'

if os.path.exists(DLC_PATH):
    bool_dlc  = True
    dlcDAdf = pd.read_pickle(rf'{DLC_PATH}')
    dlcDAdf['timestamp_session'] = dlcDAdf.apply(lambda x: np.linspace(x.trial_start_harp,x.trial_end_harp,len(x.implantSleeve_y_upsampled)), axis = 1)
else:
    bool_dlc  = False

print(bool_dlc)


#%%

snipps_implantBase_cp, _ = signal2eventsnippets(np.hstack(dlcDAdf.timestamp_session),
                                np.hstack(dlcDAdf.implantBase_y_upsampled.values),
                                DAsessiondf.cp_abs_harp.values, [-4,4], .01)
snipps_implantBase_last_lever, _ = signal2eventsnippets(np.hstack(dlcDAdf.timestamp_session),
                                np.hstack(dlcDAdf.implantBase_y_upsampled.values),
                                DAsessiondf.last_lever_abs_harp.values, [-4,4], .01)

snipps_implantSleeve_cp, _ = signal2eventsnippets(np.hstack(dlcDAdf.timestamp_session),
                                np.hstack(dlcDAdf.implantSleeve_y_upsampled.values),
                                DAsessiondf.cp_abs_harp.values, [-4,4], .01)
snipps_implantSleeve_last_lever, _ = signal2eventsnippets(np.hstack(dlcDAdf.timestamp_session),
                                np.hstack(dlcDAdf.implantSleeve_y_upsampled.values),
                                DAsessiondf.last_lever_abs_harp.values, [-4,4], .01)

snipps_topL_cp, _ = signal2eventsnippets(np.hstack(dlcDAdf.timestamp_session),
                                np.hstack(dlcDAdf.topL_y_upsampled.values),
                                DAsessiondf.cp_abs_harp.values, [-4,4], .01)
snipps_topL_last_lever, _ = signal2eventsnippets(np.hstack(dlcDAdf.timestamp_session),
                                np.hstack(dlcDAdf.topL_y_upsampled.values),
                                DAsessiondf.last_lever_abs_harp.values, [-4,4], .01)

snipps_snout_cp, _ = signal2eventsnippets(np.hstack(dlcDAdf.timestamp_session),
                                np.hstack(dlcDAdf.snout_y_upsampled.values),
                                DAsessiondf.cp_abs_harp.values, [-4,4], .01)
snipps_snout_last_lever, _ = signal2eventsnippets(np.hstack(dlcDAdf.timestamp_session),
                                np.hstack(dlcDAdf.snout_y_upsampled.values),
                                DAsessiondf.last_lever_abs_harp.values, [-4,4], .01)

## poke and lever position
pos_poke = np.nanmean(np.hstack(dlcDAdf.poke_y.values))
pos_lever = np.nanmean(np.hstack(dlcDAdf.lever_y.values))


### xy maps / trajectories could be cool to show as well -- should be done trial by trial, and then avergae of those

#snipps_implantBase_x_cp, _ = signal2eventsnippets(np.hstack(dlcDAdf.timestamp_session),
#                                np.hstack(dlcDAdf.implantBase_x_upsampled.values),
#                                DAsessiondf.cp_abs_harp.values, [-4,4], .01)
#snipps_implantBase_x_last_lever, _ = signal2eventsnippets(np.hstack(dlcDAdf.timestamp_session),
#                                np.hstack(dlcDAdf.implantBase_x_upsampled.values),
#                                DAsessiondf.last_lever_abs_harp.values, [-4,4], .01)

#plt.plot(np.nanmean(snipps_implantBase_x_cp, axis = 0))
#plt.plot(np.nanmean(snipps_implantBase_x_last_lever, axis = 0))

#fig, axs = plt.subplots(1)
#axs.plot(np.nanmean(snipps_implantBase_x_cp, axis = 0), np.nanmean(snipps_implantBase_cp, axis = 0))
#axs.invert_yaxis()


#%%
"""
.########.####..######..
.##........##..##....##.
.##........##..##.......
.######....##..##...####
.##........##..##....##.
.##........##..##....##.
.##.......####..######..
"""

for nn in range(len(neurons_sessiondf)):

    cluster_id = neurons_sessiondf.cluster_id.values[nn]
    spike_times = neurons_sessiondf.spike_times.values[nn]

    ttl_times = blocksdf.query(f'animal == "{animal}" and date == "{date}"').cp_abs_npx_time.values
    spikes_cp = align_spikes_to_ttl(spike_times, ttl_times, window=(-4,4))
    time, FR_cp = compute_FR(spikes_cp, (-4,4), binW = .1)

    ttl_times = blocksdf.query(f'animal == "{animal}" and date == "{date}"').last_lever_abs_npx_time.values
    spikes_last_lever = align_spikes_to_ttl(spike_times, ttl_times, window=(-4,4))
    time, FR_last_lever = compute_FR(spikes_last_lever, (-4,4), binW = .1)


    ## figure

    fig, axs = plt.subplots(4,2, figsize = (8,8), tight_layout = True, sharex = True, sharey = 'row')

    for trial,spikes in enumerate(spikes_cp):
        axs[1,0].plot(spikes, [trial]*len(spikes), '.', ms = .5, color = 'black')

    axs[0,0].plot(time, FR_cp, color = 'black')
    axs[2,0].plot(timeDA, snipps[:len(timeDA)], color = 'purple')
    axs[2,0].plot(timeDA, snipps_NMF[:len(timeDA)], color = 'orange')


    for trial,spikes in enumerate(spikes_last_lever):
        axs[1,1].plot(spikes, [trial]*len(spikes), '.', ms = .5, color = 'black')

    axs[0,1].plot(time, FR_last_lever, color = 'black')
    axs[2,1].plot(timeDA, snipps[-len(timeDA):], color = 'purple', label = 'reg')
    axs[2,1].plot(timeDA, snipps_NMF[-len(timeDA):], color = 'orange', label = 'NMF')
    axs[2,1].legend(frameon = False)


    ## dlc
    #poke and lever
    for ii in range(2):
        axs[3,ii].axhline(pos_lever, color = bodypart_color_dic['lever'], ls = '--')
        axs[3,ii].axhline(pos_poke, color = bodypart_color_dic['poke'], ls = '--')

    axs[3,0].plot(timeDA, np.nanmean(snipps_implantBase_cp, axis = 0), color = bodypart_color_dic['implantBase'])
    axs[3,1].plot(timeDA, np.nanmean(snipps_implantBase_last_lever, axis = 0), color = bodypart_color_dic['implantBase'])

    axs[3,0].plot(timeDA, np.nanmean(snipps_implantSleeve_cp, axis = 0), color = bodypart_color_dic['implantSleeve'])
    axs[3,1].plot(timeDA, np.nanmean(snipps_implantSleeve_last_lever, axis = 0), color = bodypart_color_dic['implantSleeve'])

    axs[3,0].plot(timeDA, np.nanmean(snipps_topL_cp, axis = 0), color = bodypart_color_dic['topL'])
    axs[3,1].plot(timeDA, np.nanmean(snipps_topL_last_lever, axis = 0), color = bodypart_color_dic['topL'])

    #axs[3,0].plot(timeDA, np.nanmean(snipps_snout_cp, axis = 0), color = bodypart_color_dic['snout'])
    #axs[3,1].plot(timeDA, np.nanmean(snipps_snout_last_lever, axis = 0), color = bodypart_color_dic['snout'])

    axs[3,0].invert_yaxis()


    axs[0,0].set_ylabel('FR (Hz)')
    axs[1,0].set_ylabel('trial #')
    axs[2,0].set_ylabel('DA (z)')
    axs[3,0].set_ylabel('y_DLC (px)')

    axs[-1,0].set_xlabel('time since transition (s)')
    axs[-1,1].set_xlabel('time since last press (s)')

    axs[-1,-1].set_xlim(-4,4)


    for ii in range(4):
        for jj in range(2):
            axs[ii,jj].axvline(0, color = 'grey', lw = .5)

    figtitle = f'{animal} {date} | experiment {determine_experiment(DAsessiondf)} | cluster {cluster_id}'
    fig.suptitle(figtitle)

    fig.savefig(rf'{PATH_SAVE_DANEURONS_FIGS}\{figtitle.replace('|','_')}.png', dpi = 300)
    plt.close()


# %%

"""
..######..########....###...
.##....##....##......##.##..
.##..........##.....##...##.
..######.....##....##.....##
.......##....##....#########
.##....##....##....##.....##
..######.....##....##.....##

spike triggered dopamine average
comparison between all DA, or no pressing epochs
"""

bool_ibl_drift = True
#bool_raw_ephys = False

DROPBOX_NEURO_PATH = rf'{DROPBOX_TASK_PATH}\ephys\{animal}'

PATH_SAVE_NEURONS_FIGS = rf'{DROPBOX_TASK_PATH}\analysis_ephys\{animal}_{date}'
if not(os.path.exists(PATH_SAVE_NEURONS_FIGS)):
    os.mkdir(PATH_SAVE_NEURONS_FIGS)

PATH_SAVE_SYNC = glob.glob(fr'{DROPBOX_NEURO_PATH}\{animal}{date}*\*')[0]

if bool_ibl_drift:
    # drift_amplitude
    ibl_sorter_path =  glob.glob(fr'{DROPBOX_NEURO_PATH}\{animal}{date}*\{animal}{date}*\ibl_sorter_results_drift_amplitude')[0]
else:
    ibl_sorter_path =  glob.glob(fr'{DROPBOX_NEURO_PATH}\{animal}{date}*\{animal}{date}*\ibl_sorter_results')[0]

#if bool_raw_ephys:
#    neuro_path = glob.glob(rf"H:\{animal}{date}*\{animal}{date}*")[1]
#    #neuro_path = glob.glob(rf"G:\EPHYS\{animal}{date}*\{animal}{date}*")[1]
#else:
#    neuro_path = 'undefined'

#%%
syncdf = pd.read_pickle(fr'{PATH_SAVE_SYNC}\syncdf.pkl')
exp = determine_experiment(syncdf)

#%%

if exp == 'c':
    hue_variable = 'n_protocols'
    color_palette = color_nprots_blocks
    hue_variable_list = nprots_list
else:
    hue_variable = 'FI'
    color_palette = color_FI_blocks
    hue_variable_list = FI_list

#%%
neuronsdf = pd.read_pickle(fr'{PATH_SAVE_SYNC}\neuronsdf.pkl')
spikes_self_aligned_all = neuronsdf.spikes_self_aligned.values
#%%
#make sure the neuronsdf have the current version of SF labels
cluster_info = pd.read_csv(rf'{ibl_sorter_path}\cluster_info.tsv', sep = '\t')
neuronsdf['SF'] = cluster_info.SF

#%%

#photometry

simpledf = photometrydf.query(f'animal == "{animal}" and date == "{date}"').get(
            ['trialno', 'blockno', 'FI', 'n_protocols',
             'lever_index', 'timestamp_session',
            'DA_session', 'DA_session_ICA',
            'trial_start_harp', 'trial_end_harp',
            'lever_rel_harp', 'lever_abs_harp',
            'poke_rel_harp', 'poke_abs_harp',
            'pump_on_harp', 'pump_off_harp',
            'cp_harp'])

simpledf['trial_duration'] = simpledf['trial_end_harp'] - simpledf['trial_start_harp']
simpledf['trialno_within_block'] = simpledf.groupby('blockno').cumcount()+1
simpledf['trialno_within_block_from_end'] = -1 * (simpledf.groupby('blockno').cumcount(ascending=False) + 1)

simpledf['animal'] = animal
simpledf['date'] = date

simpledf.rename(columns= {'timestamp_session':'time_DA',
                          'DA_session':'DA',
                          'DA_session_ICA':'DA_ICA',
                          'trial_start_harp':'trial_start',
                          'trial_end_harp':'trial_end',
                          'lever_rel_harp':'lever_rel',
                          'lever_abs_harp':'lever_abs',
                          'poke_rel_harp':'poke_rel',
                          'poke_abs_harp':'poke_abs',
                          'pump_on_harp':'rwd_onset',
                          'pump_off_harp':'rwd_offset',
                          'cp_harp':'cp'}, inplace= True)

simpledf['FI'] = (simpledf['FI']/1000).astype(int)

simpledf['cp'] = simpledf.apply(lambda x: x.cp if x.cp < x.FI else np.nan, axis = 1)
simpledf['bool_cp'] = simpledf.cp.apply(lambda x: not(np.isnan(x)))

simpledf['cp_abs'] = simpledf['cp'] + simpledf['trial_start']
simpledf['rwd_onset_abs'] = simpledf['rwd_onset'] + simpledf['trial_start']

simpledf['lever_rel_FInormalised'] = simpledf['lever_rel'] / simpledf['FI']
simpledf['cp_FInormalised'] = simpledf['cp'] / simpledf['FI']

simpledf['trial_in_block'] = simpledf.groupby(['blockno']).cumcount() + 1
simpledf['bool_new_block'] = simpledf['blockno'] != simpledf['blockno'].shift(1)

#simpledf = simpledf.reset_index(drop=True)

for key in ['blockno', 'FI', 'n_protocols']:
    simpledf[f'prev_{key}'] = simpledf.loc[simpledf['bool_new_block'], key].shift(1)
    simpledf[f'prev_{key}'] = simpledf[f'prev_{key}'].ffill()

simpledf['time_DA_rel'] = simpledf.apply(lambda x: np.array(x.time_DA) - x.trial_start, axis = 1)
simpledf['time_DA_after_cp'] = simpledf.apply(lambda x: np.array(x.time_DA_rel) - x.cp, axis = 1)

simpledf['DA_idx_after_cp'] = simpledf.time_DA_after_cp.apply(lambda x: x>=0)
simpledf['DA_after_cp'] = simpledf.apply(lambda x: np.array(x.DA)[x.DA_idx_after_cp], axis = 1)
simpledf['DA_before_cp'] = simpledf.apply(lambda x: np.array(x.DA)[~x.DA_idx_after_cp], axis = 1)


simpledf['tercile_cp_FInormalised'] = pd.qcut(simpledf['cp_FInormalised'], q=3, labels=['T1', 'T2', 'T3'])
terciles_color_dic = {'T1': 'red', 'T2': 'grey', 'T3': 'blue'}
terciles_labels = ['T1', 'T2', 'T3']
terciles_colors = ['red', 'grey', 'blue']
#%%

plt.figure()
plt.plot(simpledf.trial_duration.values, label = 'photometry')
plt.plot(syncdf.loc[0:].query('trial_duration_s > 2').trial_duration_s.values, '--', label = 'ephys')
plt.title('see if trials match')
plt.legend()
plt.show()
#%%
## to align to the neurons
simpledf['npx_trial_start'] = syncdf.query('trial_duration_s > 2').npx_time.values
#%%
plt.plot(simpledf.trial_start)
plt.plot(simpledf.npx_trial_start)
#%%
plt.plot(simpledf.npx_trial_start - simpledf.trial_start)
#%%
## DA on cp terciles split by block

simpledf['tercile_cp_withinblock'] = (
    simpledf.query('blockno < 4').groupby('blockno')['cp']
    .transform(lambda x: pd.qcut(x, q=3, labels=['T1', 'T2', 'T3']))
)

#%%

## simpledf was already stored in a place
#%%

"""
..######..####.##.....##.########..##.......########.########..########
.##....##..##..###...###.##.....##.##.......##.......##.....##.##......
.##........##..####.####.##.....##.##.......##.......##.....##.##......
..######...##..##.###.##.########..##.......######...##.....##.######..
.......##..##..##.....##.##........##.......##.......##.....##.##......
.##....##..##..##.....##.##........##.......##.......##.....##.##......
..######..####.##.....##.##........########.########.########..##......
"""

simpledf = pd.read_pickle(rf'{PATH_DANEURONS_ANALYSIS}\{animal}_{date}_simpledf.pkl')

simpledf['time_npx'] = simpledf.apply(lambda x: np.hstack(x.time_DA) + (x.npx_trial_start - x.trial_start), axis = 1)
simpledf['lever_abs_npx'] = simpledf.apply(lambda x: np.array(x.lever_rel) + x.npx_trial_start, axis = 1)

#%%
"""
..######..########....###.......########.####..######..
.##....##....##......##.##......##........##..##....##.
.##..........##.....##...##.....##........##..##.......
..######.....##....##.....##....######....##..##...####
.......##....##....#########....##........##..##....##.
.##....##....##....##.....##....##........##..##....##.
..######.....##....##.....##....##.......####..######..
"""

snippet_window = [-1,1]

for nn in range(len(neurons_sessiondf)):

    spike_times = neurons_sessiondf.spike_times.values[nn]

    STA_all, time_STA = signal2eventsnippets(np.hstack(simpledf.time_npx.values),
                                    np.hstack(simpledf.DA.values),
                                    spike_times, snippet_window, .01)

    #DA_masked_during_cp = simpledf.apply(
    #    lambda x: np.where(x.DA_idx_after_cp, np.nan, x.DA), axis=1)

    #STA_before_cp, _ = signal2eventsnippets(np.hstack(simpledf.time_npx.values),
    #                                np.hstack(DA_masked_during_cp.values),
    #                                spike_times, snippet_window, .01)


    excluding_presses_window = [.5,.5]
    idx_around_lever_presses = get_event_indices(np.hstack(simpledf.time_npx.values),
                                                np.hstack(simpledf.lever_abs_npx.values),
                                                excluding_presses_window, fs = 1000)
    DA_excluding_presses = np.array(np.hstack(simpledf.DA.values), dtype=float).copy()
    DA_excluding_presses[idx_around_lever_presses] = np.nan

    STA_excluding_presses, _ = signal2eventsnippets(np.hstack(simpledf.time_npx.values),
                                    DA_excluding_presses,
                                    spike_times, snippet_window, .01)

    excluding_presses_window_pre = [.5,0]
    idx_around_lever_presses_pre = get_event_indices(np.hstack(simpledf.time_npx.values),
                                                np.hstack(simpledf.lever_abs_npx.values),
                                                excluding_presses_window_pre, fs = 1000)
    DA_excluding_presses_pre = np.array(np.hstack(simpledf.DA.values), dtype=float).copy()
    DA_excluding_presses_pre[idx_around_lever_presses_pre] = np.nan

    STA_excluding_presses_pre, _ = signal2eventsnippets(np.hstack(simpledf.time_npx.values),
                                    DA_excluding_presses_pre,
                                    spike_times, snippet_window, .01)
    
    excluding_presses_window_nolag = [0,0]
    idx_around_lever_presses_nolag = get_event_indices(np.hstack(simpledf.time_npx.values),
                                                np.hstack(simpledf.lever_abs_npx.values),
                                                excluding_presses_window_pre, fs = 1000)
    DA_excluding_presses_nolag = np.array(np.hstack(simpledf.DA.values), dtype=float).copy()
    DA_excluding_presses_nolag[idx_around_lever_presses_nolag] = np.nan

    STA_excluding_presses_nolag, _ = signal2eventsnippets(np.hstack(simpledf.time_npx.values),
                                    DA_excluding_presses_nolag,
                                    spike_times, snippet_window, .01)

    fig, axs = plt.subplots(1,4, figsize = (16,4), tight_layout = True)

    axs[0].plot(time_STA, np.nanmean(STA_all, axis = 0), lw = 1)
    #axs[1].plot(time_STA, np.nanmean(STA_before_cp, axis = 0), lw = 1)
    axs[1].plot(time_STA, np.nanmean(STA_excluding_presses_nolag, axis = 0), lw = 1)
    axs[2].plot(time_STA, np.nanmean(STA_excluding_presses_pre, axis = 0), lw = 1)
    axs[3].plot(time_STA, np.nanmean(STA_excluding_presses, axis = 0), lw = 1)

    for ii in range(4):
        axs[ii].axvline(0, color = 'grey', ls = 'dashed', alpha =0.5)
        axs[ii].set_xlabel('time since spike (s)')

    axs[0].set_title('all DA')
    #axs[1].set_title('high pressing masked')
    axs[1].set_title(f'presses excluded {excluding_presses_window_nolag}')
    axs[2].set_title(f'presses excluded {excluding_presses_window_pre}')
    axs[3].set_title(f'presses excluded {excluding_presses_window}')

    figtitle = f'{animal} {date} | experiment {exp} | cluster {neurons_sessiondf.cluster_id.values[nn]} | STA DA comparisons'
    fig.suptitle(figtitle)

    fig.savefig(rf'{PATH_SAVE_STA_FIGS}\aligned_to_spike\{figtitle.replace("|","_")}.png', dpi = 300)
    plt.close()
# %%

"""
.####.##....##.##.....##.########.########..########.########.########......######..########....###......
..##..###...##.##.....##.##.......##.....##....##....##.......##.....##....##....##....##......##.##.....
..##..####..##.##.....##.##.......##.....##....##....##.......##.....##....##..........##.....##...##....
..##..##.##.##.##.....##.######...########.....##....######...##.....##.....######.....##....##.....##...
..##..##..####..##...##..##.......##...##......##....##.......##.....##..........##....##....#########...
..##..##...###...##.##...##.......##....##.....##....##.......##.....##....##....##....##....##.....##...
.####.##....##....###....########.##.....##....##....########.########......######.....##....##.....##...

instead of aligning DA to spikes, align spikes to DA peaks
1 - identify DA peaks
2 - align spikes to those peaks
"""

DA_full = np.hstack(simpledf.DA.values)
DA_full_z = (DA_full - np.nanmean(DA_full))/np.nanstd(DA_full)
fs = 100
#%%
template = create_da_template(fs = fs, rise_time = 0.1, decay_time=.7)
corr = correlate(DA_full_z, template, mode='same') / np.sqrt(len(template))
peaks, _ = find_peaks(corr, height = 0.5, distance = 150)
#%%
peak_times = np.hstack(simpledf.time_npx.values)[peaks]
#%%
## correction to peak times (needed due to the correlation mode 'same')

template_peak_idx = np.argmax(template) 
center_idx = len(template) // 2 

# The actual shift to align the DA APEX to t=0
# shift = (1.0s) - (0.1s) = 0.9s
correction_to_apex = (center_idx - template_peak_idx) / fs
corrected_peak_times = peak_times - center_idx/fs #correction_to_apex

#%%
plt.plot(np.hstack(simpledf.time_npx.values), np.hstack(simpledf.DA.values), lw = 1)
plt.plot(peak_times, DA_full[peaks], 'o')
plt.xlim(490,520)
plt.plot(np.arange(494,496,.01),.04+template/50)


# %%
#nn = -2


# DA peaks aligned to themselves -- sanity check for the template matching
snippes, t_snippes = signal2eventsnippets(np.hstack(simpledf.time_npx.values),
                                np.hstack(simpledf.DA.values),
                                corrected_peak_times, [-4,4], .01)
snippes = drop_nan_rows_in_matrix(snippes)


for cluster_id in neurons_sessiondf.cluster_id.values:

    spike_times = neurons_sessiondf.query(f'cluster_id == {cluster_id}').spike_times.values[0]
    spikes_aligned_DA_peaks = align_spikes_to_ttl(spike_times, corrected_peak_times, window = (-4,4))

    fig, axs = plt.subplots(3, figsize = (8,8), tight_layout = True, sharex=True)

    time, FR = compute_FR(spikes_aligned_DA_peaks, (-4,4), binW = .1)
    axs[0].plot(time, FR, color = 'black', lw = 1)

    for ii, spike in enumerate(spikes_aligned_DA_peaks):
        axs[1].plot(spike, ii*np.ones(len(spike)), '|', color = 'black')

    for ii in range(3):
        axs[ii].axvline(0, color = 'grey', ls = 'dashed')

    axs[2].plot(t_snippes, snippes.T, color = 'black', alpha = 0.01)
    axs[2].plot(t_snippes, np.nanmean(snippes, axis = 0), color = 'white')

    axs[2].set_ylim(np.nanquantile(snippes, .05), np.nanquantile(snippes, .95))
    axs[1].set_ylim(0)
    axs[2].set_xlim(-4,4)

    axs[0].set_ylabel('FR (Hz)')
    axs[1].set_ylabel('DA peak #')
    axs[2].set_ylabel('DA peaks')
    axs[2].set_xlabel('time since DA peak (s)')

    figtitle = f'{animal} {date} | experiment {exp} | cluster {cluster_id} | spikes aligned to DA peaks'
    fig.suptitle(figtitle)

    fig.savefig(rf'{PATH_SAVE_STA_FIGS}\aligned_to_DApeak\{figtitle.replace("|","_")}.png', dpi = 300)
    plt.close()
# %%

peak_times
#%%

lvr_peaks = align_spikes_to_ttl(np.hstack(simpledf.lever_abs_npx.values), corrected_peak_times, (-4,4))
#%%
fig, axs = plt.subplots(2, figsize = (8,4), tight_layout = True, sharex = True)

for ii in range(2):
    axs[ii].axvline(0, color = 'grey', ls = 'dashed')

for ii,lvr in enumerate(lvr_peaks):
    plt.plot(lvr, ii*np.ones(len(lvr)), '|', color = 'black')

time, FR_lvr = compute_FR(lvr_peaks, (-4,4), binW = .05)
latency_DApeak_press = time[np.argmax(FR_lvr)]

axs[0].text(-3.9, .5*max(zscore(FR_lvr)),
            f'latency DApeak to lever press: {latency_DApeak_press:.2f}s')#, color = 'red')

axs[0].plot(time, zscore(FR_lvr), color = 'black', lw = 1, label = 'lever presses PSTH')
axs[0].plot(t_snippes, zscore(np.nanmean(snippes, axis = 0)), color = 'purple', lw = 1, label = 'av. DA peaks')
axs[0].legend(frameon = False)

axs[0].set_ylabel('z-scored')
axs[1].set_ylabel('DA peak #')
axs[1].set_xlabel('time since DA peak (s)')
axs[1].set_ylim(0)
axs[1].set_xlim(-4,4)
# %%
# %%
plt.plot(FR_lvr)
# %%


"""
.##.....##.########..######......###.......########.####..######..
.###...###.##.......##....##....##.##......##........##..##....##.
.####.####.##.......##.........##...##.....##........##..##.......
.##.###.##.######...##...####.##.....##....######....##..##...####
.##.....##.##.......##....##..#########....##........##..##....##.
.##.....##.##.......##....##..##.....##....##........##..##....##.
.##.....##.########..######...##.....##....##.......####..######..
"""


snippes, t_snippes = signal2eventsnippets(np.hstack(simpledf.time_npx.values),
                                np.hstack(simpledf.DA.values),
                                corrected_peak_times, [-4,4], .01)
snippes = drop_nan_rows_in_matrix(snippes)

## DA around lever presses
DA_around_lever, _ = signal2eventsnippets(np.hstack(simpledf.time_npx.values),
                    np.hstack(simpledf.DA.values),
                    np.hstack(simpledf.lever_abs_npx.values), [-1,1], .01)



for cluster_id in neurons_sessiondf.cluster_id.values:

    spike_times = neurons_sessiondf.query(f'cluster_id == {cluster_id}').spike_times.values[0]
    spikes_aligned_DA_peaks = align_spikes_to_ttl(spike_times, corrected_peak_times, window = (-4,4))


    snippet_window = [-1,1]

    STA_all, time_STA = signal2eventsnippets(np.hstack(simpledf.time_npx.values),
                                    np.hstack(simpledf.DA.values),
                                    spike_times, snippet_window, .01)
    STA_all = drop_nan_rows_in_matrix(STA_all)

    excluding_presses_window = [.4,.2]
    idx_around_lever_presses = get_event_indices(np.hstack(simpledf.time_npx.values),
                                                np.hstack(simpledf.lever_abs_npx.values),
                                                excluding_presses_window, fs = 1000)
    DA_excluding_presses = np.array(np.hstack(simpledf.DA.values), dtype=float).copy()
    DA_excluding_presses[idx_around_lever_presses] = np.nan
    STA_excluding_presses, _ = signal2eventsnippets(np.hstack(simpledf.time_npx.values),
                                    DA_excluding_presses,
                                    spike_times, snippet_window, .01)
    STA_excluding_presses = drop_nan_rows_in_matrix(STA_excluding_presses)


    spikes_lever = align_spikes_to_ttl(spike_times, np.hstack(simpledf.lever_abs_npx.values), window = (-4,4))
    time, FR_spikes_lever = compute_FR(spikes_lever, (-4,4), binW = .1)


    spikes_aligned_to_presses = align_spikes_to_ttl(spike_times, np.hstack(simpledf.lever_abs_npx.values), window = (-4,4))




    fig, axs = plt.subplots(2,7, figsize = (32,8), tight_layout = True, sharex='col')


    ## STA DA around spikes

    axs[0,1].plot(time_STA, np.nanmean(STA_all, axis = 0), color = 'black')
    axs[1,1].imshow(zscore(STA_all, axis = 1), aspect = 'auto', vmin = -1, vmax = 1, cmap = 'bone', extent = (-1,1,0,len(STA_all)))

    axs[0,2].plot(time_STA, np.nanmean(STA_excluding_presses, axis = 0), color = 'black')
    axs[1,2].imshow(zscore(STA_excluding_presses, axis = 1), aspect = 'auto', vmin = -1, vmax = 1, cmap = 'bone', extent = (-1,1,0,len(STA_excluding_presses)))

    for ii in [1,2]:
        axs[-1,ii].set_xlabel('time since spike (s)')
        axs[-1,ii].set_xlabel('time since spike (s)')


    ## STA spikes around DA peaks

    time, FR = compute_FR(spikes_aligned_DA_peaks, (-4,4), binW = .1)
    axs[0,3].plot(time, FR, color = 'black')

    for ii, spike in enumerate(spikes_aligned_DA_peaks):
        axs[1,3].plot(spike, ii*np.ones(len(spike)), '|', color = 'black')

    axsDA = axs[0,3].twinx()
    axsDA.plot(t_snippes, (np.nanmean(snippes, axis = 0)), color = 'purple', label = 'av. DA peaks')
    #axs[0,2].plot(t_snippes, zscore(np.nanmean(snippes, axis = 0)), color = 'purple', lw = 1, label = 'av. DA peaks')

    axs[1,3].set_ylim(0, len(spikes_aligned_DA_peaks))
    axs[1,3].set_xlim(-4,4)

    axs[-1,3].set_xlabel('time since DA peak (s)')


    ## lever presses aligned to DA peaks

    for ii,lvr in enumerate(lvr_peaks):
        axs[1,4].plot(lvr, ii*np.ones(len(lvr)), '|', color = 'black')

    time, FR_lvr = compute_FR(lvr_peaks, (-4,4), binW = .05)
    latency_DApeak_press = time[np.argmax(FR_lvr)]

    axs[0,4].text(-3.9, .5*max((FR_lvr)),
                f'DApeak to press\npeak: {latency_DApeak_press:.2f}s')#, color = 'red')

    axs[0,4].plot(time, (FR_lvr), color = 'black', label = 'lever presses PSTH')

    axsDA2 = axs[0,4].twinx()
    axsDA2.plot(t_snippes, (np.nanmean(snippes, axis = 0)), color = 'purple', label = 'av. DA peaks')
    #axs[0,3].legend(frameon = False)

    axs[-1,4].set_xlabel('time since DA peak (s)')
    axs[-1,4].set_ylim(0, len(lvr_peaks))
    axs[-1,4].set_xlim(-4,4)


    ## DA around lever presses

    axs[0,5].plot(time_STA, np.nanmean(DA_around_lever, axis = 0), color = 'black')
    axs[1,5].imshow(zscore(DA_around_lever, axis = 1), aspect = 'auto', vmin = -1, vmax = 1, cmap = 'bone', extent = (-1,1,0,len(DA_around_lever)))

    yy_maxDA = np.nanmean(DA_around_lever, axis = 0).max()
    axs[0,5].plot(-.4, yy_maxDA, 'v', color = 'purple')
    axs[0,5].plot(.2, yy_maxDA, 'v', color = 'purple')

    for ii in range(2):
        axs[ii,5].axvline(-.4, color = 'purple', ls = 'dashed', lw = 1)
        axs[ii,5].axvline(.2, color = 'purple', ls = 'dashed', lw = 1)

    axs[0,5].text(-.1, yy_maxDA*.5, 'exclusion\nwindow', color = 'purple', ha = 'center')

    axs[-1,5].set_xlabel('time since lever press (s)')
    axs[-1,5].set_xlim(-1,1)


    ## spikes aligned to lever presses

    for ii, spike in enumerate(spikes_aligned_to_presses):
        axs[1,0].plot(spike, ii*np.ones(len(spike)), '|', color = 'black')

    t_spikes_lever, FR_spikes_lever = compute_FR(spikes_aligned_to_presses, (-4,4), binW = .1)
    axs[0,0].plot(t_spikes_lever, FR_spikes_lever, color = 'black')

    axs[-1,0].set_xlim(-4,4)
    axs[-1,0].set_ylim(0, len(spikes_aligned_to_presses))
    axs[-1,0].set_xlabel('time since lever press (s)')


    ## DA aligned to DA spikes, comparing with template (sanity check)
    axs[0,6].plot(t_snippes, np.nanmean(snippes, axis = 0), color = 'purple')
    axstemplate = axs[0,6].twinx()
    axstemplate.plot(np.linspace(0,2,len(template)),template, color = 'orange', ls = 'dashed', label = 'template')
    axs[1,6].imshow(snippes, aspect = 'auto', vmin = np.nanquantile(snippes, .05), vmax = np.nanquantile(snippes, .95), cmap = 'bone', extent = (-4,4,0,len(snippes)))
    axs[1,6].set_ylim(0,len(snippes))
    axstemplate.legend(frameon = False, loc = 'upper left')

    for ii in range(2):
        for jj in range(7):
            axs[ii,jj].axvline(0, color = 'grey', ls = 'dashed')


    ## subtitles
    axs[0,1].set_title('DA around spikes')
    axs[0,2].set_title(f'DA around spikes, excluding presses {excluding_presses_window}', fontsize = 14)
    axs[0,3].set_title('spikes aligned to DA peaks')
    axs[0,4].set_title('lever presses aligned to DA peaks')
    axs[0,5].set_title('DA around lever presses')
    axs[0,0].set_title('spikes aligned to lever presses')
    axs[0,6].set_title('DA aligned to DA peaks (sanity check)', fontsize = 14)

    axs[0,1].set_ylabel('DA (a.u.)')
    axs[0,2].set_ylabel('DA (a.u.)')
    axs[0,3].set_ylabel('FR (Hz)')
    axs[0,4].set_ylabel('press rate (Hz)')
    axs[0,5].set_ylabel('DA (a.u.)')
    axs[0,0].set_ylabel('FR (Hz)')
    axs[0,6].set_ylabel('DA (a.u.)')
    #axsDA.set_ylabel('DA (a.u.)', color = 'purple')
    #axsDA2.set_ylabel('DA (a.u.)', color = 'purple')
    #axstemplate.set_ylabel('template', color = 'orange')

    axs[1,1].set_ylabel('spike #')
    axs[1,2].set_ylabel('spike #')
    axs[1,3].set_ylabel('DA peak #')
    axs[1,4].set_ylabel('DA peak #')
    axs[1,5].set_ylabel('press #')
    axs[1,0].set_ylabel('press #')
    axs[1,6].set_ylabel('DA peak #')


    figtitle = f'{animal} {date} | experiment {exp} | cluster {cluster_id} | STA DA averages | spikes and lever presses aligned to DA peaks'
    fig.suptitle(figtitle)

    fig.savefig(rf'{PATH_SAVE_STA_FIGS}\{figtitle.replace("|","_")}.png', dpi = 300)

    plt.close()


# %%

#axs[2,1].plot(t_snippes, snippes.T, color = 'black', alpha = 0.01)
#axs[2,1].plot(t_snippes, np.nanmean(snippes, axis = 0), color = 'white')
#axs[2,1].set_ylim(np.nanquantile(snippes, .05), np.nanquantile(snippes, .95))
#axs[1,1].set_ylim(0)
#axs[2,1].set_xlim(-4,4)


# %%


# %%
