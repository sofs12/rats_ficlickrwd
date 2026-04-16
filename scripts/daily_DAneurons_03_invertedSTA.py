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
from scipy.signal import find_peaks, correlate

from pathlib import Path

from ratcode.config.paths import PATH_STORE_PICKLES, DROPBOX_TASK_PATH, PATH_DANEURONS_ANALYSIS
from ratcode.common.logging import determine_experiment
from ratcode.common.colorcodes import *
from ratcode.behavior import change_point
from ratcode.photometry.photometry import drop_nan_rows_in_matrix, get_prediction, quantile_regression, signal2eventsnippets, find_poly, segment_and_fit_function, butter_filter, mask_jumps, make_continuous, query_and_compute_snippets, plot_snippets
from ratcode.ephys.neurons import get_psths_across_cells, align_spikes_to_ttl, compute_FR, load_ibl_sorter, determine_cell_type, produce_neuron_fig, produce_mega_neuron_fig, compute_zscore, get_psths_smooth, half_gaussian_kernel, do_PCA, get_PCA_windows
from ratcode.common.dataframe import group_and_listify
from ratcode.common.time import convert_date_bonsai, convert_timestamp
from ratcode.common.math import drop_nans_matrix
from ratcode.common.colorcodes import FI_order, color_FI_blocks, rwd_order, color_rwd_blocks
from ratcode.common.plotting import remove_legend

from ratcode.init import setup

setup()
#%%
def create_da_template(fs, rise_time=0.1, decay_time=0.4):
    """Creates a synthetic DA transient template."""
    t_temp = np.arange(0, 2, 1/fs)
    # Alpha-function style: (t/tau) * exp(-t/tau)
    # Or simplified: rapid rise then exponential decay
    template = (1 - np.exp(-t_temp / rise_time)) * np.exp(-t_temp / decay_time)
    return template / np.max(template) # Normalize height

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
#%%
animal = 'Cadmium'
date = '260409'


## now looking at my old code, 01_DAneurons.py
DANEURONS_PATH_HOME = os.path.join(DROPBOX_TASK_PATH, 'analysis_DAneurons')
DANEURONS_PATH = os.path.join(DANEURONS_PATH_HOME, f'{animal}_{date}')
if not os.path.exists(DANEURONS_PATH):
    os.makedirs(DANEURONS_PATH)

PATH_SAVE_STA_FIGS = os.path.join(PATH_DANEURONS_ANALYSIS, rf'{animal}_{date}/STA_DA')
if not os.path.exists(PATH_SAVE_STA_FIGS):
    os.makedirs(PATH_SAVE_STA_FIGS)
    os.makedirs(os.path.join(PATH_SAVE_STA_FIGS,'aligned_to_spike'))
    os.makedirs(os.path.join(PATH_SAVE_STA_FIGS,'aligned_to_DApeak'))


#neurons df
EPHYS_PATH = os.path.join(DROPBOX_TASK_PATH, 'ephys', animal)
SAVE_SYNC_PATH = glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\*')[0]
IBL_SORTER_PATH =  glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\{animal}{date}*\ibl_sorter_results_drift_amplitude')[0]
neuronsdf = pd.read_pickle(fr'{SAVE_SYNC_PATH}\neuronsdf.pkl')

simpledf = pd.read_pickle(rf'{DANEURONS_PATH_HOME}\{animal}_{date}_simpledf.pkl')
simpledf['time_npx'] = simpledf.apply(lambda x: np.hstack(x.time_DA) + (x.npx_trial_start - x.trial_start), axis = 1)
simpledf['lever_abs_npx'] = simpledf.apply(lambda x: np.array(x.lever_rel) + x.npx_trial_start, axis = 1)
exp = determine_experiment(simpledf)
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


for cluster_id in neuronsdf.query('(SF == "good" or SF == "ok") and (cell_type == "MSN" or cell_type == "TAN")').cluster_id.values:

    spike_times = neuronsdf.query(f'cluster_id == {cluster_id}').spike_times.values[0]

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

    figtitle = f'{animal} {date} | exp {exp} | cluster {cluster_id} | {neuronsdf.query(f"cluster_id == {cluster_id}").cell_type.values[0]} | STA DA comparisons'
    fig.suptitle(figtitle)

    fig.savefig(rf'{PATH_SAVE_STA_FIGS}\aligned_to_spike\{figtitle.replace("|","_")}.png', dpi = 300)
    plt.close()

#%%
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

# DA peaks aligned to themselves -- sanity check for the template matching
snippes, t_snippes = signal2eventsnippets(np.hstack(simpledf.time_npx.values),
                                np.hstack(simpledf.DA.values),
                                corrected_peak_times, [-4,4], .01)
snippes = drop_nan_rows_in_matrix(snippes)


for cluster_id in neuronsdf.query('(SF == "good" or SF == "ok") and (cell_type == "MSN" or cell_type == "TAN")').cluster_id.values:

    spike_times = neuronsdf.query(f'cluster_id == {cluster_id}').spike_times.values[0]
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



for cluster_id in neuronsdf.query('(SF == "good" or SF == "ok") and (cell_type == "MSN" or cell_type == "TAN")').cluster_id.values:

    spike_times = neuronsdf.query(f'cluster_id == {cluster_id}').spike_times.values[0]
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
