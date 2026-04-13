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

"""
....###....##....##.####.##.....##....###....##.......########.....###....########.########
...##.##...###...##..##..###...###...##.##...##.......##.....##...##.##......##....##......
..##...##..####..##..##..####.####..##...##..##.......##.....##..##...##.....##....##......
.##.....##.##.##.##..##..##.###.##.##.....##.##.......##.....##.##.....##....##....######..
.#########.##..####..##..##.....##.#########.##.......##.....##.#########....##....##......
.##.....##.##...###..##..##.....##.##.....##.##.......##.....##.##.....##....##....##......
.##.....##.##....##.####.##.....##.##.....##.########.########..##.....##....##....########
"""

animal = 'Ruthenium'
date = '260327'


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


STAdf = pd.read_pickle(rf'{DANEURONS_PATH}\STAdf.pkl')
# %%

STAdf.keys()
# %%
neuronsdf.query('cluster_id in @STAdf.cluster_id').cell_type.value_counts()
# %%
STAdf['cell_type'] = neuronsdf.query('cluster_id in @STAdf.cluster_id').cell_type.values
# %%
plt.figure()
allSTAs = STAdf.query('cell_type == "MSN"').av_DA_around_spikes.values
allSTAs = np.vstack(allSTAs)
plt.imshow(zscore(allSTAs, axis = 1), aspect = 'auto')
#%%
plt.plot(np.vstack(allSTAs).T)
#%%
for i, sta in enumerate(allSTAs):
    plt.plot(sta, color = 'gray', alpha = 0.5)
# %%
index_order, loadings, PC_space = do_PCA(zscore(allSTAs, axis = 1))

# %%
plt.plot(loadings[:,0], loadings[:,1], 'o')
# %%
plt.plot(PC_space[:,0], PC_space[:,1])
# %%
plt.plot(PC_space[:,0])
plt.plot(PC_space[:,1])
#%%

fig, axs = plt.subplots(2,2, figsize = (8,4), tight_layout = True)

axs[0,0].imshow(zscore(allSTAs[index_order], axis = 1), aspect = 'auto')
axs[0,0].axvline(400, color = 'purple', ls = '--')
axs[0,0].axvline(350, color = 'purple', ls = '--')
axs[0,0].axvline(450, color = 'purple', ls = '--')


#axs[0,1].plot(PC_space[:,0], PC_space[:,1])
axs[0,1].plot(zscore(allSTAs[index_order], axis = 1)[:,400], '.')
axs[0,1].axhline(0, color = 'purple', ls = '--')


for i, sta in enumerate(allSTAs):
    axs[1,0].plot(zscore(sta), color = 'gray', alpha = 0.5)

#%%

## poor man's classification: in a vicinity of the spike, is the average DA higher or lower than farther away?
## which is roughly equivalent to deciding if the STA zscored is >0 or <0 at the time of the spike (but that's not what we're doing here)

dMSNs_index = np.mean(allSTAs[:,350:450], axis = 1) > np.mean(np.hstack([allSTAs[:,:350], allSTAs[:,450:]]), axis = 1)
iMSNs_index = ~dMSNs_index


#%%

allSTAs_dMSN = allSTAs[dMSNs_index]
allSTAs_iMSN = allSTAs[iMSNs_index]

fig, axs = plt.subplots(1,2, figsize = (8,4), tight_layout = True)
axs[0].plot(zscore(allSTAs[dMSNs_index], axis = 1).T, color = 'blue', alpha = 0.1, lw = 1)
axs[0].plot(zscore(allSTAs[iMSNs_index], axis = 1).T, color = 'red', alpha = 0.1, lw = 1)

#%%
STAdf['MSN_type'] = 'unclassified'
STAdf.loc[STAdf.query(f'cell_type == "MSN"').index[dMSNs_index], 'MSN_type'] = 'dMSN'
STAdf.loc[STAdf.query(f'cell_type == "MSN"').index[iMSNs_index], 'MSN_type'] = 'iMSN'

#%%
STAdf['animal'] = animal
STAdf['date'] = date
#%%
STAdf.to_pickle(rf'{DANEURONS_PATH}\STAdf_classified.pkl')
#%%

"""
.########...#######..########..##.....##.##..........###....########.####..#######..##....##.......###....########....########..##......##.########.
.##.....##.##.....##.##.....##.##.....##.##.........##.##......##.....##..##.....##.###...##......##.##......##.......##.....##.##..##..##.##.....##
.##.....##.##.....##.##.....##.##.....##.##........##...##.....##.....##..##.....##.####..##.....##...##.....##.......##.....##.##..##..##.##.....##
.########..##.....##.########..##.....##.##.......##.....##....##.....##..##.....##.##.##.##....##.....##....##.......########..##..##..##.##.....##
.##........##.....##.##........##.....##.##.......#########....##.....##..##.....##.##..####....#########....##.......##...##...##..##..##.##.....##
.##........##.....##.##........##.....##.##.......##.....##....##.....##..##.....##.##...###....##.....##....##.......##....##..##..##..##.##.....##
.##.........#######..##.........#######..########.##.....##....##....####..#######..##....##....##.....##....##.......##.....##..###..###..########.

align population at reward and at cp
"""

iMSN_clusters = STAdf.query(f'MSN_type == "iMSN"').index.tolist()
dMSN_clusters = STAdf.query(f'MSN_type == "dMSN"').index.tolist()

neuronsdf.query(f'cluster_id in {iMSN_clusters}').spike_times.values
#%%

simpledf['cp_abs_npx_time'] = simpledf.apply(lambda x: x.npx_trial_start + x.cp if not np.isnan(x.cp) else np.nan, axis = 1)
simpledf['lever_abs_npx_time'] = simpledf.apply(lambda x: np.array(x.lever_rel) + x.npx_trial_start, axis = 1)
simpledf['last_lever_abs_npx_time'] = simpledf.lever_abs_npx_time.apply(lambda x: x[-1])
#%%
simpledf.cp_abs_npx_time
#%%

#fig, axs = plt.subplots(4,3, tight_layout = True, figsize = (12,8), height_ratios=[1,4,1,4], sharex = 'col')
fig, axs = plt.subplots(3,3, tight_layout = True, figsize = (12,8), height_ratios=[2,4,4], sharex = True, dpi = 300)

## iMSN example

cluster_id = iMSN_clusters[2]

## transition point
spike_times = neuronsdf.query(f'cluster_id == {cluster_id}').spike_times.values[0]
ttl_times = simpledf.cp_abs_npx_time.values
lalala = align_spikes_to_ttl(spike_times, ttl_times,(-10,10))
for trial,spikes in enumerate(lalala):
    axs[1,0].plot(spikes, [trial]*len(spikes), '.', ms = 2, color = 'black')
axs[1,0].set_ylim(0,len(lalala))
time, FR = compute_FR(lalala, (-10,10), binW = .2)
axs[0,0].plot(time, FR, color = 'red')
axs[0,0].set_ylabel("iMSN FR (Hz)", color="red")
axs[0,0].tick_params(axis="y", colors="red")


## any lever press
spike_times = neuronsdf.query(f'cluster_id == {cluster_id}').spike_times.values[0]
ttl_times = np.hstack(simpledf.lever_abs_npx_time.values)
lalala = align_spikes_to_ttl(spike_times, ttl_times,(-10,10))
for trial,spikes in enumerate(lalala):
    axs[1,1].plot(spikes, [trial]*len(spikes), '.', ms = 2, color = 'black')
axs[1,1].set_ylim(0,len(lalala))
time, FR = compute_FR(lalala, (-10,10), binW = .2)
axs[0,1].plot(time, FR, color = 'red')
#axs[0,1].set_ylabel("dMSN (Hz)", color="blue")
axs[0,1].tick_params(axis="y", colors="red")


## last press
spike_times = neuronsdf.query(f'cluster_id == {cluster_id}').spike_times.values[0]
ttl_times = simpledf.last_lever_abs_npx_time.values
lalala = align_spikes_to_ttl(spike_times, ttl_times,(-10,10))
for trial,spikes in enumerate(lalala):
    axs[1,-1].plot(spikes, [trial]*len(spikes), '.', ms = 2, color = 'black')
axs[1,-1].set_ylim(0,len(lalala))
time, FR = compute_FR(lalala, (-10,10), binW = .2)
axs[0,2].plot(time, FR, color = 'red')
#axs[0,2].set_ylabel("dMSN (Hz)", color="blue")
axs[0,2].tick_params(axis="y", colors="red")


## dMSN example
#date = '250418'
cluster_id = dMSN_clusters[2]

## transition point
spike_times = neuronsdf.query(f'cluster_id == {cluster_id}').spike_times.values[0]
ttl_times = simpledf.cp_abs_npx_time.values
lalala = align_spikes_to_ttl(spike_times, ttl_times,(-10,10))
for trial,spikes in enumerate(lalala):
    axs[-1,0].plot(spikes, [trial]*len(spikes), '.', ms = 2, color = 'black')
axs[-1,0].set_ylim(0,len(lalala))
time, FR = compute_FR(lalala, (-10,10), binW = .2)

ax_twin = axs[0,0].twinx()
ax_twin.plot(time, FR, color = 'blue')
#ax_twin.set_ylabel("iMSN (Hz)", color="red", rotation = -90, labelpad=15)
ax_twin.tick_params(axis="y", colors="blue")
ax_twin.spines["right"].set_visible(True) 
ax_twin.spines["right"].set_color("blue")
ax_twin.spines["left"].set_color("red")


## any lever press
spike_times = neuronsdf.query(f'cluster_id == {cluster_id}').spike_times.values[0]
ttl_times = np.hstack(simpledf.lever_abs_npx_time.values)
lalala = align_spikes_to_ttl(spike_times, ttl_times,(-10,10))
for trial,spikes in enumerate(lalala):
    axs[-1,1].plot(spikes, [trial]*len(spikes), '.', ms = 2, color = 'black')
axs[-1,1].set_ylim(0,len(lalala))
time, FR = compute_FR(lalala, (-10,10), binW = .2)
ax_twin = axs[0,1].twinx()
ax_twin.plot(time, FR, color = 'blue')
#ax_twin.set_ylabel("iMSN (Hz)", color="red", rotation = -90, labelpad=15)
ax_twin.tick_params(axis="y", colors="blue")
ax_twin.spines["right"].set_visible(True) 
ax_twin.spines["right"].set_color("blue")
ax_twin.spines["left"].set_color("red")

## last press
spike_times = neuronsdf.query(f'cluster_id == {cluster_id}').spike_times.values[0]
ttl_times = simpledf.last_lever_abs_npx_time.values
lalala = align_spikes_to_ttl(spike_times, ttl_times,(-10,10))
for trial,spikes in enumerate(lalala):
    axs[-1,-1].plot(spikes, [trial]*len(spikes), '.', ms = 2, color = 'black')
axs[-1,-1].set_ylim(0,len(lalala))
time, FR = compute_FR(lalala, (-10,10), binW = .2)
ax_twin = axs[0,2].twinx()
ax_twin.plot(time, FR, color = 'blue')
ax_twin.set_ylabel("dMSN FR (Hz)", color="blue", rotation = -90, labelpad=15)
ax_twin.tick_params(axis="y", colors="blue")
ax_twin.spines["right"].set_visible(True) 
ax_twin.spines["right"].set_color("blue")
ax_twin.spines["left"].set_color("red")


axs[-1,0].set_xlabel('time since transition point (s)')
axs[-1,1].set_xlabel('time since press (s)')
axs[-1,2].set_xlabel('time since last press (s)')

#axs[1,0].set_ylabel('dMSN (cluster 4)', color = 'blue')
#axs[2,0].set_ylabel('iMSN (cluster 4)', color = 'red')

#ax_inset = inset_axes(axs[2,-1], width="3%", height="100%", loc='center left',
#                bbox_to_anchor=(.9, 0, 1, 1),  # x-shift, y-shift, width-scale, height-scale
#                bbox_transform=axs[2, -1].transAxes,
#                borderpad=2)
#FIs = simpledf.query(f'date == "{date}"').FI.values
#ax_inset.imshow(FIs[:, np.newaxis], aspect='auto', cmap='viridis', extent=[0, 1, 0, len(FIs)], origin = 'lower')
#ax_inset.axis('off')
#
#
#ax_inset = inset_axes(axs[1,-1], width="3%", height="100%", loc='center left',
#                bbox_to_anchor=(.9, 0, 1, 1),  # x-shift, y-shift, width-scale, height-scale
#                bbox_transform=axs[1, -1].transAxes,
#                borderpad=2)
#FIs = simpledf.query(f'date == "{date}"').FI.values
#ax_inset.imshow(FIs[:, np.newaxis], aspect='auto', cmap='viridis', extent=[0, 1, 0, len(FIs)], origin = 'lower')
#ax_inset.axis('off')

axs[-1,-1].set_xlim(-10,10)

[axs[0,ii].axvline(0, color = 'grey', ls = '--',zorder = 1) for ii in range(3)]

figtitle = 'MSN examples | spike triggered DA'
#plt.savefig(rf'{ephys_fig_path}\{figtitle.replace("|","_")}_notitle.png')
#plt.suptitle(figtitle)
#
#plt.savefig(rf'{ephys_fig_path}\{figtitle.replace("|","_")}.png')
#plt.savefig(rf'{ephys_fig_path}\{figtitle.replace("|","_")}.pdf')
#%%

#tercile_colors = ['#D95F02', '#B0B0B0', '#1B9E77']
#tercile_list = ['T1', 'T2', 'T3']
#tercile_rateH_colors = [cm.get_cmap('copper')(1-ii*.35) for ii in range(3)]

"""
..######..########.##.......##........######.....########..#######.....##.....##..######..########
.##....##.##.......##.......##.......##....##.......##....##.....##....##.....##.##....##.##......
.##.......##.......##.......##.......##.............##....##.....##....##.....##.##.......##......
.##.......######...##.......##........######........##....##.....##....##.....##..######..######..
.##.......##.......##.......##.............##.......##....##.....##....##.....##.......##.##......
.##....##.##.......##.......##.......##....##.......##....##.....##....##.....##.##....##.##......
..######..########.########.########..######........##.....#######......#######...######..########
"""
cells_to_use_DAneurons = []
animal = 'Ruthenium'
dates_DAneurons = ['260327']

#dates_to_consider = blocksdf.query(f'date == {ephys_dates_dict[animal]} and animal == "{animal}" and experiment != "c"').date.unique()
#ephys_dates_dict[animal]
for date in dates_DAneurons:
    cells_to_use_DAneurons.append(neuronsdf.query(f'date == "{date}" and cell_type == "MSN" and (SF == "good" or SF == "ok") and fr >=.2').get(['animal', 'date', 'cluster_id']).itertuples(index=False, name=None))

from itertools import chain
cells_to_use_DAneurons = list(chain.from_iterable(cells_to_use_DAneurons))
#%%
def _to_scalar_time(x):
    # None/NaN
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan

    # pandas/py datetime -> seconds
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return pd.to_datetime(x).value / 1e9  # ns -> s

    # 0-dim numpy numbers
    if isinstance(x, (np.floating, np.integer)):
        return float(x)

    # Python numeric
    if isinstance(x, (int, float)):
        return float(x)

    # Single-element containers (list/ndarray/tuple)
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return np.nan
        if len(x) == 1:
            return _to_scalar_time(x[0])
        # If you *expect* singletons, treat multi-length as invalid
        return np.nan

    # Strings that might be numeric or datetime
    if isinstance(x, str):
        # try numeric first
        v = pd.to_numeric(x, errors='coerce')
        if pd.notna(v):
            return float(v)
        # try datetime parse
        try:
            return pd.to_datetime(x).value / 1e9
        except Exception:
            return np.nan

    # Fallback
    return np.nan

def prepare_events(df, event_col):
    out = df.copy()
    # Fast-path: if already datetime64, convert once
    if np.issubdtype(out[event_col].dtype, np.datetime64):
        out[event_col] = pd.to_datetime(out[event_col]).astype('int64') / 1e9
    else:
        out[event_col] = out[event_col].apply(_to_scalar_time)
        out[event_col] = pd.to_numeric(out[event_col], errors='coerce')  # ensure float dtype
    return out

exploded_lvrs_DAneurons = simpledf.explode(['lever_abs_npx_time']).reset_index()
exploded_lvrs_DAneurons = (
    exploded_lvrs_DAneurons
      .pipe(prepare_events, event_col='lever_abs_npx_time')
      .dropna(subset=['lever_abs_npx_time'])
      .reset_index(drop=True)
)

_, _, lvr_psths_smooth_DAneurons, cells_DAneurons, _ = get_psths_across_cells(
    neuronsdf,
    exploded_lvrs_DAneurons,
    cells_to_use_DAneurons,'lever_abs_npx_time',
    pre_time = -.5, post_time = .5, quiet = True)

#%%
neuronsdf['MSN_label'] = 'unclassified'
neuronsdf.loc[neuronsdf.query(f'cluster_id in @STAdf.query("MSN_type == \'dMSN\'").cluster_id').index, 'MSN_label'] = 'dMSN'
neuronsdf.loc[neuronsdf.query(f'cluster_id in @STAdf.query("MSN_type == \'iMSN\'").cluster_id').index, 'MSN_label'] = 'iMSN'
#%%

MSN_labels = []
for cell in cells_DAneurons:
    animal = cell['animal']
    date = cell['date']
    cluster_id = cell['cluster_id']
    MSN_labels.append(neuronsdf.query(f'animal == "{animal}" and date == "{date}" and cluster_id == {cluster_id}').MSN_label.values[0])

MSN_labels = np.hstack(MSN_labels)

mask_iMSN = (MSN_labels == 'iMSN')
mask_dMSN = (MSN_labels == 'dMSN')
#%%

lvr_dMSNs = zscore(np.vstack(lvr_psths_smooth_DAneurons)[mask_dMSN], axis = 1)
lvr_iMSNs = zscore(np.vstack(lvr_psths_smooth_DAneurons)[mask_iMSN], axis = 1)

fig, axs = plt.subplots(1,2)
axs[0].imshow(lvr_dMSNs, extent = [-.5,.5,0,np.sum(mask_dMSN)], aspect = 'auto', cmap = 'magma')
axs[1].imshow(lvr_iMSNs, extent = [-.5,.5,0,np.sum(mask_iMSN)], aspect = 'auto', cmap = 'magma')

axs[0].set_title('dMSN')
axs[1].set_title('iMSN')

[axs[ii].set_xlabel('time since press (s)') for ii in range(2)]
#%%

plt.imshow(drop_nan_rows_in_matrix(zscore(np.vstack(lvr_psths_smooth_DAneurons), axis = 1)))


# %%
_, _, FI15_psths_smooth_DAneurons, cells_DAneurons, _ = get_psths_across_cells(
    neuronsdf,
    simpledf.query('FI == 15'),
    cells_to_use_DAneurons,'npx_trial_start',
    pre_time = 0, post_time = 15, quiet = True)

_, _, FI30_psths_smooth_DAneurons, cells_DAneurons, _ = get_psths_across_cells(
    neuronsdf,
    simpledf.query('FI == 30'),
    cells_to_use_DAneurons,'npx_trial_start',
    pre_time = 0, post_time = 30, quiet = True)

_, _, FI60_psths_smooth_DAneurons, cells_DAneurons, _ = get_psths_across_cells(
    neuronsdf,
    simpledf.query('FI == 60'),
    cells_to_use_DAneurons,'npx_trial_start',
    pre_time = 0, post_time = 60, quiet = True)
#%%
FIconcat_DAneurons = np.concatenate([FI15_psths_smooth_DAneurons, FI30_psths_smooth_DAneurons, FI60_psths_smooth_DAneurons], axis = 1)

plt.imshow(zscore(FIconcat_DAneurons, axis = 1), aspect = 'auto')
#%%
index_order, loadings, PC_space = do_PCA(zscore(FIconcat_DAneurons, axis = 1))
plt.imshow(zscore(FIconcat_DAneurons, axis = 1)[index_order], aspect = 'auto')
#%%
index_order_dMSNs, loadings_dMSNs, PC_space_dMSNs = do_PCA(zscore(FIconcat_DAneurons[mask_dMSN], axis = 1))
index_order_iMSNs, loadings_iMSNs, PC_space_iMSNs = do_PCA(zscore(FIconcat_DAneurons[mask_iMSN], axis = 1))

#%%
fig, axs = plt.subplots(3,2, figsize = (10,6), tight_layout = True, sharex = True)

for ii in range(2):
    axs[0,ii].plot(PC_space[:1500,ii], color = color_FI_blocks[0])
    axs[0,ii].plot(PC_space[1500:4500,ii], color = color_FI_blocks[1])
    axs[0,ii].plot(PC_space[4500:,ii], color = color_FI_blocks[2])

    axs[1,ii].plot(PC_space_dMSNs[:1500,ii], color = color_FI_blocks[0])
    axs[1,ii].plot(PC_space_dMSNs[1500:4500,ii], color = color_FI_blocks[1])
    axs[1,ii].plot(PC_space_dMSNs[4500:,ii], color = color_FI_blocks[2])

    axs[2,ii].plot(PC_space_iMSNs[:1500,ii], color = color_FI_blocks[0])
    axs[2,ii].plot(PC_space_iMSNs[1500:4500,ii], color = color_FI_blocks[1])
    axs[2,ii].plot(PC_space_iMSNs[4500:,ii], color = color_FI_blocks[2])

    axs[0,ii].set_title(f'PC{ii+1}')

axs[0,0].set_ylabel('all MSNs')
axs[1,0].set_ylabel('putative dMSNs')
axs[2,0].set_ylabel('putative iMSNs')

axs[2,0].set_xlabel('time since reward (s)')
axs[2,1].set_xlabel('time since reward (s)')

figtitle = f'{animal} {date} | exp {exp}'
fig.suptitle(figtitle)
#%%
plt.plot(PC_space_dMSNs[:,0], PC_space_dMSNs[:,1], '.', color = 'blue', alpha = 0.5)
plt.plot(PC_space_iMSNs[:,0], PC_space_iMSNs[:,1], '.', color = 'red', alpha = 0.5)
#%%
plt.plot(PC_space_dMSNs[:,0], color = 'blue')
plt.plot(PC_space_iMSNs[:,0], color = 'red')
#%%
plt.plot(PC_space_dMSNs[:,1], color = 'blue', ls = '--')
plt.plot(PC_space_iMSNs[:,1], color = 'red', ls = '--')
#%%

plt.plot(np.nanmean(zscore(FIconcat_DAneurons[mask_dMSN], axis = 1), axis = 0))
plt.plot(np.nanmean(zscore(FIconcat_DAneurons[mask_iMSN], axis = 1), axis = 0))
#%%


simpledf.tercile_cp_FInormalised




#%%

"""
.########.########.....###.....######..##.....##
....##....##.....##...##.##...##....##.##.....##
....##....##.....##..##...##..##.......##.....##
....##....########..##.....##..######..#########
....##....##...##...#########.......##.##.....##
....##....##....##..##.....##.##....##.##.....##
....##....##.....##.##.....##..######..##.....##
"""
## from DAneurons_02 old code

"""
.########.########....###....########.##.....##.########..########.....######..##..........###.....######...######..####.########.####..######.....###....########.####..#######..##....##
.##.......##.........##.##......##....##.....##.##.....##.##..........##....##.##.........##.##...##....##.##....##..##..##........##..##....##...##.##......##.....##..##.....##.###...##
.##.......##........##...##.....##....##.....##.##.....##.##..........##.......##........##...##..##.......##........##..##........##..##........##...##.....##.....##..##.....##.####..##
.######...######...##.....##....##....##.....##.########..######......##.......##.......##.....##..######...######...##..######....##..##.......##.....##....##.....##..##.....##.##.##.##
.##.......##.......#########....##....##.....##.##...##...##..........##.......##.......#########.......##.......##..##..##........##..##.......#########....##.....##..##.....##.##..####
.##.......##.......##.....##....##....##.....##.##....##..##..........##....##.##.......##.....##.##....##.##....##..##..##........##..##....##.##.....##....##.....##..##.....##.##...###
.##.......########.##.....##....##.....#######..##.....##.########.....######..########.##.....##..######...######..####.##.......####..######..##.....##....##....####..#######..##....##
"""
from sklearn.preprocessing import StandardScaler

## features DAspikes

# peak amplitude
# time to peak
# AUC
# slope before and after peak
# mean DA in pre spike and post spike windows

spike_triggered_DA_matrix = allSTAs

t0 = 400
features = []
for trace in spike_triggered_DA_matrix:
    #pre = trace[:t0]
    #post = trace[t0:]

    pre = trace[t0-100:t0]
    post = trace[t0:t0+100]
    full_trace = trace[t0-200:t0+200]

    #feat = {
    #    "pre_mean": np.mean(pre),
    #    "post_mean": np.mean(post),
    #    "delta": np.mean(post) - np.mean(pre),
    #    "peak": np.max(post),
    #    "trough": np.min(post),
    #    "auc_post": np.trapz(post),
    #    "time_to_peak": np.argmax(post),
    #    "time_to_trough": np.argmin(post),
    #}

    feat = {
        "pre_mean": np.mean(pre),
        "post_mean": np.mean(post),
        "delta": np.mean(post) - np.mean(pre),
        "peak": np.max(full_trace),
        "trough": np.min(full_trace),
        "dist_peak_trough": np.max(full_trace) - np.min(full_trace),
        "time_peak_trough": np.argmax(full_trace) - np.argmin(full_trace),
        "auc_post": np.trapz(post),
        "time_to_peak": np.argmax(full_trace),
        "time_to_trough": np.argmin(full_trace),
    }
    features.append(list(feat.values()))


X = np.array(features)
X = StandardScaler().fit_transform(X)

#%%

"""
.##.....##.##.....##....###....########.
.##.....##.###...###...##.##...##.....##
.##.....##.####.####..##...##..##.....##
.##.....##.##.###.##.##.....##.########.
.##.....##.##.....##.#########.##.......
.##.....##.##.....##.##.....##.##.......
..#######..##.....##.##.....##.##.......
"""
import hdbscan
from umap import UMAP
import matplotlib.gridspec as gridspec
#%%
spike_triggered_DA_matrix.shape
#%%
#X = zscore(spike_triggered_DA_matrix, axis = 1)#[:,320:480]
embedding = UMAP(random_state = 15, n_neighbors=50, min_dist=0.0001).fit_transform(X)

# UMAP result: 2D array of shape (n_neurons, 2)
# Let’s say it's stored in `embedding`
clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
labels = clusterer.fit_predict(embedding)
#%%
labels
#%%

t_around_spikes = np.linspace(-4,4,800)

unique_labels = np.unique(labels[labels != -1])
n_labels = len(unique_labels)

cmap = plt.get_cmap('Set2')
label_to_color = {label: cmap(i % 10) for i, label in enumerate(unique_labels)}

fig = plt.figure(figsize=(2*(n_labels+1), 2*n_labels))

gs = gridspec.GridSpec(n_labels, 2, width_ratios=[n_labels, 1])

# Main plot (left, spans all rows)
ax_main = plt.subplot(gs[:, 0])
colors = [label_to_color[l] if l in label_to_color else (0.5, 0.5, 0.5, 0.3) for l in labels]
scatter = ax_main.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=100)
#ax_main.set_title("UMAP of DA-triggered features")
ax_main.set_xlabel("UMAP 1")
ax_main.set_ylabel("UMAP 2")

# Right column: one subplot per label
for i, label in enumerate(unique_labels):
    ax = plt.subplot(gs[i, 1])
    traces = zscore(spike_triggered_DA_matrix, axis = 1)[labels == label]
    for trace in traces:
        ax.plot(t_around_spikes, trace, alpha=0.1, color = label_to_color[label], lw = 1)
    ax.plot(t_around_spikes, np.nanmean(traces, axis=0), lw=1.5, color = 'black')
    ax.set_title(f"Cluster {label}")
    ax.set_xlim(-4, 4)
    #ax.set_xlim(-.5,.5)
    if i == n_labels - 1:
        ax.set_xlabel("time since spike (s)")
    if i == n_labels // 2:
        ax.set_ylabel("DA (a.u.)")

plt.tight_layout()

figtitle = f'{animal} all | MSN clusters | UMAP + HDBSCAN'
fig.suptitle(figtitle)
#plt.savefig(rf'{DAneurons_path_home}\{animal}_all\{figtitle.split('+')[0].replace('|', '_').replace('+','')}')
#%%

plt.imshow(spike_triggered_DA_matrix)
