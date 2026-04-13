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

from pathlib import Path
from probeinterface.plotting import plot_probe


import spikeinterface.extractors as se


from ratcode.config.paths import PATH_STORE_PICKLES, DROPBOX_TASK_PATH, PATH_DATAFRAMES
from ratcode.common.logging import determine_experiment
from ratcode.common.colorcodes import *
from ratcode.behavior import change_point
from ratcode.photometry.photometry import get_prediction, quantile_regression, signal2eventsnippets, find_poly, segment_and_fit_function, butter_filter, mask_jumps, make_continuous, query_and_compute_snippets, plot_snippets
from ratcode.ephys.neurons import get_psths_across_cells, align_spikes_to_ttl, compute_FR, load_ibl_sorter, determine_cell_type, produce_neuron_fig, produce_mega_neuron_fig, compute_zscore, get_psths_smooth, half_gaussian_kernel, do_PCA, get_PCA_windows, plot_raster
from ratcode.common.dataframe import group_and_listify
from ratcode.common.time import convert_date_bonsai, convert_timestamp
from ratcode.common.math import drop_nans_matrix
from ratcode.common.colorcodes import FI_order, color_FI_blocks, rwd_order, color_rwd_blocks
from ratcode.common.plotting import remove_legend

from ratcode.init import setup


setup()
#%%

DANEURONS_PATH_HOME = os.path.join(DROPBOX_TASK_PATH, 'analysis_DAneurons')

aggregated_STAdf = []

animal_list = ['Ruthenium', 'Palladium']

for file in os.listdir(DANEURONS_PATH_HOME):
    splits = file.split('_')
    if len(splits) == 2:
        animal = splits[0]
        date = splits[1]

        if animal in animal_list:
            DANEURONS_PATH = os.path.join(DANEURONS_PATH_HOME, f'{animal}_{date}')
            STAdf = pd.read_pickle(fr'{DANEURONS_PATH}\STAdf.pkl')
            STAdf['animal'] = animal
            STAdf['date'] = date
            aggregated_STAdf.append(STAdf)

if aggregated_STAdf:
    aggregated_STAdf = pd.concat(aggregated_STAdf, ignore_index=True)

    cols_to_move = ['animal', 'date']
    remaining_cols = [c for c in aggregated_STAdf.columns if c not in cols_to_move]
    aggregated_STAdf = aggregated_STAdf[cols_to_move + remaining_cols]

else:
    print("No files matched the criteria.")

# %%

### cells are not updated. need to properly classify MSNs and stuff
STA_array = np.vstack(aggregated_STAdf.av_DA_around_spikes.values)
STA_array_z = zscore(STA_array, axis = 1)
plt.imshow(STA_array_z, aspect = 'auto')

# %%
indices = do_PCA(STA_array_z)[0]
# %%
plt.imshow(STA_array[indices])
# %%
plt.figure()
plt.plot(np.linspace(-4,4,800),STA_array_z[indices][300:].T, alpha = .01, color = 'blue')
plt.plot(np.linspace(-4,4,800),STA_array_z[indices][:300].T, alpha = .01, color = 'red')
plt.axvline(0, color = 'k', ls = '--')
plt.xlim(-4,4)

plt.show()

# %%

# go get info about the cluster
aggregated_STAdf
#%%
animal = 'Palladium'
date = '260304'
#%%
aggregated_STAdf.query(f'animal == "{animal}" and date == "{date}"')
#%%
EPHYS_PATH = os.path.join(DROPBOX_TASK_PATH, 'ephys', animal)
SAVE_SYNC_PATH = glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\*')[0]
#IBL_SORTER_PATH =  glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\{animal}{date}*\ibl_sorter_results_drift_amplitude')[0]
neuronsdf = pd.read_pickle(fr'{SAVE_SYNC_PATH}\neuronsdf.pkl')
syncdf = pd.read_pickle(fr'{SAVE_SYNC_PATH}\syncdf.pkl')

#%%
DATACLASS_PATH = rf"{DROPBOX_TASK_PATH}\analysis_ephys\{animal}_{date}_sorted_data.pkl"
sorted_data = pd.read_pickle(DATACLASS_PATH)
#%%
aggregated_jointdf = pd.read_pickle(rf'{PATH_DATAFRAMES}\aggregate_photometry_Palladium_Ruthenium.pkl')


#%%
sorted_data
#%%
neuronsdf.query('KSLabel == "good" and SF == "good"')
#%%
neuronsdf.query('cluster_id == 165 or cluster_id == 173')
#%%
sns.histplot(neuronsdf.query('cluster_id == 165').spikes_self_aligned.values[0])
#%%

DAdf = aggregated_jointdf.query(f'animal == "{animal}" and date == "{date}"')

eventalignment = 'pump_on_abs'
snipps, time = signal2eventsnippets(np.hstack(DAdf.time_DA.values),
                                    np.hstack(DAdf.DA_zscored_session.values),
                                    np.hstack(DAdf[eventalignment].values),
                                            [-4,4], .01)

#%%
cluster_id = 165

spikes = neuronsdf.query(f'cluster_id == {cluster_id}').spike_times.values[0]
#ttls = np.hstack(syncdf.lever_npx.values)
ttls = np.hstack(syncdf.rwd_onset_npx.values)

spikes_aligned = align_spikes_to_ttl(spikes, ttls, (-4,4))

fig, axs = plt.subplots(3,1, tight_layout = True)

plot_raster(axs[0], spikes_aligned, (-4,4))

#axs[1].imshow(snipps, aspect = 'auto', origin = 'lower')

axs[1].plot(time, np.nanmean(snipps, axis = 0))

axs[2].plot(np.linspace(-4,4,800),np.hstack(
    aggregated_STAdf.query(f'animal == "{animal}" and date == "{date}" and cluster_id == {cluster_id}')
    .av_DA_around_spikes.values))

fig.suptitle(f'{animal} | {date} | {cluster_id}')

# %%

aggregated_STAdf.query(f'animal == "{animal}" and date == "{date}" and cluster_id == {cluster_id}').av_DA_around_spikes.values
#psths = get_psths_smooth([165,173],ttls,-8,8,sorted_data)

# %%
plt.plot(psths[1][0], psths[1][1], '.')
# %%

#alignments, alignments_dict, key_dict, cmap_FI, cmap_nprots = compute_alignments(syncdf,bool_click, bool_cp_corrected)

#fig, axs = plt.subplots(3,len(alignments), figsize=(4*len(alignments),12), tight_layout = True, height_ratios=[1,2,1])
#for jj in range(len(alignments)):
    #axs[1, jj].sharex(axs[0, jj])

cluster_id = 165
sampling_frequency = 30000
spike_times_cluster = sorted_data.spike_times[sorted_data.spike_clusters == cluster_id]/sampling_frequency

#for jj in range(len(alignments)):
#alignment_label = alignments[jj]
#alignment_times = alignments_dict[alignment_label]
#key = key_dict[alignment_label]
df = syncdf.explode('lever_npx')

#spikes_aligned = align_spikes_to_ttl(spike_times_cluster,alignment_times, window=window)
plot_raster(axs[1,jj], spikes_aligned, (-8,8))


#%%
### this was in neurons_visualizer.py -- actually I'll work there
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