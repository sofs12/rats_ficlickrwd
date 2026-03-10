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
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler


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
....###....##....##.####.##.....##....###....##..........########.....###....########.########
...##.##...###...##..##..###...###...##.##...##..........##.....##...##.##......##....##......
..##...##..####..##..##..####.####..##...##..##..........##.....##..##...##.....##....##......
.##.....##.##.##.##..##..##.###.##.##.....##.##..........##.....##.##.....##....##....######..
.#########.##..####..##..##.....##.#########.##..........##.....##.#########....##....##......
.##.....##.##...###..##..##.....##.##.....##.##..........##.....##.##.....##....##....##......
.##.....##.##....##.####.##.....##.##.....##.########....########..##.....##....##....########
"""

animal = 'Ruthenium'
date = '260305'

## now looking at my old code, 01_DAneurons.py
DANEURONS_PATH_HOME = os.path.join(DROPBOX_TASK_PATH, 'analysis_DAneurons')
DANEURONS_PATH = os.path.join(DANEURONS_PATH_HOME, f'{animal}_{date}')
if not os.path.exists(DANEURONS_PATH):
    os.makedirs(DANEURONS_PATH)


#%%
#neurons df

EPHYS_PATH = os.path.join(DROPBOX_TASK_PATH, 'ephys', animal)

#PATH_SAVE_FIGS = os.path.join(DROPBOX_TASK_PATH, 'analysis_ephys', f'{animal}_{date}')
#if not os.path.exists(PATH_SAVE_FIGS):
#    os.makedirs(PATH_SAVE_FIGS)

SAVE_SYNC_PATH = glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\*')[0]


IBL_SORTER_PATH =  glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\{animal}{date}*\ibl_sorter_results_drift_amplitude')[0]

#NEURO_PATH =  glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\{animal}{date}*')[0]
#NEURO_PATH = glob.glob(rf"H:\{animal}{date}*\{animal}{date}*")[0]#[1]
#NEURO_PATH = glob.glob(rf"F:\EPHYS\{animal}{date}*\{animal}{date}*")[0]#[1]

#%%
neuronsdf = pd.read_pickle(fr'{SAVE_SYNC_PATH}\neuronsdf.pkl')
syncdf = pd.read_pickle(fr'{SAVE_SYNC_PATH}\syncdf.pkl')

exp = determine_experiment(syncdf)

print(exp)

if exp == 'c':
    hue_variable = 'n_protocols'
    color_palette = color_rwd_blocks
    hue_variable_list = rwd_order
else:
    hue_variable = 'FI'
    color_palette = color_FI_blocks
    hue_variable_list = FI_order


spikes_self_aligned_all = neuronsdf.spikes_self_aligned.values

#make sure the neuronsdf have the current version of SF labels
cluster_info = pd.read_csv(rf'{IBL_SORTER_PATH}\cluster_info.tsv', sep = '\t')
neuronsdf['SF'] = cluster_info.SF
#%%

sorted_data = load_ibl_sorter(IBL_SORTER_PATH, animal, date, exp)

#%%


### PHOTOMETRY DATA UP UNTIL MARCH 5, run first the new jointdf (daily_photometry.py)





## photometry
jointdf_pkl = glob.glob(rf'{DROPBOX_TASK_PATH}\analysis_photometry\{animal}_{date}*_NEWjointdf.pkl')[0]
jointdf = pd.read_pickle(jointdf_pkl)

simpledf = jointdf.get(['trialno', 'blockno', 'FI', 'n_protocols', 'lever_index'])

simpledf['trialno_within_block'] = simpledf.groupby('blockno').cumcount()+1
simpledf['trialno_within_block_from_end'] = -1 * (simpledf.groupby('blockno').cumcount(ascending=False) + 1)

simpledf['animal'] = animal
simpledf['date'] = date
simpledf['time_DA'] = jointdf['timestamp_session']
#### attention here -- ICA or regular DA
simpledf['DA'] = jointdf['DA_poly_session'] ##### careful with the ICA definitions; before this was 'DA_session', in the ICA jointdf
#simpledf['DA_ICA'] = jointdf['DA_session_ICA'] ########
simpledf['trial_start'] = jointdf['trial_start_harp']
simpledf['trial_end'] = jointdf['trial_end_harp']
simpledf['lever_rel'] = jointdf['lever_rel_harp']
simpledf['lever_abs'] = jointdf['lever_abs_harp']
simpledf['poke_rel'] = jointdf['poke_rel_harp']
simpledf['poke_abs'] = jointdf['poke_abs_harp']
simpledf['rwd_onset'] = jointdf['pump_on_harp']
simpledf['rwd_offset'] = jointdf['pump_off_harp']
#simpledf['FI'] = (simpledf['FI']/1000).astype(int)

simpledf['cp'] = jointdf['cp_harp']
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
plt.plot(jointdf.trial_duration_harp.values, label = 'photometry')
plt.plot(syncdf.loc[0:].query('trial_duration_s > 2').trial_duration_s.values, '--', label = 'ephys')
plt.title('see if trials match')
plt.legend()
plt.show()
# %%

simpledf['npx_trial_start'] =syncdf.query('trial_duration_s > 2').npx_time.values

# %%
plt.plot(simpledf.trial_start)
plt.plot(simpledf.npx_trial_start)

#%%
plt.plot(simpledf.npx_trial_start - simpledf.trial_start)
# %%
simpledf['tercile_cp_withinblock'] = (
    simpledf.query('blockno < 4').groupby('blockno')['cp']
    .transform(lambda x: pd.qcut(x, q=3, labels=['T1', 'T2', 'T3']))
)

# %%

"""
..######.....###....##.....##.########.....######..####.##.....##.########..##.......########.########..########
.##....##...##.##...##.....##.##..........##....##..##..###...###.##.....##.##.......##.......##.....##.##......
.##........##...##..##.....##.##..........##........##..####.####.##.....##.##.......##.......##.....##.##......
..######..##.....##.##.....##.######.......######...##..##.###.##.########..##.......######...##.....##.######..
.......##.#########..##...##..##................##..##..##.....##.##........##.......##.......##.....##.##......
.##....##.##.....##...##.##...##..........##....##..##..##.....##.##........##.......##.......##.....##.##......
..######..##.....##....###....########.....######..####.##.....##.##........########.########.########..##......
"""

simpledf.to_pickle(rf'{DANEURONS_PATH_HOME}\{animal}_{date}_simpledf.pkl')

# %%

"""
.########..##.....##.##.....##....########.....###...
.##.....##.##.....##.##.....##....##.....##...##.##..
.##.....##.##.....##.##.....##....##.....##..##...##.
.########..#########.##.....##....##.....##.##.....##
.##.....##.##.....##..##...##.....##.....##.#########
.##.....##.##.....##...##.##......##.....##.##.....##
.########..##.....##....###.......########..##.....##
"""

fig, axs = plt.subplots(1,2, figsize = (8, 4), tight_layout = True)
sns.histplot(ax = axs[0], data = simpledf.explode('lever_rel'), x = 'lever_rel', hue = hue_variable,
             stat = 'density', common_norm = False, element = 'step', binwidth = 4, palette = color_palette)
sns.histplot(ax = axs[1], data = simpledf.explode('lever_rel_FInormalised'), x = 'lever_rel_FInormalised', hue = hue_variable,
             stat = 'density', common_norm = False, element = 'step', binwidth = .05, palette = color_palette)
axs[0].set_xlim(0,simpledf.FI.max()*1.1)
axs[1].set_xlim(0,1.1)
axs[0].set_xlabel('time since rwd (s)')
axs[1].set_xlabel('time since rwd (FI normalised)')
figtitle = f'{animal} {date} | experiment {exp} | lever presses'
plt.suptitle(figtitle)
plt.savefig(fr'{DANEURONS_PATH}\{figtitle.replace('|','_')}')
# %%

fig, axs = plt.subplots(1,2, figsize = (8, 4), tight_layout = True)
sns.histplot(ax = axs[0], data = simpledf, x = 'cp', hue = hue_variable,
             stat = 'density', common_norm = False, element = 'step', binwidth = 4, palette = color_palette)
sns.histplot(ax = axs[1], data = simpledf, x = 'cp_FInormalised', hue = hue_variable,
             stat = 'density', common_norm = False, element = 'step', binwidth = .05, palette = color_palette)
axs[0].set_xlim(0,simpledf.FI.max())
axs[1].set_xlim(0,1)
axs[0].set_xlabel('time since rwd (s)')
axs[1].set_xlabel('time since rwd (FI normalised)')
figtitle = f'{animal} {date} | experiment {exp} | transition point'
plt.suptitle(figtitle)
plt.savefig(fr'{DANEURONS_PATH}\{figtitle.replace('|','_')}')

# %%

plt.figure()

for ii in range(3):
    for trialno in simpledf.query(f'{hue_variable} == {hue_variable_list[ii]}').trialno.values:

        sns.lineplot(data = simpledf.query(f'trialno == {trialno}').explode(['time_DA_after_cp', 'DA']),
        x = 'time_DA_after_cp', y = 'DA', color = color_palette[ii], alpha = 0.4)

plt.xlim(-2,2)

# %%
simpledf['diff_DA_around_cp'] = simpledf.query('bool_cp').apply(lambda x: np.mean(x.DA_after_cp[:100]) - np.mean(x.DA_before_cp[-200:-100]), axis = 1)

# %%
sns.scatterplot(data = simpledf,
            x = 'diff_DA_around_cp', y = 'cp_FInormalised', hue = hue_variable, palette=color_palette)

# %%


snippets, time = query_and_compute_snippets(simpledf,'bool_cp', 'cp_abs', 'time_DA', 'DA', (-3,3))
snippets_zscored = np.apply_along_axis(compute_zscore, arr = snippets, axis = 1)
fig, axs = plt.subplots(2)
plot_snippets(snippets_zscored, time, axs[0], axs[1])

# %%

df = simpledf.query('bool_cp').get(['cp'])
diff_snippets = np.max(snippets[:,250:400], axis = 1) - snippets[:,250]
#diff_snippets[weird_snippet_indx_DA_trials_around_cp] = np.nan
df['diff_DA_peak_cp'] = diff_snippets
simpledf['diff_DA_peak_cp'] = df.diff_DA_peak_cp
# %%
fig, axs = plt.subplots(3,2, tight_layout = True, figsize = (10,10))

## DA around transition point

sns.scatterplot(ax = axs[0,0], data = simpledf.query('bool_cp'), y = 'diff_DA_peak_cp', x = 'cp',
                hue = hue_variable, palette = color_palette)

sns.histplot(ax = axs[1,0], data = simpledf, x = 'diff_DA_peak_cp', hue = hue_variable, palette=color_palette, stat = 'density')

sns.scatterplot(ax = axs[2,0], data = simpledf, y = 'diff_DA_peak_cp', x = 'trialno', hue = hue_variable, palette=color_palette)



## DA around last press

sns.scatterplot(ax = axs[0,1], data = simpledf, y = 'diff_min2peak', x = 'cp_FInormalised', hue = hue_variable, palette=color_palette)





[remove_legend(axs[ii,jj]) for ii in range(3) for jj in range(2)]


plt.suptitle(f'{animal} {date} | experiment {exp} | DA peak at transition point')
#%%

snippets, time = query_and_compute_snippets(simpledf, 'FI > 0', 'rwd_onset_abs', 'time_DA', 'DA', (-1,1), delta_time = 1/100, nanify = False)

# %%
fig, axs = plt.subplots(2, figsize = (6, 4), sharex = True, tight_layout = True)
snippets_zscored_align_last = np.apply_along_axis(compute_zscore, arr = snippets, axis = 1)
plot_snippets(snippets, time, axs[0], axs[1])#, 'bool_cp', color_FI_blocks)
# %%
plt.axvline(0, color = 'black', lw = .5)

for ii in range(20,35):
    plt.plot(time, snippets[ii], color = 'grey', alpha = 0.5)

# %%
plt.axvline(0, color = 'black', lw = .5)

ii = 45
plt.plot(time, snippets_zscored_align_last[ii], color = 'grey', alpha = 0.5)

plt.plot(time[50:101], snippets_zscored_align_last[ii, 50:101])
plt.plot(time[100:150], snippets_zscored_align_last[ii, 100:150])

diff_neurons = np.ones(len(snippets_zscored_align_last))*np.nan
diff_min2peak = np.ones(len(snippets_zscored_align_last))*np.nan

for ii in range(21, len(snippets)):
    diff_neurons[ii] = np.sum(snippets_zscored_align_last[ii,100:150]) - np.sum(snippets_zscored_align_last[ii,50:101])
    diff_min2peak[ii] = np.max(snippets_zscored_align_last[ii,100:150]) - np.mean(snippets_zscored_align_last[ii,50:101])
    plt.plot(time[100:150], snippets_zscored_align_last[ii, 100:150], color = 'grey', alpha = 0.5)
#np.sum(snippets[ii,100:150]) - np.sum(snippets[ii,50:101])
# %%
len(snippets)
# %%
simpledf['diff_DA_rwd'] = diff_neurons
simpledf['diff_min2peak'] = diff_min2peak
# %%


### I don't know what's going on up there...

#%%

"""
.########.########.########...######..####.##.......########..######.
....##....##.......##.....##.##....##..##..##.......##.......##....##
....##....##.......##.....##.##........##..##.......##.......##......
....##....######...########..##........##..##.......######....######.
....##....##.......##...##...##........##..##.......##.............##
....##....##.......##....##..##....##..##..##.......##.......##....##
....##....########.##.....##..######..####.########.########..######.
"""
fig, axs = plt.subplots(2,3, figsize = (12,8),tight_layout = True, sharey = 'row', sharex = True)

snippets_lastpress_T1, time = query_and_compute_snippets(simpledf,'tercile_cp_FInormalised == "T1"', 'rwd_onset_abs', 'time_DA', 'DA', (-3,3))
plot_snippets(snippets_lastpress_T1,time, axs[0,0], axs[1,0], color_DA_traces='red', cmap = 'bone') 

snippets_lastpress_T2, time = query_and_compute_snippets(simpledf,'tercile_cp_FInormalised == "T2"', 'rwd_onset_abs', 'time_DA', 'DA', (-3,3))
plot_snippets(snippets_lastpress_T2,time, axs[0,1], axs[1,1], color_DA_traces='grey', cmap = 'bone')

snippets_lastpress_T3, time = query_and_compute_snippets(simpledf,'tercile_cp_FInormalised == "T3"', 'rwd_onset_abs', 'time_DA', 'DA', (-3,3))
plot_snippets(snippets_lastpress_T3,time, axs[0,2], axs[1,2], color_DA_traces='blue', cmap = 'bone')


for ii in range(3):
    axs[1,ii].axvline(0, color = 'black', lw = .5)

axs[0,0].set_title('early cp', color = 'red')
axs[0,1].set_title('median cp', color = 'grey')
axs[0,2].set_title('late cp', color = 'blue')

axs[0,0].set_ylabel('trials')
axs[1,0].set_ylabel('DA (deltaF/F)')

[axs[1,ii].set_xlabel('t since last press (s)') for ii in range(3)]

figtitle = f'{animal} {date} | experiment {exp} | DA around last press split by cp terciles'
plt.suptitle(figtitle)
plt.savefig(fr'{DANEURONS_PATH}\{figtitle.replace('|','_')}.png')

# %%

fig, axs = plt.subplots(2,3, figsize = (12,8), tight_layout = True, sharey = 'row', sharex = True)

snippets_cp_T1, time = query_and_compute_snippets(simpledf,'tercile_cp_FInormalised == "T1"', 'cp_abs', 'time_DA', 'DA', (-3,3))
plot_snippets(snippets_cp_T1,time, axs[0,0], axs[1,0], color_DA_traces='red', cmap = 'bone')

snippets_cp_T2, time = query_and_compute_snippets(simpledf,'tercile_cp_FInormalised == "T2"', 'cp_abs', 'time_DA', 'DA', (-3,3))
plot_snippets(snippets_cp_T2,time, axs[0,1], axs[1,1], color_DA_traces='grey', cmap = 'bone')

snippets_cp_T3, time = query_and_compute_snippets(simpledf,'tercile_cp_FInormalised == "T3"', 'cp_abs', 'time_DA', 'DA', (-3,3))
plot_snippets(snippets_cp_T3,time, axs[0,2], axs[1,2], color_DA_traces='blue', cmap = 'bone')

for ii in range(3):
    axs[1,ii].axvline(0, color = 'black', lw = .5)

axs[0,0].set_title('early cp', color = 'red')
axs[0,1].set_title('median cp', color = 'grey')
axs[0,2].set_title('late cp', color = 'blue')

axs[0,0].set_ylabel('trials')
axs[1,0].set_ylabel('DA (deltaF/F)')

[axs[1,ii].set_xlabel('t since transition point (s)') for ii in range(3)]

figtitle = f'{animal} {date} | experiment {exp} | DA around transition point split by cp terciles'
plt.suptitle(figtitle)
plt.savefig(fr'{DANEURONS_PATH}\{figtitle.replace('|','_')}.png')


# %%

fig, axs = plt.subplots(1,3, tight_layout = True, figsize = (12,4))

sns.histplot(ax = axs[0], data = simpledf, x = 'cp_FInormalised', hue = 'tercile_cp_FInormalised', palette=['red', 'grey', 'blue'],
             element = 'step')
remove_legend(axs[0])
axs[0].set_xlim(0,1)

axs[1].plot(time, np.nanmean(snippets_cp_T1, axis = 0), color = 'red')
axs[1].plot(time, np.nanmean(snippets_cp_T2, axis = 0), color = 'grey')
axs[1].plot(time, np.nanmean(snippets_cp_T3, axis = 0), color = 'blue')

axs[2].plot(time, np.nanmean(snippets_lastpress_T1, axis = 0), color = 'red')
axs[2].plot(time, np.nanmean(snippets_lastpress_T2, axis = 0), color = 'grey')
axs[2].plot(time, np.nanmean(snippets_lastpress_T3, axis = 0), color = 'blue')

for ii in range(1,3):
    axs[ii].axvline(0, color = 'black', lw = .5)

axs[0].set_xlabel('time since rwd (normalised to FI)')
axs[1].set_xlabel('time since transition point (s)')
axs[2].set_xlabel('time since last lever press (s)')

axs[0].set_ylabel('transition point distribution\n(counts)')
[axs[ii].set_ylabel('DA (deltaF/F)') for ii in [1,2]]

figtitle = f'{animal} {date} | experiment {exp} | DA split by transition points terciles'
plt.suptitle(figtitle)
plt.savefig(fr'{DANEURONS_PATH}\{figtitle.replace('|','_')}.png')

# %%


blocks = simpledf.query('blockno < 4').blockno.unique()
fig, axs = plt.subplots(len(blocks),3, figsize = (12, len(blocks)*4), tight_layout = True)

for bb in blocks:
    for ii in range(3):
        snippets, time = query_and_compute_snippets(simpledf,f'blockno == {bb} and tercile_cp_withinblock == "{terciles_labels[ii]}"', 'cp_abs', 'time_DA', 'DA', (-3,3))
        axs[bb-1,1].plot(time, np.nanmean(snippets, axis = 0), color = terciles_colors[ii])

        snippets, time = query_and_compute_snippets(simpledf,f'blockno == {bb} and tercile_cp_withinblock == "{terciles_labels[ii]}"', 'rwd_onset_abs', 'time_DA', 'DA', (-3,3))
        axs[bb-1,2].plot(time, np.nanmean(snippets, axis = 0), color = terciles_colors[ii])

    sns.histplot(ax = axs[bb-1,0], data = simpledf.query(f'blockno == {bb}'), x = 'cp',
                 hue = 'tercile_cp_withinblock', hue_order=terciles_labels, palette=terciles_colors, element = 'step')
    
    block_terciles_sequence = simpledf.query(f'blockno == {bb} and bool_cp').tercile_cp_withinblock.values

    colors = [terciles_color_dic[label] for label in block_terciles_sequence]

    cps = simpledf.query(f'blockno == {bb} and bool_cp').cp.values
    x = np.linspace(np.min(cps),np.max(cps),len(block_terciles_sequence))
    y = [1] * len(block_terciles_sequence)

    axs[bb-1,0].scatter(x, y, c=colors, s=100)
    
    axs[bb-1,0].set_ylabel(f'block {bb} | FI{simpledf.query(f'blockno == {bb}').FI.values[0]}')
    remove_legend(axs[bb-1,0])

axs[-1,1].set_xlabel('t since transition point (s)')
axs[-1,2].set_xlabel('t since last lever press (s)')


figtitle = f'{animal} {date} | experiment {exp} | DA split by transition points terciles within blocks'
plt.suptitle(figtitle)
plt.savefig(fr'{DANEURONS_PATH}\{figtitle.replace('|','_')}.png')
# %%

plt.figure(figsize = (2,4))
for ii in range(len(snippets)):
    plt.plot(time, ii*.01+snippets[ii]*2, color = 'grey', alpha = 0.2)

plt.axvline(0, color = 'black', lw = .5)
plt.axvline(.5, color = 'black', lw = .5)
plt.xlim(-1,1)
plt.title('last press')
# %%

"""
.########..##.....##.##.....##....##....##.########.##.....##.########...#######.
.##.....##.##.....##.##.....##....###...##.##.......##.....##.##.....##.##.....##
.##.....##.##.....##.##.....##....####..##.##.......##.....##.##.....##.##.....##
.########..#########.##.....##....##.##.##.######...##.....##.########..##.....##
.##.....##.##.....##..##...##.....##..####.##.......##.....##.##...##...##.....##
.##.....##.##.....##...##.##......##...###.##.......##.....##.##....##..##.....##
.########..##.....##....###.......##....##.########..#######..##.....##..#######.
"""
#considering the trials whose duration does not exceed FI in 10%
goodtrialsdf = simpledf.query('bool_cp')
print(len(goodtrialsdf))

#%%
#concatenate the trials of multiple FIs and define the PC space like that
exp = determine_experiment(syncdf)
print(f'exp {exp}')
print()
print(len(goodtrialsdf.query('FI == 15')))
print(len(goodtrialsdf.query('FI == 30')))
print(len(goodtrialsdf.query('FI == 60')))
print()
print(len(goodtrialsdf.query('n_protocols == 7')))
print(len(goodtrialsdf.query('n_protocols == 14')))
print(len(goodtrialsdf.query('n_protocols == 28')))
# %%
clusters_to_consider = neuronsdf.query('(SF == "good" or SF == "ok") and n_spikes > 1000').cluster_id
print(len(clusters_to_consider))

# %%
psthbin = 0.02
kernel = half_gaussian_kernel()

alignment = 'npx_trial_start'

if exp == 'c':
    FI = 30

    nprots = 7
    ttls = goodtrialsdf.query(f'n_protocols == {nprots}')[alignment].dropna().values
    psths_smoothed_7 = get_psths_smooth(clusters_to_consider, ttls, 0, FI, sorted_data, psthbin = psthbin, kernel = kernel)[-1]

    nprots = 14
    ttls = goodtrialsdf.query(f'n_protocols == {nprots}')[alignment].dropna().values
    psths_smoothed_14 = get_psths_smooth(clusters_to_consider, ttls, 0, FI, sorted_data, psthbin, kernel)[-1]

    nprots = 28
    ttls = goodtrialsdf.query(f'n_protocols == {nprots}')[alignment].dropna().values
    psths_smoothed_28 = get_psths_smooth(clusters_to_consider, ttls, 0, FI, sorted_data, psthbin, kernel)[-1]

    concat_for_PCA = np.concatenate([psths_smoothed_7, psths_smoothed_14, psths_smoothed_28], axis = 1)

else:
    FI = 15
    ttls = goodtrialsdf.query(f'FI == {FI}')[alignment].dropna().values
    psths_smoothed_15 = get_psths_smooth(clusters_to_consider, ttls, 0, FI, sorted_data, psthbin = psthbin, kernel = kernel)[-1]

    FI = 30
    ttls = goodtrialsdf.query(f'FI == {FI}')[alignment].dropna().values
    psths_smoothed_30 = get_psths_smooth(clusters_to_consider, ttls, 0, FI, sorted_data, psthbin, kernel)[-1]

    FI = 60
    ttls = goodtrialsdf.query(f'FI == {FI}')[alignment].dropna().values
    psths_smoothed_60 = get_psths_smooth(clusters_to_consider, ttls, 0, FI, sorted_data, psthbin, kernel)[-1]

    concat_for_PCA = np.concatenate([psths_smoothed_15, psths_smoothed_30, psths_smoothed_60], axis = 1)

#%%
concat_for_PCA = concat_for_PCA[~np.isnan(concat_for_PCA).any(axis=1)]
plt.matshow(concat_for_PCA, aspect = 'auto')

# %%

scaler = StandardScaler()
concat_for_PCA = scaler.fit_transform(concat_for_PCA)

original_index_order, loadings, PC_space = do_PCA(concat_for_PCA)
# %%
split_idx = 35
index_order = np.concatenate([original_index_order[split_idx:],original_index_order[:split_idx]])
#%%
plt.matshow(concat_for_PCA[index_order], aspect = 'auto', cmap = 'magma')
plt.title('find split index')
# %%
windows = get_PCA_windows(exp, psthbin)

# %%
fig, axs = plt.subplots(1,3, figsize = (8,4), tight_layout = True, sharey = True)

for ii in range(3):
    heatmap = concat_for_PCA[:,slice(*windows[ii])][index_order]
    vmin = np.nanquantile(heatmap,.01)
    vmax = np.nanquantile(heatmap,.99)
    axs[ii].imshow(heatmap,  vmin = vmin, vmax = vmax, aspect = 'auto', interpolation = None, cmap = 'magma')

    xticks = axs[ii].get_xticks()
    new_labels = [int(x * 2 / 100) for x in xticks]
    axs[ii].set_xticks(xticks)
    axs[ii].set_xticklabels(new_labels)

    axs[ii].set_xlim(0)


axs[0].set_ylabel('neurons (sorted)')
[axs[ii].set_xlabel('t since rwd (s)') for ii in range(3)]

[axs[ii].set_title(f'{hue_variable}{hue_variable_list[ii]}') for ii in range(3)]

figtitle = f'{animal} {date} | exp {exp} | neurons sorted by angular pos in PC space'
plt.suptitle(figtitle)

plt.savefig(fr'{DANEURONS_PATH}\{figtitle.replace('|','_')}.png')
# %%

#### individual trials

trials_conditions = []

for ii in range(3):
    var = hue_variable_list[ii]
    trials_conditions.append(simpledf.query(f'bool_cp and {var} == {hue_variable_list[ii]}').trialno.values.tolist())

# %%
psths_trials_conditions = []

for jj in range(3):
    var = hue_variable_list[jj]
    if exp == 'c':
        FI = 30
    else:
        FI = var
    n_neurons = len(clusters_to_consider)
    n_trials = len(trials_conditions[jj])
    n_timebins = windows[jj][-1] - windows[jj][0]
    psths_trials = np.zeros(([n_trials,n_neurons,n_timebins]))
    for ii in range(n_trials):
        trialno = trials_conditions[jj][ii]
        ttls = simpledf.query(f'trialno == {trialno}').npx_trial_start.dropna().values
        psths_trial = get_psths_smooth(clusters_to_consider, ttls, 0, FI, sorted_data, psthbin = psthbin, kernel = kernel)[-1]
        psths_trials[ii] = psths_trial

    psths_trials_conditions.append(psths_trials)

#%%
for tt in range(n_trials)[5:10]:
    plt.matshow(psths_trials_conditions[-1][tt][index_order], aspect = 'auto')
    plt.title(ii)

#%%

for nn in range(n_neurons)[3:10]:
    fig, axs = plt.subplots(2,3, sharex = 'col')
    for ii_cond in range(3):
        axs[0,ii_cond].plot(np.nanmean(psths_trials_conditions[ii_cond][:,nn,:], axis = 0))
        axs[1,ii_cond].matshow(psths_trials_conditions[ii_cond][:,nn,:], aspect = 'auto')

    fig.suptitle(nn)

# %%



cluster_id = index_order[5]

fig, axs = plt.subplots(2,3, tight_layout = True, sharey='row', sharex = 'col')

for ii in range(3):
    FI = FI_order[ii]
    spikes_aligned = align_spikes_to_ttl(sorted_data.spike_times[sorted_data.spike_clusters == cluster_id] / 30000, simpledf.query(f'FI == {FI} and bool_cp').npx_trial_start.values,(0,FI))
    for i, spikes in enumerate(spikes_aligned):
        axs[1,ii].plot(spikes, np.ones_like(spikes)*(i+1), '|', color='black')  # Thin vertical lines

    #plot_raster(axs[1,ii], spikes_aligned, (0,FI))

    time, firing_rate = compute_FR(spikes_aligned,(0,FI))
    axs[0,ii].plot(time,firing_rate, color = 'black', lw = 1)

axs[1,0].set_ylim(0,17.5)
[axs[1,ii].set_xlabel('t since rwd (s)') for ii in range(3)]

axs[0,0].set_ylabel('firing rate (Hz)')
axs[1,0].set_ylabel('e.g. trials')


[axs[0,ii].set_title(f'FI{FI_order[ii]}') for ii in range(3)]

figtitle = f'{animal} {date} | experiment {exp} | cluster id {cluster_id}'
plt.suptitle(figtitle)

plt.savefig(fr'{DANEURONS_PATH}\{figtitle.replace('|','_')}.png')
# %%

fig, axs = plt.subplots(3,3, figsize = (12,12), tight_layout = True, sharex = 'col')

for tercile_ii in range(3):
    for cond_ii in range(3):
        tercile_FI_trials = simpledf.query(f'tercile_cp_FInormalised == "{terciles_labels[tercile_ii]}" and {hue_variable} == {hue_variable_list[cond_ii]}').trialno.values.tolist()
        tercile_FI_idx = [trials_conditions[cond_ii].index(x) for x in tercile_FI_trials if x in trials_conditions[cond_ii]]

        heatmap = np.nanmean(psths_trials_conditions[cond_ii][tercile_FI_idx], axis = 0)[index_order]
        vmin = np.nanquantile(heatmap,.01)
        vmax = np.nanquantile(heatmap,.99)
        axs[tercile_ii,cond_ii].imshow(heatmap, vmin = vmin, vmax = vmax, aspect = 'auto', cmap = 'magma', interpolation = None)

for cond_ii in range(3):
    axs[0,cond_ii].set_title(f'{hue_variable}{hue_variable_list[cond_ii]}')

axs[0,0].set_ylabel('early transition', color = 'red')
axs[1,0].set_ylabel('mid transition', color = 'grey')
axs[2,0].set_ylabel('late transition', color = 'blue')

for ii in range(3):
    for jj in range(3):
        xticks = axs[jj,ii].get_xticks()
        new_labels = [int(x * 2 / 100) for x in xticks]
        axs[jj,ii].set_xticks(xticks)
        axs[jj,ii].set_xticklabels(new_labels)
    
    if exp == 'c':
        [axs[ii,jj].set_xlim(0,1500) for jj in range(3)]

    else:
        axs[ii,0].set_xlim(0,750)
        axs[ii,1].set_xlim(0,1500)
        axs[ii,2].set_xlim(0,3000)


[axs[2,ii].set_xlabel('t since rwd (s)') for ii in range(3)]

figtitle = f'{animal} {date} | experiment {exp} | neural activity conditioned on transition point terciles'
plt.suptitle(figtitle)
plt.savefig(fr'{DANEURONS_PATH}\{figtitle.replace('|','_')}.png')
# %%
fig, axs = plt.subplots(1,3, figsize = (12,4), tight_layout = True)

for cond_ii in range(3):
    axs[cond_ii].imshow(np.nanmean(psths_trials_conditions[cond_ii], axis = 0)[index_order], aspect = 'auto', interpolation = None, cmap = 'magma')

plt.suptitle('sanity check\ntiling is the same with psths single trial')

# %%

## lead lag analysis based on the early vs late transition
# (just computing the difference between the neural activities)


fig, axs = plt.subplots(1,3,figsize = (12,4), tight_layout = True, sharey = True)

for cond_ii in range(3): ## this is the FI for now

    # early
    tercile_ii = 0
    tercile_FI_trials = simpledf.query(f'tercile_cp_FInormalised == "{terciles_labels[tercile_ii]}" and {hue_variable} == {hue_variable_list[cond_ii]}').trialno.values.tolist()
    tercile_FI_idx = [trials_conditions[cond_ii].index(x) for x in tercile_FI_trials if x in trials_conditions[cond_ii]]
    heatmap_early = np.nanmean(psths_trials_conditions[cond_ii][tercile_FI_idx], axis = 0)[index_order]

    # late
    tercile_ii = 2
    tercile_FI_trials = simpledf.query(f'tercile_cp_FInormalised == "{terciles_labels[tercile_ii]}" and {hue_variable} == {hue_variable_list[cond_ii]}').trialno.values.tolist()
    tercile_FI_idx = [trials_conditions[cond_ii].index(x) for x in tercile_FI_trials if x in trials_conditions[cond_ii]]
    heatmap_late = np.nanmean(psths_trials_conditions[cond_ii][tercile_FI_idx], axis = 0)[index_order]

    heatmap = heatmap_early - heatmap_late

    vmin = np.nanquantile(heatmap,.01)
    vmax = np.nanquantile(heatmap,.99)
    im = axs[cond_ii].imshow(heatmap, vmin = vmin, vmax = vmax, aspect = 'auto', cmap = 'bwr', interpolation = None)

    axs[cond_ii].set_title(f'{hue_variable}{hue_variable_list[cond_ii]}')

cbar = fig.colorbar(im, ax=axs[2], orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_ticks([vmin, vmax])
cbar.set_ticklabels(['late', 'early'])


figtitle = f'{animal} {date} | experiment {exp} | difference in neural activity for early and late transition point terciles'
plt.suptitle(figtitle)
plt.savefig(fr'{DANEURONS_PATH}\{figtitle.replace('|','_')}.png')

# %%


## and the next step was tiling a la Guga, centered in 0