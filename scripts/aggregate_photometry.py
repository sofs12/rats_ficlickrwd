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
from scipy.signal import savgol_filter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors

from ratcode.config.paths import PATH_STORE_PICKLES, DROPBOX_TASK_PATH, PATH_DATAFRAMES
from ratcode.common.logging import determine_experiment
from ratcode.common.colorcodes import *
from ratcode.behavior import change_point
from ratcode.photometry.photometry import get_prediction, quantile_regression, signal2eventsnippets, find_poly, segment_and_fit_function, butter_filter, mask_jumps, make_continuous, compute_snippets_across_days, drop_nan_rows_in_matrix, bootstrap_ci, plot_snippets
from ratcode.common.dataframe import group_and_listify
from ratcode.common.time import convert_date_bonsai, convert_timestamp
from ratcode.common.math import drop_nans_matrix
from ratcode.common.colorcodes import FI_order, color_FI_blocks, rwd_order, color_rwd_blocks

from ratcode.init import setup
setup()

#%%
import matplotlib.cm as cm
tercile_colors = ['#D95F02', '#B0B0B0', '#1B9E77']
tercile_list = ['T1', 'T2', 'T3']
tercile_rateH_colors = [cm.get_cmap('copper')(1-ii*.35) for ii in range(3)]

bone_cmap = cm.get_cmap('bone')
cmap_FI = mcolors.ListedColormap(color_FI_blocks)
cmap_rwd = mcolors.ListedColormap(color_rwd_blocks)
# %%

#PHOTOMETRY_PATH = os.path.join(DROPBOX_TASK_PATH, 'photometry', animal)
PATH_SAVE_DFS = os.path.join(DROPBOX_TASK_PATH, 'analysis_photometry')
PATH_SAVE_AGGREGATE_DA_FIGS = os.path.join(PATH_SAVE_DFS, 'aggregated_DAta')
#%%

aggregated_jointdf = []

for file in os.listdir(PATH_SAVE_DFS):
    if 'jointdf' in file:
        pkl_path = os.path.join(PATH_SAVE_DFS, file)
        jointdf = pd.read_pickle(pkl_path)
        jointdf['animal'] = file.split('_')[0]
        jointdf['date'] = file.split('_')[1]
        jointdf['experiment'] = determine_experiment(jointdf)
        aggregated_jointdf.append(jointdf)

if aggregated_jointdf:
    aggregated_jointdf = pd.concat(aggregated_jointdf, ignore_index=True)

    cols_to_move = ['animal', 'date', 'experiment']
    remaining_cols = [c for c in aggregated_jointdf.columns if c not in cols_to_move]
    aggregated_jointdf = aggregated_jointdf[cols_to_move + remaining_cols]

else:
    print("No files matched the criteria.")

#%%
## zscore DA per session (so that we can aggregate)
def zscore_session_arrays(group):
    all_values = np.concatenate([np.atleast_1d(val) for val in group.values])
    
    mu = np.nanmean(all_values)
    sigma = np.nanstd(all_values)

    if sigma == 0 or np.isnan(sigma):
        return group # Return as is if we can't scale
    
    return group.apply(lambda x: (np.array(x) - mu) / sigma)

aggregated_jointdf['DA_zscored_session'] = aggregated_jointdf.groupby(['animal', 'date'])['DA_poly_session'].transform(zscore_session_arrays)

#%%

aggregated_jointdf.keys()
#%%
cols_to_drop = ['bool_block', 'trial_start_arduino', 'trial_end_arduino',
       'trial_duration_arduino', 'lever_rel_arduino',
       'count_lever', 'pump_on_arduino',
       'pump_off_arduino', 'cp_arduino',
        'poke_rel_arduino']

aggregated_jointdf.drop(cols_to_drop, axis = 1, inplace = True)
#%%
# rename columns

rename_dict = {
        'timestamp_session': 'time_DA',
        'predicted_gfp_session': 'predicted_gfp',
        'trial_start_harp': 'trial_start',
        'trial_end_harp': 'trial_end',
        'trial_duration_harp': 'trial_duration',
        'lever_rel_harp': 'lever_rel',
        't_trial_harp': 't_trial',
        'last_lever_harp': 'last_lever',
        'pump_on_harp': 'pump_on',
        'pump_off_harp': 'pump_off',
        'lever_abs_harp': 'lever_abs',
        'last_lever_abs_harp': 'last_lever_abs',
        'prelast_lever_harp': 'prelast_lever',
        'prelast_lever_abs_harp': 'prelast_lever_abs',
        'cp_harp': 'cp',
        'poke_rel_harp': 'poke_rel',
        'poke_abs_harp': 'poke_abs',
        'click_harp': 'click_on'
        #'cp_abs_harp': 'cp_abs',
        #'interpress_after_cp_harp': 'interpress_after_cp',
        #'corrected_cp_harp': 'corrected_cp',
        #'corrected_cp_abs_harp': 'corrected_cp_abs',
        #'pump_on_abs_harp': 'pump_on_abs',
        #'pump_off_abs_harp': 'pump_off_abs',
        #'preprelast_lever_abs_harp': 'preprelast_lever_abs',
        #'click_abs_harp': 'click_on_abs',
        #'t_trial_harp_normalised': 't_trial_normalised'
        }

#%%
aggregated_jointdf = aggregated_jointdf.rename(columns = rename_dict)

#%%


aggregated_jointdf['animaldate'] = aggregated_jointdf.apply(lambda x: f'{x.animal}_{x.date}', axis = 1)

aggregated_jointdf['trial_in_block'] = aggregated_jointdf.groupby(['animaldate','blockno']).cumcount() + 1

aggregated_jointdf['bool_new_block'] = aggregated_jointdf['blockno'] != aggregated_jointdf['blockno'].shift(1)

aggregated_jointdf = aggregated_jointdf.reset_index(drop=True)
#%%
for key in ['blockno', 'FI', 'n_protocols']:
    aggregated_jointdf[f'prev_{key}'] = aggregated_jointdf.loc[aggregated_jointdf['bool_new_block'], key].groupby(aggregated_jointdf['animaldate']).shift(1)
    aggregated_jointdf[f'prev_{key}'] = aggregated_jointdf[f'prev_{key}'].ffill()
#%%
aggregated_jointdf['bool_cp'] = aggregated_jointdf.cp.apply(lambda x: np.isnan(x) == False)
aggregated_jointdf['bool_cp'] = aggregated_jointdf.apply(lambda x: False if x.cp > x.FI else np.isnan(x.cp) == False, axis=1)
aggregated_jointdf['cp'] = aggregated_jointdf.apply(lambda x: np.nan if not x.bool_cp else x.cp, axis=1)

aggregated_jointdf['cp_abs'] = aggregated_jointdf.cp + aggregated_jointdf.trial_start
aggregated_jointdf['interpress_after_cp'] = aggregated_jointdf.apply(lambda x: np.diff(x.lever_rel[x.lever_rel > x.cp]), axis = 1)
#aggregated_jointdf['corrected_cp'] = aggregated_jointdf.apply(lambda x: x.cp - np.mean(x.interpress_after_cp), axis = 1)
#aggregated_jointdf['corrected_cp_abs'] = aggregated_jointdf.corrected_cp + aggregated_jointdf.trial_start

aggregated_jointdf['pump_on_abs'] = aggregated_jointdf.pump_on + aggregated_jointdf.trial_start
aggregated_jointdf['pump_off_abs'] = aggregated_jointdf.pump_off + aggregated_jointdf.trial_start

#blocksdf['click_on_abs'] = blocksdf.click_on + blocksdf.trial_start

aggregated_jointdf['preprelast_lever_abs'] = aggregated_jointdf.lever_abs.apply(lambda x: x[-3] if len(x)>2 else np.nan)

#aggregated_jointdf['FI'] = aggregated_jointdf.FI.apply(lambda x: int(x/1000))
aggregated_jointdf['trialno_within_block'] = aggregated_jointdf.groupby(['animaldate', 'blockno']).cumcount() + 1

aggregated_jointdf['cp_FInormalised'] = aggregated_jointdf.cp/aggregated_jointdf.FI

#aggregated_jointdf['DA_trial_zscored'] = aggregated_jointdf.DA.apply(lambda x: compute_zscore(x))

#%%


## from the thesis nb

#aggregated_jointdf['tercile_cp_FInormalised'] = (
#    aggregated_jointdf.query('bool_cp')
#    .groupby('animaldate')['cp_FInormalised']
#    .transform(lambda x: pd.qcut(x, q=3, labels=tercile_list))
#)
#
#aggregated_jointdf['tercile_cp_FInormalised_withinFI'] = (
#    aggregated_jointdf.query('bool_cp')
#    .groupby(['animaldate','FI','n_protocols'])['cp_FInormalised']
#    .transform(lambda x: pd.qcut(x, q=3, labels=tercile_list))
#)
## some errors with nans

#%%
def categorize_presses(presses): ## exclude the last press from the terciles categorization
    n = len(presses) #I'm excluding the last press
    if n < 4:
        return ["out"] * n
    
    thirds = np.linspace(0, n-1, 4, dtype=int)  # [0, n/3, 2n/3, n]
    labels = []
    for i in range(3):
        labels.extend([f"t{i+1}"] * (thirds[i+1] - thirds[i]))

    labels.append(['last'])
    labels = np.hstack(labels)
    return labels

aggregated_jointdf['lever_index_category'] = aggregated_jointdf.lever_rel.apply(categorize_presses)

aggregated_jointdf.to_pickle(rf'{PATH_DATAFRAMES}\aggregate_photometry_Palladium_Ruthenium.pkl')
#%%

"""
.####.##.....##.########...#######..########..########.......###.....######....######...########..########..######......###....########.########....##.....##.########.########..########
..##..###...###.##.....##.##.....##.##.....##....##.........##.##...##....##..##....##..##.....##.##.......##....##....##.##......##....##..........##.....##.##.......##.....##.##......
..##..####.####.##.....##.##.....##.##.....##....##........##...##..##........##........##.....##.##.......##.........##...##.....##....##..........##.....##.##.......##.....##.##......
..##..##.###.##.########..##.....##.########.....##.......##.....##.##...####.##...####.########..######...##...####.##.....##....##....######......#########.######...########..######..
..##..##.....##.##........##.....##.##...##......##.......#########.##....##..##....##..##...##...##.......##....##..#########....##....##..........##.....##.##.......##...##...##......
..##..##.....##.##........##.....##.##....##.....##.......##.....##.##....##..##....##..##....##..##.......##....##..##.....##....##....##..........##.....##.##.......##....##..##......
.####.##.....##.##.........#######..##.....##....##.......##.....##..######....######...##.....##.########..######...##.....##....##....########....##.....##.########.##.....##.########
"""
aggregated_jointdf = pd.read_pickle(rf'{PATH_DATAFRAMES}\aggregate_photometry_Palladium_Ruthenium.pkl')




#%%%

"""
.########.##.....##.########..######..####..######.....##....##.########.
....##....##.....##.##.......##....##..##..##....##....###...##.##.....##
....##....##.....##.##.......##........##..##..........####..##.##.....##
....##....#########.######....######...##...######.....##.##.##.########.
....##....##.....##.##.............##..##........##....##..####.##.....##
....##....##.....##.##.......##....##..##..##....##....##...###.##.....##
....##....##.....##.########..######..####..######.....##....##.########.
"""
# knobs
alignment_idx = 1
baseline_correct = False
DA_column = 'DA_zscored_session'
#DA_column = 'DA_envelope_z'

animal = 'Palladium'
df = aggregated_jointdf.query(f'animal == "{animal}"')# and date in {photometry_dates_dict[animal]}')

#df = allphotometrydf.query(f'animal == "{animal}"')


window = (-4,4)
zero_time = int((window[1] - window[0])/2*100)

alignments = ['cp_abs', 'last_lever_abs']
alignment_labels = ['transition point', 'last press']

if alignment_idx == 1:
    baseline_start = zero_time-30
    baseline_end = zero_time
else:
    baseline_start = zero_time-100
    baseline_end = zero_time-50



baseline_title = ' | baseline corrected' if baseline_correct else ''

exp = 'a'
snippets_a = []
for FI in FI_order:
    time, snippets = compute_snippets_across_days(df,
        f'experiment == "{exp}" and FI == {FI}', alignments[alignment_idx], DA_column, window)
    snippets_a.append(snippets)

exp = 'b'
snippets_b = []
for FI in FI_order:
    time, snippets = compute_snippets_across_days(df,
        f'experiment == "{exp}" and FI == {FI}', alignments[alignment_idx], DA_column, window)
    snippets_b.append(snippets)

exp = 'c'
snippets_c = []
for nprots in rwd_order:
    time, snippets = compute_snippets_across_days(df, 
        f'experiment == "{exp}" and n_protocols == {nprots}', alignments[alignment_idx], DA_column, window)
    snippets_c.append(snippets)



fig, axs = plt.subplots(1,3, figsize = (12,4), tight_layout = True, sharex = True, sharey = True)

for ii in range(3):
    if baseline_correct:
        axs[0].plot(time,np.nanmean(snippets_a[ii], axis = 0) - np.mean(np.nanmean(snippets_a[ii], axis=0)[baseline_start:baseline_end]), color = color_FI_blocks[ii])
        axs[1].plot(time,np.nanmean(snippets_b[ii], axis = 0) - np.mean(np.nanmean(snippets_b[ii], axis=0)[baseline_start:baseline_end]), color = color_FI_blocks[ii])
        axs[2].plot(time,np.nanmean(snippets_c[ii], axis = 0) - np.mean(np.nanmean(snippets_c[ii], axis=0)[baseline_start:baseline_end]), color = color_nprots_blocks[ii])
            
    else:
        axs[0].plot(time,np.nanmean(snippets_a[ii], axis = 0), color = color_FI_blocks[ii])
        axs[1].plot(time,np.nanmean(snippets_b[ii], axis = 0), color = color_FI_blocks[ii])
        axs[2].plot(time,np.nanmean(snippets_c[ii], axis = 0), color = color_rwd_blocks[ii])

    axs[ii].axvline(0, color = 'grey', lw = 1)

    axs[ii].set_xlabel(f'time since {alignment_labels[alignment_idx]} (s)')

axs[0].set_title('varying FI')
axs[1].set_title('fixed reward rate')
axs[2].set_title('varying reward magnitude')

axs[0].set_ylabel('DA (z ΔF/F)')

figtitle = f'{animal} all | DA aligned to {alignment_labels[alignment_idx]}{baseline_title} | {DA_column}'
#figtitle = f'{animal} all | DA aligned to {alignment_labels[alignment_idx]}{baseline_title}_RPE'
fig.suptitle(figtitle)
fig.savefig(fr'{PATH_SAVE_AGGREGATE_DA_FIGS}\{figtitle.replace('|', '_')}.png')
#fig.savefig(fr'{photometry_fig_path}\{figtitle.replace('|', '_')}.pdf')


#%%


animal = 'Palladium'
df = aggregated_jointdf.query(f'animal == "{animal}"')# and date in {photometry_dates_dict[animal]}')
time, snippets_cp = compute_snippets_across_days(df,f'bool_cp', 'last_lever_abs','DA_zscored_session', (-4,4))
time, snippets_nocp = compute_snippets_across_days(df,f'bool_cp == False', 'last_lever_abs','DA_zscored_session', (-4,4))

alignment_idx = 1
DA_column = 'DA_zscored_session'

fig, axs = plt.subplots(2,3, tight_layout = True, figsize = (12,8))


### exp a ####

exp = 'a'
snippets_a = []
FI_a = []
for FI in FI_order:
    time, snippets = compute_snippets_across_days(df,
        f'experiment == "{exp}" and FI == {FI}', alignments[alignment_idx], DA_column, window)
    snippets = drop_nan_rows_in_matrix(snippets)
    FI_vals = FI*np.ones(len(snippets))
    snippets_a.append(snippets)
    FI_a.append(FI_vals)


snippets_a = np.vstack(snippets_a)
FI_a = np.hstack(FI_a)

vmin,vmax = np.nanquantile(snippets_a,[.01,.99])
axs[0,0].imshow(snippets_a, vmin = vmin, vmax = vmax, aspect = 'auto', cmap = bone_cmap,
                extent = [time[0], time[-1], len(snippets_a), 1])

for ii in range(3):
    axs[1,0].plot(time, np.nanmedian(snippets_a[FI_a == FI_order[ii]], axis = 0), color = color_FI_blocks[ii])

ax_FI = inset_axes(axs[0,0], width="4%", height="100%", loc="center left", borderpad=0)
ax_FI.matshow(FI_a.reshape(len(FI_a),1), aspect = 'auto', cmap = cmap_FI)
ax_FI.set_axis_off()
ax_FI = inset_axes(axs[0,0], width="4%", height="100%", loc="center right", borderpad=0)
ax_FI.matshow(FI_a.reshape(len(FI_a),1), aspect = 'auto', cmap = cmap_FI)
ax_FI.set_axis_off()



### exp b ####

exp = 'b'
snippets_b = []
FI_b = []
for FI in FI_order:
    time, snippets = compute_snippets_across_days(df,
        f'experiment == "{exp}" and FI == {FI}', alignments[alignment_idx], DA_column, window)
    snippets = drop_nan_rows_in_matrix(snippets)
    FI_vals = FI*np.ones(len(snippets))
    snippets_b.append(snippets)
    FI_b.append(FI_vals)


snippets_b = np.vstack(snippets_b)
FI_b = np.hstack(FI_b)

vmin,vmax = np.nanquantile(snippets_b,[.01,.99])
axs[0,1].imshow(snippets_b, vmin = vmin, vmax = vmax, aspect = 'auto', cmap = bone_cmap,
                extent = [time[0], time[-1], len(snippets_b), 1])

for ii in range(3):
    axs[1,1].plot(time, np.nanmedian(snippets_b[FI_b == FI_order[ii]], axis = 0), color = color_FI_blocks[ii])

ax_FI = inset_axes(axs[0,1], width="4%", height="100%", loc="center left", borderpad=0)
ax_FI.matshow(FI_b.reshape(len(FI_b),1), aspect = 'auto', cmap = cmap_FI)
ax_FI.set_axis_off()
ax_FI = inset_axes(axs[0,1], width="4%", height="100%", loc="center right", borderpad=0)
ax_FI.matshow(FI_b.reshape(len(FI_b),1), aspect = 'auto', cmap = cmap_FI)
ax_FI.set_axis_off()



### exp c ####

exp = 'c'
snippets_c = []
rwd_c = []
for nprots in rwd_order:
    time, snippets = compute_snippets_across_days(df,
        f'experiment == "{exp}" and n_protocols == {nprots}', alignments[alignment_idx], DA_column, window)
    snippets = drop_nan_rows_in_matrix(snippets)
    rwd_vals = nprots*np.ones(len(snippets))
    snippets_c.append(snippets)
    rwd_c.append(rwd_vals)


snippets_c = np.vstack(snippets_c)
rwd_c = np.hstack(rwd_c)

vmin,vmax = np.nanquantile(snippets_c,[.01,.99])
axs[0,2].imshow(snippets_c, vmin = vmin, vmax = vmax, aspect = 'auto', cmap = bone_cmap,
                extent = [time[0], time[-1], len(snippets_c), 1])

for ii in range(3):
    axs[1,2].plot(time, np.nanmedian(snippets_c[rwd_c == rwd_order[ii]], axis = 0), color = color_rwd_blocks[ii])

ax_rwd = inset_axes(axs[0,2], width="4%", height="100%", loc="center left", borderpad=0)
ax_rwd.matshow(rwd_c.reshape(len(rwd_c),1), aspect = 'auto', cmap = cmap_rwd)
ax_rwd.set_axis_off()
ax_rwd = inset_axes(axs[0,2], width="4%", height="100%", loc="center right", borderpad=0)
ax_rwd.matshow(rwd_c.reshape(len(rwd_c),1), aspect = 'auto', cmap = cmap_rwd)
ax_rwd.set_axis_off()


axs[1,0].sharey(axs[1,1])
axs[1,2].sharey(axs[1,1])


for ii in range(3):
    axs[1,ii].axvline(0, color = 'grey', lw = 1)
    axs[1,ii].set_xlabel(f'time since {alignment_labels[alignment_idx]} (s)')

axs[0,0].set_title('varying FI')
axs[0,1].set_title('fixed reward rate')
axs[0,2].set_title('varying reward magnitude')

axs[1,0].set_ylabel('DA (a.u.)')

figtitle = f'{animal} all | DA aligned to {alignment_labels[alignment_idx]}{baseline_title} | with heatmap'
#figtitle = f'{animal} all | DA aligned to {alignment_labels[alignment_idx]}{baseline_title}_RPE'
fig.suptitle(figtitle)
fig.savefig(fr'{PATH_SAVE_AGGREGATE_DA_FIGS}\{figtitle.replace('|', '_')}.png')
#%%


#### some issues with Palladium's data
## for sure juumps in the DA session data; look at day by day weird stuff. maybe we just need to acquire more clean data from exp c from this guy


plt.plot(df.query('experiment == "c" and n_protocols == 7').DA_zscored_session.apply(lambda x: np.mean(x)).values)
plt.plot(df.query('experiment == "c" and n_protocols == 14').DA_zscored_session.apply(lambda x: np.mean(x)).values)
plt.plot(df.query('experiment == "c" and n_protocols == 28').DA_zscored_session.apply(lambda x: np.mean(x)).values)
#%%

dates = df.query('experiment == "c" and n_protocols == 7').date.unique()
#%%
time, snippets = compute_snippets_across_days(df,
    f'experiment == "c" and n_protocols == 7 and date in {list(dates)}', alignments[alignment_idx], 'DA_poly_session', window)
#snippets = drop_nan_rows_in_matrix(snippets)
plt.figure()
plt.imshow(snippets, aspect = 'auto')
#plt.title(date)
plt.show()
#%%
df.query('experiment == "c"').date.unique()
#%%
for rr in rwd_order:
    plt.figure()
    sns.histplot(np.hstack(snippets_c[rwd_c == rr]))
    plt.show()

#%%
plt.imshow(zscore(snippets_c, axis = 1))
#%%
for rr in rwd_order:
    plt.plot(np.nanmean(zscore(snippets_c, axis = 1)[rwd_c == rr], axis = 0))


#%%

aggregated_jointdf['tercile_cp_FInormalised'] = (
    aggregated_jointdf.query('bool_cp')
    .groupby('animaldate')['cp_FInormalised']
    .transform(lambda x: pd.qcut(x, q=3, labels=tercile_list))
)

aggregated_jointdf['tercile_cp_FInormalised_withinFI'] = (
    aggregated_jointdf.query('animal in ["Zirconium", "Niobium"] and bool_cp')
    .groupby(['animaldate','FI','n_protocols'])['cp_FInormalised']
    .transform(lambda x: pd.qcut(x, q=3, labels=tercile_list))
)
#%%

aggregated_jointdf['count_lever'] = aggregated_jointdf.lever_rel.apply(lambda x: len(x))
#%%
"""
.########.##.....##.########..##........#######..########..########.########.....########..########
.##........##...##..##.....##.##.......##.....##.##.....##.##.......##.....##....##.....##.##......
.##.........##.##...##.....##.##.......##.....##.##.....##.##.......##.....##....##.....##.##......
.######......###....########..##.......##.....##.##.....##.######...##.....##....##.....##.######..
.##.........##.##...##........##.......##.....##.##.....##.##.......##.....##....##.....##.##......
.##........##...##..##........##.......##.....##.##.....##.##.......##.....##....##.....##.##......
.########.##.....##.##........########..#######..########..########.########.....########..##......
"""

def align_DA_to_ttls(blocksdf, ttl_column, DA_column = 'DA_zscored_per_session', alignment_window=(-2, 2), bin_size=0.01):
    """
    Align DA activity to TTL events for each session in blocksdf.

    Parameters:
    -----------
    blocksdf : pd.DataFrame
        DataFrame containing DA activity and TTL events. Must include 'animaldate', 'time_DA', and 'DA'.
    ttl_column : str
        Column name in blocksdf containing TTL events to align DA activity to.
    alignment_window : tuple, optional
        Time window around TTL events to align DA activity (default: (-2, 2)).
    bin_size : float, optional
        Size of bins for alignment (default: 0.01 seconds).

    Returns:
    --------
    aligned_snippets : dict
        Dictionary where keys are 'animaldate' and values are aligned DA snippets for each session.
    alignment_time : np.array
        Array of time bins used for alignment.
    """
    aligned_snippets = {}
    all_snippets = []  # List to collect all snippets

    n_timestamps = 1 + (alignment_window[1]-alignment_window[0])/bin_size

    for animaldate in blocksdf['animaldate'].unique():
        session_df = blocksdf.query(f'animaldate == "{animaldate}"')
        time_DA = np.hstack(session_df['time_DA'].values)
        DA_signal = np.hstack(session_df[DA_column].values)
        ttl_events = session_df[ttl_column].values  # Use the specified TTL column

        # Align DA activity to TTL events
        snippets, time = signal2eventsnippets(time_DA, DA_signal, ttl_events, alignment_window, bin_size, nanify=False)
        aligned_snippets[animaldate] = snippets
        all_snippets.append(snippets)

    all_snippets = np.vstack(all_snippets)

    return aligned_snippets, time, all_snippets
#%%
aggregated_jointdf.animaldate[aggregated_jointdf.apply(lambda x: type(x.lever_index), axis = 1) == float].unique()
#%%
#exploded_df = aggregated_jointdf.query(f'date in {list(dates)}').explode(['lever_abs','lever_index', 'lever_index_category'])
exploded_df = aggregated_jointdf.explode(['lever_abs','lever_index', 'lever_index_category'])
exploded_df = exploded_df[exploded_df['lever_index'].notna()]  # Remove NaNs
exploded_df['lever_index'] = pd.to_numeric(exploded_df['lever_index'], errors='coerce')  # Ensure numeric

_, time, all_snippets = align_DA_to_ttls(exploded_df, 'lever_abs', 'DA_zscored_session', alignment_window=(-2, 2), bin_size=0.01)
# Discard rows with NaNs
cleaned_snippets = all_snippets[~np.isnan(all_snippets).any(axis=1)]

#%%


fig, axs = plt.subplots(1,2, tight_layout = True, figsize = (8,4))

rwd_color = bone_cmap(.2)
non_rwd_color = bone_cmap(.7)

t_4 = np.linspace(-2,2,401)

animal = 'Ruthenium'
non_rwd_mask = (exploded_df['lever_index'] < -1) & (exploded_df['animal'] == animal)
rwd_mask = (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)

median, lower, upper = bootstrap_ci(all_snippets[non_rwd_mask], n_boot=500, smooth_sigma=5)
axs[0].plot(t_4, median, color = non_rwd_color, lw = 2, label = 'unrewarded press')
axs[0].fill_between(t_4, lower, upper, color=non_rwd_color, alpha=0.2)

median, lower, upper = bootstrap_ci(all_snippets[rwd_mask], n_boot=500, smooth_sigma=5)
axs[0].plot(t_4, median, color = rwd_color, lw = 2, label = 'rewarded press')
axs[0].fill_between(t_4, lower, upper, color=rwd_color, alpha=0.2)

#axs[0].plot(np.nanmean(all_snippets[non_rwd_mask], axis = 0), color = bone_cmap(0.7), lw = 2)
#axs[0].plot(np.nanmean(all_snippets[rwd_mask], axis = 0), color = bone_cmap(0.2), lw = 2)

animal = 'Palladium'
non_rwd_mask = (exploded_df['lever_index'] < -1) & (exploded_df['animal'] == animal)
rwd_mask = (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
#axs[1].plot(np.nanmean(all_snippets[non_rwd_mask], axis = 0), color = bone_cmap(0.7), lw = 2)
#axs[1].plot(np.nanmean(all_snippets[rwd_mask], axis = 0), color = bone_cmap(0.2), lw = 2)

median, lower, upper = bootstrap_ci(all_snippets[non_rwd_mask], n_boot=500, smooth_sigma=5)
axs[1].plot(t_4, median, color = non_rwd_color, lw = 2, label = 'unrewarded')
axs[1].fill_between(t_4, lower, upper, color=non_rwd_color, alpha=0.2)

median, lower, upper = bootstrap_ci(all_snippets[rwd_mask], n_boot=500, smooth_sigma=5)
axs[1].plot(t_4, median, color = rwd_color, lw = 2, label = 'rewarded')
axs[1].fill_between(t_4, lower, upper, color=rwd_color, alpha=0.2)

[axs[ii].axvline(0, color = 'grey', ls = '--') for ii in range(2)]
[axs[ii].set_xlabel('time since lever press (s)') for ii in range(2)]

axs[0].set_ylabel('DA (z ΔF/F)')

axs[1].legend(frameon = False, fontsize = 12)

axs[0].set_title('Ruthenium')
axs[1].set_title('Palladium')

figtitle = 'DA aligned to rewarded and unrewarded lever presses'
plt.suptitle(figtitle)

plt.savefig(rf'{PATH_SAVE_AGGREGATE_DA_FIGS}\{figtitle.replace('|','_')}.png')
#plt.savefig(rf'{photometry_fig_path}\{figtitle.replace('|','_')}.pdf')

#%%

fig, axs = plt.subplots(1,2, tight_layout = True, figsize = (8,4))

labels = ['initial', 'middle', 'pre-terminal']

animal = 'Ruthenium'
mask_presses_t1 = (exploded_df['bool_cp'] == True) & (exploded_df['lever_index_category'] == 't1') & (exploded_df['animal'] == animal)
mask_presses_t2 = (exploded_df['bool_cp'] == True) & (exploded_df['lever_index_category'] == 't2') & (exploded_df['animal'] == animal)
mask_presses_t3 = (exploded_df['bool_cp'] == True) & (exploded_df['lever_index_category'] == 't3') & (exploded_df['animal'] == animal)
mask_presses_last = (exploded_df['bool_cp'] == True) & (exploded_df['lever_index_category'] == 'last') & (exploded_df['animal'] == animal)
masks = [mask_presses_t1, mask_presses_t2, mask_presses_t3]

for ii, mask in enumerate(masks):

    color = bone_cmap(.8-.2*ii) ## start with 0.2 if including the last lever press
    
    median, lower, upper = bootstrap_ci(all_snippets[mask], n_boot=50, smooth_sigma=5)
    axs[0].plot(t_4, median, color = color, lw = 2, label = labels[ii])
    axs[0].fill_between(t_4, lower, upper, color=color, alpha=0.2)


animal = 'Palladium'
mask_presses_t1 = (exploded_df['bool_cp'] == True) & (exploded_df['lever_index_category'] == 't1') & (exploded_df['animal'] == animal)
mask_presses_t2 = (exploded_df['bool_cp'] == True) & (exploded_df['lever_index_category'] == 't2') & (exploded_df['animal'] == animal)
mask_presses_t3 = (exploded_df['bool_cp'] == True) & (exploded_df['lever_index_category'] == 't3') & (exploded_df['animal'] == animal)
mask_presses_last = (exploded_df['bool_cp'] == True) & (exploded_df['lever_index_category'] == 'last') & (exploded_df['animal'] == animal)
masks = [mask_presses_t1, mask_presses_t2, mask_presses_t3]

for ii, mask in enumerate(masks):

    color = bone_cmap(.8-.2*ii)
    
    median, lower, upper = bootstrap_ci(all_snippets[mask], n_boot=50, smooth_sigma=5)
    axs[1].plot(t_4, median, color = color, lw = 2, label = labels[ii])
    axs[1].fill_between(t_4, lower, upper, color=color, alpha=0.2)

[axs[ii].axvline(0, color = 'grey', ls = '--') for ii in range(2)]
[axs[ii].set_xlabel('time since lever press (s)') for ii in range(2)]

axs[0].set_ylabel('DA (z ΔF/F)')

axs[1].legend(frameon = False, fontsize = 12)

axs[0].set_title('Ruthenium')
axs[1].set_title('Palladium')

figtitle = 'DA aligned to unrewarded lever presses | split by early vs late presses'
#figtitle = 'DA aligned to rewarded and unrewarded lever presses | split by early vs late presses'
plt.suptitle(figtitle)

plt.savefig(rf'{PATH_SAVE_AGGREGATE_DA_FIGS}\{figtitle.replace('|','_')}.png')
#plt.savefig(rf'{photometry_fig_path}\{figtitle.replace('|','_')}.pdf')


#%%
#aggregated_jointdf['rateH'] = aggregated_jointdf.query('bool_cp').apply(lambda x: len(x.lever_rel[x.lever_rel >= x.cp])/(x.trial_duration - x.cp), axis = 1)
#
#aggregated_jointdf['tercile_rateH'] = (
#    aggregated_jointdf['rateH']
#    #.groupby(['FI'])['cp']
#    .transform(terciles)
#)
#%%
fig, axs = plt.subplots(1,2, tight_layout = True, figsize = (8,4))

color_non_transition = '#1b6a9e'
labels = ['early', 'intermediate', 'late', 'later-than-FI']

animal = 'Ruthenium'

mask_cp_T1 = (exploded_df['tercile_cp_FInormalised'] == 'T1') & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
mask_cp_T2 = (exploded_df['tercile_cp_FInormalised'] == 'T2') & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
mask_cp_T3 = (exploded_df['tercile_cp_FInormalised'] == 'T3') & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
#mask_cp_trial = (exploded_df['bool_cp'] == True) & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
mask_nocp_trial = (exploded_df['bool_cp'] == False) & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
masks = [mask_cp_T1,mask_cp_T2, mask_cp_T3, mask_nocp_trial]

for ii, mask in enumerate(masks):

    color = tercile_colors[ii] if ii < 3 else color_non_transition
    
    median, lower, upper = bootstrap_ci(all_snippets[mask], n_boot=50, smooth_sigma=5)
    axs[0].plot(t_4, median, color = color, lw = 2, label = labels[ii])
    axs[0].fill_between(t_4, lower, upper, color=color, alpha=0.2)


animal = 'Palladium'

mask_cp_T1 = (exploded_df['tercile_cp_FInormalised'] == 'T1') & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
mask_cp_T2 = (exploded_df['tercile_cp_FInormalised'] == 'T2') & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
mask_cp_T3 = (exploded_df['tercile_cp_FInormalised'] == 'T3') & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
#mask_cp_trial = (exploded_df['bool_cp'] == True) & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
mask_nocp_trial = (exploded_df['bool_cp'] == False) & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
masks = [mask_cp_T1,mask_cp_T2, mask_cp_T3, mask_nocp_trial]

for ii, mask in enumerate(masks):

    color = tercile_colors[ii] if ii < 3 else color_non_transition
    
    median, lower, upper = bootstrap_ci(all_snippets[mask], n_boot=50, smooth_sigma=5)
    axs[1].plot(t_4, median, color = color, lw = 2, label = labels[ii])
    axs[1].fill_between(t_4, lower, upper, color=color, alpha=0.2)


[axs[ii].axvline(0, color = 'grey', ls = '--') for ii in range(2)]
[axs[ii].set_xlabel('time since lever press (s)') for ii in range(2)]

axs[0].set_ylabel('DA (z ΔF/F)')

axs[1].legend(frameon = False, fontsize = 11)

axs[0].set_title('Ruthenium')
axs[1].set_title('Palladium')

figtitle = 'DA aligned to last lever press | conditioned by transition point'
plt.suptitle(figtitle)

plt.savefig(rf'{PATH_SAVE_AGGREGATE_DA_FIGS}\{figtitle.replace('|','_')}.png')
#plt.savefig(rf'{photometry_fig_path}\{figtitle.replace('|','_')}.pdf')

#%%
fig, axs = plt.subplots(1,2, tight_layout = True, figsize = (8,4))

labels = ['early', 'intermediate', 'late', 'later-than-FI']

animal = 'Ruthenium'
df = aggregated_jointdf.query(f'animal == "{animal}"')# and date in {photometry_dates_dict[animal]}')
_, snippets_cp_T1 = compute_snippets_across_days(df,f'bool_cp and tercile_cp_FInormalised == "T1"', 'cp_abs','DA_zscored_session', (-2,2))
_, snippets_cp_T2 = compute_snippets_across_days(df,f'bool_cp and tercile_cp_FInormalised == "T2"', 'cp_abs','DA_zscored_session', (-2,2))
_, snippets_cp_T3 = compute_snippets_across_days(df,f'bool_cp and tercile_cp_FInormalised == "T3"', 'cp_abs','DA_zscored_session', (-2,2))
_, snippets_cp_afterFI = compute_snippets_across_days(df,f'bool_cp == False and count_lever < 3', 'last_lever_abs','DA_zscored_session', (-2,2))

all_cp_snippets = [snippets_cp_T1, snippets_cp_T2, snippets_cp_T3]

for ii, snippets in enumerate(all_cp_snippets):
    color = tercile_colors[ii] if ii < 3 else color_non_transition

    median, lower, upper = bootstrap_ci(snippets, n_boot=50, smooth_sigma=5)
    axs[0].plot(t_4, median, color = color, lw = 2, label = labels[ii])
    axs[0].fill_between(t_4, lower, upper, color=color, alpha=0.2)

## pre lever later-than-FI presses
mask_nocp_trial = (exploded_df['count_lever'] < 3) & (exploded_df['bool_cp'] == False) & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
median, lower, upper = bootstrap_ci(all_snippets[mask_nocp_trial], n_boot=50, smooth_sigma=5)
axs[0].plot(t_4[:200], median[:200], color = color_non_transition, lw = 2, label = labels[ii])
axs[0].fill_between(t_4[:200], lower[:200], upper[:200], color=color_non_transition, alpha=0.2)


animal = 'Palladium'
df = aggregated_jointdf.query(f'animal == "{animal}"')# and date in {photometry_dates_dict[animal]}')
_, snippets_cp_T1 = compute_snippets_across_days(df,f'bool_cp and tercile_cp_FInormalised == "T1"', 'cp_abs','DA_zscored_session', (-2,2))
_, snippets_cp_T2 = compute_snippets_across_days(df,f'bool_cp and tercile_cp_FInormalised == "T2"', 'cp_abs','DA_zscored_session', (-2,2))
_, snippets_cp_T3 = compute_snippets_across_days(df,f'bool_cp and tercile_cp_FInormalised == "T3"', 'cp_abs','DA_zscored_session', (-2,2))
_, snippets_cp_afterFI = compute_snippets_across_days(df,f'bool_cp == False and count_lever == 1', 'last_lever_abs','DA_zscored_session', (-2,2))

all_cp_snippets = [snippets_cp_T1, snippets_cp_T2, snippets_cp_T3]

for ii, snippets in enumerate(all_cp_snippets):
    color = tercile_colors[ii] if ii < 3 else color_non_transition

    median, lower, upper = bootstrap_ci(snippets, n_boot=50, smooth_sigma=5)
    axs[1].plot(t_4, median, color = color, lw = 2, label = labels[ii])
    axs[1].fill_between(t_4, lower, upper, color=color, alpha=0.2)

## pre lever later-than-FI presses
mask_nocp_trial = (exploded_df['count_lever'] < 3) & (exploded_df['bool_cp'] == False) & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
median, lower, upper = bootstrap_ci(all_snippets[mask_nocp_trial], n_boot=50, smooth_sigma=5)
axs[1].plot(t_4[:200], median[:200], color = color_non_transition, lw = 2, label = labels[-1])
axs[1].fill_between(t_4[:200], lower[:200], upper[:200], color=color_non_transition, alpha=0.2)


[axs[ii].axvline(0, color = 'grey', ls = '--') for ii in range(2)]
[axs[ii].set_xlabel('time since lever press (s)') for ii in range(2)]

axs[0].set_ylabel('DA (z ΔF/F)')

axs[1].legend(frameon = False, fontsize = 11)

axs[0].set_title('Ruthenium')
axs[1].set_title('Palladium')

figtitle = 'DA aligned to transition point | conditioned by transition point'
plt.suptitle(figtitle)

plt.savefig(rf'{PATH_SAVE_AGGREGATE_DA_FIGS}\{figtitle.replace('|','_')}.png')
#plt.savefig(rf'{photometry_fig_path}\{figtitle.replace('|','_')}.pdf')
#%%
fig, axs = plt.subplots(1,2, tight_layout = True, figsize = (8,4))

labels = ['early', 'intermediate', 'late', 'later-than-FI']

animal1 = 'Ruthenium'
animal2 = 'Palladium'

animal = animal1
df = aggregated_jointdf.query(f'animal == "{animal}"')# and date in {photometry_dates_dict[animal]}')
_, snippets_cp_T1 = compute_snippets_across_days(df,f'bool_cp and experiment == "a" and tercile_cp_FInormalised == "T1"', 'last_lever_abs','DA_zscored_session', (-2,2))
_, snippets_cp_T2 = compute_snippets_across_days(df,f'bool_cp and experiment == "a" and tercile_cp_FInormalised == "T2"', 'last_lever_abs','DA_zscored_session', (-2,2))
_, snippets_cp_T3 = compute_snippets_across_days(df,f'bool_cp and experiment == "a" and tercile_cp_FInormalised == "T3"', 'last_lever_abs','DA_zscored_session', (-2,2))
_, snippets_cp_afterFI = compute_snippets_across_days(df,f'bool_cp == False and experiment == "a" and count_lever < 3', 'last_lever_abs','DA_zscored_session', (-2,2))

all_cp_snippets = [snippets_cp_T1, snippets_cp_T2, snippets_cp_T3]

for ii, snippets in enumerate(all_cp_snippets):
    color = tercile_colors[ii] if ii < 3 else color_non_transition

    median, lower, upper = bootstrap_ci(snippets, n_boot=50, smooth_sigma=5)
    axs[0].plot(t_4, median, color = color, lw = 2, label = labels[ii])
    axs[0].fill_between(t_4, lower, upper, color=color, alpha=0.2)

## pre lever later-than-FI presses
mask_nocp_trial = (exploded_df['count_lever'] < 3) & (exploded_df['bool_cp'] == False) & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
median, lower, upper = bootstrap_ci(all_snippets[mask_nocp_trial], n_boot=50, smooth_sigma=5)
axs[0].plot(t_4, median, color = color_non_transition, lw = 2, label = labels[ii])
axs[0].fill_between(t_4, lower, upper, color=color_non_transition, alpha=0.2)


animal = animal2
df = aggregated_jointdf.query(f'animal == "{animal}"')# and date in {photometry_dates_dict[animal]}')
_, snippets_cp_T1 = compute_snippets_across_days(df,f'bool_cp and experiment == "a" and tercile_cp_FInormalised == "T1"', 'last_lever_abs','DA_zscored_session', (-2,2))
_, snippets_cp_T2 = compute_snippets_across_days(df,f'bool_cp and experiment == "a" and tercile_cp_FInormalised == "T2"', 'last_lever_abs','DA_zscored_session', (-2,2))
_, snippets_cp_T3 = compute_snippets_across_days(df,f'bool_cp and experiment == "a" and tercile_cp_FInormalised == "T3"', 'last_lever_abs','DA_zscored_session', (-2,2))
_, snippets_cp_afterFI = compute_snippets_across_days(df,f'bool_cp == False and experiment == "a" and count_lever == 1', 'last_lever_abs','DA_zscored_session', (-2,2))

all_cp_snippets = [snippets_cp_T1, snippets_cp_T2, snippets_cp_T3]

for ii, snippets in enumerate(all_cp_snippets):
    color = tercile_colors[ii] if ii < 3 else color_non_transition

    median, lower, upper = bootstrap_ci(snippets, n_boot=50, smooth_sigma=5)
    axs[1].plot(t_4, median, color = color, lw = 2, label = labels[ii])
    axs[1].fill_between(t_4, lower, upper, color=color, alpha=0.2)

## pre lever later-than-FI presses
mask_nocp_trial = (exploded_df['count_lever'] < 3) & (exploded_df['bool_cp'] == False) & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
median, lower, upper = bootstrap_ci(all_snippets[mask_nocp_trial], n_boot=50, smooth_sigma=5)
axs[1].plot(t_4, median, color = color_non_transition, lw = 2, label = labels[-1])
axs[1].fill_between(t_4, lower, upper, color=color_non_transition, alpha=0.2)


[axs[ii].axvline(0, color = 'grey', ls = '--') for ii in range(2)]
[axs[ii].set_xlabel('time since last press (s)') for ii in range(2)]

axs[0].set_ylabel('DA (z ΔF/F)')

axs[1].legend(frameon = False, fontsize = 11)

axs[0].set_title(animal1)
axs[1].set_title(animal2)

figtitle = 'DA aligned to last press | conditioned by transition point | exp a same rwd magnitude'
plt.suptitle(figtitle)

plt.savefig(rf'{PATH_SAVE_AGGREGATE_DA_FIGS}\{figtitle.replace('|','_')}.png')
#plt.savefig(rf'{photometry_fig_path}\{figtitle.replace('|','_')}.pdf')

#%%
animal = animal1

n_bot = 500

t_4 = np.linspace(-2,2,401)
color_non_transition = '#1b6a9e'
labels = ['early', 'intermediate', 'late', 'later-than-FI']

fig, axs = plt.subplots(1,5, tight_layout = True, figsize = (12,4), width_ratios=[3,1,1,1,3], sharey = True)


df = aggregated_jointdf.query(f'animal == "{animal}"')# and date in {photometry_dates_dict[animal]}')
_, snippets_cp_T1 = compute_snippets_across_days(df,f'bool_cp and tercile_cp_FInormalised == "T1"', 'cp_abs','DA_zscored_session', (-2,2))
_, snippets_cp_T2 = compute_snippets_across_days(df,f'bool_cp and tercile_cp_FInormalised == "T2"', 'cp_abs','DA_zscored_session', (-2,2))
_, snippets_cp_T3 = compute_snippets_across_days(df,f'bool_cp and tercile_cp_FInormalised == "T3"', 'cp_abs','DA_zscored_session', (-2,2))
_, snippets_cp_afterFI = compute_snippets_across_days(df,f'bool_cp == False and count_lever < 3', 'last_lever_abs','DA_zscored_session', (-2,2))
_, snippets_cp_afterFI = compute_snippets_across_days(df,f'bool_cp == False and count_lever == 1', 'last_lever_abs','DA_zscored_session', (-2,2))

all_cp_snippets = [snippets_cp_T1, snippets_cp_T2, snippets_cp_T3]

for ii, snippets in enumerate(all_cp_snippets):
    color = tercile_colors[ii] if ii < 3 else color_non_transition

    median, lower, upper = bootstrap_ci(snippets, n_boot=n_bot, smooth_sigma=5)
    axs[0].plot(t_4, median, color = color, lw = 2, label = labels[ii])
    axs[0].fill_between(t_4, lower, upper, color=color, alpha=0.2)



mask_cp_T1 = (exploded_df['tercile_cp_FInormalised'] == 'T1') & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
mask_cp_T2 = (exploded_df['tercile_cp_FInormalised'] == 'T2') & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
mask_cp_T3 = (exploded_df['tercile_cp_FInormalised'] == 'T3') & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
masks = [mask_cp_T1,mask_cp_T2, mask_cp_T3]

for ii, mask in enumerate(masks):

    color = tercile_colors[ii] if ii < 3 else color_non_transition
    
    median, lower, upper = bootstrap_ci(all_snippets[mask], n_boot=n_bot, smooth_sigma=5)
    axs[-1].plot(t_4, median, color = color, lw = 2, label = labels[ii])
    axs[-1].fill_between(t_4, lower, upper, color=color, alpha=0.2)




## pre lever later-than-FI presses
mask_nocp_trial = (exploded_df['count_lever'] < 3) & (exploded_df['bool_cp'] == False) & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
median_noncp, lower_noncp, upper_noncp = bootstrap_ci(all_snippets[mask_nocp_trial], n_boot=50, smooth_sigma=5)

axs[0].plot(t_4[:200], median_noncp[:200], color = color_non_transition, lw = 2, label = labels[-1])
axs[0].fill_between(t_4[:200], lower_noncp[:200], upper_noncp[:200], color=color_non_transition, alpha=0.2)
axs[0].plot(t_4[200:], median_noncp[200:], color = color_non_transition, lw = 2, ls = ':')

axs[-1].plot(t_4[:200], median_noncp[:200], color = color_non_transition, lw = 2, ls = ':')
axs[-1].plot(t_4[200:], median_noncp[200:], color = color_non_transition, lw = 2, label = labels[-1])
axs[-1].fill_between(t_4[200:], lower_noncp[200:], upper_noncp[200:], color=color_non_transition, alpha=0.2)



## ALIGNMENT LEVER PRESSES: INITIAL TO PRE-TERMINAL
for jj, tercile_category in enumerate(['t1', 't2', 't3']):
    mask_cp_T1 = (exploded_df['tercile_cp_FInormalised'] == 'T1') & (exploded_df['lever_index_category'] == tercile_category) & (exploded_df['animal'] == animal)
    mask_cp_T2 = (exploded_df['tercile_cp_FInormalised'] == 'T2') & (exploded_df['lever_index_category'] == tercile_category) & (exploded_df['animal'] == animal)
    mask_cp_T3 = (exploded_df['tercile_cp_FInormalised'] == 'T3') & (exploded_df['lever_index_category'] == tercile_category) & (exploded_df['animal'] == animal)
    masks = [mask_cp_T1,mask_cp_T2, mask_cp_T3]

    for ii, mask in enumerate(masks):
        color = tercile_colors[ii] if ii < 3 else color_non_transition
        median, lower, upper = bootstrap_ci(all_snippets[mask], n_boot=n_bot, smooth_sigma=5)
        axs[1+jj].plot(t_4[100:-100], median[100:-100], color = color, lw = 2, label = labels[ii])
        axs[1+jj].fill_between(t_4[100:-100], lower[100:-100], upper[100:-100], color=color, alpha=0.2)


axs[-1].legend(frameon = False, fontsize = 12)

axs[0].set_xlabel('time since transition point (s)')
axs[2].set_xlabel('time since press (s)')
axs[-1].set_xlabel('time since last press (s)')

[axs[ii].axvline(0,color = 'black', ls = '--', lw = 1) for ii in range(5)]

axs[0].set_ylabel('DA (z ΔF/F)')

figtitle = f'DA | conditioned on transition point | {animal}'

plt.suptitle(figtitle)
plt.savefig(rf'{PATH_SAVE_AGGREGATE_DA_FIGS}\{figtitle.replace("|","_")}.png')
#plt.savefig(rf'{photometry_fig_path}\{figtitle.replace("|","_")}.pdf')

#%%

## same as above but split by experiment

animal = animal2
exp = 'c'

n_bot = 500

t_4 = np.linspace(-2,2,401)
color_non_transition = '#1b6a9e'
labels = ['early', 'intermediate', 'late', 'later-than-FI']

fig, axs = plt.subplots(1,5, tight_layout = True, figsize = (12,4), width_ratios=[3,1,1,1,3], sharey = True)


df = aggregated_jointdf.query(f'animal == "{animal}" and experiment == "{exp}"')# and date in {photometry_dates_dict[animal]}')
_, snippets_cp_T1 = compute_snippets_across_days(df,f'bool_cp and tercile_cp_FInormalised == "T1"', 'cp_abs','DA_zscored_session', (-2,2))
_, snippets_cp_T2 = compute_snippets_across_days(df,f'bool_cp and tercile_cp_FInormalised == "T2"', 'cp_abs','DA_zscored_session', (-2,2))
_, snippets_cp_T3 = compute_snippets_across_days(df,f'bool_cp and tercile_cp_FInormalised == "T3"', 'cp_abs','DA_zscored_session', (-2,2))
_, snippets_cp_afterFI = compute_snippets_across_days(df,f'bool_cp == False and count_lever < 3', 'last_lever_abs','DA_zscored_session', (-2,2))
_, snippets_cp_afterFI = compute_snippets_across_days(df,f'bool_cp == False and count_lever == 1', 'last_lever_abs','DA_zscored_session', (-2,2))

all_cp_snippets = [snippets_cp_T1, snippets_cp_T2, snippets_cp_T3]

for ii, snippets in enumerate(all_cp_snippets):
    color = tercile_colors[ii] if ii < 3 else color_non_transition

    median, lower, upper = bootstrap_ci(snippets, n_boot=n_bot, smooth_sigma=5)
    axs[0].plot(t_4, median, color = color, lw = 2, label = labels[ii])
    axs[0].fill_between(t_4, lower, upper, color=color, alpha=0.2)



mask_cp_T1 = (exploded_df['experiment'] == exp) & (exploded_df['tercile_cp_FInormalised'] == 'T1') & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
mask_cp_T2 = (exploded_df['experiment'] == exp) & (exploded_df['tercile_cp_FInormalised'] == 'T2') & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
mask_cp_T3 = (exploded_df['experiment'] == exp) & (exploded_df['tercile_cp_FInormalised'] == 'T3') & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
masks = [mask_cp_T1,mask_cp_T2, mask_cp_T3]

for ii, mask in enumerate(masks):

    color = tercile_colors[ii] if ii < 3 else color_non_transition
    
    median, lower, upper = bootstrap_ci(all_snippets[mask], n_boot=n_bot, smooth_sigma=5)
    axs[-1].plot(t_4, median, color = color, lw = 2, label = labels[ii])
    axs[-1].fill_between(t_4, lower, upper, color=color, alpha=0.2)




## pre lever later-than-FI presses
mask_nocp_trial = (exploded_df['count_lever'] < 3) & (exploded_df['bool_cp'] == False) & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
median_noncp, lower_noncp, upper_noncp = bootstrap_ci(all_snippets[mask_nocp_trial], n_boot=50, smooth_sigma=5)

axs[0].plot(t_4[:200], median_noncp[:200], color = color_non_transition, lw = 2, label = labels[-1])
axs[0].fill_between(t_4[:200], lower_noncp[:200], upper_noncp[:200], color=color_non_transition, alpha=0.2)
axs[0].plot(t_4[200:], median_noncp[200:], color = color_non_transition, lw = 2, ls = ':')

axs[-1].plot(t_4[:200], median_noncp[:200], color = color_non_transition, lw = 2, ls = ':')
axs[-1].plot(t_4[200:], median_noncp[200:], color = color_non_transition, lw = 2, label = labels[-1])
axs[-1].fill_between(t_4[200:], lower_noncp[200:], upper_noncp[200:], color=color_non_transition, alpha=0.2)



## ALIGNMENT LEVER PRESSES: INITIAL TO PRE-TERMINAL
for jj, tercile_category in enumerate(['t1', 't2', 't3']):
    mask_cp_T1 = (exploded_df['experiment'] == exp) & (exploded_df['tercile_cp_FInormalised'] == 'T1') & (exploded_df['lever_index_category'] == tercile_category) & (exploded_df['animal'] == animal)
    mask_cp_T2 = (exploded_df['experiment'] == exp) & (exploded_df['tercile_cp_FInormalised'] == 'T2') & (exploded_df['lever_index_category'] == tercile_category) & (exploded_df['animal'] == animal)
    mask_cp_T3 = (exploded_df['experiment'] == exp) & (exploded_df['tercile_cp_FInormalised'] == 'T3') & (exploded_df['lever_index_category'] == tercile_category) & (exploded_df['animal'] == animal)
    masks = [mask_cp_T1,mask_cp_T2, mask_cp_T3]

    for ii, mask in enumerate(masks):
        color = tercile_colors[ii] if ii < 3 else color_non_transition
        median, lower, upper = bootstrap_ci(all_snippets[mask], n_boot=n_bot, smooth_sigma=5)
        axs[1+jj].plot(t_4[100:-100], median[100:-100], color = color, lw = 2, label = labels[ii])
        axs[1+jj].fill_between(t_4[100:-100], lower[100:-100], upper[100:-100], color=color, alpha=0.2)


axs[-1].legend(frameon = False, fontsize = 12)

axs[0].set_xlabel('time since transition point (s)')
axs[2].set_xlabel('time since press (s)')
axs[-1].set_xlabel('time since last press (s)')

[axs[ii].axvline(0,color = 'black', ls = '--', lw = 1) for ii in range(5)]

axs[0].set_ylabel('DA (z ΔF/F)')

figtitle = f'DA | conditioned on transition point | {animal} | experiment {exp}'

plt.suptitle(figtitle)
plt.savefig(rf'{PATH_SAVE_AGGREGATE_DA_FIGS}\{figtitle.replace("|","_")}.png')
#plt.savefig(rf'{photometry_fig_path}\{figtitle.replace("|","_")}.pdf')



#%%

"""
....###....##.......##...........######...#######..##....##.########...######.
...##.##...##.......##..........##....##.##.....##.###...##.##.....##.##....##
..##...##..##.......##..........##.......##.....##.####..##.##.....##.##......
.##.....##.##.......##..........##.......##.....##.##.##.##.##.....##..######.
.#########.##.......##..........##.......##.....##.##..####.##.....##.......##
.##.....##.##.......##..........##....##.##.....##.##...###.##.....##.##....##
.##.....##.########.########.....######...#######..##....##.########...######.

not from the thesis nb -- split by conditions: block types within experiment types

"""
animal = animal2

alignment = 'cp_abs'
alignment_label = 'transition point'
#alignment = 'last_lever_abs'
#alignment_label = 'last press'

fig, axs = plt.subplots(3,3, tight_layout = True, figsize = (12,12), sharex = True, sharey = True)

for jj, exp in enumerate(['a','b','c']):
    if exp == 'c':
        variable = 'n_protocols'
        order = rwd_order
    else:
        variable = 'FI'
        order = FI_order

    for kk, var in enumerate(order):
        df = aggregated_jointdf.query(f'animal == "{animal}" and experiment == "{exp}" and {variable} == {var}')
        _, snippets_cp_T1 = compute_snippets_across_days(df,f'bool_cp and tercile_cp_FInormalised == "T1"', alignment,'DA_zscored_session', (-2,2))
        _, snippets_cp_T2 = compute_snippets_across_days(df,f'bool_cp and tercile_cp_FInormalised == "T2"', alignment,'DA_zscored_session', (-2,2))
        _, snippets_cp_T3 = compute_snippets_across_days(df,f'bool_cp and tercile_cp_FInormalised == "T3"', alignment,'DA_zscored_session', (-2,2))
        #_, snippets_cp_afterFI = compute_snippets_across_days(df,f'bool_cp == False and count_lever < 3', 'last_lever_abs','DA_zscored_session', (-2,2))
        #_, snippets_cp_afterFI = compute_snippets_across_days(df,f'bool_cp == False and count_lever == 1', 'last_lever_abs','DA_zscored_session', (-2,2))

        all_cp_snippets = [snippets_cp_T1, snippets_cp_T2, snippets_cp_T3]

        for ii, snippets in enumerate(all_cp_snippets):
            color = tercile_colors[ii] if ii < 3 else color_non_transition

            median, lower, upper = bootstrap_ci(snippets, n_boot=n_bot, smooth_sigma=5)
            axs[kk,jj].plot(t_4, median, color = color, lw = 2, label = labels[ii])
            axs[kk,jj].fill_between(t_4, lower, upper, color=color, alpha=0.2)

        axs[kk,jj].set_title(f'{variable} {var} - {exp}')
        axs[kk,jj].axvline(0,color = 'black', ls = '--', lw = 1)

        axs[kk,0].set_ylabel('DA (z ΔF/F)')

    axs[-1,jj].set_xlabel(f'time since {alignment_label} (s)')


figtitle = f'DA | conditioned on {alignment_label} | {animal} | per block type within experimental condition'

plt.suptitle(figtitle)
plt.savefig(rf'{PATH_SAVE_AGGREGATE_DA_FIGS}\{figtitle.replace("|","_")}.png')
#%%

animal = 'Ruthenium'
exp = 'b'

if exp == "c":
    var = 'n_protocols'
    order = rwd_order
else:
    var = 'FI'
    order = FI_order


fig, axs = plt.subplots(2,3, tight_layout = True, figsize = (12,6), height_ratios = [2,1])#, width_ratios=[1,2,4])

for ii, cond in enumerate(order):

    df = aggregated_jointdf.query(f'animal == "{animal}" and experiment == "{exp}" and {var} == {cond}')
    
    FI = 30 if exp == 'c' else cond
    ## nanify False is not working, halp! -- something weird is going on with the nanify and delta_t; might need to restart stuff
    t_long, snippets_long = compute_snippets_across_days(df,
        'FI > 0', 'last_lever_abs', 'DA_zscored_session', (-2,FI*1.05))

    cps = df.cp.values
    sorted_cps = cps[np.argsort(cps)]
    #next_rwd = aggregated_jointdf.query('bool_cp and animal == "Ruthenium" and experiment == "a" and FI == 30').apply(lambda x: x.cp_abs - x.trial_start, axis = 1).values

    data = snippets_long[np.argsort(cps)][~np.isnan(sorted_cps)]

    #plot_snippets(data, t_long, axs[0,ii], axs[1,ii])

    vmin,vmax = np.nanquantile(data, [.01,.99])
    axs[0,ii].imshow(zscore(data, axis = 1), aspect = 'auto', vmin = vmin, vmax = vmax,
               extent = [t_long[0],t_long[-1],0,len(data)], origin = 'lower', cmap = 'bone')
    axs[0,ii].plot(sorted_cps, np.arange(0,len(cps)), '|', color = 'purple')
    axs[1,ii].plot(t_long, np.nanmean(data, axis = 0), lw = 1)
    #axs[1,ii].plot(t_long, np.nanmean(zscore(data, axis = 1), axis = 0))

    axs[0,ii].set_title(f'{var} {cond}')

axs[0,0].set_ylabel('trials sorted by transition point')
axs[1,0].set_ylabel('DA (z ΔF/F)')

[axs[1,ii].set_xlabel('time since reward delivery (s)') for ii in range(3)]

figtitle = f'{animal} | experiment {exp} | DA within trial sorted by transition point'

plt.suptitle(figtitle)
plt.savefig(rf'{PATH_SAVE_AGGREGATE_DA_FIGS}\{figtitle.replace("|","_")}.png')
#%%

np.argsort(df.cp.values)
#%%
cps
#%%
len(sorted_cps)
#%%

plt.figure(figsize = (8,10))


for ii in range(105):
    tt = np.argsort(cps)[ii]

    #plt.plot(t_long, ii*5+ data[ii], 'k', lw = 1)
    plt.plot(np.arange(0,len(df.DA_zscored_session.values[tt]))/100,5*ii+df.DA_zscored_session.values[tt], 'k', lw = .5)
    plt.plot(cps[tt], 5*ii, 'o', color = 'orange')
    for lvr in df.lever_rel.values[tt]:
        plt.plot(lvr, 5*ii, '.', color = 'teal')

plt.xlim(0,62)
plt.ylim(0,(ii+1)*5)
#plt.title(tt)

#%%
ii = 0
tt = np.argsort(cps)[ii]

plt.plot(data[ii,200:8000],
         df.DA_zscored_session.values[tt][:6000])
#%%
plt.plot(np.arange(0,len(df.DA_zscored_session.values[tt]))/100,df.DA_zscored_session.values[tt], 'k', lw = 1)

plt.plot(cps[tt], 5*ii, '.', color = 'purple')


#%%

"""
..######..########..##.......####.########....########..##....##....########.##.....##.########..########.########..####.##.....##.########.##....##.########....###....##...........######...#######..##....##.########..####.########.####..#######..##....##..######.
.##....##.##.....##.##........##.....##.......##.....##..##..##.....##........##...##..##.....##.##.......##.....##..##..###...###.##.......###...##....##......##.##...##..........##....##.##.....##.###...##.##.....##..##.....##.....##..##.....##.###...##.##....##
.##.......##.....##.##........##.....##.......##.....##...####......##.........##.##...##.....##.##.......##.....##..##..####.####.##.......####..##....##.....##...##..##..........##.......##.....##.####..##.##.....##..##.....##.....##..##.....##.####..##.##......
..######..########..##........##.....##.......########.....##.......######......###....########..######...########...##..##.###.##.######...##.##.##....##....##.....##.##..........##.......##.....##.##.##.##.##.....##..##.....##.....##..##.....##.##.##.##..######.
.......##.##........##........##.....##.......##.....##....##.......##.........##.##...##........##.......##...##....##..##.....##.##.......##..####....##....#########.##..........##.......##.....##.##..####.##.....##..##.....##.....##..##.....##.##..####.......##
.##....##.##........##........##.....##.......##.....##....##.......##........##...##..##........##.......##....##...##..##.....##.##.......##...###....##....##.....##.##..........##....##.##.....##.##...###.##.....##..##.....##.....##..##.....##.##...###.##....##
..######..##........########.####....##.......########.....##.......########.##.....##.##........########.##.....##.####.##.....##.########.##....##....##....##.....##.########.....######...#######..##....##.########..####....##....####..#######..##....##..######.
"""

animal = animal2

t_4 = np.linspace(-2,2,401)


fig, axs = plt.subplots(2,3, tight_layout = True, figsize = (12,6), sharey = 'row')

for ii,FI in enumerate(FI_order):
    cond_color = color_FI_blocks[ii]

    ## a
    cond_mask = (exploded_df['experiment'] == 'a') & (exploded_df['FI'] == FI) & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
    median, lower, upper = bootstrap_ci(all_snippets[cond_mask], n_boot=500, smooth_sigma=5)
    axs[0,0].plot(t_4, median, color = cond_color, lw = 2)
    axs[0,0].fill_between(t_4, lower, upper, color=cond_color, alpha=0.2)

    cond_mask = (exploded_df['experiment'] == 'a') & (exploded_df['FI'] == FI) & (exploded_df['lever_index'] != -1) & (exploded_df['animal'] == animal)
    median, lower, upper = bootstrap_ci(all_snippets[cond_mask], n_boot=500, smooth_sigma=5)
    axs[1,0].plot(t_4, median, color = cond_color, lw = 2)
    axs[1,0].fill_between(t_4, lower, upper, color=cond_color, alpha=0.2)

    ## b
    cond_mask = (exploded_df['experiment'] == 'b') & (exploded_df['FI'] == FI) & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
    median, lower, upper = bootstrap_ci(all_snippets[cond_mask], n_boot=500, smooth_sigma=5)
    axs[0,1].plot(t_4, median, color = cond_color, lw = 2)
    axs[0,1].fill_between(t_4, lower, upper, color=cond_color, alpha=0.2)

    cond_mask = (exploded_df['experiment'] == 'b') & (exploded_df['FI'] == FI) & (exploded_df['lever_index'] != -1) & (exploded_df['animal'] == animal)
    median, lower, upper = bootstrap_ci(all_snippets[cond_mask], n_boot=500, smooth_sigma=5)
    axs[1,1].plot(t_4, median, color = cond_color, lw = 2)
    axs[1,1].fill_between(t_4, lower, upper, color=cond_color, alpha=0.2)

for ii, rwd in enumerate(rwd_order): ## c
    cond_color = color_rwd_blocks[ii]

    cond_mask = (exploded_df['experiment'] == 'c') & (exploded_df['n_protocols'] == rwd) & (exploded_df['lever_index'] == -1) & (exploded_df['animal'] == animal)
    median, lower, upper = bootstrap_ci(all_snippets[cond_mask], n_boot=500, smooth_sigma=5)
    axs[0,2].plot(t_4, median, color = cond_color, lw = 2)
    axs[0,2].fill_between(t_4, lower, upper, color=cond_color, alpha=0.2)

    cond_mask = (exploded_df['experiment'] == 'c') & (exploded_df['n_protocols'] == rwd) & (exploded_df['lever_index'] != -1) & (exploded_df['animal'] == animal)
    median, lower, upper = bootstrap_ci(all_snippets[cond_mask], n_boot=500, smooth_sigma=5)
    axs[1,2].plot(t_4, median, color = cond_color, lw = 2)
    axs[1,2].fill_between(t_4, lower, upper, color=cond_color, alpha=0.2)

for kk in range(2):
    axs[kk,0].set_ylabel('DA (z ΔF/F)')

    for jj in range(3):    
        axs[kk,jj].axvline(0,color = 'black', ls = '--', lw = 1)

for jj in range(3):
    axs[0,jj].set_xlabel(f'time since last lever press (s)')
    axs[1,jj].set_xlabel(f'time since unrewarded lever press (s)')


axs[0,0].set_title('varying FI, fixed rwd')
axs[0,1].set_title('fixed rwd rate')
axs[0,2].set_title('FI30, varying rwd')


figtitle = f'DA | {animal} | around lever presses | split per experimental condition'

plt.suptitle(figtitle)
plt.savefig(rf'{PATH_SAVE_AGGREGATE_DA_FIGS}\{figtitle.replace("|","_")}.png')

#%%

#%%
## this in principle should be the same as above; and it is!! I had a bug above in which I was filtering the exploded_df to account only for exp c
"""
.##....##.########.##......##.....######..##....##.####.########..########..########.########..######.
.###...##.##.......##..##..##....##....##.###...##..##..##.....##.##.....##.##..........##....##....##
.####..##.##.......##..##..##....##.......####..##..##..##.....##.##.....##.##..........##....##......
.##.##.##.######...##..##..##.....######..##.##.##..##..########..########..######......##.....######.
.##..####.##.......##..##..##..........##.##..####..##..##........##........##..........##..........##
.##...###.##.......##..##..##....##....##.##...###..##..##........##........##..........##....##....##
.##....##.########..###..###......######..##....##.####.##........##........########....##.....######.
"""
# Define your columns
colname = 'DA_session_zscored'
eventalignment = 'nonrwd_lever_abs'
## ['cp_abs','nonrwd_lever_abs','rwd_lever_abs']

# Storage for snippets and metadata arrays
all_snipps = []
meta = {
    'animal': [], 'date': [], 'experiment': [], 'FI': [], 'n_protocols': []
}

# Group by all 5 metadata factors
meta_cols = ['animal', 'date', 'experiment', 'FI', 'n_protocols']  # Adjust 'n_protocols' if your column name is different

for (ani, date, exp, fi, rwd), daydf in aggregated_jointdf.groupby(meta_cols):
    
    ts = np.hstack(daydf.timestamp_session.values)
    sig = np.hstack(daydf[colname].values)
    
    evs = np.hstack([np.atleast_1d(x) for x in daydf[eventalignment].values])
    
    snipps, time = signal2eventsnippets(ts, sig, evs, [-4, 4], 0.01)
    
    num_snipps = snipps.shape[0]
    if num_snipps > 0:
        all_snipps.append(snipps)
        
        meta['animal'].append(np.full(num_snipps, ani))
        meta['date'].append(np.full(num_snipps, date))
        meta['experiment'].append(np.full(num_snipps, exp))
        meta['FI'].append(np.full(num_snipps, fi))
        meta['n_protocols'].append(np.full(num_snipps, rwd))

snipps_matrix = np.vstack(all_snipps)
animal_idx = np.concatenate(meta['animal'])
date_idx = np.concatenate(meta['date'])
exp_idx = np.concatenate(meta['experiment'])
FI_idx = np.concatenate(meta['FI'])
rwd_idx = np.concatenate(meta['n_protocols'])

snipps_matrix = drop_nans_matrix(snipps_matrix)

#%%

nonrwd_snipps_Pa = snipps_matrix
#rwd_snipps_Pa = snipps_matrix
#%%

## not sure where exp_list is defined, but it's part of the globals somewhere

fig, axs = plt.subplots(1,3, figsize = (12,4), tight_layout = True, sharey = True)

for ii, exp in enumerate(exp_list):
    if exp != 'c':
        for jj, FI in enumerate(FI_order):
            mask = (animal_idx == animal) & (exp_idx == exp) & (FI_idx == FI)
            subset = snipps_matrix[mask]

            axs[ii].plot(time, np.nanmean(subset, axis = 0), color = color_FI_blocks[jj])
    else:
        for jj, rwd in enumerate(rwd_order):
            mask = (animal_idx == animal) & (exp_idx == exp) & (rwd_idx == rwd)
            subset = snipps_matrix[mask]

            axs[ii].plot(time, np.nanmean(subset, axis = 0), color = color_rwd_blocks[jj])

    axs[ii].axvline(0, ls = '--', color = 'grey')
    axs[ii].set_title(f'experiment {exp}')
    axs[ii].set_xlabel('time since last lever press (s)')

axs[0].set_ylabel('av. DA across sessions (dF/F)')

figtitle = f'{animal} | all sessions aggregated | channel {colname} aligned to {eventalignment}'
plt.suptitle(figtitle)

plt.savefig(rf'{PATH_SAVE_DFS}\{figtitle.replace("|","_")}.png', dpi = 300)
#%%

fig, axs = plt.subplots(1,2, tight_layout = True, figsize = (8,4), sharey = True)

axs[0].plot(time, np.nanmean(nonrwd_snipps_Ru, axis = 0))
axs[0].plot(time, np.nanmean(rwd_snipps_Ru, axis = 0))

axs[1].plot(time, np.nanmean(nonrwd_snipps_Pa, axis = 0))
axs[1].plot(time, np.nanmean(rwd_snipps_Pa, axis = 0))

for ii in range(2):
    axs[ii].axvline(0, ls = '--', color = 'grey')
#%%
from matplotlib import cm

fig, axs = plt.subplots(3,2, tight_layout = True, figsize = (8,8), sharex = True)

lvr_rwd_labels = ['nonrwd', 'rwd']
bone_cmap = cm.get_cmap('bone')
lvr_rwd_colors = [bone_cmap(0.7), bone_cmap(0.2)]

for ii,data in enumerate([nonrwd_snipps_Ru, rwd_snipps_Ru, nonrwd_snipps_Pa, rwd_snipps_Pa]):

    data_z = zscore(data, axis = 1)
    vmin = np.nanquantile(data_z, .05)
    vmax = np.nanquantile(data_z, .95)
    axs[ii%2,int(ii/2)].imshow(data_z, aspect = 'auto', vmin = vmin, vmax = vmax, cmap = 'bone', origin = 'lower',
               extent = [time[0], time[-1], 0, data.shape[0]])

    axs[-1,int(ii/2)].plot(time, np.nanmean(data, axis = 0), label = lvr_rwd_labels[ii%2], color = lvr_rwd_colors[ii%2])

for ii in range(2):   
    axs[-1,ii].axvline(0, ls = '--', color = 'grey')
    axs[-1,ii].set_xlabel('time since lever press (s)')

axs[-1,0].legend(frameon = False)

axs[0,0].set_title('Ruthenium')
axs[0,1].set_title('Palladium')

axs[0,0].set_ylabel('non-rewarded')
axs[1,0].set_ylabel('rewarded')
axs[2,0].set_ylabel('av. DA on lever presses')

figtitle = 'DA aligned to lever presses | rewarded vs unrewarded'
plt.suptitle(figtitle)

plt.savefig(rf'{PATH_SAVE_DFS}\{figtitle.replace("|","_")}.png', dpi = 300)

#axs[1,1].imshow(rwd_snipps_Ru, aspect = 'auto', vmin = np.nanquantile(rwd_snipps_Ru, .05), vmax = np.nanquantile(rwd_snipps_Ru, .95), cmap = 'bone', origin = 'lower',
#           extent = [time[0], time[-1], 0, rwd_snipps_Ru.shape[0]])

#%%

"""
..#######..##.......########.....########.########...#######..##.....##....########.....###....####.##.......##....##
.##.....##.##.......##.....##....##.......##.....##.##.....##.###...###....##.....##...##.##....##..##........##..##.
.##.....##.##.......##.....##....##.......##.....##.##.....##.####.####....##.....##..##...##...##..##.........####..
.##.....##.##.......##.....##....######...########..##.....##.##.###.##....##.....##.##.....##..##..##..........##...
.##.....##.##.......##.....##....##.......##...##...##.....##.##.....##....##.....##.#########..##..##..........##...
.##.....##.##.......##.....##....##.......##....##..##.....##.##.....##....##.....##.##.....##..##..##..........##...
..#######..########.########.....##.......##.....##..#######..##.....##....########..##.....##.####.########....##...
"""

PATH_SAVE_FIGS = os.path.join(DROPBOX_TASK_PATH, 'analysis_photometry', f'{animal}_{date}')
if not os.path.exists(PATH_SAVE_FIGS):
    os.makedirs(PATH_SAVE_FIGS)

#PATH_SAVE_ICA = os.path.join(DROPBOX_TASK_PATH, 'analysis_photometry', '00_all_sessions_ICA_snippets')
#%%

#%%

"""
.########.##.....##.########....########.####..######..
....##....##.....##.##..........##........##..##....##.
....##....##.....##.##..........##........##..##.......
....##....#########.######......######....##..##...####
....##....##.....##.##..........##........##..##....##.
....##....##.....##.##..........##........##..##....##.
....##....##.....##.########....##.......####..######..
"""

fig, axs = plt.subplots(5,3, tight_layout = True, figsize = (12,12), sharex = True)

colors_colname = ['grey','red', 'green', 'purple']

for ii,eventalignment in enumerate(['cp_abs','nonrwd_lever_abs','rwd_lever_abs']):
    for jj,colname in enumerate(['ds_continuous_encoder','deltaF_poly_tdtomato', 'deltaF_poly_gfp', 'DA_poly_session']):

        snipps, time = signal2eventsnippets(downharpdf.timestamp_session,
                                        downharpdf[colname],
                                        np.hstack(jointdf[eventalignment].values),
                                        [-4,4], .01)
        
        snipps = drop_nans_matrix(snipps)

        if jj == 0:
            snipps = zscore(snipps, axis = 1)
            #snipps - snipps[:, 0][:, np.newaxis]
            
        
        vmin,vmax = np.nanquantile(snipps, [.05,.95])
        axs[jj,ii].imshow(snipps, aspect = 'auto', vmin = vmin, vmax = vmax, cmap = 'bone', origin = 'lower',
                          extent = [time[0],time[-1],0,len(snipps)])

        snipps_mean = np.nanmean(snipps, axis = 0)
        if jj != 0:
            axs[-1,ii].plot(time, snipps_mean, color = colors_colname[jj])
        else:
            axs_encoder = axs[-1,ii].twinx()
            axs_encoder.plot(time, snipps_mean, color = colors_colname[jj])

        axs[jj,0].set_ylabel(colname, color = colors_colname[jj])

    axs[-1,ii].set_xlabel(f't since {eventalignment} (s)')
    axs[-1,ii].axvline(0, ls = '--', color = 'grey')

figtitle = f'{animal} {date} | experiment {determine_experiment(jointdf)} | channel traces aligned to events'

fig.suptitle(figtitle)

plt.savefig(rf'{PATH_SAVE_FIGS}\{figtitle.replace('|','_')}.png', dpi = 300)

#%%

"""
.########.##.....##.########.
.##........##...##..##.....##
.##.........##.##...##.....##
.######......###....########.
.##.........##.##...##.......
.##........##...##..##.......
.########.##.....##.##.......
""" 

exp = determine_experiment(jointdf)

fig, axs = plt.subplots(1,3, tight_layout = True, sharey = True, figsize = (12,4))

if exp == 'c': ## untested
    variable = 'n_protocols'
    colorcode = color_rwd_blocks
    variable_list = rwd_order

else:
    variable = 'FI'
    colorcode = color_FI_blocks
    variable_list = FI_order


for ii,eventalignment in enumerate(['cp_abs','nonrwd_lever_abs','rwd_lever_abs']):

    for jj, variable_value in enumerate(variable_list):
        snipps, time = signal2eventsnippets(downharpdf.timestamp_session,
                                        downharpdf['DA_poly_session'],
                                        np.hstack(jointdf.query(f'{variable} == {variable_value}')[eventalignment].values),
                                        [-4,4], .01)
    
        axs[ii].plot(time, np.nanmean(snipps, axis = 0), color = colorcode[jj])
    
    axs[ii].set_xlabel(f't since {eventalignment} (s)')
    axs[ii].axvline(0, ls = '--', color = 'grey')

axs[0].set_ylabel('DA_poly_session')

figtitle = f'{animal} {date} | experiment {exp} | averages split by block condition'

fig.suptitle(figtitle)

plt.savefig(rf'{PATH_SAVE_FIGS}\{figtitle.replace('|','_')}.png', dpi = 300)

#%%




#%%

"""
.##.....##.########.########..########
.##.....##.##.......##.....##.##......
.##.....##.##.......##.....##.##......
.#########.######...########..######..
.##.....##.##.......##...##...##......
.##.....##.##.......##....##..##......
.##.....##.########.##.....##.########

might be trash
"""

"""
tt = 22
fig, axs = plt.subplots(5, tight_layout = True, figsize = (8,6))

tomato = np.hstack(jointdf.tdtomato_poly_flat[tt])
gfp = np.hstack(jointdf.gfp_poly_flat[tt])
encoder = np.hstack(jointdf.ds_continuous_encoder[tt])
time = np.arange(0,len(tomato))/100

DA_q = gfp - quantile_regression(tomato, gfp, .5)

DA_q_norm = DA_q/(quantile_regression(tomato, gfp,.95) - quantile_regression(tomato, gfp,.05))

axs[0].plot(time, tomato, color = 'red', lw = 1)
axs[0].plot(time, gfp, color = 'green', lw = 1)

axs_encoder = axs[0].twinx()
axs_encoder.plot(time, encoder, color = 'blue', lw = 1, alpha = 0.5)

axs[1].plot(time, DA_q, color = 'purple', lw = 1)

axs[2].plot(time, DA_q_norm, color = 'teal', lw = 1)


from sklearn.decomposition import FastICA, NMF

X = np.column_stack([tomato, gfp])
ica = FastICA(n_components = 2, random_state = 42)
signals_recovered = ica.fit_transform(X)

axs[3].plot(time, signals_recovered[:,0]/5, lw = 1)
axs[3].plot(time, signals_recovered[:,1]/5, lw = 1)

X = X - np.min(X) if np.min(X) < 0 else X
nmf = NMF(n_components=2, init='nndsvda', random_state=42, max_iter=1000)
W = nmf.fit_transform(X)
#H = nmf.components_
axs[4].plot(W[:,0], lw = 1)
axs[4].plot(W[:,1], lw = 1)


axs[0].set_ylabel('raw')
axs[1].set_ylabel('regression')
axs[2].set_ylabel('q-normalized')
axs[3].set_ylabel('ICA')
axs[4].set_ylabel('NMF')

figtitle = f'{animal} {date} | experiment {determine_experiment(bhvdf)} | trial {tt}'
fig.suptitle(figtitle)
#%%
plt.plot(time, encoder)
plt.plot(time[1:], np.diff(encoder))
#%%


#%%
plt.plot(tomato, gfp, '.')

#%%

plt.plot(np.nanmean(snipps_DA_rwd, axis = 0), color = 'purple', lw = 1)

#%%
tt = 36
plt.plot(zscore(np.hstack(jointdf.query(f'trialno == {tt}').tdtomato_poly_flat)), color = 'red', lw = 1)
plt.plot(1+zscore(np.hstack(jointdf.query(f'trialno == {tt}').gfp_poly_flat)), color = 'green', lw = 1)

plt.plot(10+zscore(np.hstack(jointdf.query(f'trialno == {tt}').DA_poly_session)), color = 'purple', lw = 1)
#%%

from scipy.signal import decimate, butter, filtfilt

tdtomato = decimate(harpdf.tdtomato.values, q=10)
gfp = decimate(harpdf.gfp.values, q=10)

DA = gfp - get_prediction(tdtomato, gfp)

#%%

ss = 35000
ee = ss + 6000

fig, axs = plt.subplots(2, figsize = (6,4))

axs[0].plot(tdtomato[ss:ee], lw = 1)
axs[0].plot(gfp[ss:ee], lw = 1)
axs[0].plot(-1+DA[ss:ee], color = 'purple', lw = 1)
axs[0].plot(gfp[ss:ee] - get_prediction(tdtomato[ss:ee], gfp[ss:ee]), color = 'orange', lw = 1)

axs[1].plot(downharpdf.ds_tdtomato[ss:ee])
axs[1].plot(downharpdf.ds_gfp[ss:ee])
axs[1].plot(-1+downharpdf.DA_poly_session[ss:ee], color = 'purple', lw = 1)
axs[1].plot(downharpdf.ds_gfp[ss:ee] - get_prediction(downharpdf.ds_tdtomato[ss:ee], downharpdf.ds_gfp[ss:ee]), color = 'orange', lw = 1)
#%%



plt.plot(DA[ss:ee])
plt.plot(downharpdf.DA_poly_session[ss:ee].values)
#%%


#%%
print(calculate_snr(tdtomato))
print(calculate_snr(gfp))
#%%
print(calculate_snr(downharpdf.tdtomato))
print(calculate_snr(downharpdf.gfp))




#%%
fig, axs = plt.subplots(4,3, tight_layout = True, figsize = (12,10), sharey = 'row', sharex = True)

#axs[0,0].plot(time, np.nanmean(snipps_0_cp, axis = 0), color = 'blue', lw = 1)
#axs[0,0].plot(time, np.nanmean(snipps_1_cp, axis = 0), color = 'orange', lw = 1)
#
#axs[0,1].plot(time, np.nanmean(snipps_0_nonrwd, axis = 0), color = 'blue', lw = 1)
#axs[0,1].plot(time, np.nanmean(snipps_1_nonrwd, axis = 0), color = 'orange', lw = 1)
#
#axs[0,2].plot(time, np.nanmean(snipps_0_rwd, axis = 0), color = 'blue', lw = 1)
#axs[0,2].plot(time, np.nanmean(snipps_1_rwd, axis = 0), color = 'orange', lw = 1)
#
## constrainted ICA
#axs[1,0].plot(time, np.nanmean(snipps_cICA_dlight_cp, axis = 0), color = 'blue', lw = 1)
#axs[1,0].plot(time, np.nanmean(snipps_cICA_motion_cp, axis = 0), color = 'orange', lw = 1)
#
#axs[1,1].plot(time, np.nanmean(snipps_cICA_dlight_nonrwd, axis = 0), color = 'blue', lw = 1)
#axs[1,1].plot(time, np.nanmean(snipps_cICA_motion_nonrwd, axis = 0), color = 'orange', lw = 1)
#
#axs[1,2].plot(time, np.nanmean(snipps_cICA_dlight_rwd, axis = 0), color = 'blue', lw = 1, label = 'DA')
#axs[1,2].plot(time, np.nanmean(snipps_cICA_motion_rwd, axis = 0), color = 'orange', lw = 1, label = 'motion')
#axs[1,2].legend(frameon = False)

## regression DA for comparison
axs[2,0].plot(time, np.nanmean(snipps_DA_cp, axis = 0), color = 'purple', lw = 1)
axs[2,1].plot(time, np.nanmean(snipps_DA_nonrwd, axis = 0), color = 'purple', lw = 1)
axs[2,2].plot(time, np.nanmean(snipps_DA_rwd, axis = 0), color = 'purple', lw = 1)

## NMF
#axs[3,0].plot(time, np.nanmean(snipps_NMF_cp, axis = 0), color = 'blue', lw = 1)
#axs[3,1].plot(time, np.nanmean(snipps_NMF_nonrwd, axis = 0), color = 'blue', lw = 1)
#axs[3,2].plot(time, np.nanmean(snipps_NMF_rwd, axis = 0), color = 'blue', lw = 1, label = 'DA')
#axs[3,0].plot(time, np.nanmean(snipps_NMFmotion_cp, axis = 0), color = 'orange', lw = 1)
#axs[3,1].plot(time, np.nanmean(snipps_NMFmotion_nonrwd, axis = 0), color = 'orange', lw = 1)
#axs[3,2].plot(time, np.nanmean(snipps_NMFmotion_rwd, axis = 0), color = 'orange', lw = 1, label = 'motion')
#axs[3,2].legend(frameon = False)

for ii in range(3):
    for jj in range(4):
        axs[jj,ii].axvline(0, color = 'grey', lw = 0.5, ls = '--')

axs[-1,0].set_xlabel('time since transition (s)')
axs[-1,1].set_xlabel('time since non rwd press (s)')
axs[-1,2].set_xlabel('time since rwd press (s)')

axs[0,0].set_ylabel('ICA')
axs[1,0].set_ylabel('cICA')
axs[2,0].set_ylabel('regression DA')
axs[3,0].set_ylabel('NMF')

figtitle = f"{animal} {date} | photometry ICA snippets around events"
fig.suptitle(figtitle)

#%%
fig.savefig(rf'{PATH_SAVE_ICA}\{figtitle.replace('|','_')}.png', dpi = 300)


jointdf.to_pickle(rf'{PATH_SAVE_ICA}\jointdf_{animal}_{date}_photometry_ICA.pkl')
# %%

plt.imshow(snipps_DA_rwd, aspect = 'auto')



# %%
jointdf.keys()
# %%
"""