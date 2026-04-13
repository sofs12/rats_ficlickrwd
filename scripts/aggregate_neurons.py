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
from ratcode.ephys.neurons import get_psths_across_cells, align_spikes_to_ttl, compute_FR, load_ibl_sorter, determine_cell_type, produce_neuron_fig, produce_mega_neuron_fig, compute_zscore, get_psths_smooth, half_gaussian_kernel, do_PCA, get_PCA_windows, plot_raster
from ratcode.common.dataframe import group_and_listify
from ratcode.common.time import convert_date_bonsai, convert_timestamp
from ratcode.common.math import drop_nans_matrix
from ratcode.common.colorcodes import FI_order, color_FI_blocks, rwd_order, color_rwd_blocks
from ratcode.common.plotting import remove_legend

from ratcode.init import setup


setup()
#%%
def waveform_polarization_area(waveform):
    """
    Determine spike waveform polarization using area under the curve:
    - Integrates above and below zero (baseline)
    - Returns 'neg', 'pos', or 'balanced'
    """
    area_above = np.trapz(waveform[waveform > 0])
    area_below = np.trapz(waveform[waveform < 0])

    if abs(area_below) > abs(area_above):
        return 'neg'
    elif abs(area_above) > abs(area_below):
        return 'pos'
    else:
        return 'balanced'
#%%

## goal here is to aggregate all neurons from Ruthenium and Palladium; check how I used to do this in the past
## can be per animal and then join. check the currently_aggregating folder

## ok so in the past I did this per animal, so let's keep that motif

#%%
animal = 'Ruthenium'
#%%

EPHYS_PATH = os.path.join(DROPBOX_TASK_PATH, 'ephys', animal)
#dropbox_neuro_path = rf'{dropbox_path}\ephys\{animal}'

PATH_ANALYSIS_EPHYS = os.path.join(DROPBOX_TASK_PATH, 'analysis_ephys')
#path_analysis_ephys = r'D:\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\Data\FIClickRwd\analysis_ephys'
#%%

dates = []
for session in glob.glob(fr'{EPHYS_PATH}\{animal}*'):
    session_title = session.split('\\')[-1].split('_')[0]

    dates.append(re.search(r'\d+', session_title).group())
#%%
len(dates)
#%%

dates_Ruthenium = ['260219',
 '260220',
 #'260223', ## bad bhv - viv watered them
 '260224',
 '260225',
 '260226',
 '260227',
 '260228',
 '260303',
 '260304',
 '260305',
 '260306',
 '260308',
 '260310',
 '260311',
 '260312',
 '260318',
 '260319',
 '260320',
 '260323',
 '260324',
 '260325',
 '260327',
 '260330']

dates_Palladium = ['260218',
 '260219',
 '260220',
 #'260223', ## bad bhv - viv watered them
 '260224',
 '260225',
 '260226',
 '260227',
 '260302',
 '260303',
 '260304',
 '260306',
 '260308',
 '260309',
 '260310',
 '260311',
 '260312',
 '260318',
 #'260319',
 '260319', ## session 260319b is usable! 36mins with DA; but for now let's leave it out
 #'260320',
 #'260320',
 '260323',
 #'260324' ## only two blocks
 '260327',
 '260330']

#%%

if animal == 'Ruthenium':
    dates = dates_Ruthenium
elif animal == 'Palladium':
    dates = dates_Palladium
else:
    dates = []
#%%
animalneurondf = pd.DataFrame(columns = ['date','animal','animaldate','experiment','sorted_data', 'good_clusters', 'ok_clusters'])

animalneurondf['date'] = dates
animalneurondf['animal'] = animal

animalneurondf['animaldate'] = animalneurondf.apply(lambda x: f'{x.animal}_{x.date}', axis = 1)
#%%

for ii in range(len(animalneurondf)): ## ready to run
    animal = animalneurondf.loc[ii].animal
    date = animalneurondf.loc[ii].date
    print(f'{animal} {date}')

    SAVE_SYNC_PATH = glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\*')[0]
    neuronsdf = pd.read_pickle(fr'{SAVE_SYNC_PATH}\neuronsdf.pkl')

    DATACLASS_PATH = rf"{DROPBOX_TASK_PATH}\analysis_ephys\{animal}_{date}_sorted_data.pkl"
    with open(DATACLASS_PATH, "rb") as f:
        sorted_data = pickle.load(f)

    animalneurondf.loc[ii,'experiment'] = sorted_data.exp
    animalneurondf.loc[ii,'sorted_data'] = sorted_data

    #possible_ibl_paths = glob.glob(fr"D:\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\Data\FIClickRwd\ephys\{animal}\{animal}{date}*\{animal}{date}*\ibl_sorter_results*")
    #if len(possible_ibl_paths) == 1:
    #    SFpath = rf'{possible_ibl_paths[0]}\cluster_SF.tsv'
    #else:
    #    SFpath = rf'{possible_ibl_paths[1]}\cluster_SF.tsv'

    #SFdf = pd.read_csv(SFpath, sep = '\t')

    animalneurondf.at[ii,'good_clusters'] = neuronsdf.query('SF == "good"').cluster_id.values.tolist()
    animalneurondf.at[ii,'ok_clusters'] = neuronsdf.query('SF == "ok"').cluster_id.values.tolist()
#%%
animalneurondf
#%%
animalneurondf.to_pickle(rf"{PATH_ANALYSIS_EPHYS}\{animal}_animalneurondf.pkl")

## ran until here for both animals -- updated only Palladium (24 March)
#%%


"""
.########.########...#######..##.....##.....######..##.......########....###....##....##.......###.....######....######...########..########..######......###....########.########......###..#######..##.......########......######...#######..########..########.###..
.##.......##.....##.##.....##.###...###....##....##.##.......##.........##.##...###...##......##.##...##....##..##....##..##.....##.##.......##....##....##.##......##....##...........##...##.....##.##.......##.....##....##....##.##.....##.##.....##.##.........##.
.##.......##.....##.##.....##.####.####....##.......##.......##........##...##..####..##.....##...##..##........##........##.....##.##.......##.........##...##.....##....##..........##....##.....##.##.......##.....##....##.......##.....##.##.....##.##..........##
.######...########..##.....##.##.###.##....##.......##.......######...##.....##.##.##.##....##.....##.##...####.##...####.########..######...##...####.##.....##....##....######......##....##.....##.##.......##.....##....##.......##.....##.##.....##.######......##
.##.......##...##...##.....##.##.....##....##.......##.......##.......#########.##..####....#########.##....##..##....##..##...##...##.......##....##..#########....##....##..........##....##.....##.##.......##.....##....##.......##.....##.##.....##.##..........##
.##.......##....##..##.....##.##.....##....##....##.##.......##.......##.....##.##...###....##.....##.##....##..##....##..##....##..##.......##....##..##.....##....##....##...........##...##.....##.##.......##.....##....##....##.##.....##.##.....##.##.........##.
.##.......##.....##..#######..##.....##.....######..########.########.##.....##.##....##....##.....##..######....######...##.....##.########..######...##.....##....##....########......###..#######..########.########......######...#######..########..########.###..
"""

neuronsdf_list = []

for date in dates: ## and then we need to check the bool_ibl_drift
    print(date)

    bhv_pkl = glob.glob(rf"{DROPBOX_TASK_PATH}\analysis\{animal}_{date}_*.pkl")[0]
    bhvdf = pd.read_pickle(bhv_pkl)

    EPHYS_PATH = os.path.join(DROPBOX_TASK_PATH, 'ephys', animal)
    SAVE_SYNC_PATH = glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\*')[0]
    #IBL_SORTER_PATH =  glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\{animal}{date}*\ibl_sorter_results_drift_amplitude')[0]
    neuronsdf = pd.read_pickle(fr'{SAVE_SYNC_PATH}\neuronsdf.pkl')

    #fig_save_path, save_sync_path, ibl_sorter_path, neuro_path = define_all_paths(animal,date,
    #    bool_ibl_drift=False, bool_raw_ephys=False)

    #neuronsdf['date'] = date

    #spikes_self_aligned_all = neuronsdf.spikes_self_aligned.values
    
    #make sure the neuronsdf have the current version of SF labels
    #cluster_info = pd.read_csv(rf'{IBL_SORTER_PATH}\cluster_info.tsv', sep = '\t')
    #neuronsdf['SF'] = cluster_info.SF

    ## keep only the good or ok neurons
    neuronsdf = neuronsdf.query('SF == "good" or SF == "ok"')

    ## to determine cell type features
    #exp = determine_experiment(bhvdf)
    #start_time = time.time()
    #sorted_data = load_ibl_sorter(IBL_SORTER_PATH, animal, date, exp)
    #print(f"load_ibl_sorter took {time.time() - start_time:.2f} seconds")
    #syncdf = pd.read_pickle(fr'{SAVE_SYNC_PATH}\syncdf.pkl')

    #trough_to_peak = []
    #interspike_ratio = []
    #spike_suppression = []
    #waveforms = []
    #for cluster_id in tqdm(neuronsdf.cluster_id.values, desc=f"Extracting features {date}"):
    #    try:        
    #        trough_to_peak_ms, long_interspike_ratio, post_spike_suppression_ms, mean_waveform = extract_features_cell_type(cluster_id, sorted_data, syncdf)
    #    except Exception as e:
    #        print(f'error in cluster_id {cluster_id}')
    #        trough_to_peak_ms = np.nan
    #        long_interspike_ratio = np.nan
    #        post_spike_suppression_ms = np.nan
    #        mean_waveform = np.nan
    #    
    #    trough_to_peak.append(trough_to_peak_ms)
    #    interspike_ratio.append(long_interspike_ratio)
    #    spike_suppression.append(post_spike_suppression_ms)
    #    waveforms.append(mean_waveform)
    #
    #neuronsdf['trough_to_peak_ms'] = trough_to_peak
    #neuronsdf['long_interspike_ratio'] = interspike_ratio
    #neuronsdf['post_spike_suppression_ms'] = spike_suppression
    #neuronsdf['mean_waveform'] = waveforms

    neuronsdf_list.append(neuronsdf)

#%%
aggregated_neuronsdf = pd.concat(neuronsdf_list, ignore_index=True)
#%%

aggregated_neuronsdf.keys()
#%%
aggregated_neuronsdf['cell_polarization'] = aggregated_neuronsdf.mean_waveform.apply(lambda x: waveform_polarization_area(x) if type(x) != float else np.nan)
#aggregated_neuronsdf['cell_type'] = aggregated_neuronsdf.apply(lambda x: classify_cell_type_with_features(x.trough_to_peak_ms, x.long_interspike_ratio, x.post_spike_suppression_ms), axis = 1)
#%%
#aggregated_neuronsdf['animal'] = animal
aggregated_neuronsdf['date_cluster_id'] = aggregated_neuronsdf.apply(lambda x: f'{x.date}_{x.cluster_id}', axis = 1)
#%%
#temp_path = rf'D:\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\Data\FIClickRwd\analysis_ephys\currently_aggregating'

aggregated_neuronsdf.to_pickle(rf'{PATH_ANALYSIS_EPHYS}\{animal}_aggregated_neuronsdf.pkl')
#%%

"""
..######..##....##.##....##..######..########..########..######.
.##....##..##..##..###...##.##....##.##.....##.##.......##....##
.##.........####...####..##.##.......##.....##.##.......##......
..######.....##....##.##.##.##.......##.....##.######....######.
.......##....##....##..####.##.......##.....##.##.............##
.##....##....##....##...###.##....##.##.....##.##.......##....##
..######.....##....##....##..######..########..##........######.

aggregate syncdfs -- bridge clock info between bhv and npx

"""


syncdf_list = []

for date in dates:
    print(date)

    #bhv_pkl = glob.glob(rf"{DROPBOX_TASK_PATH}\analysis\{animal}_{date}_*.pkl")[0]
    #bhvdf = pd.read_pickle(bhv_pkl)

    EPHYS_PATH = os.path.join(DROPBOX_TASK_PATH, 'ephys', animal)
    SAVE_SYNC_PATH = glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\*')[0]
    syncdf = pd.read_pickle(fr'{SAVE_SYNC_PATH}\syncdf.pkl')
    syncdf['animal'] = animal
    syncdf['date'] = date
    syncdf['trialno'] = syncdf.index+1
    syncdf['experiment'] = determine_experiment(syncdf)

    syncdf_list.append(syncdf)

#%%
aggregated_syncdf = pd.concat(syncdf_list, ignore_index=True)
#%%

aggregated_syncdf.to_pickle(rf'{PATH_DATAFRAMES}\{animal}_aggregated_syncdf.pkl')





#%%
"""
.########.########..######..########.####.##....##..######.....
....##....##.......##....##....##.....##..###...##.##....##....
....##....##.......##..........##.....##..####..##.##..........
....##....######....######.....##.....##..##.##.##.##...####...
....##....##.............##....##.....##..##..####.##....##....
....##....##.......##....##....##.....##..##...###.##....##....
....##....########..######.....##....####.##....##..######.....

just reading and seeing what we need

this whole section should be moved to another .py file

MOVING TO A NOTEBOOK

"""

PATH_SAVE_EPHYS_AGG_FIGS = rf'{PATH_ANALYSIS_EPHYS}\aggregated_ephys'

animal = 'Ruthenium'

aggregated_neuronsdf = pd.read_pickle(rf'{PATH_ANALYSIS_EPHYS}\{animal}_aggregated_neuronsdf.pkl')

blocksdf = pd.read_pickle(rf'{PATH_DATAFRAMES}\blocksdf_Ruthenium_Palladium.pkl')

aggregated_syncdf = pd.read_pickle(rf'{PATH_DATAFRAMES}\{animal}_aggregated_syncdf.pkl')


#%%

aggregated_neuronsdf.query('(SF == "good" or SF == "ok") and cell_type == "MSN"')
# %%
sns.histplot(aggregated_neuronsdf.ch)
# %%
sns.histplot(aggregated_neuronsdf.query('date != "260218"').depth)
# %%
plt.plot(aggregated_neuronsdf.depth, '.')
# %%
aggregated_neuronsdf.query('date == "260219"').depth
# %%
"""
..######..########.##.......##........######.....########..#######.....##.....##..######..########
.##....##.##.......##.......##.......##....##.......##....##.....##....##.....##.##....##.##......
.##.......##.......##.......##.......##.............##....##.....##....##.....##.##.......##......
.##.......######...##.......##........######........##....##.....##....##.....##..######..######..
.##.......##.......##.......##.............##.......##....##.....##....##.....##.......##.##......
.##....##.##.......##.......##.......##....##.......##....##.....##....##.....##.##....##.##......
..######..########.########.########..######........##.....#######......#######...######..########
"""
cells_to_use_neurons = []
animal = 'Ruthenium'
dates_neurons = dates_Ruthenium

#dates_to_consider = blocksdf.query(f'date == {ephys_dates_dict[animal]} and animal == "{animal}" and experiment != "c"').date.unique()
#ephys_dates_dict[animal]
for date in dates_neurons:
    cells_to_use_neurons.append(aggregated_neuronsdf.query(f'date == "{date}" and cell_type == "MSN" and (SF == "good" or SF == "ok") and fr >=.2').get(['animal', 'date', 'cluster_id']).itertuples(index=False, name=None))

from itertools import chain
cells_to_use_neurons = list(chain.from_iterable(cells_to_use_neurons))

# %%

aggregated_syncdf['npx_trial_start'] = aggregated_syncdf['npx_time']

_, _, FI15a_psths_smooth, cells15a, _ = get_psths_across_cells(
    aggregated_neuronsdf,
    aggregated_syncdf.query('experiment == "a" and FI == 15'),
    cells_to_use_neurons,'npx_trial_start',
    pre_time = 0, post_time = 15, quiet = True)

_, _, FI30a_psths_smooth, cells30a, _ = get_psths_across_cells(
    aggregated_neuronsdf,
    aggregated_syncdf.query('experiment == "a" and FI == 30'),
    cells_to_use_neurons,'npx_trial_start',
    pre_time = 0, post_time = 30, quiet = True)

_, _, FI60a_psths_smooth, cells60a, _ = get_psths_across_cells(
    aggregated_neuronsdf,
    aggregated_syncdf.query('experiment == "a" and FI == 60'),
    cells_to_use_neurons,'npx_trial_start',
    pre_time = 0, post_time = 60, quiet = True)
#%%
_, _, FI15b_psths_smooth, cells15b, _ = get_psths_across_cells(
    aggregated_neuronsdf,
    aggregated_syncdf.query('experiment == "b" and FI == 15'),
    cells_to_use_neurons,'npx_trial_start',
    pre_time = 0, post_time = 15, quiet = True)

_, _, FI30b_psths_smooth, cells30b, _ = get_psths_across_cells(
    aggregated_neuronsdf,
    aggregated_syncdf.query('experiment == "b" and FI == 30'),
    cells_to_use_neurons,'npx_trial_start',
    pre_time = 0, post_time = 30, quiet = True)

_, _, FI60b_psths_smooth, cells60b, _ = get_psths_across_cells(
    aggregated_neuronsdf,
    aggregated_syncdf.query('experiment == "b" and FI == 60'),
    cells_to_use_neurons,'npx_trial_start',
    pre_time = 0, post_time = 60, quiet = True)
#%%
_, _, rwd7c_psths_smooth, cells7c, _ = get_psths_across_cells(
    aggregated_neuronsdf,
    aggregated_syncdf.query('experiment == "c" and n_protocols == 7'),
    cells_to_use_neurons,'npx_trial_start',
    pre_time = 0, post_time = 30, quiet = True)

_, _, rwd14c_psths_smooth, cells14c, _ = get_psths_across_cells(
    aggregated_neuronsdf,
    aggregated_syncdf.query('experiment == "c" and n_protocols == 14'),
    cells_to_use_neurons,'npx_trial_start',
    pre_time = 0, post_time = 30, quiet = True)

_, _, rwd28c_psths_smooth, cells28c, _ = get_psths_across_cells(
    aggregated_neuronsdf,
    aggregated_syncdf.query('experiment == "c" and n_protocols == 28'),
    cells_to_use_neurons,'npx_trial_start',
    pre_time = 0, post_time = 30, quiet = True)
#%%
FIconcat_a = np.concatenate([FI15a_psths_smooth, FI30a_psths_smooth, FI60a_psths_smooth], axis = 1)
FIconcat_b = np.concatenate([FI15b_psths_smooth, FI30b_psths_smooth, FI60b_psths_smooth], axis = 1)
FIconcat_c = np.concatenate([rwd7c_psths_smooth, rwd14c_psths_smooth, rwd28c_psths_smooth], axis = 1)
#%%
index_order_a, loadings_a, PC_space_a = do_PCA(zscore(FIconcat_a, axis = 1))
plt.imshow(zscore(FIconcat_a, axis = 1)[index_order_a], aspect = 'auto')
#%%
index_order_b, loadings_b, PC_space_b = do_PCA(zscore(FIconcat_b, axis = 1))
plt.imshow(zscore(FIconcat_b, axis = 1)[index_order_b], aspect = 'auto')
#%%
index_order_c, loadings_c, PC_space_c = do_PCA(zscore(FIconcat_c, axis = 1))
plt.imshow(zscore(FIconcat_c, axis = 1)[index_order_c], aspect = 'auto')

#%%
fig, axs = plt.subplots(4,3,tight_layout = True, figsize = (12,10), height_ratios=[2,1,1,1])

t_15 = np.arange(0,15,.01)
t_30 = np.arange(0,30,.01)
t_60 = np.arange(0,60,.01)

axs[0,0].plot(PC_space_a[:1500,0], PC_space_a[:1500,1], color = color_FI_blocks[0])
axs[0,0].plot(PC_space_a[1500:4500,0], PC_space_a[1500:4500,1], color = color_FI_blocks[1])
axs[0,0].plot(PC_space_a[4500:,0], PC_space_a[4500:,1], color = color_FI_blocks[2])

axs[0,1].plot(PC_space_b[:1500,0], PC_space_b[:1500,1], color = color_FI_blocks[0])
axs[0,1].plot(PC_space_b[1500:4500,0], PC_space_b[1500:4500,1], color = color_FI_blocks[1])
axs[0,1].plot(PC_space_b[4500:,0], PC_space_b[4500:,1], color = color_FI_blocks[2])

axs[0,2].plot(PC_space_c[:3000,0], PC_space_c[:3000,1], color = color_rwd_blocks[0])
axs[0,2].plot(PC_space_c[3000:6000,0], PC_space_c[3000:6000,1], color = color_rwd_blocks[1])
axs[0,2].plot(PC_space_c[6000:,0], PC_space_c[6000:,1], color = color_rwd_blocks[2])

for ii in range(3):
    axs[ii+1,0].plot(t_15, PC_space_a[:1500,ii], color = color_FI_blocks[0])
    axs[ii+1,0].plot(t_30, PC_space_a[1500:4500,ii], color = color_FI_blocks[1])
    axs[ii+1,0].plot(t_60, PC_space_a[4500:,ii], color = color_FI_blocks[2])

    axs[ii+1,1].plot(t_15, PC_space_b[:1500,ii], color = color_FI_blocks[0])
    axs[ii+1,1].plot(t_30, PC_space_b[1500:4500,ii], color = color_FI_blocks[1])
    axs[ii+1,1].plot(t_60, PC_space_b[4500:,ii], color = color_FI_blocks[2])

    axs[ii+1,2].plot(t_30, PC_space_c[:3000,ii], color = color_rwd_blocks[0])
    axs[ii+1,2].plot(t_30, PC_space_c[3000:6000,ii], color = color_rwd_blocks[1])
    axs[ii+1,2].plot(t_30, PC_space_c[6000:,ii], color = color_rwd_blocks[2])


for ii in range(3):
    axs[0,ii].set_ylabel('PC2')
    axs[0,ii].set_xlabel('PC1')

axs[1,0].set_ylabel('PC1')
axs[2,0].set_ylabel('PC2')
axs[3,0].set_ylabel('PC3')

[axs[3,ii].set_xlabel('time since rwd (s)') for ii in range(3)]

axs[0,0].set_title('varying FI, fixed rwd')
axs[0,1].set_title('fixed rwd rate')
axs[0,2].set_title('FI30, varying rwd')


## invert axis to compare the shape (PCA is invariant to sign)
axs[0,1].invert_xaxis()
axs[0,2].invert_xaxis()
axs[0,2].invert_yaxis()

axs[1,1].invert_yaxis()
axs[1,2].invert_yaxis()

axs[2,2].invert_yaxis()

axs[3,1].invert_yaxis()


figtitle = f'{animal} | aggregate across days | PC space | MSNs across experimental conditions'
fig.suptitle(figtitle)

fig.savefig(rf'{PATH_SAVE_EPHYS_AGG_FIGS}\{figtitle.replace('|','_')}.png', dpi = 300)
#%%
def sort_index_order(split_index, index_order, concat_for_PCA, ax = None):
    index_order_sorted = np.concatenate([index_order[split_index:], index_order[:split_index]])
    if ax == None:
        plt.imshow(concat_for_PCA[index_order_sorted], aspect = 'auto', origin = 'lower', vmin = -1, vmax = 2)
    else:
        ax.imshow(concat_for_PCA[index_order_sorted], aspect = 'auto', origin = 'lower', vmin = -1, vmax = 2)
    return index_order_sorted


fig, axs = plt.subplots(3,tight_layout = True, figsize = (12,8))
sort_index_order(120,index_order_a, zscore(FIconcat_a, axis = 1), axs[0])
sort_index_order(170,index_order_b, zscore(FIconcat_b, axis = 1), axs[1])
sort_index_order(260,index_order_c, zscore(FIconcat_c, axis = 1), axs[2])

axs[0].set_ylabel('varying FI, fixed rwd')
axs[1].set_ylabel('fixed rwd rate')
axs[2].set_ylabel('FI30, varying rwd')

figtitle = f'{animal} | aggregate across days | tiling | MSNs across experimental conditions'
fig.suptitle(figtitle)

fig.savefig(rf'{PATH_SAVE_EPHYS_AGG_FIGS}\{figtitle.replace('|','_')}.png', dpi = 300)
# %%

"""
..######..########.
.##....##.##.....##
.##.......##.....##
.##.......########.
.##.......##.......
.##....##.##.......
..######..##.......

conditioned on cp terciles

"""
blocksdf['cp_FInormalised'] = blocksdf.apply(lambda x: x.cp / x.FI, axis = 1)

fig, axs = plt.subplots(1,3, figsize = (12,4), tight_layout = True, sharex = True)

sns.histplot(ax= axs[0], data = blocksdf.query(f'animal == "{animal}" and experiment == "a" and click == False'),
             x = 'cp_FInormalised', hue = 'FI', palette=color_FI_blocks,
             element = 'step', stat = 'density', common_norm = False)
sns.histplot(ax= axs[1], data = blocksdf.query(f'animal == "{animal}" and experiment == "b" and click == False'),
             x = 'cp_FInormalised', hue = 'FI', palette=color_FI_blocks,
             element = 'step', stat = 'density', common_norm=False)
sns.histplot(ax= axs[2], data = blocksdf.query(f'animal == "{animal}" and experiment == "c" and click == False'),
             x = 'cp_FInormalised', hue = 'n_protocols', palette=color_rwd_blocks,
             element = 'step', stat = 'density', common_norm=False)

axs[0].set_xlim(0,1)
axs[1].legend([], frameon = False)

[axs[ii].set_xlabel('time since reward (normalized to FI)') for ii in range(3)]

figtitle = f'{animal} | bhv | transition point normalized to FI'
fig.suptitle(figtitle)
fig.savefig(rf'{PATH_SAVE_EPHYS_AGG_FIGS}\{figtitle.replace('|','_')}.png', dpi = 300)

# %%
blocksdf.query(f'animal == "{animal}" and experiment == "a" and click == False').cp_FInormalised
# %%
from matplotlib import colormaps as cm

tercile_colors = ['#D95F02', '#B0B0B0', '#1B9E77']
tercile_list = ['T1', 'T2', 'T3']
tercile_rateH_colors = [cm.get_cmap('copper')(1-ii*.35) for ii in range(3)]



blocksdf['tercile_cp_FInormalised'] = (
    blocksdf.query('bool_cp')
    .groupby('animaldate')['cp_FInormalised']
    .transform(lambda x: pd.qcut(x, q=3, labels=tercile_list))
)
#%%
aggregated_syncdf['cp_rel'] = aggregated_syncdf.apply(lambda x: x.cp - x.npx_trial_start, axis = 1)
aggregated_syncdf['cp_FInormalised'] = aggregated_syncdf.apply(lambda x: x.cp_rel / x.FI, axis = 1)
aggregated_syncdf['animaldate'] = syncdf.apply(lambda x: f'{x.animal} {x.date}', axis = 1)
aggregated_syncdf['tercile_cp_FInormalised'] = (
    aggregated_syncdf.query('bool_cp')
    #.groupby('animaldate')
    ['cp_FInormalised']
    .transform(lambda x: pd.qcut(x, q=3, labels=tercile_list))
)
#%%
aggregated_syncdf.tercile_cp_FInormalised.values
#%%

sns.histplot(aggregated_syncdf.cp_FInormalised)


#%%
#blocksdf['tercile_cp_FInormalised_withinFI'] = (
#    allphotometrydf.query('animal in ["Zirconium", "Niobium"] and bool_cp')
#    .groupby(['animaldate','FI','n_protocols'])['cp_FInormalised']
#    .transform(lambda x: pd.qcut(x, q=3, labels=tercile_list))
#)
#%%
sns.histplot(blocksdf.query('tercile_cp_FInormalised == "T1"').cp_FInormalised)
sns.histplot(blocksdf.query('tercile_cp_FInormalised == "T2"').cp_FInormalised)
sns.histplot(blocksdf.query('tercile_cp_FInormalised == "T3"').cp_FInormalised)

# %%
aggregated_syncdf.cp
# %%
blocksdf.query('tercile_cp_FInormalised == "T1"').get(['animal', 'date', 'trialno_within_session'])
# %%
fig, axs = plt.subplots(1,2, figsize = (8,4), tight_layout = True)
sns.histplot(ax = axs[0], data = blocksdf, x = 'cp_FInormalised', hue = 'tercile_cp_FInormalised', element = 'step')
sns.histplot(ax = axs[1], data = aggregated_syncdf, x = 'cp_FInormalised', hue = 'tercile_cp_FInormalised', element='step')

for ii in range(2): 
    axs[ii].legend(frameon = False)
    axs[ii].set_xlim(0,1)
# %%


_, _, FI30_T1_psths_smooth, cells_FI30_T1, _ = get_psths_across_cells(
    aggregated_neuronsdf,
    aggregated_syncdf.query('FI == 30 and tercile_cp_FInormalised == "T1"'),
    cells_to_use_neurons,'npx_trial_start',
    pre_time = 0, post_time = 30, quiet = True)

#%%
_, _, FI30_T2_psths_smooth, cells_FI30_T2, _ = get_psths_across_cells(
    aggregated_neuronsdf,
    aggregated_syncdf.query('FI == 30 and tercile_cp_FInormalised == "T2"'),
    cells_to_use_neurons,'npx_trial_start',
    pre_time = 0, post_time = 30, quiet = True)

_, _, FI30_T3_psths_smooth, cells_FI30_T3, _ = get_psths_across_cells(
    aggregated_neuronsdf,
    aggregated_syncdf.query('FI == 30 and tercile_cp_FInormalised == "T3"'),
    cells_to_use_neurons,'npx_trial_start',
    pre_time = 0, post_time = 30, quiet = True)

#%%

def filter_common_cells(cell_lists, matrices):
    """
    Filters matrices to include only rows corresponding to cells common across all cell_lists.

    Parameters
    ----------
    cell_lists : list of list[dict]
        A list where each element is a list of dicts representing cells.
    matrices : list of np.ndarray
        A list of matrices aligned with the cell_lists (same number of elements).
        Each matrix should have rows aligned to cells in the corresponding cell_list.

    Returns
    -------
    filtered_matrices : list of np.ndarray
        The matrices filtered to only include rows corresponding to common cells.
    common_cells : list of dict
        The list of common cell dicts.
    """

    # Convert lists of dicts to sets of tuples for comparison
    sets = [{tuple(d.items()) for d in cells} for cells in cell_lists]

    # Find intersection across all sets
    common = set.intersection(*sets)

    # Convert back to dicts
    common_cells = [dict(t) for t in common]

    # For each cell list, find indices of common cells
    index_lists = []
    for cells in cell_lists:
        indices = []
        for entry in common_cells:
            for i, d in enumerate(cells):
                if d == entry:
                    indices.append(i)
        index_lists.append(indices)

    # Filter each matrix by its indices
    filtered_matrices = [
        mat[idx, ...] for mat, idx in zip(matrices, index_lists)
    ]

    return filtered_matrices, common_cells
#%%

filtered, common_cells = filter_common_cells(
    [cells_FI30_T1, cells_FI30_T2, cells_FI30_T3],
    [np.vstack(FI30_T1_psths_smooth), np.vstack(FI30_T2_psths_smooth), np.vstack(FI30_T3_psths_smooth)]
)

filtered_concat = np.concatenate(filtered, axis = 1)

#%%
plt.imshow(zscore(filtered_concat, axis = 1), aspect = 'auto')

#%%
index_order_terciles, loadings_terciles, PC_space_terciles = do_PCA(zscore(filtered_concat, axis = 1))

# %%
plt.plot(PC_space_terciles[:3000,0], PC_space_terciles[:3000,1])
plt.plot(PC_space_terciles[3000:6000,0], PC_space_terciles[3000:6000,1])
plt.plot(PC_space_terciles[6000:,0], PC_space_terciles[6000:,1])

#plt.plot(PC_space_terciles[:,1])

#%%
fig, axs = plt.subplots(1,2, figsize = (12,4), tight_layout = True)
plot_first2PCs_w_start(PC_space_terciles[:3000,:], axs[0], tercile_colors[0], 1 )
#%%

fig, axs = plt.subplots(1, 2, figsize=(12,6),
                         subplot_kw={"projection": "3d"})

proj = [-20,-20,-50]

add_3d_line_w_start_matplotlib(axs[0], PC_space_terciles[:3000,:], color = tercile_colors[0],
                               legend = 'T1', smooth_sigma = 20, 
                                project_axis=['x','y','z'], location_project_axis=proj)

add_3d_line_w_start_matplotlib(axs[0], PC_space_terciles[3000:6000,:], color = tercile_colors[1],
                               legend = 'T2', smooth_sigma = 20, 
                                project_axis=['x','y','z'], location_project_axis=proj)

add_3d_line_w_start_matplotlib(axs[0], PC_space_terciles[6000:,:], color = tercile_colors[2],
                               legend = 'T3', smooth_sigma = 20, 
                                project_axis=['x','y','z'], location_project_axis=proj)


# %%
"""
.########..##........#######..########.########.####.##....##..######..
.##.....##.##.......##.....##....##.......##.....##..###...##.##....##.
.##.....##.##.......##.....##....##.......##.....##..####..##.##.......
.########..##.......##.....##....##.......##.....##..##.##.##.##...####
.##........##.......##.....##....##.......##.....##..##..####.##....##.
.##........##.......##.....##....##.......##.....##..##...###.##....##.
.##........########..#######.....##.......##....####.##....##..######..

from thesis nb

"""
from scipy.ndimage import gaussian_filter1d

def plot_first2PCs_w_start(PCs_matrix, ax, color, alpha):
    ax.plot(PCs_matrix[:,0], PCs_matrix[:,1], color = color, alpha = alpha)
    ax.plot(PCs_matrix[0,0], PCs_matrix[0,1], color = color, marker = 'o', alpha = alpha)

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

fig, axs = plt.subplots(2,2, figsize = (8,8), tight_layout = True, dpi = 200)#, width_ratios = [10,1])

colors_t_levers = [bone_cmap(.8-.2*ii) for ii in range(3)]


data_FI = data_FI30[index_order_sorted_FI30]
[vmin,vmax] = np.quantile(data_FI, [.01,.99])
axs[0,0].imshow(data_FI, aspect = 'auto', origin='lower', cmap = 'magma',
                        vmin = vmin, vmax = vmax,
                        extent = [0,30, 1, len(data)])


#lvrs_concat = np.concatenate([lvrs_T1_forPCA, lvrs_T2_forPCA, lvrs_T3_forPCA], axis = 1)
parent_ax = axs[0,1]
parent_spec = parent_ax.get_subplotspec()
parent_ax.remove()  # remove the original to free the slot

subgs = parent_spec.subgridspec(1, 3, wspace=0.05)  # width_ratios=[2,1,1] if you want
ax01_0 = fig.add_subplot(subgs[0, 0])
ax01_1 = fig.add_subplot(subgs[0, 1], sharey=ax01_0)
ax01_2 = fig.add_subplot(subgs[0, 2], sharey=ax01_0)

data_press = lvrs_T1_forPCA[lvrs_T1_order_sorted]
[vmin,vmax] = np.quantile(data_press, [.01,.99])
ax01_0.imshow(data_press, aspect = 'auto', origin='lower', cmap = 'magma',
                        vmin = vmin, vmax = vmax,
                        extent = [-.5,.5, 1, len(data)])

data_press = lvrs_T2_forPCA[lvrs_T2_order_sorted]
[vmin,vmax] = np.quantile(data_press, [.01,.99])
ax01_1.imshow(data_press, aspect = 'auto', origin='lower', cmap = 'magma',
                        vmin = vmin, vmax = vmax,
                        extent = [-.5,.5, 1, len(data)])

data_press = lvrs_T3_forPCA[lvrs_T3_order_sorted]
[vmin,vmax] = np.quantile(data_press, [.01,.99])
ax01_2.imshow(data_press, aspect = 'auto', origin='lower', cmap = 'magma',
                        vmin = vmin, vmax = vmax,
                        extent = [-.5,.5, 1, len(data)])

for ax in (ax01_1, ax01_2):
    ax.tick_params(axis='y', which='both', left=False, labelleft=False)
    ax.spines['left'].set_visible(False)

#data_press = lvrs_concat[index_order_sorted_lvr]
#[vmin,vmax] = np.quantile(data_press, [.01,.99])
#axs[0,1].imshow(data_press, aspect = 'auto', origin='lower', cmap = 'magma',
#                        vmin = vmin, vmax = vmax,
#                        extent = [-.5,.5, 1, len(data)])

#axs[1,0].plot(PC_space_FI30[:,0], PC_space_FI30[:,1]) -- yep, they match to before, all good, sanity check right here
#axs[1,1].plot(PC_space_lvr[:,0], PC_space_lvr[:,1])




plot_first2PCs_w_start(gaussian_filter1d(X_in_Xspace,10,axis = 0), axs[1,0], color_FI_blocks[1], 1)
plot_first2PCs_w_start(gaussian_filter1d(Y1_in_Xspace,1,axis = 0), axs[1,0], colors_t_levers[0], 1)
plot_first2PCs_w_start(gaussian_filter1d(Y2_in_Xspace,1,axis = 0), axs[1,0], colors_t_levers[1], 1)
plot_first2PCs_w_start(gaussian_filter1d(Y3_in_Xspace,1,axis = 0), axs[1,0], colors_t_levers[2], 1)


plot_first2PCs_w_start(gaussian_filter1d(X_in_lvrspace,10,axis = 0), axs[1,1], color_FI_blocks[1], 1)
plot_first2PCs_w_start(gaussian_filter1d(pcs[:100],1, axis = 0), axs[1,1],  colors_t_levers[0], 1)
plot_first2PCs_w_start(gaussian_filter1d(pcs[100:200],1,axis = 0), axs[1,1],colors_t_levers[1], 1)
plot_first2PCs_w_start(gaussian_filter1d(pcs[200:],1,axis = 0), axs[1,1],   colors_t_levers[2],1)



axs[1,0].text(14, 18.5, f'{np.round(variance_explained_by_subspace(XX, pca_X,2)*100,2)}% var. explained', fontsize=8, color=color_FI_blocks[1])
axs[1,0].text(15, 17,   f'{np.round(variance_explained_by_subspace(YY1, pca_X,2)*100,2)}% var. explained', fontsize=8, color=colors_t_levers[0])
axs[1,0].text(15, 15.5, f'{np.round(variance_explained_by_subspace(YY2, pca_X,2)*100,2)}% var. explained', fontsize=8, color=colors_t_levers[1])
axs[1,0].text(15, 14,   f'{np.round(variance_explained_by_subspace(YY3, pca_X,2)*100,2)}% var. explained', fontsize=8, color=colors_t_levers[2])


axs[1,1].text(-18, -20, f'{np.round(variance_explained_by_subspace(XX, pca_lvrspace,2)*100,2)}% var. explained', fontsize=8, color=color_FI_blocks[1])
axs[1,1].text(-18, -22.5,   f'{np.round(variance_explained_by_subspace(YY1, pca_lvrspace,2)*100,2)}% var. explained', fontsize=8, color=colors_t_levers[0])
axs[1,1].text(-18, -25, f'{np.round(variance_explained_by_subspace(YY2, pca_lvrspace,2)*100,2)}% var. explained', fontsize=8,     color=colors_t_levers[1])
axs[1,1].text(-18, -27.5,   f'{np.round(variance_explained_by_subspace(YY3, pca_lvrspace,2)*100,2)}% var. explained', fontsize=8, color=colors_t_levers[2])


axs[0,0].set_ylabel('cell #\n(sorted by FI30-aligned PCs)')
ax01_0.set_ylabel('cell #\n(sorted by lever-aligned PCs)')

axs[0,0].set_xlabel('time since reward (s)')
ax01_1.set_xlabel('time since press (s)')

axs[1,0].set_xlabel('FI30-aligned PC1')
axs[1,0].set_ylabel('FI30-aligned PC2')

axs[1,1].set_xlabel('lever-aligned PC1')
axs[1,1].set_ylabel('lever-aligned PC2')

figtitle = f'no click animals | cells in varying FI experiments | interval vs lever \nlever presses categorized from initial to final'
plt.savefig(rf'{ephys_fig_path}\{figtitle.replace("|","_").replace('\n','_')}_notitle.png')
plt.savefig(rf'{ephys_fig_path}\{figtitle.replace("|","_").replace('\n','_')}_notitle.pdf')

plt.suptitle(figtitle)
plt.savefig(rf'{ephys_fig_path}\{figtitle.replace("|","_").replace('\n','_')}.png')
plt.savefig(rf'{ephys_fig_path}\{figtitle.replace("|","_").replace('\n','_')}.pdf')



fig, axs = plt.subplots(1, 2, figsize=(12,6),
                         subplot_kw={"projection": "3d"})

proj = [-20,-20,-50]

add_3d_line_w_start_matplotlib(axs[0], X_in_Xspace, color = color_FI_blocks[1],
                               legend = 'FI', smooth_sigma = 20, 
                                project_axis=['x','y','z'], location_project_axis=proj)

add_3d_line_w_start_matplotlib(axs[0], Y1_in_Xspace, color = colors_t_levers[0],
                               legend = 'FI', smooth_sigma = 5, 
                                project_axis=['x','y','z'], location_project_axis=proj)

add_3d_line_w_start_matplotlib(axs[0], Y2_in_Xspace, color = colors_t_levers[1],
                               legend = 'FI', smooth_sigma = 5, 
                                project_axis=['x','y','z'], location_project_axis=proj)

add_3d_line_w_start_matplotlib(axs[0], Y3_in_Xspace, color = colors_t_levers[2],
                               legend = 'FI', smooth_sigma = 5, 
                                project_axis=['x','y','z'], location_project_axis=proj)


proj = [-40,-40,-40]


add_3d_line_w_start_matplotlib(axs[1], X_in_Yspace, color = color_FI_blocks[1],
                               legend = 'FI', smooth_sigma = 20, 
                                project_axis=['x','y','z'], location_project_axis=proj)

add_3d_line_w_start_matplotlib(axs[1], Y1_in_Yspace, color = colors_t_levers[0],
                               legend = 'FI', smooth_sigma = 5, 
                                project_axis=['x','y','z'], location_project_axis=proj)

add_3d_line_w_start_matplotlib(axs[1], Y2_in_Yspace, color = colors_t_levers[1],
                               legend = 'FI', smooth_sigma = 5, 
                                project_axis=['x','y','z'], location_project_axis=proj)

add_3d_line_w_start_matplotlib(axs[1], Y3_in_Yspace, color = colors_t_levers[2],
                               legend = 'FI', smooth_sigma = 5, 
                                project_axis=['x','y','z'], location_project_axis=proj)


for ii in range(2):
    axs[ii].set_xlabel('PC1')
    axs[ii].set_ylabel('PC2')
    axs[ii].set_zlabel('PC3')
    axs[ii].grid(False)

axs[0].view_init(elev=45, azim=45)   # elevation=30°, azimuth=45°
axs[1].view_init(elev=30, azim=55)   # elevation=30°, azimuth=45°


figtitle = 'interval vs lever presses | FI30 | 3d'

plt.suptitle(figtitle)
plt.savefig(rf'{ephys_fig_path}\{figtitle.replace("|","_")}.png')
plt.savefig(rf'{ephys_fig_path}\{figtitle.replace("|","_")}.pdf')



