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

"""
....###....##....##.####.##.....##....###....##......
...##.##...###...##..##..###...###...##.##...##......
..##...##..####..##..##..####.####..##...##..##......
.##.....##.##.##.##..##..##.###.##.##.....##.##......
.#########.##..####..##..##.....##.#########.##......
.##.....##.##...###..##..##.....##.##.....##.##......
.##.....##.##....##.####.##.....##.##.....##.########
"""

animal = 'Cadmium'
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


dates_Cadmium = ['260409',
                 '260413',
                 '260421']

#%%
dic_animals_dates = {
    'Ruthenium': dates_Ruthenium,
    'Palladium': dates_Palladium,
    'Cadmium': dates_Cadmium
}
#%%
animalneurondf = pd.DataFrame(columns = ['date','animal','animaldate','experiment','sorted_data', 'good_clusters', 'ok_clusters'])

animalneurondf['date'] = dic_animals_dates[animal]
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

for date in dic_animals_dates[animal]: ## and then we need to check the bool_ibl_drift
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

for date in dic_animals_dates[animal]:
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

# %%
