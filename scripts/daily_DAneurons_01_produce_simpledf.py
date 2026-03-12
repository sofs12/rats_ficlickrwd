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


from pathlib import Path
from probeinterface.plotting import plot_probe
import argparse

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


def main():
    parser = argparse.ArgumentParser(description='Extract TTLs from neuropixel recording and correct geometry after ibl sorter')
    parser.add_argument('animal', type=str, help='Name of the animal (e.g. Ruthenium)')
    parser.add_argument('date', type=str, help='Date of the session format yymmdd (e.g. 260225)')
    args = parser.parse_args()
    animal = args.animal
    date = args.date

    setup()

    DANEURONS_PATH_HOME = os.path.join(DROPBOX_TASK_PATH, 'analysis_DAneurons')
    DANEURONS_PATH = os.path.join(DANEURONS_PATH_HOME, f'{animal}_{date}')
    if not os.path.exists(DANEURONS_PATH):
        os.makedirs(DANEURONS_PATH)

    #neurons df
    EPHYS_PATH = os.path.join(DROPBOX_TASK_PATH, 'ephys', animal)
    SAVE_SYNC_PATH = glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\*')[0]
    IBL_SORTER_PATH =  glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\{animal}{date}*\ibl_sorter_results_drift_amplitude')[0]

    syncdf = pd.read_pickle(fr'{SAVE_SYNC_PATH}\syncdf.pkl')
    exp = determine_experiment(syncdf)

    print(f'Producing simpledf for animal {animal} on date {date}; experiment {exp}')

    if exp == 'c':
        hue_variable = 'n_protocols'
        color_palette = color_rwd_blocks
        hue_variable_list = rwd_order
    else:
        hue_variable = 'FI'
        color_palette = color_FI_blocks
        hue_variable_list = FI_order

    neuronsdf = pd.read_pickle(fr'{SAVE_SYNC_PATH}\neuronsdf.pkl')
    spikes_self_aligned_all = neuronsdf.spikes_self_aligned.values

    #make sure the neuronsdf have the current version of SF labels -- will have to add another step before this
    cluster_info = pd.read_csv(rf'{IBL_SORTER_PATH}\cluster_info.tsv', sep = '\t')
    neuronsdf['SF'] = cluster_info.SF

    sorted_data = load_ibl_sorter(IBL_SORTER_PATH, animal, date, exp)

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

    simpledf['cp'] = jointdf['cp_harp']
    simpledf['cp'] = simpledf.apply(lambda x: x.cp if x.cp < x.FI else np.nan, axis = 1)
    simpledf['bool_cp'] = simpledf.cp.apply(lambda x: not(np.isnan(x)))

    simpledf['cp_abs'] = simpledf['cp'] + simpledf['trial_start']
    simpledf['rwd_onset_abs'] = simpledf['rwd_onset'] + simpledf['trial_start']

    simpledf['lever_rel_FInormalised'] = simpledf['lever_rel'] / simpledf['FI']
    simpledf['cp_FInormalised'] = simpledf['cp'] / simpledf['FI']

    simpledf['trial_in_block'] = simpledf.groupby(['blockno']).cumcount() + 1
    simpledf['bool_new_block'] = simpledf['blockno'] != simpledf['blockno'].shift(1)

    for key in ['blockno', 'FI', 'n_protocols']:
        simpledf[f'prev_{key}'] = simpledf.loc[simpledf['bool_new_block'], key].shift(1)
        simpledf[f'prev_{key}'] = simpledf[f'prev_{key}'].ffill()

    simpledf['time_DA_rel'] = simpledf.apply(lambda x: np.array(x.time_DA) - x.trial_start, axis = 1)
    simpledf['time_DA_after_cp'] = simpledf.apply(lambda x: np.array(x.time_DA_rel) - x.cp, axis = 1)

    simpledf['DA_idx_after_cp'] = simpledf.time_DA_after_cp.apply(lambda x: x>=0)
    simpledf['DA_after_cp'] = simpledf.apply(lambda x: np.array(x.DA)[x.DA_idx_after_cp], axis = 1)
    simpledf['DA_before_cp'] = simpledf.apply(lambda x: np.array(x.DA)[~x.DA_idx_after_cp], axis = 1)

    simpledf['tercile_cp_FInormalised'] = pd.qcut(simpledf['cp_FInormalised'], q=3, labels=['T1', 'T2', 'T3'])


    plt.figure()
    plt.plot(jointdf.trial_duration_harp.values, label = 'photometry')
    plt.plot(syncdf.loc[0:].query('trial_duration_s > 2').trial_duration_s.values, '--', label = 'ephys')
    plt.title('see if trials match')
    plt.legend()
    plt.show()

    simpledf['npx_trial_start'] = syncdf.query('trial_duration_s > 2').npx_time.values

    simpledf['tercile_cp_withinblock'] = (
        simpledf.groupby('blockno')['cp']
        .transform(lambda x: pd.qcut(x, q=3, labels=['T1', 'T2', 'T3']))
    )

    simpledf.to_pickle(rf'{DANEURONS_PATH_HOME}\{animal}_{date}_simpledf.pkl')

if __name__ == '__main__':
    main()
