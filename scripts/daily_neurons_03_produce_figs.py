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
import argparse
import sys

from pathlib import Path
from probeinterface.plotting import plot_probe

import spikeinterface.extractors as se

import pickle 
from tqdm import tqdm

from ratcode.config.paths import PATH_STORE_PICKLES, DROPBOX_TASK_PATH
from ratcode.common.logging import determine_experiment
from ratcode.common.colorcodes import *
from ratcode.behavior import change_point
from ratcode.ephys.neurons import get_psths_across_cells, align_spikes_to_ttl, compute_FR, load_ibl_sorter, determine_cell_type, produce_neuron_fig, produce_mega_neuron_fig, extract_features_cell_type, classify_cell_type_with_features
from ratcode.common.dataframe import group_and_listify
from ratcode.common.time import convert_date_bonsai, convert_timestamp
from ratcode.common.math import drop_nans_matrix
from ratcode.common.colorcodes import FI_order, color_FI_blocks, rwd_order, color_rwd_blocks

from ratcode.init import setup

def main():
    parser = argparse.ArgumentParser(description='Extract TTLs from neuropixel recording and correct geometry after ibl sorter')
    parser.add_argument('animal', type=str, help='Name of the animal (e.g. Ruthenium)')
    parser.add_argument('date', type=str, help='Date of the session format yymmdd (e.g. 260225)')
    args = parser.parse_args()
    animal = args.animal
    date = args.date

    setup()



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


    if os.path.exists(fr'{SAVE_SYNC_PATH}/syncdf.pkl'):
        syncdf = pd.read_pickle(fr'{SAVE_SYNC_PATH}/syncdf.pkl')
        print(f'syncdf loaded for {animal} {date}')
    else:
        print('syncdf not found, run daily_neurons_02_manual_sync.py to align npx to bhv data')
        print('FATAL ERROR, killing script')
        ## if we're here, kill the script
        sys.exit(1)




    """
    ..######...#######..########..########..########.##........#######...######...########.....###....##.....##
    .##....##.##.....##.##.....##.##.....##.##.......##.......##.....##.##....##..##.....##...##.##...###...###
    .##.......##.....##.##.....##.##.....##.##.......##.......##.....##.##........##.....##..##...##..####.####
    .##.......##.....##.########..########..######...##.......##.....##.##...####.########..##.....##.##.###.##
    .##.......##.....##.##...##...##...##...##.......##.......##.....##.##....##..##...##...#########.##.....##
    .##....##.##.....##.##....##..##....##..##.......##.......##.....##.##....##..##....##..##.....##.##.....##
    ..######...#######..##.....##.##.....##.########.########..#######...######...##.....##.##.....##.##.....##

    compute autocorrelogram (spikes_self_aligned_all), and store it in neuronsdf

    run this once
    """
    print(f'reading from ibl sorter: {IBL_SORTER_PATH}')

    spike_times = np.load(rf'{IBL_SORTER_PATH}\spike_times.npy')
    spike_clusters = np.load(rf'{IBL_SORTER_PATH}\spike_clusters.npy')

    cluster_info = pd.read_csv(rf'{IBL_SORTER_PATH}\cluster_info.tsv', sep = '\t')

    print('computing autocorrelogram. this takes time...')
    spikes_self_aligned_all = []

    # in seconds
    window_start = -.2
    window_end = .2
    binW = .001

    ## historically I do this for all cells, but it's a bit of a waste of time tbh
    #for cluster_id in cluster_info.query('SF == "good" or SF == "ok"').cluster_id:
    ## do a load bar here
    for cluster_id in cluster_info.cluster_id:
        cluster_spikes = spike_times[spike_clusters == cluster_id]/sampling_frequency

        spikes_self_aligned = np.hstack(align_spikes_to_ttl(cluster_spikes,cluster_spikes,(window_start,window_end)))
        spikes_self_aligned = spikes_self_aligned[spikes_self_aligned!=0]

        spikes_self_aligned_all.append(spikes_self_aligned)


    """
    ..######...#######..########..########.########.########.....########.....###....########....###......
    .##....##.##.....##.##.....##....##....##.......##.....##....##.....##...##.##......##......##.##.....
    .##.......##.....##.##.....##....##....##.......##.....##....##.....##..##...##.....##.....##...##....
    ..######..##.....##.########.....##....######...##.....##....##.....##.##.....##....##....##.....##...
    .......##.##.....##.##...##......##....##.......##.....##....##.....##.#########....##....#########...
    .##....##.##.....##.##....##.....##....##.......##.....##....##.....##.##.....##....##....##.....##...
    ..######...#######..##.....##....##....########.########.....########..##.....##....##....##.....##...
    """

    exp = determine_experiment(syncdf)
    sorted_data = load_ibl_sorter(IBL_SORTER_PATH, animal, date, exp)

    DATACLASS_PATH = rf"{DROPBOX_TASK_PATH}\analysis_ephys\{animal}_{date}_sorted_data.pkl"

    with open(DATACLASS_PATH, 'wb') as f:
        pickle.dump(sorted_data, f)



    """
    .##....##.########.##.....##.########...#######..##....##..######.....########..########
    .###...##.##.......##.....##.##.....##.##.....##.###...##.##....##....##.....##.##......
    .####..##.##.......##.....##.##.....##.##.....##.####..##.##..........##.....##.##......
    .##.##.##.######...##.....##.########..##.....##.##.##.##..######.....##.....##.######..
    .##..####.##.......##.....##.##...##...##.....##.##..####.......##....##.....##.##......
    .##...###.##.......##.....##.##....##..##.....##.##...###.##....##....##.....##.##......
    .##....##.########..#######..##.....##..#######..##....##..######.....########..##......
    """
    neuronsdf = cluster_info #.query('n_spikes > 1000 and KSLabel in ["good","mua"]')

    neuronsdf['spike_times'] = neuronsdf.cluster_id.apply(lambda x: sorted_data.spike_times[sorted_data.spike_clusters == x]/sorted_data.sampling_frequency)
    neuronsdf['spikes_self_aligned'] = spikes_self_aligned_all
    #neuronsdf['cell_type'] = neuronsdf.apply(lambda x: determine_cell_type(x.cluster_id,sorted_data,syncdf) if x.group == 'good' else np.nan, axis = 1)

    ## new part, with neuronsdf updated -- need to do this here

    ## run this to identify the cells

    trough_to_peak = []
    interspike_ratio = []
    spike_suppression = []
    waveforms_ms = []
    waveforms = []
    for cluster_id in tqdm(neuronsdf.cluster_id.values, desc=f"Extracting features {animal} {date}"):
        if cluster_id in neuronsdf.query('SF == "good" or SF == "ok"').cluster_id.values:
            try:        
                trough_to_peak_ms, long_interspike_ratio, post_spike_suppression_ms, waveform_ms, mean_waveform = extract_features_cell_type(cluster_id, sorted_data, syncdf)
            except Exception as e:
                print(f'error in cluster_id {cluster_id}')
                trough_to_peak_ms = np.nan
                long_interspike_ratio = np.nan
                post_spike_suppression_ms = np.nan
                waveform_ms = np.nan
                mean_waveform = np.nan

        else:
            trough_to_peak_ms = np.nan
            long_interspike_ratio = np.nan
            post_spike_suppression_ms = np.nan
            waveform_ms = np.nan
            mean_waveform = np.nan


        trough_to_peak.append(trough_to_peak_ms)
        interspike_ratio.append(long_interspike_ratio)
        spike_suppression.append(post_spike_suppression_ms)
        waveforms_ms.append(waveform_ms)
        waveforms.append(mean_waveform)
    neuronsdf['trough_to_peak_ms'] = trough_to_peak
    neuronsdf['long_interspike_ratio'] = interspike_ratio
    neuronsdf['post_spike_suppression_ms'] = spike_suppression
    neuronsdf['waveform_ms'] = waveforms_ms
    neuronsdf['mean_waveform'] = waveforms

    neuronsdf['cell_type'] = neuronsdf.apply(lambda x: classify_cell_type_with_features(x.trough_to_peak_ms,
                            x.long_interspike_ratio, x.post_spike_suppression_ms), axis = 1)

    print('total good or ok SF labelled clusters')
    print(len(neuronsdf.query('SF == "good" or SF == "ok"').cluster_id))
    print()

    neuronsdf['animal'] = animal
    neuronsdf['date'] = date
    
    neuronsdf.to_pickle(fr'{SAVE_SYNC_PATH}\neuronsdf.pkl')




    """
    ..######....#######...#######..########.
    .##....##..##.....##.##.....##.##.....##
    .##........##.....##.##.....##.##.....##
    .##...####.##.....##.##.....##.##.....##
    .##....##..##.....##.##.....##.##.....##
    .##....##..##.....##.##.....##.##.....##
    ..######....#######...#######..########.
    """

    print(rf'DATA REFERING TO {animal} on {date}')

    KSlabels = pd.read_csv(rf'{IBL_SORTER_PATH}\cluster_KSLabel.tsv', sep='\t')
    cluster_info = pd.read_csv(rf'{IBL_SORTER_PATH}\cluster_info.tsv', sep = '\t')

    good_clusters = KSlabels.query('KSLabel == "good"').cluster_id.values
    mua_clusters = KSlabels.query('KSLabel == "mua"').cluster_id.values

    print(f'total clusters: {len(KSlabels)}')
    print(f'good clusters: {len(good_clusters)}')
    print(f'mua clusters: {len(mua_clusters)}')


    rising_edges = syncdf.query('trial_duration_s > 2').npx_time.values


    print('quick figures being produced')

    for cluster in good_clusters:
        produce_neuron_fig(cluster, rising_edges, sorted_data, window = (-10,10), save_fig=True, fig_save_path=PATH_SAVE_FIGS)

        #produce_mega_neuron_fig(cluster_id, sorted_data, syncdf, neuronsdf, PATH_SAVE_FIGS, bool_click = False)

    ## check if I want to keep both -- SEE BELOW UNDER THE FOLDERS

    #for cluster in mua_clusters:
    #    produce_neuron_fig(cluster, rising_edges, sorted_data, sorting_label='mua')



    """
    ..######..########....##..........###....########..########.##......
    .##....##.##..........##.........##.##...##.....##.##.......##......
    .##.......##..........##........##...##..##.....##.##.......##......
    ..######..######......##.......##.....##.########..######...##......
    .......##.##..........##.......#########.##.....##.##.......##......
    .##....##.##..........##.......##.....##.##.....##.##.......##......
    ..######..##..........########.##.....##.########..########.########
    """

    print('multiple alignment figures being produced')

    SFgood_path = rf'{PATH_SAVE_FIGS}\SF_good'
    if not(os.path.exists(SFgood_path)):
        os.makedirs(SFgood_path)

    SFok_path = rf'{PATH_SAVE_FIGS}\SF_ok'
    if not(os.path.exists(SFok_path)):
        os.makedirs(SFok_path)



    SFgood = cluster_info.query('SF == "good"').cluster_id.values
    for cluster_id in SFgood:
        produce_mega_neuron_fig(cluster_id, sorted_data, syncdf, neuronsdf, fig_save_path=SFgood_path, bool_click = False, bool_cp_corrected = False)

    SFok = cluster_info.query('SF == "ok"').cluster_id.values
    for cluster_id in SFok:
        produce_mega_neuron_fig(cluster_id, sorted_data, syncdf, neuronsdf, fig_save_path=SFok_path, bool_click = False, bool_cp_corrected = False)

    print('all done! :)')
    print('')


if __name__ == '__main__':
    main()