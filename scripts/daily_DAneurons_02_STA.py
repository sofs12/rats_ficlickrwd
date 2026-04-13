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

def main():
    parser = argparse.ArgumentParser(description='Extract TTLs from neuropixel recording and correct geometry after ibl sorter')
    parser.add_argument('animal', type=str, help='Name of the animal (e.g. Ruthenium)')
    parser.add_argument('date', type=str, help='Date of the session format yymmdd (e.g. 260225)')
    args = parser.parse_args()
    animal = args.animal
    date = args.date

    setup()

    ## now looking at my old code, 01_DAneurons.py
    DANEURONS_PATH_HOME = os.path.join(DROPBOX_TASK_PATH, 'analysis_DAneurons')
    DANEURONS_PATH = os.path.join(DANEURONS_PATH_HOME, f'{animal}_{date}')
    if not os.path.exists(DANEURONS_PATH):
        os.makedirs(DANEURONS_PATH)


    #neurons df
    EPHYS_PATH = os.path.join(DROPBOX_TASK_PATH, 'ephys', animal)
    SAVE_SYNC_PATH = glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\*')[0]
    IBL_SORTER_PATH =  glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\{animal}{date}*\ibl_sorter_results_drift_amplitude')[0]
    neuronsdf = pd.read_pickle(fr'{SAVE_SYNC_PATH}\neuronsdf.pkl')

    simpledf = pd.read_pickle(rf'{DANEURONS_PATH_HOME}\{animal}_{date}_simpledf.pkl')


    # from 01_DAneurons, on the MSN classification. compute the spike triggered DA average for every neuron

    ### will move the dopamine times to npx times -- need to double check that rwd onset and trial start are the same
    ###this is the actual conversion!!!
    #plt.plot(simpledf.rwd_onset_abs.values[:-1],
    #simpledf.npx_trial_start.values[1:], '.')


    #da_ttl_clock = simpledf.rwd_onset_abs.values[:-1]
    #npx_ttl_clock = simpledf.npx_trial_start.values[1:]

    da_ttl_clock = simpledf.trial_start.values
    npx_ttl_clock = simpledf.npx_trial_start.values

    da_ttl_clock = da_ttl_clock[np.isnan(npx_ttl_clock) == False]
    npx_ttl_clock = npx_ttl_clock[np.isnan(npx_ttl_clock) == False]


    t_DA = np.hstack(simpledf.time_DA.values)
    t_DA_npx_clock = np.hstack(simpledf.time_DA.values) + np.mean(-da_ttl_clock + npx_ttl_clock)
    DA = np.hstack(simpledf.DA.values)


    #plt.plot(da_ttl_clock[:4], npx_ttl_clock[:4], '.')


    # Perform linear regression to transform DA times into NPX times

    # Reshape the data for sklearn
    da_ttl_clock_reshaped = da_ttl_clock.reshape(-1, 1)

    # Fit the linear regression model
    regressor = LinearRegression()
    regressor.fit(da_ttl_clock_reshaped, npx_ttl_clock)

    # Extract the slope and intercept
    slope = regressor.coef_[0]
    intercept = regressor.intercept_

    # Transform DA times into NPX times
    t_DA_npx_clock = t_DA * slope + intercept


    if not os.path.exists(fr'{DANEURONS_PATH}\spike_triggered_DA'):
        os.makedirs(fr'{DANEURONS_PATH}\spike_triggered_DA')

    wstart,wend = -4,4
    n_timepoints = (wend - wstart)*100
    t_around_spikes = np.linspace(wstart, wend, n_timepoints)
    clusters_to_consider = neuronsdf.query('(SF == "good" or SF == "ok")  and cell_type != "FSI" and n_spikes > 1000').cluster_id.unique()

    av_allneurons_DA_spikes = np.zeros((len(clusters_to_consider), n_timepoints))

    for idx, cluster_id in enumerate(clusters_to_consider):
        neuron_spikes = neuronsdf.query(f'cluster_id == {cluster_id}').spike_times.values[0]

        DA_around_spikes = np.zeros((n_timepoints,len(neuron_spikes)))

        for spike_idx in range(len(neuron_spikes)):
            DA_spike = DA[(t_DA_npx_clock > neuron_spikes[spike_idx] + wstart) & (t_DA_npx_clock < neuron_spikes[spike_idx] + wend)]
            if len(DA_spike) < n_timepoints:
                DA_spike = np.nan*np.ones(n_timepoints)
            DA_around_spikes[:,spike_idx] = DA_spike[:n_timepoints]

        av_DA_around_spikes = np.nanmean(DA_around_spikes, axis = 1)

        av_allneurons_DA_spikes[idx] = av_DA_around_spikes

        plt.figure(figsize=(6, 4))
        plt.plot(t_around_spikes,av_DA_around_spikes)
        plt.axvline(0, color = 'grey', linestyle = '--', alpha = 0.5)
        plt.xlabel('t around spike (s)')
        plt.ylabel('DA')
        figtitle = f'{animal} {date} | cluster {cluster_id} | DA around spikes'
        plt.title(figtitle, fontsize = 16)
        plt.tight_layout()
        plt.savefig(fr'{DANEURONS_PATH}\spike_triggered_DA\{figtitle.replace('|','_')}')
        plt.close()
 
    ## save df of STAs
    df = pd.DataFrame()
    df['cluster_id'] = clusters_to_consider
    df['av_DA_around_spikes'] = av_allneurons_DA_spikes.tolist()
    df.to_pickle(fr'{DANEURONS_PATH}\STAdf.pkl')

if __name__ == '__main__':
    main()

