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
import statsmodels.api as sm
import argparse

from ratcode.config.paths import PATH_STORE_PICKLES, DROPBOX_TASK_PATH
from ratcode.common.logging import determine_experiment
from ratcode.common.colorcodes import *
from ratcode.behavior import change_point
from ratcode.photometry.photometry import get_prediction, quantile_regression, signal2eventsnippets, find_poly, segment_and_fit_function, butter_filter, mask_jumps, make_continuous
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

    PHOTOMETRY_PATH = os.path.join(DROPBOX_TASK_PATH, 'photometry', animal)

    PATH_SAVE_DFS = os.path.join(DROPBOX_TASK_PATH, 'analysis_photometry')

    PATH_SAVE_FIGS = os.path.join(DROPBOX_TASK_PATH, 'analysis_photometry', f'{animal}_{date}')
    if not os.path.exists(PATH_SAVE_FIGS):
        os.makedirs(PATH_SAVE_FIGS)


    bhv_pkl = glob.glob(rf"{DROPBOX_TASK_PATH}\analysis\{animal}_{date}_*.pkl")[0]
    bhvdf = pd.read_pickle(bhv_pkl)

    bhvdf['cp'] = bhvdf.apply(lambda x: change_point.accepted_cp_Gallistel(x.trialno, 2, bhvdf, 'lever_rel', True)[0] if len(x.lever_rel)> 0 else np.nan, axis = 1)
    bhvdf['cp'] = bhvdf.apply(lambda x: change_point.validate_cp(x.cp, x.lever_rel) if len(x.lever_rel) > 0 else np.nan, axis = 1)

    bhvdf['bool_cp'] = np.isnan(bhvdf.cp.values) == False

    bhvdf.drop(bhvdf.query('trial_duration < 200').index, inplace = True)
    bhvdf.reset_index(drop = True, inplace = True)
    bhvdf['trialno'] = bhvdf.index + 1


    exp = determine_experiment(bhvdf)
    print(f'RUNNING DAILY PHOTOMETRY FOR\nanimal {animal}\ndate {date}\nexperiment {exp}')


    downharpdf = pd.read_pickle(rf'{PATH_SAVE_DFS}\{animal}_{date}_downharpdf.pkl')
    jointdf = pd.read_pickle(rf'{PATH_SAVE_DFS}\{animal}_{date}_NEWjointdf.pkl')

    print('downharpdf and jointdf loaded')


    jointdf['click_abs_harp'] = jointdf.trial_start_harp + jointdf.click_harp



    fig, axs = plt.subplots(2,2, figsize = (8,8), tight_layout = True, sharey='row')

    colname = 'DA_poly_session'

    eventalignment = 'click_abs_harp'
    snipps, time = signal2eventsnippets(downharpdf.timestamp_session,
                                        downharpdf[colname],
                                        np.hstack(jointdf[eventalignment].values),
                                        [-4,4], .01)
    #snipps_heatmap = zscore(snipps, axis = 1)
    axs[0,0].imshow(snipps, aspect = 'auto',
                    vmin = np.nanquantile(snipps, .05), vmax = np.nanquantile(snipps, .95),
                    cmap = 'bone', origin = 'lower',
                  extent = [time[0],time[-1],0,len(snipps)])
    axs[1,0].plot(time, np.nanmean(snipps, axis = 0), color = 'purple')

    eventalignment = 'last_lever_abs_harp'
    snipps, time = signal2eventsnippets(downharpdf.timestamp_session,
                                        downharpdf[colname],
                                        np.hstack(jointdf[eventalignment].values),
                                        [-4,4], .01)
    #snipps_heatmap = zscore(snipps, axis = 1)
    axs[0,1].imshow(snipps, aspect = 'auto',
                    vmin = np.nanquantile(snipps, .05), vmax = np.nanquantile(snipps, .95),
                    cmap = 'bone', origin = 'lower',
                  extent = [time[0],time[-1],0,len(snipps)])
    axs[1,1].plot(time, np.nanmean(snipps, axis = 0), color = 'purple')

    for ii in range(2):
        axs[1,ii].axvline(0, ls = '--', color = 'grey')

    axs[1,0].set_xlabel('time since click (s)')
    axs[1,1].set_xlabel('time since last lever press (s)')

    axs[0,0].set_ylabel('trial #')
    axs[1,0].set_ylabel('DA_poly_session')

    figtitle = f'{animal} {date} | experiment {exp} | click and last lever press aligned'
    fig.suptitle(figtitle)

    fig.savefig(rf'{PATH_SAVE_FIGS}\{figtitle.replace("|","_")}.png', dpi = 300)




if __name__ == '__main__':
    main()
