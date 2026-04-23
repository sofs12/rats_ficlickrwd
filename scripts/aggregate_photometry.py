'''
this file automatically aggregates all photometry data (dfs) that is stored in the analysis_photometry folder
'''

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
from datetime import date

from ratcode.config.paths import PATH_STORE_PICKLES, DROPBOX_TASK_PATH, PATH_DATAFRAMES
from ratcode.common.logging import determine_experiment
from ratcode.common.colorcodes import *
from ratcode.behavior import change_point
from ratcode.photometry.photometry import get_prediction, quantile_regression, signal2eventsnippets, find_poly, segment_and_fit_function, butter_filter, mask_jumps, make_continuous, compute_snippets_across_sessions, drop_nan_rows_in_matrix, bootstrap_ci, plot_snippets
from ratcode.common.dataframe import group_and_listify
from ratcode.common.time import convert_date_bonsai, convert_timestamp
from ratcode.common.math import drop_nans_matrix
from ratcode.common.colorcodes import FI_order, color_FI_blocks, rwd_order, color_rwd_blocks

from ratcode.init import setup
setup()

## zscore DA per session (so that we can aggregate)
def zscore_session_arrays(group):
    all_values = np.concatenate([np.atleast_1d(val) for val in group.values])
    
    mu = np.nanmean(all_values)
    sigma = np.nanstd(all_values)

    if sigma == 0 or np.isnan(sigma):
        return group # Return as is if we can't scale
    
    return group.apply(lambda x: (np.array(x) - mu) / sigma)


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



PATH_SAVE_DFS = os.path.join(DROPBOX_TASK_PATH, 'analysis_photometry')
PATH_SAVE_AGGREGATE_DA_FIGS = os.path.join(PATH_SAVE_DFS, 'aggregated_DAta')


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



aggregated_jointdf['DA_zscored_session'] = aggregated_jointdf.groupby(['animal', 'date'])['DA_poly_session'].transform(zscore_session_arrays)

cols_to_drop = ['bool_block', 'trial_start_arduino', 'trial_end_arduino',
       'trial_duration_arduino', 'lever_rel_arduino',
       'count_lever', 'pump_on_arduino',
       'pump_off_arduino', 'cp_arduino',
        'poke_rel_arduino']

aggregated_jointdf.drop(cols_to_drop, axis = 1, inplace = True)

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

aggregated_jointdf = aggregated_jointdf.rename(columns = rename_dict)

aggregated_jointdf['animaldate'] = aggregated_jointdf.apply(lambda x: f'{x.animal}_{x.date}', axis = 1)
aggregated_jointdf['trial_in_block'] = aggregated_jointdf.groupby(['animaldate','blockno']).cumcount() + 1
aggregated_jointdf['bool_new_block'] = aggregated_jointdf['blockno'] != aggregated_jointdf['blockno'].shift(1)

aggregated_jointdf = aggregated_jointdf.reset_index(drop=True)

for key in ['blockno', 'FI', 'n_protocols']:
    aggregated_jointdf[f'prev_{key}'] = aggregated_jointdf.loc[aggregated_jointdf['bool_new_block'], key].groupby(aggregated_jointdf['animaldate']).shift(1)
    aggregated_jointdf[f'prev_{key}'] = aggregated_jointdf[f'prev_{key}'].ffill()

aggregated_jointdf['bool_cp'] = aggregated_jointdf.cp.apply(lambda x: np.isnan(x) == False)
aggregated_jointdf['bool_cp'] = aggregated_jointdf.apply(lambda x: False if x.cp > x.FI else np.isnan(x.cp) == False, axis=1)
aggregated_jointdf['cp'] = aggregated_jointdf.apply(lambda x: np.nan if not x.bool_cp else x.cp, axis=1)

aggregated_jointdf['cp_abs'] = aggregated_jointdf.cp + aggregated_jointdf.trial_start
aggregated_jointdf['interpress_after_cp'] = aggregated_jointdf.apply(lambda x: np.diff(x.lever_rel[x.lever_rel > x.cp]), axis = 1)
#aggregated_jointdf['corrected_cp'] = aggregated_jointdf.apply(lambda x: x.cp - np.mean(x.interpress_after_cp), axis = 1)
#aggregated_jointdf['corrected_cp_abs'] = aggregated_jointdf.corrected_cp + aggregated_jointdf.trial_start

aggregated_jointdf['pump_on_abs'] = aggregated_jointdf.pump_on + aggregated_jointdf.trial_start
aggregated_jointdf['pump_off_abs'] = aggregated_jointdf.pump_off + aggregated_jointdf.trial_start

aggregated_jointdf['click_on_abs'] = aggregated_jointdf.click_on + aggregated_jointdf.trial_start

aggregated_jointdf['preprelast_lever_abs'] = aggregated_jointdf.lever_abs.apply(lambda x: x[-3] if len(x)>2 else np.nan)

aggregated_jointdf['trialno_within_block'] = aggregated_jointdf.groupby(['animaldate', 'blockno']).cumcount() + 1

aggregated_jointdf['cp_FInormalised'] = aggregated_jointdf.cp/aggregated_jointdf.FI

#aggregated_jointdf['DA_trial_zscored'] = aggregated_jointdf.DA.apply(lambda x: compute_zscore(x))


aggregated_jointdf['lever_index_category'] = aggregated_jointdf.lever_rel.apply(categorize_presses)


## df saved with the data it is produced (to avoid weird rewriting issues; ultimately we only care about the latest/full dataset)
today = date.today()
formatted_date = today.strftime("%y%m%d")

print(f'aggregate photometry data saved to {PATH_DATAFRAMES}\nfilename: aggregate_photometry_{formatted_date}.pkl')
aggregated_jointdf.to_pickle(rf'{PATH_DATAFRAMES}\aggregate_photometry_{formatted_date}.pkl')
