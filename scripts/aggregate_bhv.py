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
import datetime

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

# %%
aggregated_bhvdf = []

animal_list = ['Ruthenium', 'Palladium']

for file in os.listdir(PATH_STORE_PICKLES):
    animal = file.split('_')[0]
    if animal in animal_list:
        pkl_path = os.path.join(PATH_STORE_PICKLES, file)
        bhvdf = pd.read_pickle(pkl_path)
        bhvdf['animal'] = file.split('_')[0]
        bhvdf['date'] = file.split('_')[1]
        bhvdf['experiment'] = determine_experiment(bhvdf)
        aggregated_bhvdf.append(bhvdf)

if aggregated_bhvdf:
    aggregated_bhvdf = pd.concat(aggregated_bhvdf, ignore_index=True)

    cols_to_move = ['animal', 'date', 'experiment']
    remaining_cols = [c for c in aggregated_bhvdf.columns if c not in cols_to_move]
    aggregated_bhvdf = aggregated_bhvdf[cols_to_move + remaining_cols]

else:
    print("No files matched the criteria.")

#%%
aggregated_bhvdf.keys()
#%%
#blocksdf = pd.read_pickle(f'{PATH_DATAFRAMES}/blocksdf_july25_thesis_dataset.pkl')

#%%

"""
..######...#######..##.....##.########..##.......########.########.########....########..########
.##....##.##.....##.###...###.##.....##.##.......##..........##....##..........##.....##.##......
.##.......##.....##.####.####.##.....##.##.......##..........##....##..........##.....##.##......
.##.......##.....##.##.###.##.########..##.......######......##....######......##.....##.######..
.##.......##.....##.##.....##.##........##.......##..........##....##..........##.....##.##......
.##....##.##.....##.##.....##.##........##.......##..........##....##..........##.....##.##......
..######...#######..##.....##.##........########.########....##....########....########..##......

blocksdf for historical reasons (to distinguish from the single condition sessions)
"""

blocksdf = aggregated_bhvdf.reset_index().query('bool_block == True').reset_index(drop = True)

blocksdf['bool_blocks'] = True
blocksdf.drop(columns = ['bool_block'], inplace = True)

blocksdf['bool_lever'] = blocksdf.lever_rel.apply(lambda x: bool(len(x)))
blocksdf['count_lever'] = blocksdf.lever_rel.apply(lambda x: len(x))

blocksdf['FI'] = blocksdf.FI.apply(lambda x: int(x/1000))

blocksdf['bool_pump'] = blocksdf.pump_duration.apply(lambda x: bool(len(x)))
blocksdf = blocksdf.drop(blocksdf.query('bool_pump == False').index)

blocksdf['trialno_within_session'] = blocksdf.trialno
blocksdf['trialno'] = blocksdf.index + 1

#cp with theta = 0, just for the logodds part (tests with 3)
blocksdf['cpseries'] = blocksdf.apply(lambda x: change_point.accepted_cp_Gallistel(x.trialno, 3, blocksdf, 'lever_rel', False) if len(x.lever_rel) > 0 else x.lever_rel, axis = 1)
#%%
blocksdf['lencpseries'] = blocksdf.cpseries.apply(lambda x: len(x))
#%%
print(blocksdf.query('lencpseries == 0'))
#%%
#for now dropping the lines where this happens
#need to drop the days in which this happens? -- need to confirm why it is that this happens - hardware malfunction?
blocksdf.drop(blocksdf.query('lencpseries == 0 ').index, inplace=True)
#%%
blocksdf['cp_logodds'] = blocksdf.cpseries.apply(lambda x: x[2])
blocksdf['cp_logodds'] = blocksdf.cp_logodds.apply(lambda x: x[0] if len(x)> 0 else np.nan)
#%%
blocksdf['cp_pre'] = blocksdf.apply(lambda x: x.cpseries[0][0] if type(x.cpseries[0]) == np.ndarray else np.nan, axis = 1)#x.cpseries[0][0] < x.FI else np.nan, axis = 1)
blocksdf['cp_beforeFI'] = blocksdf.apply(lambda x: x.cp_pre < x.FI, axis = 1)
blocksdf['bool_cp'] = blocksdf.apply(lambda x: x.cp_beforeFI, axis = 1)#and x.rateH_pre < 10, axis = 1)
blocksdf['cp'] = blocksdf.apply(lambda x: x.cp_pre if x.bool_cp else np.nan, axis = 1)
#%%
blocksdf['lever_rel_s'] = blocksdf.lever_rel.apply(lambda x: x/1000)
blocksdf['presses_after_cp'] = blocksdf.apply(lambda x: x.lever_rel_s[x.lever_rel_s > x.cp] if x.bool_cp == True else np.nan, axis = 1)

blocksdf['cp'] = blocksdf.apply(lambda x: x.presses_after_cp[0] if x.bool_cp == True else np.nan, axis = 1)
#%%
#dropping sessions that are early in training
blocksdf = blocksdf.query('FI > 0').reset_index(drop = True)

#%%
blocksdf['cp_normalised'] = blocksdf.apply(lambda x: x.cp / x.FI, axis = 1)

#blocksdf.drop(columns = ['cpseries', 'cp_pre', 'cp_beforeFI'], inplace = True)

blocksdf['last_press'] = blocksdf.lever_rel.apply(lambda x: x[-1]/1000)
blocksdf['high_duration'] = blocksdf.apply(lambda x: x.last_press - x.cp if x.bool_cp == True else np.nan, axis = 1)
blocksdf['bool_cp'] = blocksdf.apply(lambda x: False if x.high_duration == 0 else x.bool_cp, axis = 1)

#rateH defined until the end of the trial, when there is a transition
blocksdf['rateH'] = blocksdf.apply(lambda x: len(x.presses_after_cp)/x.high_duration if x.bool_cp == True else np.nan, axis = 1)    


blocksdf['session_vars'] = blocksdf.apply(lambda x: f'FI {x.FI} click {bool(x.click)}', axis = 1)
blocksdf['FI_nprots'] = blocksdf.apply(lambda x: f'FI{x.FI} rwd{int(x.n_protocols)}', axis = 1)

blocksdf['animaldate'] = blocksdf.apply(lambda x: f'{x.animal} {x.date}', axis = 1)
blocksdf['first_press_beforeFI'] = blocksdf.apply(lambda x: True if x.first_press_s < x.FI else False, axis = 1)

# pump rwd rate
blocksdf.pump_duration = blocksdf.pump_duration.apply(lambda x: x[0] if type(x)!= float else np.nan)
blocksdf['pump_volume'] = blocksdf.pump_duration.apply(lambda x: 3.45/33*x)
blocksdf['pump_rwdrate'] = blocksdf.apply(lambda x: x.pump_volume/(x.FI/60), axis = 1)
blocksdf['pump_rwdrate_clean'] = blocksdf.pump_rwdrate.apply(lambda x: "I" if x < 80 else "II")
#%%
blocksdf['date_mmdd'] = blocksdf.date.apply(lambda x: x[2:])
blocksdf['datetime'] = blocksdf.date.apply(lambda x: datetime.datetime.strptime(x, '%y%m%d').date())
blocksdf['date_FI'] = blocksdf.apply(lambda x: f"{x.date_mmdd}\nFI{x.FI}", axis = 1)

blocksdf['interpress'] = blocksdf.lever_rel.apply(lambda x: np.diff(x)/1000)

blocksdf['interpress_aftercp'] = blocksdf.presses_after_cp.apply(lambda x: np.diff(x) if (type(x) == np.ndarray and len(x)>1) else np.nan)
blocksdf['presses_cp2FI'] = blocksdf.apply(lambda x: x.presses_after_cp[x.presses_after_cp < x.FI] if x.bool_cp == True else np.nan, axis = 1)
blocksdf['interpress_cp2FI'] = blocksdf.presses_cp2FI.apply(lambda x: np.diff(x) if type(x)!=float else np.nan)

blocksdf = blocksdf.drop('bool_blocks', axis = 1)

blocksdf['nprots_over_FI'] = blocksdf.apply(lambda x: x.n_protocols / x.FI, axis = 1)
blocksdf['nprots_approx'] = blocksdf.n_protocols.apply(lambda x: 14 if x == 15 else (28 if x == 30 else x))

blocksdf = blocksdf.drop(blocksdf.query('nprots_approx == 60').index)

# this is using the approx value, i.e. 15 becomes 14 and 30 becomes 28
blocksdf['nprots_over_FI'] = blocksdf.apply(lambda x: np.round(x.nprots_approx / x.FI,2), axis = 1)


#blocksessdf

blocksessdf = pd.DataFrame()
blocksessdf['animaldate'] = blocksdf.animaldate.unique()
blocksessdf['FI_list'] = blocksessdf.animaldate.apply(lambda x: blocksdf.query(f'animaldate == "{x}"').FI.unique())
blocksessdf['rwd_list'] = blocksessdf.animaldate.apply(lambda x: blocksdf.query(f'animaldate == "{x}"').nprots_approx.unique())
blocksessdf['rwdrate_list'] = blocksessdf.animaldate.apply(lambda x: blocksdf.query(f'animaldate == "{x}"').nprots_over_FI.unique())
blocksessdf['FI_len'] = blocksessdf.FI_list.apply(lambda x: len(x))
blocksessdf['nprots_len'] = blocksessdf.rwd_list.apply(lambda x: len(x))
blocksessdf['rwdrate_len'] = blocksessdf.rwdrate_list.apply(lambda x: len(x))
blocksessdf['rwdrate_matched'] = blocksessdf.rwdrate_len.apply(lambda x: True if x == 1 else False)
blocksessdf['experiment'] = blocksessdf.apply(lambda x: 'a' if x.FI_len > 1 and x.nprots_len == 1 else ('b' if x.rwdrate_matched == True and x.nprots_len > 1 else ('c' if x.nprots_len > 1 and x.FI_len == 1 else 'other')), axis = 1)
#%%
experiment_FI_dic = {
    'a': [15,30,60],
    'b': [15,30,60],    
    'c': [30],
    'other': []
}

experiment_nprots_dic = {
    'a': [14],
    'b': [7,14,28],    
    'c': [7,14,28],
    'other': []
}

blocksessdf['consider_FI'] = blocksessdf.experiment.map(experiment_FI_dic)
blocksessdf['consider_nprots'] = blocksessdf.experiment.map(experiment_nprots_dic)
blocksessdf['bool_consider_FI'] = blocksessdf.apply(lambda x: True if x.consider_FI == list(np.sort(x.FI_list)) else False, axis = 1)
blocksessdf['bool_consider_nprots'] = blocksessdf.apply(lambda x: True if x.consider_nprots == list(np.sort(x.rwd_list)) else False, axis = 1)
blocksessdf['bool_consider_session'] = blocksessdf.bool_consider_FI & blocksessdf.bool_consider_nprots


blocksdf['experiment'] = blocksdf.animaldate.apply(lambda x: blocksessdf.query(f'animaldate == "{x}"').experiment.unique()[0])
blocksdf['FI_len'] = blocksdf.animaldate.apply(lambda x: blocksessdf.query(f'animaldate == "{x}"').FI_len.unique()[0])
blocksdf['nprots_len'] = blocksdf.animaldate.apply(lambda x: blocksessdf.query(f'animaldate == "{x}"').nprots_len.unique()[0])
blocksdf['bool_consider_session'] = blocksdf.animaldate.apply(lambda x: blocksessdf.query(f'animaldate == "{x}"').bool_consider_session.unique()[0])

#%%
blocksdf['trialno_within_session'] = blocksdf.reset_index(drop = True).index.values + 1
#true if block has changed - i.e. true in the first trial of the new block
#if blocksdf.experiment.values[0] == 'c':
#    blocksdf['bool_block_changed'] = blocksdf.nprots_approx.shift(1) != blocksdf.nprots_approx
#else:
#    blocksdf['bool_block_changed'] = blocksdf.FI.shift(1) != blocksdf.FI
#
#blocksdf.loc[blocksdf.index[0], 'bool_block_changed'] = False

nprots_changed = blocksdf.nprots_approx.shift(1) != blocksdf.nprots_approx
FI_changed = blocksdf.FI.shift(1) != blocksdf.FI

blocksdf['bool_block_changed'] = nprots_changed + FI_changed

#showing that the distributions of cp don't change accross blocks in experiment c
blocksdf['blockno'] = blocksdf.bool_block_changed.cumsum() + 1
blocksdf['nprots'] = blocksdf.nprots_approx


# flag steady state if we're at the 5th or more transition trial in the block
gg = blocksdf.groupby(['animaldate', 'blockno']).groups
cumsum_cp_in_block = []

for key in list(gg.keys()):
    cumsum_cp_in_block.append(blocksdf.loc[gg[key]].bool_cp.cumsum().values)

blocksdf['cumsum_cp_in_block'] = np.hstack(cumsum_cp_in_block)
blocksdf['bool_steady_state'] = blocksdf.cumsum_cp_in_block.apply(lambda x: True if x >= 5 else False)


blocksdf.drop(blocksdf.query('bool_consider_session == False').index, inplace = True)
#%%
#blocksdf.to_pickle(rf'{PATH_DATAFRAMES}\blocksdf_Ruthenium_Palladium.pkl')
# %%
