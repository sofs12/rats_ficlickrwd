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
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import zscore

from ratcode.config.paths import PATH_STORE_PICKLES, DROPBOX_TASK_PATH, PATH_STORE_PHOTOMETRY_PICKLES
from ratcode.common.logging import determine_experiment
from ratcode.common.colorcodes import *
from ratcode.behavior import change_point
from ratcode.common.time import convert_date_bonsai, convert_timestamp
from ratcode.photometry.photometry import butter_filter, quantile_regression, get_prediction, segment_and_fit_function, mask_jumps, find_poly
from ratcode.common.dataframe import group_and_listify
from ratcode.ephys.neurons import get_psths_across_cells
from ratcode.common.math import drop_nan_rows_in_matrix

from ratcode.init import setup
setup()

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import HuberRegressor 
# %%

from scipy.stats import zscore

def get_colors_and_windows(exp, psth_bin = 0.01):
    color_palette = color_nprots_blocks if exp == 'c' else color_FI_blocks

    if exp == 'c':
        cond_I = 'rwd7'
        cond_II = 'rwd14'
        cond_III = 'rwd28'

        window_I = (0,int(30/psth_bin))
        window_II = (int(30/psth_bin),int((30*2)/psth_bin))
        window_III = (int((30*2)/psth_bin),int((30*3)/psth_bin))

    else:
        cond_I = 'FI15'
        cond_II = 'FI30'
        cond_III = 'FI60'

        window_I = (0,int(15/psth_bin))
        window_II = (int(15/psth_bin),int((15+30)/psth_bin))
        window_III = (int((15+30)/psth_bin),int((15+30+60)/psth_bin))


    all_windows = [window_I,window_II,window_III]
    all_conds = [cond_I, cond_II, cond_III]

    return all_windows,all_conds,color_palette

def sort_index_order(split_index, index_order, concat_for_PCA):
    index_order_sorted = np.concatenate([index_order[split_index:], index_order[:split_index]])
    plt.imshow(concat_for_PCA[index_order_sorted], aspect = 'auto', origin = 'lower', vmin = -1, vmax = 2)
    return index_order_sorted
#%%
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
Strontium_ephys_dates = [
 '250219',
 '250220',
 '250221',
 '250225',
 '250226',
 '250227',
 '250228',
 '250304',
 '250305',
 '250306',
 '250307',
 '250311',
 '250313',
 '250314',
 '250318',
 '250319',
 #'250325',
 #'250328'
 ]

Technetium_ephys_dates = [
 '250618',
 #'250619', -- 29 mins recording, so not all blocks came
 #'250620', -- it's only 15 mins
 '250623',
 '250624',
 '250625',
 '250626'
 ]


Niobium_ephys_dates = [
 '250618',
 #'250619', only 16 mins
 '250620',
 '250624',
 '250625',
 '250626',
 '250627',
 '250628'
]


Zirconium_ephys_dates = [
 '250321',
 '250325',
 '250326',
 '250327',
 '250328',
 '250401',
 '250418',
 '250419',
 '250422',
 #'250427', unsortable
 '250428',
 '250429',
 #'250430', sortable but terrible
 '250501',
 '250502',
 '250503',
 '250504'
 ]

ephys_dates_dict = {
    'Strontium': Strontium_ephys_dates,
    'Zirconium': Zirconium_ephys_dates,
    'Niobium': Niobium_ephys_dates,
    'Technetium': Technetium_ephys_dates
    }
# %%

"""
.##........#######.....###....########.....########..########
.##.......##.....##...##.##...##.....##....##.....##.##......
.##.......##.....##..##...##..##.....##....##.....##.##......
.##.......##.....##.##.....##.##.....##....##.....##.######..
.##.......##.....##.#########.##.....##....##.....##.##......
.##.......##.....##.##.....##.##.....##....##.....##.##......
.########..#######..##.....##.########.....########..##......
"""

PATH_DATAFRAMES = rf'C:\Users\Admin\Documents\git\ratanalysis\dfs'

unidf = pd.read_pickle(f'{PATH_DATAFRAMES}/unidf.pkl')
#blocksdf = pd.read_pickle('dfs/blocksdf_03mar.pkl')
#blocksdf = pd.read_pickle('dfs/blocksdf_july25_thesis_dataset.pkl')

ephys_fig_path = rf'D:\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\Data\FIClickRwd\analysis_ephys_thesis'
blocksdf = pd.read_pickle(rf'{ephys_fig_path}\dfs\blocksdf.pkl')
all_aggregated_neuronsdf = pd.read_pickle(rf'{ephys_fig_path}\dfs\all_aggregated_neuronsdf.pkl')


#%%
animal = 'Zirconium' ## single animal
bool_multiple_animals = False

dates_to_consider = ephys_dates_dict[animal]
cells_to_use = list(all_aggregated_neuronsdf.query(f'animal == "{animal}" and date in {list(dates_to_consider)} and cell_type == "MSN" and KSLabel == "good"').get(['animal', 'date', 'cluster_id']).itertuples(index=False, name=None))
neuronsdf = all_aggregated_neuronsdf
bhvdf = blocksdf

smooths_15 = get_psths_across_cells(
    neuronsdf, bhvdf.query('experiment != "c"'), cells_to_use, event_name='npx_trial_start',
    query_condition='FI == 15', pre_time = 0, post_time = 15
)[2]

smooths_30 = get_psths_across_cells(
    neuronsdf, bhvdf.query('experiment != "c"'), cells_to_use, event_name='npx_trial_start',
    query_condition='FI == 30', pre_time = 0, post_time = 30
)[2]

smooths_60 = get_psths_across_cells(
    neuronsdf, bhvdf.query('experiment != "c"'), cells_to_use, event_name='npx_trial_start',
    query_condition='FI == 60', pre_time = 0, post_time = 60
)[2]

smooths_for_PCA = drop_nan_rows_in_matrix(
    np.concatenate([zscore(smooths_15, axis = 1),
                    zscore(smooths_30, axis = 1),
                    zscore(smooths_60, axis = 1)], axis = 1))

#plt.imshow(smooths_for_PCA, aspect = 'auto', vmin = -1, vmax = 3)
# %%

plt.imshow(smooths_for_PCA, aspect = 'auto')
# %%
animal = 'Niobium'
date = '250624'

spike_times = all_aggregated_neuronsdf.query(f'animal == "{animal}" and date == "{date}" and SF == "good"').spike_times.values[0]
# %%

spike_times = spike_times[:1000]

plt.plot(downharpdf.timestamp_session[:20000], downharpdf.dlight_pure[:20000], lw = 1)
plt.plot(spike_times, np.ones(len(spike_times)), '|')

# %%
