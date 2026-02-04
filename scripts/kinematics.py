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
from ratcode.photometry.photometry import signal2eventsnippets, butter_filter, quantile_regression, get_prediction, segment_and_fit_function, mask_jumps, find_poly
from ratcode.common.dataframe import group_and_listify, get_dlc_df
from ratcode.ephys.neurons import get_psths_across_cells, align_spikes_to_ttl, compute_FR
from ratcode.common.math import drop_nan_rows_in_matrix

from ratcode.init import setup
setup()

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import HuberRegressor 

def interpolate_array(arr):
    # Mask for non-NaN values
    mask = ~np.isnan(arr)
    
    # Indices of the array
    idx = np.arange(len(arr))
    
    # 1. Linear Interpolation (Handles gaps between valid numbers)
    # Note: np.interp does not extrapolate by default, 
    # it constant-fills the ends (which acts as bfill/ffill)
    return np.interp(idx, idx[mask], arr[mask])

#%%
"""
....###....##....##.####.##.....##....###....##..........########.....###....########.########
...##.##...###...##..##..###...###...##.##...##..........##.....##...##.##......##....##......
..##...##..####..##..##..####.####..##...##..##..........##.....##..##...##.....##....##......
.##.....##.##.##.##..##..##.###.##.##.....##.##..........##.....##.##.....##....##....######..
.#########.##..####..##..##.....##.#########.##..........##.....##.#########....##....##......
.##.....##.##...###..##..##.....##.##.....##.##..........##.....##.##.....##....##....##......
.##.....##.##....##.####.##.....##.##.....##.########....########..##.....##....##....########
"""

animal = 'Zirconium'
date = '250425'


PATH_SAVE_DANEURONS_FIGS = os.path.join(DROPBOX_TASK_PATH, rf'analysis_DAneurons/{animal}_{date}/clusters_DA_DLC')
if not os.path.exists(PATH_SAVE_DANEURONS_FIGS):
    os.makedirs(PATH_SAVE_DANEURONS_FIGS)


DLC_PATH = rf'{PATH_STORE_PHOTOMETRY_PICKLES}/{animal}_{date}_dlcDAdf.pkl'
if os.path.exists(DLC_PATH):
    bool_dlc  = True
    dlcDAdf = pd.read_pickle(rf'{DLC_PATH}')
else:
    bool_dlc  = False
print(bool_dlc)
dlcDAdf['timestamp_session'] = dlcDAdf.apply(lambda x: np.linspace(x.trial_start_harp, x.trial_end_harp, len(x.implantSleeve_y)), axis = 1)

DLC_PATH_SIDE_FULL = glob.glob(os.path.join(DROPBOX_TASK_PATH, 'video', animal, f'{animal}_20{date[:2]}-{date[2:4]}-{date[4:]}*.h5'))[0]
_, nancoords_full = get_dlc_df(DLC_PATH_SIDE_FULL)

PATH_ANALYSIS_VIDEO = os.path.join(DROPBOX_TASK_PATH,'analysis_video')

#%%
## poke and lever position
poke_x = np.nanmean(np.hstack(nancoords_full.poke.x.values))
poke_y = np.nanmean(np.hstack(nancoords_full.poke.y.values))

lever_x = np.nanmean(np.hstack(nancoords_full.lever.x.values))
lever_y = np.nanmean(np.hstack(nancoords_full.lever.y.values))


window = [-2,2]

fig, axs = plt.subplots(3,2, figsize = (8,8), tight_layout = True, sharey = 'row', sharex='row', height_ratios=[2,1,1])

for bodypart in ['implantBase', 'implantSleeve', 'snout', 'topL']:
    #trace_x = np.hstack(dlcDAdf[f'{bodypart}_x'].values)
    #trace_y = np.hstack(dlcDAdf[f'{bodypart}_y'].values)
    trace_x = nancoords_full[bodypart].x[np.hstack(dlcDAdf.frameno_session.values)].values
    trace_y = nancoords_full[bodypart].y[np.hstack(dlcDAdf.frameno_session.values)].values

    trace_x = interpolate_array(trace_x)
    trace_y = interpolate_array(trace_y)

    snipps_x_cp, time_dlc = signal2eventsnippets(np.hstack(dlcDAdf.timestamp_session),
                                trace_x, dlcDAdf.cp_abs.values, window, .01)
    snipps_y_cp, _ = signal2eventsnippets(np.hstack(dlcDAdf.timestamp_session),
                                trace_y, dlcDAdf.cp_abs.values, window, .01)
    
    #snipps_x_nonrwd_lever, _ = signal2eventsnippets(np.hstack(dlcDAdf.timestamp_session),
    #                            trace_x, np.hstack(dlcDAdf.nonrwd_lever_abs.values), [-.5,.5], .01)
    #snipps_y_nonrwd_lever, _ = signal2eventsnippets(np.hstack(dlcDAdf.timestamp_session),
    #                            trace_y, np.hstack(dlcDAdf.nonrwd_lever_abs.values), [-.5,.5], .01)

    snipps_x_last_lever, _ = signal2eventsnippets(np.hstack(dlcDAdf.timestamp_session),
                                trace_x, np.hstack(dlcDAdf.rwd_lever_abs.values), window, .01)
    snipps_y_last_lever, _ = signal2eventsnippets(np.hstack(dlcDAdf.timestamp_session),
                                trace_y, np.hstack(dlcDAdf.rwd_lever_abs.values), window, .01)
    

    axs[0,0].plot(snipps_x_cp.T, snipps_y_cp.T, color = bodypart_color_dic[bodypart], alpha = 0.05)
    #axs[1].plot(snipps_x_nonrwd_lever.T, snipps_y_nonrwd_lever.T, color = bodypart_color_dic[bodypart], alpha = 0.05)    
    axs[0,1].plot(snipps_x_last_lever.T, snipps_y_last_lever.T, color = bodypart_color_dic[bodypart], alpha = 0.05)

    ## trajectory midpoint
    axs[0,0].plot(np.nanmean(snipps_x_cp.T[200]), np.nanmean(snipps_y_cp.T[200]), 'v', color = bodypart_color_dic[bodypart], markeredgecolor = 'white')
    axs[0,1].plot(np.nanmean(snipps_x_cp.T[200]), np.nanmean(snipps_y_cp.T[200]), 'v', color = bodypart_color_dic[bodypart], markeredgecolor = 'white')

    axs[1,0].plot(time_dlc, snipps_x_cp.T, color = bodypart_color_dic[bodypart], alpha = 0.05)
    axs[1,1].plot(time_dlc, snipps_x_last_lever.T, color = bodypart_color_dic[bodypart], alpha = 0.05)
    
    axs[2,0].plot(time_dlc, snipps_y_cp.T, color = bodypart_color_dic[bodypart], alpha = 0.05)
    axs[2,1].plot(time_dlc, snipps_y_last_lever.T, color = bodypart_color_dic[bodypart], alpha = 0.05)

    #axs[2,0].plot(time_dlc, np.nanmean(snipps_y_cp, axis = 0), color = bodypart_color_dic[bodypart])
    #axs[2,1].plot(time_dlc, np.nanmean(snipps_y_last_lever, axis = 0), color = bodypart_color_dic[bodypart])


#axs[0].plot(snipps_implantBase_x_cp.T, snipps_implantBase_cp.T, color = bodypart_color_dic['implantBase'], alpha = 0.1)
#axs[1].plot(snipps_implantBase_x_last_lever.T, snipps_implantBase_last_lever.T, color = bodypart_color_dic['implantBase'], alpha = 0.1)

for ii in range(2):
    axs[2,ii].plot(0,0, 'v', color = 'grey')#, markeredgecolor = 'white')

for ii in range(2):
    axs[0,ii].plot(poke_x,poke_y,'x', color = bodypart_color_dic['poke'])
    axs[0,ii].plot(lever_x,lever_y,'x', color = bodypart_color_dic['lever'])
    axs[1,ii].axhline(poke_x, color = bodypart_color_dic['poke'], lw = 1, ls = '--')
    axs[1,ii].axhline(lever_x, color = bodypart_color_dic['lever'], lw = 1, ls = '--')
    axs[2,ii].axhline(poke_y, color = bodypart_color_dic['poke'], lw = 1, ls = '--')
    axs[2,ii].axhline(lever_y, color = bodypart_color_dic['lever'], lw = 1, ls = '--')

    for jj in range(1,3):
        axs[jj,ii].axvline(0, color = 'grey', lw = 1, ls = '--')


axs[0,0].invert_yaxis()
axs[2,0].invert_yaxis()

axs[0,0].set_ylabel('y (px)')
for ii in range(2):
    axs[0,ii].set_xlabel('x (px)')

axs[1,0].set_ylabel('x (px)')
axs[2,0].set_ylabel('y (px)')

for ii in range(1,3):
    axs[ii,0].set_xlabel('time since transition (s)')
    axs[ii,1].set_xlabel('time since last press (s)')

axs[1,0].set_xlim(-2,2)
axs[2,0].set_xlim(-2,2)


axs[0,0].set_title('around transition time')
axs[0,1].set_title('around last lever press')


figtitle = rf'{animal} {date} | kinematics aligned to events'
fig.suptitle(figtitle)

fig.savefig(rf'{PATH_ANALYSIS_VIDEO}\{figtitle.replace('|', '_')}.png', dpi = 300)

#%%

