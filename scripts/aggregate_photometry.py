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
setup()


# %%

animal = 'Palladium'


# %%

PHOTOMETRY_PATH = os.path.join(DROPBOX_TASK_PATH, 'photometry', animal)
PATH_SAVE_DFS = os.path.join(DROPBOX_TASK_PATH, 'analysis_photometry')


#%%

aggregated_jointdf = []

for file in os.listdir(PATH_SAVE_DFS):
    if animal in file:
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
#%%

## zscore DA per session (so that we can aggregate)


def zscore_session_arrays(group):
    # 1. Flatten all arrays in the session to find the "true" session mean/std
    all_values = np.concatenate([np.atleast_1d(val) for val in group.values])
    
    mu = np.nanmean(all_values)
    sigma = np.nanstd(all_values)
    
    if sigma == 0 or np.isnan(sigma):
        return group # Return as is if we can't scale
    
    # 2. Apply the Z-score to each array in the column
    return group.apply(lambda x: (np.array(x) - mu) / sigma)

# Apply the grouping
aggregated_jointdf['DA_session_zscored'] = aggregated_jointdf.groupby(['animal', 'date'])['DA_poly_session'].transform(zscore_session_arrays)

#%%

# Define your columns
colname = 'DA_session_zscored'
eventalignment = 'nonrwd_lever_abs'
## ['cp_abs','nonrwd_lever_abs','rwd_lever_abs']

# Storage for snippets and metadata arrays
all_snipps = []
meta = {
    'animal': [], 'date': [], 'experiment': [], 'FI': [], 'n_protocols': []
}

# Group by all 5 metadata factors
meta_cols = ['animal', 'date', 'experiment', 'FI', 'n_protocols']  # Adjust 'n_protocols' if your column name is different

for (ani, date, exp, fi, rwd), daydf in aggregated_jointdf.groupby(meta_cols):
    
    # 1. Extract data for this specific combination
    ts = np.hstack(daydf.timestamp_session.values)
    sig = np.hstack(daydf[colname].values)
    
    # Ensure eventalignment cells are handled whether they are scalars or arrays
    evs = np.hstack([np.atleast_1d(x) for x in daydf[eventalignment].values])
    
    # 2. Generate snippets
    snipps, time = signal2eventsnippets(ts, sig, evs, [-4, 4], 0.01)
    
    # 3. If snippets were found, record them and the metadata
    num_snipps = snipps.shape[0]
    if num_snipps > 0:
        all_snipps.append(snipps)
        
        # Create metadata arrays of length N for this batch
        meta['animal'].append(np.full(num_snipps, ani))
        meta['date'].append(np.full(num_snipps, date))
        meta['experiment'].append(np.full(num_snipps, exp))
        meta['FI'].append(np.full(num_snipps, fi))
        meta['n_protocols'].append(np.full(num_snipps, rwd))

# 4. Collapse everything into final flat arrays
snipps_matrix = np.vstack(all_snipps)
animal_idx = np.concatenate(meta['animal'])
date_idx = np.concatenate(meta['date'])
exp_idx = np.concatenate(meta['experiment'])
FI_idx = np.concatenate(meta['FI'])
rwd_idx = np.concatenate(meta['n_protocols'])

# Final Clean (optional but recommended based on your snippet)
snipps_matrix = drop_nans_matrix(snipps_matrix)

#%%

nonrwd_snipps_Pa = snipps_matrix
#rwd_snipps_Pa = snipps_matrix
#%%

## not sure where exp_list is defined, but it's part of the globals somewhere

fig, axs = plt.subplots(1,3, figsize = (12,4), tight_layout = True, sharey = True)

for ii, exp in enumerate(exp_list):
    if exp != 'c':
        for jj, FI in enumerate(FI_order):
            mask = (animal_idx == animal) & (exp_idx == exp) & (FI_idx == FI)
            subset = snipps_matrix[mask]

            axs[ii].plot(time, np.nanmean(subset, axis = 0), color = color_FI_blocks[jj])
    else:
        for jj, rwd in enumerate(rwd_order):
            mask = (animal_idx == animal) & (exp_idx == exp) & (rwd_idx == rwd)
            subset = snipps_matrix[mask]

            axs[ii].plot(time, np.nanmean(subset, axis = 0), color = color_rwd_blocks[jj])

    axs[ii].axvline(0, ls = '--', color = 'grey')
    axs[ii].set_title(f'experiment {exp}')
    axs[ii].set_xlabel('time since last lever press (s)')

axs[0].set_ylabel('av. DA across sessions (dF/F)')

figtitle = f'{animal} | all sessions aggregated | channel {colname} aligned to {eventalignment}'
plt.suptitle(figtitle)

plt.savefig(rf'{PATH_SAVE_DFS}\{figtitle.replace("|","_")}.png', dpi = 300)
#%%

fig, axs = plt.subplots(1,2, tight_layout = True, figsize = (8,4), sharey = True)

axs[0].plot(time, np.nanmean(nonrwd_snipps_Ru, axis = 0))
axs[0].plot(time, np.nanmean(rwd_snipps_Ru, axis = 0))

axs[1].plot(time, np.nanmean(nonrwd_snipps_Pa, axis = 0))
axs[1].plot(time, np.nanmean(rwd_snipps_Pa, axis = 0))

for ii in range(2):
    axs[ii].axvline(0, ls = '--', color = 'grey')
#%%
from matplotlib import cm

fig, axs = plt.subplots(3,2, tight_layout = True, figsize = (8,8), sharex = True)

lvr_rwd_labels = ['nonrwd', 'rwd']
bone_cmap = cm.get_cmap('bone')
lvr_rwd_colors = [bone_cmap(0.7), bone_cmap(0.2)]

for ii,data in enumerate([nonrwd_snipps_Ru, rwd_snipps_Ru, nonrwd_snipps_Pa, rwd_snipps_Pa]):

    data_z = zscore(data, axis = 1)
    vmin = np.nanquantile(data_z, .05)
    vmax = np.nanquantile(data_z, .95)
    axs[ii%2,int(ii/2)].imshow(data_z, aspect = 'auto', vmin = vmin, vmax = vmax, cmap = 'bone', origin = 'lower',
               extent = [time[0], time[-1], 0, data.shape[0]])

    axs[-1,int(ii/2)].plot(time, np.nanmean(data, axis = 0), label = lvr_rwd_labels[ii%2], color = lvr_rwd_colors[ii%2])

for ii in range(2):   
    axs[-1,ii].axvline(0, ls = '--', color = 'grey')
    axs[-1,ii].set_xlabel('time since lever press (s)')

axs[-1,0].legend(frameon = False)

axs[0,0].set_title('Ruthenium')
axs[0,1].set_title('Palladium')

axs[0,0].set_ylabel('non-rewarded')
axs[1,0].set_ylabel('rewarded')
axs[2,0].set_ylabel('av. DA on lever presses')

figtitle = 'DA aligned to lever presses | rewarded vs unrewarded'
plt.suptitle(figtitle)

plt.savefig(rf'{PATH_SAVE_DFS}\{figtitle.replace("|","_")}.png', dpi = 300)

#axs[1,1].imshow(rwd_snipps_Ru, aspect = 'auto', vmin = np.nanquantile(rwd_snipps_Ru, .05), vmax = np.nanquantile(rwd_snipps_Ru, .95), cmap = 'bone', origin = 'lower',
#           extent = [time[0], time[-1], 0, rwd_snipps_Ru.shape[0]])

#%%

"""
..#######..##.......########.....########.########...#######..##.....##....########.....###....####.##.......##....##
.##.....##.##.......##.....##....##.......##.....##.##.....##.###...###....##.....##...##.##....##..##........##..##.
.##.....##.##.......##.....##....##.......##.....##.##.....##.####.####....##.....##..##...##...##..##.........####..
.##.....##.##.......##.....##....######...########..##.....##.##.###.##....##.....##.##.....##..##..##..........##...
.##.....##.##.......##.....##....##.......##...##...##.....##.##.....##....##.....##.#########..##..##..........##...
.##.....##.##.......##.....##....##.......##....##..##.....##.##.....##....##.....##.##.....##..##..##..........##...
..#######..########.########.....##.......##.....##..#######..##.....##....########..##.....##.####.########....##...
"""

PATH_SAVE_FIGS = os.path.join(DROPBOX_TASK_PATH, 'analysis_photometry', f'{animal}_{date}')
if not os.path.exists(PATH_SAVE_FIGS):
    os.makedirs(PATH_SAVE_FIGS)

#PATH_SAVE_ICA = os.path.join(DROPBOX_TASK_PATH, 'analysis_photometry', '00_all_sessions_ICA_snippets')
#%%

#%%

"""
.########.##.....##.########....########.####..######..
....##....##.....##.##..........##........##..##....##.
....##....##.....##.##..........##........##..##.......
....##....#########.######......######....##..##...####
....##....##.....##.##..........##........##..##....##.
....##....##.....##.##..........##........##..##....##.
....##....##.....##.########....##.......####..######..
"""

fig, axs = plt.subplots(5,3, tight_layout = True, figsize = (12,12), sharex = True)

colors_colname = ['grey','red', 'green', 'purple']

for ii,eventalignment in enumerate(['cp_abs','nonrwd_lever_abs','rwd_lever_abs']):
    for jj,colname in enumerate(['ds_continuous_encoder','deltaF_poly_tdtomato', 'deltaF_poly_gfp', 'DA_poly_session']):

        snipps, time = signal2eventsnippets(downharpdf.timestamp_session,
                                        downharpdf[colname],
                                        np.hstack(jointdf[eventalignment].values),
                                        [-4,4], .01)
        
        snipps = drop_nans_matrix(snipps)

        if jj == 0:
            snipps = zscore(snipps, axis = 1)
            #snipps - snipps[:, 0][:, np.newaxis]
            
        
        vmin,vmax = np.nanquantile(snipps, [.05,.95])
        axs[jj,ii].imshow(snipps, aspect = 'auto', vmin = vmin, vmax = vmax, cmap = 'bone', origin = 'lower',
                          extent = [time[0],time[-1],0,len(snipps)])

        snipps_mean = np.nanmean(snipps, axis = 0)
        if jj != 0:
            axs[-1,ii].plot(time, snipps_mean, color = colors_colname[jj])
        else:
            axs_encoder = axs[-1,ii].twinx()
            axs_encoder.plot(time, snipps_mean, color = colors_colname[jj])

        axs[jj,0].set_ylabel(colname, color = colors_colname[jj])

    axs[-1,ii].set_xlabel(f't since {eventalignment} (s)')
    axs[-1,ii].axvline(0, ls = '--', color = 'grey')

figtitle = f'{animal} {date} | experiment {determine_experiment(jointdf)} | channel traces aligned to events'

fig.suptitle(figtitle)

plt.savefig(rf'{PATH_SAVE_FIGS}\{figtitle.replace('|','_')}.png', dpi = 300)

#%%

"""
.########.##.....##.########.
.##........##...##..##.....##
.##.........##.##...##.....##
.######......###....########.
.##.........##.##...##.......
.##........##...##..##.......
.########.##.....##.##.......
""" 

exp = determine_experiment(jointdf)

fig, axs = plt.subplots(1,3, tight_layout = True, sharey = True, figsize = (12,4))

if exp == 'c': ## untested
    variable = 'n_protocols'
    colorcode = color_rwd_blocks
    variable_list = rwd_order

else:
    variable = 'FI'
    colorcode = color_FI_blocks
    variable_list = FI_order


for ii,eventalignment in enumerate(['cp_abs','nonrwd_lever_abs','rwd_lever_abs']):

    for jj, variable_value in enumerate(variable_list):
        snipps, time = signal2eventsnippets(downharpdf.timestamp_session,
                                        downharpdf['DA_poly_session'],
                                        np.hstack(jointdf.query(f'{variable} == {variable_value}')[eventalignment].values),
                                        [-4,4], .01)
    
        axs[ii].plot(time, np.nanmean(snipps, axis = 0), color = colorcode[jj])
    
    axs[ii].set_xlabel(f't since {eventalignment} (s)')
    axs[ii].axvline(0, ls = '--', color = 'grey')

axs[0].set_ylabel('DA_poly_session')

figtitle = f'{animal} {date} | experiment {exp} | averages split by block condition'

fig.suptitle(figtitle)

plt.savefig(rf'{PATH_SAVE_FIGS}\{figtitle.replace('|','_')}.png', dpi = 300)

#%%

"""
..######.....###....##.....##.########....########..########..######.
.##....##...##.##...##.....##.##..........##.....##.##.......##....##
.##........##...##..##.....##.##..........##.....##.##.......##......
..######..##.....##.##.....##.######......##.....##.######....######.
.......##.#########..##...##..##..........##.....##.##.............##
.##....##.##.....##...##.##...##..........##.....##.##.......##....##
..######..##.....##....###....########....########..##........######.
"""

#downharpdf.to_pickle(rf'{PATH_SAVE_DFS}\{animal}_{date}_downharpdf.pkl')
#jointdf.to_pickle(rf'{PATH_SAVE_DFS}\{animal}_{date}_NEWjointdf.pkl')

#%%

calculate_snr(downharpdf.DA_poly_session)

#%%



#%%
#eventalignment = 'rwd_lever_abs'
#for colname in ['deltaF_poly_tdtomato', 'deltaF_poly_gfp']:#= 'DA_poly_session'
#    snipps, time = signal2eventsnippets(downharpdf.timestamp_session,
#                                        downharpdf[colname],
#                                        np.hstack(jointdf[eventalignment].values),
#                                        [-6,6], .01)
#    plt.plot(time, np.nanmean(snipps, axis = 0))
#
#snipps, time = signal2eventsnippets(downharpdf.timestamp_session,
#                                        downharpdf['DA_poly_session'],
#                                        np.hstack(jointdf[eventalignment].values),
#                                        [-6,6], .01)
#plt.plot(time, 0.05+np.nanmean(snipps, axis = 0))

#%%

"""
..#######..##.....##....###....##.......####.########.##....##....##.....##.########.########.########..####..######...######....
.##.....##.##.....##...##.##...##........##.....##.....##..##.....###...###.##..........##....##.....##..##..##....##.##....##...
.##.....##.##.....##..##...##..##........##.....##......####......####.####.##..........##....##.....##..##..##.......##.........
.##.....##.##.....##.##.....##.##........##.....##.......##.......##.###.##.######......##....########...##..##........######....
.##..##.##.##.....##.#########.##........##.....##.......##.......##.....##.##..........##....##...##....##..##.............##...
.##....##..##.....##.##.....##.##........##.....##.......##.......##.....##.##..........##....##....##...##..##....##.##....##...
..#####.##..#######..##.....##.########.####....##.......##.......##.....##.########....##....##.....##.####..######...######....

- peak to noise ratio
- photobleaching decay constant
- motion sensitivity coefficient
- total dynamic range

"""






#%%

"""
.##.....##.########.########..########
.##.....##.##.......##.....##.##......
.##.....##.##.......##.....##.##......
.#########.######...########..######..
.##.....##.##.......##...##...##......
.##.....##.##.......##....##..##......
.##.....##.########.##.....##.########

might be trash
"""

"""
tt = 22
fig, axs = plt.subplots(5, tight_layout = True, figsize = (8,6))

tomato = np.hstack(jointdf.tdtomato_poly_flat[tt])
gfp = np.hstack(jointdf.gfp_poly_flat[tt])
encoder = np.hstack(jointdf.ds_continuous_encoder[tt])
time = np.arange(0,len(tomato))/100

DA_q = gfp - quantile_regression(tomato, gfp, .5)

DA_q_norm = DA_q/(quantile_regression(tomato, gfp,.95) - quantile_regression(tomato, gfp,.05))

axs[0].plot(time, tomato, color = 'red', lw = 1)
axs[0].plot(time, gfp, color = 'green', lw = 1)

axs_encoder = axs[0].twinx()
axs_encoder.plot(time, encoder, color = 'blue', lw = 1, alpha = 0.5)

axs[1].plot(time, DA_q, color = 'purple', lw = 1)

axs[2].plot(time, DA_q_norm, color = 'teal', lw = 1)


from sklearn.decomposition import FastICA, NMF

X = np.column_stack([tomato, gfp])
ica = FastICA(n_components = 2, random_state = 42)
signals_recovered = ica.fit_transform(X)

axs[3].plot(time, signals_recovered[:,0]/5, lw = 1)
axs[3].plot(time, signals_recovered[:,1]/5, lw = 1)

X = X - np.min(X) if np.min(X) < 0 else X
nmf = NMF(n_components=2, init='nndsvda', random_state=42, max_iter=1000)
W = nmf.fit_transform(X)
#H = nmf.components_
axs[4].plot(W[:,0], lw = 1)
axs[4].plot(W[:,1], lw = 1)


axs[0].set_ylabel('raw')
axs[1].set_ylabel('regression')
axs[2].set_ylabel('q-normalized')
axs[3].set_ylabel('ICA')
axs[4].set_ylabel('NMF')

figtitle = f'{animal} {date} | experiment {determine_experiment(bhvdf)} | trial {tt}'
fig.suptitle(figtitle)
#%%
plt.plot(time, encoder)
plt.plot(time[1:], np.diff(encoder))
#%%


#%%
plt.plot(tomato, gfp, '.')

#%%

plt.plot(np.nanmean(snipps_DA_rwd, axis = 0), color = 'purple', lw = 1)

#%%
tt = 36
plt.plot(zscore(np.hstack(jointdf.query(f'trialno == {tt}').tdtomato_poly_flat)), color = 'red', lw = 1)
plt.plot(1+zscore(np.hstack(jointdf.query(f'trialno == {tt}').gfp_poly_flat)), color = 'green', lw = 1)

plt.plot(10+zscore(np.hstack(jointdf.query(f'trialno == {tt}').DA_poly_session)), color = 'purple', lw = 1)
#%%

from scipy.signal import decimate, butter, filtfilt

tdtomato = decimate(harpdf.tdtomato.values, q=10)
gfp = decimate(harpdf.gfp.values, q=10)

DA = gfp - get_prediction(tdtomato, gfp)

#%%

ss = 35000
ee = ss + 6000

fig, axs = plt.subplots(2, figsize = (6,4))

axs[0].plot(tdtomato[ss:ee], lw = 1)
axs[0].plot(gfp[ss:ee], lw = 1)
axs[0].plot(-1+DA[ss:ee], color = 'purple', lw = 1)
axs[0].plot(gfp[ss:ee] - get_prediction(tdtomato[ss:ee], gfp[ss:ee]), color = 'orange', lw = 1)

axs[1].plot(downharpdf.ds_tdtomato[ss:ee])
axs[1].plot(downharpdf.ds_gfp[ss:ee])
axs[1].plot(-1+downharpdf.DA_poly_session[ss:ee], color = 'purple', lw = 1)
axs[1].plot(downharpdf.ds_gfp[ss:ee] - get_prediction(downharpdf.ds_tdtomato[ss:ee], downharpdf.ds_gfp[ss:ee]), color = 'orange', lw = 1)
#%%



plt.plot(DA[ss:ee])
plt.plot(downharpdf.DA_poly_session[ss:ee].values)
#%%


#%%
print(calculate_snr(tdtomato))
print(calculate_snr(gfp))
#%%
print(calculate_snr(downharpdf.tdtomato))
print(calculate_snr(downharpdf.gfp))




#%%
fig, axs = plt.subplots(4,3, tight_layout = True, figsize = (12,10), sharey = 'row', sharex = True)

#axs[0,0].plot(time, np.nanmean(snipps_0_cp, axis = 0), color = 'blue', lw = 1)
#axs[0,0].plot(time, np.nanmean(snipps_1_cp, axis = 0), color = 'orange', lw = 1)
#
#axs[0,1].plot(time, np.nanmean(snipps_0_nonrwd, axis = 0), color = 'blue', lw = 1)
#axs[0,1].plot(time, np.nanmean(snipps_1_nonrwd, axis = 0), color = 'orange', lw = 1)
#
#axs[0,2].plot(time, np.nanmean(snipps_0_rwd, axis = 0), color = 'blue', lw = 1)
#axs[0,2].plot(time, np.nanmean(snipps_1_rwd, axis = 0), color = 'orange', lw = 1)
#
## constrainted ICA
#axs[1,0].plot(time, np.nanmean(snipps_cICA_dlight_cp, axis = 0), color = 'blue', lw = 1)
#axs[1,0].plot(time, np.nanmean(snipps_cICA_motion_cp, axis = 0), color = 'orange', lw = 1)
#
#axs[1,1].plot(time, np.nanmean(snipps_cICA_dlight_nonrwd, axis = 0), color = 'blue', lw = 1)
#axs[1,1].plot(time, np.nanmean(snipps_cICA_motion_nonrwd, axis = 0), color = 'orange', lw = 1)
#
#axs[1,2].plot(time, np.nanmean(snipps_cICA_dlight_rwd, axis = 0), color = 'blue', lw = 1, label = 'DA')
#axs[1,2].plot(time, np.nanmean(snipps_cICA_motion_rwd, axis = 0), color = 'orange', lw = 1, label = 'motion')
#axs[1,2].legend(frameon = False)

## regression DA for comparison
axs[2,0].plot(time, np.nanmean(snipps_DA_cp, axis = 0), color = 'purple', lw = 1)
axs[2,1].plot(time, np.nanmean(snipps_DA_nonrwd, axis = 0), color = 'purple', lw = 1)
axs[2,2].plot(time, np.nanmean(snipps_DA_rwd, axis = 0), color = 'purple', lw = 1)

## NMF
#axs[3,0].plot(time, np.nanmean(snipps_NMF_cp, axis = 0), color = 'blue', lw = 1)
#axs[3,1].plot(time, np.nanmean(snipps_NMF_nonrwd, axis = 0), color = 'blue', lw = 1)
#axs[3,2].plot(time, np.nanmean(snipps_NMF_rwd, axis = 0), color = 'blue', lw = 1, label = 'DA')
#axs[3,0].plot(time, np.nanmean(snipps_NMFmotion_cp, axis = 0), color = 'orange', lw = 1)
#axs[3,1].plot(time, np.nanmean(snipps_NMFmotion_nonrwd, axis = 0), color = 'orange', lw = 1)
#axs[3,2].plot(time, np.nanmean(snipps_NMFmotion_rwd, axis = 0), color = 'orange', lw = 1, label = 'motion')
#axs[3,2].legend(frameon = False)

for ii in range(3):
    for jj in range(4):
        axs[jj,ii].axvline(0, color = 'grey', lw = 0.5, ls = '--')

axs[-1,0].set_xlabel('time since transition (s)')
axs[-1,1].set_xlabel('time since non rwd press (s)')
axs[-1,2].set_xlabel('time since rwd press (s)')

axs[0,0].set_ylabel('ICA')
axs[1,0].set_ylabel('cICA')
axs[2,0].set_ylabel('regression DA')
axs[3,0].set_ylabel('NMF')

figtitle = f"{animal} {date} | photometry ICA snippets around events"
fig.suptitle(figtitle)

#%%
fig.savefig(rf'{PATH_SAVE_ICA}\{figtitle.replace('|','_')}.png', dpi = 300)


jointdf.to_pickle(rf'{PATH_SAVE_ICA}\jointdf_{animal}_{date}_photometry_ICA.pkl')
# %%

plt.imshow(snipps_DA_rwd, aspect = 'auto')



# %%
jointdf.keys()
# %%
"""