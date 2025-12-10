#%%
# %% bootstrap local package imports
import sys
from pathlib import Path

# This assumes this file is in rats_ficlickrwd/scripts/
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # .../rats_ficlickrwd

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Project root on sys.path:", PROJECT_ROOT)
#%%
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression

from ratcode.dlc.video import *
from ratcode.common.colorcodes import *
from ratcode.common.logging import *


# %%
dropbox_video_path = r"D:\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\data\FIClickRwd\video"
save_dlc_pkl_path = r"D:\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\Data\FIClickRwd\analysis_video"

os.listdir(save_dlc_pkl_path)
# %%


"""
..######...########.########....########.####.##.......########..######.
.##....##..##..........##.......##........##..##.......##.......##....##
.##........##..........##.......##........##..##.......##.......##......
.##...####.######......##.......######....##..##.......######....######.
.##....##..##..........##.......##........##..##.......##.............##
.##....##..##..........##.......##........##..##.......##.......##....##
..######...########....##.......##.......####.########.########..######.
"""

thres = 0.7
#%%
date = '250618'
animal = 'Niobium'
camera = 'side' # top OR side
camera_label = '_top' if camera == 'top' else ''

# %%
#dlc_files = glob.glob(fr"{dropbox_video_path}\{animal}\{animal}_{date}_{camera}_trials\{animal}_{date}_{camera}_trialno_*.h5")

nandlcdf_top = pd.read_pickle(fr'{save_dlc_pkl_path}\{animal}_{date}_top_nandlcdf.pkl')
nandlcdf_side = pd.read_pickle(fr'{save_dlc_pkl_path}\{animal}_{date}_side_nandlcdf.pkl')
# %%

#jointdf 
jointdf_pkl = glob.glob(rf"D:\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\data\FIClickRwd\analysis_photometry\{animal}_{date}*_NEWjointdf.pkl")[0]
jointdf = pd.read_pickle(jointdf_pkl)

# %%

path_to_save_subfolder = os.path.join(save_dlc_pkl_path, f"{animal}_{date}")

if not os.path.exists(path_to_save_subfolder):
    os.makedirs(path_to_save_subfolder)

#%%
trialno = 92

t_trial = np.hstack(jointdf.loc[trialno-1].timestamp_session) - jointdf.loc[trialno-1].timestamp_session[0]
t_dlc = np.linspace(0,t_trial[-1], len(nandlcdf_side.loc[trialno]))

fig, axs = plt.subplots(5,1, figsize = (12,10))

for bodypart in ['lever','poke', 'topL', 'topR','implantSleeve','implantBase', 'snout']:
    axs[0].plot(t_dlc, nandlcdf_side.loc[trialno][bodypart]['x'].values, color = bodypart_color_dic[bodypart])
    axs[1].plot(t_dlc, nandlcdf_side.loc[trialno][bodypart]['y'].values, color = bodypart_color_dic[bodypart])

axs[0].set_ylabel('x dlc coords (pixels)')

axs[1].invert_yaxis()
axs[1].set_ylabel('y dlc coords (pixels)')

axs[2].plot(t_trial, zscore(jointdf.loc[trialno-1].DA_session))
axs[2].set_ylabel('DA  from regression')

tdtomato = jointdf.loc[trialno-1].deltaF_tdtomato
gfp = jointdf.loc[trialno-1].deltaF_gfp

axs[3].plot(t_trial, zscore(tdtomato), color = 'red')
axs[3].plot(t_trial, zscore(gfp), color = 'green')

axs[4].plot(t_trial, zscore(jointdf.loc[trialno-1].DA_session), label = 'plain regression')

for bodypart in ['implantSleeve', 'topL']:
    axs[2].plot(t_dlc, zscore(-nandlcdf_side.loc[trialno][bodypart]['y'].interpolate().to_numpy(), nan_policy='omit'), color = bodypart_color_dic[bodypart])

    y = nandlcdf_side.loc[trialno][bodypart]['y']
    y_filled = y.interpolate(limit_direction='both')
    dlc_resampled = np.interp(t_trial, t_dlc, y_filled.to_numpy())

    X = np.column_stack([tdtomato, dlc_resampled])
    y = gfp

    model = LinearRegression().fit(X, y)
    gfp_pred = model.predict(X)
    residual = y - gfp_pred

    axs[4].plot(t_trial, zscore(residual), label = f'using {bodypart}', color = bodypart_color_dic['implantSleeve'])

plt.legend()

#%%

full_session_topL_y = nandlcdf_side.loc[:, 'topL']['y']
full_session_implantSleeve_y = nandlcdf_side.loc[:, 'implantSleeve']['y']
#%%

t_session = np.hstack(jointdf.timestamp_session) - jointdf.timestamp_session[0][0]
t_dlc = np.linspace(0,t_session[-1], len(full_session_topL_y))

full_session_topL_y_filled = full_session_topL_y.interpolate(limit_direction = 'both')
full_session_implantSleeve_y_filled = full_session_implantSleeve_y.interpolate(limit_direction = 'both')

full_session_topL_y_resampled = np.interp(t_session, t_dlc, full_session_topL_y_filled.to_numpy())
full_session_implantSleeve_y_resampled = np.interp(t_session, t_dlc, full_session_implantSleeve_y_filled.to_numpy())

# %%

full_session_tdtomato = np.hstack(jointdf.deltaF_tdtomato.values)
full_session_gfp = np.hstack(jointdf.deltaF_gfp.values)

# %%

X = np.column_stack([full_session_tdtomato, full_session_implantSleeve_y_resampled])
y = full_session_gfp
model = LinearRegression().fit(X, y)
gfp_pred = model.predict(X)
residual = y - gfp_pred
# %%
plt.plot(residual, lw = .7)
plt.plot(np.hstack(jointdf.DA_session.values), lw = .7)
# %%
jointdf_exploded = jointdf.explode(['timestamp_session','DA_session','deltaF_tdtomato','deltaF_gfp'])
jointdf_exploded['DA_session_dlc'] = residual


# %%
tt = 15
plt.plot(jointdf_exploded.query(f'trialno == {tt}').DA_session.values, lw = .7)
plt.plot(jointdf_exploded.query(f'trialno == {tt}').DA_session_dlc.values, lw = .7)
# %%
