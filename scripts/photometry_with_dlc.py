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
#import statsmodels as sm
import statsmodels.formula.api as smf
from scipy.stats import zscore
from scipy import interpolate
from sklearn.preprocessing import StandardScaler

from ratcode.config.paths import PATH_STORE_PICKLES, DROPBOX_TASK_PATH, PATH_STORE_PHOTOMETRY_PICKLES
from ratcode.common.logging import determine_experiment
from ratcode.common.colorcodes import *
from ratcode.behavior import change_point
from ratcode.common.time import convert_date_bonsai, convert_timestamp
from ratcode.photometry.photometry import butter_filter, quantile_regression, get_prediction, segment_and_fit_function, mask_jumps, find_poly
from ratcode.common.dataframe import group_and_listify, get_dlc_df
from ratcode.common.plotting import plot_XY
from ratcode.common.signals import signal2eventsnippets, detect_rising_edge

from ratcode.init import setup
setup()

def resample_dlc_to_timebase(dlc_time, dlc_signals, target_time):
    f = interpolate.interp1d(dlc_time, dlc_signals, axis = 0, bounds_error=False)
    return f(target_time)
# %%

animal = 'Zirconium'
date = '250429'

PHOTOMETRY_PATH = os.path.join(DROPBOX_TASK_PATH, 'photometry', animal)

# full video
DLC_PATH_SIDE_FULL = glob.glob(os.path.join(DROPBOX_TASK_PATH, 'video', animal, f'{animal}_20{date[:2]}-{date[2:4]}-{date[4:]}*.h5'))[0]

_, nancoords_full = get_dlc_df(DLC_PATH_SIDE_FULL)

#%%
DLC_PATH_SIDE = os.path.join(DROPBOX_TASK_PATH, 'video', animal, f'{animal}_{date}_side_trials')
DLC_PATH_TOP = os.path.join(DROPBOX_TASK_PATH, 'video', animal, f'{animal}_{date}_top_trials')


PATH_SAVE_PHOTOMETRY_FIGS = os.path.join(DROPBOX_TASK_PATH, rf'analysis_photometry/{animal}_{date}')
if not os.path.exists(PATH_SAVE_PHOTOMETRY_FIGS):
    os.makedirs(PATH_SAVE_PHOTOMETRY_FIGS)

#PATH_SAVE_FIGS = os.path.join(DROPBOX_TASK_PATH, 'analysis_photometry', f'{animal}_{date}')
#if not os.path.exists(PATH_SAVE_FIGS):
#    os.makedirs(PATH_SAVE_FIGS)

# %%
side_h5_files = glob.glob(rf'{DLC_PATH_SIDE}/*.h5')
# %%
trialnos = []
for file in side_h5_files:
    trialno = file.split('\\')[-1].split('_')[4][:3]
    trialnos.append(int(trialno))

# %%

bhv_pkl = glob.glob(rf"{DROPBOX_TASK_PATH}\analysis\{animal}_{date}_*.pkl")[0]
bhvdf = pd.read_pickle(bhv_pkl)

bhvdf['cp'] = bhvdf.apply(lambda x: change_point.accepted_cp_Gallistel(x.trialno, 2, bhvdf, 'lever_rel', True)[0] if len(x.lever_rel)> 0 else np.nan, axis = 1)
bhvdf['cp'] = bhvdf.apply(lambda x: change_point.validate_cp(x.cp, x.lever_rel) if len(x.lever_rel) > 0 else np.nan, axis = 1)

bhvdf['bool_cp'] = np.isnan(bhvdf.cp.values) == False

bhvdf.drop(bhvdf.query('trial_duration < 200').index, inplace = True)
bhvdf.reset_index(drop = True, inplace = True)
bhvdf['trialno'] = bhvdf.index + 1
# %%

PATH_STORE_PHOTOMETRY_PICKLES = os.path.join(DROPBOX_TASK_PATH, r'analysis_photometry')
downharpdf = pd.read_pickle(f'{PATH_STORE_PHOTOMETRY_PICKLES}/{animal}_{date}_kernel_regression_downharpdf.pkl')
#downharpdf = pd.read_pickle(f'{PATH_STORE_PHOTOMETRY_PICKLES}/{animal}_{date}_downharpdf.pkl')

# %%

jointdf = group_and_listify(downharpdf, 'trialno', ['timestamp_session', 'denoised_tdtomato', 'tdtomato_standardized', 'denoised_gfp', 'DA_poly_session' , 'DA_quantile_reg', 'DA_kernelreg'])

jointdf['trial_start_harp'] = jointdf.timestamp_session.apply(lambda x: x[0])
jointdf['trial_end_harp'] = jointdf.timestamp_session.apply(lambda x: x[-1])
jointdf['trial_duration_harp'] = jointdf.trial_end_harp - jointdf.trial_start_harp

jointdf.drop(jointdf.query('trial_duration_harp < 2').index, inplace = True)
jointdf.reset_index(drop = True, inplace = True)
jointdf['trialno'] = jointdf.index + 1
# %%
plt.figure()
plt.plot(jointdf.trialno, jointdf.trial_duration_harp, label = 'harp')
plt.plot(bhvdf.trialno, bhvdf.trial_duration/1000, '--', label = 'bhv')
plt.title('trial duration (i.e. identity) ok?')
plt.legend()
plt.show()
# %%

jointdf['blockno'] = bhvdf.blockno
jointdf['FI'] = bhvdf.FI
jointdf['click']  = bhvdf.click
jointdf['n_protocols'] = bhvdf.n_protocols
jointdf['bool_block'] = bhvdf.bool_block

jointdf['trial_start_arduino'] = bhvdf.trial_start
jointdf['trial_end_arduino'] = bhvdf.trial_end
jointdf['trial_duration_arduino'] = bhvdf.trial_duration

jointdf['lever_rel_arduino'] = bhvdf.lever_rel

jointdf['lever_rel_harp'] = jointdf.apply(lambda x:
                                convert_timestamp(x.lever_rel_arduino,
                                [0, x.trial_duration_arduino],
                                [0, x.trial_duration_harp]), axis = 1)


jointdf['t_trial_harp'] = jointdf.apply(lambda x: np.hstack(x.timestamp_session) - x.trial_start_harp, axis = 1)

# %%

"""
.########.....###....########.##.....##..######.
.##.....##...##.##......##....##.....##.##....##
.##.....##..##...##.....##....##.....##.##......
.########..##.....##....##....#########..######.
.##........#########....##....##.....##.......##
.##........##.....##....##....##.....##.##....##
.##........##.....##....##....##.....##..######.
"""

PATH_DA_AND_DLC = os.path.join(DROPBOX_TASK_PATH, rf'analysis_photometry/{animal}_{date}/DA_and_dlc')
if not os.path.exists(PATH_DA_AND_DLC):
    os.makedirs(PATH_DA_AND_DLC)
    
#PATH_DLC_REGRESSION = os.path.join(PATH_DA_AND_DLC, 'implantBase_dlc_regression')
#if not os.path.exists(PATH_DLC_REGRESSION):
#    os.makedirs(PATH_DLC_REGRESSION)

## create folders
for folder_name in ['full_session_regression', 'session_vs_trial', 'sliding_window_regression']:
    new_folder = os.path.join(PATH_DA_AND_DLC, folder_name)
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)


#%%

"""
.########..##........#######..########
.##.....##.##.......##.....##....##...
.##.....##.##.......##.....##....##...
.########..##.......##.....##....##...
.##........##.......##.....##....##...
.##........##.......##.....##....##...
.##........########..#######.....##...
"""

def plot_dlc_and_photometry(trial_index):
    tt = trialnos[trial_index]

    coords, nancoords = get_dlc_df(side_h5_files[trial_index])

    fig, axs = plt.subplots(4,1, figsize = (12,8), tight_layout = True, sharex = True)

    t_trial = np.hstack(jointdf.query(f'trialno == {tt}').t_trial_harp.values[0])
    t_dlc = np.linspace(0,t_trial[-1], len(nancoords))

    for lever in jointdf.query(f'trialno == {tt}').lever_rel_harp.values[0]:
        for ii in range(4):
            axs[ii].axvline(lever, color = 'grey', lw = 1)

    axs[0].plot(t_trial, zscore(jointdf.query(f'trialno == {tt}').denoised_tdtomato.values[0]), lw = .5, color = 'red')
    axs[0].plot(t_trial, zscore(jointdf.query(f'trialno == {tt}').denoised_gfp.values[0]), lw = .5, color = 'green')

    axs[1].plot(t_trial, zscore(jointdf.query(f'trialno == {tt}').DA_poly_session.values[0]), lw = .5)

    for bodypart in ['lever','poke', 'topL', 'topR', 'earL', 'earR', 'implantSleeve','implantBase', 'snout']:
        axs[2].plot(t_dlc, nancoords[bodypart].y.values, color = bodypart_color_dic[bodypart], lw = 1)
        axs[3].plot(t_dlc, nancoords[bodypart].x.values, color = bodypart_color_dic[bodypart], lw = 1)

    axs[2].invert_yaxis()


    axs[-1].set_xlabel('time (s)')

    axs[0].set_ylabel('photometry')
    axs[1].set_ylabel('DA')
    axs[2].set_ylabel('DLC y')
    axs[3].set_ylabel('DLC x')


    figtitle = f'{animal} {date} | trial {tt} | photometry and DLC'
    plt.suptitle(figtitle)

    fig.savefig(f'{PATH_DA_AND_DLC}/{figtitle.replace('|', '_')}.png', dpi = 300)

    plt.close(fig)

    return tt, nancoords, t_dlc, t_trial

"""
.########..########..######...########..########..######...######..####..#######..##....##
.##.....##.##.......##....##..##.....##.##.......##....##.##....##..##..##.....##.###...##
.##.....##.##.......##........##.....##.##.......##.......##........##..##.....##.####..##
.########..######...##...####.########..######....######...######...##..##.....##.##.##.##
.##...##...##.......##....##..##...##...##.............##.......##..##..##.....##.##..####
.##....##..##.......##....##..##....##..##.......##....##.##....##..##..##.....##.##...###
.##.....##.########..######...##.....##.########..######...######..####..#######..##....##
"""

def do_trial_dlc_regression(tt, nancoords, t_dlc, t_trial):
    ## do this session wide, but for now I can do it trial by trial

    dlc_implant = -nancoords.implantBase.y.values
    dlc_implant_filled = pd.Series(dlc_implant).interpolate(limit_direction='both').to_numpy()

    tdtomato = jointdf.query(f'trialno == {tt}').denoised_tdtomato.values[0]
    gfp = jointdf.query(f'trialno == {tt}').denoised_gfp.values[0]

    interp_func = interpolate.interp1d(t_dlc, dlc_implant_filled, kind='linear')
    dlc_interpolated = interp_func(t_trial)

    X = np.column_stack([tdtomato, dlc_interpolated])
    y = gfp

    model = sm.QuantReg(y, sm.add_constant(X)).fit()
    gfp_pred = model.predict(sm.add_constant(X))

    residual = y - gfp_pred

    return tdtomato, gfp, dlc_interpolated, residual

def plot_dlc_regression_results(tt, tdtomato, gfp, dlc_interpolated, residual, t_trial):
    fig, axs = plt.subplots(2,1, figsize = (12,6), sharex = True, tight_layout = True)

    for lever in jointdf.query(f'trialno == {tt}').lever_rel_harp.values[0]:
        for ii in range(2):
            axs[ii].axvline(lever, color = 'grey', lw = 1, alpha = 0.5)

    axs[0].plot(t_trial, zscore(gfp), color = 'green', lw = .5, label = 'gfp')
    axs[0].plot(t_trial, zscore(tdtomato), color = 'red', lw = .5, label = 'tdtomato')

    axs[0].plot(t_trial, zscore(dlc_interpolated), color = bodypart_color_dic['implantBase'], lw = 1, label = 'implantBase_y')

    axs[1].plot(t_trial, zscore(jointdf.query(f'trialno == {tt}').DA_poly_session.values[0]), lw = .5, label = 'robust no dlc')
    axs[1].plot(t_trial, zscore(residual), color = 'purple', lw = .5, label = 'quantile reg pred')


    axs[0].set_ylabel('regressors (zscored)')
    axs[0].legend(loc = 'lower right', ncols = 3, frameon = False)

    axs[1].set_ylabel('DA comparison (zscored)')
    axs[1].legend(loc = 'lower right', ncols = 2, frameon = False)

    axs[-1].set_xlabel('time (s)')

    figtitle = f'{animal} {date} | trial {tt} | quantile regression (tdTomato, implantBase_y)'
    plt.suptitle(figtitle)

    fig.savefig(f'{PATH_DLC_REGRESSION}/{figtitle.replace('|', '_')}.png', dpi = 300)

    plt.close(fig)

#%%

"""
.########.....###....########..######..##.....##....########..##........#######..########..######.
.##.....##...##.##......##....##....##.##.....##....##.....##.##.......##.....##....##....##....##
.##.....##..##...##.....##....##.......##.....##....##.....##.##.......##.....##....##....##......
.########..##.....##....##....##.......#########....########..##.......##.....##....##.....######.
.##.....##.#########....##....##.......##.....##....##........##.......##.....##....##..........##
.##.....##.##.....##....##....##....##.##.....##....##........##.......##.....##....##....##....##
.########..##.....##....##.....######..##.....##....##........########..#######.....##.....######.
"""

for trial_index in range(len(trialnos)):
    tt, nancoords, t_dlc, t_trial = plot_dlc_and_photometry(trial_index)
    tdtomato, gfp, dlc_interpolated, residual = do_trial_dlc_regression(tt, nancoords, t_dlc, t_trial)
    plot_dlc_regression_results(tt, tdtomato, gfp, dlc_interpolated, residual, t_trial)



#%%

fig, axs = plt.subplots(1,2, figsize = (10,5), sharey=True)

for bodypart in ['lever','poke', 'implantSleeve','implantBase']:
    plot_XY(nancoords, bodypart, color = bodypart_color_dic[bodypart], ax = axs[0])

for bodypart in ['lever','poke', 'topL','topR']:
    plot_XY(nancoords, bodypart, color = bodypart_color_dic[bodypart], ax = axs[1])

axs[0].invert_yaxis()
#axs[1].invert_yaxis()

#%%
#plt.figure()
#plt.plot([t_dlc, t_dlc],
#         [nancoords.implantSleeve.x.values - nancoords.implantBase.x.values,
#        nancoords.implantSleeve.y.values - nancoords.implantBase.y.values])
##          )#, c = t_dlc)
#plt.show()
#%%
"""
.########.########.....###....##.....##.########
.##.......##.....##...##.##...###...###.##......
.##.......##.....##..##...##..####.####.##......
.######...########..##.....##.##.###.##.######..
.##.......##...##...#########.##.....##.##......
.##.......##....##..##.....##.##.....##.##......
.##.......##.....##.##.....##.##.....##.########
show frame (to really see if there is a problem with the TTL sync)
"""

import cv2

#VIDEO_PATH = rf"D:\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\Data\FIClickRwd\video\Niobium\Niobium_2025-06-24T12_37_06DLC_resnet50_side-implant-earsNov13shuffle1_100000_labeled.mp4"

#VIDEO_PATH = os.path.join(DROPBOX_TASK_PATH, 'video', animal, f'{animal}_{date}_side_trials',f'{animal}_{date}_side_trialno_001')
VIDEO_PATH = rf"D:\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\Data\FIClickRwd\video\Niobium\Niobium_250624_side_trials\Niobium_250624_side_trialno_010DLC_resnet50_side-implant-earsNov13shuffle1_100000_labeled.mp4"

trial_index = 9
coords, nancoords = get_dlc_df(side_h5_files[trial_index])


cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total_frames != len(nancoords):
    raise ValueError ('mismatch between frame count and dlc labels\n' \
    f'video has {total_frames} frames, but dlc has {len(nancoords)} labels')

fig, axs = plt.subplots(1,5, figsize = (20,4), tight_layout = True)

target_frame_index = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
axs[0].imshow(frame_rgb)

target_frame_index = int(total_frames*.25)
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
axs[1].imshow(frame_rgb)

target_frame_index = int(total_frames*.5)
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
axs[2].imshow(frame_rgb)

target_frame_index = int(total_frames*.75)
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
axs[3].imshow(frame_rgb)

target_frame_index = int(total_frames-1)
cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_index)
ret, frame = cap.read()
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
axs[4].imshow(frame_rgb)

cap.release()

for ii in range(5):
    axs[ii].set_axis_off()
    #axs[ii].set_yticks('')


figtitle = f'{animal} {date} | trial {trial_index+1}'
plt.suptitle(figtitle)

#%%

duration_bhv = bhvdf.query(f'trialno == {trial_index+1}').trial_duration.values[0]
duration_dlc = len(nancoords)

print(duration_bhv)
print(duration_dlc)

#%%

lever_rels =  bhvdf.query(f'trialno == {trial_index+1}').lever_rel.values[0]

lever_rels_in_dlc = (lever_rels * duration_dlc/duration_bhv).astype(int)
#%%
cap = cv2.VideoCapture(VIDEO_PATH)

index = lever_rels_in_dlc[0]
print(index)

fig, axs = plt.subplots(3,4, figsize = (12,8))
for ii,frame_index in enumerate(np.arange(index-100, index+130,20)):#lever_rels_in_dlc[:2]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    row = int(ii/4)
    col = ii%4
    axs[row,col].imshow(frame_rgb)
    if frame_index == index:
        title_color = 'red'
    else:
        title_color = 'black'
    axs[row,col].set_title(frame_index, color = title_color)
    axs[row,col].set_axis_off()
    #plt.show()

#%%
fig, axs = plt.subplots()
for bodypart in ['snout', 'topL', 'topR', 'implantBase', 'lever', 'poke']:
    axs.plot(nancoords[bodypart].y[index-500:index+500], color = bodypart_color_dic[bodypart])

axs.axvline(index)
axs.invert_yaxis()
#%%
fig, axs = plt.subplots(figsize = (12,4))
for bodypart in ['lever']:#, 'topL', 'topR', 'snout', 'implantBase', 'implantSleeve']:
    axs.plot(nancoords[bodypart].y, color = bodypart_color_dic[bodypart], lw = 1)
for lever in lever_rels_in_dlc:
    axs.axvline(lever, lw = 1, color = 'grey')

axs.invert_yaxis()

#%%
fig, axs = plt.subplots()

for bodypart in ['snout', 'topL', 'implantBase', 'lever', 'poke']:
    plot_XY(nancoords[index-100:], bodypart, bodypart_color_dic[bodypart])

axs.invert_yaxis()

#%%

meta = pd.read_csv(rf"D:\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\Data\FIClickRwd\metadata\Niobium\Niobium_Metadata_2025-06-24T12_37_05.csv",
                   header = None)

meta[1] = meta[1]- meta[1][0]
#%%
#plt.plot(meta[0])
frames_skipped = np.diff(meta[1])

print(np.sum(frames_skipped[frames_skipped>1]))

plt.plot(frames_skipped)
plt.plot(meta[0])
#%%

#%%
len(meta) - 247


#%%
from scipy.stats import linregress

def convert_timestamp(timepoint_in_A, time_A, time_B):


    slope, intercept, _, _, _ = linregress(time_A, time_B)
    timepoint_in_B = timepoint_in_A*slope + intercept
    
    return timepoint_in_B

convert_timestamp(100, np.arange(duration_bhv), np.arange(duration_dlc))


#%%

"""
.########.########.##......
....##.......##....##......
....##.......##....##......
....##.......##....##......
....##.......##....##......
....##.......##....##......
....##.......##....########
"""

## import gpio from camera and ttl from photometry

camera_view = 'side'

ftoken_bhv = glob.glob(rf"{DROPBOX_TASK_PATH}\behavior\{animal}\{animal}_**_{date}**")[0]
ftoken_pkl = glob.glob(rf"{DROPBOX_TASK_PATH}\analysis\{animal}_{date}**")[0]

camera_prefix = '_top' if camera_view == 'top' else ''
fps = 150 if camera_view == 'side' else 60

ftoken_gpio = glob.glob(rf"{DROPBOX_TASK_PATH}\metadata\{animal}\{animal}_Metadata{camera_prefix}_20{date[:2]}-{date[2:4]}-{date[-2:]}**")[0]
ftoken_video = glob.glob(rf"{DROPBOX_TASK_PATH}\video\{animal}\{animal}**{camera_prefix}_20{date[:2]}-{date[2:4]}-{date[-2:]}**")[0]
ftoken_photometry = glob.glob(rf"{DROPBOX_TASK_PATH}\photometry\{animal}\{animal}_Photometry_in2_20{date[:2]}-{date[2:4]}-{date[-2:]}**")[0]
#%%

gpiodf = pd.read_csv(ftoken_gpio, names=['ttl', 'frame', 'bla'], header=0)
gpiodf['frame_session'] = gpiodf.frame - gpiodf.frame[0]

in2 = pd.read_csv(ftoken_photometry, header = None)
in2.columns = ['in2', 'timestamp']
in2['gpio'] = in2.in2/2**16*20
in2['ttl_bool'] = in2.gpio.apply(lambda x: int(x>1))
in2['timestamp'] = in2.timestamp - in2.timestamp.values[0]
#%%
bhv_ttls = np.histogram(bhvdf.trial_start.values,np.arange(bhvdf.trial_start.values[0],bhvdf.trial_start.values[-1],1))[0]
side_ttls = gpiodf.ttl.values

bhv_t_s = np.linspace(0,len(bhv_ttls)/1000, len(bhv_ttls))
side_t_s = np.linspace(0,len(side_ttls)/150,len(side_ttls))
photometry_t_s = np.linspace(0,len(in2)/1000, len(in2))

fig, axs = plt.subplots(2)

#axs[1].plot(np.diff(bhvdf.trial_start)/1000)
axs[0].plot(bhv_t_s, bhv_ttls, label = 'bhv')
axs[0].plot(side_t_s, side_ttls/2, label = 'side')
axs[0].plot(photometry_t_s, in2.ttl_bool/5, label = 'DA')

#plt.xlim(85,92)

fig.suptitle(f'{animal} {date}')
#%%

"""
.########.##.....##.##.......##......
.##.......##.....##.##.......##......
.##.......##.....##.##.......##......
.######...##.....##.##.......##......
.##.......##.....##.##.......##......
.##.......##.....##.##.......##......
.##........#######..########.########
""" 

plt.plot(zscore(nancoords_full.implantBase.y, nan_policy='omit'))
plt.plot(side_ttls)

#%%
side_frameno = gpiodf.frame.values-gpiodf.frame.values[0]

ttl_frameno_dlc = detect_rising_edge(side_ttls,side_frameno)



#%%
ttl_photometry = detect_rising_edge(downharpdf.ttl_bool.values, downharpdf.timestamp_session.values)
ttl_dlc = detect_rising_edge(side_ttls,side_t_s)

alignment_window = [-15,15]

snippets_gfp, alignment = signal2eventsnippets(downharpdf.timestamp_session,downharpdf.denoised_gfp, ttl_photometry,alignment_window,1/1000)
snippets_tdtomato, alignment = signal2eventsnippets(downharpdf.timestamp_session,downharpdf.denoised_tdtomato, ttl_photometry, alignment_window,1/1000)
snippets_DA, alignment_DA = signal2eventsnippets(downharpdf.timestamp_session,downharpdf.DA_poly_session, ttl_photometry, alignment_window,1/1000)

snippets_sleeve, alignment_dlc = signal2eventsnippets(side_t_s,nancoords_full.implantSleeve.y.values[:-1],ttl_dlc,alignment_window,1/150)
snippets_base, alignment = signal2eventsnippets(side_t_s,nancoords_full.implantBase.y.values[:-1],ttl_dlc,alignment_window,1/150)
snippets_snout, alignment = signal2eventsnippets(side_t_s,nancoords_full.snout.y.values[:-1],ttl_dlc,alignment_window,1/150)
snippets_topL, alignment = signal2eventsnippets(side_t_s,nancoords_full.topL.y.values[:-1],ttl_dlc,alignment_window,1/150)
snippets_topR, alignment = signal2eventsnippets(side_t_s,nancoords_full.topR.y.values[:-1],ttl_dlc,alignment_window,1/150)
snippets_earL, alignment = signal2eventsnippets(side_t_s,nancoords_full.earL.y.values[:-1],ttl_dlc,alignment_window,1/150)
snippets_earR, alignment = signal2eventsnippets(side_t_s,nancoords_full.earR.y.values[:-1],ttl_dlc,alignment_window,1/150)

#%%
fig, axs = plt.subplots(1,4, tight_layout = True, figsize = (12,6))
axs[0].imshow(snippets_sleeve, aspect = 'auto', origin = 'lower')
axs[1].imshow(snippets_base, aspect = 'auto', origin = 'lower')
axs[2].imshow(snippets_tdtomato, aspect = 'auto', origin = 'lower')
axs[3].imshow(snippets_DA, aspect = 'auto', origin = 'lower')

axs[0].set_title('implant sleeve')
axs[1].set_title('implant base')
axs[2].set_title('tdtomato')
axs[3].set_title('DA')

#%%

plt.plot(alignment_DA, zscore(np.nanmean(snippets_DA, axis = 0)))
plt.plot(alignment_dlc, zscore(np.nanmean(snippets_base, axis = 0)), color = bodypart_color_dic['implantBase'])
plt.plot(alignment_dlc, zscore(np.nanmean(snippets_sleeve, axis = 0)), color = bodypart_color_dic['implantSleeve'])
#plt.plot(alignment_dlc, zscore(np.nanmean(snippets_snout, axis = 0)), color = bodypart_color_dic['snout'])
#plt.plot(alignment_dlc, zscore(np.nanmean(snippets_topL, axis = 0)), color = bodypart_color_dic['topL'])
#plt.plot(alignment_dlc, zscore(np.nanmean(snippets_topR, axis = 0)), color = bodypart_color_dic['topR'])
#plt.plot(alignment_dlc, zscore(np.nanmean(snippets_earL, axis = 0)), color = bodypart_color_dic['earL'])
#plt.plot(alignment_dlc, zscore(np.nanmean(snippets_earR, axis = 0)), color = bodypart_color_dic['earR'])

#axs[3].imshow(snippets_topL, aspect = 'auto')
plt.xlim(-5,2)

#%%

raw_photometry_ttls = in2.ttl_bool.values
raw_photometry_t = in2.timestamp.values

#%%
ttl_raw_photometry = detect_rising_edge(raw_photometry_ttls, raw_photometry_t)

#%%

## ref trial
plt.plot(jointdf.trial_duration_harp, color = 'black')

#ttl_dlc_dropped = np.delete(ttl_dlc,[0])
#plt.plot(np.diff(ttl_dlc_dropped))

ttl_raw_photometry_dropped = np.delete(ttl_raw_photometry,[0,21,42,63,85])
ttl_frameno_dlc_dropped = np.delete(ttl_frameno_dlc,[0,21,62])

plt.plot(np.diff(ttl_raw_photometry_dropped))
plt.plot(np.diff(ttl_frameno_dlc_dropped)/150)

#%%
plt.plot(ttl_frameno_dlc_dropped,ttl_raw_photometry_dropped,'.')
#%%

dlcDAdf = pd.DataFrame()
dlcDAdf['trialno'] = jointdf.trialno
dlcDAdf['trial_start_harp'] = jointdf.trial_start_harp.values
dlcDAdf['trial_end_harp'] = jointdf.trial_end_harp.values
dlcDAdf['trial_duration_harp'] = jointdf.trial_end_harp - jointdf.trial_start_harp
dlcDAdf['trial_start_frameno'] = ttl_frameno_dlc_dropped[:-1]
dlcDAdf['trial_end_frameno'] = ttl_frameno_dlc_dropped[1:]

split_idx = np.where(np.isin(gpiodf.frame_session.values,ttl_frameno_dlc_dropped ))[0]
dlcDAdf['frameno_session'] = np.split(gpiodf.frame_session.values, split_idx)[1:-1]
dlcDAdf['frame_count'] = dlcDAdf.frameno_session.apply(lambda x: len(x))
dlcDAdf['trial_fps'] = dlcDAdf.apply(lambda x: x.frame_count/x.trial_duration_harp, axis = 1) 
#%%
plt.plot(dlcDAdf.trial_fps)
plt.axhline(150)
plt.ylim(148,152)
plt.title('trial fps (approx 150?)')
#%%

split_idx_harp = np.where(np.isin(downharpdf.timestamp_session.values,dlcDAdf.trial_start_harp.values))[0][1:]

dlcDAdf['denoised_tdtomato'] = np.split(downharpdf.denoised_tdtomato.values, split_idx_harp)
dlcDAdf['denoised_gfp'] = np.split(downharpdf.denoised_gfp.values, split_idx_harp)
#%%
for bodypart in ['implantSleeve', 'implantBase', 'snout', 'topL','poke','lever']:
    coord_dlc_interpolated = nancoords_full[bodypart]['y'].interpolate(method = 'linear').bfill().ffill().values
    dlcDAdf[f'{bodypart}_y'] = np.split(coord_dlc_interpolated,split_idx)[1:-1]

#%%
dlcDAdf['len_tdtomato'] = dlcDAdf.denoised_tdtomato.apply(lambda x: len(x))
dlcDAdf['len_gfp'] = dlcDAdf.denoised_gfp.apply(lambda x: len(x))

plt.plot(dlcDAdf.len_tdtomato == dlcDAdf.len_gfp)
plt.title('should be 1')
#%%
dlcDAdf['timestamp_trial'] = dlcDAdf.apply(lambda x: np.linspace(0,x.trial_duration_harp,x.len_tdtomato), axis = 1)
dlcDAdf['frametimes_trial'] = dlcDAdf.apply(lambda x: np.linspace(0,x.trial_duration_harp,x.frame_count), axis = 1)

for bodypart in ['implantSleeve', 'implantBase', 'snout', 'topL','poke','lever']:
    dlcDAdf[f'{bodypart}_y_upsampled'] = dlcDAdf.apply(lambda x: resample_dlc_to_timebase(x.frametimes_trial, x[f'{bodypart}_y'], x.timestamp_trial), axis = 1)


#%%

## len check
dlcDAdf['len_dlc'] = dlcDAdf.implantBase_y_upsampled.apply(lambda x: len(x))

plt.plot(dlcDAdf.len_tdtomato == dlcDAdf.len_dlc)
plt.title('should be 1')


#%%

"""
.########.##.....##.##.......##..........########..########..######...########..########..######...######..####..#######..##....##
.##.......##.....##.##.......##..........##.....##.##.......##....##..##.....##.##.......##....##.##....##..##..##.....##.###...##
.##.......##.....##.##.......##..........##.....##.##.......##........##.....##.##.......##.......##........##..##.....##.####..##
.######...##.....##.##.......##..........########..######...##...####.########..######....######...######...##..##.....##.##.##.##
.##.......##.....##.##.......##..........##...##...##.......##....##..##...##...##.............##.......##..##..##.....##.##..####
.##.......##.....##.##.......##..........##....##..##.......##....##..##....##..##.......##....##.##....##..##..##.....##.##...###
.##........#######..########.########....##.....##.########..######...##.....##.########..######...######..####..#######..##....##
"""
tdtomato_full = np.hstack(dlcDAdf.denoised_tdtomato.values)
gfp_full = np.hstack(dlcDAdf.denoised_gfp.values)

cols = []
for bodypart in ['implantSleeve', 'implantBase', 'snout', 'topL']:
    cols.append(np.hstack(dlcDAdf[f'{bodypart}_y_upsampled'].values))

cols = np.column_stack(cols)

scaler = StandardScaler()
X = scaler.fit_transform(np.column_stack([tdtomato_full, cols]))
X = sm.add_constant(X)
y = gfp_full

rlm = sm.RLM(y,X, M = sm.robust.norms.HuberT())
res = rlm.fit()

print(res.summary())

#%%
labels = ["intercept", "tdtomato", 'implantSleeve', 'implantBase', 'snout', 'topL']
 
# Extract parameters and CI
params = res.params                  # shape (k,)
conf = res.conf_int()                # shape (k, 2)

# Compute error bars
lower_err = params - conf[:, 0]
upper_err = conf[:, 1] - params
err = np.vstack([lower_err, upper_err])

x = np.arange(len(params)-1)

plt.figure(figsize=(4, 4))
plt.errorbar(
    x, params[1:],
    yerr=err[1:,1:],
    fmt='o', capsize=5
)

plt.axhline(0, color='k', linestyle='--', alpha=0.6)

plt.xticks(x, labels[1:], rotation=45, ha='right')
plt.ylabel("coefficient value")

figtitle = f'{animal} {date} | reg coeffs'
plt.title(figtitle)

plt.tight_layout()
plt.show()


y_pred = res.fittedvalues
raw_residuals = y - res.fittedvalues
dlcDAdf['DA_fullregression'] = np.split(raw_residuals, split_idx_harp)


#%%

"""
.########.....###....########..########.####....###....##..........########..########..######...########..########..######...######..####..#######..##....##..######.
.##.....##...##.##...##.....##....##.....##....##.##...##..........##.....##.##.......##....##..##.....##.##.......##....##.##....##..##..##.....##.###...##.##....##
.##.....##..##...##..##.....##....##.....##...##...##..##..........##.....##.##.......##........##.....##.##.......##.......##........##..##.....##.####..##.##......
.########..##.....##.########.....##.....##..##.....##.##..........########..######...##...####.########..######....######...######...##..##.....##.##.##.##..######.
.##........#########.##...##......##.....##..#########.##..........##...##...##.......##....##..##...##...##.............##.......##..##..##.....##.##..####.......##
.##........##.....##.##....##.....##.....##..##.....##.##..........##....##..##.......##....##..##....##..##.......##....##.##....##..##..##.....##.##...###.##....##
.##........##.....##.##.....##....##....####.##.....##.########....##.....##.########..######...##.....##.########..######...######..####..#######..##....##..######.
"""

#scaler = StandardScaler()
X = scaler.fit_transform(np.column_stack([tdtomato_full]))
X = sm.add_constant(X)

rlm = sm.RLM(y,X, M = sm.robust.norms.HuberT())
res = rlm.fit()

print(res.summary())

y_pred = res.fittedvalues
raw_residuals = y - res.fittedvalues
dlcDAdf['DA_tdtomato'] = np.split(raw_residuals, split_idx_harp)

#%%
X = scaler.fit_transform(np.column_stack([tdtomato_full, cols[:,0]]))
X = sm.add_constant(X)

rlm = sm.RLM(y,X, M = sm.robust.norms.HuberT())
res = rlm.fit()

print(res.summary())

y_pred = res.fittedvalues
raw_residuals = y - res.fittedvalues
dlcDAdf['DA_tdtomato_sleeve'] = np.split(raw_residuals, split_idx_harp)

#%%

X = scaler.fit_transform(np.column_stack([tdtomato_full, cols[:,1]]))
X = sm.add_constant(X)

rlm = sm.RLM(y,X, M = sm.robust.norms.HuberT())
res = rlm.fit()

print(res.summary())

y_pred = res.fittedvalues
raw_residuals = y - res.fittedvalues
dlcDAdf['DA_tdtomato_base'] = np.split(raw_residuals, split_idx_harp)
#%%

X = scaler.fit_transform(np.column_stack([tdtomato_full, cols[:,:2]]))
X = sm.add_constant(X)

rlm = sm.RLM(y,X, M = sm.robust.norms.HuberT())
res = rlm.fit()

print(res.summary())

y_pred = res.fittedvalues
raw_residuals = y - res.fittedvalues
dlcDAdf['DA_tdtomato_implant'] = np.split(raw_residuals, split_idx_harp)
#%%

"""
..#######..##.....##....###....##....##.########.########..########..######..
.##.....##.##.....##...##.##...###...##....##....##.....##.##.......##....##.
.##.....##.##.....##..##...##..####..##....##....##.....##.##.......##.......
.##.....##.##.....##.##.....##.##.##.##....##....########..######...##...####
.##..##.##.##.....##.#########.##..####....##....##...##...##.......##....##.
.##....##..##.....##.##.....##.##...###....##....##....##..##.......##....##.
..#####.##..#######..##.....##.##....##....##....##.....##.########..######..
"""

X = np.column_stack([tdtomato_full, cols])
X = sm.add_constant(X)

y = gfp_full


quantiles = [.01, .5, .99]

models = {}
fits = {}

for q in quantiles:
    mod = sm.QuantReg(y, X)
    res_q = mod.fit(q=q)
    models[q] = mod
    fits[q] = res_q

#%%

qhat_001 = fits[.01].predict(X)
qhat_050 = fits[.5].predict(X)
qhat_099 = fits[.99].predict(X)

#%%
denom = (qhat_099 - qhat_001)
eps = np.finfo(float).eps
denom = np.where(denom < eps, np.nan, denom)

DA_qr = (y - qhat_050) / denom


#%%
dlcDAdf['DA_qr'] = np.split(DA_qr, split_idx_harp)


#%%

dlcDAdf['trial_duration_arduino'] = jointdf.trial_duration_arduino/1000
#%%
dlcDAdf.get(['trial_duration_arduino', 'trial_duration_harp'])

#%%
"""
.########..##........#######..########.########.####.##....##..######..
.##.....##.##.......##.....##....##.......##.....##..###...##.##....##.
.##.....##.##.......##.....##....##.......##.....##..####..##.##.......
.########..##.......##.....##....##.......##.....##..##.##.##.##...####
.##........##.......##.....##....##.......##.....##..##..####.##....##.
.##........##.......##.....##....##.......##.....##..##...###.##....##.
.##........########..#######.....##.......##....####.##....##..######..
""" 

## I think this plotting should be trash
for tt in range(0,len(dlcDAdf)):

    fig, axs = plt.subplots(3, tight_layout = True, sharex = True, figsize = (12,6))

    for lever in bhvdf.lever_rel[tt]/1000:
        for ax in axs:
            ax.axvline(lever, color = 'grey', lw = .5)

    #plt.plot(dlcDAdf.timestamp_trial[tt], zscore(jointdf.DA_poly_session[tt]), lw = .5)
    axs[0].plot(dlcDAdf.timestamp_trial[tt], zscore(dlcDAdf.denoised_tdtomato[tt]), color = 'red', lw = .5, label = 'tdtomato')
    axs[0].plot(dlcDAdf.timestamp_trial[tt], zscore(dlcDAdf.denoised_gfp[tt]), color = 'green', lw = .5, label = 'dlight')

    for bodypart in ['poke', 'lever', 'implantBase', 'implantSleeve', 'snout', 'topL']:
        axs[1].plot(dlcDAdf.timestamp_trial[tt], dlcDAdf[f'{bodypart}_y_upsampled'][tt], color = bodypart_color_dic[bodypart], lw = 1, label = bodypart)

    axs[1].invert_yaxis()

    axs[2].plot(dlcDAdf.timestamp_trial[tt], zscore(dlcDAdf.DA_tdtomato[tt]), lw = 0.5, label = 'robust ~ tdtomato')
    axs[2].plot(dlcDAdf.timestamp_trial[tt], zscore(dlcDAdf.DA_fullregression[tt]), lw = 0.5, label = ' robust ~ tdtomato, dlc all')
    #axs[2].plot(dlcDAdf.timestamp_trial[tt], dlcDAdf.DA_tdtomato_sleeve[tt], lw = 0.5)
    #axs[2].plot(dlcDAdf.timestamp_trial[tt], dlcDAdf.DA_tdtomato_base[tt], lw = 0.5)
    #axs[2].plot(dlcDAdf.timestamp_trial[tt], dlcDAdf.DA_tdtomato_implant[tt], lw = 0.5)
    axs[2].plot(dlcDAdf.timestamp_trial[tt], zscore(dlcDAdf.DA_qr[tt]), lw = 0.5, label = 'quantile norm ~ tdtomato, dlc all')


    axs[0].set_ylabel('raw photometry (z)')
    axs[1].set_ylabel('dlc y (px)')
    axs[2].set_ylabel('DA comparison (z)')

    axs[0].legend(frameon = False, ncols = 2)
    axs[1].legend(frameon = False, ncols = 6)
    axs[2].legend(frameon = False, ncols = 3)

    axs[0].set_xlim(0, dlcDAdf['trial_duration_harp'][tt])

    figtitle = f'{animal} {date} | trial {tt+1} | full session regression with dlc'
    fig.suptitle(figtitle)

    plt.savefig(rf"{PATH_DA_AND_DLC}\full_session_regression\{figtitle.replace('|', '_')}.png", dpi = 300)
    plt.close()

#%%

fig, axs = plt.subplots(1,3, tight_layout = True, figsize = (12,4))
axs[0].plot(dlcDAdf.denoised_tdtomato[tt], dlcDAdf.denoised_gfp[tt], '.')
axs[1].plot(dlcDAdf.denoised_tdtomato[tt], dlcDAdf.DA_fullregression[tt], '.')
axs[2].plot(dlcDAdf.denoised_tdtomato[tt], dlcDAdf.DA_qr[tt], '.')


#%%
"""
.########..########.########..####.##.....##....###....########.####.##.....##.########
.##.....##.##.......##.....##..##..##.....##...##.##......##.....##..##.....##.##......
.##.....##.##.......##.....##..##..##.....##..##...##.....##.....##..##.....##.##......
.##.....##.######...########...##..##.....##.##.....##....##.....##..##.....##.######..
.##.....##.##.......##...##....##...##...##..#########....##.....##...##...##..##......
.##.....##.##.......##....##...##....##.##...##.....##....##.....##....##.##...##......
.########..########.##.....##.####....###....##.....##....##....####....###....########

using the derivative to do the regression

"""

## angle
for bodypart in ['implantSleeve', 'implantBase']:
    coord_dlc_interpolated = nancoords_full[bodypart]['x'].interpolate(method = 'linear').bfill().ffill().values
    dlcDAdf[f'{bodypart}_x'] = np.split(coord_dlc_interpolated,split_idx)[1:-1]

for bodypart in ['implantSleeve', 'implantBase']:
    dlcDAdf[f'{bodypart}_x_upsampled'] = dlcDAdf.apply(lambda x: resample_dlc_to_timebase(x.frametimes_trial, x[f'{bodypart}_x'], x.timestamp_trial), axis = 1)

dy = dlcDAdf.implantSleeve_y_upsampled - dlcDAdf.implantBase_y_upsampled
dx = dlcDAdf.implantSleeve_x_upsampled - dlcDAdf.implantBase_x_upsampled
#%%

tt = 21

pos_trial = dlcDAdf.implantSleeve_y_upsampled[tt][1:]
implant_angle_trial = np.arctan2(dy[tt], dx[tt])
#derivative_angle_38 = np.diff(implant_angle_38)
#%%

derivative_trial = dlcDAdf.implantSleeve_y_upsampled.apply(lambda x: np.diff(x))[tt]
tdtomato_trial = dlcDAdf.denoised_tdtomato[tt][1:]
gfp_trial = dlcDAdf.denoised_gfp[tt][1:]
plt.plot(zscore(pos_trial))
plt.plot(zscore(derivative_trial))
plt.plot(zscore(implant_angle_trial))
#plt.plot(zscore(derivative_angle_38))
plt.plot(5+zscore(tdtomato_trial), color = 'red')
plt.plot(5+zscore(gfp_trial), color = 'green')

#%%

scaler = StandardScaler()
X = scaler.fit_transform(np.column_stack([tdtomato_trial,
                                        pos_trial,
                                        derivative_trial,
                                        implant_angle_trial[1:]]))
X = sm.add_constant(X)
y = gfp_trial

rlm = sm.RLM(y,X, M = sm.robust.norms.HuberT())
res = rlm.fit()

print(res.summary())

#%%

fig, axs = plt.subplots(4, tight_layout = True, figsize = (12,8), sharex = True)

axs[0].plot(zscore(tdtomato_trial), color = 'red', lw = 1)
axs[0].plot(zscore(gfp_trial), color = 'green', lw = 1)
axs[1].plot(zscore(gfp_trial-res.fittedvalues), lw =1)
axs[2].plot(zscore(gfp_trial-quantile_regression(tdtomato_trial,gfp_trial)), lw =1)
axs[3].plot(zscore(dlcDAdf.DA_qr[tt]), lw =1)

axs[0].set_ylabel('raw photometry')
axs[1].set_ylabel('pos, angle, deriv | trial')
axs[2].set_ylabel('no dlc | trial')
axs[3].set_ylabel('dlc | session')

figtitle = f'{animal} {date} | trialno {tt+1}'
plt.suptitle(figtitle)

#%%

labels = ["tdtomato", 'pos', 'derivative', 'angle']
 
params = res.params                  # shape (k,)
conf = res.conf_int()                # shape (k, 2)

lower_err = params - conf[:, 0]
upper_err = conf[:, 1] - params
err = np.vstack([lower_err, upper_err])

x = np.arange(len(params)-1)

plt.figure(figsize=(4, 4))
plt.errorbar(
    x, params[1:],
    yerr=err[1:,1:],
    fmt='o', capsize=5, color = 'darkblue'
)

plt.axhline(0, color='k', linestyle='--', alpha=0.6)

plt.xticks(x, labels, rotation=45, ha='right')
plt.ylabel("coefficient value")

figtitle = f'{animal} {date} | trial {tt+1} | reg coeffs'
plt.title(figtitle, fontsize = 14)

plt.tight_layout()
plt.show()
#%%

"""
.########.##.....##.##.......##...........######..########..######...######..####..#######..##....##
.##.......##.....##.##.......##..........##....##.##.......##....##.##....##..##..##.....##.###...##
.##.......##.....##.##.......##..........##.......##.......##.......##........##..##.....##.####..##
.######...##.....##.##.......##...........######..######....######...######...##..##.....##.##.##.##
.##.......##.....##.##.......##................##.##.............##.......##..##..##.....##.##..####
.##.......##.....##.##.......##..........##....##.##.......##....##.##....##..##..##.....##.##...###
.##........#######..########.########.....######..########..######...######..####..#######..##....##

now using the position, derivative and implant angle as predictors in the full session

"""

# adding the previous implant sleeve position to the start of the array, so that the derivative has matching lenght
dlcDAdf['last_position'] = dlcDAdf.implantSleeve_y_upsampled.apply(lambda x: x[-1])
position_0 = dlcDAdf.implantSleeve_y_upsampled[0][0]
dlcDAdf['previous_last_position'] = np.hstack([position_0, dlcDAdf.last_position.values[:-1]])

dlcDAdf['extra_dim_position'] = dlcDAdf.apply(lambda x: np.hstack([x.previous_last_position, x.implantSleeve_y_upsampled]), axis = 1)

dlcDAdf['implantSleeve_y_deriv'] = dlcDAdf.extra_dim_position.apply(lambda x: np.diff(x))

dlcDAdf['dy'] = dlcDAdf.apply(lambda x: x.implantSleeve_y_upsampled - x.implantBase_y_upsampled, axis = 1)
dlcDAdf['dx'] = dlcDAdf.apply(lambda x: x.implantSleeve_x_upsampled - x.implantBase_x_upsampled, axis = 1)

#%%
def compute_implant_angle(dy, dx):
    return np.arctan2(dy,dx)

dlcDAdf['implant_angle'] = dlcDAdf.apply(lambda x: compute_implant_angle(x.dy, x.dx), axis = 1)

#%%
pos_full = np.hstack(dlcDAdf.implantSleeve_y_upsampled.values)
derivative_full = np.hstack(dlcDAdf.implantSleeve_y_deriv.values)
angle_full = np.hstack(dlcDAdf.implant_angle.values)


#%%
scaler = StandardScaler()
X = scaler.fit_transform(np.column_stack([tdtomato_full[:15000],
                                        pos_full[:15000],
                                        derivative_full[:15000],
                                        angle_full[:15000]]))
X = sm.add_constant(X)
y = gfp_full[:15000]

rlm = sm.RLM(y,X, M = sm.robust.norms.HuberT())
res = rlm.fit()

print(res.summary())
#%%
labels = ["tdtomato", 'pos', 'derivative', 'angle']
 
params = res.params                  # shape (k,)
conf = res.conf_int()                # shape (k, 2)

lower_err = params - conf[:, 0]
upper_err = conf[:, 1] - params
err = np.vstack([lower_err, upper_err])

x = np.arange(len(params)-1)

plt.figure(figsize=(4, 4))
plt.errorbar(
    x, params[1:],
    yerr=err[1:,1:],
    fmt='o', capsize=5, color = 'darkblue'
)

plt.axhline(0, color='k', linestyle='--', alpha=0.6)

plt.xticks(x, labels, rotation=45, ha='right')
plt.ylabel("coefficient value")

figtitle = f'{animal} {date} | trial {tt+1} | reg coeffs'
plt.title(figtitle, fontsize = 14)

plt.tight_layout()
plt.show()

#%%

DA_full_dlc = gfp_full[:15000] - res.fittedvalues
#%%
tt = 2
plt.plot(zscore(np.split(DA_full_dlc,split_idx_harp)[tt]), lw = 1, alpha = 0.5, label = 'with dlc')
plt.plot(zscore(dlcDAdf.DA_tdtomato_base[tt]), lw = 1, alpha = 0.5, label = 'without dlc')
plt.legend()
#%%

plt.plot(DA_full_dlc)
plt.plot(-1+np.hstack(dlcDAdf.DA_tdtomato_base.values)[:15000])
#%%


## in the limit we do trial by trial and then patch, and compare against the original formulation (for )
def do_dlc_regression(regressor_cols, y_col):
    scaler = StandardScaler()
    X = scaler.fit_transform(np.column_stack(regressor_cols))

    X = sm.add_constant(X)
    y = y_col

    rlm = sm.RLM(y,X, M = sm.robust.norms.HuberT())
    res = rlm.fit()

    residuals = y  - res.fittedvalues
    betas = res.params
    conf_ints = res.conf_int()

    return residuals, betas, conf_ints

#%%

dlcDAdf['DA_dlc_trial_all'] = dlcDAdf.apply(lambda x: do_dlc_regression([x.denoised_tdtomato,
                                           x.implantSleeve_y_upsampled,
                                           x.implantSleeve_y_deriv,
                                           x.implant_angle],
                                           x.denoised_gfp), axis = 1)

dlcDAdf['DA_dlc_trial'] = dlcDAdf.DA_dlc_trial_all.apply(lambda x: x[0])
dlcDAdf['DA_dlc_trial_betas'] = dlcDAdf.DA_dlc_trial_all.apply(lambda x: x[1])
dlcDAdf['DA_dlc_trial_conf_ints'] = dlcDAdf.DA_dlc_trial_all.apply(lambda x: x[2])
#%%

plt.axhline(0, color = 'grey', ls = '--')
for ii in range(len(dlcDAdf)):
    plt.plot(dlcDAdf.DA_dlc_trial_betas[ii][1:], '.')
#%%

"""
.########..##........#######..########
.##.....##.##.......##.....##....##...
.##.....##.##.......##.....##....##...
.########..##.......##.....##....##...
.##........##.......##.....##....##...
.##........##.......##.....##....##...
.##........########..#######.....##...
"""

for tt in range(len(dlcDAdf)):

    fig, axs = plt.subplots(4, tight_layout = True, figsize = (12,8), sharex = True)


    for lever in bhvdf.lever_rel[tt]/1000:
        for ax in axs:
            ax.axvline(lever, color = 'grey', lw = .5)

    axs[0].plot(dlcDAdf.timestamp_trial[tt], zscore(dlcDAdf.denoised_tdtomato[tt]), color = 'red', lw = .5, label = 'tdtomato')
    axs[0].plot(dlcDAdf.timestamp_trial[tt], zscore(dlcDAdf.denoised_gfp[tt]), color = 'green', lw = .5, label = 'dlight')
    for bodypart in ['poke', 'lever', 'implantBase', 'implantSleeve', 'snout', 'topL']:
        axs[1].plot(dlcDAdf.timestamp_trial[tt], dlcDAdf[f'{bodypart}_y_upsampled'][tt], color = bodypart_color_dic[bodypart], lw = 1, label = bodypart)
    axs[1].invert_yaxis()

    axs[0].plot(dlcDAdf.timestamp_trial[tt], zscore(dlcDAdf.implant_angle[tt]), lw =1, color = 'grey', label = 'implant angle')


    axs[2].plot(dlcDAdf.timestamp_trial[tt], zscore(dlcDAdf.DA_dlc_trial[tt]), lw =.5, label = 'dlc trial')
    axs[2].plot(dlcDAdf.timestamp_trial[tt], zscore(dlcDAdf.denoised_gfp[tt]-quantile_regression(dlcDAdf.denoised_tdtomato[tt], dlcDAdf.denoised_gfp[tt])), lw =.5, label = 'no dlc trial')
    axs[3].plot(dlcDAdf.timestamp_trial[tt], zscore(dlcDAdf.DA_qr[tt]), lw =.5, label = 'session dlc')
    axs[3].plot(dlcDAdf.timestamp_trial[tt], zscore(dlcDAdf.denoised_gfp[tt]-quantile_regression(dlcDAdf.denoised_tdtomato[tt], dlcDAdf.denoised_gfp[tt])), lw =.5, label = 'trial no dlc')

    axs[0].set_ylabel('raw photometry')
    axs[1].set_ylabel('dlc')
    axs[2].set_ylabel('dlc vs no dlc | trial')
    axs[3].set_ylabel('session vs trial')

    betas = dlcDAdf.DA_dlc_trial_betas[tt]
    axs[2].text(0.5, 5, f'predictors: tdtomato {np.round(betas[1],3)}, implantSleeve pos {np.round(betas[2],3)}, deriv {np.round(betas[3],3)}, implant angle {np.round(betas[-1],3)}')


    axs[0].legend(frameon = False, ncols = 3)
    axs[1].legend(frameon = False, ncols = 6)
    axs[2].legend(frameon = False, ncols = 2)
    axs[3].legend(frameon = False, ncols = 2)

    axs[0].set_xlim(0, dlcDAdf['trial_duration_harp'][tt])
    figtitle = f'{animal} {date} | trial {tt+1} | trial vs full session | dlc vs no dlc'
    fig.suptitle(figtitle)

    plt.savefig(rf"{PATH_DA_AND_DLC}\session_vs_trial\{figtitle.replace('|', '_')}.png", dpi = 300)
    plt.close()

#%%

plt.plot(np.hstack(dlcDAdf.DA_dlc_trial.values),np.hstack(dlcDAdf.DA_tdtomato_base.values), '.')

#%%
tt = 18
plt.plot(dlcDAdf.DA_dlc_trial[tt], label = 'trial with dlc')
plt.plot(-.2+dlcDAdf.DA_tdtomato_base[tt], label = 'session without dlc')
plt.legend()

#%%
all_new_DA = np.hstack(dlcDAdf.DA_dlc_trial.values)

time_all_session = np.linspace(dlcDAdf.trial_start_harp.values[0], dlcDAdf.trial_end_harp.values[-1], len(all_new_DA))
snippsDA, _ = signal2eventsnippets(time_all_session, all_new_DA, dlcDAdf.trial_start_harp.values, [-2,2], .01)


#%%


"""
..######..##.......####.########..####.##....##..######......##......##.####.##....##.########...#######..##......##
.##....##.##........##..##.....##..##..###...##.##....##.....##..##..##..##..###...##.##.....##.##.....##.##..##..##
.##.......##........##..##.....##..##..####..##.##...........##..##..##..##..####..##.##.....##.##.....##.##..##..##
..######..##........##..##.....##..##..##.##.##.##...####....##..##..##..##..##.##.##.##.....##.##.....##.##..##..##
.......##.##........##..##.....##..##..##..####.##....##.....##..##..##..##..##..####.##.....##.##.....##.##..##..##
.##....##.##........##..##.....##..##..##...###.##....##.....##..##..##..##..##...###.##.....##.##.....##.##..##..##
..######..########.####.########..####.##....##..######.......###..###..####.##....##.########...#######...###..###.

start with a sliding window of 120 seconds
step size: 1s
"""

window_size_s = 120
step_size_s = 1
fs = 100

window_size_i = window_size_s * fs
step_size_i = step_size_s * fs

total_len = len(tdtomato_full)

all_starts = np.arange(
    0, 
    total_len - window_size_i + 1, 
    step_size_i
).astype(int)

final_dopamine_signal = np.zeros(total_len)

print(f"Total original length: {total_len}")
print(f"Total full windows to fit: {len(all_starts)}")

last_filled_index = 0

for i, ss in enumerate(all_starts):
    ee = ss + window_size_i
    
    X_window = np.column_stack([
        tdtomato_full[ss:ee], pos_full[ss:ee],
        derivative_full[ss:ee], angle_full[ss:ee]
    ])
    y_window = gfp_full[ss:ee] 

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_window)
    X_scaled = sm.add_constant(X_scaled)
    
    rlm = sm.RLM(y_window, X_scaled, M=sm.robust.norms.HuberT())
    res = rlm.fit()
    residuals = y_window - res.fittedvalues

    if i == 0:
        segment_to_save = residuals
    else:
        segment_to_save = residuals[-step_size_i:]
        
    insert_start_index = ss + (window_size_i - len(segment_to_save))
    insert_end_index = ss + window_size_i
    
    final_dopamine_signal[insert_start_index:insert_end_index] = segment_to_save
    last_filled_index = insert_end_index


if last_filled_index < total_len:
    
    final_start = last_filled_index - window_size_i + step_size_i
    final_end = total_len

    last_full_window_start = all_starts[-1]
    last_full_window_end = last_full_window_start + window_size_i
    
    X_window = np.column_stack([
        tdtomato_full[last_full_window_start:last_full_window_end],
        pos_full[last_full_window_start:last_full_window_end],
        derivative_full[last_full_window_start:last_full_window_end],
        angle_full[last_full_window_start:last_full_window_end]
    ])
    y_window = gfp_full[last_full_window_start:last_full_window_end] 

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_window)
    X_scaled = sm.add_constant(X_scaled)
    
    rlm = sm.RLM(y_window, X_scaled, M=sm.robust.norms.HuberT())
    res = rlm.fit()
    
    X_remaining = np.column_stack([
        tdtomato_full[last_filled_index:final_end],
        pos_full[last_filled_index:final_end],
        derivative_full[last_filled_index:final_end],
        angle_full[last_filled_index:final_end]
    ])
    y_remaining = gfp_full[last_filled_index:final_end]
    
    X_remaining_scaled = scaler.transform(X_remaining)
    X_remaining_scaled = sm.add_constant(X_remaining_scaled)

    predicted_values = np.dot(X_remaining_scaled, res.params)
    
    final_segment_residuals = y_remaining - predicted_values
    
    final_dopamine_signal[last_filled_index:final_end] = final_segment_residuals

print(f"Final dopamine signal length after correction: {len(final_dopamine_signal)}")

#%%

snipps, _ = signal2eventsnippets(time_all_session, final_dopamine_signal, dlcDAdf.trial_start_harp.values, [-2,2], .01)

#%%
plt.imshow(snipps, aspect = 'auto')
plt.show()
plt.plot(np.nanmean(snipps, axis = 0))
plt.plot(np.nanmean(snippsDA, axis = 0))
#plt.plot(np.nanmedian(snipps, axis = 0))


#%%

"""
.########..##........######..########.....###....########..########
.##.....##.##.......##....##.##.....##...##.##...##.....##.##......
.##.....##.##.......##.......##.....##..##...##..##.....##.##......
.##.....##.##.......##.......##.....##.##.....##.##.....##.######..
.##.....##.##.......##.......##.....##.#########.##.....##.##......
.##.....##.##.......##....##.##.....##.##.....##.##.....##.##......
.########..########..######..########..##.....##.########..##......
"""

dlcDAdf['DA_movingwindows'] = np.split(final_dopamine_signal, split_idx_harp)
#%%
tt = 42
plt.figure(figsize = (12,4))
plt.plot(6+zscore(dlcDAdf.DA_movingwindows[tt]), lw = .5, label = 'dlc moving window')
plt.plot(zscore(dlcDAdf.DA_qr[tt]), lw = .5, label = 'session qr')
plt.legend()
#%%
"""
.########..##........#######..########
.##.....##.##.......##.....##....##...
.##.....##.##.......##.....##....##...
.########..##.......##.....##....##...
.##........##.......##.....##....##...
.##........##.......##.....##....##...
.##........########..#######.....##...
"""

for tt in range(len(dlcDAdf)):

    fig, axs = plt.subplots(4, tight_layout = True, figsize = (12,8), sharex = True)


    for lever in bhvdf.lever_rel[tt]/1000:
        for ax in axs:
            ax.axvline(lever, color = 'grey', lw = .5)

    axs[0].plot(dlcDAdf.timestamp_trial[tt], zscore(dlcDAdf.denoised_tdtomato[tt]), color = 'red', lw = .5, label = 'tdtomato')
    axs[0].plot(dlcDAdf.timestamp_trial[tt], zscore(dlcDAdf.denoised_gfp[tt]), color = 'green', lw = .5, label = 'dlight')
    for bodypart in ['poke', 'lever', 'implantBase', 'implantSleeve', 'snout', 'topL']:
        axs[1].plot(dlcDAdf.timestamp_trial[tt], dlcDAdf[f'{bodypart}_y_upsampled'][tt], color = bodypart_color_dic[bodypart], lw = 1, label = bodypart)
    axs[1].invert_yaxis()

    axs[0].plot(dlcDAdf.timestamp_trial[tt], zscore(dlcDAdf.implant_angle[tt]), lw =1, color = 'grey', label = 'implant angle')

    axs[2].plot(dlcDAdf.timestamp_trial[tt], zscore(dlcDAdf.DA_dlc_trial[tt]), lw =.5, label = 'trial dlc')
    axs[2].plot(dlcDAdf.timestamp_trial[tt], zscore(dlcDAdf.DA_movingwindows[tt]), lw =.5, label = 'moving window dlc')
    axs[3].plot(dlcDAdf.timestamp_trial[tt], zscore(dlcDAdf.DA_qr[tt]), lw =.5, label = 'session dlc')
    axs[3].plot(dlcDAdf.timestamp_trial[tt], zscore(dlcDAdf.DA_movingwindows[tt]), lw =.5, label = 'moving window dlc')

    axs[0].set_ylabel('raw photometry')
    axs[1].set_ylabel('dlc')
    axs[2].set_ylabel('trial comparison')
    axs[3].set_ylabel('session comparison')

    axs[0].legend(frameon = False, ncols = 3)
    axs[1].legend(frameon = False, ncols = 6)
    axs[2].legend(frameon = False, ncols = 2)
    axs[3].legend(frameon = False, ncols = 2)

    axs[0].set_xlim(0, dlcDAdf['trial_duration_harp'][tt])
    figtitle = f'{animal} {date} | trial {tt+1} | moving windows regression | vs trial or vs session'
    fig.suptitle(figtitle)

    plt.savefig(rf"{PATH_DA_AND_DLC}\sliding_window_regression\{figtitle.replace('|', '_')}.png", dpi = 300)
    plt.close()
# %%

bhvdf['cp_abs'] = bhvdf.cp+bhvdf.trial_start
dlcDAdf['cp_abs'] = dlcDAdf.trial_start_harp + bhvdf.cp
dlcDAdf['lever_abs'] = dlcDAdf.trial_start_harp + bhvdf.lever_rel/1000
dlcDAdf['rwd_lever_abs'] = dlcDAdf.lever_abs.apply(lambda x: x[-1])
dlcDAdf['nonrwd_lever_abs'] = dlcDAdf.lever_abs.apply(lambda x: x[x!=x[-1]])

#%%

snipps_cp, alignment_cp = signal2eventsnippets(time_all_session, final_dopamine_signal, dlcDAdf.cp_abs.dropna().values, [-10,10], .01)

# %%
plt.plot(alignment_cp, np.nanmean(snipps_cp, axis = 0))
plt.axvline(0, color = 'grey', ls = 'dashed')
# %%

snipps_all_lvrs, _ = signal2eventsnippets(time_all_session, final_dopamine_signal,
                                          np.hstack(dlcDAdf.lever_abs.values), [-2,2], .01)
snipps_rwd_lvrs, _ = signal2eventsnippets(time_all_session, final_dopamine_signal,
                                          np.hstack(dlcDAdf.rwd_lever_abs.values), [-2,2], .01)
snipps_nonrwd_lvrs, _ = signal2eventsnippets(time_all_session, final_dopamine_signal,
                                          np.hstack(dlcDAdf.nonrwd_lever_abs.values), [-2,2], .01)

#%%
#plt.plot(np.nanmean(snipps_all_lvrs, axis = 0))
plt.plot(np.nanmean(snipps_rwd_lvrs, axis = 0))
plt.plot(np.nanmean(snipps_nonrwd_lvrs, axis = 0))

# %%
tt = 37

fig, axs = plt.subplots(figsize = (12,4))

for lever in bhvdf.lever_rel[tt]:
    plt.axvline(lever/10, color = 'grey')

plt.plot(dlcDAdf.implantSleeve_y_upsampled[tt])
plt.plot(dlcDAdf.topL_y_upsampled[tt])
plt.plot(dlcDAdf.lever_y_upsampled[tt])
plt.plot(dlcDAdf.poke_y_upsampled[tt])

plt.plot(1000-100*zscore(dlcDAdf.DA_movingwindows[tt]))

axs.invert_yaxis()
#plt.xlim(2000)
#%%
"""
..######.....###....##.....##.########....########..##........######..########.....###....########..########
.##....##...##.##...##.....##.##..........##.....##.##.......##....##.##.....##...##.##...##.....##.##......
.##........##...##..##.....##.##..........##.....##.##.......##.......##.....##..##...##..##.....##.##......
..######..##.....##.##.....##.######......##.....##.##.......##.......##.....##.##.....##.##.....##.######..
.......##.#########..##...##..##..........##.....##.##.......##.......##.....##.#########.##.....##.##......
.##....##.##.....##...##.##...##..........##.....##.##.......##....##.##.....##.##.....##.##.....##.##......
..######..##.....##....###....########....########..########..######..########..##.....##.########..##......
"""

dlcDAdf.to_pickle(rf'{PATH_STORE_PHOTOMETRY_PICKLES}/{animal}_{date}_dlcDAdf.pkl')



#%%

#%%

"""
.########..########....###....##..........########..####.########...#######.
.##.....##.##.........##.##...##..........##.....##..##..##.....##.##.....##
.##.....##.##........##...##..##..........##.....##..##..##.....##.......##.
.########..######...##.....##.##..........##.....##..##..########......###..
.##...##...##.......#########.##..........##.....##..##..##...........##....
.##....##..##.......##.....##.##..........##.....##..##..##.................
.##.....##.########.##.....##.########....########..####.##...........##....

compare lever presses that happen after rearing up vs when the animal is already up
if the "dip" is present in both, it points towards this being a feature
"""
## by hand for trial 38 (tt=37) Niobium 250624

lever_index_alreadyup = [1,2,3,4,5,6,7,10,11,12,13,15,16,17,18,19,20,21,22,23,24,25,27]
lever_index_afterrearing = [0,8,9,14,26]
#%%
len(bhvdf.lever_rel[tt])
#%%


snipps_alreadyup, _ = signal2eventsnippets(time_all_session, final_dopamine_signal,
                                          np.hstack(dlcDAdf.nonrwd_lever_abs[tt][lever_index_alreadyup]), [-2,2], .01)

snipps_afterrearing, _ = signal2eventsnippets(time_all_session, final_dopamine_signal,
                                          np.hstack(dlcDAdf.nonrwd_lever_abs[tt][lever_index_afterrearing]), [-2,2], .01)


#%%
plt.plot(np.nanmean(snipps_afterrearing, axis = 0))
plt.plot(np.nanmean(snipps_alreadyup, axis = 0))



# %%
plt.plot(dlcDAdf.implantSleeve_y_deriv[tt])
plt.axhline(10)
# %%
dlcDAdf.implantSleeve_y_deriv[tt] == 10
# %%

"""
....###....##.......########.########.########..##....##....###....########.####.##.....##.########
...##.##...##..........##....##.......##.....##.###...##...##.##......##.....##..##.....##.##......
..##...##..##..........##....##.......##.....##.####..##..##...##.....##.....##..##.....##.##......
.##.....##.##..........##....######...########..##.##.##.##.....##....##.....##..##.....##.######..
.#########.##..........##....##.......##...##...##..####.#########....##.....##...##...##..##......
.##.....##.##..........##....##.......##....##..##...###.##.....##....##.....##....##.##...##......
.##.....##.########....##....########.##.....##.##....##.##.....##....##....####....###....########

try to regress out the dlc from tdtomato and from gfp first, and then use the regression for DA

(I don't really think this makes sense)
"""

animal = 'Niobium'
date = '250617'
dlcDAdf = pd.read_pickle(rf'{PATH_STORE_PHOTOMETRY_PICKLES}/{animal}_{date}_dlcDAdf.pkl')

# %%

tdtomato_full = np.hstack(dlcDAdf.denoised_tdtomato.values)
gfp_full = np.hstack(dlcDAdf.denoised_gfp.values)

# %%
angle_full = np.hstack(dlcDAdf.implant_angle.values)
# %%

do_dlc_regression(np.vstack([tdtomato_full,
                             angle_full]),
                             gfp_full)

#%%

tdtomato_minus_angle = tdtomato_full - quantile_regression(angle_full, tdtomato_full)
# %%
gfp_minus_angle = gfp_full - quantile_regression(angle_full, gfp_full)

# %%
plt.plot(tdtomato_minus_angle)
plt.plot(gfp_minus_angle)
plt.xlim(0,10000)
# %%
DA_minus_angle = gfp_minus_angle - quantile_regression(tdtomato_minus_angle, gfp_minus_angle)
# %%
plt.plot(zscore(DA_minus_angle), lw = 0.5)
plt.plot(-2+zscore(np.hstack(dlcDAdf.DA_qr.values)), lw = 0.5)
plt.xlim(60*100,130*100)
# %%
