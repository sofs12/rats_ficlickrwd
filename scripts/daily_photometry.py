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
from scipy.stats import zscore


from ratcode.config.paths import PATH_STORE_PICKLES, DROPBOX_TASK_PATH
from ratcode.common.logging import determine_experiment
from ratcode.common.colorcodes import *
from ratcode.behavior import change_point
from ratcode.photometry.photometry import get_prediction, quantile_regression, signal2eventsnippets, find_poly, segment_and_fit_function, butter_filter, mask_jumps
from ratcode.common.dataframe import group_and_listify
from ratcode.common.time import convert_date_bonsai, convert_timestamp

from ratcode.init import setup
setup()
# %%

animal = 'Rhodium'
date = '260204'

# %%

PHOTOMETRY_PATH = os.path.join(DROPBOX_TASK_PATH, 'photometry', animal)

PATH_SAVE_FIGS = os.path.join(DROPBOX_TASK_PATH, 'analysis_photometry', f'{animal}_{date}')
if not os.path.exists(PATH_SAVE_FIGS):
    os.makedirs(PATH_SAVE_FIGS)

#PATH_SAVE_ICA = os.path.join(DROPBOX_TASK_PATH, 'analysis_photometry', '00_all_sessions_ICA_snippets')


bhv_pkl = glob.glob(rf"{DROPBOX_TASK_PATH}\analysis\{animal}_{date}_*.pkl")[0]
bhvdf = pd.read_pickle(bhv_pkl)

bhvdf['cp'] = bhvdf.apply(lambda x: change_point.accepted_cp_Gallistel(x.trialno, 2, bhvdf, 'lever_rel', True)[0] if len(x.lever_rel)> 0 else np.nan, axis = 1)
bhvdf['cp'] = bhvdf.apply(lambda x: change_point.validate_cp(x.cp, x.lever_rel) if len(x.lever_rel) > 0 else np.nan, axis = 1)

bhvdf['bool_cp'] = np.isnan(bhvdf.cp.values) == False

bhvdf.drop(bhvdf.query('trial_duration < 200').index, inplace = True)
bhvdf.reset_index(drop = True, inplace = True)
bhvdf['trialno'] = bhvdf.index + 1

for file in os.listdir(PHOTOMETRY_PATH):

    if convert_date_bonsai(date) in file:
        if "in0" in file:
            in0_path = os.path.join(PHOTOMETRY_PATH, file)
        if "in1" in file:
            in1_path = os.path.join(PHOTOMETRY_PATH, file)
        if "in2" in file:
            in2_path = os.path.join(PHOTOMETRY_PATH, file)
        #if "in3" in file: ## encoder
        #    in3_path = os.path.join(photometry_path, file)

in0 = pd.read_csv(in0_path, header = None)
in0.columns = ['in0', 'timestamp0']

in1 = pd.read_csv(in1_path, header = None)
in1.columns = ['in1', 'timestamp1']

in2 = pd.read_csv(in2_path, header = None)
in2.columns = ['in2', 'timestamp2']

## this is the encoder
#in3 = pd.read_csv(in3_path, header = None)
#in3.columns = ['in3', 'timestamp3']

harpdf = pd.concat([in0, in1, in2], axis = 1)
harpdf['timestamp_comp'] = (harpdf.timestamp0 == harpdf.timestamp1)*(harpdf.timestamp0 == harpdf.timestamp2)
harpdf.drop(harpdf.query('timestamp_comp == False').index, inplace=True)
harpdf.rename(columns = {'timestamp1': 'timestamp'}, inplace=True)
harpdf.drop(['timestamp0', 'timestamp_comp', 'timestamp2'], axis = 1, inplace=True)

harpdf['tdtomato'] = harpdf.in0/2**16*20
harpdf['gfp'] = harpdf.in1/2**16*20
harpdf['gpio'] = harpdf.in2/2**16*20

#harpdf['encoder'] = in3.in3/2**16*20

harpdf['ttl_bool'] = harpdf.gpio.apply(lambda x: int(x>1))
harpdf['diff_ttl'] = harpdf.ttl_bool.diff()

harpdf['ttl_rising_edge'] = harpdf.ttl_bool*harpdf.diff_ttl > 0
harpdf['timestamp_session'] = harpdf.timestamp - harpdf.timestamp[0]

harpdf['trialno'] = harpdf.ttl_rising_edge.cumsum() - 1
harpdf.drop(harpdf.query('trialno < 1').index, inplace=True)
#in the behaviour I always drop the last trial
harpdf.drop(harpdf.query(f'trialno == {harpdf.trialno.max()}').index, inplace=True)

print(f'total # trials: {harpdf.trialno.values[-1]}')
# %%

start = 0
end = -1

gpio_offset = np.min([np.mean(harpdf.gfp.values),np.mean(harpdf.tdtomato.values)])

plt.figure()

plt.plot(harpdf.timestamp_session[start:end], harpdf.gpio[start:end]/5+gpio_offset, color = 'grey', alpha = 0.5)

plt.plot(harpdf.timestamp_session[start:end], harpdf.tdtomato[start:end]-.2, color = 'red')
plt.plot(harpdf.timestamp_session[start:end], harpdf.gfp[start:end], color = 'green')

#plt.plot(harpdf.timestamp_session[start:end], harpdf.encoder[start:end], color = 'blue', alpha = 0.5)

plt.xlabel('t (s)')
plt.ylabel('V')

plt.title(f'{animal}_{date}')
plt.show()

#%%
tt = 4

tomato = harpdf.query(f'trialno == {tt}').tdtomato.values
gfp = harpdf.query(f'trialno == {tt}').gfp.values

prediction = get_prediction(tomato,gfp)

time = np.arange(len(tomato))*.001

fig, axs = plt.subplots(2, figsize = (6,4), tight_layout = True, sharex = True)
axs[0].plot(time, zscore(tomato), color = 'red', lw = .5)
#axs[0].plot(prediction, color = 'grey', lw = 1)
axs[0].plot(time, zscore(gfp), color = 'green', lw = .5)
axs[1].plot(time, zscore(gfp-prediction), color = 'purple', lw = .5)

axs[-1].set_xlabel('time in trial (s)')

fig.suptitle(f'{animal} {date} | trial {tt}')
# %%


"""
.########...#######..##......##.##....##..######.....###....##.....##.########..##.......########
.##.....##.##.....##.##..##..##.###...##.##....##...##.##...###...###.##.....##.##.......##......
.##.....##.##.....##.##..##..##.####..##.##........##...##..####.####.##.....##.##.......##......
.##.....##.##.....##.##..##..##.##.##.##..######..##.....##.##.###.##.########..##.......######..
.##.....##.##.....##.##..##..##.##..####.......##.#########.##.....##.##........##.......##......
.##.....##.##.....##.##..##..##.##...###.##....##.##.....##.##.....##.##........##.......##......
.########...#######...###..###..##....##..######..##.....##.##.....##.##........########.########
"""

downsample_factor = 10
fs = 1000 #sampling frequency
fs = fs/downsample_factor
nyquist = 0.5 * fs

harpdf['ds_tdtomato'] = harpdf['tdtomato'].rolling(2 * downsample_factor, center=True, min_periods=1).mean()
harpdf['ds_gfp'] = harpdf['gfp'].rolling(2 * downsample_factor, center=True, min_periods=1).mean()

downharpdf = harpdf.iloc[::downsample_factor].reset_index(drop = True)
# %%

#lowpass filter to remove the high freq noise
high_cutoff = 20

downharpdf['denoised_tdtomato'] = butter_filter(downharpdf.ds_tdtomato, high_cutoff, fs, 'low') 
downharpdf['denoised_gfp'] = butter_filter(downharpdf.ds_gfp, high_cutoff, fs, 'low')
#%%

## jumps
fig, axs = plt.subplots(1,3, figsize = (16,4), tight_layout = True)

cutoffs = [.01, .0015, 0.001]

for ii in range(3):
    cutoff = cutoffs[ii]
    axs[ii].plot(downharpdf.ds_tdtomato, color = 'red')
    axs[ii].plot(butter_filter(downharpdf.ds_tdtomato, cutoff, fs, 'low'), color = 'black')
    axs[ii].plot(downharpdf.ds_gfp, color = 'green')
    axs[ii].plot(butter_filter(downharpdf.ds_gfp, cutoff, fs, 'low'), color = 'black')
    axs[ii].set_title(cutoffs[ii])
#%%
jump_threshold_tdtomato = 4
jump_threshold_gfp = 3.5

fig, axs = plt.subplots(2)
axs[1].plot(np.abs(zscore(np.diff(butter_filter(downharpdf.ds_gfp, 0.01, fs, 'low')))))
axs[0].plot(np.abs(zscore(np.diff(butter_filter(downharpdf.ds_tdtomato, 0.01, fs, 'low')))))
axs[1].plot(downharpdf.ds_gfp-3, color = 'green', alpha = 0.5)
axs[0].plot(downharpdf.ds_tdtomato-4, color = 'red', alpha = 0.5)

axs[0].axhline(jump_threshold_tdtomato)
axs[1].axhline(jump_threshold_gfp)

#%%

plt.figure()
plt.plot(downharpdf.ds_tdtomato)
plt.plot(downharpdf.ds_gfp)
#%%

np.where(np.abs(zscore(np.diff(butter_filter(downharpdf.ds_gfp, 0.01, fs, 'low'))))>5)
#%%
## for when I want to manually delete bad data
#plt.plot(downharpdf.ds_tdtomato[:122000])
#plt.plot(downharpdf.ds_gfp[:89226])
downharpdf.loc[89226:, 'ds_tdtomato'] = np.nan
downharpdf.loc[89226:, 'ds_gfp'] = np.nan
#%%

#if a channel saturates

downharpdf.loc[downharpdf.query('ds_gfp > 9.99').ds_gfp.index, 'ds_gfp'] = np.nan

#%%

##USING POLY
downharpdf['poly_tdtomato'] = segment_and_fit_function(downharpdf.timestamp_session.values, mask_jumps(downharpdf.ds_tdtomato, thres = jump_threshold_tdtomato), function = 'poly')
#downharpdf['poly_tdtomato'] = segment_and_fit_function(downharpdf.timestamp_session.values, mask_jumps(downharpdf.ds_gfp, thres = jump_threshold_gfp), function = 'poly')
downharpdf['poly_gfp'] = segment_and_fit_function(downharpdf.timestamp_session.values, mask_jumps(downharpdf.ds_gfp, thres = jump_threshold_gfp), function = 'poly')

downharpdf['tdtomato_poly_flat'] = downharpdf.ds_tdtomato - downharpdf.poly_tdtomato
downharpdf['gfp_poly_flat'] = downharpdf.ds_gfp - downharpdf.poly_gfp

downharpdf['clean_poly_tdtomato'] = downharpdf.tdtomato_poly_flat + np.mean(downharpdf.ds_tdtomato)
downharpdf['clean_poly_gfp'] = downharpdf.gfp_poly_flat + np.mean(downharpdf.ds_gfp)

#define the baseline as the 10th percentile
F0_tdtomato = np.nanquantile(downharpdf.clean_poly_tdtomato,.1)
F0_gfp = np.nanquantile(downharpdf.clean_poly_gfp,.1)

downharpdf['deltaF_poly_tdtomato'] = (downharpdf.clean_poly_tdtomato - F0_tdtomato)/F0_tdtomato
downharpdf['deltaF_poly_gfp'] = (downharpdf.clean_poly_gfp - F0_gfp)/F0_gfp

downharpdf['predicted_poly_gfp_session'] = get_prediction(downharpdf.deltaF_poly_tdtomato, downharpdf.deltaF_poly_gfp)
downharpdf['DA_poly_session'] = downharpdf.deltaF_poly_gfp - downharpdf.predicted_poly_gfp_session

downharpdf['tdtomato_poly_flat'] = downharpdf.ds_tdtomato - downharpdf.poly_tdtomato
downharpdf['gfp_poly_flat'] = downharpdf.ds_gfp - downharpdf.poly_gfp

#%%


fig, axs = plt.subplots(3, figsize = (12,6), tight_layout = True, sharex = True)

axs[0].plot(downharpdf.timestamp_session, downharpdf.ds_tdtomato, color = 'red')
axs[0].plot(downharpdf.timestamp_session, downharpdf.poly_tdtomato, color = 'white')
axs[0].plot(downharpdf.timestamp_session, downharpdf.ds_gfp, color = 'green')
axs[0].plot(downharpdf.timestamp_session, downharpdf.poly_gfp, color = 'white')

axs[1].plot(downharpdf.timestamp_session, downharpdf.tdtomato_poly_flat, color = 'red')
axs[1].plot(downharpdf.timestamp_session, downharpdf.gfp_poly_flat, color = 'green')

axs[2].plot(downharpdf.timestamp_session, downharpdf.DA_poly_session, color = 'purple')

axs[0].set_ylabel('ds data and fits')
axs[1].set_ylabel('flattened = ds - fit')
axs[2].set_ylabel('DA')

figtitle = f'{animal} {date} | exp {determine_experiment(bhvdf)} | DA from robust regression | signals flattened via 3rd order polynomial'
fig.suptitle(figtitle)


# %%
## USING EXPONENTIAL

#downharpdf['exp_tdtomato'] = segment_and_fit_function(downharpdf.timestamp_session.values, mask_jumps(downharpdf.ds_tdtomato, thres = jump_threshold_tdtomato), function = 'exp')
#downharpdf['exp_gfp'] = segment_and_fit_function(downharpdf.timestamp_session.values, mask_jumps(downharpdf.ds_gfp, thres = jump_threshold_gfp), function = 'exp')
#
#downharpdf['tdtomato_exp_flat'] = downharpdf.ds_tdtomato - downharpdf.exp_tdtomato
#downharpdf['gfp_exp_flat'] = downharpdf.ds_gfp - downharpdf.exp_gfp
#
#downharpdf['clean_exp_tdtomato'] = downharpdf.tdtomato_exp_flat + np.mean(downharpdf.ds_tdtomato)
#downharpdf['clean_exp_gfp'] = downharpdf.gfp_exp_flat + np.mean(downharpdf.ds_gfp)
#
##define the baseline as the 10th percentile
#F0_tdtomato = np.nanquantile(downharpdf.clean_exp_tdtomato,.1)
#F0_gfp = np.nanquantile(downharpdf.clean_exp_gfp,.1)
#
#downharpdf['deltaF_exp_tdtomato'] = (downharpdf.clean_exp_tdtomato - F0_tdtomato)/F0_tdtomato
#downharpdf['deltaF_exp_gfp'] = (downharpdf.clean_exp_gfp - F0_gfp)/F0_gfp
#
#downharpdf['predicted_exp_gfp_session'] = get_prediction(downharpdf.deltaF_exp_tdtomato, downharpdf.deltaF_exp_gfp)
#downharpdf['DA_exp_session'] = downharpdf.deltaF_exp_gfp - downharpdf.predicted_exp_gfp_session
#
# %%

figtitle = f'{animal}_{date}_polyVSexp'
fig.suptitle(figtitle)

#plt.savefig(rf'{path_save_figs}\{figtitle}.png', facecolor = 'white')

# %%
## NEED TO DO MORE STUFF HERE!

#%%
"""
.......##..#######..####.##....##.########.########..########
.......##.##.....##..##..###...##....##....##.....##.##......
.......##.##.....##..##..####..##....##....##.....##.##......
.......##.##.....##..##..##.##.##....##....##.....##.######..
.##....##.##.....##..##..##..####....##....##.....##.##......
.##....##.##.....##..##..##...###....##....##.....##.##......
..######...#######..####.##....##....##....########..##......
"""
## again, almost full repetition from photometry_interactive
## new part is the ICA column

jointdf = group_and_listify(downharpdf, 'trialno', ['timestamp_session', 'tdtomato_poly_flat', 'gfp_poly_flat','DA_poly_session'])

jointdf['trial_start_harp'] = jointdf.timestamp_session.apply(lambda x: x[0])
jointdf['trial_end_harp'] = jointdf.timestamp_session.apply(lambda x: x[-1])
jointdf['trial_duration_harp'] = jointdf.trial_end_harp - jointdf.trial_start_harp

jointdf.drop(jointdf.query('trial_duration_harp < 2').index, inplace = True)
jointdf.reset_index(drop = True, inplace = True)
jointdf['trialno'] = jointdf.index + 1

#jointdf['trialno'] = jointdf.trialno + 11 
plt.figure()
plt.plot(jointdf.trialno, jointdf.trial_duration_harp, label = 'harp')
plt.plot(bhvdf.trialno, bhvdf.trial_duration/1000, '--', label = 'bhv')
plt.title('trial duration (i.e. identity) ok?')
plt.legend()
plt.show()


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

jointdf['lever_abs_harp'] = jointdf.apply(lambda x: x.lever_rel_harp + x.trial_start_harp, axis = 1)



#%%

tt = 5

all_lever_presses = np.hstack(jointdf.query(f'trialno == {tt}').lever_rel_harp.values)
t_trial = np.hstack(jointdf.query(f'trialno == {tt}').t_trial_harp.values)

fig, axs = plt.subplots(2,1, figsize = (10,6), sharex = True, tight_layout = True)

for lvr in all_lever_presses:
    for ii in range(2):
        axs[ii].axvline(lvr, color = 'grey', lw = 0.5)

axs[0].plot(t_trial, zscore(np.hstack(jointdf.query(f'trialno == {tt}').tdtomato_poly_flat)), color = 'red', lw = 1)
axs[0].plot(t_trial, zscore(np.hstack(jointdf.query(f'trialno == {tt}').gfp_poly_flat)), color = 'green', lw = 1)
axs[0].set_ylabel('signals')

axs[1].plot(t_trial, zscore(np.hstack(jointdf.query(f'trialno == {tt}').DA_poly_session)), color = 'purple', lw = 1)
axs[1].set_ylabel('regression DA')


figtitle = f"{animal} {date} | trial {tt} | photometry ICA"
fig.suptitle(figtitle)

#%%
jointdf['cp_abs'] = jointdf.trial_start_harp + bhvdf.cp
jointdf['rwd_lever_abs'] = jointdf.lever_abs_harp.apply(lambda x: x[-1])
jointdf['nonrwd_lever_abs'] = jointdf.lever_abs_harp.apply(lambda x: x[x!=x[-1]])
# %%

#%%
### regular regression for comparison
snipps_DA_cp, time = signal2eventsnippets(downharpdf.timestamp_session, downharpdf.DA_poly_session,
                                np.hstack(jointdf.cp_abs.values), [-4,4], .01)
snipps_DA_rwd, _ = signal2eventsnippets(downharpdf.timestamp_session, downharpdf.DA_poly_session,
                                np.hstack(jointdf.rwd_lever_abs.values), [-4,4], .01)
snipps_DA_nonrwd, _ = signal2eventsnippets(downharpdf.timestamp_session, downharpdf.DA_poly_session,
                                np.hstack(jointdf.nonrwd_lever_abs.values), [-4,4], .01)
#%%
fig, axs = plt.subplots(2,3, tight_layout = True, figsize = (12,4))
vmin,vmax = np.nanquantile(snipps_DA_rwd, [.05,.95])
axs[0,2].imshow(snipps_DA_rwd, aspect = 'auto', vmin = vmin, vmax = vmax, cmap = 'bone')
#%%

plt.plot(np.nanmean(snipps_DA_rwd, axis = 0), color = 'purple', lw = 1)

#%%
tt = 11
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
def calculate_snr(signal):
    mean_signal = np.mean(signal)
    std_noise = np.std(signal)
    snr = mean_signal / std_noise
    return snr

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
