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

from ratcode.init import setup
setup()
# %%

animal = 'Technetium'
date = '250618'

PHOTOMETRY_PATH = os.path.join(DROPBOX_TASK_PATH, 'photometry', animal)

PATH_SAVE_FIGS = os.path.join(DROPBOX_TASK_PATH, 'analysis_photometry', f'{animal}_{date}')
if not os.path.exists(PATH_SAVE_FIGS):
    os.makedirs(PATH_SAVE_FIGS)

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

print(harpdf.trialno.values[-1])
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
# %%

"""
.########...######.
.##.....##.##....##
.##.....##.##......
.##.....##..######.
.##.....##.......##
.##.....##.##....##
.########...######.

and low pass
"""

downsample_factor = 10
fs = 1000 #sampling frequency
fs = fs/downsample_factor
nyquist = 0.5 * fs

harpdf['ds_tdtomato'] = harpdf['tdtomato'].rolling(2 * downsample_factor, center=True, min_periods=1).mean()
harpdf['ds_gfp'] = harpdf['gfp'].rolling(2 * downsample_factor, center=True, min_periods=1).mean()

downharpdf = harpdf.iloc[::downsample_factor].reset_index(drop = True)



high_cutoff = 20
#lowpass to remove the high freq noise
downharpdf['denoised_tdtomato'] = butter_filter(downharpdf.ds_tdtomato, high_cutoff, fs, 'low') 
downharpdf['denoised_gfp'] = butter_filter(downharpdf.ds_gfp, high_cutoff, fs, 'low')

# %%
##USING POLY

jump_threshold_tdtomato = 8
jump_threshold_gfp = 8
downharpdf['poly_tdtomato'] = segment_and_fit_function(downharpdf.timestamp_session.values, mask_jumps(downharpdf.denoised_tdtomato, thres = jump_threshold_tdtomato), function = 'poly')
#downharpdf['poly_tdtomato'] = segment_and_fit_function(downharpdf.timestamp_session.values, mask_jumps(downharpdf.ds_gfp, thres = jump_threshold_gfp), function = 'poly')
downharpdf['poly_gfp'] = segment_and_fit_function(downharpdf.timestamp_session.values, mask_jumps(downharpdf.denoised_gfp, thres = jump_threshold_gfp), function = 'poly')

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

#%%
downharpdf['deltaF_tdtomato'] = downharpdf['deltaF_poly_tdtomato']
downharpdf['deltaF_gfp'] = downharpdf['deltaF_poly_gfp']
#%%

plt.figure(figsize = (6,6), tight_layout = True)

sns.scatterplot(data = downharpdf, x = 'deltaF_tdtomato', y = 'deltaF_gfp', s = 10, alpha = 0.1, edgecolor = None, facecolor = 'slategrey')

sns.lineplot(x = downharpdf.deltaF_poly_tdtomato, y = quantile_regression(downharpdf.deltaF_poly_tdtomato, downharpdf.deltaF_poly_gfp), color = 'teal', lw = 1, label = 'q-reg (0.5)')

sns.lineplot(x = downharpdf.deltaF_poly_tdtomato, y = quantile_regression(downharpdf.deltaF_poly_tdtomato, downharpdf.deltaF_poly_gfp, quantile = 0.05), color = 'black', lw = 1)
sns.lineplot(x = downharpdf.deltaF_poly_tdtomato, y = quantile_regression(downharpdf.deltaF_poly_tdtomato, downharpdf.deltaF_poly_gfp, quantile = 0.95), color = 'black', lw = 1)
sns.lineplot(x = downharpdf.deltaF_poly_tdtomato, y = quantile_regression(downharpdf.deltaF_poly_tdtomato, downharpdf.deltaF_poly_gfp, quantile = 0.99), color = 'black', ls = 'dashed', lw = 1)
sns.lineplot(x = downharpdf.deltaF_poly_tdtomato, y = quantile_regression(downharpdf.deltaF_poly_tdtomato, downharpdf.deltaF_poly_gfp, quantile = 0.01), color = 'black', ls = 'dashed', lw = 1)

sns.lineplot(x = downharpdf.deltaF_poly_tdtomato, y = get_prediction(downharpdf.deltaF_poly_tdtomato, downharpdf.deltaF_poly_gfp), color = 'forestgreen', lw = 1, label = 'huber')

figtitle= f'{animal}_{date}_regressions'
plt.suptitle(figtitle)

plt.legend()
# %%

"""
..######..##.....##.########..##.....##.########.########......#######..##.....##....###....##....##.########.####.##.......########..######....
.##....##.##.....##.##.....##.##.....##.##.......##.....##....##.....##.##.....##...##.##...###...##....##.....##..##.......##.......##....##...
.##.......##.....##.##.....##.##.....##.##.......##.....##....##.....##.##.....##..##...##..####..##....##.....##..##.......##.......##.........
.##.......##.....##.########..##.....##.######...##.....##....##.....##.##.....##.##.....##.##.##.##....##.....##..##.......######....######....
.##.......##.....##.##...##....##...##..##.......##.....##....##..##.##.##.....##.#########.##..####....##.....##..##.......##.............##...
.##....##.##.....##.##....##....##.##...##.......##.....##....##....##..##.....##.##.....##.##...###....##.....##..##.......##.......##....##...
..######...#######..##.....##....###....########.########......#####.##..#######..##.....##.##....##....##....####.########.########..######....

will be trash

"""

tdtomato = downharpdf.deltaF_tdtomato.values
gfp = downharpdf.deltaF_gfp.values


x = np.asarray(tdtomato, dtype=np.float64)
y = np.asarray(gfp, dtype=np.float64)

X = np.column_stack([x, x**2, x**3])
X = sm.add_constant(X)

model_50 = sm.QuantReg(y, X).fit(q=0.5)
model_01 = sm.QuantReg(y, X).fit(q=.01)
#model_05 = sm.QuantReg(y, X).fit(q=.05)  ## there is not much difference here
#model_95 = sm.QuantReg(y, X).fit(q=.95)
model_99 = sm.QuantReg(y, X).fit(q=.99)

y_pred_50 = model_50.predict(X)
y_pred_01 = model_01.predict(X)
#y_pred_05 = model_05.predict(X)
#y_pred_95 = model_95.predict(X)
y_pred_99 = model_99.predict(X)
# %%


plt.plot(x,y, 'o', alpha=0.1, color = 'slategrey', markersize=5)
plt.plot(x, y_pred_50, color='teal', lw=1, label='Quantile 0.5')
plt.plot(x, y_pred_01, color='black', lw=1, label='Quantile 0.01')
plt.plot(x, y_pred_99, color='black', lw=1, label='Quantile 0.99')
# %%

from scipy.stats import rankdata, norm
import numpy as np

def gaussianize(v):
    r = rankdata(v) / (len(v)+1)
    return norm.ppf(r)

Xg = gaussianize(x)
Yg = gaussianize(y)
# %%
fig, axs = plt.subplots(1,2, figsize = (12,6), tight_layout = True)
axs[0].plot(x, y, 'o', alpha=0.1, color = 'slategrey', markersize=5)
axs[1].plot(Xg, Yg, 'o', alpha=0.1, color = 'slategrey', markersize=5)
# %%
plt.plot(Xg, color = 'red')
plt.plot(Yg, color = 'green')
# %%
DAg = Yg-get_prediction(Xg, Yg)
DA = y - get_prediction(x, y)
plt.plot(Xg, DAg, 'o', alpha=0.1, color = 'slategrey', markersize=5)
# %%
downharpdf['DA'] = DA
downharpdf['DAg'] = DAg

fig, axs = plt.subplots(2, figsize = (6,6), tight_layout = True)
tt = 20
axs[1].plot(zscore(downharpdf.query(f'trialno == {tt}').DA.values), lw = .5)
axs[1].plot(zscore(downharpdf.query(f'trialno == {tt}').DAg.values), lw = .5)


# %%



"""
....###....##.......########.########.########..##....##....###....########.####.##.....##.########....########..####.########..########.##.......####.##....##.########
...##.##...##..........##....##.......##.....##.###...##...##.##......##.....##..##.....##.##..........##.....##..##..##.....##.##.......##........##..###...##.##......
..##...##..##..........##....##.......##.....##.####..##..##...##.....##.....##..##.....##.##..........##.....##..##..##.....##.##.......##........##..####..##.##......
.##.....##.##..........##....######...########..##.##.##.##.....##....##.....##..##.....##.######......########...##..########..######...##........##..##.##.##.######..
.#########.##..........##....##.......##...##...##..####.#########....##.....##...##...##..##..........##.........##..##........##.......##........##..##..####.##......
.##.....##.##..........##....##.......##....##..##...###.##.....##....##.....##....##.##...##..........##.........##..##........##.......##........##..##...###.##......
.##.....##.########....##....########.##.....##.##....##.##.....##....##....####....###....########....##........####.##........########.########.####.##....##.########

ds and filer, and then this (no flattening with polyfits)

"""

x = downharpdf.denoised_tdtomato.values
y = downharpdf.denoised_gfp.values

x_mean, x_std = x.mean(), x.std()
x_s = (x - x_mean) / x_std

plt.plot(x_s, y, '.')

#%%
from statsmodels.nonparametric.kernel_regression import KernelReg

kr = KernelReg(endog = y, exog = x_s, var_type='c', bw = [.3])
#%%
fhat, _ = kr.fit(x_s)
#%%

dopamine_resid = y - fhat
#%%

downharpdf['DA_kernelreg'] = dopamine_resid

#%%

downharpdf['DA_quantile_reg'] = (y - quantile_regression(x_s, y, quantile = 0.5))/(quantile_regression(x_s,y,quantile = .99) - quantile_regression(x_s,y,quantile = .01))
#%%
downharpdf['tdtomato_standardized'] = x_s
#%%
PATH_STORE_PHOTOMETRY_PICKLES = os.path.join(DROPBOX_TASK_PATH, r'analysis_photometry')
downharpdf.to_pickle(f'{PATH_STORE_PHOTOMETRY_PICKLES}/{animal}_{date}_kernel_regression_downharpdf.pkl')

# %%

PATH_SAVE_PHOTOMETRY_FIGS = os.path.join(DROPBOX_TASK_PATH, rf'analysis_photometry/{animal}_{date}')
if not os.path.exists(PATH_SAVE_PHOTOMETRY_FIGS):
    os.makedirs(PATH_SAVE_PHOTOMETRY_FIGS)
#%%

fig, axs = plt.subplots(1,4, figsize = (14,4), tight_layout = True)

axs[0].plot(x_s, y, '.', alpha = 0.2)
axs[1].plot(x_s, downharpdf.DA_poly_session, '.', alpha = 0.2)
axs[2].plot(x_s, downharpdf.DA_quantile_reg, '.', alpha = 0.2)
axs[3].plot(x_s, downharpdf.DA_kernelreg, '.', alpha = 0.2)

axs[0].set_title('raw data')
axs[1].set_title('robust regression')
axs[2].set_title('quantile normalized regression')
axs[3].set_title('kernel regression')

axs[0].set_ylabel('dLight')
axs[1].set_ylabel('DA (std reg)')
axs[2].set_ylabel('DA (quantile reg)')
axs[3].set_ylabel('DA (kernel reg)')

[axs[ii].set_xlabel('tdTomato (std)') for ii in range(4)]
#axs[0,1].plot(downharpdf.timestamp_session, x, lw = .5, color = 'red')
#axs[0,1].plot(downharpdf.timestamp_session, y, lw = .5, color = 'green')

figtitle = f'{animal}_{date}_regression_comparison'
plt.suptitle(figtitle)
plt.savefig(f'{PATH_SAVE_PHOTOMETRY_FIGS}/{figtitle}.png', dpi = 300)
#%%

#downharpdf = pd.read_pickle(f'{PATH_STORE_PHOTOMETRY_PICKLES}/{animal}_{date}_kernel_regression_downharpdf.pkl')

# %%
"""
.......##..#######..####.##....##.########.########..########
.......##.##.....##..##..###...##....##....##.....##.##......
.......##.##.....##..##..####..##....##....##.....##.##......
.......##.##.....##..##..##.##.##....##....##.....##.######..
.##....##.##.....##..##..##..####....##....##.....##.##......
.##....##.##.....##..##..##...###....##....##.....##.##......
..######...#######..####.##....##....##....########..##......
"""

jointdf = group_and_listify(downharpdf, 'trialno', ['timestamp_session', 'denoised_tdtomato', 'tdtomato_standardized', 'denoised_gfp', 'DA_poly_session' , 'DA_quantile_reg', 'DA_kernelreg'])

jointdf['trial_start_harp'] = jointdf.timestamp_session.apply(lambda x: x[0])
jointdf['trial_end_harp'] = jointdf.timestamp_session.apply(lambda x: x[-1])
jointdf['trial_duration_harp'] = jointdf.trial_end_harp - jointdf.trial_start_harp

jointdf.drop(jointdf.query('trial_duration_harp < 2').index, inplace = True)
jointdf.reset_index(drop = True, inplace = True)
jointdf['trialno'] = jointdf.index + 1
# %%
#jointdf['trialno'] = jointdf.trialno + 11 
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
def figure_regression_comparison(tt):
    fig, axs = plt.subplots(5,2, figsize = (8,6), tight_layout = True, sharex='col',
                            width_ratios = [4,1], height_ratios = [10,10,10,10,1])

    time_in_trial = np.hstack(jointdf.query(f'trialno == {tt}').t_trial_harp.values)
    lever_times = np.hstack(jointdf.query(f'trialno == {tt}').lever_rel_harp.values)

    axs[-1,0].imshow(time_in_trial[np.newaxis, :], aspect="auto", cmap='viridis',
                  interpolation = None, extent = [time_in_trial[0],time_in_trial[-1],0,1])
    axs[-1,0].spines['left'].set_visible(False)
    axs[-1,0].yaxis.set_visible(False)    

    axs[0,0].plot(time_in_trial, zscore(np.hstack(jointdf.query(f'trialno == {tt}').denoised_tdtomato.values)), color = 'red', lw = .5)
    axs[0,0].plot(time_in_trial, zscore(np.hstack(jointdf.query(f'trialno == {tt}').denoised_gfp.values)), color = 'green', lw = .5)

    axs[1,0].plot(time_in_trial, zscore(np.hstack(jointdf.query(f'trialno == {tt}').DA_poly_session.values)), lw = .5, label = 'standard reg')
    axs[2,0].plot(time_in_trial, zscore(np.hstack(jointdf.query(f'trialno == {tt}').DA_quantile_reg.values)), lw = .5, label = 'quantile normalized reg')
    axs[3,0].plot(time_in_trial, zscore(np.hstack(jointdf.query(f'trialno == {tt}').DA_kernelreg.values)), lw = .5, label = 'kernel reg')

    for ll in lever_times:
        for ii in range(4):
            axs[ii,0].axvline(x = ll, color = 'grey', lw = .5)

    sns.scatterplot(ax = axs[0,1], x = np.hstack(jointdf.query(f'trialno == {tt}').tdtomato_standardized.values),
                    y = np.hstack(jointdf.query(f'trialno == {tt}').denoised_gfp.values), s = 5, c = time_in_trial)
    sns.scatterplot(ax = axs[1,1], x = np.hstack(jointdf.query(f'trialno == {tt}').tdtomato_standardized.values),
                    y = np.hstack(jointdf.query(f'trialno == {tt}').DA_poly_session.values), s = 5, c = time_in_trial)
    sns.scatterplot(ax = axs[2,1], x = np.hstack(jointdf.query(f'trialno == {tt}').tdtomato_standardized.values),
                    y = np.hstack(jointdf.query(f'trialno == {tt}').DA_quantile_reg.values), s = 5, c = time_in_trial)
    sns.scatterplot(ax = axs[3,1], x = np.hstack(jointdf.query(f'trialno == {tt}').tdtomato_standardized.values),
                    y = np.hstack(jointdf.query(f'trialno == {tt}').DA_kernelreg.values), s = 5, c = time_in_trial)


    axs[-1,0].set_xlabel('time in trial (s)')
    axs[-1,1].remove()


    axs[0,0].set_ylabel('raw data')
    axs[1,0].set_ylabel('robust')
    axs[2,0].set_ylabel('quantile norm')
    axs[3,0].set_ylabel('kernel')

    figtitle = f'{animal} {date} | trial {tt} | regression comparisons'
    plt.suptitle(figtitle)
    plt.close(fig)

    PATH_SAVE_REG_COMP = f'{PATH_SAVE_PHOTOMETRY_FIGS}/DA_reg_comparisons'
    if not os.path.exists(PATH_SAVE_REG_COMP):
        os.makedirs(PATH_SAVE_REG_COMP)

    fig.savefig(f'{PATH_SAVE_REG_COMP}/{figtitle.replace('|', '_')}.png', dpi = 300)
# %%
for tt in jointdf.trialno:
    figure_regression_comparison(tt)

#%%
# %%

"""
.########..##..........###....##....##..######...########...#######..##.....##.##....##.########.
.##.....##.##.........##.##....##..##..##....##..##.....##.##.....##.##.....##.###...##.##.....##
.##.....##.##........##...##....####...##........##.....##.##.....##.##.....##.####..##.##.....##
.########..##.......##.....##....##....##...####.########..##.....##.##.....##.##.##.##.##.....##
.##........##.......#########....##....##....##..##...##...##.....##.##.....##.##..####.##.....##
.##........##.......##.....##....##....##....##..##....##..##.....##.##.....##.##...###.##.....##
.##........########.##.....##....##.....######...##.....##..#######...#######..##....##.########.
"""

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import signal

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
# The scipy import is not strictly needed for this version, but kept for general utility.
from scipy import signal

def global_quantile_regression_correction(dlight_signal, tdtomato_signal, quantile_level=0.05):
    """
    Performs GLOBAL Quantile Regression to fit a single linear relationship
    across the entire trace. Useful for comparing against local fit results.
    """
    
    print(f"--- Global Fit: Quantile Regression at the {quantile_level*100:.0f}th percentile ---")

    # 1. Prepare data (Add a constant for the intercept)
    X = sm.add_constant(tdtomato_signal)
    Y = dlight_signal.ravel()

    # 2. Fit the Quantile Regression Model
    model = sm.QuantReg(Y, X)
    results = model.fit(q=quantile_level)
    
    # 3. Predict the baseline/artifact (F_fit)
    F_fit = results.predict(X)
    
    # 4. Calculate the corrected signal (Delta F/F)
    corrected_signal_dff = (dlight_signal - F_fit) / F_fit
    
    return corrected_signal_dff, F_fit

def local_quantile_correction(dlight_signal, tdtomato_signal, sampling_rate=30, 
                              window_size_sec=5, quantile_level=0.05):
    """
    Performs a Local Quantile Regression using a small, centered, sliding window.
    This is designed to adapt to rapid, instantaneous baseline jumps that change 
    the underlying linear relationship (a and b coefficients) between the channels.

    Args:
        dlight_signal (np.array): The activity-dependent signal (Y).
        tdtomato_signal (np.array): The motion-insensitive signal (X).
        sampling_rate (int): The frequency of the data (Hz).
        window_size_sec (int): The duration of the sliding window in seconds.
        quantile_level (float): The percentile level for regression (e.g., 0.05).

    Returns:
        np.array: The motion-corrected DLight signal (Delta F/F).
        np.array: The predicted artifact trace (F_fit).
    """
    
    window_size_points = int(window_size_sec * sampling_rate)
    n_points = len(dlight_signal)
    F_fit = np.zeros(n_points)
    
    print(f"--- Local Fit: Quantile Regression ({window_size_sec}s window, {quantile_level*100:.0f}th percentile) ---")

    # Use a small buffer around the current index for local fitting
    for i in range(n_points):
        # Define the centered window indices
        start = max(0, i - window_size_points // 2)
        end = min(n_points, i + window_size_points // 2)

        X_window = tdtomato_signal[start:end]
        Y_window = dlight_signal[start:end].ravel()
        
        # Ensure enough data points for a stable fit
        if len(X_window) < 30: # Use a minimum of 1 second of data (30 points)
            if i > 0:
                F_fit[i] = F_fit[i-1]
            continue

        # 1. Prepare and Fit the Quantile Regression Model Locally
        X_window_const = sm.add_constant(X_window)
        model = sm.QuantReg(Y_window, X_window_const)
        
        try:
            results = model.fit(q=quantile_level, max_iter=200) # Reduced iterations for speed
            
            # 2. Predict the baseline for the current single point 'i'
            # We predict using the local coefficients on the single point at X[i]
            X_current = sm.add_constant(tdtomato_signal[i])
            F_fit[i] = results.predict(X_current)[0]

        except Exception as e:
            # Handle cases where the local fit fails (e.g., singular matrix)
            if i > 0:
                F_fit[i] = F_fit[i-1]
            else:
                F_fit[i] = tdtomato_signal[i] # Crude fallback

    # 3. Calculate the corrected signal (Delta F/F)
    corrected_signal_dff = (dlight_signal - F_fit) / F_fit

    return corrected_signal_dff, F_fit
#%%
corrected_dlight_global, f_fit_global = global_quantile_regression_correction(
    y, x_s, quantile_level=0.05
)

corrected_dlight_local, f_fit_local = local_quantile_correction(
    y, x_s, sampling_rate=100, window_size_sec=5, quantile_level=0.05
)
#%%

"""
..#######..##.......########.
.##.....##.##.......##.....##
.##.....##.##.......##.....##
.##.....##.##.......##.....##
.##.....##.##.......##.....##
.##.....##.##.......##.....##
..#######..########.########.
"""

corrected, F_fit = local_quantile_correction(y, x_s, quantile_level=0.01)
#%%

plt.plot(corrected)
plt.plot(downharpdf.DA_poly_session.values, alpha = .5)


downharpdf['DA_quantile_reg_v2'] = corrected
jointexpdf = group_and_listify(downharpdf, 'trialno', ['DA_quantile_reg_v2'])
jointdf['DA_quantile_reg_v2'] = jointexpdf.DA_quantile_reg_v2
#%%
tt = 10
plt.plot(zscore(jointdf.query(f'trialno == {tt}').DA_quantile_reg_v2.values[0]), lw = .5)
plt.plot(zscore(jointdf.query(f'trialno == {tt}').DA_poly_session.values[0]), lw = .5)
#%%

"""
.########...######.....###...
.##.....##.##....##...##.##..
.##.....##.##........##...##.
.########..##.......##.....##
.##........##.......#########
.##........##....##.##.....##
.##.........######..##.....##
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def pca_correction(dlight_signal, tdtomato_signal):
    """
    Performs Principal Component Analysis (PCA) correction on two-channel fiber photometry data.

    Assumes the first principal component (PC1) represents the shared artifact (high variance)
    and the second component (PC2) represents the true neural signal (independent variance).

    Args:
        dlight_signal (np.array): The activity-dependent signal (DLight).
        tdtomato_signal (np.array): The motion-insensitive signal (tdTomato).

    Returns:
        np.array: The motion-corrected DLight signal (reconstructed from PC2).
        PCA: The fitted PCA object for inspection.
    """
    
    print("--- Starting PCA Correction ---")

    # 1. Stack the two channels into a single data matrix
    # X has shape (n_samples, n_features), where n_features=2
    X_raw = np.vstack([dlight_signal, tdtomato_signal]).T
    
    # 2. Standardize the data
    # PCA is sensitive to the scale of the features. We must scale the signals
    # to have zero mean and unit variance before running PCA.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    
    # 3. Apply PCA
    # We only need 2 components since we have 2 features
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 4. Identify and Isolate the Neural Signal Component (PC2)
    # PC1 (index 0) captures the maximum variance (artifact)
    # PC2 (index 1) captures the remaining variance (putative neural signal)
    
    # Create a reconstructed matrix that zeros out PC1 (the artifact)
    X_corrected_pca = np.copy(X_pca)
    
    # Zero out the first component (PC1) to remove the artifact
    # If the artifact is complex, you might need to zero out higher components too, 
    # but for 2-channel data, PC1 is the primary artifact axis.
    X_corrected_pca[:, 0] = 0 
    
    # 5. Reconstruct the Corrected Signal
    # Inverse transform the data back into the original feature space (DLight and tdTomato)
    X_corrected_scaled = pca.inverse_transform(X_corrected_pca)
    
    # Inverse transform the data back to the original scale
    X_corrected = scaler.inverse_transform(X_corrected_scaled)
    
    # The corrected DLight signal is the first column
    corrected_dlight_signal = X_corrected[:, 0]
    
    print("Variance Explained by Components:")
    print(f"PC1 (Artifact): {pca.explained_variance_ratio_[0]*100:.2f}%")
    print(f"PC2 (Signal): {pca.explained_variance_ratio_[1]*100:.2f}%")
    
    # Calculate Delta F/F using the corrected signal (F) and the original DLight mean (F0)
    # Using the mean of the corrected signal as a pseudo-F0 for DFF calculation 
    # is a common simplifying step after PCA/ICA.
    F0 = np.mean(corrected_dlight_signal)
    corrected_dff = (corrected_dlight_signal - F0) / F0

    return corrected_dff, pca, X_pca
# %%

y = downharpdf.denoised_gfp.values
x_s = downharpdf.denoised_tdtomato.values


pca_corrected_dff, pca_model, X_pca = pca_correction(y, x_s)
# %%
plt.plot(pca_corrected_dff)
# %%
downharpdf['DA_pca'] = pca_corrected_dff
jointexpdf = group_and_listify(downharpdf, 'trialno', ['DA_pca'])
jointdf['DA_pca'] = jointexpdf.DA_pca
# %%
tt = 22
plt.plot(zscore(jointdf.query(f'trialno == {tt}').DA_poly_session.values[0]), lw = .5)
plt.plot(zscore(jointdf.query(f'trialno == {tt+1}').DA_pca.values[0]), lw = .5)

# %%
plt.plot(X_pca[:,0])
plt.plot(X_pca[:,1])
plt.plot(x_s+5)
plt.plot(y)
# %%
