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
from ratcode.photometry.photometry import butter_filter, quantile_regression, get_prediction, segment_and_fit_function, mask_jumps, find_poly, signal2eventsnippets
from ratcode.common.dataframe import group_and_listify

from ratcode.init import setup
setup()

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import HuberRegressor 
from sklearn.decomposition import FastICA

# %%

from scipy.optimize import minimize
from scipy.stats import kurtosis
from sklearn.preprocessing import StandardScaler

def constrained_ica(X, reference, mu=0.5):
    """
    X: Observed signals (n_samples, 2) - dLight and tdTomato channels
    reference: The signal we want to 'lock onto' (tdTomato channel)
    mu: Constraint weight (0 = pure ICA, high = pure correlation)
    """
    # 1. Preprocessing: Center and Whiten (Standardize)
    scaler = StandardScaler()
    X_white = scaler.fit_transform(X)
    ref_norm = (reference - np.mean(reference)) / np.std(reference)
    
    n_samples = X_white.shape[0]

    # 2. Objective Function: Maximize Kurtosis (ICA) + Correlation (Constraint)
    # Note: We minimize the negative to 'maximize'
    def objective(w):
        w = w / np.linalg.norm(w) # Ensure w is a unit vector
        source_est = np.dot(X_white, w)
        
        # Independence term (Non-Gaussianity via Kurtosis)
        indep_loss = -np.abs(kurtosis(source_est))
        
        # Constraint term (Correlation with tdTomato reference)
        # We want high correlation, so we minimize negative correlation
        corr = np.corrcoef(source_est, ref_norm)[0, 1]
        const_loss = -np.abs(corr)
        
        return (1 - mu) * indep_loss + mu * const_loss

    # 3. Optimize for Component 1 (The 'Reference-like' component)
    res = minimize(objective, x0=np.array([1, 0]), method='Nelder-Mead')
    w1 = res.x / np.linalg.norm(res.x)
    comp_ref = np.dot(X_white, w1)

    # 4. Extract Component 2 (The orthogonal 'Pure Signal' component)
    # In 2D, the second vector is just the orthogonal counterpart
    w2 = np.array([-w1[1], w1[0]])
    comp_signal = np.dot(X_white, w2)

    return comp_signal, comp_ref
#%%
### repetition from photometry_interactive

"""
....###....##....##.####.##.....##....###....##.......########.....###....########.########
...##.##...###...##..##..###...###...##.##...##.......##.....##...##.##......##....##......
..##...##..####..##..##..####.####..##...##..##.......##.....##..##...##.....##....##......
.##.....##.##.##.##..##..##.###.##.##.....##.##.......##.....##.##.....##....##....######..
.#########.##..####..##..##.....##.#########.##.......##.....##.#########....##....##......
.##.....##.##...###..##..##.....##.##.....##.##.......##.....##.##.....##....##....##......
.##.....##.##....##.####.##.....##.##.....##.########.########..##.....##....##....########
"""

animal = 'Niobium'
date = '250624'
PHOTOMETRY_PATH = os.path.join(DROPBOX_TASK_PATH, 'photometry', animal)

PATH_SAVE_FIGS = os.path.join(DROPBOX_TASK_PATH, 'analysis_photometry', f'{animal}_{date}')
if not os.path.exists(PATH_SAVE_FIGS):
    os.makedirs(PATH_SAVE_FIGS)

PATH_SAVE_ICA = os.path.join(DROPBOX_TASK_PATH, 'analysis_photometry', '00_all_sessions_ICA_snippets')


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

print(harpdf.trialno.values[-1])


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

downharpdf['predicted_poly_gfp_session'] = get_prediction(downharpdf.deltaF_poly_tdtomato, downharpdf.deltaF_poly_gfp)[0]
downharpdf['DA_poly_session'] = downharpdf.deltaF_poly_gfp - downharpdf.predicted_poly_gfp_session


downharpdf['deltaF_tdtomato'] = downharpdf['deltaF_poly_tdtomato']
downharpdf['deltaF_gfp'] = downharpdf['deltaF_poly_gfp']

# %%

"""
.####..######.....###...
..##..##....##...##.##..
..##..##........##...##.
..##..##.......##.....##
..##..##.......#########
..##..##....##.##.....##
.####..######..##.....##
"""

ica = FastICA(n_components=2, random_state=0)
components = ica.fit_transform(downharpdf[['deltaF_tdtomato', 'deltaF_gfp']].values)  # Reconstruct signals

downharpdf['ICA_0'] = components[:,0]
downharpdf['ICA_1'] = components[:,1]


## constrained ICA

clean_dlight, isolated_motion = constrained_ica(downharpdf[['ds_gfp', 'ds_tdtomato']].values, downharpdf.ds_tdtomato.values, mu=0.8)

if np.max(clean_dlight) < np.abs(np.min(clean_dlight)):
    clean_dlight *= -1

downharpdf['constrained_ICA_dlight'] = clean_dlight
downharpdf['constrained_ICA_motion'] = isolated_motion

# %%

"""
.##....##.##.....##.########
.###...##.###...###.##......
.####..##.####.####.##......
.##.##.##.##.###.##.######..
.##..####.##.....##.##......
.##...###.##.....##.##......
.##....##.##.....##.##......
"""

from sklearn.decomposition import NMF


green_obs = downharpdf['ds_gfp'].values
red_obs = downharpdf['ds_tdtomato'].values


# 1. Prepare data (NMF requires strictly POSITIVE data)
# Shift data so the minimum value is 0 or a small positive constant
offset_green = np.min(green_obs)
offset_red = np.min(red_obs)
X_positive = np.c_[green_obs - offset_green, red_obs - offset_red]

# 2. Fit NMF
# 'mu' solver with 'kullback-leibler' is often more robust for biological peaks
model = NMF(n_components=2, init='random', random_state=0, solver='mu')
H = model.fit_transform(X_positive)  # These are your separated traces
W = model.components_              # This is how they are mixed

# 3. Identify your signals
# One component will have the transients (DA), one will have the motion.
# You can identify them by checking which one correlates more with the Red channel.
corr0 = np.corrcoef(H[:, 0], red_obs)[0, 1]
corr1 = np.corrcoef(H[:, 1], red_obs)[0, 1]

dlight_pure = H[:, 1] if corr0 > corr1 else H[:, 0]
motion_pure = H[:, 0] if corr0 > corr1 else H[:, 1]
# %%

plt.plot(motion_pure)
plt.plot(dlight_pure)
# %%

downharpdf['motion_pure'] = motion_pure
downharpdf['dlight_pure'] = dlight_pure


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

jointdf = group_and_listify(downharpdf, 'trialno', ['timestamp_session', 'deltaF_tdtomato', 'deltaF_gfp','DA_poly_session','ICA_0','ICA_1', 'constrained_ICA_dlight','constrained_ICA_motion', 'motion_pure', 'dlight_pure'])

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

tt = 30

all_lever_presses = np.hstack(jointdf.query(f'trialno == {tt}').lever_rel_harp.values)
t_trial = np.hstack(jointdf.query(f'trialno == {tt}').t_trial_harp.values)

fig, axs = plt.subplots(5,1, figsize = (10,6), sharex = True, tight_layout = True)

for lvr in all_lever_presses:
    for ii in range(4):
        axs[ii].axvline(lvr, color = 'grey', lw = 0.5)

axs[0].plot(t_trial, zscore(np.hstack(jointdf.query(f'trialno == {tt}').deltaF_tdtomato)), color = 'red', lw = 1)
axs[0].plot(t_trial, zscore(np.hstack(jointdf.query(f'trialno == {tt}').deltaF_gfp)), color = 'green', lw = 1)
axs[0].set_ylabel('signals')

axs[1].plot(t_trial, zscore(np.hstack(jointdf.query(f'trialno == {tt}').DA_poly_session)), color = 'purple', lw = 1)
axs[1].set_ylabel('regression DA')

axs[2].plot(t_trial, np.hstack(jointdf.query(f'trialno == {tt}').ICA_0), color = 'blue', lw = 1, label = 'ICA 0')
axs[2].plot(t_trial, np.hstack(jointdf.query(f'trialno == {tt}').ICA_1), color = 'orange', lw = 1, label = 'ICA 1')
axs[2].set_ylabel('ICA')
axs[2].legend(frameon = False)

axs[3].plot(t_trial, np.hstack(jointdf.query(f'trialno == {tt}').constrained_ICA_dlight), color = 'blue', lw = 1, label = 'DA')
axs[3].plot(t_trial, np.hstack(jointdf.query(f'trialno == {tt}').constrained_ICA_motion), color = 'orange', lw = 1, label = 'motion')
axs[3].set_ylabel('cICA')
axs[3].legend(frameon = False)

axs[4].plot(t_trial, np.hstack(jointdf.query(f'trialno == {tt}').dlight_pure), lw = 1, label = 'DA')
axs[4].plot(t_trial, np.hstack(jointdf.query(f'trialno == {tt}').motion_pure), lw = 1, label = 'motion')
axs[4].set_ylabel('NMF')
axs[4].legend(frameon = False)

figtitle = f"{animal} {date} | trial {tt} | photometry ICA"
fig.suptitle(figtitle)

#%%
jointdf['cp_abs'] = jointdf.trial_start_harp + bhvdf.cp
jointdf['rwd_lever_abs'] = jointdf.lever_abs_harp.apply(lambda x: x[-1])
jointdf['nonrwd_lever_abs'] = jointdf.lever_abs_harp.apply(lambda x: x[x!=x[-1]])
# %%

snipps_0_cp, _ = signal2eventsnippets(downharpdf.timestamp_session, downharpdf.ICA_0,
                                np.hstack(jointdf.cp_abs.values), [-4,4], .01)
snipps_1_cp, _ = signal2eventsnippets(downharpdf.timestamp_session, downharpdf.ICA_1,
                                np.hstack(jointdf.cp_abs.values), [-4,4], .01)

snipps_0_rwd, _ = signal2eventsnippets(downharpdf.timestamp_session, downharpdf.ICA_0,
                                np.hstack(jointdf.rwd_lever_abs.values), [-4,4], .01)
snipps_1_rwd, _ = signal2eventsnippets(downharpdf.timestamp_session, downharpdf.ICA_1,
                                np.hstack(jointdf.rwd_lever_abs.values), [-4,4], .01)

snipps_0_nonrwd, time = signal2eventsnippets(downharpdf.timestamp_session, downharpdf.ICA_0,
                                np.hstack(jointdf.nonrwd_lever_abs.values), [-4,4], .01)
snipps_1_nonrwd, _ = signal2eventsnippets(downharpdf.timestamp_session, downharpdf.ICA_1,
                                np.hstack(jointdf.nonrwd_lever_abs.values), [-4,4], .01)

### cICA
snipps_cICA_dlight_cp, time = signal2eventsnippets(downharpdf.timestamp_session, downharpdf.constrained_ICA_dlight,
                                np.hstack(jointdf.cp_abs.values), [-4,4], .01)
snipps_cICA_motion_cp, _ = signal2eventsnippets(downharpdf.timestamp_session, downharpdf.constrained_ICA_motion,
                                np.hstack(jointdf.cp_abs.values), [-4,4], .01)
snipps_cICA_dlight_rwd, _ = signal2eventsnippets(downharpdf.timestamp_session, downharpdf.constrained_ICA_dlight,
                                np.hstack(jointdf.rwd_lever_abs.values), [-4,4], .01)
snipps_cICA_motion_rwd, _ = signal2eventsnippets(downharpdf.timestamp_session, downharpdf.constrained_ICA_motion,
                                np.hstack(jointdf.rwd_lever_abs.values), [-4,4], .01)
snipps_cICA_dlight_nonrwd, _ = signal2eventsnippets(downharpdf.timestamp_session, downharpdf.constrained_ICA_dlight,
                                np.hstack(jointdf.nonrwd_lever_abs.values), [-4,4], .01)
snipps_cICA_motion_nonrwd, _ = signal2eventsnippets(downharpdf.timestamp_session, downharpdf.constrained_ICA_motion,
                                np.hstack(jointdf.nonrwd_lever_abs.values), [-4,4], .01)


### regular regression for comparison
snipps_DA_cp, _ = signal2eventsnippets(downharpdf.timestamp_session, downharpdf.DA_poly_session,
                                np.hstack(jointdf.cp_abs.values), [-4,4], .01)
snipps_DA_rwd, _ = signal2eventsnippets(downharpdf.timestamp_session, downharpdf.DA_poly_session,
                                np.hstack(jointdf.rwd_lever_abs.values), [-4,4], .01)
snipps_DA_nonrwd, _ = signal2eventsnippets(downharpdf.timestamp_session, downharpdf.DA_poly_session,
                                np.hstack(jointdf.nonrwd_lever_abs.values), [-4,4], .01)

### NMF
snipps_NMF_cp, _ = signal2eventsnippets(downharpdf.timestamp_session, downharpdf.dlight_pure,
                                np.hstack(jointdf.cp_abs.values), [-4,4], .01)
snipps_NMF_rwd, _ = signal2eventsnippets(downharpdf.timestamp_session, downharpdf.dlight_pure,
                                np.hstack(jointdf.rwd_lever_abs.values), [-4,4], .01)
snipps_NMF_nonrwd, _ = signal2eventsnippets(downharpdf.timestamp_session, downharpdf.dlight_pure,
                                np.hstack(jointdf.nonrwd_lever_abs.values), [-4,4], .01)
snipps_NMFmotion_cp, _ = signal2eventsnippets(downharpdf.timestamp_session, downharpdf.motion_pure,
                                np.hstack(jointdf.cp_abs.values), [-4,4], .01)
snipps_NMFmotion_rwd, _ = signal2eventsnippets(downharpdf.timestamp_session, downharpdf.motion_pure,
                                np.hstack(jointdf.rwd_lever_abs.values), [-4,4], .01)
snipps_NMFmotion_nonrwd, _ = signal2eventsnippets(downharpdf.timestamp_session, downharpdf.motion_pure,
                                np.hstack(jointdf.nonrwd_lever_abs.values), [-4,4], .01)
#%%
fig, axs = plt.subplots(4,3, tight_layout = True, figsize = (12,10), sharey = 'row', sharex = True)

axs[0,0].plot(time, np.nanmean(snipps_0_cp, axis = 0), color = 'blue', lw = 1)
axs[0,0].plot(time, np.nanmean(snipps_1_cp, axis = 0), color = 'orange', lw = 1)

axs[0,1].plot(time, np.nanmean(snipps_0_nonrwd, axis = 0), color = 'blue', lw = 1)
axs[0,1].plot(time, np.nanmean(snipps_1_nonrwd, axis = 0), color = 'orange', lw = 1)

axs[0,2].plot(time, np.nanmean(snipps_0_rwd, axis = 0), color = 'blue', lw = 1)
axs[0,2].plot(time, np.nanmean(snipps_1_rwd, axis = 0), color = 'orange', lw = 1)

# constrainted ICA
axs[1,0].plot(time, np.nanmean(snipps_cICA_dlight_cp, axis = 0), color = 'blue', lw = 1)
axs[1,0].plot(time, np.nanmean(snipps_cICA_motion_cp, axis = 0), color = 'orange', lw = 1)

axs[1,1].plot(time, np.nanmean(snipps_cICA_dlight_nonrwd, axis = 0), color = 'blue', lw = 1)
axs[1,1].plot(time, np.nanmean(snipps_cICA_motion_nonrwd, axis = 0), color = 'orange', lw = 1)

axs[1,2].plot(time, np.nanmean(snipps_cICA_dlight_rwd, axis = 0), color = 'blue', lw = 1, label = 'DA')
axs[1,2].plot(time, np.nanmean(snipps_cICA_motion_rwd, axis = 0), color = 'orange', lw = 1, label = 'motion')
axs[1,2].legend(frameon = False)

## regression DA for comparison
axs[2,0].plot(time, np.nanmean(snipps_DA_cp, axis = 0), color = 'purple', lw = 1)
axs[2,1].plot(time, np.nanmean(snipps_DA_nonrwd, axis = 0), color = 'purple', lw = 1)
axs[2,2].plot(time, np.nanmean(snipps_DA_rwd, axis = 0), color = 'purple', lw = 1)

## NMF
axs[3,0].plot(time, np.nanmean(snipps_NMF_cp, axis = 0), color = 'blue', lw = 1)
axs[3,1].plot(time, np.nanmean(snipps_NMF_nonrwd, axis = 0), color = 'blue', lw = 1)
axs[3,2].plot(time, np.nanmean(snipps_NMF_rwd, axis = 0), color = 'blue', lw = 1, label = 'DA')
axs[3,0].plot(time, np.nanmean(snipps_NMFmotion_cp, axis = 0), color = 'orange', lw = 1)
axs[3,1].plot(time, np.nanmean(snipps_NMFmotion_nonrwd, axis = 0), color = 'orange', lw = 1)
axs[3,2].plot(time, np.nanmean(snipps_NMFmotion_rwd, axis = 0), color = 'orange', lw = 1, label = 'motion')
axs[3,2].legend(frameon = False)

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




