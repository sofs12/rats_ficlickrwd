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
import statsmodels.api as sm
import argparse

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


def main():
    parser = argparse.ArgumentParser(description='Extract TTLs from neuropixel recording and correct geometry after ibl sorter')
    parser.add_argument('animal', type=str, help='Name of the animal (e.g. Ruthenium)')
    parser.add_argument('date', type=str, help='Date of the session format yymmdd (e.g. 260225)')
    parser.add_argument('bool_encoder', type = str, help = 'In case the encoder malfunctions, flag this boolean as false')
    args = parser.parse_args()
    animal = args.animal
    date = args.date
    bool_encoder = args.bool_encoder

    setup()

    #animal = 'Ruthenium'
    #date = '260312'
    ## in case the encoder malfunctions (fully), flag this as False
    #bool_encoder = True


    PHOTOMETRY_PATH = os.path.join(DROPBOX_TASK_PATH, 'photometry', animal)

    PATH_SAVE_DFS = os.path.join(DROPBOX_TASK_PATH, 'analysis_photometry')

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


    exp = determine_experiment(bhvdf)
    print(f'RUNNING DAILY PHOTOMETRY FOR\nanimal {animal}\ndate {date}\nexperiment {exp}')


    for file in os.listdir(PHOTOMETRY_PATH):

        if convert_date_bonsai(date) in file:
            if "in0" in file:
                in0_path = os.path.join(PHOTOMETRY_PATH, file)
            if "in1" in file:
                in1_path = os.path.join(PHOTOMETRY_PATH, file)
            if "in2" in file:
                in2_path = os.path.join(PHOTOMETRY_PATH, file)
            if "in3" in file: ## encoder
                in3_path = os.path.join(PHOTOMETRY_PATH, file)

    in0 = pd.read_csv(in0_path, header = None)
    in0.columns = ['in0', 'timestamp0']

    in1 = pd.read_csv(in1_path, header = None)
    in1.columns = ['in1', 'timestamp1']

    in2 = pd.read_csv(in2_path, header = None)
    in2.columns = ['in2', 'timestamp2']

    ## this is the encoder // rotary joint angular position
    in3 = pd.read_csv(in3_path, header = None)
    in3.columns = ['in3', 'timestamp3']

    print('harp files read')

    harpdf = pd.concat([in0, in1, in2], axis = 1)
    harpdf['timestamp_comp'] = (harpdf.timestamp0 == harpdf.timestamp1)*(harpdf.timestamp0 == harpdf.timestamp2)
    harpdf.drop(harpdf.query('timestamp_comp == False').index, inplace=True)
    harpdf.rename(columns = {'timestamp1': 'timestamp'}, inplace=True)
    harpdf.drop(['timestamp0', 'timestamp_comp', 'timestamp2'], axis = 1, inplace=True)

    harpdf['tdtomato'] = harpdf.in0/2**16*20
    harpdf['gfp'] = harpdf.in1/2**16*20
    harpdf['gpio'] = harpdf.in2/2**16*20
    harpdf['encoder'] = in3.in3/2**16*20


    if bool_encoder: 
        harpdf['continuous_encoder'] = make_continuous(harpdf['encoder'].values, min_val=harpdf.encoder.min(), max_val=harpdf.encoder.max())
    else:
        harpdf['continuous_encoder'] = 0

    window = 51 ## must be odd
    harpdf['encoder_vel'] = savgol_filter(harpdf['continuous_encoder'], window_length=window, polyorder=2, deriv=1)
    harpdf['encoder_vel_abs'] = np.abs(harpdf.encoder_vel)

    harpdf['ttl_bool'] = harpdf.gpio.apply(lambda x: int(x>1))
    harpdf['diff_ttl'] = harpdf.ttl_bool.diff()

    harpdf['ttl_rising_edge'] = harpdf.ttl_bool*harpdf.diff_ttl > 0
    harpdf['timestamp_session'] = harpdf.timestamp - harpdf.timestamp[0]

    harpdf['trialno'] = harpdf.ttl_rising_edge.cumsum() - 1
    harpdf.drop(harpdf.query('trialno < 1').index, inplace=True)
    #in the behaviour I always drop the last trial
    harpdf.drop(harpdf.query(f'trialno == {harpdf.trialno.max()}').index, inplace=True)

    print(f'total # trials: {harpdf.trialno.values[-1]}')

    start = 0
    end = -1

    gpio_offset = np.min([np.mean(harpdf.gfp.values),np.mean(harpdf.tdtomato.values)])

    #plt.figure()
    #plt.plot(harpdf.timestamp_session[start:end], harpdf.gpio[start:end]/5+gpio_offset, color = 'grey', alpha = 0.5)
    #plt.plot(harpdf.timestamp_session[start:end], harpdf.tdtomato[start:end]-.2, color = 'red')
    #plt.plot(harpdf.timestamp_session[start:end], harpdf.gfp[start:end], color = 'green')
    #plt.plot(harpdf.timestamp_session[start:end], harpdf.encoder[start:end], color = 'blue', alpha = 0.5)
    #plt.plot(harpdf.timestamp_session[start:end], zscore(harpdf.continuous_encoder[start:end]), color = 'pink')
    #plt.xlabel('t (s)')
    #plt.ylabel('V')
    #plt.title(f'{animal}_{date}')
    #plt.ylim(0)
    #plt.show()


    """
    .########...#######..##......##.##....##..######.....###....##.....##.########..##.......########
    .##.....##.##.....##.##..##..##.###...##.##....##...##.##...###...###.##.....##.##.......##......
    .##.....##.##.....##.##..##..##.####..##.##........##...##..####.####.##.....##.##.......##......
    .##.....##.##.....##.##..##..##.##.##.##..######..##.....##.##.###.##.########..##.......######..
    .##.....##.##.....##.##..##..##.##..####.......##.#########.##.....##.##........##.......##......
    .##.....##.##.....##.##..##..##.##...###.##....##.##.....##.##.....##.##........##.......##......
    .########...#######...###..###..##....##..######..##.....##.##.....##.##........########.########
    """


    ####### might be worth revisiting this!

    downsample_factor = 10
    fs = 1000 #sampling frequency
    fs = fs/downsample_factor
    nyquist = 0.5 * fs


    for colname in ['tdtomato', 'gfp', 'encoder', 'continuous_encoder', 'encoder_vel']:
        harpdf[f'ds_{colname}'] = harpdf[colname].rolling(2 * downsample_factor, center=True, min_periods=1).mean()

    downharpdf = harpdf.iloc[::downsample_factor].reset_index(drop = True)
    
    #lowpass filter to remove the high freq noise
    high_cutoff = 20

    downharpdf['denoised_tdtomato'] = butter_filter(downharpdf.ds_tdtomato, high_cutoff, fs, 'low') 
    downharpdf['denoised_gfp'] = butter_filter(downharpdf.ds_gfp, high_cutoff, fs, 'low')

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

    jump_threshold_tdtomato = 10
    jump_threshold_gfp = 10

    fig, axs = plt.subplots(2)
    axs[1].plot(np.abs(zscore(np.diff(butter_filter(downharpdf.ds_gfp, 0.01, fs, 'low')))))
    axs[0].plot(np.abs(zscore(np.diff(butter_filter(downharpdf.ds_tdtomato, 0.01, fs, 'low')))))
    axs[1].plot(downharpdf.ds_gfp-3, color = 'green', alpha = 0.5)
    axs[0].plot(downharpdf.ds_tdtomato-4, color = 'red', alpha = 0.5)

    axs[0].axhline(jump_threshold_tdtomato)
    axs[1].axhline(jump_threshold_gfp)


    #plt.figure()
    #plt.plot(downharpdf.ds_tdtomato)
    #plt.plot(downharpdf.ds_gfp)
    #plt.plot(downharpdf.ds_continuous_encoder)


    #np.where(np.abs(zscore(np.diff(butter_filter(downharpdf.ds_tdtomato, 0.01, fs, 'low'))))>4)
    
    ## for when I want to manually delete bad data
    #plt.plot(downharpdf.ds_tdtomato[:122000])
    #plt.plot(downharpdf.ds_gfp[:57500])
    #downharpdf.loc[57500:, 'ds_tdtomato'] = np.nan
    #downharpdf.loc[57500:, 'ds_gfp'] = np.nan


    #if a channel saturates
    downharpdf.loc[downharpdf.query('ds_gfp > 9.99').ds_gfp.index, 'ds_gfp'] = np.nan


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



    fig, axs = plt.subplots(3, figsize = (12,6), tight_layout = True, sharex = True)

    time_session_mins = downharpdf.timestamp_session.values
    time_session_mins = (time_session_mins - time_session_mins[0])/60

    axs[0].plot(time_session_mins, downharpdf.ds_tdtomato, color = 'red')
    axs[0].plot(time_session_mins, downharpdf.poly_tdtomato, color = 'white')
    axs[0].plot(time_session_mins, downharpdf.ds_gfp, color = 'green')
    axs[0].plot(time_session_mins, downharpdf.poly_gfp, color = 'white')

    axs[1].plot(time_session_mins, downharpdf.tdtomato_poly_flat, color = 'red')
    axs[1].plot(time_session_mins, downharpdf.gfp_poly_flat, color = 'green')

    axs[2].plot(time_session_mins, downharpdf.DA_poly_session, color = 'purple')

    axs[0].set_ylabel('ds data and fits')
    axs[1].set_ylabel('flattened = ds - fit')
    axs[2].set_ylabel('DA')

    axs[-1].set_xlabel('time in session (min)')

    figtitle = f'{animal} {date} | exp {exp} | DA from robust regression | signals flattened via 3rd order polynomial'
    fig.suptitle(figtitle)

    plt.savefig(rf"{PATH_SAVE_FIGS}\{figtitle.replace('|', '_')}.png", dpi = 300)



    fig, axs = plt.subplots(2,1, tight_layout = True, figsize = (4,8), sharex = True)

    axs[0].plot(downharpdf.deltaF_poly_tdtomato.values, downharpdf.deltaF_poly_gfp.values, '.', color = 'grey', alpha = 0.2)
    axs[0].plot(downharpdf.deltaF_poly_tdtomato.values, get_prediction(downharpdf.deltaF_poly_tdtomato.values, downharpdf.deltaF_poly_gfp.values), color = 'black', lw = 1)
    axs[1].plot(downharpdf.deltaF_poly_tdtomato.values, downharpdf.DA_poly_session, '.', color = 'grey', alpha = 0.2)

    axs[0].set_ylabel('dLight (dF/F)')
    axs[1].set_ylabel('DA from regression')
    axs[1].set_xlabel('tdTomato (dF/F)')

    figtitle = f'{animal} {date} | exp {exp} | scatter'

    fig.suptitle(figtitle)

    plt.savefig(rf"{PATH_SAVE_FIGS}\{figtitle.replace('|', '_')}.png", dpi = 300)


    """
    .########..########..######...########..########..######...######..####..#######..##....##....########..########..######...#######..########..########.########.
    .##.....##.##.......##....##..##.....##.##.......##....##.##....##..##..##.....##.###...##....##.....##.##.......##....##.##.....##.##.....##.##.......##.....##
    .##.....##.##.......##........##.....##.##.......##.......##........##..##.....##.####..##....##.....##.##.......##.......##.....##.##.....##.##.......##.....##
    .########..######...##...####.########..######....######...######...##..##.....##.##.##.##....##.....##.######...##.......##.....##.##.....##.######...########.
    .##...##...##.......##....##..##...##...##.............##.......##..##..##.....##.##..####....##.....##.##.......##.......##.....##.##.....##.##.......##...##..
    .##....##..##.......##....##..##....##..##.......##....##.##....##..##..##.....##.##...###....##.....##.##.......##....##.##.....##.##.....##.##.......##....##.
    .##.....##.########..######...##.....##.########..######...######..####..#######..##....##....########..########..######...#######..########..########.##.....##
    """

    if bool_encoder == False: ## this is not working FIX
        downharpdf['DA_encoder_session'] = downharpdf['DA_poly_session']

    else:
        tdtomato = np.hstack(downharpdf.tdtomato_poly_flat.values)
        gfp = np.hstack(downharpdf.gfp_poly_flat.values)
        continuous_encoder = np.hstack(downharpdf.ds_continuous_encoder.values)
        encoder_vel = np.hstack(downharpdf.ds_encoder_vel.values)

        data_stack = np.column_stack([tdtomato, continuous_encoder, encoder_vel])
        is_valid = ~np.isnan(data_stack).any(axis=1)

        X_raw = np.column_stack([
            zscore(tdtomato[is_valid]), 
            zscore(continuous_encoder[is_valid]), 
            zscore(encoder_vel[is_valid])
        ])
        y_valid = gfp[is_valid]


        X_with_const = sm.add_constant(X_raw)

        mod = sm.QuantReg(y_valid, X_with_const)
        res = mod.fit(q = 0.5)

        residuals = np.full(len(gfp), np.nan)
        residuals[is_valid] = res.resid

        print(res.summary())

        downharpdf['DA_encoder_session'] = residuals

    ## save downharpdf
    downharpdf.to_pickle(rf'{PATH_SAVE_DFS}\{animal}_{date}_downharpdf.pkl')

    """
    .......##..#######..####.##....##.########.########..########
    .......##.##.....##..##..###...##....##....##.....##.##......
    .......##.##.....##..##..####..##....##....##.....##.##......
    .......##.##.....##..##..##.##.##....##....##.....##.######..
    .##....##.##.....##..##..##..####....##....##.....##.##......
    .##....##.##.....##..##..##...###....##....##.....##.##......
    ..######...#######..####.##....##....##....########..##......
    """

    jointdf = group_and_listify(downharpdf, 'trialno', ['timestamp_session', 'tdtomato_poly_flat', 'gfp_poly_flat','DA_poly_session',
                                                        'ds_encoder', 'ds_continuous_encoder', 'ds_encoder_vel',
                                                        'DA_encoder_session'])

    jointdf['trial_start_harp'] = jointdf.timestamp_session.apply(lambda x: x[0])
    jointdf['trial_end_harp'] = jointdf.timestamp_session.apply(lambda x: x[-1])
    jointdf['trial_duration_harp'] = jointdf.trial_end_harp - jointdf.trial_start_harp

    jointdf.drop(jointdf.query('trial_duration_harp < 2').index, inplace = True)
    jointdf.reset_index(drop = True, inplace = True)
    jointdf['trialno'] = jointdf.index + 1

    #jointdf['trialno'] = jointdf.trialno + 11 
    #plt.figure()
    #plt.plot(jointdf.trialno, jointdf.trial_duration_harp, label = 'harp')
    #plt.plot(bhvdf.trialno, bhvdf.trial_duration/1000, '--', label = 'bhv')
    #plt.title('trial duration (i.e. identity) ok?')
    #plt.legend()
    #plt.show()


    jointdf['blockno'] = bhvdf.blockno
    jointdf['FI'] = bhvdf.FI/1000
    jointdf['FI'] = jointdf['FI'].astype(int)
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

    jointdf['cp_abs'] = jointdf.trial_start_harp + bhvdf.cp
    jointdf['rwd_lever_abs'] = jointdf.lever_abs_harp.apply(lambda x: x[-1])
    jointdf['nonrwd_lever_abs'] = jointdf.lever_abs_harp.apply(lambda x: x[x!=x[-1]])


    ## a bunch of new things added, on march 5th -- should rerun this for Ruthenium and Palladium
    jointdf['count_lever'] = jointdf.lever_rel_harp.apply(lambda x: len(x) if type(x) == np.ndarray else 0)
    jointdf = jointdf.query('count_lever > 0')

    jointdf['last_lever_harp'] = jointdf.apply(lambda x: x.lever_rel_harp[-1] if x.count_lever > 0 else np.nan, axis = 1)

    jointdf['pump_on_arduino'] = bhvdf.pump_rel.apply(lambda x: int(x[0]) if len(x)> 0 else np.nan)

    jointdf['pump_off_arduino'] = bhvdf.pump_rel + bhvdf.pump_duration
    jointdf['pump_off_arduino'] = jointdf.pump_off_arduino.apply(lambda x: int(x[0]) if len(x)> 0 else np.nan)

    jointdf['pump_on_harp'] = jointdf.apply(lambda x: convert_timestamp(x.pump_on_arduino,
                                    [0, x.trial_duration_arduino],
                                    [0, x.trial_duration_harp]), axis = 1)
    jointdf['pump_off_harp'] = jointdf.apply(lambda x: convert_timestamp(x.pump_off_arduino,
                                    [0, x.trial_duration_arduino],
                                    [0, x.trial_duration_harp]), axis = 1)


    # NEW - 14 AUGUST
    if jointdf.click.unique()[0] == 1:
        jointdf['click_arduino'] = bhvdf.click_rel.apply(lambda x: int(x[0]) if len(x)> 0 else np.nan)
        jointdf['click_harp'] = jointdf.apply(lambda x: convert_timestamp(x.click_arduino,
                                    [0, x.trial_duration_arduino],
                                    [0, x.trial_duration_harp]), axis = 1)

    jointdf['lever_abs_harp'] = jointdf.lever_rel_harp + jointdf.trial_start_harp
    jointdf['last_lever_abs_harp'] = jointdf.last_lever_harp + jointdf.trial_start_harp

    jointdf['prelast_lever_harp'] = jointdf.apply(lambda x: x.lever_rel_harp[-2] if x.count_lever > 1 else np.nan, axis = 1)
    jointdf['prelast_lever_abs_harp'] = jointdf.prelast_lever_harp + jointdf.trial_start_harp

    jointdf['diff_lever'] = jointdf.lever_rel_harp.apply(lambda x: np.diff(x))
    jointdf['lever_index'] = jointdf.count_lever.apply(lambda x: np.arange(x) - x)
    jointdf['lever_index_minuslast'] = jointdf.lever_index.apply(lambda x: x[:-1])                                

    jointdf['cp_arduino'] = bhvdf.cp*1000
    jointdf['cp_harp'] = jointdf.apply(lambda x: convert_timestamp(x.cp_arduino,
                                    [0, x.trial_duration_arduino],
                                    [0, x.trial_duration_harp]), axis = 1)


    #NEW - 23 JULY

    jointdf['poke_rel_arduino'] = bhvdf.poke_rel

    jointdf['poke_rel_harp'] = jointdf.apply(lambda x:
                                    convert_timestamp(x.poke_rel_arduino,
                                    [0, x.trial_duration_arduino],
                                    [0, x.trial_duration_harp]), axis = 1)

    jointdf['poke_abs_harp'] = jointdf.poke_rel_harp + jointdf.trial_start_harp


    jointdf.to_pickle(rf'{PATH_SAVE_DFS}\{animal}_{date}_NEWjointdf.pkl')


    """
    .########.########..####....###....##..........########.####..######..
    ....##....##.....##..##....##.##...##..........##........##..##....##.
    ....##....##.....##..##...##...##..##..........##........##..##.......
    ....##....########...##..##.....##.##..........######....##..##...####
    ....##....##...##....##..#########.##..........##........##..##....##.
    ....##....##....##...##..##.....##.##..........##........##..##....##.
    ....##....##.....##.####.##.....##.########....##.......####..######..
    """

    #### ADD IF CONDITION TO ONLY PLOT IF THERE ARE NOT NANS / if the fig exists

    print('plotting trial figs...')

    for tt in jointdf.trialno.values:
        all_lever_presses = np.hstack(jointdf.query(f'trialno == {tt}').lever_rel_harp.values)
        t_trial = np.hstack(jointdf.query(f'trialno == {tt}').t_trial_harp.values)

        tomato = np.hstack(jointdf.query(f'trialno == {tt}').tdtomato_poly_flat)
        continuous_encoder = np.hstack(jointdf.query(f'trialno == {tt}').ds_continuous_encoder)
        encoder_vel = np.hstack(jointdf.query(f'trialno == {tt}').ds_encoder_vel)
        gfp = np.hstack(jointdf.query(f'trialno == {tt}').gfp_poly_flat)



        fig, axs = plt.subplots(2,1, figsize = (10,4), sharex = True, tight_layout = True)

        for lvr in all_lever_presses:
            for ii in range(2):
                axs[ii].axvline(lvr, color = 'grey', lw = 0.5)

        axs[0].plot(t_trial, zscore(np.hstack(jointdf.query(f'trialno == {tt}').tdtomato_poly_flat)), color = 'red', lw = 1, label = 'tdTomato')
        axs[0].plot(t_trial, zscore(np.hstack(jointdf.query(f'trialno == {tt}').gfp_poly_flat)), color = 'green', lw = 1, label = 'dLight')

        axs_encoder = axs[0].twinx()
        axs_encoder.plot(t_trial, continuous_encoder, lw = 1, alpha = 0.5, label = 'enc_pos')
        #axs[0].plot(t_trial, zscore(encoder_vel), color = 'orange', lw = 1, alpha = 0.5)

        axs[0].set_ylabel('signals')

        axs[1].plot(t_trial, zscore(np.hstack(jointdf.query(f'trialno == {tt}').DA_poly_session)), color = 'teal', lw = 1, label = 'reg w tdTomato only')
        axs[1].set_ylabel('DA')

        ## no longer doing regressions on a trial by trial basis
        #da_trial = np.hstack(jointdf.query(f'trialno == {tt}').gfp_poly_flat) - get_prediction(np.hstack(jointdf.query(f'trialno == {tt}').tdtomato_poly_flat),np.hstack(jointdf.query(f'trialno == {tt}').gfp_poly_flat))
        #axs[1].plot(t_trial, zscore(da_trial), color = 'teal', lw = 1, label = 'robust trial')



        axs[1].plot(t_trial, zscore(np.hstack(jointdf.query(f'trialno == {tt}').DA_encoder_session)), color = 'purple', lw = 1, label = 'reg also w encoder pos and vel')

        ## trial regression with encoder - skipped
        #X = np.column_stack([zscore(tomato), zscore(continuous_encoder), zscore(encoder_vel)])
        #X_with_const = sm.add_constant(X)
        #mod = sm.QuantReg(gfp, X_with_const)
        #res = mod.fit(q = 0.5)

        #axs[2].plot(t_trial, zscore(res.resid), color = 'teal', lw = 1, label = 'encoder trial')
        #axs[2].set_ylabel('tdTomato, enc_pos, _vel', fontsize = 12)

        for ii in range(2):
            axs[ii].legend(frameon = False, fontsize = 8)

        axs[-1].set_xlabel('time since reward (s)')

        figtitle = f"{animal} {date} | trial {tt} | regressions with tdTomato and encoder data"
        fig.suptitle(figtitle)

        plt.savefig(rf"{PATH_SAVE_FIGS}\{figtitle.replace('|', '_')}.png", dpi = 300)
        plt.close()



    """
    .########.##.....##.########....########.####..######..
    ....##....##.....##.##..........##........##..##....##.
    ....##....##.....##.##..........##........##..##.......
    ....##....#########.######......######....##..##...####
    ....##....##.....##.##..........##........##..##....##.
    ....##....##.....##.##..........##........##..##....##.
    ....##....##.....##.########....##.......####..######..
    """

    print('plotting heatmaps aligned to events for the different channels')

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    from matplotlib.colors import ListedColormap
    cmap_FI = ListedColormap(list(color_FI_blocks))
    cmap_rwd = ListedColormap(list(color_rwd_blocks))

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

    if exp == 'c':
        exp_cond_values = jointdf['n_protocols'].values
        cmap_cond = cmap_rwd
    else:
        exp_cond_values = jointdf['FI'].values
        cmap_cond = cmap_FI

    for loc in ['center left', 'center right']:
        ax_index = inset_axes(axs[-2,-1], width="5%", height="100%", loc=loc, borderpad=0)
        ax_index.matshow(exp_cond_values.reshape(len(exp_cond_values),1), aspect = 'auto', cmap = cmap_cond, origin = 'lower')
        ax_index.set_axis_off()

    figtitle = f'{animal} {date} | experiment {determine_experiment(jointdf)} | channel traces aligned to events'

    fig.suptitle(figtitle)

    plt.savefig(rf"{PATH_SAVE_FIGS}\{figtitle.replace('|', '_')}.png", dpi = 300)


    """
    .########.##.....##.########.
    .##........##...##..##.....##
    .##.........##.##...##.....##
    .######......###....########.
    .##.........##.##...##.......
    .##........##...##..##.......
    .########.##.....##.##.......
    """ 

    print('plotting DA transients split by block condition')

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

    plt.savefig(rf"{PATH_SAVE_FIGS}\{figtitle.replace('|', '_')}.png", dpi = 300)



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



if __name__ == '__main__':
    main()

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