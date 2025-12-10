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

from ratcode.config.paths import PATH_STORE_PICKLES, DROPBOX_TASK_PATH
from ratcode.common.logging import determine_experiment
from ratcode.common.colorcodes import *
from ratcode.behavior import change_point

from ratcode.init import setup
setup()
# %%
animal = 'Ruthenium'
date = '251114'
# %%
THIS_PICKLE_PATH = glob.glob(rf"{PATH_STORE_PICKLES}\{animal}_{date}*.pkl")[0]
# %%
df = pd.read_pickle(THIS_PICKLE_PATH)
# %%
plt.plot(df.trial_duration)
# %%
sns.histplot(x = np.hstack(df.first_press_s.values))#, hue = 'FI')
# %%
if df.bool_block[0]:
    df = df[df.lever_rel.apply(lambda x: len(x)) != 0]
    df.reset_index(inplace = True, drop = True)
    df.trialno = df.index+1

df['FI'] = (df.FI/1000).astype(int)
#%%
df['lever_rel_s'] = df.lever_rel.apply(lambda x: x/1000)
#%%
df['cp'] = df.trialno.apply(lambda x: change_point.accepted_cp_Gallistel(x,2,df,'lever_rel',True)[0])
#%%    
df['count_lever'] = df.lever_rel.apply(lambda x: len(x))
df['cp_after_FI'] = df.cp > df.FI
df['cp'] = df.apply(lambda x: np.nan if (x.count_lever < 3 or x.cp_after_FI) else x.cp, axis = 1)
df['bool_cp'] = ~df.cp.isna()
#%%
df['cp_normalised'] = df.cp/df.FI

#%%

def align_lvr_on_cp(cp, lvr_array, exclude_zero = True):

    if np.isnan(cp):
        aligned = np.nan
    
    else:
        aligned = lvr_array/1000 - cp

        if exclude_zero:
            cpindex = np.where(aligned >= 0)[0][0]
            aligned = np.hstack(list(set(aligned) - set([aligned[cpindex]])))

    return aligned


df['lvr_aligned_cp'] = df.apply(lambda x: align_lvr_on_cp(x.cp, x.lever_rel), axis = 1)

    df['interpress_s'] = df.lever_rel.apply(lambda x: np.diff(x)/1000)

    df['lever_after_cp'] = df.apply(lambda x: x.lever_rel_s[x.lever_rel_s > x.cp], axis = 1)
    df['interpress_after_cp'] = df.lever_after_cp.apply(lambda x: np.diff(x))

    df['press_rate_av'] = df.lever_after_cp.apply(lambda x: len(x)/(x[-1] - x[0]) if len(x)> 0 else np.nan)
    df['inverse_interpress_av'] = df.interpress_after_cp.apply(lambda x: 1/np.mean(x))

    df['block_transition'] = df.blockno.diff().fillna(0).astype(bool)
    df['trial_since_block_transition'] = df.groupby('blockno').cumcount()

    df['cp_trial_since_block_transition'] = df.groupby('blockno').apply(
        lambda group: group['bool_cp'].cumsum()-1
    ).reset_index(drop=True)

    df['cp_trial_since_block_transition'] = df['cp_trial_since_block_transition'].where(df['bool_cp'], np.nan)
#%%

exp = determine_experiment(df)

figtitle = f'{animal} {date} | experiment {exp}'


fig, axs = plt.subplots(2,3, figsize = (10,10), facecolor='w', tight_layout = True, sharex = 'col')
#plt.figure(figsize = (20,10), facecolor='w', tight_layout = True)

plt.suptitle(figtitle)

#gs = plt.GridSpec(1,2)

#plt.show()

sns.histplot(ax = axs[0,0], data = df, x = 'cp', hue = 'FI',
             palette = color_FI_blocks, element = 'step', stat = 'density', common_norm=False)

sns.scatterplot(ax = axs[1,0], data = df.explode('lever_rel_s'), y = 'trialno', x = 'lever_rel_s',
                color = 'grey', s = 10)

sns.histplot(ax = axs[0,1], data = df, x = 'cp_normalised', hue = 'FI',
             palette = color_FI_blocks, element = 'step', stat = 'density', common_norm=False)


axs[-1,0].set_xlabel('time since reward (s)')


plt.savefig(rf'{DROPBOX_TASK_PATH}/analysis_plots/daily_reports/{figtitle.replace('|','_')}.png', transparent = False)

#%%

def produce_daily_fig_BLOCKS():

    exp = determine_experiment(df)
    figtitle = f'{animal} {date} | experiment {exp}'

    plt.figure(figsize = (20,20), facecolor='w', tight_layout = True)
    plt.suptitle(figtitle)
    #skeleton
    gs = gridspec.GridSpec(5,4)

    ax_rasters = plt.subplot(gs[0:3,0:2])
    ax_cpsession = plt.subplot(gs[0:3,2])
    ax_pressrate_session = plt.subplot(gs[0:3,3])

    ax_trialrate = plt.subplot(gs[4,0])
    ax_firstpress = plt.subplot(gs[3,0])

    ax_cp = plt.subplot(gs[3,2])
    ax_cpnormalised = plt.subplot(gs[3,1])

    ax_interpress = plt.subplot(gs[4,2])
    ax_pressrate = plt.subplot(gs[4,3])

    ax_press_rate_box = plt.subplot(gs[3,3])
    ax_cpblocktransition = plt.subplot(gs[4,1])

    ##### rasters
    sns.scatterplot(ax = ax_rasters, data = df.explode('lever_rel_s').reset_index(), x = 'lever_rel_s', y = 'trialno', hue = 'FI',
                    hue_order=[15,30,60], palette=color_FI_blocks, s = 20)
    ax_rasters.set_xlim(0,60*1.05)
    ax_rasters.set_ylim(0)
    ax_rasters.set_xlabel('t since rwd (s)')
    ax_rasters.set_ylabel('trial #')
    #ax_rasters.set_title('session overview')


    ###interpress interval
    sns.histplot(ax = ax_interpress, data = df.explode('interpress_s').reset_index().query('interpress_s < 3'), x = 'interpress_s', hue = 'FI', hue_order=FI_list,
                 palette = color_FI_blocks, stat = 'density', common_norm=False, element = 'step', binwidth=.25)
    ax_interpress.set_xlabel('t (s)')
    ax_interpress.set_ylabel('interpress interval dist')


    ###theoretical limit - trial rate
    ax_trialrate.plot(df.FI.cumsum(), df.trialno, label = 'FI rate', color = 'black', ls = 'dashed')
    ax_trialrate.plot(df.trial_duration_s.cumsum(), df.trialno, label = 'trial rate', color = 'grey')
    #ax_trialrate.set_title('trial number versus theoretical limit')
    ax_trialrate.set_xlabel('t (s)')
    ax_trialrate.set_ylabel('trial #')
    ax_trialrate.set_xlim(0)
    ax_trialrate.set_ylim(0)
    ax_trialrate.legend()

    ###first press dist
    sns.histplot(ax = ax_firstpress, data = df, x = 'first_press_s', hue = 'FI', hue_order=FI_list, palette=color_FI_blocks,
                 stat = 'density', common_norm=False, element = 'step')
    ax_firstpress.set_ylabel('first press dist')
    ax_firstpress.set_xlabel('t (s)')
    ax_firstpress.set_xlim(0, 60 * 1.5)

    ##### cp distribution - considering theta = 2 and only looking at the first cp
    sns.histplot(ax = ax_cp, data = df.explode('cp').reset_index(), x = 'cp', hue = 'FI', hue_order=FI_list, palette=color_FI_blocks,
                 stat = 'density', common_norm=False, element = 'step')
    ax_cp.set_ylabel('change point dist')
    ax_cp.set_xlabel('t (s)')
    ax_cp.set_xlim(0, 60 * 1.05)

    ### cp normalised
    sns.histplot(ax = ax_cpnormalised, data = df.explode('cp_normalised').reset_index(), x = 'cp_normalised', hue = 'FI', hue_order=FI_list, palette=color_FI_blocks,
                 stat = 'density', common_norm=False, element = 'step')
    ax_cpnormalised.set_ylabel('change point normalised dist')
    ax_cpnormalised.set_xlabel('t normalised to FI')
    ax_cpnormalised.set_xlim(0, 1.05)


    ###pressing rate aligned on cp
    binsize = 0.5
    for ii in range(3):
        fi = FI_list[ii]
        presses = np.hstack(df.query(f'FI == {fi}').lvr_aligned_cp.dropna().values)
        ntrials = len(df.query(f'FI == {fi}').lvr_aligned_cp.dropna())
        rate = np.histogram(presses, np.arange(-2,6,binsize))
        ax_pressrate.plot(rate[1][:-1], rate[0]/(binsize * ntrials), color = color_FI_blocks[ii])
    #ax_pressrate.set_title('pressing rate aligned on cp')
    ax_pressrate.set_xlabel('t since transition (s)')
    ax_pressrate.set_ylabel('pressing rate (Hz)')
    ax_pressrate.set_ylim(0)


    ###cp throughout the session
    sns.scatterplot(ax = ax_cpsession, data = df, y = 'trialno', x = 'cp',
                    hue = 'FI', hue_order = FI_list, palette=color_FI_blocks)
    ax_cpsession.set_xlabel('trial #')
    ax_cpsession.set_ylabel('cp (s)')
    ax_cpsession.set_ylim(0)
    ax_cpsession.set_xlim(0,60)

    ### pressing rate in the session
    sns.scatterplot(ax = ax_pressrate_session, data = df, y = 'trialno', x = 'press_rate_av',
                    hue = 'FI', hue_order=FI_list, palette=color_FI_blocks)
    sns.scatterplot(ax = ax_pressrate_session, data = df, y = 'trialno', x = 'inverse_interpress_av',
                    hue = 'FI', hue_order=FI_list, palette=color_FI_blocks, marker='s')
    ax_pressrate_session.set_xlabel('trial #')
    ax_pressrate_session.set_ylabel('av pressing rate')
    ax_pressrate_session.set_ylim(0)
    ax_pressrate_session.set_xlim(0)
    ax_pressrate_session.annotate('circle = average pressing rate (counts/t)\nsquare = inverse of the interpress interval',
                                  xy = (.02,.02) ,xycoords = ('axes fraction'))


    ### boxplot pressing rates
    aux_press_rate_av = df.explode(['press_rate_av']).dropna().get(['press_rate_av', 'FI']).reset_index(drop = True)
    aux_inverse_interpress_av = df.explode(['inverse_interpress_av']).dropna().get(['inverse_interpress_av', 'FI']).reset_index(drop = True)
    aux_press_rate_av.rename(columns = {'press_rate_av' : 'press_rate'}, inplace = True)
    aux_inverse_interpress_av.rename(columns = {'inverse_interpress_av' : 'press_rate'}, inplace = True)
    aux_press_rate_av['method'] = 'press_rate_av'
    aux_inverse_interpress_av['method'] = 'inverse_interpress_av'
    bb = pd.concat([aux_press_rate_av, aux_inverse_interpress_av]).reset_index(drop = True)

    sns.boxplot(ax = ax_press_rate_box , data = bb, x = 'press_rate', y = 'FI',
                hue = 'method', orient = 'h', showfliers = False,
                hue_order = ['press_rate_av', 'inverse_interpress_av'], palette=['#b0b0b0', '#e3e3e3'])

    ax_press_rate_box.sharex(ax_pressrate_session)


    ### cp after block transition
    blocks_w_cp = df.groupby('blockno')['cp'].apply(lambda x: x.notnull().any()).index.tolist()
    cc = []
    yy = []
    for bb in blocks_w_cp:
        #color corresponds to the color of the FI of the previous block, or the current one if it's the first block of the session
        if bb == 1:
            cc.append(df.query(f'blockno == {bb}').FI.values[0])
        else:
            cc.append(df.query(f'blockno == {bb-1}').FI.values[0]) 
        yy.append(df.query(f'blockno == {bb}').cp_normalised.dropna().values[0])

    cc = [color_FI_blocks_dic[ii] for ii in cc]

    sns.lineplot(ax = ax_cpblocktransition, data = df, x = 'cp_trial_since_block_transition', y = 'cp_normalised', hue = 'FI', estimator = None, units = 'blockno',
                 hue_order = FI_list, palette=color_FI_blocks)
    ax_cpblocktransition.scatter(np.zeros(len(blocks_w_cp)), yy, c = cc, s = 30)
    ax_cpblocktransition.annotate('linecolor = current block\npoint at zero = prev block', xy = (0.5,.92) ,xycoords = ('axes fraction'))
    ax_cpblocktransition.set_xlabel('transition trial # since block transition')
    ax_cpblocktransition.set_ylabel('cp (normalised)')

    figures.remove_legend([ax_cpblocktransition, ax_interpress, ax_rasters, ax_cpnormalised, ax_cp, ax_cpsession, ax_pressrate_session])

    plt.savefig(rf'{dropbox_path}/analysis_plots/daily_reports/{figtitle.replace('|','_')}.png', transparent = False)

# %%
