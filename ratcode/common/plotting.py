import matplotlib.pyplot as plt
import matplotlib



click_order = [False, True]
click_palette = ['#bababa', '#7a7a7a'] #following the order of the click_order



#color_pal_dic = {
#    "FI 30 click False": "#81a6fc",
#    "FI 30 click True": "#2e6dff",
#    "FI 60 click False": "#77d674",
#    "FI 60 click True": "#0b6b07"
#}

#conds_block_order = list(color_blocks_dic.keys())

#color_pal = []
#for cond in session_vars_order:
#    color_pal.append(color_pal_dic[cond])

#color palette for different rewards (1st is the smaller rwd)
newc = ['#FF7E6B','#8C5E58']


color_nprots_blocks = ["#636f82", "#81a6fc", "#2e6dff"]

color_blocks_dic = {
    "FI15 rwd7": "#cba6e3",
    "FI15 rwd14": "#b979e0",
    "FI15 rwd15": "#b979e0",
    "FI15 rwd28": "#980cf0",
    "FI15 rwd30": "#980cf0",
    "FI30 rwd7": "#879aad",
    "FI30 rwd14": "#81a6fc",
    "FI30 rwd15": "#81a6fc",
    "FI30 rwd28": "#2e6dff",
    "FI30 rwd30": "#2e6dff",
    "FI30 rwd60": "#2e6dff", # shouldn't exist
    "FI60 rwd7": "#87ad93",    
    "FI60 rwd14": "#77d674",
    "FI60 rwd15": "#77d674",
    "FI60 rwd28": "#0b6b07",
    "FI60 rwd30": "#0b6b07"
}

agg_blocks_FI_dic = {
    15: "#cba6e3",
    30: "#81a6fc",
    60: "#77d674"
}

color_FI_blocks = list(agg_blocks_FI_dic.values())



### from figures\figures.py


""" figures.py
This module takes care of everything that is figure and plotting related
"""

color_blocks_dic = {
    "FI15 rwd7": "#cba6e3",
    "FI15 rwd14": "#b979e0",
    "FI15 rwd15": "#b979e0",
    "FI15 rwd28": "#980cf0",
    "FI15 rwd30": "#980cf0",
    "FI30 rwd7": "#879aad",
    "FI30 rwd14": "#81a6fc",
    "FI30 rwd15": "#81a6fc",
    "FI30 rwd28": "#2e6dff",
    "FI30 rwd30": "#2e6dff",
    "FI30 rwd60": "#2e6dff", # shouldn't exist
    "FI60 rwd7": "#87ad93",    
    "FI60 rwd14": "#77d674",
    "FI60 rwd15": "#77d674",
    "FI60 rwd28": "#0b6b07",
    "FI60 rwd30": "#0b6b07"
}

conds_block_order = list(color_blocks_dic.keys())



def verify_2_session_conds(animal, date, blocksdf):
    '''
    Checks if a daily figure should be produced for the blocks experiments.
    If so, it produces the figure, via the produce_daily_2FI function

    Parameters:
    animal: 
    date: 
    blocksdf: dataframe where the animal and date are
    '''

    newdf = blocksdf.query(f'animal == "{animal}" and date == "{date}"')

    llorder = [conds_block_order.index(condition) for condition in newdf.FI_nprots.unique()]
    zzorder = zip(llorder, newdf.FI_nprots.unique())
    this_sess_conds = [x for _,x in sorted(zzorder)]
    this_sess_colors = [color_blocks_dic[cond] for cond in this_sess_conds]

    if len(this_sess_conds) > 1:
        produce_daily_2FI(newdf, this_sess_conds, this_sess_colors)

    else:
        print(f'no blocks found for {animal} {date}')


def produce_daily_2FI(newdf, this_sess_conds, this_sess_colors):
    '''
    Saves daily figure (blocks experiments)
    '''

    trials = newdf.reset_index().index.values + 1
    #newdf.trialno.values
    newdf['trialno_within_session'] = trials

    figtitle = f'{newdf.animal.values[0]} {newdf.date.values[0]} BLOCKS'

    newdf['count_lever'] = newdf.apply(lambda x: len(x.lever_rel) if x.bool_lever == True else 0, axis = 1)
    newdf['out_of_FI'] = newdf.apply(lambda x: 'beforeFI' if x.first_press_s < x.FI else 'afterFI', axis = 1)


    # trials where the first press is done before FI termination
    beforeFI_trials = newdf.query('out_of_FI == "beforeFI"').trialno.values

    #lever dataframe
    levertrue = newdf.query('bool_lever == True')
    leverdf = pd.DataFrame(columns = ['trial #', 't (s)', 'trial duration (s)'])
    leverdf['trial #'] = np.hstack(levertrue.apply(lambda x: np.full(x.count_lever, x.trialno_within_session), axis = 1).values)
    leverdf['t (s)'] = np.hstack(levertrue.lever_rel.values/1000)
    leverdf['trial duration (s)'] = np.hstack(levertrue.apply(lambda x: np.full(x.count_lever, x.trial_duration), axis = 1).values/1000)
    leverdf['kind'] = 'lever'
    leverdf['FI'] = np.hstack(levertrue.apply(lambda x: np.full(x.count_lever, x.FI), axis = 1).values)
    leverdf['FI_nprots'] = np.hstack(levertrue.apply(lambda x: np.full(x.count_lever, x.FI_nprots), axis = 1).values)

    #valve dataframe
    valvetrue = newdf.query('bool_pump == True')
    valvedf = pd.DataFrame(columns = ['trial #', 't (s)'])
    valvedf['trial #'] = np.hstack(valvetrue.trialno_within_session.values)
    valvedf['t (s)'] = np.hstack(valvetrue.pump_rel.values/1000)
    valvedf['kind'] = 'valve'
    valvedf['FI'] = valvetrue.FI.values
    valvedf['FI_nprots'] = valvetrue.FI_nprots.values

    #cp
    cptrue = newdf.query('bool_cp == True')
    cpdf = pd.DataFrame(columns = ['trial #', 't (s)'])
    cpdf['trial #'] = np.hstack(cptrue.trialno_within_session.values)
    cpdf['t (s)'] = np.hstack(cptrue.cp.values)
    cpdf['kind'] = 'cp'
    cpdf['FI'] = cptrue.FI.values
    cpdf['FI_nprots'] = cptrue.FI_nprots.values

    #rateH
    rateHdf = pd.DataFrame(columns = ['trial #', 'rate (Hz)'])
    rateHdf['trial #'] = np.hstack(cptrue.trialno_within_session.values)
    rateHdf['rate (Hz)'] = np.hstack(cptrue.rateH.values)
    rateHdf['kind'] = 'rateH'
    rateHdf['FI'] = cptrue.FI.values
    rateHdf['FI_nprots'] = cptrue.FI_nprots.values

    alljointdf = pd.concat([leverdf, valvedf, cpdf])

    newdf['all_presses_1s_bins'] = newdf.apply(lambda x: np.arange(0, np.ceil(x.trial_duration_s)), axis = 1)
    newdf['all_presses_1s_bins_toplot'] = newdf['all_presses_1s_bins'].apply(lambda x: x[:-1])
    newdf['all_presses_1s_count'] = newdf.apply(lambda x: np.histogram(x.lever_rel/1000, bins = x.all_presses_1s_bins)[0], axis = 1)
    newdf['all_presses_trialsno'] = newdf.apply(lambda x: np.full(len(x.all_presses_1s_count), x.trialno), axis = 1)

    # for ax3
    all_levers_toplotdf = pd.DataFrame()
    all_levers_toplotdf['trial #'] = np.hstack(newdf.apply(lambda x: np.full(len(x.all_presses_1s_count), x.trialno_within_session), axis = 1))
    all_levers_toplotdf['t (s)'] = np.hstack(newdf.apply(lambda x: np.arange(0, np.ceil(x.trial_duration_s))[:-1], axis = 1))
    all_levers_toplotdf['pressing rate (Hz)'] = np.hstack(newdf.apply(lambda x: np.histogram(x.lever_rel/1000, bins = x.all_presses_1s_bins)[0], axis = 1))
    all_levers_toplotdf['FI'] = np.hstack(newdf.apply(lambda x: np.full(len(x.all_presses_1s_count), x.FI), axis = 1))
    all_levers_toplotdf['FI_nprots'] = np.hstack(newdf.apply(lambda x: np.full(len(x.all_presses_1s_count), x.FI_nprots), axis = 1))
    all_levers_toplotdf['t normalised'] = all_levers_toplotdf.apply(lambda x: x['t (s)']/ x.FI, axis = 1)

    # for ax4
    interdf = pd.DataFrame()
    interdf['t (s)'] = np.hstack(newdf.interpress_aftercp.dropna().values)
    interdf['FI'] = np.hstack(newdf.apply(lambda x: np.full(len(x.interpress_aftercp), x.FI) if x.bool_cp == True else [], axis = 1))
    interdf.FI = interdf.FI.apply(lambda x: int(x))
    interdf['FI_nprots'] = np.hstack(newdf.apply(lambda x: np.full(len(x.interpress_aftercp), x.FI_nprots) if x.bool_cp == True else [], axis = 1))
    optimaltrialdf = pd.DataFrame()
    first_trial_start = newdf.trial_start.values[0]/1000
    optimaltrialdf['actual trial #'] = newdf.trialno_within_session - 1 
    optimaltrialdf['t (s)'] = newdf.trial_start/1000 - first_trial_start

    optimaltrialdf['FI'] = newdf.FI
    optimaltrialdf['theoretical trial #'] = optimaltrialdf.apply(lambda x: x['t (s)']/x.FI, axis = 1)


    newdf['lever_aligned_cp'] = newdf.apply(lambda x: x.lever_rel - x.cp*1000 if x.bool_cp == True else np.nan, axis = 1)
    newdf['lever_aligned_cp_dropfirstpress'] = newdf.lever_aligned_cp.apply(lambda x: np.delete(x, np.where(x>0)[0][0]) if type(x) == np.ndarray else np.nan)

    newdf['last_press_beforeFI_index'] = newdf.apply(lambda x: np.where(x.lever_rel < x.FI *1000)[0], axis = 1)
    newdf['bool_last_press'] = newdf.last_press_beforeFI_index.apply(lambda x: True if len(x)>0 else False)
    newdf['last_press_beforeFI'] = newdf.apply(lambda x: x.lever_rel[x.last_press_beforeFI_index[-1]] if x.bool_last_press == True else [], axis = 1)
    newdf['last_press_beforeFI_aligned_cp'] = newdf.apply(lambda x: x.last_press_beforeFI - x.cp*1000 if x.bool_last_press == True else [], axis = 1)

    newdf['FI_aligned_cp'] = newdf.apply(lambda x: (x.FI - x.cp)*1000, axis = 1)

    newdf['FI_max_cap'] = newdf.FI.apply(lambda x: x/3*1000)
    newdf['FI_aligned_cp_capped'] = newdf.apply(lambda x: x.FI_aligned_cp if x.FI_aligned_cp < x.FI_max_cap else (x.FI_max_cap if x.bool_cp == True else np.nan), axis = 1)


    xlower = -2000
    binw = 250
    newdf['bins'] = newdf.apply(lambda x: np.arange(xlower, int(x.FI_aligned_cp_capped), binw) if x.bool_cp == True and x.bool_last_press == True else [], axis = 1)
    newdf['counts_cp'] = newdf.apply(lambda x: np.histogram(x.lever_aligned_cp_dropfirstpress, x.bins)[0]*1000/binw if x.bool_cp == True and x.bool_last_press == True else [], axis = 1)
    newdf['len_counts'] = newdf.counts_cp.apply(lambda x: len(x))
    newdf['middle'] = newdf.apply(lambda x: x.bins[:x.len_counts] + binw/2 if x.bool_cp == True and x.bool_last_press == True else [], axis = 1)

    newdf['FI_nprots_long'] = newdf.apply(lambda x: [x.FI_nprots]*x.len_counts, axis = 1)

    newdf['animal_long'] = newdf.apply(lambda x: [x.animal]*x.len_counts, axis = 1)

    press_currbio = pd.DataFrame()
    press_currbio['middle'] = np.hstack(newdf.query(f'bool_cp == True').middle.values)
    press_currbio['middle_s'] = press_currbio.middle.apply(lambda x: x/1000)
    press_currbio['counts_cp'] = np.hstack(newdf.query(f'bool_cp == True').counts_cp.values)
    press_currbio['FI_nprots'] = np.hstack(newdf.query(f'bool_cp == True').FI_nprots_long.values)
    press_currbio['animal'] = np.hstack(newdf.query('bool_cp == True').animal_long.values)

    #when do animals press for the first time - trials with cp versus trials with no cp
    firstpressVScptrial = np.empty((2,2))
    firstpressVScptrial[1,1] = len(newdf.query('out_of_FI == "beforeFI" and bool_cp == True' ))
    firstpressVScptrial[1,0] = len(newdf.query('out_of_FI == "beforeFI" and bool_cp == False' ))
    firstpressVScptrial[0,1] = len(newdf.query('out_of_FI == "afterFI" and bool_cp == True' ))
    firstpressVScptrial[0,0] = len(newdf.query('out_of_FI == "afterFI" and bool_cp == False' ))


    binW = 1
    FImax = newdf.FI.max()
    plt.figure(figsize = (20,20), facecolor='w')
    plt.suptitle(figtitle)
    #skeleton
    gs = gridspec.GridSpec(5,4)

    ax_scatter = plt.subplot(gs[1:4,0:2])
    ax_avresp = plt.subplot(gs[0,0:2])
    ax_avresp_norm = plt.subplot(gs[0,2:4])


    ax_1stdist = plt.subplot(gs[4,0:1])
    ax_cpdist = plt.subplot(gs[4,1:2])


    ax_blocks = plt.subplot(gs[3,2:4])


    ax4 = plt.subplot(gs[1,2])
    ax5 = plt.subplot(gs[2,2])
    ax8 = plt.subplot(gs[4,2])
    ax9 = plt.subplot(gs[1,3])
    ax10 = plt.subplot(gs[2,3])

    ax13 = plt.subplot(gs[4,3])


    ##### rasters / session overview
    sns.scatterplot(ax = ax_scatter, data = alljointdf, x = 't (s)', y = 'trial #', hue = 'kind', marker = '|' )
    ax_scatter.set_xlim(0,FImax*1.05)
    ax_scatter.set_ylim(0, max (alljointdf['trial #'].values))
    ax_scatter.legend([], frameon = False)
    ax_scatter.set_title('session overview')

    ##### average pressing rate
    sns.lineplot(ax = ax_avresp, data = all_levers_toplotdf, x = 't (s)', y = 'pressing rate (Hz)', hue = 'FI_nprots', hue_order=this_sess_conds, palette=this_sess_colors, errorbar=('ci', 0))#'sd')
    ax_avresp.set_xlim(0,FImax*1.05)
    ax_avresp.set_ylim(0)
    ax_avresp.set_title('average response rate over session')
    ax_avresp.legend([], frameon = False)

    #### average pressing rate - normalised time
    sns.lineplot(ax = ax_avresp_norm, data = all_levers_toplotdf, x = 't normalised', y = 'pressing rate (Hz)', hue = 'FI_nprots', hue_order = this_sess_conds, palette = this_sess_colors, errorbar=('ci', 0))
    ax_avresp_norm.set_title('average response rate over session - normalised time')
    ax_avresp_norm.set_xlim(0, 1.05)

    #### block structure
    sns.lineplot(ax = ax_blocks, data = newdf, x = 'trialno_within_session', y = 'FI', label = 'FI')
    sns.lineplot(ax = ax_blocks, data = newdf, x = 'trialno_within_session', y = 'n_protocols', label = 'n_protocols')


    ##### interpress interval
    sns.histplot(ax = ax4, data = interdf, x = 't (s)', hue = 'FI_nprots', stat = 'density', fill = False, element = 'step', common_norm = False,  binwidth = .1, hue_order=this_sess_conds, palette=this_sess_colors)
    ax4.set_xlim(0,3)
    ax4.set_ylim(0)
    ax4.set_xlabel('t (s)')
    ax4.set_title('interpress interval distribution')
    ax4.legend([], frameon = False)


    ##### first press distribution
    sns.histplot(ax = ax_1stdist, data = newdf, x = 'first_press_s', binwidth = binW, stat = 'density', hue = 'FI_nprots', element = 'step', common_norm=False, hue_order=this_sess_conds, palette=this_sess_colors)
    ax_1stdist.set_title('first press distribution')
    ax_1stdist.set_xlabel('t (s)')
    ax_1stdist.set_xlim(0, FImax * 1.05)
    ax_1stdist.legend([], frameon = False)


    ##### cp distribution - considering theta = 2 and only looking at the first cp
    sns.histplot(ax = ax_cpdist, data = newdf, x = 'cp', hue = 'FI_nprots', stat = 'density', binwidth = binW, element = 'step', common_norm = False, hue_order=this_sess_conds, palette=this_sess_colors)
    ax_cpdist.set_title('change point distribution (theta = 2, 1st cp)')
    ax_cpdist.set_xlabel('t (s)')
    ax_cpdist.set_xlim(0, FImax * 1.05)
    ax_cpdist.legend([], frameon = False)


    ##### pressing rate aligned on cp // current biology style
    sns.lineplot(ax = ax9, data = press_currbio, x = 'middle_s', y = 'counts_cp', hue = 'FI_nprots', hue_order=this_sess_conds, palette=this_sess_colors, ci = 0)
    ax9.set_title('pressing rate aligned on cp')
    ax9.set_xlabel('t since cp (s)')
    ax9.set_ylabel('pressing rate (Hz)')
    ax9.set_ylim(0)
    ax9.legend([], frameon = False)


    ##### cp vs trialno - single trial
    sns.scatterplot(ax = ax5, data = newdf.query('bool_cp == True'), x = 'trialno_within_session', y = 'cp', marker = '.', hue = 'FI_nprots', hue_order=this_sess_conds, palette=this_sess_colors)
    ax5.set_title('single trial cp over session')
    ax5.set_xlabel('trial #')
    ax5.set_ylabel('cp (s)')
    ax5.legend([], frameon = False)

    ##### pressing rate after cp vs trialno - single trial
    sns.scatterplot(ax = ax10, data = newdf.query('bool_cp == True'), x = 'trialno_within_session', y = 'rateH', marker = '.', hue = 'FI_nprots', hue_order=this_sess_conds, palette=this_sess_colors)
    ax10.set_title('pressing rate after cp over session')
    ax10.set_xlabel('trial #')
    ax10.set_ylabel('pressing rate (Hz)')
    ax10.legend([], frameon = False)


    sns.boxplot(ax = ax8, data = newdf, y = 'rateH', x = 'FI_nprots', hue = 'FI_nprots', hue_order=this_sess_conds, palette=this_sess_colors, showfliers = False, order = this_sess_conds, dodge = False)
    ax8.legend([], frameon = False)

    sns.heatmap(ax = ax13, data = firstpressVScptrial, annot = True, fmt = ".0f", yticklabels = ['after FI', 'before FI'], xticklabels = ['False', 'True'], cbar = False)
    ax13.set_ylabel('first press')
    ax13.set_xlabel('trial with cp?')
    ax13.set_title('trial type count')

    plt.tight_layout()
    plt.subplots_adjust(top = .95)

    plt.savefig('daily_reports/' + figtitle, transparent = False) 

## doing this!!
def produce_daily_noblocks(newdf):#, this_sess_conds, this_sess_colors):
    '''
    Saves the daily figure of a no blocks experiment
    '''

    trials = newdf.reset_index().index.values + 1
    #newdf.trialno.values
    newdf['trialno'] = trials

    FI = newdf.FI.unique()[0]
    nprots = newdf.n_protocols.unique()[0]
    click = newdf.click.unique()[0]

    figtitle = f'{newdf.animal.values[0]} {newdf.date.values[0]} NO BLOCKS - FI{FI} n_protocols{nprots} click{bool(click)}'

    newdf['count_lever'] = newdf.apply(lambda x: len(x.lever_rel) if x.bool_lever == True else 0, axis = 1)
    newdf['out_of_FI'] = newdf.apply(lambda x: 'beforeFI' if x.first_press_s < x.FI else 'afterFI', axis = 1)

    # trials where the first press is done before FI termination
    beforeFI_trials = newdf.query('out_of_FI == "beforeFI"').trialno.values

    #lever dataframe
    levertrue = newdf.query('bool_lever == True')
    leverdf = pd.DataFrame(columns = ['trial #', 't (s)', 'trial duration (s)'])
    leverdf['trial #'] = np.hstack(levertrue.apply(lambda x: np.full(x.count_lever, x.trialno), axis = 1).values)
    leverdf['t (s)'] = np.hstack(levertrue.lever_rel.values/1000)
    leverdf['trial duration (s)'] = np.hstack(levertrue.apply(lambda x: np.full(x.count_lever, x.trial_duration), axis = 1).values/1000)
    leverdf['kind'] = 'lever'

    #valve dataframe
    valvedf = pd.DataFrame(columns = ['trial #', 't (s)'])
    valvedf['trial #'] = np.hstack(newdf.trialno.values)
    valvedf['t (s)'] = np.hstack(newdf.pump_rel.values/1000)
    valvedf['kind'] = 'valve'

    #cp
    cptrue = newdf.query('bool_cp == True')
    cpdf = pd.DataFrame(columns = ['trial #', 't (s)'])
    cpdf['trial #'] = np.hstack(cptrue.trialno_within_session.values)
    cpdf['t (s)'] = np.hstack(cptrue.cp.values)
    cpdf['kind'] = 'cp'

    #rateH
    rateHdf = pd.DataFrame(columns = ['trial #', 'rate (Hz)'])
    rateHdf['trial #'] = np.hstack(cptrue.trialno_within_session.values)
    rateHdf['rate (Hz)'] = np.hstack(cptrue.rateH.values)
    rateHdf['kind'] = 'rateH'

    alljointdf = pd.concat([leverdf, valvedf, cpdf])

    newdf['all_presses_1s_bins'] = newdf.apply(lambda x: np.arange(0, np.ceil(x.trial_duration_s)), axis = 1)
    newdf['all_presses_1s_bins_toplot'] = newdf['all_presses_1s_bins'].apply(lambda x: x[:-1])
    newdf['all_presses_1s_count'] = newdf.apply(lambda x: np.histogram(x.lever_rel/1000, bins = x.all_presses_1s_bins)[0], axis = 1)
    newdf['all_presses_trialsno'] = newdf.apply(lambda x: np.full(len(x.all_presses_1s_count), x.trialno), axis = 1)

    # for ax3
    all_levers_toplotdf = pd.DataFrame()
    all_levers_toplotdf['trial #'] = np.hstack(newdf.all_presses_trialsno)
    all_levers_toplotdf['t (s)'] = np.hstack(newdf.all_presses_1s_bins_toplot)
    all_levers_toplotdf['pressing rate (Hz)'] = np.hstack(newdf.all_presses_1s_count)


    #for ax5
    #how many trials the animal could do assuming all trials have FI duration
    #vs
    #how many trials the animal actually does
    optimaltrialdf = pd.DataFrame(columns = ['t (s)', 'actual trial #',  'theoretical trial #', 'session'])
    first_trial_start = newdf.trial_start.values[0]/1000
    optimaltrialdf['actual trial #'] = newdf.trialno - 1 
    optimaltrialdf['t (s)'] = newdf.trial_start/1000 - first_trial_start
    optimaltrialdf['theoretical trial #'] = optimaltrialdf['t (s)'].apply(lambda x: x/FI)

    newdf['cp2'] = newdf.apply(lambda x: change_point.accepted_cp_Gallistel(x.trialno, 2 , newdf, 'lever_rel', bool_onlyfirst=True)[0], axis=1)
    newdf['bool_cp'] = newdf.cp2.apply(lambda x: True if x < FI else False)
    newdf['cp'] = newdf.apply(lambda x: x.cp2 if x.bool_cp == True else np.nan, axis = 1)

    # for ax8
    newdf['all_presses_aftercp'] = newdf.apply(lambda x: x.lever_rel[x.lever_rel >= x.cp*1000]/1000 if x.bool_cp == True else np.nan, axis = 1)
    newdf['rate_aftercp'] = newdf.apply(lambda x: len(x.all_presses_aftercp)/(x.trial_duration_s - x.all_presses_aftercp[0]) if type(x.all_presses_aftercp) == np.ndarray else np.nan, axis = 1)
    newdf['all_presses_after10sbeforecp'] = newdf.apply(lambda x: x.lever_rel[x.lever_rel >= x.cp*1000-10000]/1000 if x.cp != np.nan else np.nan, axis = 1)

    #aux df for pressing rate after cp
    withcp = newdf.query('bool_cp == True')
    cpdf = pd.DataFrame(columns = ['trialno', 'cp', 'all_presses_after10sbeforecp', 'trial_duration_s', 'time_spent_after10sbeforecp'])
    cpdf.trialno = withcp.trialno
    cpdf.cp = withcp.cp
    cpdf.all_presses_after10sbeforecp = withcp.all_presses_after10sbeforecp
    cpdf.trial_duration_s = withcp.trial_duration_s
    cpdf.time_spent_after10sbeforecp = cpdf.trial_duration_s - cpdf.cp - 10

    cpdf['presses_1s_bins'] = cpdf.apply(lambda x: np.arange(np.floor(x.all_presses_after10sbeforecp[0]), np.ceil(x.trial_duration_s)), axis = 1)
    cpdf['presses_1s_count'] = cpdf.apply(lambda x: np.histogram(x.all_presses_after10sbeforecp, bins = x.presses_1s_bins)[0], axis = 1)
    cpdf['presses_1s_bins_toplot'] = cpdf['presses_1s_bins'].apply(lambda x: x[:-1])
    cpdf['presses_1s_bins_toplot_alignedcpminus10'] = cpdf.apply(lambda x: x.presses_1s_bins_toplot - x.cp, axis = 1)
    cpdf['trials_len_bins'] = cpdf.apply(lambda x: np.full(len(x.presses_1s_bins_toplot),x.trialno), axis = 1)
    # need to make this tidy data to be able to plot
    plotcpdf = pd.DataFrame()
    plotcpdf['trial #'] = np.hstack(cpdf.trials_len_bins)
    plotcpdf['presses_1s_bins_toplot_alignedcpminus10'] = np.hstack(cpdf.presses_1s_bins_toplot_alignedcpminus10)
    plotcpdf['presses_1s_count'] = np.hstack(cpdf.presses_1s_count)

    #when do animals press for the first time - trials with cp versus trials with no cp
    firstpressVScptrial = np.empty((2,2))
    firstpressVScptrial[1,1] = len(newdf.query('out_of_FI == "beforeFI" and bool_cp == True' ))
    firstpressVScptrial[1,0] = len(newdf.query('out_of_FI == "beforeFI" and bool_cp == False' ))
    firstpressVScptrial[0,1] = len(newdf.query('out_of_FI == "afterFI" and bool_cp == True' ))
    firstpressVScptrial[0,0] = len(newdf.query('out_of_FI == "afterFI" and bool_cp == False' ))


    #for now else is just FI = 30, careful if including more intervals
    binW = 1 if FI == 60 else 1

    plt.figure(figsize = (20,20), facecolor='w')
    plt.suptitle(figtitle)

    #skeleton
    gs = gridspec.GridSpec(5,4)
    ax1 = plt.subplot(gs[0,0:2])
    ax2 = plt.subplot(gs[1:4,0:2])
    ax3 = plt.subplot(gs[4,0:2])
    ax4 = plt.subplot(gs[0,2])
    ax5 = plt.subplot(gs[1,2])
    ax6 = plt.subplot(gs[2,2:4])
    ax7 = plt.subplot(gs[3,2:4])
    ax8 = plt.subplot(gs[4,2])
    ax9 = plt.subplot(gs[0,3])
    ax10 = plt.subplot(gs[1,3])
    #ax11 = plt.subplot(gs[2,3])
    #ax12 = plt.subplot(gs[3,3])
    ax13 = plt.subplot(gs[4,3])


    ##### distributions lever + licks
    sns.histplot(alljointdf, x = 't (s)', hue = 'kind', ax = ax1, stat = 'density', common_norm=False, binwidth = 1)
    ax1.set_xlim(0,FI*1.05)
    ax1.set_title('licks and lever press distributions')

    ##### rasters
    sns.scatterplot(data = alljointdf, x = 't (s)', y = 'trial #', hue = 'kind', ax = ax2, marker = '|' )
    ax2.set_xlim(0,FI * 1.05)
    ax2.set_ylim(0, max (alljointdf['trial #'].values))
    ax2.legend([], frameon = False)
    ax2.set_title('session overview')

    ##### pressing rate
    sns.lineplot(data = all_levers_toplotdf, x = 't (s)', y = 'pressing rate (Hz)', ax = ax3)#, ci = 'sd')
    ax3.set_xlim(0,FI*1.05)
    ax3.set_ylim(0)
    ax3.set_title('average response rate over session')
    ax3.annotate('95% CI', xy = (0.05,.92) ,xycoords = ('axes fraction'))


    ##### interpress interval
    sns.histplot(np.hstack(newdf.lever_rel.apply(lambda x: np.diff(x)).values/1000), stat = 'density', ax = ax4, binwidth = .25)
    ax4.set_xlim(0,6)
    ax4.set_ylim(0)
    ax4.set_xlabel('t (s)')
    ax4.set_title('interpress interval distribution')
    ax4.annotate('binwidth = 250ms', xy = (0.6,.92) ,xycoords = ('axes fraction'))


    ##### trial number vs theoretical limit
    sns.lineplot(ax = ax5, data = optimaltrialdf, y = 'actual trial #', x = 'theoretical trial #', label = 'data') # hue = 'session')
    sns.lineplot(ax = ax5, data = optimaltrialdf, x = 'theoretical trial #', y = 'theoretical trial #', label = 'theoretical limit')
    ax5.set_title('trial number versus theoretical limit')
    ax5.set_xlim(0)
    ax5.set_ylim(0)


    ##### first press distribution
    sns.histplot(newdf.first_press_s, binwidth = binW ,ax = ax6, stat = 'density')
    ax6.set_title('first press distribution')
    ax6.set_xlabel('t (s)')
    ax6.set_xlim(0, FI * 1.5)
    ax6.annotate(f'binwidth = {binW}s', xy = (0.05,.92), xycoords = ('axes fraction'))


    ##### cp distribution - considering theta = 2 and only looking at the first cp
    sns.histplot(np.hstack(newdf.query('bool_cp == True').cp.values), ax = ax7, stat = 'density', binwidth = binW)
    ax7.set_title('change point distribution (theta = 2, 1st cp)')
    ax7.set_xlabel('t (s)')
    ax7.set_xlim(0, FI * 1.05)
    ax7.annotate(f'binwidth = {binW}s', xy = (0.05,.92), xycoords = ('axes fraction'))


    ##### average pressing rate aligned on cp
    sns.lineplot(ax = ax8, data = plotcpdf, x = 'presses_1s_bins_toplot_alignedcpminus10', y = 'presses_1s_count')#, ci = 'sd')#, hue = 'trial #')
    ax8.set_title('pressing rate aligned on cp')
    ax8.set_xlabel('t since cp (s)')
    ax8.set_ylabel('pressing rate (Hz)')
    ax8.set_xlim(-10, 2/3*FI)
    ax8.set_ylim(0)
    ax8.annotate('95% CI', xy = (0.05,.92) ,xycoords = ('axes fraction'))


    ##### cp vs trialno - single trial
    sns.scatterplot(data = newdf.query('bool_cp == True'), x = 'trialno', y = 'cp', marker = '.', ax = ax9)
    ax9.set_title('single trial cp over session')
    ax9.set_xlabel('trial #')
    ax9.set_ylabel('cp (s)')
    ax9.set_xlim(0)

    ##### pressing rate after cp vs trialno - single trial
    sns.scatterplot(data = newdf.query('bool_cp == True'), x = 'trialno', y = 'rate_aftercp', marker = '.', ax = ax10)
    ax10.set_title('pressing rate after cp over session')
    ax10.set_xlabel('trial #')
    ax10.set_ylabel('pressing rate (Hz)')
    ax10.set_xlim(0)

    sns.heatmap(firstpressVScptrial, annot = True, fmt = ".0f", yticklabels = ['after FI', 'before FI'], xticklabels = ['False', 'True'], cbar = False, ax = ax13)
    ax13.set_ylabel('first press')
    ax13.set_xlabel('trial with cp?')
    ax13.set_title('trial type count')


    plt.tight_layout()
    plt.subplots_adjust(top = .95)

    plt.savefig('daily_reports/' + figtitle, transparent = False)

    plt.show()




    

class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


#function for plotting the change point overview - distributions as function of theta
def plot_cp_theta(dataframe, fig_title, theta_range):
    '''
    NEED TO REVISIT THIS
    Return a figure (saves it)

    Parameters:
    dataframe: 
    fig_title: 
    theta_range: present theta range the input as [a,b]
    '''
    
    #fig skeleton

    plt.figure(figsize = (10,8), facecolor= 'w')
    plt.suptitle(fig_title)
    
    gs = gridspec.GridSpec(2,2)
    ax1 = plt.subplot(gs[0:2,0])
    ax2 = plt.subplot(gs[0,1])
    ax3 = plt.subplot(gs[1,1])


    #get "histogram" heatmap AX1

    cp_df = pd.DataFrame(columns = ['trialno', 'theta', 'cp_accepted', 'cp_all', 'LogOdds_all', 'rates', 'slices', 'rates_long'])

    #only consider trials that were correct

    for th in range(theta_range[0], theta_range[1]):
        cp_df_temp =  pd.DataFrame(columns = ['trialno', 'theta', 'cp_accepted', 'cp_all', 'LogOdds_all', 'rates', 'slices', 'rates_long'])
        cp_df_temp.trialno = dataframe.trialno
        cp_df_temp.theta = th

        cp_df_temp.cp_accepted = cp_df_temp.apply(lambda x: change_point.accepted_cp_Gallistel(x.trialno, th , df = dataframe)[0], axis=1)
        cp_df_temp.cp_all = cp_df_temp.apply(lambda x: change_point.accepted_cp_Gallistel(x.trialno, th , df = dataframe)[1], axis=1)
        cp_df_temp.LogOdds_all = cp_df_temp.apply(lambda x: change_point.accepted_cp_Gallistel(x.trialno, th , df = dataframe)[2], axis=1)
        cp_df_temp.rates = cp_df_temp.apply(lambda x: change_point.accepted_cp_Gallistel(x.trialno, th , df = dataframe)[3], axis=1)
        cp_df_temp.slices = cp_df_temp.apply(lambda x: change_point.accepted_cp_Gallistel(x.trialno, th , df = dataframe)[4], axis=1)
        cp_df_temp.rates_long = cp_df_temp.apply(lambda x: np.hstack([np.full(np.diff(x.slices)[i], x.rates[i]) for i in range(len(x.rates))]), axis = 1) 

        cp_df = pd.concat([cp_df, cp_df_temp])

    cp_df['no_cps'] = cp_df['cp_accepted'].apply(lambda x: len(x))


    
    C = np.empty((theta_range[1], max(cp_df.no_cps)+1))
    C[:] = np.nan

    for th in range(theta_range[0], theta_range[1]):
        s =  np.bincount(cp_df.groupby('theta').get_group(th).no_cps.values)
        C[th][:len(s)] = s


    # first cp distribution AX2
    firstcp_df = cp_df.query('no_cps != 0 and theta!=0')
    firstcp_df['first_cp'] = firstcp_df['cp_accepted'].apply(lambda x: x[0])

    # median, mean and var AX3
    theta_groups = list(firstcp_df.groupby('theta').groups.keys())

    medianstheta = pd.DataFrame()
    medianstheta['theta'] = theta_groups

    medians = []
    means = []
    variances = []

    for theta in theta_groups:
        medians.append(np.median(np.hstack(firstcp_df.groupby('theta').get_group(theta).first_cp)))
        means.append(np.mean(np.hstack(firstcp_df.groupby('theta').get_group(theta).first_cp)))
        variances.append(np.var(np.hstack(firstcp_df.groupby('theta').get_group(theta).first_cp)))

    medianstheta['median'] = medians
    medianstheta['mean'] = means
    medianstheta['variance'] = variances

    ## ax1
    sns.heatmap(C, annot = True, fmt = '.0f', cbar = False, ax = ax1)
    ax1.set_title('cp counts for different sensitivities theta')
    ax1.set_xlabel('# cps')
    ax1.set_ylabel('theta')

    ## ax2
    sns.kdeplot(data = firstcp_df, x = 'first_cp', hue = 'theta', ax = ax2, legend = False, common_norm=False)
    ax2.annotate('lighter colors -> smaller thetas', xy = (0.05,.92) ,xycoords = ('axes fraction'))
    ax2.set_title('first cp distribution for different values of theta') 

    ## ax3
    sns.lineplot(data = medianstheta, x = 'theta', y = 'median', label = 'median', ax = ax3)
    sns.lineplot(data = medianstheta, x = 'theta', y = 'mean', label = 'mean', ax = ax3)
    sns.lineplot(data = medianstheta, x = 'theta', y = 'variance', label = 'variance', ax = ax3)
    ax3.set_ylabel('value')
    ax3.set_title('Distribution variables for different sensitivities theta')

    plt.tight_layout()
    plt.subplots_adjust(top = .9)
    plt.savefig('change_point_analysis//multicp_' + fig_title, transparent = False)
    plt.show()


def trial_raster(trial, df, array_name, bool_savefig = False):
    '''
    Return a dictionary. Display (and can save) a figure

    Parameters:
    trial: trialno in the dataframe
    df: dataframe where the info is
    array_name: list; name of the array from where to extract the information - e.g. lever_rel
    bool_savefig: save or not the figure (default False)

    '''

    plt.figure(figsize = (8,1), tight_layout = True)

    dic = dict()

    for array in array_name:
        if array == 'cp' or array == 'valve_rel_s':
            mm = 'x' 

        else:
            mm = '|'

        raster = np.hstack(df.query(f'trialno == {trial}')[array])#.values[0]
        
        sns.scatterplot(y = np.ones(len(raster))*trial, x = raster, marker =  mm,
        linewidth=1, label = f'{array}')
        
        dic[array] = raster
    
    plt.xlim(0)
    plt.ylim(trial-.1, trial+.1)
    plt.title(trial)
    plt.yticks([])
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    if bool_savefig:
        plt.savefig(f'trials/trialraster_{trial}', transparent = False, facecolor = 'white') 
        plt.savefig(f'trials/trialraster_{trial}.eps', format = 'eps', transparent = False, facecolor = 'white') 

    plt.show()

    return dic

def swarm_cp_proportion(condition_to_split, propdf, bool_savefig = False):
    '''
    Display (and can save) a figure

    Parameters:
    condition_to_split: experimental condition, e.g. FI, click
    propdf: dataframe with the proportions of transition points
    bool_savefig: save or not the figure (default False)
    '''

    fig = plt.figure(figsize=(19, 8))

    fig.suptitle(f'Proportion of cp trials split by {condition_to_split}')

    outer = gridspec.GridSpec(1, 4, wspace=0.2, hspace=0.2)
    outer.update(left = .05, right = .95)

    for i in range(4):
        inner = gridspec.GridSpecFromSubplotSpec(1, 2,
                        subplot_spec=outer[i], wspace=0, hspace=0.1)

        axx = plt.Subplot(fig, outer[i])
        axx.set_title(animal_order[i])

        if condition_to_split == 'rwd':    

            sns.swarmplot(ax = axx, data = propdf.query(f'animal == "{animal_order[i]}" and rwdmod != 2'), 
                    x = 'session_vars', y = '#cp/#trials', hue = 'rwd_uL',
                    palette=newc)


            axx.set_ylabel('')
            axx.set_xlabel('')
            if i==0:
                axx.set_ylabel('Proportion of cp trials')


            axx.set_ylim(0,1)
            axx.set_xticklabels(['FI 30\nclick False', 'FI 30\nclick True', 'FI 60\nclick False', 'FI 60\nclick True'])

            if i != 3:
               axx.get_legend().remove()    

            fig.add_subplot(axx)

        else:    
            axx.set_xticks([])
            axx.set_yticks([])

            axx.set_xlabel('Reward rate (uL/min)', labelpad = 22)

            fig.add_subplot(axx)

            for j in range(2):

                ax = plt.Subplot(fig, inner[j])
                if condition_to_split == 'click':
                    if j == 0:
                        sns.swarmplot(ax = ax, data = propdf.query(f'animal == "{animal_order[i]}" and rwdmod != 2 and FI == 30'), 
                        x = 'rwdrate_uLmin', y = '#cp/#trials', hue = condition_to_split,
                        palette=color_pal[:2])

                    else:
                        sns.swarmplot(ax = ax, data = propdf.query(f'animal == "{animal_order[i]}" and rwdmod != 2 and FI == 60'), 
                        x = 'rwdrate_uLmin', y = '#cp/#trials', hue = condition_to_split,
                        palette=color_pal[2:])

                if condition_to_split == 'FI':
                    if j == 0:
                        sns.swarmplot(ax = ax, data = propdf.query(f'animal == "{animal_order[i]}" and rwdrate == .05 and click == 0'), 
                        x = 'rwdrate_uLmin', y = '#cp/#trials', hue = condition_to_split,
                        palette=color_pal[0:3:2])

                    else:
                        sns.swarmplot(ax = ax, data = propdf.query(f'animal == "{animal_order[i]}" and rwdrate == .05 and click == 1'), 
                        x = 'rwdrate_uLmin', y = '#cp/#trials', hue = condition_to_split,
                        palette=color_pal[1:4:2])

                ax.set_ylabel('')
                ax.set_xlabel('')
                if i==0 and j==0:
                    ax.set_ylabel('Proportion of cp trials')

                if j == 1:  
                    ax.set_yticks([])
                    ax.set_ylabel('')
                    ax.spines['left'].set_visible(False)
                if j == 0:
                    ax.spines['right'].set_visible(False)

                ax.set_ylim(0,1)

                if i == 3:
                    ax.legend().set_title(None)
                    ax.legend(frameon = False, loc = 'upper center')
                    if condition_to_split == 'click':
                        [ax.get_legend().get_texts()[k].set_text(session_vars_order[k+j*2]) for k in [0,1]]
                    if condition_to_split == 'FI':
                        [ax.get_legend().get_texts()[k].set_text(session_vars_order[j+k*2]) for k in [0,1]]


                else:
                    ax.get_legend().remove()    

                fig.add_subplot(ax)
    
    if bool_savefig:
        plt.savefig(f'summary/allanimals_propcp_split{condition_to_split}_orderedbyconds_violin', transparent = False, facecolor = 'white') 
        plt.savefig(f'summary/allanimals_propcp_split{condition_to_split}_orderedbyconds_violin.eps', format = 'eps', transparent = False, facecolor = 'white') 


def violin_comparison(bhv_feature, condition_to_split, binwidth=0.2, bool_savefig = False):

    FI = [30,60]
    click = [0,1]

    if bhv_feature == 'rateH':
        ymin = 0
        ymax = 5.2
        ylabel = '(High) pressing rate (Hz)'
        bhv_feature_title = 'Pressing rate'

    if bhv_feature == 'cp':
        ymin = 0
        ymax = 65
        ylabel = 'Transition point (s)'
        bhv_feature_title = 'Transition point'        

    if bhv_feature == 'cp_normalised':
        ymin = 0
        ymax = 1.1
        ylabel = 'Transition point (normalised)'
        bhv_feature_title = 'Normalised transition point'      

    #fig = plt.figure(figsize=(19, 8))
    fig = plt.figure(figsize=(16,4))

    fig.suptitle(f'{bhv_feature_title} split by {condition_to_split}')

    outer = gridspec.GridSpec(1, 4, wspace=0.2, hspace=0.2)
    outer.update(left = .05, right = .95)
    for i in range(4):
        inner = gridspec.GridSpecFromSubplotSpec(1, 2,
                    subplot_spec=outer[i], wspace=0, hspace=0.1)

        axx = plt.Subplot(fig, outer[i])
        axx.set_title(animal_order[i])
        
        axx.set_xlabel('Reward rate (uL/min)', labelpad = 22)

        if condition_to_split == 'rwd':


            sns.violinplot(ax = axx, data = unidf.query(f'animal == "{animal_order[i]}" and rwdmod != 2 and bool_cp == True'),#[unidf.cp3_highrate<5], 
                x = 'session_vars', y = bhv_feature, hue = 'rwd_uL',
                palette=newc,
                order = session_vars_order, 
                inner = 'quartile', bw = binwidth,
                cut = 0, split = True)


            axx.set_ylabel('')
            axx.set_xlabel('')
            if i==0:
                axx.set_ylabel(ylabel)

            axx.set_ylim(ymin,ymax)

            axx.set_xticklabels(['FI 30\nclick False', 'FI 30\nclick True', 'FI 60\nclick False', 'FI 60\nclick True'])

            if i != 3:
               axx.get_legend().remove()    

            fig.add_subplot(axx)
        
        else:

            fig.add_subplot(axx)

            axx.set_xticks([])
            axx.set_yticks([])

            for j in range(2):
                ax = plt.Subplot(fig, inner[j])
            
                if condition_to_split == 'FI':

                    if j == 0:
                        sns.violinplot(ax = ax, data = unidf.query(f'animal == "{animal_order[i]}" and rwdrate == .05 and click == {click[j]} and bool_cp == True'),#[unidf.rateH<5], 
                            x = 'rwdrate_uLmin', y = bhv_feature, hue = condition_to_split,
                            palette=color_pal[0:3:2], 
                            inner = 'quartile',
                            cut = 0, split = True, bw = binwidth)

                    else:
                        sns.violinplot(ax = ax, data = unidf.query(f'animal == "{animal_order[i]}" and rwdrate == .05 and click == {click[j]} and bool_cp == True'),#[unidf.rateH<5], 
                                x = 'rwdrate_uLmin', y = bhv_feature, hue = condition_to_split,
                                palette=color_pal[1:4:2], 
                                inner = 'quartile',
                                cut = 0, split = True, bw = binwidth)  

                    

                if condition_to_split == 'click':

                    if j == 0:
                        sns.violinplot(ax = ax, data = unidf.query(f'animal == "{animal_order[i]}" and rwdmod != 2 and FI == {FI[j]} and bool_cp == True'), 
                            x = 'rwdrate_uLmin', y = bhv_feature, hue = condition_to_split,
                            palette=color_pal[:2],
                            inner = 'quartile',
                            cut = 0, split = True, bw = binwidth)

                    else:
                        sns.violinplot(ax = ax, data = unidf.query(f'animal == "{animal_order[i]}" and rwdmod != 2 and FI == {FI[j]} and bool_cp == True'),#[unidf.cp3_highrate<5], 
                            x = 'rwdrate_uLmin', y = bhv_feature, hue = condition_to_split,
                            palette=color_pal[2:], 
                            inner = 'quartile',
                            cut = 0, split = True, bw = binwidth)  


                if i == 3:
                    ax.legend().set_title(None)
                    ax.legend(frameon = False, loc = 'upper center')
                    
                    if condition_to_split == 'FI':
                        [ax.get_legend().get_texts()[k].set_text(session_vars_order[j+k*2]) for k in [0,1]]

                    if condition_to_split == 'click':
                        [ax.get_legend().get_texts()[k].set_text(session_vars_order[k+j*2]) for k in [0,1]]
                else:
                    ax.get_legend().remove()    

                ax.set_ylabel('')
                ax.set_xlabel('')
                if i==0 and j==0:
                    ax.set_ylabel(ylabel)

                if j == 1:  
                    ax.set_yticks([])
                    ax.set_ylabel('')
                    ax.spines['left'].set_visible(False)
                if j == 0:
                    ax.spines['right'].set_visible(False)
                ax.set_ylim(ymin, ymax)
                
                
                fig.add_subplot(ax)


    if bool_savefig:
        plt.savefig(f'summary/allanimals_{bhv_feature}_split{condition_to_split}_orderedbyconds_violin', transparent = False, facecolor = 'white') 
        plt.savefig(f'summary/allanimals_{bhv_feature}_split{condition_to_split}_orderedbyconds_violin.eps', format = 'eps', transparent = False, facecolor = 'white') 

def lineplot_quantiles(bhv_feature, condition_to_split, bool_savefig = False):
    
    titledic = dict()
    titledic['cp'] = 'Transition point'
    titledic['rate'] = 'Pressing rate'
    titledic['prop'] = 'Proportion of cp trials'

    fig, axs = plt.subplots( 1,4, sharex = True, sharey = True, figsize=(16, 4), constrained_layout = True)
    plt.suptitle(f'{titledic[bhv_feature]} split by {condition_to_split}')

    #features of behaviour
    if bhv_feature == 'cp':
        xx = [16, 57]
    
    if bhv_feature == 'rate':
        xx = [0.4,3.5]
    
    if bhv_feature == 'prop':
        xx = [0.3,1]

    yy = xx

    #axes
    if condition_to_split == 'FI':
        FIx = 30
        clickx = 0
        rwdx = 1.5
        
        FIy = 60
        clicky = 0
        rwdy = 3
        
        FIx1 = 30
        clickx1 = 1
        rwdx1 = 1.5
        
        FIy1 = 60
        clicky1 = 1
        rwdy1 = 3

        lab = 'click false rwd rate 0.05'
        lab1 = 'click true rwd rate 0.05'

        yylabel = 'FI 60'
        xxlabel = 'FI 30'

        if bhv_feature == 'cp':
            xx = np.array([16,29])
            yy = 2*xx

    if condition_to_split == 'click':
        FIx = 30
        clickx = 0
        rwdx = 1.5
        
        FIy = 30
        clicky = 1
        rwdy = 3
        
        FIx1 = 60
        clickx1 = 0
        rwdx1 = 1.5
        
        FIy1 = 60
        clicky1 = 1
        rwdy1 = 3

        lab = ' FI 30 rwd rate 0.05'
        lab1 = 'FI 60 rwd rate 0.05'

        yylabel = 'click true'
        xxlabel ='click false'
    
    if condition_to_split == 'rwd':
        FIx = 30
        clickx = 0
        rwdx = 1.5
        
        FIy = 30
        clicky = 0
        rwdy = 1.5
        
        FIx1 = 30
        clickx1 = 1
        rwdx1 = 1.5
        
        FIy1 = 30
        clicky1 = 1
        rwdy1 = 1.5
        
        FIx2 = 60
        clickx2 = 0
        rwdx2 = 1.5
        
        FIy2 = 60
        clicky2 = 0
        rwdy2 = 3

        lab = 'FI 30 rwd rate 0.05'
        lab1 = 'FI 30 rwd rate 0.1'
        lab2 = 'FI 60 rwd rate 0.1'
        
        yylabel = 'rwd 3'
        xxlabel ='rwd 1.5'

    axs[0].set_ylabel(yylabel)
    [axs[i].set_xlabel(xxlabel) for i in range(4)]

    for i in range(4):

        sns.lineplot(ax = axs[i], x = xx, y = yy, linestyle = '--', color = 'grey')

        sns.lineplot(ax = axs[i], x = quantilesdf.loc[idx[animal_order[i], FIx,clickx,rwdx]][bhv_feature],
        y = quantilesdf.loc[idx[animal_order[i], FIy,clicky,rwdy]][bhv_feature], label = lab,
        marker = 'o')

        sns.lineplot(ax = axs[i], x = quantilesdf.loc[idx[animal_order[i], FIx1,clickx1,rwdx1]][bhv_feature],
        y = quantilesdf.loc[idx[animal_order[i],FIy1,clicky1,rwdy1]][bhv_feature], label = lab1,
        marker = 'o')

        if condition_to_split == 'rwd':
            sns.lineplot(ax = axs[i], x = quantilesdf.loc[idx[animal_order[i], FIx2,clickx2,rwdx2]][bhv_feature],
                y = quantilesdf.loc[idx[animal_order[i], FIy2,clicky2,rwdy2]][bhv_feature], label = lab2,
                marker = 'o')

        axs[i].set_title(animal_order[i])



    if bool_savefig:
        plt.savefig(f'summary/allanimals_quantiles_{bhv_feature}_{condition_to_split}', transparent = False, facecolor = 'white')     
        plt.savefig(f'summary/allanimals_quantiles_{bhv_feature}_{condition_to_split}.eps', transparent = False, facecolor = 'white', format = 'eps')     



def remove_legend(axes):
    '''
    Removes the legend from axes of the figure

    Parameters:
    axes: (axis or list) axes to remove the legend; name given in the current fig
    '''

    if type(axes) != list:
        axes = [axes]

    for ax in axes:
        ax.legend([],[], frameon = False)


def produce_fig_titles(title_str, title_type = 'sup'):
    '''
    Prints the fig title and converts to a string to use as name to save.
    spaces get replaced by underscores and parenthesis are dropped

    Parameters:
    title_str: title of the figure, how if will be printed at the top
    title_type: str, either 'sup' for suptitle (subplots), or 'regular' for title (regular fig)
    '''

    if title_type == 'sup':
        plt.suptitle(title_str)

    if title_type == 'regular':
        plt.title(title_str)


    return title_str.lower().replace(' ', '_').replace('(', '').replace(')','')


def plot_XY(coordsdf, bodypart, color, ax = None):
    """
    to be used with the dlc dataframes
    """
    if ax == None: 
        plt.plot(coordsdf[bodypart].x, coordsdf[bodypart].y, '.', color = color, ms = 2)
    else: 
        ax.plot(coordsdf[bodypart].x, coordsdf[bodypart].y, '.', color = color, ms = 2)
