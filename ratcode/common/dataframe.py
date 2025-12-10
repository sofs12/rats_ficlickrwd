import numpy as np
import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt


def combine_df_elements(df, cols):
    '''
    Retuns a dataframe

    Parameters:
    df: dataframe to get the information
    cols: columns to group the dataframe by 
    '''
    
    dickey = list(df.groupby(cols).groups.keys())

    combined_df = pd.DataFrame(columns = cols)

    for ii in range(len(dickey)):
        key = dickey[ii]

        for kk in range(len(cols)):
            combined_df.loc[ii, cols[kk]] = key[kk]

    return combined_df

def count_dictionary_values(dic):
    '''
    Return a dictionary with
    keys: same as the original
    values: length of the list of the values of the original

    Parameters:
    dic: dictionary whose keys should be lists
    '''

    len_dic = {}

    for key in dic.keys():
        len_dic[key] = len(dic[key])

    return len_dic

def piecewise_linear(x, x0, k):
    '''
    Return an array

    Parameters:
    x: ndarray or scalar; initially thought out to be used with a ndarray
    x0: change point of the two branches of the function (domain)
    k: slope of the second linear funtion
    '''
    return np.piecewise(x, [x < x0], [lambda x:0, lambda x: k*(x-x0)])

def slidingWindow(df, ntrials= 10, window_step = 1, bool_plot = False):
    '''
    Return an array

    NEED TO REVISIT THIS

    Parameters:
    df:
    ntrials:
    window_step:
    bool_plot: boolean; [Y] produces and saves a plot
    '''
    
    all_sessions = df.session.unique()

    slidingdf = pd.DataFrame()
    auxdf = pd.DataFrame(columns=['date', 'session', 'first_trial', 'last_trial', 'switch', 'second_slope', 'av_decay_raster'])

    #add an empty row at the end of every session
    empt = np.empty(5)
    empt[:] = np.nan    
    
    #first trial of every session
    first_trials = []

    all_ravs = []

    for sess in all_sessions:

        dff = df.query(f'session == {sess}')

        all_trials = dff.trialno.values

        first_trials.append(all_trials[0])

        FI = dff.FI.values[0]
        FI01s = FI * 10


        for tt0 in np.arange(0, len(all_trials)-ntrials+1, window_step):
            current_trials = all_trials[tt0: tt0+ntrials]

            x = dff.query(f'trialno in {current_trials.tolist()}')

            auxdf['date'] = x.date.unique()
            auxdf['session'] = x.session.unique()
            auxdf['first_trial'] = current_trials[0]
            auxdf['last_trial'] = current_trials[-1]

            #rasters averaged
            rav = x.decay_raster_reshaped.sum()/ntrials
            rav = rav[:FI01s]

            #all_ravs.append(rav)
     
            auxdf['av_decay_raster'] = pd.Series([rav])

            t = np.arange(0,FI, 0.1)

            #test first with a linear fit; if the slope is 0, consider this as a no switch trial and set the slope to zero (or to wtv value very close to zero was the result of the linear fit)
            res = scipy.stats.linregress(t, rav)#[:FI01s])
            
            if res.slope < 1e-3 or res.intercept > 0:                 
                auxdf['switch'] = np.nan
                auxdf['second_slope'] = np.nan #res.slope

                if bool_plot:
                    plt.figure(figsize=(16,8))
                    plt.plot(t, rav)#[:FI01s])
                    plt.ylim(0,2)
                    plt.title(f'[no switch] session {sess} - trials [{current_trials[0]},{current_trials[-1]}]')
                    plt.xlabel('t (s)')
                    plt.ylabel('pressing rate (Hz)')
                    #if save_plot:
                    #    plt.savefig(f'stationary_trials/{animal}/s{sess}_1stt{current_trials[0]}', transparent = False) # get titles automatically!
                    plt.show()

            
            else:
                #piecewise fit
                p , e = scipy.optimize.curve_fit(piecewise_linear, t, rav[:FI01s], maxfev = 1000)

                auxdf['switch'] = p[0]
                auxdf['second_slope'] = p[-1]

                if bool_plot:
                    plt.figure(figsize=(16,8))
                    plt.plot(t, rav)#[:FI01s])
                    plt.plot(t, piecewise_linear(t, *p))
                    plt.ylim(0,2)
                    plt.title(f'session {sess} - trials [{current_trials[0]},{current_trials[-1]}]')
                    plt.xlabel('t (s)')
                    plt.ylabel('pressing rate (Hz)')
                    #if save_plot:
                    #    plt.savefig(f'stationary_trials/{animal}/s{sess}_t{current_trials[0]}', transparent = False) # get titles automatically!
 
                    plt.show()

            slidingdf = pd.concat([slidingdf, auxdf], ignore_index=True)


        slidingdf.loc[slidingdf.index[-1]+1] = [x.date.unique()[0], x.session.unique()[0], *empt]

    return np.array(first_trials), slidingdf


def quantile_list(df, animal, FI, click, rwd, bhv_feature, quantiles = np.arange(.1,1,.1)):
    '''
    Return an array

    SHOULD SPLIT THIS INTO MULTIPLE FUNCTIONS; ONE TO LOOK AT MULTIPLE DFS,
    AND ANOTHER ONE TO JUST COMPUTE THE QUANTILES

    Parameters:
    df:
    animal:
    FI: 
    click: 
    rwd: 
    bhv_feature: 
    quantiles:
    '''

    if 'newbool_cp' not in df.keys():
        df['newbool_cp'] = True

    dist = np.hstack(df.query(f'animal == "{animal}" and FI == {FI} and click == {click} and actual_rwd == {rwd}')[bhv_feature].dropna().values)
    
    return np.quantile(dist, quantiles)


'''
def get_pressing_currBio(df, binw = 500):
    
    #Return a tuple (dataframe df, dataframe curr bio)
    #
    #Dataframe df is the same as the parameter, with extra columns
    #Dataframe curr bio carries information about the press rate aligned on transition
    #
    #Parameters:
    #newdf: dataframe with the 
    #binw: bin size, in ms (default 500)

    df['lever_aligned_cp'] = df.apply(lambda x: x.lever_rel - x.cp*1000 if x.bool_cp == True else np.nan, axis = 1)
    df['lever_aligned_cp_dropfirstpress'] = df.lever_aligned_cp.apply(lambda x: np.delete(x, np.where(x==0)) if type(x) == np.ndarray else np.nan)
    df['last_press_beforeFI_index'] = df.apply(lambda x: np.where(x.lever_rel < x.FI *1000)[0], axis = 1)
    df['bool_last_press'] = df.last_press_beforeFI_index.apply(lambda x: True if len(x)>0 else False)
    df['last_press_beforeFI'] = df.apply(lambda x: x.lever_rel[x.last_press_beforeFI_index[-1]] if x.bool_last_press == True else [], axis = 1)
    df['last_press_beforeFI_aligned_cp'] = df.apply(lambda x: x.last_press_beforeFI - x.cp*1000 if x.bool_last_press == True else [], axis = 1)
    df['FI_aligned_cp'] = df.apply(lambda x: (x.FI - x.cp)*1000, axis = 1)
    df['FI_max_cap'] = df.FI.apply(lambda x: x/3*1000)
    df['FI_aligned_cp_capped'] = df.apply(lambda x: x.FI_aligned_cp if x.FI_aligned_cp < x.FI_max_cap else (x.FI_max_cap if x.bool_cp == True else np.nan), axis = 1)

    df['FI_nprots'] = df.apply(lambda x: f'FI{x.FI} rwd{x.nprots_approx}', axis = 1)

    xlower = -2000

    df['bins'] = df.apply(lambda x: np.arange(xlower, int(x.FI_aligned_cp_capped), binw) if x.bool_cp == True and x.bool_last_press == True else [], axis = 1)
    df['counts_cp'] = df.apply(lambda x: np.histogram(x.lever_aligned_cp_dropfirstpress, x.bins)[0]*1000/binw if x.bool_cp == True and x.bool_last_press == True else [], axis = 1)
    df['len_counts'] = df.counts_cp.apply(lambda x: len(x))
    df['middle'] = df.apply(lambda x: x.bins[:x.len_counts] + binw/2 if x.bool_cp == True and x.bool_last_press == True else [], axis = 1)
    df['FI_long'] = df.apply(lambda x: [x.FI]*x.len_counts, axis = 1)
    df['nprots_long'] = df.apply(lambda x: [x.nprots_approx]*x.len_counts, axis = 1)
    df['FI_nprots_long'] = df.apply(lambda x: [x.FI_nprots]*x.len_counts, axis = 1)
    df['animal_long'] = df.apply(lambda x: [x.animal]*x.len_counts, axis = 1)

    press_currbio = pd.DataFrame()
    press_currbio['middle'] = np.hstack(df.query(f'bool_cp == True').middle.values)
    press_currbio['middle_s'] = press_currbio.middle.apply(lambda x: x/1000)
    press_currbio['counts_cp'] = np.hstack(df.query(f'bool_cp == True').counts_cp.values)
    press_currbio['FI_nprots'] = np.hstack(df.query(f'bool_cp == True').FI_nprots_long.values)
    press_currbio['FI'] = np.hstack(df.query(f'bool_cp == True').FI_long.values)
    press_currbio['nprots'] = np.hstack(df.query(f'bool_cp == True').nprots_long.values)
    press_currbio['animal'] = np.hstack(df.query('bool_cp == True').animal_long.values)

    return df, press_currbio
'''

def fraction_within_boundaries(dist_array, max_bound, min_bound = 0):
    '''
    Return a float, corresponding to the fraction of the distribution between two boundaries

    Parameters:
    dist_array: array with the distribution
    max_bound: top/ maximum boundary
    min_bound: lower/ minimum boundary (default = 0)
    '''

    uu = dist_array[dist_array < max_bound]
    ll = uu[uu > min_bound]
    
    return len(ll)/len(dist_array)

# this is the one used in the notebook
def get_pressing_currBio(newdf, care_about_reward = False, binw = 500):
    newdf['lever_aligned_cp'] = newdf.apply(lambda x: x.lever_rel - x.cp*1000 if x.bool_cp == True else np.nan, axis = 1)
    newdf['lever_aligned_cp_dropfirstpress'] = newdf.lever_aligned_cp.apply(lambda x: np.delete(x, np.where(x>=0)[0][0]) if type(x) == np.ndarray else np.nan)
    newdf['last_press_beforeFI_index'] = newdf.apply(lambda x: np.where(x.lever_rel < x.FI *1000)[0], axis = 1)
    newdf['bool_last_press'] = newdf.last_press_beforeFI_index.apply(lambda x: True if len(x)>0 else False)
    newdf['last_press_beforeFI'] = newdf.apply(lambda x: x.lever_rel[x.last_press_beforeFI_index[-1]] if x.bool_last_press == True else [], axis = 1)
    newdf['last_press_beforeFI_aligned_cp'] = newdf.apply(lambda x: x.last_press_beforeFI - x.cp*1000 if x.bool_last_press == True else [], axis = 1)
    newdf['FI_aligned_cp'] = newdf.apply(lambda x: (x.FI - x.cp)*1000, axis = 1)
    newdf['FI_max_cap'] = newdf.FI.apply(lambda x: x/3*1000)
    newdf['FI_aligned_cp_capped'] = newdf.apply(lambda x: x.FI_aligned_cp if x.FI_aligned_cp < x.FI_max_cap else (x.FI_max_cap if x.bool_cp == True else np.nan), axis = 1)

    #if care_about_reward:
    #    newdf['FI_nprots'] = newdf.apply(lambda x: f'FI{x.FI} rwd{x.nprots_approx}', axis = 1)

    xlower = -2000

    newdf['bins'] = newdf.apply(lambda x: np.arange(xlower, int(x.FI_aligned_cp_capped), binw) if x.bool_cp == True and x.bool_last_press == True else [], axis = 1)
    newdf['counts_cp'] = newdf.apply(lambda x: np.histogram(x.lever_aligned_cp_dropfirstpress, x.bins)[0]*1000/binw if x.bool_cp == True and x.bool_last_press == True else [], axis = 1)
    newdf['len_counts'] = newdf.counts_cp.apply(lambda x: len(x))
    newdf['middle'] = newdf.apply(lambda x: x.bins[:x.len_counts] + binw/2 if x.bool_cp == True and x.bool_last_press == True else [], axis = 1)
    newdf['FI_long'] = newdf.apply(lambda x: [x.FI]*x.len_counts, axis = 1)

        
    if care_about_reward:## before this was nprotocols
        newdf['nprots_long'] = newdf.apply(lambda x: [x.n_protocols]*x.len_counts, axis = 1)
        newdf['FI_nprots_long'] = newdf.apply(lambda x: [x.FI_nprots]*x.len_counts, axis = 1)
    
    newdf['animal_long'] = newdf.apply(lambda x: [x.animal]*x.len_counts, axis = 1)        

    press_currbio = pd.DataFrame()
    
    if len(newdf.query(f'bool_cp == True').middle.values) > 0: # if there are no cps the press_currbio df is returned empty
        press_currbio['middle'] = np.hstack(newdf.query(f'bool_cp == True').middle.values)
        press_currbio['middle_s'] = press_currbio.middle.apply(lambda x: x/1000)
        press_currbio['counts_cp'] = np.hstack(newdf.query(f'bool_cp == True').counts_cp.values)
        press_currbio['FI'] = np.hstack(newdf.query(f'bool_cp == True').FI_long.values)
    
        if care_about_reward:
            press_currbio['FI_nprots'] = np.hstack(newdf.query(f'bool_cp == True').FI_nprots_long.values)
            press_currbio['nprots'] = np.hstack(newdf.query(f'bool_cp == True').nprots_long.values)
        
        press_currbio['animal'] = np.hstack(newdf.query('bool_cp == True').animal_long.values)

    return newdf, press_currbio

def get_average_pressing(press_currbio):

    '''
    Return a dataframe with average pressing rate

    Parameters:
    press_currbio: dataframe resulting from the get_pressing_currBio function
    '''

    if len(press_currbio) > 0:

        middle_dic = press_currbio.groupby('middle_s').groups

        average_middle = []

        for key in middle_dic.keys():
            aa = []
            aa.append(press_currbio.loc[middle_dic[key]].counts_cp.values)
            average_middle.append(np.average(aa[0]))

        average_press_df = pd.DataFrame()
        average_press_df['middle_s'] = middle_dic.keys()
        average_press_df['av_pressing_rate'] = average_middle
    
    else:
        average_press_df = np.nan

    return average_press_df


def get_outsdf(outsdf):
    '''
    Return a dataframe

    Parameters:
    outsdf: dataframe with information about the blocks (added to the original df??)
    improve this description
    '''

    #outsdf = uniblocks.query(f'bool_cp == True and animal == "{animal_order_blocks[i]}"').reset_index(drop = True)

    # the block changing is not properly defined in the code before :(
    outsdf['bool_block_changed'] = outsdf.FI.shift(1) # FI because this is an experiment a
    outsdf['bool_block_changed'] = outsdf.apply(lambda x: x.FI != x.bool_block_changed, axis = 1)

    outsdf['bool_session_changed'] = outsdf.date.shift(1)
    outsdf['bool_session_changed'] = outsdf.apply(lambda x: x.date != x.bool_session_changed, axis = 1)

    session_start = outsdf.query('bool_session_changed == True').index.values # this is the first trial of the new session
    block_start = np.unique(np.sort(np.hstack([session_start, outsdf.query('bool_block_changed == True').index.values])))
    block_end = (block_start-1)[1:]
    first_blocks_start = np.sort(np.array(list(set(block_start).intersection(set(session_start)))))
    first_blocks_end = np.array([block_end[np.where((block_end>session_start[ii]) == True)[0][0]] for ii in range(len(session_start))])
    all_FIs = outsdf.loc[block_start].FI.values

    outsdf['curr_ind'] = outsdf.index
    outsdf['block_start'] = outsdf.apply(lambda x: block_start[np.where(block_start <= x.curr_ind)[0][-1]], axis = 1)
    outsdf['trialno_within_block'] = outsdf.apply(lambda x: x.curr_ind - x.block_start, axis = 1)


    first_blocks_bool = np.full(len(outsdf), False)
    previous_FI = np.full(len(outsdf), 0)

    for ind in outsdf.index.values:
        ii = [np.where(ind >= first_blocks_start)][0][0][-1]

        if ind <= first_blocks_end[ii]: #first block in the session
            first_blocks_bool[ind] = True

        else: #within session, it makes sense to consider previous FI
            ii_prev = [np.where(ind >= block_start)][0][0][-1]
            previous_FI[ind] = all_FIs[ii_prev-1]

    outsdf['bool_first_block'] = first_blocks_bool
    outsdf['previous_FI'] = previous_FI

    return outsdf


def get_lvrzoom(uniblocks, i, i_exp):
    '''
    Return a dataframe.
    Use only for blocks sessions

    Parameters:
    uniblocks: dataframe that aggregates the blocks experiments
    i: index for animal in animal_order_blocks
    i_exp: index for the experiment in [a,b,c]
    '''

    exp_list = ['a', 'b', 'c']

    outsdf = uniblocks.query(f'bool_cp == True and experiment == "{exp_list[i_exp]}" and animal == "{animal_order_blocks[i]}"')

    if i_exp == 2:
        outsdf = outsdf.sort_values(by = ['nprots'])
    else:
        outsdf = outsdf.sort_values(by = ['FI'])
    outsdf = outsdf.reset_index(drop = True)
    outsdf['trialno'] = outsdf.index.values
    outsdf['lever_aligned_cp'] = outsdf.apply(lambda x: x.lever_rel/1000 - x.cp, axis = 1)
    
    # drop the press that is the transition (to avoid a very visible line splitting the figure)
    outsdf.lever_aligned_cp = outsdf.lever_aligned_cp.apply(lambda x: np.delete(x, np.where(x==0)[0]))
    
    lvrzoom = pd.DataFrame()
    for key in ['trialno', 'FI', 'nprots']:
        lvrzoom[key] = np.hstack(outsdf.apply(lambda x: np.full(x.count_lever - 1, x[key]), axis = 1)) # (count_lever - 1) because of the dropping of the first press 
    lvrzoom['lever_aligned_cp'] = np.hstack(outsdf.lever_aligned_cp.values)

    return lvrzoom


def midpoint_between_array_elements(original_array):
    """
    Compute the mean between consecutive elements of a 1-d array. The resulting array has len(original) - 1.
    Return an array.
    """

    mean_array = np.ndarray(len(original_array)-1)

    for ii in range(len(original_array)-1):
        mean_array[ii] = np.mean([original_array[ii+1],original_array[ii]])

    return mean_array




""" dataframes.py
This module completes information regarding dataframes
"""

#from ratcode.globe.globe import *

def populateDataFrame(bhv_log_file, eventcode_ref):

    print('populating df...')

    bhv_data = pd.read_csv(bhv_log_file, sep = '\t', header = None, names = ['code', 'timestamp'])

    df = pd.DataFrame(columns=['animal', 'date', 'experimenter', 'timestamp', 'eventcode', 'eventname', 'state'])

    df.timestamp = bhv_data.timestamp
    df.eventcode = bhv_data.code

    x = bhv_log_file.split('\\') #goes to the last directory
    x = x[-1].split('.') #gets rid of the file extension
    x = x[0].split('_')

    df['animal'] = x[0]
    df['date'] = x[2]
    df['experimenter'] = x[3]

    #eventcodes to eventnames
    df['eventname'] = df['eventcode'].map(lambda code: str(code))
    df = df.replace({'eventname' : eventcode_ref})

    df['FI'] = df.query('eventname == "FI_SESSION"').timestamp.values[0]
    df['click'] = df.query('eventname == "CLICK_SESSION"').timestamp.values[0]
    df['n_protocols'] = df.query('eventname == "N_PROTOCOL"').timestamp.values[0]

    # new trial happens when TRIAL_PREP appears 
    df['trialno'] = np.where(df['eventname']== 'TRIAL_PREP', 1, 0)
    df['trialno'] = df['trialno'].cumsum(axis=0)


    # current state - this takes a while - better way of doing this?
    # event codes with 1st number 2 are reserved for states
    current_state = 'none'
    for i in range(len(df)):
        if str(df['eventcode'][i])[0] == '2':
            current_state = df['eventname'][i]

        df['state'][i] = current_state


    # lever, poke and lick detection
    df['lever'] = np.where(df['eventname'] == 'LEVER_PRESSED',1,0)
    df['poke'] = np.where(df['eventname'] == 'POKE_IN', 1,0)
    df['lick'] = np.where(df['eventname'] == 'LICK_IN', 1,0)

    return df



def populateDataFrame_BLOCKS(bhv_log_file, eventcode_ref):

    print('populating BLOCKS df...')

    bhv_data = pd.read_csv(bhv_log_file, sep = '\t', header = None, names = ['code', 'timestamp'])

    df = pd.DataFrame(columns=['animal', 'date', 'experimenter', 'timestamp', 'eventcode', 'eventname', 'state'])

    df.timestamp = bhv_data.timestamp
    df.eventcode = bhv_data.code

    x = bhv_log_file.split('\\') #goes to the last directory
    x = x[-1].split('.') #gets rid of the file extension
    x = x[0].split('_')

    df['animal'] = x[0]
    df['date'] = x[2]
    df['experimenter'] = x[3]

    #eventcodes to eventnames
    df['eventname'] = df['eventcode'].map(lambda code: str(code))
    df = df.replace({'eventname' : eventcode_ref})

    df['blockno'] = df.eventname.apply(lambda x: 1 if x in ['SESSION_PREP','BLOCK_PREP'] else 0)
    df['blockno'] = df['blockno'].cumsum(axis=0)

    FI_dict = dict(zip(df.blockno.unique(), np.hstack([0,df.query('eventname == "FI_SESSION"').timestamp.values])))
    click_dict = dict(zip(df.blockno.unique(), np.hstack([0,df.query('eventname == "CLICK_SESSION"').timestamp.values])))
    n_protocols_dict = dict(zip(df.blockno.unique(), np.hstack([0,df.query('eventname == "N_PROTOCOL"').timestamp.values])))
    df['FI'] = df.blockno.map(FI_dict)
    df['click'] = df.blockno.map(click_dict)
    df['n_protocols'] = df.blockno.map(n_protocols_dict)

    # the way this is done, there is always a 0 before the variables are defined, hence the > 2 instead of > 1
    bool_change_FI = len(df.FI.dropna().unique()) > 2
    bool_change_click = len(df.click.dropna().unique()) > 2
    bool_change_nprot = len(df.n_protocols.dropna().unique()) > 2

    df['blocks_FI'] = bool(bool_change_FI)
    df['blocks_click'] = bool(bool_change_click)
    df['blocks_nprot'] = bool(bool_change_nprot)

    if not bool_change_FI:
        df['FI'] = df.FI.dropna().unique()[-1]

    if not bool_change_click:
        df['click'] = df.click.dropna().unique()[-1]
    
    if not bool_change_nprot:
        df['n_protocols'] = df.n_protocols.dropna().unique()[-1]


    df['trialno'] = np.where(df['eventname']== 'TRIAL_PREP', 1, 0)
    df['trialno'] = df['trialno'].cumsum(axis=0)


    # current state - this takes a while - better way of doing this?
    # event codes with 1st number 2 are reserved for states
    current_state = 'none'
    for i in range(len(df)):
        if str(df['eventcode'][i])[0] == '2':
            current_state = df['eventname'][i]

        df['state'][i] = current_state


    # lever, poke and lick detection
    df['lever'] = np.where(df['eventname'] == 'LEVER_PRESSED',1,0)
    df['poke'] = np.where(df['eventname'] == 'POKE_IN', 1,0)
    df['lick'] = np.where(df['eventname'] == 'LICK_IN', 1,0)

    return df



def df_to_new_df(df):

    print('converting to new...')

    newdf = pd.DataFrame(columns = ['trialno', 'trial_start','trial_end', 'trial_duration','lever_abs', 'lever_rel', 'leverup_abs', 'leverup_rel', 'poke_abs', 'poke_rel', 'lick_abs', 'lick_rel', 'pump_abs', 'pump_abs_stop', 'pump_rel', 'pump_duration', 'first_press_s'])

    trials = df.query('eventname == "TRIAL_PREP"').trialno.values
    newdf.trialno = trials

    newdf['trial_start'] = df.query('eventname == "TRIAL_PREP"').timestamp.values
    newdf['trial_end'] = newdf['trial_start'].shift(-1)
    newdf['trial_duration'] = newdf['trial_end'] - newdf['trial_start']
    newdf['trial_duration_s'] = newdf['trial_duration']/1000
    newdf['trial_duration_s'] = newdf['trial_duration_s'].astype('float')

    newdf['lever_abs'] = newdf.apply(lambda x: df.query(f'eventname == "LEVER_PRESSED" and trialno == {x.trialno}').timestamp.values, axis = 1)
    newdf['lever_rel'] = newdf.apply(lambda x: x.lever_abs - x.trial_start, axis = 1)
    
    ## new -- implemented 20240206
    newdf['leverup_abs'] = newdf.apply(lambda x: df.query(f'eventname == "LEVER_RELEASED" and trialno == {x.trialno}').timestamp.values, axis = 1)
    newdf['leverup_rel'] = newdf.apply(lambda x: x.leverup_abs - x.trial_start, axis = 1)

    newdf['poke_abs'] = newdf.apply(lambda x: df.query(f'eventname == "POKE_IN" and trialno == {x.trialno}').timestamp.values, axis = 1)
    newdf['poke_rel'] = newdf.apply(lambda x: x.poke_abs - x.trial_start, axis = 1)

    newdf['lick_abs'] = newdf.apply(lambda x: df.query(f'eventname == "LICK_IN" and trialno == {x.trialno}').timestamp.values, axis = 1)
    newdf['lick_rel'] = newdf.apply(lambda x: x.lick_abs - x.trial_start, axis = 1)

    newdf['first_press_s'] = newdf['lever_rel'].apply(lambda x: x[0]/1000 if len(x) > 0 else x)
    
    newdf = newdf.drop(newdf.index[-1])

    #if valve:
    #    newdf['time_valve'] = newdf.apply(lambda x: df.query(f'eventname == "TIME_VALVE" and trialno == {x.trialno}').timestamp.values, axis = 1)
    newdf['pump_abs'] = newdf.apply(lambda x: df.query(f'eventname == "PUMP_ON" and trialno == {x.trialno}').timestamp.values, axis = 1)
    newdf['pump_abs_stop'] = newdf.apply(lambda x: df.query(f'eventname == "PUMP_OFF" and trialno == {x.trialno}').timestamp.values, axis = 1)
    
    newdf['pump_rel'] = newdf.apply(lambda x: x.pump_abs - x.trial_start, axis = 1)
    newdf['pump_duration'] = newdf.apply(lambda x: x.pump_abs_stop - x.pump_abs, axis = 1)

    newdf['FI'] = df.FI.unique()[0]
    newdf['click'] = df.click.unique()[0]
    newdf['n_protocols'] = df.n_protocols.unique()[0]

    return newdf



def df_to_new_df_BLOCKS(df):

    print('converting to new...')

    newdf = pd.DataFrame(columns = ['trialno', 'trial_start','trial_end', 'trial_duration','lever_abs', 'lever_rel', 'poke_abs', 'poke_rel', 'lick_abs', 'lick_rel', 'pump_abs', 'pump_abs_stop', 'pump_rel', 'pump_duration', 'first_press_s'])

    trials = df.query('eventname == "TRIAL_PREP"').trialno.values
    newdf.trialno = trials
    
    newdf['trial_start'] = df.query('eventname == "TRIAL_PREP"').timestamp.values 
    newdf['trial_end'] = newdf['trial_start'].shift(-1)
    
    newdf['trial_duration'] = newdf['trial_end'] - newdf['trial_start']
    newdf['trial_duration_s'] = newdf['trial_duration']/1000
    newdf['trial_duration_s'] = newdf['trial_duration_s'].astype('float')

    newdf['lever_abs'] = newdf.apply(lambda x: df.query(f'eventname == "LEVER_PRESSED" and trialno == {x.trialno}').timestamp.values, axis = 1)
    newdf['lever_rel'] = newdf.apply(lambda x: x.lever_abs - x.trial_start, axis = 1)
    
    ## new -- implemented 20240206
    newdf['leverup_abs'] = newdf.apply(lambda x: df.query(f'eventname == "LEVER_RELEASED" and trialno == {x.trialno}').timestamp.values, axis = 1)
    newdf['leverup_rel'] = newdf.apply(lambda x: x.leverup_abs - x.trial_start, axis = 1)

    newdf['poke_abs'] = newdf.apply(lambda x: df.query(f'eventname == "POKE_IN" and trialno == {x.trialno}').timestamp.values, axis = 1)
    newdf['poke_rel'] = newdf.apply(lambda x: x.poke_abs - x.trial_start, axis = 1)

    newdf['lick_abs'] = newdf.apply(lambda x: df.query(f'eventname == "LICK_IN" and trialno == {x.trialno}').timestamp.values, axis = 1)
    newdf['lick_rel'] = newdf.apply(lambda x: x.lick_abs - x.trial_start, axis = 1)

    newdf['first_press_s'] = newdf['lever_rel'].apply(lambda x: x[0]/1000 if len(x) > 0 else x)
    
    newdf = newdf.drop(newdf.index[-1])

    newdf['pump_abs'] = newdf.apply(lambda x: df.query(f'eventname == "PUMP_ON" and trialno == {x.trialno}').timestamp.values, axis = 1)
    newdf['pump_abs_stop'] = newdf.apply(lambda x: df.query(f'eventname == "PUMP_OFF" and trialno == {x.trialno}').timestamp.values, axis = 1)
    
    newdf['pump_rel'] = newdf.apply(lambda x: x.pump_abs - x.trial_start, axis = 1)
    newdf['pump_duration'] = newdf.apply(lambda x: x.pump_abs_stop - x.pump_abs, axis = 1)

    newdf['blockno'] = newdf.trialno.apply(lambda x: df.query(f'trialno == {x}').blockno.unique()[0])
    newdf['FI'] = newdf.trialno.apply(lambda x: df.query(f'trialno == {x}').FI.unique()[0])
    newdf['click'] = newdf.trialno.apply(lambda x: df.query(f'trialno == {x}').click.unique()[0])
    newdf['n_protocols'] = newdf.trialno.apply(lambda x: df.query(f'trialno == {x}').n_protocols.unique()[0])

    ## new -- implemented 20240814
    if newdf.click.unique()[0] == 1: 
        newdf['click_abs'] = newdf.apply(lambda x: df.query(f'eventname == "CLICK_ON" and trialno == {x.trialno}').timestamp.values, axis = 1)
        newdf['click_rel'] = newdf.apply(lambda x: x.click_abs - x.trial_start, axis = 1)

    return newdf





def complete_aggregate(aggregatedf):
    
    datekeys = list(aggregatedf.groupby('date').groups.keys())
    aggregatedf['session'] = aggregatedf['date'].apply(lambda x: datekeys.index(x) + 1) # so that the first session is session 1 instead of session 0
    aggregatedf['trialno_withinsession'] = aggregatedf.trialno
    aggregatedf['trialno'] = aggregatedf.index +1

    allsessions_trial_start = []
    allsessions_trial_number = []

    #get list of the new first trial number of each session, and the start time of each session
    for date in datekeys:
        allsessions_trial_start.append(aggregatedf.groupby('date').get_group(date).trial_start.values[0])
        allsessions_trial_number.append(aggregatedf.groupby('date').get_group(date).trialno.values[0])
    
    aggregatedf['bool_lever'] = ~aggregatedf.lever_rel.isna()
    aggregatedf['bool_poke'] = ~aggregatedf.poke_rel.isna()
    aggregatedf['bool_lick'] = ~aggregatedf.lick_rel.isna()
    aggregatedf['bool_valve'] = ~aggregatedf.valve_rel.isna()
    aggregatedf['count_lever'] = aggregatedf.apply(lambda x: len(x.lever_rel) if x.bool_lever == True else 0, axis = 1)
    aggregatedf['count_poke'] = aggregatedf.apply(lambda x: len(x.poke_rel) if x.bool_poke == True else 0, axis = 1)
    aggregatedf['count_lick'] = aggregatedf.apply(lambda x: len(x.lick_rel) if x.bool_lick == True else 0, axis = 1)

    cp_series =  aggregatedf.apply(lambda x: change_point.accepted_cp_Gallistel(x.trialno, 2, aggregatedf, 'lever_rel', True) if len(x.lever_rel) > 0 else x.lever_rel, axis = 1)
    aggregatedf['cp_pre'] = cp_series.apply(lambda x: x[0] if len(x) > 0 else np.nan)
    aggregatedf['rate_pre'] = cp_series.apply(lambda x: x[3] if len(x) > 0 else np.nan)
    aggregatedf['rateH_pre'] = aggregatedf.rate_pre.apply(lambda x: x[-1] if type(x) == np.ndarray else x)
    aggregatedf['cp_beforeFI'] = aggregatedf.apply(lambda x: x.cp_pre < x.FI, axis = 1)
    aggregatedf['bool_cp'] = aggregatedf.apply(lambda x: x.cp_beforeFI and x.rateH_pre < 10, axis = 1)
    aggregatedf['cp'] = aggregatedf.apply(lambda x: x.cp_pre if x.bool_cp else np.nan, axis = 1)
    aggregatedf['rateH'] = aggregatedf.apply(lambda x: x.rateH_pre if x.bool_cp else np.nan, axis = 1)

    aggregatedf['session_vars'] = aggregatedf.apply(lambda x: f'FI {x.FI} click {bool(x.click)}', axis = 1)
    aggregatedf['actual_rwd'] = aggregatedf.rwdmod.apply(lambda x: 1.5 if x == 1 else x)
    aggregatedf['rwd_uL'] = aggregatedf.actual_rwd.apply(lambda x: rwd_dic[x] if x in rwd_dic.keys() else np.nan)
    aggregatedf['session_vars_w_rwdmod'] = aggregatedf.apply(lambda x: f'FI {x.FI}\nclick {bool(x.click)}\nrwd {x.actual_rwd}', axis = 1)
    aggregatedf['rwdrate'] = aggregatedf.apply(lambda x: x.actual_rwd / x.FI, axis = 1)
    aggregatedf['rwdrate_uL'] = aggregatedf.apply(lambda x: x.rwd_uL / (x.FI/60), axis = 1)


    aggregatedf['cp_normalised'] = aggregatedf.apply(lambda x: x.cp / x.FI, axis = 1)
    aggregatedf['animaldate'] = aggregatedf.apply(lambda x: f'{x.animal} {x.date}', axis = 1)
    aggregatedf['first_press_beforeFI'] = aggregatedf.apply(lambda x: True if x.first_press_s < x.FI else False, axis = 1)


    # pump rwd rate
    aggregatedf.pump_duration = aggregatedf.pump_duration.apply(lambda x: x[0] if type(x)!= float else np.nan)
    aggregatedf['pump_volume'] = aggregatedf.pump_duration.apply(lambda x: 3.45/33*x)
    aggregatedf['pump_rwdrate'] = aggregatedf.apply(lambda x: x.pump_volume/(x.FI/60), axis = 1)
    aggregatedf['pump_rwdrate_clean'] = aggregatedf.pump_rwdrate.apply(lambda x: "I" if x < 80 else "II")

    aggregatedf['date_mmdd'] = aggregatedf.date.apply(lambda x: x[2:])
    aggregatedf['datetime'] = aggregatedf.date.apply(lambda x: datetime.datetime.strptime(x, '%y%m%d').date())
    aggregatedf['date_FI'] = aggregatedf.apply(lambda x: f"{x.date_mmdd}\nFI{x.FI}", axis = 1)

    aggregatedf['interpress'] = aggregatedf.lever_rel.apply(lambda x: np.diff(x)/1000)
    aggregatedf['presses_after_cp'] = aggregatedf.apply(lambda x: x.lever_rel[x.lever_rel>x.cp*1000]/1000 if x.bool_cp == True else np.nan, axis = 1)
    aggregatedf['presses_cp2FI'] = aggregatedf.apply(lambda x: x.presses_after_cp[x.presses_after_cp < x.FI] if x.bool_cp == True else np.nan, axis = 1)
    aggregatedf['interpress_aftercp'] = aggregatedf.presses_cp2FI.apply(lambda x: np.diff(x) if type(x)!=float else np.nan)
    
    return aggregatedf


def produce_trialtypesdf_per_experiment(animal_list, experiment, click_blocks):
    '''
    Return a dataframe with the number and fractions of trials of each type

    Parameters:
    animal_list: list of animals to look at data
    experiment: experiment label [a,b,c] 
    click_blocks: dataframe for blocks experiments, animals with click
    '''

    #ideally this doesn't live here (it's already defined in the globe)
    extra_conds_dic = {
    'a': 'experiment == "a" and FI_len == 3 and nprots == 14',
    'b': 'experiment == "b" and FI_len == 3 and nprots_len == 3',
    'c': 'experiment == "c" and FI == 30 and nprots_len == 3'
    }

    trialtypesdf = pd.DataFrame()

    trialtypesdf['animal'] = animal_list 
    trialtypesdf['experiment'] = experiment
    trialtypesdf['no_total'] = trialtypesdf.apply(lambda x: len(click_blocks.query(f'animal == "{x.animal}" and {extra_conds_dic[x.experiment]}')), axis = 1)
    trialtypesdf['no_transition'] = trialtypesdf.apply(lambda x: len(click_blocks.query(f'bool_cp == True and animal == "{x.animal}" and {extra_conds_dic[x.experiment]}')), axis = 1)
    trialtypesdf['no_RT'] = trialtypesdf.apply(lambda x: len(click_blocks.query(f'count_lever == 1 and RT < 10 and animal == "{x.animal}" and {extra_conds_dic[x.experiment]}')), axis = 1)
    trialtypesdf['no_engaged no transition'] = trialtypesdf.apply(lambda x: len(click_blocks.query(f'count_lever > 1 and bool_cp == False and animal == "{x.animal}" and {extra_conds_dic[x.experiment]}')), axis = 1)
    trialtypesdf['no_disengaged'] = trialtypesdf.apply(lambda x: len(click_blocks.query(f'count_lever == 1 and RT >=10 and animal == "{x.animal}" and {extra_conds_dic[x.experiment]}')), axis = 1)

    trialtypesdf.set_index('animal', inplace = True)

    for key in trialtypesdf.keys()[2:]:
        trialtypesdf[f'{key[3:]}'] = trialtypesdf.apply(lambda x: x[key]/x.no_total if x.no_total > 0 else 0, axis = 1)

    return trialtypesdf



def delete_consumption_time(df, FI, alpha):

    '''
    Return a dataframe

    Parameters:
    df: dataframe to extract cp distribution. Must have 'cp' and 'FI' as columns
    FI: FI or list of FIs to aggregate
    alpha: consumption time (non scalable time in the beginning of the trial)
    '''

    delete_consumption_time_df = pd.DataFrame()
    delete_consumption_time_df['percentiles'] = np.arange(0,1.01,.01)

    if type(FI) == int:
        FI = list(FI)
    
    for fi in FI:
        delete_consumption_time_df[fi] = np.quantile(df.query(f'FI == {fi}').cp.dropna().values - alpha, np.arange(0,1.01,.01))/(fi-alpha)

    delete_consumption_time_df['alpha'] = alpha

    delete_consumption_time_df = pd.melt(delete_consumption_time_df, id_vars = ['percentiles','alpha'], value_vars = FI)
    delete_consumption_time_df = delete_consumption_time_df.rename(columns={'variable': 'FI', 'value': 'normalised time'})

    return delete_consumption_time_df


def expand_df(df, main_col, other_cols):
    """
    Convert a dataframe with arrays to a dataframe with single values.
    Return a dataframe.

    Parameters:

    df: dataframe to be expanded
    main_col: column to unstack - this is the column with the arrays; it will determine the lenght of the dataframe
    other_cols: columns to add as extra identifiers or labels of the main_col
    """
    
    if type(other_cols) == str:
        other_cols = [other_cols]

    for col in other_cols:
        df[f'{col}_expanded'] = df.apply(lambda x: np.full(len(x[main_col]), x[col]), axis = 1)

    expandf = pd.DataFrame()
    
    for col in other_cols:
        expandf[col] = np.hstack(df[f'{col}_expanded'])
    
    expandf[main_col] = np.hstack(df[main_col])

    return expandf


def normalise_column(df, col_name, divisor = 'FI'):
    '''
    Adds a dataframe column to the original dataframe.
    The new column is named [col_name]_normalised

    Parameters:
    df: dataframe to do the normalisation operation
    col_name: string or list; name of the column(s) to normalise
    divisor: column nameused to normalise (default is FI) 
    '''
    if type(col_name) == str:
        col_name = [col_name]
        
    for col in col_name:
        df[f'{col}_normalised'] = df.apply(lambda x: x[col] / x[divisor], axis = 1)
    
def sliceByEvent(eventname):
    slicedf = pd.DataFrame()
    slicedf['t'] = df.query(f'eventname == "{eventname}"').timestamp
    slicedf['trialno'] = df.query(f'eventname == "{eventname}"').trialno
    slicedf['kind'] = eventname

    return slicedf

def tidyToArrays(df, aggregate_col):
    uniqueidx = pd.unique(df[aggregate_col])
    
    gg = df.groupby(aggregate_col)
    
    arrdf = pd.DataFrame()
    arrdf[aggregate_col] = uniqueidx
    arrdf[df.kind.values[0]] = arrdf['trialno'].apply(lambda x: gg.get_group(x).t.values)

    return arrdf

def arraysToTidy(df, unpack_col):

    df['count_unpack'] = df[unpack_col].apply(lambda x: len(x))
    col_list = list(df.columns)
    col_list.remove('count_unpack')
    col_list.remove(unpack_col)

    tempdf = pd.DataFrame()

    for col in col_list:
        tempdf[col] = df.apply(lambda x: np.full(x['count_unpack'], x[col]), axis = 1)

    tempdf[unpack_col] = df[unpack_col]
    
    tidydf = pd.DataFrame()
    l = col_list
    l.append(unpack_col)

    for col in l:
        tidydf[col] = np.hstack(tempdf[col].values)

    return tidydf


def group_and_listify(df, group_col, list_cols):
    """
    Groups the DataFrame by a specified column and creates lists of values for the specified columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    group_col (str): The column to group by.
    list_cols (list): The columns for which to create lists of values.

    Returns:
    pd.DataFrame: A DataFrame with grouped values turned into lists.
    """
    grouped_df = df.groupby(group_col).apply(
        lambda x: pd.Series({col: x[col].tolist() for col in list_cols})
    ).reset_index()
    
    return grouped_df


def get_dlc_df(labels_path, thres = 0.7):
    '''
    Returns a dataframe with the labeled coordinates (coords) and a second dataframe nanified when coordinates are below a threshold (default 0.7)
    '''

    coords = pd.read_hdf(labels_path)
    coords.columns = coords.columns.droplevel()
    bodyparts = coords.columns.get_level_values('bodyparts').unique()

    nancoords = coords.copy()

    # Iterate through each body part and apply the threshold
    for bodypart in nancoords.columns.levels[0]:
        likelihood_col = (bodypart, 'likelihood')
        x_col = (bodypart, 'x')
        y_col = (bodypart, 'y')

        # Mask coordinates with NaN where likelihood is below the threshold
        mask = nancoords[likelihood_col] < thres
        nancoords.loc[mask, x_col] = np.nan
        nancoords.loc[mask, y_col] = np.nan

    return coords, nancoords

def read_gpio_into_df(ttl_timestamps_path, camera_view):
    if not os.path.exists(ttl_timestamps_path):
        print(f"Error: File {ttl_timestamps_path} does not exist.")
        return

    if camera_view == 'side':
            gpiodf = pd.read_csv(ttl_timestamps_path, names=['ttl', 'frame', 'bla'], header=0)
    if camera_view == 'top':
            gpiodf = pd.read_csv(ttl_timestamps_path, names=['frame', 'bla', 'ttl'], header=0)
            threshold = gpiodf.ttl.max() * 0.5  
            gpiodf['ttl'] = (gpiodf['ttl'] > threshold).astype(int)

    gpiodf['frame_session'] = gpiodf.frame - gpiodf.frame.values[0]
    gpiodf['bool_ttl'] = (gpiodf['ttl'].shift(1, fill_value=0) == 0) & (gpiodf['ttl'] == 1)
    gpiodf['trialno'] = gpiodf.bool_ttl.cumsum()+1

    return gpiodf 