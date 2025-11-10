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