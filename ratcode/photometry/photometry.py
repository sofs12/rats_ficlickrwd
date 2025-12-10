#%%
#from ratcode.globe.globe import *
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.optimize
import scipy.optimize
from sklearn.linear_model import HuberRegressor 
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from statsmodels.tsa.stattools import acf
import statsmodels.formula.api as smf


#def compute_zscore(arr):
#    return (arr-np.nanmean(arr))/np.nanstd(arr)

def get_prediction(X,y): ## need to allow this to have NaNs
    model = HuberRegressor()
    X = np.array(X).reshape(-1,1)
    y = np.array(y).reshape(-1,1)

    valid_indices = ~np.isnan(X).ravel() & ~np.isnan(y).ravel()

    X_valid = X[valid_indices]
    y_valid = y[valid_indices]
    
    model.fit(X_valid,y_valid)

    predictions = np.full(len(X), np.nan)
    valid_predictions = model.predict(X_valid)
    predictions[valid_indices] = valid_predictions

    return predictions

def quantile_regression(X, y, quantile = 0.5):
    data = pd.DataFrame()
    data['X'] = X
    data['y'] = y
    model = smf.quantreg('y ~ X', data)
    result = model.fit(q=quantile)
    line = model.predict(result.params)

    #dealing with nans, to return a line with the same size as the original data
    transformed_with_nans = np.full_like(X, np.nan)
    not_nan_indices = ~np.isnan(X)*~np.isnan(y)
    transformed_with_nans[not_nan_indices] = line

    return transformed_with_nans

def find_poly(x,y,n = 3):
    coefficients = np.polyfit(x, y, n)
    poly = np.poly1d(coefficients)

    y_fit = poly(x)

    return y_fit

def butter_filter(data, cutoff, fs, bandtype,order=2):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype = bandtype, analog=False)
    y = filtfilt(b, a, data)
    return y

def segment_and_fit_function(xx, yy_nans, function = 'poly',n_poly = 3):
    # Mask indicating NaN regions
    mask = ~np.isnan(yy_nans)

    # Find indices where the signal is not NaN
    segments = np.split(xx, np.where(~mask)[0])
    y_segments = np.split(yy_nans, np.where(~mask)[0])

    # Filter out empty segments
    segments = [seg for seg in segments if len(seg) > 1]
    y_segments = [seg for seg in y_segments if len(seg) > 1]


    polyfit_results = np.full(len(yy_nans), np.nan)

    for seg_x, seg_y in zip(segments, y_segments):
        if np.all(np.isnan(seg_y)):
            continue
        mask = ~np.isnan(seg_y)

        if function == 'poly':
            function_fit = find_poly(seg_x[mask], seg_y[mask], n_poly)

        if function == 'exp':

            xx_exp = np.linspace(0,len(seg_y[mask])/100, len(seg_y[mask]))
            popt, pcov = scipy.optimize.curve_fit(double_exponential, xx_exp, seg_y[mask])
            function_fit = double_exponential(xx_exp, *popt)

        seg_mask = ~np.isnan(seg_y)
        polyfit_results[np.isin(xx, seg_x[seg_mask])] = function_fit 

    return polyfit_results

def mask_jumps(x, cutoff = 0.01, fs = 100, thres = 5):
    filtered = butter_filter(x, cutoff, fs, 'low')
    
    jump_mask = (np.abs(get_zscore(np.diff(filtered))) > thres) == False
    jump_mask = np.hstack([jump_mask, True])

    x_masked = np.where(jump_mask, x, np.nan)

    return x_masked

def get_zscore(x):
    return (x - np.nanmean(x))/np.nanstd(x)

def validate_cp(cp, lever_array):

    if len(lever_array)<3:
        cp = np.nan

    if round(lever_array[-1]/1000,0) == round(cp,0):
        cp = np.nan

    return cp

def exponential_decay(t, A, tau, C):
    return A * np.exp(-t/tau) + C

def exp_decay_tau(t, tau):
    return np.exp(-t/tau)

#https://github.com/ThomasAkam/photometry_preprocessing/blob/master/Photometry%20data%20preprocessing.ipynb
def double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
    '''Compute a double exponential function with constant offset.
    Parameters:
    t       : Time vector in seconds.
    const   : Amplitude of the constant offset. 
    amp_fast: Amplitude of the fast component.  
    amp_slow: Amplitude of the slow component.  
    tau_slow: Time constant of slow component in seconds.
    tau_multiplier: Time constant of fast component relative to slow. 
    '''
    tau_fast = tau_slow*tau_multiplier
    return const+amp_slow*np.exp(-t/tau_slow)+amp_fast*np.exp(-t/tau_fast)

def matrix_last_lever(df, column, before, after, zscored = False):

    lastlever = []

    for ii in range(1,len(df)-1):
        cc = df[column].values[ii] * 1000
        if not(np.isnan(cc)):
            cc = int(cc)
            da = np.hstack(df.loc[cc - before : cc + after, :].DA.values)
            
            if len(da) > before+after+1:
                da = da[:before+after+1] #silly way to ensure everything has the same dimension
            if len(da) < before+after+1:
                da = np.hstack([da, np.nan])
            
            if zscored:
                da = get_zscore(da)

            lastlever.append(da)

    return lastlever

#def signal2eventsnippets(time, signal, event_times, alignment_window, delta_time, nanify=False):
#    """
#    Takes a SIGNAL as input and outputs a matrix of signal SNIPPETS
#    spanning ALIGNMENT_WINDOW around EVENT_TIMES.
#
#    Parameters:
#    - time: array-like, time points of the signal
#    - signal: array-like, signal values corresponding to the time points
#    - event_times: array-like, times of the events around which snippets are extracted
#    - alignment_window: tuple or list, (start, end) of the alignment window around each event time
#    - delta_time: float, time step for the alignment
#    - nanify: bool, if True, overlapping snippets will be nanified
#
#    Returns:
#    - snippets: 2D NumPy array of shape (n_events, n_samples), the extracted snippets
#    - alignment_time: 1D NumPy array, the time points for the alignment
#    """
#
#    # Number of events
#    n_events = len(event_times)
#    
#    # Adjust alignment window to match the delta_time
#    alignment_window = delta_time * np.round(np.array(alignment_window) / delta_time)
#    alignment_time = np.arange(alignment_window[0], alignment_window[1] + delta_time, delta_time)
#    n_samples = len(alignment_time)
#
#    # Preallocation
#    #snippets = np.full((n_events, n_samples+1), np.nan)
#
#    snipps = []
#    # Iterate through events
#    for ii in range(n_events):
#        onset_time = event_times[ii] + alignment_window[0]
#        offset_time = event_times[ii] + alignment_window[1]
#        snippet_time = np.arange(onset_time, offset_time + delta_time/2, delta_time)
#        
#        snippet_flags = (snippet_time > time[0] - delta_time / 2) & (snippet_time < time[-1] + delta_time / 2)
#        time_flags = (time > onset_time - delta_time / 2) & (time < offset_time + delta_time / 2)
#    
#        snipps.append(signal[time_flags])
#
#    max_length = max(len(row) for row in snipps)
#    snippets = np.full((len(snipps), max_length), np.nan)
#    for ii, row in enumerate(snipps):
#        snippets[ii,:len(row)] = row
#
#    # Nanify overlapping snippets
#    if nanify:
#        iei = np.diff(np.concatenate(([0], event_times)))
#        valid_mask = np.zeros_like(snippets, dtype=bool)
#        
#        for i in range(n_events):
#            values = (alignment_time > -iei[i]) & (alignment_time < (iei[i + 1] if i + 1 < len(iei) else np.inf))
#            if len(values) == len(valid_mask[i,:]):
#                valid_mask[i, :] = values
#            elif len(values) > len(valid_mask[i,:]):
#                valid_mask[i,:] = values[:len(valid_mask[i,:])]
#            else:
#                valid_mask[i,:len(values)] = values
#        
#        snippets[~valid_mask] = np.nan
#
#    return snippets, alignment_time

def signal2eventsnippets(time, signal, event_times, alignment_window, delta_time, nanify=False):
    ## using this one!
    """
    Takes a SIGNAL as input and outputs a matrix of signal SNIPPETS
    spanning ALIGNMENT_WINDOW around EVENT_TIMES.

    Parameters:
    - time: array-like, time points of the signal
    - signal: array-like, signal values corresponding to the time points
    - event_times: array-like, times of the events around which snippets are extracted
    - alignment_window: tuple or list, (start, end) of the alignment window around each event time
    - delta_time: float, time step for the alignment
    - nanify: bool, if True, overlapping snippets will be nanified

    Returns:
    - snippets: 2D NumPy array of shape (n_events, n_samples), the extracted snippets
    - alignment_time: 1D NumPy array, the time points for the alignment
    """
    # Number of events
    n_events = len(event_times)
    
    # Adjust alignment window to match the delta_time
    n_samples = int(np.round((alignment_window[1] - alignment_window[0]) / delta_time)) + 1
    alignment_time = np.linspace(alignment_window[0], alignment_window[1], n_samples)

    # Preallocate snippets array
    snippets = np.full((n_events, n_samples), np.nan)

    # Iterate through events
    for ii in range(n_events):
        onset_time = event_times[ii] + alignment_window[0]
        offset_time = event_times[ii] + alignment_window[1]

        # Find indices of the signal within the alignment window
        snippet_flags = (time >= onset_time) & (time <= offset_time)
        snippet_time = time[snippet_flags]
        snippet_signal = signal[snippet_flags]

        # Interpolate or pad the snippet to match alignment_time
        if len(snippet_signal) > 0:
            snippet_interp = np.interp(alignment_time, snippet_time - event_times[ii], snippet_signal)
            snippets[ii, :] = snippet_interp

    # Nanify overlapping snippets
    if nanify:
        iei = np.diff(np.concatenate(([0], event_times)))
        valid_mask = np.zeros_like(snippets, dtype=bool)
        
        for i in range(n_events):
            values = (alignment_time > -iei[i]) & (alignment_time < (iei[i + 1] if i + 1 < len(iei) else np.inf))
            if len(values) == len(valid_mask[i, :]):
                valid_mask[i, :] = values
            elif len(values) > len(valid_mask[i, :]):
                valid_mask[i, :] = values[:len(valid_mask[i, :])]
            else:
                valid_mask[i, :len(values)] = values
        
        snippets[~valid_mask] = np.nan

    return snippets, alignment_time

def drop_nans_matrix(matrix):
    boolnan = []
    
    for row in matrix:
        boolnan.append(np.all(np.isnan(row)))
    
    return matrix[~np.array(boolnan)]

def drop_nans_array(array):
    return array[~np.isnan(array)]



def convert_timestamp(time_A, ref_A, ref_B):

    '''
    Converts time_A into time_B (A the old, B the new time frame)
    time_A: time to convert; can be a float or array
    ref_A: set of values in the original time reference
    ref_B: set of values in the new reference frame
    '''

    regress = scipy.stats.linregress(ref_A, ref_B)

    time_B = regress.intercept + regress.slope * time_A

    return time_B

def clean_outliers(time_series, z_score_threshold=3, interpolate = False):
    """
    Identifies outliers in a time series, either Nanifying them or replacing them with interpolated values.

    Parameters:
    - time_series: A pandas Series or numpy array representing the time series data.
    - z_score_threshold: A multiplier for the standard deviation to set the threshold for outliers.

    Returns:
    - A pandas Series or numpy array with outliers replaced by interpolated values.
    """
    if isinstance(time_series, np.ndarray):
        time_series = pd.Series(time_series)
    
    diffs = time_series.diff().abs()
    
    mean_diff = diffs.mean()
    std_diff = diffs.std()
    
    threshold = mean_diff + z_score_threshold * std_diff
    
    outliers = diffs > threshold
    
    clean_series = time_series.copy()
    
    clean_series[outliers] = np.nan
    
    if interpolate == True:
        result_series = clean_series.interpolate(method='linear', limit_direction='both')
    
    else:
        result_series = clean_series

    return result_series

def mask_array(array1, array2, range_val):
    '''
    array1: values around which the range is defined
    array2: values to be masked
    range_val: range around each value in array1 (window, it's +- these units)
    '''

    mask = np.zeros_like(array2, dtype=bool)

    for value in array1:
        mask |= (array2 >= (value - range_val)) & (array2 <= (value + range_val))

    masked_array = np.where(mask, np.nan, array2)

    return mask, masked_array    

def determine_trial_zeros(trialno):
    if len(str(trialno)) == 1:
        pre_trial_zeros = '00'
    elif len(str(trialno)) == 2:
        pre_trial_zeros = '0'
    else:
        pre_trial_zeros = ''

    return pre_trial_zeros


def query_and_compute_snippets(df, query_condition, events_column, time_column, signal_column, alignment_window, delta_time = 1/100, nanify = True):

    time = np.hstack(df[time_column].values)
    signal = np.hstack(df[signal_column].values)

    event_times = df.query(query_condition)[events_column].values
    event_times = event_times[~np.isnan(event_times)].flatten()
    snippets, alignment_time = signal2eventsnippets(time, signal, event_times, alignment_window, delta_time, nanify)

    return snippets, alignment_time

def plot_snippets(snippets, alignment_time, ax_heatmap, ax_mean, q_min=.01, q_max=.99, bool_plot_mean = True, cmap = 'viridis', color_DA_traces = 'blue'):

    v_min = np.nanquantile(np.hstack(snippets), q_min)
    v_max = np.nanquantile(np.hstack(snippets), q_max)

    ax_heatmap.imshow(snippets, aspect = 'auto', vmin = v_min, vmax = v_max, interpolation = 'nearest',
                extent = [alignment_time[0], alignment_time[-1], len(snippets), 1], cmap = cmap)
    if bool_plot_mean:
        ax_mean.plot(alignment_time, np.nanmean(snippets, axis = 0), color = color_DA_traces)
# %%

"""
.########.####..######...##.....##.########..########..######.
.##........##..##....##..##.....##.##.....##.##.......##....##
.##........##..##........##.....##.##.....##.##.......##......
.######....##..##...####.##.....##.########..######....######.
.##........##..##....##..##.....##.##...##...##.............##
.##........##..##....##..##.....##.##....##..##.......##....##
.##.......####..######....#######..##.....##.########..######.
"""
def figure_multiple_alignments(df, animaldate, path_save_figs, bool_normalise = False):
    FI_list = [15,30,60]
    nprots_list = [7,14,28]

    alignment_window = (-10,10)

    exp = df.experiment.unique()[0]
    bool_click = df.click.unique()[0]

    if bool_click:
        events_list =  ['pump_on_abs','pump_off_abs', 'corrected_cp_abs', 'cp_abs', 'preprelast_lever_abs', 'prelast_lever_abs', 'click_on_abs','last_lever_abs']
    else:
        events_list =  ['pump_on_abs','pump_off_abs', 'corrected_cp_abs', 'cp_abs', 'preprelast_lever_abs', 'prelast_lever_abs', 'last_lever_abs']

    fig, axs = plt.subplots(1,len(events_list), figsize = (len(events_list)*4,4), tight_layout = True, sharey = True)

    jj = 0
    for align_event in events_list:

        for ii in range(3):
            if exp == "c":
                snippets, alignment_time = query_and_compute_snippets(df, f'n_protocols == {nprots_list[ii]}', align_event, 'timestamp_session', 'DA', alignment_window)

                min_len = np.min([len(alignment_time), len(snippets[0])])

                sns.lineplot(ax = axs[jj], x = alignment_time[:min_len], y = np.nanmean(snippets, axis = 0)[:min_len],
                        label = nprots_list[ii], lw = .5, color = rwd_dict[nprots_list[ii]])


            else:
                if bool_normalise: # choosing the same "relative" time window in exps with varying FI, if decided to normalise
                    t_boundary = int(FI_list[ii]/3)
                    alignment_window = (-t_boundary,t_boundary)    
                
                snippets, alignment_time = query_and_compute_snippets(df, f'FI == {FI_list[ii]*1000}', align_event, 'timestamp_session', 'DA', alignment_window)
                
                if bool_normalise:
                    alignment_time = alignment_time/FI_list[ii]

                min_len = np.min([len(alignment_time), len(snippets[0])])

                sns.lineplot(ax = axs[jj], x = alignment_time[:min_len], y = np.nanmean(snippets, axis = 0)[:min_len],
                        label = FI_list[ii], lw = .5, color = FI_dict[FI_list[ii]])

        axs[jj].set_xlabel(align_event)
        jj+=1


    text_click = '_click' if bool_click else ''

    text_normalise = '_FInormalised' if bool_normalise else ''

    figtitle = f'{animaldate}_exp_{exp}{text_click}{text_normalise}_block_multiple_alignments'
    plt.suptitle(figtitle)

    fig.savefig(f'{path_save_figs}/multiple_alignments/{figtitle}.png', facecolor = 'white')

def figure_multiple_alignments_w_raster(df, path_save_figs, DA_column = 'DA', bool_normalise = False, bool_zscore = False, alignment_window = (-5,5)):

    ref_df = df
    FI_list = [15,30,60]
    nprots_list = [7,14,28]
    exp = df.experiment.unique()[0]
    bool_click = df.click.unique()[0]
    animaldate = df.animaldate.unique()[0]

    global_min = float('inf')
    global_max = float('-inf')
    
    print(bool_click)

    if bool_click:
        events_list =  ['pump_on_abs','pump_off_abs', 'corrected_cp_abs', 'cp_abs', 'preprelast_lever_abs', 'prelast_lever_abs', 'click_on_abs','last_lever_abs']
    else:
        events_list =  ['pump_on_abs','pump_off_abs', 'corrected_cp_abs', 'cp_abs', 'preprelast_lever_abs', 'prelast_lever_abs', 'last_lever_abs']

    fig, axs = plt.subplots(2,len(events_list), figsize = (len(events_list)*4,10),
                            tight_layout = True, height_ratios = [2,1], sharex = True)
    jj = 0

    for align_event in events_list:
        df = ref_df

        for ii in range(3):
            if exp == "c":
                snippets, alignment_time = query_and_compute_snippets(df, f'n_protocols == {nprots_list[ii]}', align_event, 'timestamp_session', DA_column, alignment_window)
                if bool_zscore:
                    snippets = np.apply_along_axis(compute_zscore, axis = 1, arr = snippets)
                    snippets = snippets[~np.isnan(snippets).any(axis=1)]
                min_len = np.min([len(alignment_time), len(snippets[0])])
                sns.lineplot(ax = axs[1,jj], x = alignment_time[:min_len], y = np.nanmean(snippets, axis = 0)[:min_len],
                        label = nprots_list[ii], lw = 1, color = rwd_dict[nprots_list[ii]])
                
            else:
                if bool_normalise: # choosing the same "relative" time window in exps with varying FI, if decided to normalise
                    t_boundary = int(FI_list[ii]/3)
                    alignment_window = (-t_boundary,t_boundary)    

                snippets, alignment_time = query_and_compute_snippets(df, f'FI == {FI_list[ii]}', align_event, 'timestamp_session', DA_column, alignment_window)
                if bool_zscore and len(snippets) > 0:
                    snippets = np.apply_along_axis(compute_zscore, axis = 1, arr = snippets)
                    snippets = snippets[~np.isnan(snippets).any(axis=1)]

                if bool_normalise:
                    alignment_time = alignment_time/FI_list[ii]
                
                if len(snippets) == 0:
                    continue

                min_len = np.min([len(alignment_time), len(snippets[0])])
                sns.lineplot(ax = axs[1,jj], x = alignment_time[:min_len], y = np.nanmean(snippets, axis = 0)[:min_len],
                    label = FI_list[ii], lw = 1, color = FI_dict[FI_list[ii]])

            global_min = min(global_min, np.nanmin(np.nanmean(snippets, axis = 0)))
            global_max = max(global_max, np.nanmax(np.nanmean(snippets, axis = 0)))

        snippets, alignment_time = query_and_compute_snippets(df, f'FI > 0', align_event, 'timestamp_session', DA_column, alignment_window)
        if bool_zscore and len(snippets) > 0:
            snippets = np.apply_along_axis(compute_zscore, axis = 1, arr = snippets)
            snippets = snippets[~np.isnan(snippets).any(axis=1)]

        plot_snippets(snippets, alignment_time, axs[0,jj], axs[0,jj], bool_plot_mean = False)

        axs[0,jj].invert_yaxis()

        if 'cp' in align_event:
            df = df.query('bool_cp')
            df['trialno'] = np.arange(1,len(df)+1)

        for trialno in df.trialno.unique():
            y_start = (trialno - 1) / snippets.shape[0]
            y_end = trialno / snippets.shape[0]

            n_protocols = df.query(f'trialno == {trialno}').n_protocols.values[0]
            FI = int(df.query(f'trialno == {trialno}').FI.values[0])
            
            color = rwd_dict[n_protocols] if exp == 'c' else FI_dict[FI]

            axs[0,jj].axvspan(alignment_window[0], alignment_window[0]+.5, ymin = y_start, ymax = y_end, color = color)
            axs[0,jj].axvspan(alignment_window[1]-.5, alignment_window[1], ymin = y_start, ymax = y_end, color = color)

        axs[1,jj].set_xlabel(align_event)

        axs[1,jj].axvline(0, color = 'grey', lw = .5)
        axs[0,jj].axvline(0, color = 'white', lw = .5)

        #if jj < len(events_list):
        #    axs[1,jj].sharey(axs[1,0])

        if jj > 0:
            figures.remove_legend(axs[1,jj])

        jj+=1

    globals_dist = np.abs(global_max - global_min)
    [axs[1,jj].set_ylim(global_min-.05*globals_dist, global_max+.05*globals_dist) for jj in range(len(events_list))]

    text_click = '_click' if bool_click else ''
    text_normalise = '_FInormalised' if bool_normalise else ''
    text_zscore = '_zscored' if bool_zscore else ''
    figtitle = f'{animaldate}_exp_{exp}{text_click}{text_zscore}{text_normalise}_block_multiple_alignments_raster'
    plt.suptitle(figtitle)

    fig.savefig(fr'{path_save_figs}/{figtitle}.png', facecolor = 'white')

# %%
