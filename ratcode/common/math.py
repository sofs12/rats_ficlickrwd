import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def decay_conv(array, tau = 1000):
    '''
    Return an array

    Parameters:
    array: input to be convolved. Typically an array of zeros and ones
    tau: time constant of the exponential. Default set for miliseconds units
    '''

    t = np.arange(0,len(array))
    g = np.exp(-t/tau)
    conv = np.convolve(array,g)[:len(t)] * (t[1]-t[0])

    return conv

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

def signal2eventsnippets(time, signal, event_times, alignment_window, delta_time, nanify=False):
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
    alignment_window = delta_time * np.round(np.array(alignment_window) / delta_time)
    alignment_time = np.arange(alignment_window[0], alignment_window[1] + delta_time, delta_time)
    n_samples = len(alignment_time)

    # Preallocation
    #snippets = np.full((n_events, n_samples+1), np.nan)

    snipps = []
    # Iterate through events
    for ii in range(n_events):
        onset_time = event_times[ii] + alignment_window[0]
        offset_time = event_times[ii] + alignment_window[1]
        snippet_time = np.arange(onset_time, offset_time + delta_time/2, delta_time)
        
        snippet_flags = (snippet_time > time[0] - delta_time / 2) & (snippet_time < time[-1] + delta_time / 2)
        time_flags = (time > onset_time - delta_time / 2) & (time < offset_time + delta_time / 2)
    
        snipps.append(signal[time_flags])

    max_length = max(len(row) for row in snipps)
    snippets = np.full((len(snipps), max_length), np.nan)
    for ii, row in enumerate(snipps):
        snippets[ii,:len(row)] = row

    # Nanify overlapping snippets
    if nanify:
        iei = np.diff(np.concatenate(([0], event_times)))
        valid_mask = np.zeros_like(snippets, dtype=bool)
        
        for i in range(n_events):
            valid_mask[i, :] = (alignment_time > -iei[i]) & (alignment_time < (iei[i + 1] if i + 1 < len(iei) else np.inf))
        
        snippets[~valid_mask] = np.nan

    return snippets, alignment_time

def drop_nans_matrix(matrix):
    boolnan = []
    
    for row in matrix:
        boolnan.append(np.all(np.isnan(row)))
    
    return matrix[~np.array(boolnan)]

def drop_nans_array(array):
    return array[~np.isnan(array)]

def drop_nan_rows_in_matrix(matrix):
    """
    Remove any rows from a NumPy array that contain NaN values.

    Parameters
    ----------
    matrix : np.ndarray
        Input 2D array.

    Returns
    -------
    np.ndarray
        Array with rows containing NaNs removed.
    """
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)

    mask = ~np.isnan(matrix).any(axis=1)
    return matrix[mask]