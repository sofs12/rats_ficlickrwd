import numpy as np

def detect_rising_edge(ttl_trace, full_time_vector):
    diff_trace = np.diff(ttl_trace.astype(int))
    rising_edge_indices = np.where(diff_trace == 1)[0] + 1
    ttl_timestamps = full_time_vector[rising_edge_indices]

    return ttl_timestamps

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