import numpy as np


def compute_rate_aligned_cp(df, conditions, count_lever = 3, binW = 0.1, time_window = (-3,5)):
    events = df.query(f'bool_cp and count_lever > {count_lever} and {conditions}').lever_aligned_cp.values
    bins = np.arange(time_window[0],time_window[1],binW)
    counts, bins = np.histogram(np.hstack(events),bins)
    bins = bins[:-1]
    counts = counts/(binW*len(events))
    return bins, counts

def get_lever_press_matrix(df, time_window = [-3,6], bin_width=0.1):
    
    bins = np.arange(time_window[0], time_window[1] + bin_width, bin_width)

    n_trials = len(df)
    lever_press_matrix = np.full((len(df), len(bins)-1), np.nan)

    for i, trial_data in enumerate(df.lever_aligned_cp):
        if isinstance(trial_data, np.ndarray):
            counts, _ = np.histogram(trial_data, bins=bins)
            lever_press_matrix[i, :len(counts)] = counts

    # nanify after the moment the trial is over
    for idx, val in enumerate(df.lever_aligned_cp.apply(lambda x: x[-1]).values):
        if val < bins[-1]:
            bin_idx = np.digitize(val, bins) - 1  # subtract 1 for 0-based index
            lever_press_matrix[idx, bin_idx+1:] = np.nan

    #for i in range(lever_press_matrix.shape[0]):
    #    last_one_index = np.where(lever_press_matrix[i] == 1)[0]
    #    if len(last_one_index) > 0:
    #        lever_press_matrix[i, last_one_index[-1] + 1:] = np.nan

    frac_finished_trials = np.sum(np.isnan(lever_press_matrix), axis = 0)/n_trials
    trials_still_ongoing = (1-frac_finished_trials)*n_trials
    mean_rate = np.nansum(lever_press_matrix, axis = 0)/trials_still_ongoing/bin_width
                          
    SEM = np.nanstd(lever_press_matrix, axis=0) / np.sqrt(trials_still_ongoing)

    return lever_press_matrix, bins[:-1]+bin_width/2, mean_rate, SEM, frac_finished_trials