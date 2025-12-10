#%%
## need to be running in the neuro environment
import os
#os.chdir('c:\\Users\\Admin\\Documents\\git\\ratanalysis')
#from ratcode.globe.globe import *
#import ratcode.utils.change_point as change_point
#from ratcode.globe.globe import *

nprots_order_blocks = [7,14,28]
color_nprots_blocks = ["#636f82", "#81a6fc", "#2e6dff"]

FI_list = [15,30,60]
nprots_list = [7,14,28]
exp_list = ['a', 'b', 'c']

agg_blocks_FI_dic = {
    15: "#cba6e3",
    30: "#81a6fc",
    60: "#77d674"
}
color_FI_blocks = list(agg_blocks_FI_dic.values())

from scipy.signal import lfilter
import spikeinterface.full as si
import spikeinterface.extractors as se
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss
import spikeinterface.qualitymetrics as sqm
import spikeinterface.exporters as sexp
import spikeinterface.widgets as sw


import warnings
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import glob
import seaborn as sns
import pandas as pd
from pathlib import Path
import shutil
import cmath
from scipy.ndimage import convolve1d
from matplotlib.colors import ListedColormap
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
import scipy.signal as signal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats


plt.rc('axes', titlesize=18)     # fontsize of the axes title
plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=13)    # legend fontsize
plt.rc('font', size=13)          # controls default text sizes
plt.rc('axes.spines', top = False, right = False) #equivalent to sns.despine(top = True)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['figure.titlesize'] = 'large'


"""
..######...##........#######..########.....###....##........######.
.##....##..##.......##.....##.##.....##...##.##...##.......##....##
.##........##.......##.....##.##.....##..##...##..##.......##......
.##...####.##.......##.....##.########..##.....##.##........######.
.##....##..##.......##.....##.##.....##.#########.##.............##
.##....##..##.......##.....##.##.....##.##.....##.##.......##....##
..######...########..#######..########..##.....##.########..######.
""" 

sampling_frequency = 30000
window = (-8,8)

cmap_FI = ListedColormap(list(color_FI_blocks))
cmap_nprots = ListedColormap(list(color_nprots_blocks))


def compute_alignments(syncdf, bool_click): # might need readjusting for no click sessions, needs testing
    exp = determine_experiment(syncdf)
    #bool_click = (len(np.unique(np.hstack(syncdf.click))) > 1)

    lvr = drop_nan(np.hstack(syncdf.lever_npx.values))
    poke = drop_nan(np.hstack(syncdf.poke_npx.values))
    rwd_onset = drop_nan(np.hstack(syncdf.rwd_onset_npx.values))
    cp = drop_nan(np.hstack(syncdf.cp.values))
    cp_corrected = drop_nan(np.hstack(syncdf.cp_corrected.values))
    if bool_click:
        click_onset = drop_nan(np.hstack(syncdf.click.values))
    rwd_offset = drop_nan(np.hstack(syncdf.query('trial_duration_s > 2').npx_time.values))

    key_dict = {
        'rwd_offset': 'npx_time',
        'poke': 'poke_npx',
        'lvr': 'lever_npx',
        'cp_corrected': 'cp_corrected',
        'cp': 'cp',
        'click': 'click',
        'rwd_onset': 'rwd_onset_npx'
    }

    cmap_FI = ListedColormap(list(color_FI_blocks))
    cmap_nprots = ListedColormap(list(color_nprots_blocks))

    if exp == 'a':
        cmap_nprots = ListedColormap(list([color_nprots_blocks[1],color_nprots_blocks[1]]))
    if exp == 'c':
        cmap_FI = ListedColormap(list([color_FI_blocks[1],color_FI_blocks[1]]))

    if bool_click:
        alignments_dict = {'rwd_offset':rwd_offset, 'rwd_onset':rwd_onset, 'lvr':lvr, 'poke':poke,
                       'click': click_onset, 'cp': cp, 'cp_corrected':cp_corrected}
        alignments = ['rwd_offset', 'poke', 'lvr', 'cp_corrected', 'cp', 'click', 'rwd_onset']
    else:
        alignments_dict = {'rwd_offset':rwd_offset, 'rwd_onset':rwd_onset, 'lvr':lvr, 'poke':poke,
                       'cp': cp, 'cp_corrected':cp_corrected}
        alignments = ['rwd_offset', 'poke', 'lvr', 'cp_corrected', 'cp', 'rwd_onset']

    return alignments, alignments_dict, key_dict, cmap_FI, cmap_nprots
#%%
def define_all_paths(animal, date, bool_ibl_drift = False, bool_raw_ephys = False):
    dropbox_path = r'D:\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\Data\FIClickRwd'
    dropbox_neuro_path = rf'{dropbox_path}\ephys\{animal}'

    fig_save_path = rf'{dropbox_path}\analysis_ephys\{animal}_{date}'
    if not(os.path.exists(fig_save_path)):
        os.mkdir(fig_save_path)

    save_sync_path = glob.glob(fr'{dropbox_neuro_path}\{animal}{date}*\*')[0]

    if bool_ibl_drift:
        # drift_amplitude
        ibl_sorter_path =  glob.glob(fr'{dropbox_neuro_path}\{animal}{date}*\{animal}{date}*\ibl_sorter_results_drift_amplitude')[0]
    else:
        ibl_sorter_path =  glob.glob(fr'{dropbox_neuro_path}\{animal}{date}*\{animal}{date}*\ibl_sorter_results')[0]

    if bool_raw_ephys:
        neuro_path = glob.glob(rf"H:\{animal}{date}*\{animal}{date}*")[1]
        #neuro_path = glob.glob(rf"G:\EPHYS\{animal}{date}*\{animal}{date}*")[1]
    else:
        neuro_path = 'undefined'
    
    return fig_save_path, save_sync_path, ibl_sorter_path, neuro_path

#%%

"""
.########.##.....##.##....##..######..########.####..#######..##....##..######.
.##.......##.....##.###...##.##....##....##.....##..##.....##.###...##.##....##
.##.......##.....##.####..##.##..........##.....##..##.....##.####..##.##......
.######...##.....##.##.##.##.##..........##.....##..##.....##.##.##.##..######.
.##.......##.....##.##..####.##..........##.....##..##.....##.##..####.......##
.##.......##.....##.##...###.##....##....##.....##..##.....##.##...###.##....##
.##........#######..##....##..######.....##....####..#######..##....##..######.
"""

def detect_rising_edges(arr):
    # Ensure it's a 1D array
    #arr = arr.flatten()
    return np.where((arr[:-1] == 0) & (arr[1:] > 1))[0] + 1

def align_spikes_to_ttl(spike_times, ttl_times, window=(-0.1, 0.3)):
    """
    Aligns spike times to TTL pulses.
    
    Parameters:
    - spike_times: array of spike times (seconds)
    - ttl_times: array of TTL event times (seconds)
    - window: time window around each TTL event (start, end) in seconds
    
    Returns:
    - aligned_spikes: list of arrays, each containing spike times relative to a TTL pulse
    """
    aligned_spikes = []
    
    for ttl in ttl_times:
        # Find spikes in the time window around each TTL event
        spikes_relative = spike_times - ttl
        spikes_in_window = spikes_relative[(spikes_relative >= window[0]) & (spikes_relative <= window[1])]
        aligned_spikes.append(spikes_in_window)

    return aligned_spikes


def align_cluster_spikes(cluster_id, spike_times, spike_clusters, ttl_times, window=(-0.1, 0.3)):
    cluster_spikes = spike_times[spike_clusters == cluster_id]
    return align_spikes_to_ttl(cluster_spikes, ttl_times, window)


def compute_FR(spikes_aligned, window, binW = 0.1):
    """
    computes the firing rate from a matrix of spike times
    returns time, firing rate
    """

    counts = np.histogram(np.concatenate(spikes_aligned), bins = np.arange(window[0], window[1], binW))

    return counts[1][:-1], counts[0]/(binW*len(spikes_aligned))

def compute_corrected_cp(lvr_array, cp):
    lvr_array = lvr_array[lvr_array>=cp]
    interpress = np.diff(lvr_array)
    return cp - np.mean(interpress)

def pad_array(small_array, large_array):
    pad_size = len(large_array) - len(small_array)

    return np.pad(small_array, (pad_size, 0), mode = 'constant', constant_values=np.nan)

def drop_nan(array):
    return array[~np.isnan(array)]

def moving_average(array, window_size = 5):
    kernel = np.ones(window_size)/window_size
    return np.convolve(array, kernel, mode = 'same')

def compute_zscore(arr):
    return (arr-np.nanmean(arr))/np.nanstd(arr)

def determine_experiment(syncdf):
    lenFI = len(syncdf.FI.unique())
    lenprots = len(syncdf.n_protocols.unique())

    if lenFI == 3 and lenprots == 1:
        exp = 'a'
    if lenFI == 3 and lenprots == 3:
        exp = 'b'
    if lenFI == 1 and lenprots ==3:
        exp = 'c'

    return exp
#%%
def determine_cell_type(cluster_id, sorted_data, syncdf):

    spike_times_cluster = sorted_data.spike_times[sorted_data.spike_clusters == cluster_id]/sorted_data.sampling_frequency

    template_id = sorted_data.spike_templates[sorted_data.spike_clusters==cluster_id][0]
    template_waveform = sorted_data.templates[template_id]

    peak_amplitudes = np.max(template_waveform, axis = 0) - np.min(template_waveform, axis = 0)
    min_lim, max_lim = np.quantile(peak_amplitudes[peak_amplitudes!=0],[.05,.95])#, binwidth=.001)

    channels_to_consider_idx = np.where(peak_amplitudes > min_lim)[0]
    channels_to_consider = sorted_data.channel_map[channels_to_consider_idx]

    waveform_ms = np.arange(0,sorted_data.templates.shape[1]/sorted_data.sampling_frequency*1000,1/sampling_frequency*1000)
    mean_waveform = np.mean(sorted_data.templates[template_id,:,channels_to_consider],axis = 0)


    template_duration_FSIs = .4 #ms 400us 
    post_spike_suppression_TANs_ms = 40 #40 ms


    #trough-to-peak time
    trough = np.argmin(mean_waveform)
    peak = trough + np.argmax(mean_waveform[trough:])
    trough_to_peak_ms = (peak - trough)/sorted_data.sampling_frequency*1000
    print(f'trough to peak (ms): {trough_to_peak_ms}')


    if trough_to_peak_ms <= template_duration_FSIs:
        print(f'smaller than {template_duration_FSIs}. either FSI or unidentified interneuron')

        #compute long interspike interval
        ISI = np.diff(spike_times_cluster) # in seconds
        total_recording_time = syncdf.npx_time.dropna().values[-1] - syncdf.npx_time.dropna().values[0] # in seconds
        long_interspike_ratio = len(ISI[ISI > 2])/total_recording_time*100
        print(f'long interspike ratio (%): {long_interspike_ratio}')

        if long_interspike_ratio > 10:
            cell_type = 'unidentified interneuron'

        else:
            cell_type = 'FSI'

    else:
        print(f'larger than {template_duration_FSIs}. either MSN or TAN')
        print('computing post spike suppression')

        autocorrelogram_900ms = np.hstack(align_spikes_to_ttl(spike_times_cluster,spike_times_cluster,(0,.9)))
        autocorrelogram_900ms = autocorrelogram_900ms[autocorrelogram_900ms!=0]

        counts, bins = np.histogram(autocorrelogram_900ms, bins = np.arange(0,.9,.001))

        #average FR (Hz) in bins 600 to 900 ms
        av_FR_600_to_900 = np.sum(counts[600:900])/300

        #plt.plot(bins[:-1],counts)
        #plt.axhline(av_FR_600_to_900)

        post_spike_suppression_ms = np.where(counts > av_FR_600_to_900)[0][0]
        #post_spike_suppression_ms = len(np.where(counts < av_FR_600_to_900)[0])
        print(f'suppression: {post_spike_suppression_ms} ms')

        if post_spike_suppression_ms > post_spike_suppression_TANs_ms:
            print(f'suppression larger than {post_spike_suppression_TANs_ms} ms')
            cell_type = 'TAN'
        else:
            print(f'suppression smaller than {post_spike_suppression_TANs_ms} ms')
            cell_type = 'MSN'

    print(f'cell identification: {cell_type}')

    return cell_type
#%%
"""
.########..##........#######..########....########.##.....##.##....##..######..########.####..#######..##....##..######.
.##.....##.##.......##.....##....##.......##.......##.....##.###...##.##....##....##.....##..##.....##.###...##.##....##
.##.....##.##.......##.....##....##.......##.......##.....##.####..##.##..........##.....##..##.....##.####..##.##......
.########..##.......##.....##....##.......######...##.....##.##.##.##.##..........##.....##..##.....##.##.##.##..######.
.##........##.......##.....##....##.......##.......##.....##.##..####.##..........##.....##..##.....##.##..####.......##
.##........##.......##.....##....##.......##.......##.....##.##...###.##....##....##.....##..##.....##.##...###.##....##
.##........########..#######.....##.......##........#######..##....##..######.....##....####..#######..##....##..######.
"""
def plot_raster(ax,spikes_aligned, window):
    for i, spikes in enumerate(spikes_aligned):
        ax.scatter(spikes, np.ones_like(spikes)*i, color='black', s = .05) #before s = 0.01
    ax.set_xlim(window[0],window[1])
    ax.set_ylim(-0.5, len(spikes_aligned) - 0.5)
    ax.axvline(0, color = 'purple', lw = 1)
#%%
def produce_neuron_fig(cluster_id, alignment_times, sorted_data, sorting_label = 'good', window = (-10,10), save_fig = True, fig_save_path = None):
    """
    quick plot, aligning only to TTL -- for other alignments see produce_neuron_bhv_fig
    """

    spike_times_cluster = sorted_data.spike_times[sorted_data.spike_clusters == cluster_id]/sorted_data.sampling_frequency

    spikes_aligned = align_spikes_to_ttl(spike_times_cluster,alignment_times, window=window)

    fig, axs = plt.subplots(2, figsize=(6,4), tight_layout = True, sharex = 'col')

    for i, spikes in enumerate(spikes_aligned):
        #plt.vlines(spikes, i - 0.4, i + 0.4, color='black', linewidth=0.2)  # Thin vertical lines
        axs[1].scatter(spikes, np.ones_like(spikes)*i, color='black', s = .01)  # Thin vertical lines
        #axs[1].plot(spikes, np.ones_like(spikes)*i, color='black', marker = '.', lw = .01)  # Thin vertical lines

    axs[1].set_xlim(window[0],window[1])

    axs[1].set_ylim(-0.5, len(spikes_aligned) - 0.5)
    axs[1].axvline(0, color = 'purple', lw = 1)

    #sns.histplot(np.concatenate(spikes_aligned), ax = axs[0], stat = 'density', element = 'step', binwidth=.05, color = 'grey')
    time, firing_rate = compute_FR(spikes_aligned, window)
    axs[0].plot(time, firing_rate, color = 'grey')
    axs[0].axvline(0, color = 'purple', lw = 1)
    axs[0].set_ylabel('firing rate (Hz)')

    axs[1].set_ylabel('trials')
    axs[1].set_yticks([])
    axs[1].set_xlabel('time since TTL = last lever press (s)')
    axs[1].set_ylim(0)

    #axs[0,1].plot(np.mean(templates[cluster_id], axis = 1))

    ax_inset = inset_axes(axs[1], width="30%", height="30%", loc='upper right')
    ax_inset.plot(np.mean(sorted_data.templates[sorted_data.spike_templates[sorted_data.spike_clusters == cluster_id][0]], axis = 1), color='purple')
    ax_inset.axis('off')


    figtitle = f'cluster_id {cluster_id} - {sorting_label} - {sorted_data.date}'
    plt.suptitle(figtitle)
    if save_fig:
        plt.savefig(rf'{fig_save_path}/{figtitle}.png', facecolor = 'white')
        plt.close()

#%%
def half_gaussian_kernel(std_ms = 500, bin_size_ms = 20):
    std_bins = std_ms/bin_size_ms
    size = int(3*std_bins)
    x = np.arange(0,size)

    kernel = stats.norm.pdf(x,loc = 0, scale = std_bins)
    kernel /= kernel.sum()
    return kernel

def get_psths_smooth(cluster_ids, ttls, pre_time, post_time, sorted_data, psthbin = 0.01, kernel = half_gaussian_kernel()):
    
    '''
    all arguments in seconds! returns in ms
    '''
    
    #psthbin = 10  # Bin width in ms
    #pre_time = 0 #1000  # Time before event in ms
    #post_time = 60000 #2000  # Time after event in ms

    #convert to ms -- give some time before and after so that we can smooth without messing up the edges, and only then select the window
    pre_time = (pre_time-10)*1000
    post_time = (post_time+10)*1000
    psthbin = psthbin*1000

    time_bins = np.arange(pre_time, post_time, psthbin)
    n_neurons = len(cluster_ids)
    n_trials = len(ttls)

    spike_counts = np.zeros((n_trials, n_neurons, len(time_bins) - 1))

    for trial_idx, ttl in enumerate(ttls):
        for neuron_idx, cluster in enumerate(cluster_ids):
            neuron_spikes = sorted_data.spike_times[sorted_data.spike_clusters == cluster]/sorted_data.sampling_frequency  # Get spikes for this neuron
            aligned_spikes = neuron_spikes - ttl  # Align spikes to TTL
            aligned_spikes = aligned_spikes*1000 # to work in ms
            spike_counts[trial_idx, neuron_idx, :] = np.histogram(aligned_spikes, bins=time_bins)[0]

    psths = np.nanmean(spike_counts, axis=0)  # Average over trials
    
    psths_smoothed = convolve1d(psths,kernel,axis = 1,mode = 'reflect')

    psths_smoothed_zscored = np.apply_along_axis(compute_zscore, axis = 1, arr = psths_smoothed)

    pad = int((10*1000)/psthbin)

    return spike_counts[:,pad-1:-pad], psths[:,pad-1:-pad], psths_smoothed[:,pad-1:-pad], psths_smoothed_zscored[:,pad-1:-pad]

#def get_psths_smooth_neuron_by_neuron(spike_times, ttls, pre_time = -4, post_time = 4, psthbin = 0.01, kernel = half_gaussian_kernel()):
#    
#    '''
#    all arguments in seconds! returns in ms
#    '''
#    
#    #psthbin = 10  # Bin width in ms
#    #pre_time = 0 #1000  # Time before event in ms
#    #post_time = 60000 #2000  # Time after event in ms
#
#    #convert to ms -- give some time before and after so that we can smooth without messing up the edges, and only then select the window
#    pre_time = (pre_time-10)*1000
#    post_time = (post_time+10)*1000
#    psthbin = psthbin*1000
#
#    n_trials = len(ttls)
#
#    time_bins = np.arange(pre_time, post_time, psthbin)
#
#    spike_counts = np.zeros((n_trials, len(time_bins) - 1))
#
#    for trial_idx, ttl in enumerate(ttls):
#        aligned_spikes = spike_times - ttl  # Align spikes to TTL
#        aligned_spikes = aligned_spikes*1000 # to work in ms
#        spike_counts[trial_idx, :] = np.histogram(aligned_spikes, bins=time_bins)[0]/(psthbin/1000) # to return in spks/s
#
#    psths = np.nanmean(spike_counts, axis=0)  # Average over trials
#
#    #psths_smoothed = convolve1d(psths,kernel,mode = 'reflect')
#    psths_smoothed = np.apply_along_axis(lambda s: lfilter(kernel, [1.0], s), axis=0, arr=psths)
#    #psths_smoothed_zscored = np.apply_along_axis(compute_zscore, axis = 1, arr = psths_smoothed)
#    pad = int((10*1000)/psthbin)
#
#    return spike_counts[:,pad-1:-pad], psths[pad-1:-pad], psths_smoothed[pad-1:-pad] #, psths_smoothed_zscored[:,pad-1:-pad]

import numpy as np
from scipy.signal import lfilter
#%%
def get_psths_smooth_neuron_by_neuron(
    spike_times, ttls, pre_time=-4, post_time=4, psthbin=0.01, kernel=None,
    pad_seconds=10.0  # keep, but converted to an integer number of bins
):
    """
    All args in seconds. Returns:
      - spike_counts: [n_trials, n_core_bins] in spks/s
      - psth: [n_core_bins]
      - psth_smoothed: [n_core_bins]
    """

    if kernel is None:
        kernel = np.array([1.0])

    # integer ms grid
    psthbin_ms = int(round(psthbin * 1000.0))
    pre_ms_core  = int(round(pre_time  * 1000.0))
    post_ms_core = int(round(post_time * 1000.0))

    # core duration (must be an integer # of bins ideally)
    core_ms = post_ms_core - pre_ms_core
    n_core_bins_exact = core_ms / psthbin_ms
    n_core_bins = int(round(n_core_bins_exact))

    # If not exactly divisible, snap the post boundary to the nearest bin count
    if not np.isclose(n_core_bins_exact, n_core_bins):
        post_ms_core = pre_ms_core + n_core_bins * psthbin_ms
        core_ms = post_ms_core - pre_ms_core

    # make padding an integer number of bins
    pad_ms_target = int(round(pad_seconds * 1000.0))
    pad_bins = int(np.ceil(pad_ms_target / psthbin_ms))
    pad_ms_exact = pad_bins * psthbin_ms  # exact multiple

    # total bins (padded)
    n_bins_total = n_core_bins + 2 * pad_bins

    # construct exact bin edges on the integer grid
    start_ms = pre_ms_core - pad_ms_exact
    time_bins = start_ms + np.arange(n_bins_total + 1, dtype=int) * psthbin_ms

    # histogram per trial
    n_trials = len(ttls)
    spike_counts = np.zeros((n_trials, n_bins_total), dtype=float)
    for trial_idx, ttl in enumerate(ttls):
        aligned_spikes_ms = (spike_times - ttl) * 1000.0
        spike_counts[trial_idx, :] = np.histogram(aligned_spikes_ms, bins=time_bins)[0] / (psthbin_ms / 1000.0)

    psth = np.nanmean(spike_counts, axis=0)
    psth_smoothed = lfilter(kernel, [1.0], psth)

    # trim exactly the integer number of pad bins
    core_counts   = spike_counts[:, pad_bins : pad_bins + n_core_bins]
    core_psth     = psth[pad_bins : pad_bins + n_core_bins]
    core_smoothed = psth_smoothed[pad_bins : pad_bins + n_core_bins]

    # sanity check: for your case core=-6000..15000, bin=150 ms -> 140 bins
    assert core_psth.shape[0] == n_core_bins, (
        f"Got {core_psth.shape[0]} bins, expected {n_core_bins}. "
        f"(bin={psthbin_ms} ms, core={pre_ms_core}..{post_ms_core} ms, pad_bins={pad_bins})"
    )

    return core_counts, core_psth, core_smoothed





def get_psths_across_cells(
    neural_df,
    bhv_df,
    cells_to_use,
    event_name,
    query_condition=None,  # New argument
    pre_time=-4,
    post_time=4,
    psthbin=0.01,
    kernel=half_gaussian_kernel(),
    quiet = True,
):
    """
    Parameters:
    - neural_df: dataframe with neurons (each row = one unit)
    - bhv_df: dataframe with trials
    - cells_to_use: list of (animal, date, cluster_id) tuples
    - event_name: name of the behavioral timestamp to align to
    - query_condition: string used to filter bhv_df (e.g. 'trial_type == "FI30"')
    Returns:
        - all_psth_matrices: list of spike count matrices [n_trials x n_bins] per neuron
        - all_psths: list of raw psth vectors per neuron
        - all_smoothed: list of smoothed psths
    """
    all_psth_matrices = []
    all_psths = []
    all_smoothed = []
    cell_id_dict = []
    all_trialno = []

    for animal, date, cluster_id in cells_to_use:
        
        # --- Get spikes for this neuron
        neuron = neural_df[(neural_df['animal'] == animal) & (neural_df['date'] == date) & (neural_df['cluster_id'] == cluster_id)]
        if neuron.empty:
            if quiet == False:
                print(f"Neuron {cluster_id} on {date} not found. Skipping.")
            continue
        spike_times = neuron.iloc[0]['spike_times']

        # --- Get behavior rows for this session
        session_trials = bhv_df[(bhv_df['animal'] == animal) & (bhv_df['date'] == date)]
        if session_trials.empty:
            if quiet == False:
                print(f"No behavior data found for {date}. Skipping.")
            continue

        # --- Apply condition if provided
        if query_condition:
            try:
                session_trials = session_trials.query(query_condition)
            except Exception as e:
                print(f"Query failed for condition '{query_condition}': {e}")
                continue

        if event_name not in session_trials.columns:
            raise ValueError(f"Event '{event_name}' not found in behavior dataframe.")

        #ttls = session_trials[event_name].dropna().values
        ttls = session_trials[event_name].values
        nan_mask = np.isnan(ttls)
        ttls = ttls[~nan_mask]
        trials = session_trials['trialno'].values[~nan_mask]

        if len(ttls) == 0:
            print(f"No valid TTLs found for neuron {cluster_id} on {date} after filtering. Skipping.")
            continue

        # --- Compute PSTH
        counts, raw, smoothed = get_psths_smooth_neuron_by_neuron(
            spike_times, ttls,
            pre_time=pre_time,
            post_time=post_time,
            psthbin=psthbin,
            kernel=kernel
        )

        all_psth_matrices.append(counts)
        all_psths.append(raw)
        all_smoothed.append(smoothed)
        cell_id_dict.append({
            'animal': animal,
            'date': date,
            'cluster_id': cluster_id
            })
        all_trialno.append(trials)

    return all_psth_matrices, all_psths, all_smoothed, cell_id_dict, all_trialno
#%%
def plot_templates(cluster_id, axs, sorted_data, cluster_info):
    template_id = sorted_data.spike_templates[sorted_data.spike_clusters==cluster_id][0]
    template_waveform = sorted_data.templates[template_id]

    peak_amplitudes = np.max(template_waveform, axis = 0) - np.min(template_waveform, axis = 0)
    min_lim, max_lim = np.quantile(peak_amplitudes[peak_amplitudes!=0],[.05,.95])#, binwidth=.001)

    channels_to_consider_idx = np.where(peak_amplitudes > min_lim)[0]
    channels_to_consider = sorted_data.channel_map[channels_to_consider_idx]
    channels_to_consider_positions = sorted_data.channel_positions[channels_to_consider]

    #best_channel_idx = np.argmax(peak_amplitudes)
    #best_channel = channel_map[best_channel_idx]

    best_channel = cluster_info.query(f'cluster_id == {cluster_id}').ch.values[0]
    #depth = cluster_info.query(f'cluster_id == {cluster_id}').depth.values
    #shank = cluster_info.query(f'cluster_id == {cluster_id}').sh.values

    waveform_ms = np.arange(0,sorted_data.templates.shape[1]/sorted_data.sampling_frequency*1000,1/sampling_frequency*1000)
    mean_waveform = np.mean(sorted_data.templates[template_id,:,channels_to_consider],axis = 0)
    non_null_timestamps = np.where(mean_waveform!=0)

    axs.plot(waveform_ms[non_null_timestamps],np.hstack(sorted_data.templates[template_id,non_null_timestamps,best_channel]), color = 'grey', lw = 3)

    for channel in channels_to_consider:
        axs.plot(waveform_ms[non_null_timestamps],np.hstack(sorted_data.templates[template_id,non_null_timestamps,sorted_data.channel_map[channel]]), color = 'grey', alpha = 0.2)

    #axs.annotate(f'channel {best_channel}', xy = (2,min_lim))

#%%
def plot_probe_sofia(cluster_id, axs, sorted_data, cluster_info):
    template_id = sorted_data.spike_templates[sorted_data.spike_clusters==cluster_id][0]
    template_waveform = sorted_data.templates[template_id]

    peak_amplitudes = np.max(template_waveform, axis = 0) - np.min(template_waveform, axis = 0)
    min_lim, max_lim = np.quantile(peak_amplitudes[peak_amplitudes!=0],[.05,.95])#, binwidth=.001)

    channels_to_consider_idx = np.where(peak_amplitudes > min_lim)[0]
    channels_to_consider = sorted_data.channel_map[channels_to_consider_idx]
    
    best_channel = cluster_info.query(f'cluster_id == {cluster_id}').ch.values[0]
    
    #unelegant way of drawing the probe geometry
    y_positions = np.arange(0, 10006, 15)
    all_shanks_x = [27,59,123,155,219,251,315,347]
    x_positions = np.tile(all_shanks_x,len(y_positions)//len(all_shanks_x))
    x_positions = np.concatenate([x_positions,all_shanks_x[1::2]])

    axs.plot(x_positions,y_positions,'.', color = 'grey', alpha = 0.2)
    axs.plot(sorted_data.channel_positions[:,0], sorted_data.channel_positions[:,1], '.', color = 'grey', alpha = 0.1)

    for channel in channels_to_consider:
        channel_pos = sorted_data.channel_positions[channel]
        axs.plot(channel_pos[0],channel_pos[1],'.', color = 'purple', alpha = 0.2)
        if channel == best_channel:
            axs.plot(channel_pos[0],channel_pos[1],'.', color = '#3b2f80')
#%%
def produce_mega_neuron_fig(cluster_id, sorted_data, syncdf, neuronsdf, fig_save_path, bool_click, window = (-8,8), save_fig = True):

    alignments, alignments_dict, key_dict, cmap_FI, cmap_nprots = compute_alignments(syncdf,bool_click)

    fig, axs = plt.subplots(3,len(alignments), figsize=(4*len(alignments),12), tight_layout = True, height_ratios=[1,2,1])
    for jj in range(len(alignments)):
        axs[1, jj].sharex(axs[0, jj])

    spike_times_cluster = sorted_data.spike_times[sorted_data.spike_clusters == cluster_id]/sampling_frequency

    for jj in range(len(alignments)):
        alignment_label = alignments[jj]
        alignment_times = alignments_dict[alignment_label]
        key = key_dict[alignment_label]
        df = syncdf.explode(key)

        spikes_aligned = align_spikes_to_ttl(spike_times_cluster,alignment_times, window=window)
        plot_raster(axs[1,jj], spikes_aligned, (-8,8))


        for cond_ii in range(3):
            if sorted_data.exp == 'c':
                nprots = nprots_list[cond_ii]
                ttls = df.query(f'n_protocols == {nprots}')[key]
                color_palette = color_nprots_blocks

            else:
                FI = FI_list[cond_ii]
                ttls = df.query(f'FI == {FI}')[key]
                color_palette = color_FI_blocks
            
            spikes_aligned = align_spikes_to_ttl(spike_times_cluster,ttls, window=window)
            time, firing_rate = compute_FR(spikes_aligned, window, binW = .25)
            axs[0,jj].plot(time, firing_rate, color = color_palette[cond_ii])
        axs[0,jj].set_ylabel('firing rate (Hz)')
        axs[0,jj].axvline(0, color = 'purple', lw = 1)
        axs[1,jj].set_xlabel(f'time since {alignment_label} (s)')

        ax_FI = inset_axes(axs[1,jj], width="5%", height="100%", loc="center left", borderpad=0)
        ax_nprots = inset_axes(axs[1,jj], width="2.5%", height="100%", loc="center left", borderpad=0)

        ax_FI2 = inset_axes(axs[1,jj], width="5%", height="100%", loc="center right", borderpad=0)
        ax_nprots2 = inset_axes(axs[1,jj], width="2.5%", height="100%", loc="center right", borderpad=0)

        matching_rows = df[df[key].isin(alignment_times)]
        FI_values = matching_rows.FI.values
        nprots_values = matching_rows.n_protocols.values
        
        ax_FI.matshow(FI_values.reshape(len(FI_values),1), aspect = 'auto', cmap = cmap_FI)
        ax_FI.set_xticks([])  
        ax_FI.set_yticks([])
        ax_FI.invert_yaxis()
        ax_FI2.matshow(FI_values.reshape(len(FI_values),1), aspect = 'auto', cmap = cmap_FI)
        ax_FI2.set_xticks([])  
        ax_FI2.set_yticks([])
        ax_FI2.invert_yaxis()

        ax_nprots.matshow(nprots_values.reshape(len(nprots_values),1), aspect = 'auto', cmap = cmap_nprots)
        ax_nprots.set_xticks([])  
        ax_nprots.set_yticks([])
        ax_nprots.invert_yaxis()
        ax_nprots2.matshow(nprots_values.reshape(len(nprots_values),1), aspect = 'auto', cmap = cmap_nprots)
        ax_nprots2.set_xticks([])  
        ax_nprots2.set_yticks([])
        ax_nprots2.invert_yaxis()

        for ax_inset in [ax_FI, ax_nprots, ax_FI2,ax_nprots2]:
            for spine in ax_inset.spines.values():
                spine.set_visible(False)

    # cluster info -- annotation top left
    cluster_line = neuronsdf.query(f'cluster_id == {cluster_id}')
    cluster_info_text = f"""
    {cluster_line['cell_type'].values[0]}
    KSLabel: {cluster_line['KSLabel'].values[0]}
    n_spikes: {cluster_line['n_spikes'].values[0]}
    channel: {cluster_line['ch'].values[0]}
    depth: {cluster_line['depth'].values[0]}
    shank: {cluster_line['sh'].values[0]}
    """
    if bool_click:
        fig.text(0.5, 0.21, cluster_info_text, ha='center', va='center', fontsize=14)
    else:
        fig.text(0.43, 0.21, cluster_info_text, ha='center', va='center', fontsize=14)


    ##### new stuff

    lastrow_gs = axs[2, 0].get_gridspec()
    [fig.delaxes(axs[2, col]) for col in range(3)]

    if bool_click:
        axs[2, 0] = fig.add_subplot(lastrow_gs[2, :3])  # Expand across 3 columns
    else:
        axs[2,0] = fig.add_subplot(lastrow_gs[2,:2]) # if there is no click, then there's one less few column
    cluster_amplitudes = sorted_data.amplitudes[sorted_data.spike_clusters == cluster_id]
    
    axs[2,0].plot(spike_times_cluster/60, cluster_amplitudes,'.', color = 'black', ms = 1)
    axs[2,0].set_xlabel('time in session (min)')
    axs[2,0].set_xlim(0,spike_times_cluster[-1]/60)
    axs[2,0].set_ylabel('amplitude (pseudo volts)')

    ax_FR = axs[2,0].twinx()
    color = 'tab:blue'
    ax_FR.set_ylabel('mean firing rate (Hz)', color = color)
    sns.histplot(ax = ax_FR, x = spike_times_cluster/60, binwidth=1, weights = 1/60, element = 'step', alpha = 0.5)
    ax_FR.tick_params(axis = 'y', labelcolor=color)
    ax_FR.spines['right'].set_visible(True)
    ax_FR.spines['right'].set_color(color)

    ax_FI = inset_axes(axs[2,0], width="100%", height="10%", loc="upper center", borderpad=0)
    ax_nprots = inset_axes(axs[2,0], width="100%", height="5%", loc="upper center", borderpad=0)
    FI_values = syncdf.loc[syncdf.npx_time.dropna().index, 'FI'].values #syncdf.FI.values
    nprots_values = syncdf.loc[syncdf.npx_time.dropna().index, 'n_protocols'].values #syncdf.n_protocols.values
    ax_FI.matshow(FI_values.reshape(1,len(FI_values)), aspect = 'auto', cmap = cmap_FI)
    ax_FI.set_axis_off()
    ax_nprots.matshow(nprots_values.reshape(1,len(nprots_values)), aspect = 'auto', cmap = cmap_nprots)
    ax_nprots.set_axis_off()

    if bool_click:
        fig.delaxes(axs[2,3])
    #else:
    #    fig.delaxes(axs[2,2])

    ax_distamplitudes = inset_axes(axs[2,0], width="10%", height="100%", loc="center right", borderpad=-12)
    sns.histplot(ax = ax_distamplitudes, y = cluster_amplitudes, stat = 'density', element='step', color = 'grey')
    #ax_distamplitudes.set_axis_off()

    ax_probe = inset_axes(axs[2,0], width="5%", height="120%", loc="center right", borderpad=-21)
    plot_probe_sofia(cluster_id, ax_probe, sorted_data, neuronsdf)
    ax_probe.set_axis_off()

    plot_templates(cluster_id, axs[2,-3], sorted_data, neuronsdf)
    axs[2,-3].set_ylabel('template waveform (pseudo volts)')
    axs[2,-3].set_xlabel('time (ms)')
    #axs[2,4].set_axis_off()

    ISI = np.diff(spike_times_cluster)*1000 # in ms
    if 1/np.mean(ISI)*1000 > 10:
        binISI = .1
    else:
        binISI = 1
    
    sns.histplot(ax = axs[2,-1], x = ISI, stat = 'count', element = 'step', binwidth=binISI, color = 'grey')
    axs[2,-1].set_xlim(-2,100)
    axs[2,-1].set_xlabel('time (ms)')
    axs[2,-1].set_title('ISI')

    window_start = -.2
    window_end = .2
    sns.histplot(ax = axs[2,-2], x = neuronsdf.query(f'cluster_id == {cluster_id}').spikes_self_aligned.values[0]*1000, stat = 'frequency', element = 'step', binwidth = 1, color = 'grey')
    axs[2,-2].set_xlim(window_start*1000,window_end*1000)
    axs[2,-2].set_xlabel('time (ms)')
    axs[2,-2].set_ylabel('spikes / ms')
    axs[2,-2].set_title('autocorrelogram')


    figtitle = f'{sorted_data.animal} | {sorted_data.date} | experiment {sorted_data.exp} | cluster_id {cluster_id}'
    plt.suptitle(figtitle)
    if save_fig:
        figtitle = figtitle.replace('|','-')
        plt.savefig(rf'{fig_save_path}/{figtitle}.png', facecolor = 'white')
        plt.close()
#%%
def produce_neuron_bhv_fig(cluster_id, alignments, sorted_data, syncdf, cluster_info, fig_save_path, window = (-8,8), save_fig = True):

    fig, axs = plt.subplots(2,len(alignments), figsize=(4*len(alignments),6), tight_layout = True, sharex = 'col', height_ratios = [1,1])

    spike_times_cluster = sorted_data.spike_times[sorted_data.spike_clusters == cluster_id]/sampling_frequency

    for jj in range(len(alignments)):
        alignment_label = alignments[jj]
        alignment_times = alignments_dict[alignment_label]
        key = key_dict[alignment_label]
        df = syncdf.explode(key)

        spikes_aligned = align_spikes_to_ttl(spike_times_cluster,alignment_times, window=window)
        plot_raster(axs[1,jj], spikes_aligned)


        for cond_ii in range(3):
            if sorted_data.exp == 'c':
                nprots = nprots_list[cond_ii]
                ttls = df.query(f'n_protocols == {nprots}')[key]
                color_palette = color_nprots_blocks

            else:
                FI = FI_list[cond_ii]
                ttls = df.query(f'FI == {FI}')[key]
                color_palette = color_FI_blocks
            
            spikes_aligned = align_spikes_to_ttl(spike_times_cluster,ttls, window=window)
            time, firing_rate = compute_FR(spikes_aligned, window, binW = 0.1)
            axs[0,jj].plot(time, firing_rate, color = color_palette[cond_ii])
        #axs[0,jj].plot(time, firing_rate, color = 'grey')
        axs[0,jj].set_ylabel('firing rate (Hz)')
        axs[0,jj].axvline(0, color = 'purple', lw = 1)
        axs[1,jj].set_xlabel(f'time since {alignment_label} (s)')

        ax_FI = inset_axes(axs[1,jj], width="5%", height="100%", loc="center left", borderpad=0)
        ax_nprots = inset_axes(axs[1,jj], width="2.5%", height="100%", loc="center left", borderpad=0)

        ax_FI2 = inset_axes(axs[1,jj], width="5%", height="100%", loc="center right", borderpad=0)
        ax_nprots2 = inset_axes(axs[1,jj], width="2.5%", height="100%", loc="center right", borderpad=0)

        matching_rows = df[df[key].isin(alignment_times)]
        FI_values = matching_rows.FI.values
        nprots_values = matching_rows.n_protocols.values
        
        ax_FI.matshow(FI_values.reshape(len(FI_values),1), aspect = 'auto', cmap = cmap_FI)
        ax_FI.set_xticks([])  
        ax_FI.set_yticks([])
        ax_FI.invert_yaxis()
        ax_FI2.matshow(FI_values.reshape(len(FI_values),1), aspect = 'auto', cmap = cmap_FI)
        ax_FI2.set_xticks([])  
        ax_FI2.set_yticks([])
        ax_FI2.invert_yaxis()

        ax_nprots.matshow(nprots_values.reshape(len(nprots_values),1), aspect = 'auto', cmap = cmap_nprots)
        ax_nprots.set_xticks([])  
        ax_nprots.set_yticks([])
        ax_nprots.invert_yaxis()
        ax_nprots2.matshow(nprots_values.reshape(len(nprots_values),1), aspect = 'auto', cmap = cmap_nprots)
        ax_nprots2.set_xticks([])  
        ax_nprots2.set_yticks([])
        ax_nprots2.invert_yaxis()

        for ax_inset in [ax_FI, ax_nprots, ax_FI2,ax_nprots2]:
            for spine in ax_inset.spines.values():
                spine.set_visible(False)

    ax_template = fig.add_axes([0.7,.89,.1,.1])
    ax_template.plot(np.mean(sorted_data.templates[sorted_data.spike_templates[sorted_data.spike_clusters == cluster_id][0]], axis = 1), color='purple')
    ax_template.axis('off')

    # cluster info -- annotation top left
    cluster_info_text = f"""
    KSLabel: {cluster_info.loc[cluster_id, 'KSLabel']}     n_spikes: {cluster_info.loc[cluster_id, 'n_spikes']}     amplitude: {cluster_info.loc[cluster_id, 'Amplitude']}     FR: {cluster_info.loc[cluster_id, 'fr']}
    contamPct: {cluster_info.loc[cluster_id, 'ContamPct']}     channel: {cluster_info.loc[cluster_id, 'ch']}     depth: {cluster_info.loc[cluster_id, 'depth']}     shank: {cluster_info.loc[cluster_id, 'sh']}
    """
    fig.text(0.02, 0.95, cluster_info_text, ha='left', va='center', fontsize=14)

    figtitle = f'cluster_id {cluster_id} - multiple alignments'
    plt.suptitle(figtitle)
    if save_fig:
        plt.savefig(rf'{fig_save_path}/{figtitle}.png', facecolor = 'white')
        plt.close()

# %%

"""
.########...######.....###...
.##.....##.##....##...##.##..
.##.....##.##........##...##.
.########..##.......##.....##
.##........##.......#########
.##........##....##.##.....##
.##.........######..##.....##
"""

def do_PCA(concat_for_PCA, quiet = False, bool_return_theta = False):
    pca = PCA()
    PC_space = pca.fit_transform(concat_for_PCA.T)

    explained_variance = pca.explained_variance_ratio_

    if quiet == False:
        plt.figure()
        plt.plot(np.cumsum(explained_variance) * 100, marker='o', linestyle='-')
        plt.xlabel("PC #")
        plt.ylabel("cumulative variance explained (%)")
        plt.title("PCA explained variance")
        #plt.savefig(fr'{fig_save_path}/PCA_variance.png')
        plt.show()

    loadings = pca.components_

    neurons_theta = []
    neurons_rho = []

    if quiet == False:
        fig, axs = plt.subplots(2)
    
    for neuron in range(loadings.shape[1]):
        if quiet == False:
            axs[0].plot(loadings[0,neuron], loadings[1,neuron], '.', color = 'grey')
        complex_neuron = complex(loadings[0,neuron], loadings[1,neuron])
        rho, theta = cmath.polar(complex_neuron)
        neurons_theta.append(theta)
        neurons_rho.append(rho)
    
    if quiet == False:
        axs[0].axvline(0, color = 'black', lw = 1)
        axs[0].axhline(0, color = 'black', lw = 1)

    index_order = np.argsort(neurons_theta)

    if quiet == False:
        axs[1].plot(np.array(neurons_theta)[index_order], '.', color = 'grey')
        plt.show()

        plt.figure()
        plt.matshow(concat_for_PCA[index_order], aspect = 'auto', cmap = 'magma')
        plt.show()

    if bool_return_theta:
        return index_order, loadings, PC_space, neurons_theta
    else:
        return index_order, loadings, PC_space
#%%
def get_PCA_windows(exp, psthbin):
    if exp == 'c':
        #window_I = (0,int(30/psthbin))
        #window_II = (window_I[-1]+1,window_I[-1]+1+int(30/psthbin))
        #window_III = (window_II[-1]+1,window_II[-1]+1+int(30/psthbin))
        window_I = (0,int(30/psthbin))
        window_II = (window_I[-1]+1,window_I[-1]+int(30/psthbin))
        window_III = (window_II[-1]+1,window_II[-1]+int(30/psthbin))

    else: 
        #window_I = (0,int(15/psthbin))
        #window_II = (window_I[-1]+1,window_I[-1]+1+int(30/psthbin))
        #window_III = (window_II[-1]+1,window_II[-1]+1+int(60/psthbin))
        window_I = (0,int(15/psthbin))
        window_II = (window_I[-1]+1,window_I[-1]+int(30/psthbin))
        window_III = (window_II[-1]+1,window_II[-1]+int(60/psthbin))
    return [window_I,window_II,window_III]


# %%
"""
.....#######..########...######......###....##....##.####..######..########
....##.....##.##.....##.##....##....##.##...###...##..##..##....##.##......
....##.....##.##.....##.##.........##...##..####..##..##..##.......##......
....##.....##.########..##...####.##.....##.##.##.##..##...######..######..
....##.....##.##...##...##....##..#########.##..####..##........##.##......
....##.....##.##....##..##....##..##.....##.##...###..##..##....##.##......
.....#######..##.....##..######...##.....##.##....##.####..######..########
"""

def build_X_matrix(df_neurons, df_ttls, bin_size=0.01, window=(0, 2), spike_col='spike_trains', ttl_col='npx_time'):
    """
    Converts spike trains into an aligned 3D matrix (time x neuron x trial).
    
    Args:
        df_neurons (pd.DataFrame): One row per neuron. `spike_col` contains spike times as lists.
        df_ttls (pd.DataFrame): One column with trial start times.
        bin_size (float): Width of each time bin (in same units as spike times).
        window (tuple): Time window around TTL to extract (start, end).
        spike_col (str): Name of the column in df_neurons with spike times.
        ttl_col (str): Name of the column in df_ttls with trial starts.
    
    Returns:
        X (np.ndarray): Array of shape (T, N, K), where T = time bins, N = neurons, K = trials.
        time_vector (np.ndarray): The bin centers, useful for plotting.
    """
    ttls = df_ttls[ttl_col].values
    n_trials = len(ttls)
    n_neurons = len(df_neurons)

    t_start, t_end = window
    time_bins = np.arange(t_start, t_end + bin_size, bin_size)
    T = len(time_bins) - 1  # because histogram uses edges
    time_vector = time_bins[:-1] + bin_size / 2  # bin centers

    X = np.zeros((T, n_neurons, n_trials))

    for neuron_idx, spikes in enumerate(df_neurons[spike_col]):
        spikes = np.asarray(spikes)
        for trial_idx, trial_start in enumerate(ttls):
            aligned_spikes = spikes - trial_start
            trial_spikes = aligned_spikes[(aligned_spikes >= t_start) & (aligned_spikes < t_end)]
            counts, _ = np.histogram(trial_spikes, bins=time_bins)
            X[:, neuron_idx, trial_idx] = counts

    return X, time_vector
# %%

from dataclasses import dataclass

@dataclass
class IBLSorter:
    spike_times  : np.ndarray
    spike_clusters  : np.ndarray
    templates  : np.ndarray
    channel_map : np.ndarray
    channel_positions : np.ndarray
    spike_templates  : np.ndarray
    template_features : np.ndarray
    pc_features  : np.ndarray
    amplitudes : np.ndarray
    sampling_frequency: int
    animal: str
    date: str
    exp: str

def load_ibl_sorter(ibl_sorter_path, animal, date, exp):
    return IBLSorter(
        spike_times = np.load(rf'{ibl_sorter_path}\spike_times.npy'),
        spike_clusters = np.load(rf'{ibl_sorter_path}\spike_clusters.npy'),
        templates = np.load(rf'{ibl_sorter_path}\templates.npy'),
        channel_map = np.load(rf'{ibl_sorter_path}\channel_map.npy'),
        channel_positions = np.load(rf'{ibl_sorter_path}\channel_positions.npy'),
        spike_templates = np.load(rf'{ibl_sorter_path}\spike_templates.npy'),
        template_features = np.load(rf'{ibl_sorter_path}\template_features.npy'),
        pc_features = np.load(rf'{ibl_sorter_path}\pc_features.npy'),
        amplitudes = np.load(rf'{ibl_sorter_path}\amplitudes.npy'),
        sampling_frequency = 30000,
        animal = animal, #these last should be parte of some session data
        date = date,  #these last should be parte of some session data
        exp = exp #these last should be parte of some session data
    )
# %%
