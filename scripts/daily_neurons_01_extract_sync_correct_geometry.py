import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # .../rats_ficlickrwd

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Project root on sys.path:", PROJECT_ROOT)


import os
import glob
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import zscore
from scipy.signal import lfilter

from pathlib import Path
from probeinterface.plotting import plot_probe

import spikeinterface.extractors as se

from ratcode.config.paths import PATH_STORE_PICKLES, DROPBOX_TASK_PATH
from ratcode.common.colorcodes import *
from ratcode.behavior import change_point
from ratcode.ephys.neurons import get_psths_across_cells, align_spikes_to_ttl, compute_FR, load_ibl_sorter, determine_cell_type, produce_neuron_fig, produce_mega_neuron_fig
from ratcode.common.dataframe import group_and_listify
from ratcode.common.time import convert_date_bonsai, convert_timestamp
from ratcode.common.math import drop_nans_matrix
from ratcode.common.colorcodes import FI_order, color_FI_blocks, rwd_order, color_rwd_blocks
from ratcode.init import setup

def main():
    parser = argparse.ArgumentParser(description='Extract TTLs from neuropixel recording and correct geometry after ibl sorter')
    parser.add_argument('animal', type=str, help='Name of the animal (e.g. Ruthenium)')
    parser.add_argument('date', type=str, help='Date of the session format yymmdd (e.g. 260225)')
    args = parser.parse_args()
    animal = args.animal
    date = args.date

    setup()

    EPHYS_PATH = os.path.join(DROPBOX_TASK_PATH, 'ephys', animal)

    PATH_SAVE_FIGS = os.path.join(DROPBOX_TASK_PATH, 'analysis_ephys', fr'{animal}_{date}')
    if not os.path.exists(PATH_SAVE_FIGS):
        os.makedirs(PATH_SAVE_FIGS)

    SAVE_SYNC_PATH = glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\*')[0]
    
    
    IBL_SORTER_PATH =  glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\{animal}{date}*\ibl_sorter_results_drift_amplitude')[0]

    ## CAREFUL HERE!!
    NEURO_PATH =  glob.glob(fr'{EPHYS_PATH}\{animal}{date}*\{animal}{date}*')[0]
    #NEURO_PATH = glob.glob(rf"H:\{animal}{date}*\{animal}{date}*")[0]#[1]
    #NEURO_PATH = glob.glob(rf"F:\EPHYS\{animal}{date}*\{animal}{date}*")[0]#[1]

    raw_rec = se.read_spikeglx(NEURO_PATH, load_sync_channel=False)

    sync_rec = se.read_spikeglx(NEURO_PATH, load_sync_channel=True)
    sync_data = sync_rec.get_traces(channel_ids=[sync_rec.get_channel_ids()[-1]])

    sampling_frequency = int(raw_rec.get_sampling_frequency())
    print(f'Recording time (min): {len(sync_data)/sampling_frequency/60}')
    print()

    ## run this only once to update the channel positions if they are in the old format (x = 1 or 2 for shank identity instead of actual x position)
    if os.path.exists(Path(fr'{IBL_SORTER_PATH}\channel_positions_original.npy')):
        print('channel_positions.npy had already been updated')

    else:
        print('channel_positions.npy updated')
        file_position = Path(fr'{IBL_SORTER_PATH}\channel_positions.npy')
        xy = np.load(file_position)
        shank = np.load(file_position.with_name('channel_shanks.npy'))
        if len(np.unique(xy[:, 0])) == 2:
            np.save(file_position.with_name('channel_positions_original.npy'), xy)
            xy_new = xy.copy()
            xy_new[:, 0] = xy_new[:, 0] + shank.astype(np.float32) * 32 * 3
            np.save(file_position, xy_new)

    probe = raw_rec.get_probe()

    plt.figure(figsize=(4,6))
    plot_probe(probe)
    plt.savefig(fr'{PATH_SAVE_FIGS}\probe_geometry.png')
    plt.close()


    ## detect TTLs rising edge -- this step takes some time

    print(f'Started extracting TTLs for {animal} {date}...')


    # Parameters
    chunk_duration_minutes = 2  # Duration of each chunk in minutes (adjust based on memory)
    chunk_size = chunk_duration_minutes * 60 * sampling_frequency  # Number of samples per chunk

    # Load the sync channel from the recording
    ttl_channel = sync_rec.get_channel_ids()[-1]  # Assuming the TTL channel is the last one
    num_samples = sync_rec.get_num_frames()

    # To store rising edges
    rising_edges = []
    above_thres = []

    # Process in chunks
    for start in range(0, num_samples, chunk_size):
        end = min(start + chunk_size, num_samples)
        sync_data = sync_rec.get_traces(start_frame=start, end_frame=end, channel_ids=[ttl_channel]).ravel()

        #plt.figure()
        #plt.plot(sync_data)
        #plt.title(start//chunk_size+1)
        #plt.show()

        # Convert to binary TTL (0 or 1) and detect rising edges
        binary_ttl = (sync_data > 20).astype(np.uint8)
        chunk_rising_edges = np.where(np.diff(binary_ttl) == 1)[0] + start
        rising_edges.extend(chunk_rising_edges)

        # new - discard with voltages > 100
        #above_thres.extend(np.where(sync_data > 100)[0])

        print(f"Processed chunk {start // chunk_size + 1} / {num_samples // chunk_size}")


    # Convert rising_edges list to a numpy array
    rising_edges = np.array(rising_edges)
    rising_edges = rising_edges/sampling_frequency


    # Save or use the rising edges as needed (for example, saving to file)
    np.save(fr'{SAVE_SYNC_PATH}/rising_edges.npy', rising_edges)
    print()
    print(f"Total rising edges detected: {len(rising_edges)}")
    print(fr'syncdf saved to {SAVE_SYNC_PATH}/rising_edges.npy')
    

if __name__ == "__main__":
    main()