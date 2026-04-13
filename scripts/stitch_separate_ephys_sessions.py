#%%
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # .../rats_ficlickrwd

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Project root on sys.path:", PROJECT_ROOT)

import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import zscore
from scipy.signal import lfilter
from scipy.signal import savgol_filter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
#import argparse
import re 
import pickle
import time
from tqdm import tqdm

from pathlib import Path
from probeinterface.plotting import plot_probe


import spikeinterface.extractors as se


from ratcode.config.paths import PATH_STORE_PICKLES, DROPBOX_TASK_PATH, PATH_DATAFRAMES
from ratcode.common.logging import determine_experiment
from ratcode.common.colorcodes import *
from ratcode.behavior import change_point
from ratcode.photometry.photometry import get_prediction, quantile_regression, signal2eventsnippets, find_poly, segment_and_fit_function, butter_filter, mask_jumps, make_continuous, query_and_compute_snippets, plot_snippets
from ratcode.ephys.neurons import get_psths_across_cells, align_spikes_to_ttl, compute_FR, load_ibl_sorter, determine_cell_type, produce_neuron_fig, produce_mega_neuron_fig, compute_zscore, get_psths_smooth, half_gaussian_kernel, do_PCA, get_PCA_windows, plot_raster, extract_features_cell_type, classify_cell_type_with_features
from ratcode.common.dataframe import group_and_listify
from ratcode.common.time import convert_date_bonsai, convert_timestamp
from ratcode.common.math import drop_nans_matrix
from ratcode.common.colorcodes import FI_order, color_FI_blocks, rwd_order, color_rwd_blocks
from ratcode.common.plotting import remove_legend

from ratcode.init import setup


setup()
# %%
import spikeinterface.full as si
from pathlib import Path
#%%
# 1. Define your paths
session1_path = Path('path/to/session1/folder')
session2_path = Path('path/to/session2/folder')

# 2. Load the recordings
# Using read_spikeglx, but works for read_openephys as well
rec1 = si.read_spikeglx(session1_path, stream_id='imec0.ap')
rec2 = si.read_spikeglx(session2_path, stream_id='imec0.ap')

# 3. Concatenate as Segments
# This creates a single recording object with segment_index 0 and 1
multisegment_rec = si.concatenate_recordings([rec1, rec2])

print(f"Total segments: {multisegment_rec.get_num_segments()}")
print(f"Total duration: {multisegment_rec.get_total_duration():.2f} seconds")

# 4. Optional: Add Probe Information
# If your probe isn't automatically loaded from metadata
# probe = si.read_probeinterface('your_probe_file.json')
# multisegment_rec = multisegment_rec.set_probe(probe)

# 5. Pre-processing (Common for Neuropixels)
# It is highly recommended to bandpass and common-reference before sorting
rec_processed = si.bandpass_filter(multisegment_rec, freq_min=300, freq_max=6000)
rec_processed = si.common_reference(rec_processed, reference='global', operator='median')

# 6. Run the Sorter (Kilosort 4 is recommended for NPX)
# This will sort both segments together, maintaining unit identity
sorting = si.run_sorter(
    sorter_name='kilosort4', 
    recording=rec_processed, 
    output_folder='sorter_output_stitched',
    verbose=True
)

# 7. Save the result
sorting.save(folder='my_final_sorting')
# %%
