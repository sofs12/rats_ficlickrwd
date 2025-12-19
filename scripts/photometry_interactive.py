#%%
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # .../rats_ficlickrwd

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Project root on sys.path:", PROJECT_ROOT)

# %%
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import zscore

from ratcode.config.paths import PATH_STORE_PICKLES, DROPBOX_TASK_PATH, PATH_STORE_PHOTOMETRY_PICKLES
from ratcode.common.logging import determine_experiment
from ratcode.common.colorcodes import *
from ratcode.behavior import change_point
from ratcode.common.time import convert_date_bonsai, convert_timestamp
from ratcode.photometry.photometry import butter_filter, quantile_regression, get_prediction, segment_and_fit_function, mask_jumps, find_poly
from ratcode.common.dataframe import group_and_listify

from ratcode.init import setup
setup()

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import HuberRegressor 

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

    return predictions, model

# %%

"""
....###....##....##.####.##.....##....###....##.......########.....###....########.########
...##.##...###...##..##..###...###...##.##...##.......##.....##...##.##......##....##......
..##...##..####..##..##..####.####..##...##..##.......##.....##..##...##.....##....##......
.##.....##.##.##.##..##..##.###.##.##.....##.##.......##.....##.##.....##....##....######..
.#########.##..####..##..##.....##.#########.##.......##.....##.#########....##....##......
.##.....##.##...###..##..##.....##.##.....##.##.......##.....##.##.....##....##....##......
.##.....##.##....##.####.##.....##.##.....##.########.########..##.....##....##....########
"""

animal = 'Krypton'
date = '240823'

PHOTOMETRY_PATH = os.path.join(DROPBOX_TASK_PATH, 'photometry', animal)

PATH_SAVE_FIGS = os.path.join(DROPBOX_TASK_PATH, 'analysis_photometry', f'{animal}_{date}')
if not os.path.exists(PATH_SAVE_FIGS):
    os.makedirs(PATH_SAVE_FIGS)



bhv_pkl = glob.glob(rf"{DROPBOX_TASK_PATH}\analysis\{animal}_{date}_*.pkl")[0]
bhvdf = pd.read_pickle(bhv_pkl)

bhvdf['cp'] = bhvdf.apply(lambda x: change_point.accepted_cp_Gallistel(x.trialno, 2, bhvdf, 'lever_rel', True)[0] if len(x.lever_rel)> 0 else np.nan, axis = 1)
bhvdf['cp'] = bhvdf.apply(lambda x: change_point.validate_cp(x.cp, x.lever_rel) if len(x.lever_rel) > 0 else np.nan, axis = 1)

bhvdf['bool_cp'] = np.isnan(bhvdf.cp.values) == False

bhvdf.drop(bhvdf.query('trial_duration < 200').index, inplace = True)
bhvdf.reset_index(drop = True, inplace = True)
bhvdf['trialno'] = bhvdf.index + 1

for file in os.listdir(PHOTOMETRY_PATH):

    if convert_date_bonsai(date) in file:
        if "in0" in file:
            in0_path = os.path.join(PHOTOMETRY_PATH, file)
        if "in1" in file:
            in1_path = os.path.join(PHOTOMETRY_PATH, file)
        if "in2" in file:
            in2_path = os.path.join(PHOTOMETRY_PATH, file)
        #if "in3" in file: ## encoder
        #    in3_path = os.path.join(photometry_path, file)

in0 = pd.read_csv(in0_path, header = None)
in0.columns = ['in0', 'timestamp0']

in1 = pd.read_csv(in1_path, header = None)
in1.columns = ['in1', 'timestamp1']

in2 = pd.read_csv(in2_path, header = None)
in2.columns = ['in2', 'timestamp2']

## this is the encoder
#in3 = pd.read_csv(in3_path, header = None)
#in3.columns = ['in3', 'timestamp3']

harpdf = pd.concat([in0, in1, in2], axis = 1)
harpdf['timestamp_comp'] = (harpdf.timestamp0 == harpdf.timestamp1)*(harpdf.timestamp0 == harpdf.timestamp2)
harpdf.drop(harpdf.query('timestamp_comp == False').index, inplace=True)
harpdf.rename(columns = {'timestamp1': 'timestamp'}, inplace=True)
harpdf.drop(['timestamp0', 'timestamp_comp', 'timestamp2'], axis = 1, inplace=True)

harpdf['tdtomato'] = harpdf.in0/2**16*20
harpdf['gfp'] = harpdf.in1/2**16*20
harpdf['gpio'] = harpdf.in2/2**16*20

#harpdf['encoder'] = in3.in3/2**16*20

harpdf['ttl_bool'] = harpdf.gpio.apply(lambda x: int(x>1))
harpdf['diff_ttl'] = harpdf.ttl_bool.diff()

harpdf['ttl_rising_edge'] = harpdf.ttl_bool*harpdf.diff_ttl > 0
harpdf['timestamp_session'] = harpdf.timestamp - harpdf.timestamp[0]

harpdf['trialno'] = harpdf.ttl_rising_edge.cumsum() - 1
harpdf.drop(harpdf.query('trialno < 1').index, inplace=True)
#in the behaviour I always drop the last trial
harpdf.drop(harpdf.query(f'trialno == {harpdf.trialno.max()}').index, inplace=True)

print(harpdf.trialno.values[-1])


"""
.########...######.
.##.....##.##....##
.##.....##.##......
.##.....##..######.
.##.....##.......##
.##.....##.##....##
.########...######.

and low pass
"""

downsample_factor = 10
fs = 1000 #sampling frequency
fs = fs/downsample_factor
nyquist = 0.5 * fs

harpdf['ds_tdtomato'] = harpdf['tdtomato'].rolling(2 * downsample_factor, center=True, min_periods=1).mean()
harpdf['ds_gfp'] = harpdf['gfp'].rolling(2 * downsample_factor, center=True, min_periods=1).mean()

downharpdf = harpdf.iloc[::downsample_factor].reset_index(drop = True)



high_cutoff = 20
#lowpass to remove the high freq noise
downharpdf['denoised_tdtomato'] = butter_filter(downharpdf.ds_tdtomato, high_cutoff, fs, 'low') 
downharpdf['denoised_gfp'] = butter_filter(downharpdf.ds_gfp, high_cutoff, fs, 'low')


##USING POLY

jump_threshold_tdtomato = 8
jump_threshold_gfp = 8
downharpdf['poly_tdtomato'] = segment_and_fit_function(downharpdf.timestamp_session.values, mask_jumps(downharpdf.denoised_tdtomato, thres = jump_threshold_tdtomato), function = 'poly')
#downharpdf['poly_tdtomato'] = segment_and_fit_function(downharpdf.timestamp_session.values, mask_jumps(downharpdf.ds_gfp, thres = jump_threshold_gfp), function = 'poly')
downharpdf['poly_gfp'] = segment_and_fit_function(downharpdf.timestamp_session.values, mask_jumps(downharpdf.denoised_gfp, thres = jump_threshold_gfp), function = 'poly')

downharpdf['tdtomato_poly_flat'] = downharpdf.ds_tdtomato - downharpdf.poly_tdtomato
downharpdf['gfp_poly_flat'] = downharpdf.ds_gfp - downharpdf.poly_gfp

downharpdf['clean_poly_tdtomato'] = downharpdf.tdtomato_poly_flat + np.mean(downharpdf.ds_tdtomato)
downharpdf['clean_poly_gfp'] = downharpdf.gfp_poly_flat + np.mean(downharpdf.ds_gfp)

#define the baseline as the 10th percentile
F0_tdtomato = np.nanquantile(downharpdf.clean_poly_tdtomato,.1)
F0_gfp = np.nanquantile(downharpdf.clean_poly_gfp,.1)

downharpdf['deltaF_poly_tdtomato'] = (downharpdf.clean_poly_tdtomato - F0_tdtomato)/F0_tdtomato
downharpdf['deltaF_poly_gfp'] = (downharpdf.clean_poly_gfp - F0_gfp)/F0_gfp

downharpdf['predicted_poly_gfp_session'] = get_prediction(downharpdf.deltaF_poly_tdtomato, downharpdf.deltaF_poly_gfp)[0]
downharpdf['DA_poly_session'] = downharpdf.deltaF_poly_gfp - downharpdf.predicted_poly_gfp_session


downharpdf['deltaF_tdtomato'] = downharpdf['deltaF_poly_tdtomato']
downharpdf['deltaF_gfp'] = downharpdf['deltaF_poly_gfp']



"""
.########..##........######.
.##.....##.##.......##....##
.##.....##.##.......##......
.##.....##.##.......##......
.##.....##.##.......##......
.##.....##.##.......##....##
.########..########..######.
"""
DLC_PATH = rf'{PATH_STORE_PHOTOMETRY_PICKLES}/{animal}_{date}_dlcDAdf.pkl'

if os.path.exists(DLC_PATH):
    bool_dlc  = True
    dlcDAdf = pd.read_pickle(rf'{DLC_PATH}')
else:
    bool_dlc  = False


"""
.########..########..######...########..########..######...######..####..#######..##....##....########.....###....########.....###....##.....##.########.########.########.########...######.
.##.....##.##.......##....##..##.....##.##.......##....##.##....##..##..##.....##.###...##....##.....##...##.##...##.....##...##.##...###...###.##..........##....##.......##.....##.##....##
.##.....##.##.......##........##.....##.##.......##.......##........##..##.....##.####..##....##.....##..##...##..##.....##..##...##..####.####.##..........##....##.......##.....##.##......
.########..######...##...####.########..######....######...######...##..##.....##.##.##.##....########..##.....##.########..##.....##.##.###.##.######......##....######...########...######.
.##...##...##.......##....##..##...##...##.............##.......##..##..##.....##.##..####....##........#########.##...##...#########.##.....##.##..........##....##.......##...##.........##
.##....##..##.......##....##..##....##..##.......##....##.##....##..##..##.....##.##...###....##........##.....##.##....##..##.....##.##.....##.##..........##....##.......##....##..##....##
.##.....##.########..######...##.....##.########..######...######..####..#######..##....##....##........##.....##.##.....##.##.....##.##.....##.########....##....########.##.....##..######.

regression parameters: slope, intercept, r2, pvalue computed in a sliding window
sliding window parameters: 0.5s size, 0.1s step (starting point)

"""

tdtomato = downharpdf.deltaF_tdtomato.values
gfp = downharpdf.deltaF_gfp.values

window_size_s = .5
step_size_s = .1
fs = 100 #Hz after downsampling
window_size = int(window_size_s * fs)
step_size = int(step_size_s * fs)

window_starts = np.arange(0, len(tdtomato) - window_size, step_size)

starts = []
slopes = []
intercepts = []
r_squareds = []
p_values = []

for start in window_starts:
    end = start + window_size
    x_window = tdtomato[start:end]
    y_window = gfp[start:end]
    
    X = sm.add_constant(x_window)
    model = sm.OLS(y_window, X)
    results = model.fit()
    
    slope = results.params[1]
    intercept = results.params[0]
    r_squared = results.rsquared
    p_value = results.pvalues[1]

    starts.append(start)
    slopes.append(slope)
    intercepts.append(intercept)
    r_squareds.append(r_squared)
    p_values.append(p_value)

    #print(f"Window {start}-{end}: Slope={slope}, Intercept={intercept}, R²={r_squared}, p-value={p_value}")




jointdf = group_and_listify(downharpdf, 'trialno', ['timestamp_session', 'denoised_tdtomato', 'denoised_gfp'])

jointdf['trial_start_harp'] = jointdf.timestamp_session.apply(lambda x: x[0])
jointdf['trial_end_harp'] = jointdf.timestamp_session.apply(lambda x: x[-1])
jointdf['trial_duration_harp'] = jointdf.trial_end_harp - jointdf.trial_start_harp

jointdf.drop(jointdf.query('trial_duration_harp < 2').index, inplace = True)
jointdf.reset_index(drop = True, inplace = True)
jointdf['trialno'] = jointdf.index + 1

#jointdf['trialno'] = jointdf.trialno + 11 
plt.figure()
plt.plot(jointdf.trialno, jointdf.trial_duration_harp, label = 'harp')
plt.plot(bhvdf.trialno, bhvdf.trial_duration/1000, '--', label = 'bhv')
plt.title('trial duration (i.e. identity) ok?')
plt.legend()
plt.show()


jointdf['blockno'] = bhvdf.blockno
jointdf['FI'] = bhvdf.FI
jointdf['click']  = bhvdf.click
jointdf['n_protocols'] = bhvdf.n_protocols
jointdf['bool_block'] = bhvdf.bool_block

jointdf['trial_start_arduino'] = bhvdf.trial_start
jointdf['trial_end_arduino'] = bhvdf.trial_end
jointdf['trial_duration_arduino'] = bhvdf.trial_duration

jointdf['lever_rel_arduino'] = bhvdf.lever_rel

jointdf['lever_rel_harp'] = jointdf.apply(lambda x:
                                convert_timestamp(x.lever_rel_arduino,
                                [0, x.trial_duration_arduino],
                                [0, x.trial_duration_harp]), axis = 1)


jointdf['t_trial_harp'] = jointdf.apply(lambda x: np.hstack(x.timestamp_session) - x.trial_start_harp, axis = 1)

jointdf['lever_abs_harp'] = jointdf.apply(lambda x: x.lever_rel_harp + x.trial_start_harp, axis = 1)



# 1. Define time axis
time_session_s = downharpdf.timestamp_session.values
starts_s = time_session_s[0] + np.array(starts) / 100 + window_size_s/2  # Centering the time points

all_lever_presses = np.hstack(jointdf.lever_abs_harp.values)
rwd_lever_presses = np.hstack(jointdf.lever_abs_harp.apply(lambda x: x[-1]))
nonrwd_lever_presses = np.setdiff1d(all_lever_presses, rwd_lever_presses)

shapes = []
for press_time in nonrwd_lever_presses:
    shapes.append(
        dict(
            type="line",
            xref="x",     # Reference the x-axis
            yref="paper", # Reference the "paper" (0 to 1 scale of the whole plot)
            x0=press_time,
            x1=press_time,
            y0=0,         # Bottom of the plot area
            y1=1,         # Top of the plot area
            line=dict(
                color="rgba(56, 64, 66, 0.5)", # dark with transparency
                width=1,
            ),
            layer="below", # Put them behind the data so they don't hide your traces
        )
    )

shapes_rwd = []
for press_time in rwd_lever_presses:
    shapes_rwd.append(
        dict(
            type="line",
            xref="x",     # Reference the x-axis
            yref="paper", # Reference the "paper" (0 to 1 scale of the whole plot)
            x0=press_time,
            x1=press_time,
            y0=0,         # Bottom of the plot area
            y1=1,         # Top of the plot area
            line=dict(
                color="rgba(0, 147, 252, 1)", # blue with transparency
                width=1,
            ),
            layer="below", # Put them behind the data so they don't hide your traces
        )
    )


nrows = 5 + bool_dlc
# 2. Create subplots
fig = make_subplots(
    rows=nrows, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0,
    #subplot_titles=("deltaF/F signals", "DA", "slopes", "intercepts", "R2", "p-values")
)

if bool_dlc:
    # DLC
    for bodypart in ['lever', 'poke', 'implantBase', 'implantSleeve', 'snout', 'topL']:
        fig.add_trace(go.Scatter(x=time_session_s, y=np.hstack(dlcDAdf[bodypart + '_y_upsampled'].values), name = bodypart, line=dict(color=bodypart_color_dic[bodypart])), row=1, col=1)

# Row 1: Raw Signals
fig.add_trace(go.Scatter(x=time_session_s, y=downharpdf.deltaF_tdtomato, name='tdTomato', line=dict(color='red')), row=1+bool_dlc, col=1)
fig.add_trace(go.Scatter(x=time_session_s, y=downharpdf.deltaF_gfp, name='GFP', line=dict(color='green')), row=1+bool_dlc, col=1)

# Row 2: DA Signal
predicted, model = get_prediction(downharpdf.deltaF_tdtomato, downharpdf.deltaF_gfp)
da_signal = downharpdf.deltaF_gfp.values - predicted
fig.add_trace(go.Scatter(x=time_session_s, y=da_signal, name='DA Signal', line=dict(color='blue')), row=2+bool_dlc, col=1)

# Rows 3-6: Using np.array(starts)/100
fig.add_trace(go.Scatter(x=starts_s, y=slopes, name='Slope', line=dict(color='purple')), row=3+bool_dlc, col=1)
fig.add_trace(go.Scatter(x=starts_s, y=intercepts, name='Intercept', line=dict(color='orange')), row=4+bool_dlc, col=1)
fig.add_trace(go.Scatter(x=starts_s, y=r_squareds, name='R2', line=dict(color='teal')), row=5+bool_dlc, col=1)
#fig.add_trace(go.Scatter(x=starts_s, y=p_values, name='p-value', line=dict(color='grey')), row=6, col=1)


## comparison with the huber regression slope and intercept (session wide)
fig.add_hline(y=model.coef_[0], line_dash="dash", line_color="grey", row=3+bool_dlc, col=1, name = 'session slope')
fig.add_hline(y=model.intercept_, line_dash="dash", line_color="grey", row=4+bool_dlc, col=1, name = 'session intercept')


# 3. Update Axes (Spike lines and Range Selector)
# We apply matches='x' so all subplots treat the X-axis as one entity
fig.update_xaxes(
    showspikes=True,
    spikemode='across',
    spikesnap='cursor',
    spikethickness=1,
    spikedash='dash',
    spikecolor="#999999",
    matches='x' 
)

fig.update_xaxes(
    title_text="time (s)",
    row=6, col=1
)

if bool_dlc:
    fig.update_yaxes(title_text="DLC y", autorange = 'reversed', row=1, col=1)

fig.update_yaxes(title_text="deltaF/F signal", row=1+bool_dlc, col=1)
fig.update_yaxes(title_text="DA (robust)", row=2+bool_dlc, col=1)
fig.update_yaxes(title_text="slope", row=3+bool_dlc, col=1)
fig.update_yaxes(title_text="intercept", row=4+bool_dlc, col=1)
fig.update_yaxes(title_text="R2", row=5+bool_dlc, col=1)

#fig.update_traces(
#    xaxis='x', # Explicitly link to the first x-axis
#    showlegend=True
#)

# Force the spikes again globally
fig.update_layout(
    hovermode='x unified',
    spikedistance=-1,
    hoverdistance=-1
)

# 4. Final Layout Adjustments (Hover properties go here!)
fig.update_layout(
    height=900,
    width=1700,
    title_text=f"{animal} {date} | session wide photometry | sliding window {window_size_s}s, step {step_size_s}s",
    hovermode='x unified', # Global property
    hoverdistance=-1,      # Global property
    template="simple_white",
    showlegend=False,
    shapes=shapes + shapes_rwd
)

#fig.update_layout({ax:{"showspikes":True} for ax in fig.to_dict()["layout"] if ax[0:3]=="xax"})


#fig.show()

# Save the figure
#fig.write_html(rf"{PATH_SAVE_FIGS}/{animal}_{date}_interactive_regressions.html", include_plotlyjs='cdn')
fig.write_html(rf"{DROPBOX_TASK_PATH}/analysis_photometry/00_all_sessions_interactive/{animal}_{date}_interactive_regressions.html", include_plotlyjs='cdn')
# %%
