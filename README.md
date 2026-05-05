# rats_ficlickrwd


practical stuff to organize

Basic commands and organization within the FIClickRwd folder (Dropbox\Data)

### pipeline to produce a behavior daily figure

- run the script produce_block_pickle.py. this automatically looks for new behavioral files and translates them into a dataframe that is stored in a pickle. destination folder is "analysis". No need to give any arguments regarding animal or date

- run the script daily_summary. it is made to be run from terminal with arguments animal date. e.g. 

```
python scripts/daily_summary.py Palladium 260302
```

destination folder is "analysis_bhv\animal"

### pipeline to analyze photometry and produce daily photometry figs

- run the script daily_photometry, from terminal. Arguments are animal date bool_encoder. This later one should always be True, unless for some reason the encoder malfunctions (physically the magnet got out of place and the read is completely flat throughout the session) and in that case we should flag the bool_encoder argument as False. Destination folder is "analysis_photometry\animal_date". 

```
python scripts/daily_photometry animal date True
```


### pipeline to produce sort and analyze the ephys data

- On LAMBDA, go to user spike and run the script ```run_ibl_sort_sofia_drift_amplitude.py```. The command line expects as argument the home recording folder only (no more need to specify the paths to .bin or .meta):

```
python run_ibl_sort_sofia_drift_amplitude.py /media/spike/PortableSSD/Palladium260302_imec0/
```

a new folder will be created on the same directory as the meta and bin files. this is the results of the ibl sorter

- move to personal computer and use phy to inspect the cells. I do this directly on the Portable SSD. The reason for doing it directly in the Portable SSD is twofold: 1) the dropbox path is too long and phy throws errors (it is possible to fix this though -- confer with your favorite LLM); 2) my dropbox is in a hard disk, not SSD, so accessing spike data is very slow

```
H:
cd H:\PortableSSD\recording_folder\..\ibl_sorter_results_drift_amplitude
conda activate phy2
phy template-gui params.py
```

(note that H is the port it connects as in my computer; adjust as needed)

- copy the recording folder to dropbox, to "ephys\animal"

- run the group of numbered scripts of daily_neurons. 

  - 01 reads the sync pulses sent during the experiment and it's what we will use to syncronize the ephys data stream to the remaining peripherals. It also corrects the probe geometry from the ibl sorter (there is a bug there that assumes a linear geometry with just one shank, and this reindexes the channels to account for the 4 shanks geometry)

  ```
  python scripts/daily_neurons_01_extract_sync_correct_geometry.py animal date
  ```

  - 02 needs manual intervention. Run it in interactive mode and adjust the ttls that need deleting so that npx trial duration matches that of bhv. This mismatch stems from the fact that I use a TTL both on block change and on trial start. The job here is usually eliminating the TTLs resulting from block change. This could, in principle, be automatized but it's not that much work and it also comes in handy when the npx doesn't cover the full bhv session (e.g. got disconnected or was connected later). The output of this is a dataframe, syncdf, that has trial start times in both arduino (bhv) and npx times.

  ```
  MANUAL / INTERACTIVE MODE  scripts/daily_neurons_02_manual_sync.py
  ```

  - 03 is again automatic. It gets the syncdf and data from the IBL sorter and computes a bunch of stuff including autocorrelograms and classifies into cell types. It also produces the daily neuron figs. Destination folder is analysis_ephys\animal_date

  ```
  python script/daily_neurons_03_produce_figs.py animal date
  ```

### pipeline to merge dopamine and neural data

- create the simpledf (conversion between DA and npx clocks). Run from terminal:

```
python scripts/daily_DAneurons_01_produce_simpledf.py animal date
```

- produce spike triggered dopamine averages. Run from terminal:
```
python scripts/daily_DAneurons_02_STA.py animal date
```

### how to aggregate data

- aggregate_photometry.py can be ran from terminal with no parameters. It will aggregate all dfs stored in the analysis_photometry folder and automatically save a dataframe with all the information in the dfs folder. This will be named ```aggregate_photometry_YYmmdd.pkl``` (YYmmdd the date in which the aggregated df was produced).
If there are sessions that are trash and should not be aggregated, move the corresponding dfs (_NEWjointdf.pkl and _downharpdf.pkl) to the trash subfolder. One can also select the dates to use a posteriori and filter out the days in the aggregated dataframe