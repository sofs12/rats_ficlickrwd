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

- run the script daily_photometry. currently under development; the ideia is that it can be run from terminal automatically with the same arguments as the daily_summary script. destination folder is "analysis_photometry\animal_date"


### pipeline to produce sort and analyze the ephys data

- ON LAMBDA, on user spike, run the script run_ibl_sort_sofia_drift_CHECK.py script. SPECIFY THE PATHS AND HOW TO ACTIVATE THE ENVIRONMENTS

```
python run_ibl_sort_sofia LA LA LA /media/spike/PortableSSD/Palladium260302_imec0/Palladium260302_imec0_LALALA/Palladium260302_imec0lalalala.meta /media/spike/PortableSSD/Palladium260302_imec0/Palladium260302_imec0_LALALA/Palladium260302_imec0lalalala.bin
```

a new folder will be created on the same directory as the meta and bin files. this is the results of the ibl sorter

- move to personal computer and use phy to inspect the cells. I do this directly on the Portable SSD (the dropbox path is too long and phy throws errors)

```
H:
cd H:\PortableSSD\recording_folder\..\ibl_sorter
conda activate phy2
phy template-gui params.py
```

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
````
python scripts/daily_DAneurons_02_STA.py animal date
```

