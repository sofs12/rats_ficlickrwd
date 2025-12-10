import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # .../rats_ficlickrwd

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Project root on sys.path:", PROJECT_ROOT)

import os 
from ratcode.common import logging, dataframe
from ratcode.config.paths import PATH_TO_GET_BHV_FILES, PATH_STORE_PICKLES

eventcode_ref, eventcode_ref_reverse = logging.getEventCodes(rf"{PROJECT_ROOT}\ratcode\config\FIClickRwd_EventCodes.h")

animals_folders = os.listdir(PATH_TO_GET_BHV_FILES)
animals_folders.remove('Zero')

path_to_animals_bhv_files = []
path2 = []

all_animal_files = []

already_pickled = os.listdir(PATH_STORE_PICKLES)
already_pickled_combined = '\t'.join(already_pickled)

for animal in animals_folders:
    path_to_animal_folder = os.path.join(PATH_TO_GET_BHV_FILES, animal)

    animal_files = os.listdir(path_to_animal_folder)

    for file in animal_files:

        if('trash' not in file):

            split_file = file.split('_')
            if (split_file[0] + '_' + split_file[2] not in already_pickled_combined):
                # create a pickle only for new files
                path_to_file = os.path.join(path_to_animal_folder, file)
                all_animal_files.append(path_to_file)

for file in all_animal_files:
    print(file)

    df_BLOCKS = dataframe.populateDataFrame_BLOCKS(file, eventcode_ref)
    newdf_BLOCKS = dataframe.df_to_new_df_BLOCKS(df_BLOCKS)
    newdf_BLOCKS['bool_block'] = True
    figtitle = df_BLOCKS.animal[0] + '_' + df_BLOCKS.date[0] + '_BLOCKS'
    newdf_BLOCKS.to_pickle(PATH_STORE_PICKLES + '\\' + figtitle + '.pkl', compression= None)
