#%%
## all base paths
import os
from pathlib import Path

DROPBOX_ROOT = r"D:\\Learning Lab Dropbox\\Learning Lab Team Folder\\Patlab protocols"

DROPBOX_DATA_LOCATION = os.path.join(DROPBOX_ROOT,'Data')

# for the pickles / daily figures
DROPBOX_TASK_PATH = os.path.join(DROPBOX_DATA_LOCATION, r'FIClickRwd')
PATH_STORE_PICKLES = os.path.join(DROPBOX_TASK_PATH, r'analysis')
#PICKLE_LOCATION = PATH_STORE_PICKLES

PATH_TO_GET_BHV_FILES = os.path.join(DROPBOX_TASK_PATH, r'behavior')

PATH_STORE_PHOTOMETRY_PICKLES = os.path.join(DROPBOX_TASK_PATH, r'analysis_photometry')
# %%
