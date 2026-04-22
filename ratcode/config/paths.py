## all base paths
import os
from pathlib import Path
import json

## get dropbox path, computer agnostic
def get_dropbox_root():
    # 1. Find the Dropbox metadata file
    json_path = os.path.expandvars(r'%APPDATA%\Dropbox\info.json')
    if not os.path.exists(json_path):
        json_path = os.path.expandvars(r'%LOCALAPPDATA%\Dropbox\info.json')

    try:
        with open(json_path, 'r') as f:
            info = json.load(f)

            full_path = info.get('business', info.get('personal'))['path']
            
            dropbox_parent = os.path.dirname(full_path)
            
            return os.path.join(dropbox_parent, "Learning Lab Team Folder", "Patlab protocols")
            
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None

DROPBOX_ROOT = get_dropbox_root()

if DROPBOX_ROOT:
    print(f"Dropbox path set to: {DROPBOX_ROOT}")
else:
    print("Could not find Dropbox. Check if the app is installed.")

## get this automatically so that it is computer agnostic!
#DROPBOX_ROOT = r"D:\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols"
#DROPBOX_ROOT = r"C:\Users\admin\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols"

DROPBOX_DATA_LOCATION = os.path.join(DROPBOX_ROOT,'Data')

# for the pickles / daily figures
DROPBOX_TASK_PATH = os.path.join(DROPBOX_DATA_LOCATION, 'FIClickRwd')
PATH_STORE_PICKLES = os.path.join(DROPBOX_TASK_PATH, 'analysis')
#PICKLE_LOCATION = PATH_STORE_PICKLES

PATH_TO_GET_BHV_FILES = os.path.join(DROPBOX_TASK_PATH, 'behavior')

PATH_STORE_PHOTOMETRY_PICKLES = os.path.join(DROPBOX_TASK_PATH, 'analysis_photometry')

PATH_BHV_ANALYSIS = os.path.join(DROPBOX_TASK_PATH, 'analysis_bhv')
PATH_EPHYS_ANALYSIS = os.path.join(DROPBOX_TASK_PATH, 'analysis_ephys')
PATH_DANEURONS_ANALYSIS = os.path.join(DROPBOX_TASK_PATH, 'analysis_DAneurons')

PATH_DATAFRAMES = os.path.join(DROPBOX_TASK_PATH,'dfs')

