## all base paths
import os

os.chdir(r'C:\Users\Admin\Documents\git\ratanalysis')

dropbox_data_location = r'D:\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\Data'

# for the pickles / daily figures
dropbox_path = os.path.join(dropbox_data_location, r'FIClickRwd')
path_store_pickles = os.path.join(dropbox_path, r'analysis')
pickle_location = path_store_pickles

path_to_get_bhv_files = os.path.join(dropbox_path, r'behavior')
