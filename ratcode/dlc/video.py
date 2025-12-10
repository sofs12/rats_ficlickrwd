import cv2
import h5py
import numpy as np
import pandas as pd
import os

def read_gpio_into_df(ttl_timestamps_path, camera_view):
    if not os.path.exists(ttl_timestamps_path):
        print(f"Error: File {ttl_timestamps_path} does not exist.")
        return

    if camera_view == 'side':
            gpiodf = pd.read_csv(ttl_timestamps_path, names=['ttl', 'frame', 'bla'], header=0)
    if camera_view == 'top':
            gpiodf = pd.read_csv(ttl_timestamps_path, names=['frame', 'bla', 'ttl'], header=0)
            threshold = gpiodf.ttl.max() * 0.5  
            gpiodf['ttl'] = (gpiodf['ttl'] > threshold).astype(int)

    gpiodf['frame_session'] = gpiodf.frame - gpiodf.frame.values[0]
    gpiodf['bool_ttl'] = (gpiodf['ttl'].shift(1, fill_value=0) == 0) & (gpiodf['ttl'] == 1)
    gpiodf['trialno'] = gpiodf.bool_ttl.cumsum()+1

    return gpiodf 


def cut_video_by_ttl(video_path, ttl_timestamps, animal, date, camera_view = 'side'):
    if not os.path.exists(video_path):
        print(f"Error: File {video_path} does not exist.")
        return

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}.")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output directory
    video_dir = os.path.dirname(video_path)
    output_dir = os.path.join(video_dir, f"{animal}_{date}_{camera_view}_trials")
    os.makedirs(output_dir, exist_ok=True)

    video_title = f'{animal}_{date}_{camera_view}'

    chunk_index = 0
    frame_count = 0
    out = None
    ttl_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_count / fps

        # Check if we need to start a new chunk
        if ttl_index < len(ttl_timestamps) and frame_count >= ttl_timestamps[ttl_index]:
            if out:
                out.release()
            chunk_index += 1
            output_path = os.path.join(output_dir, f"{video_title}_trialno_{chunk_index:03d}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'avc1' for H.264
            #output_path = os.path.join(output_dir, f"{video_title}_trialno_{chunk_index:03d}.avi")
            #fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            print(f"chunk {chunk_index} saved")
            ttl_index += 1

        if out:
            out.write(frame)
        frame_count += 1

    if out:
        out.release()
    cap.release()
    print(f"all done!")


def get_dlc_df(labels_path, thres = 0.7):
    '''
    Returns a dataframe with the labeled coordinates (coords) and a second dataframe nanified when coordinates are below a threshold (default 0.7)
    '''

    coords = pd.read_hdf(labels_path)
    coords.columns = coords.columns.droplevel()
    bodyparts = coords.columns.get_level_values('bodyparts').unique()


    nancoords = coords.copy()

    # Iterate through each body part and apply the threshold
    for bodypart in nancoords.columns.levels[0]:
        likelihood_col = (bodypart, 'likelihood')
        x_col = (bodypart, 'x')
        y_col = (bodypart, 'y')

        # Mask coordinates with NaN where likelihood is below the threshold
        mask = nancoords[likelihood_col] < thres
        nancoords.loc[mask, x_col] = np.nan
        nancoords.loc[mask, y_col] = np.nan

    return coords, nancoords