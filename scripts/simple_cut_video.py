#%%
import sys
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # .../rats_ficlickrwd

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("Project root on sys.path:", PROJECT_ROOT)

from ratcode.init import setup
setup()

from ratcode.config.paths import PATH_STORE_PICKLES, DROPBOX_TASK_PATH

#%%
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

from ratcode.dlc.video import read_gpio_into_df, cut_video_by_ttl



#%%

animal = "Niobium"
#date = "250501" 
#camera_view = 'side'
date = "250624"
camera_view = 'top'

#%%

ftoken_bhv = glob.glob(rf"{DROPBOX_TASK_PATH}\behavior\{animal}\{animal}_**_{date}**")[0]
ftoken_pkl = glob.glob(rf"{DROPBOX_TASK_PATH}\analysis\{animal}_{date}**")[0]

camera_prefix = '_top' if camera_view == 'top' else ''
fps = 150 if camera_view == 'side' else 60

ftoken_gpio = glob.glob(rf"{DROPBOX_TASK_PATH}\metadata\{animal}\{animal}_Metadata{camera_prefix}_20{date[:2]}-{date[2:4]}-{date[-2:]}**")[0]
ftoken_video = glob.glob(rf"{DROPBOX_TASK_PATH}\video\{animal}\{animal}**{camera_prefix}_20{date[:2]}-{date[2:4]}-{date[-2:]}**")[0]
#%%
bhvdf = pd.read_pickle(ftoken_pkl)
bhvdf = bhvdf[bhvdf.trial_duration_s >= 2].reset_index(drop=True)
bhvdf['trialno'] = bhvdf.index +1

gpiodf = read_gpio_into_df(ftoken_gpio,camera_view)

trialsdf = gpiodf.groupby('trialno').agg(lambda x: x.tolist()).reset_index()
trialsdf['first_frame'] = trialsdf.frame_session.apply(lambda x: x[1])
trialsdf['len_frames'] = trialsdf.frame_session.apply(lambda x: len(x))
trialsdf['trial_duration_s'] = trialsdf.len_frames/fps
trialsdf.drop(trialsdf.query('trialno == 1').index, inplace=True)
trialsdf.drop(trialsdf.query('trial_duration_s < 2').index, inplace = True)
trialsdf.reset_index(drop = True, inplace = True)
trialsdf['trialno'] = trialsdf.index + 1

#%%
plt.figure()
plt.plot(bhvdf.trialno, bhvdf.trial_duration_s.values, label = 'bhv')
plt.plot(trialsdf.trialno, trialsdf.trial_duration_s.values, '--', label = 'gpio')
plt.title('trials duration match?')
plt.legend()
#%%

ttl_timestamps = trialsdf.first_frame.values
cut_video_by_ttl(ftoken_video,ttl_timestamps, animal, date, camera_view=camera_view)

# %%

"""
....###....##....##.##....##..#######..########....###....########.########....##.....##.####.########..########..#######...######.
...##.##...###...##.###...##.##.....##....##......##.##......##....##..........##.....##..##..##.....##.##.......##.....##.##....##
..##...##..####..##.####..##.##.....##....##.....##...##.....##....##..........##.....##..##..##.....##.##.......##.....##.##......
.##.....##.##.##.##.##.##.##.##.....##....##....##.....##....##....######......##.....##..##..##.....##.######...##.....##..######.
.#########.##..####.##..####.##.....##....##....#########....##....##...........##...##...##..##.....##.##.......##.....##.......##
.##.....##.##...###.##...###.##.....##....##....##.....##....##....##............##.##....##..##.....##.##.......##.....##.##....##
.##.....##.##....##.##....##..#######.....##....##.....##....##....########.......###....####.########..########..#######...######.
"""
import os
import cv2
import numpy as np
from collections import defaultdict

import cv2

class IntervalIndicatorSeconds:
    """
    Keeps a pointer while you iterate frames in order.
    intervals_sec: list of (start_sec, end_sec) in *video seconds*.
    """
    def __init__(self, intervals_sec):
        self.intervals = sorted((min(a,b), max(a,b)) for a,b in intervals_sec)
        self.i = 0

    def is_active_time(self, t_sec: float) -> bool:
        # advance pointer if current interval ended before current time
        while self.i < len(self.intervals) and self.intervals[self.i][1] < t_sec:
            self.i += 1
        if self.i >= len(self.intervals):
            return False
        a, b = self.intervals[self.i]
        return a <= t_sec <= b

def draw_lever_circle(frame, active, center=(60, 60), radius=22, color=(60, 255, 60), thickness=-1):
    """
    Draw a filled circle when 'active' is True; otherwise draw an outline (or skip).
    """
    if active:
        cv2.circle(frame, center, radius, color, thickness, lineType=cv2.LINE_AA)
    else:
        # faint outline when inactive (optional)
        cv2.circle(frame, center, radius, (120,120,120), 2, lineType=cv2.LINE_AA)

class IntervalIndicatorSeconds:
    """Efficiently checks whether current time is within any (start,end) interval."""
    def __init__(self, intervals):
        self.intervals = sorted(intervals)
        self.i = 0
    def is_active(self, t):
        while self.i < len(self.intervals) and self.intervals[self.i][1] < t:
            self.i += 1
        if self.i >= len(self.intervals):
            return False
        a, b = self.intervals[self.i]
        return a <= t <= b

def annotate_video(
    video_path,
    ttl_timestamps,                     # list[int] -> trial start frame indices
    event_tracks=None,                  # dict[str, list[int]] e.g. {"cue": [...], "lick": [...], "reward": [...]}
    static_text=None,                   # dict[str, str] e.g. {"animal": "R123", "date": "2025-09-15", "view":"side"}
    roi_rects=None,                     # list[tuple[int,int,int,int]] -> (x,y,w,h) rectangles to draw
    colors=None,                        # dict[str, tuple(B,G,R)] for event colors
    output_path=None,                   # if None, writes alongside input with _annotated suffix
    font_scale=0.6,
    thickness=1,
    timeline_height=24,                 # px height of bottom timeline
    trial_label_y=30,                   # px from top
):
    """
    Draws per-frame annotations using OpenCV and writes an annotated MP4.
    All event lists are in FRAME INDICES (same clock as ttl_timestamps).
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video {video_path}")

    fps         = cap.get(cv2.CAP_PROP_FPS)
    w           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Prepare defaults
    if event_tracks is None:
        event_tracks = {}
    if static_text is None:
        static_text = {}
    if roi_rects is None:
        roi_rects = []
    if colors is None:
        # default palette
        colors = defaultdict(lambda: (200,200,200), {
            "ttl": (60,180,255),       # orange-ish (BGR)
            "cue": (255,170,30),
            "lick": (220,220,220),
            "poke": (120,255,120),
            "reward": (60,255,120),
            "text": (255,255,255),
            "shadow": (0,0,0),
            "roi": (60,255,255),
        })

    # Output path
    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_annotated.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Convert lists to sets for O(1) membership
    event_sets = {k: set(v) for k, v in event_tracks.items()}
    ttl_set = set(ttl_timestamps)

    # For trial bookkeeping
    ttl_sorted = sorted(ttl_timestamps)
    next_ttl_idx = 0
    current_trial = 0
    current_trial_start = None

    # Helper: draw outlined text for contrast
    def put_text(img, text, org, color_text, scale=font_scale, thick=thickness):
        x, y = org
        # shadow
        cv2.putText(img, text, (x+1, y+1), cv2.FONT_HERSHEY_SIMPLEX, scale, colors["shadow"], thick+2, cv2.LINE_AA)
        # main
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color_text, thick, cv2.LINE_AA)

    # Helper: draw bottom timeline section
    def draw_timeline(img, frame_idx, trial_start_frame, trial_end_frame_guess=None):
        if timeline_height <= 0:
            return

        # timeline rect
        tl_h = timeline_height
        y0 = h - tl_h
        cv2.rectangle(img, (0, y0), (w-1, h-1), (30,30,30), -1)

        # Where are we within the trial?
        if trial_start_frame is not None:
            # define a visible window for the timeline
            if trial_end_frame_guess is None:
                # assume a 10s window if trial end unknown
                window_frames = int(fps * 10)
                trial_end_frame_guess = trial_start_frame + window_frames
            total = max(1, trial_end_frame_guess - trial_start_frame)
            pos = np.clip(int((frame_idx - trial_start_frame) / total * (w-1)), 0, w-1)
            # progress bar
            cv2.rectangle(img, (0, y0), (pos, h-1), (70,70,70), -1)

            # draw tick marks for events inside this trial window
            for name, s in event_sets.items():
                # we only draw ticks that are within [start, end_guess)
                # To keep it simple/fast, sample around frame_idx +/- window
                # But we can compute projected x from frame to window
                # To avoid scanning all s, you can prefilter per-trial if big; here we keep it simple.
                pass

            # draw TTL start marker at x=0 of timeline
            cv2.line(img, (0, y0), (0, h-1), colors["ttl"], 2)

    # To draw per-frame event markers on the actual image (small dots near the bottom-left)
    def draw_event_markers(img, frame_idx, y_base):
        x0 = 12
        gap = 10
        i = 0
        for name, s in event_sets.items():
            if frame_idx in s:
                cv2.circle(img, (x0 + i*gap, y_base), 4, colors.get(name, (255,255,255)), -1, lineType=cv2.LINE_AA)
                i += 1

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect trial boundaries
        if frame_idx in ttl_set:
            current_trial += 1
            current_trial_start = frame_idx

        # Static labels (top-left)
        y = 22
        for k in ("animal", "date", "view"):
            if k in static_text:
                put_text(frame, f"{k}: {static_text[k]}", (10, y), colors["text"])
                y += int(18 * font_scale + 6)

        # Trial label and time since start (top-right)
        if current_trial_start is not None:
            # time since trial start (s)
            dt = (frame_idx - current_trial_start) / fps
            label = f"Trial {current_trial:03d} | t= {dt:5.2f}s"
        else:
            label = "Trial --- | t=  --.--s"

        # right align
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        put_text(frame, label, (w - tw - 10, trial_label_y), colors["text"])

        # Event markers (tiny dots along a baseline near bottom-left)
        draw_event_markers(frame, frame_idx, y_base=h - timeline_height - 10 if timeline_height > 0 else h - 10)

        # Optional ROIs
        for (x, y, rw, rh) in roi_rects:
            cv2.rectangle(frame, (x, y), (x+rw, y+rh), colors["roi"], 2)

        # Timeline bar
        draw_timeline(frame, frame_idx, current_trial_start)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Annotated video written to: {output_path}")
#%%
import cv2
import numpy as np

class IntervalIndicatorSeconds:
    """Efficiently checks whether current time is within any (start,end) interval."""
    def __init__(self, intervals):
        self.intervals = sorted(intervals)
        self.i = 0
    def is_active(self, t):
        while self.i < len(self.intervals) and self.intervals[self.i][1] < t:
            self.i += 1
        if self.i >= len(self.intervals):
            return False
        a, b = self.intervals[self.i]
        return a <= t <= b

def annotate_video(
    video_path,
    ttl_timestamps,
    event_tracks=None,
    static_text=None,
    output_path=None,
    font_scale=0.6,
    thickness=1,
    circle_center=(70, 70),
    circle_radius=24,
    circle_color=(255,255,255),
):
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps  = cap.get(cv2.CAP_PROP_FPS)
    w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Default output path
    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_annotated{ext}"

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    # Handle event_tracks and lever intervals
    event_tracks = event_tracks or {}
    lever_intervals = event_tracks.get("lever_intervals", [])
    lever_indicator = IntervalIndicatorSeconds(lever_intervals)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_time = frame_idx / fps

        # ----- draw lever circle -----
        if lever_indicator.is_active(current_time):
            cv2.circle(frame, circle_center, circle_radius, circle_color, -1, lineType=cv2.LINE_AA)
        else:
            cv2.circle(frame, circle_center, circle_radius, (120,120,120), 2, lineType=cv2.LINE_AA)

        # ----- optional: other event markers -----
        for name, times in event_tracks.items():
            if name == "lever_intervals":
                continue  # skip the interval list
            # draw a dot or text for instantaneous events (within ±1 frame)
            for t in times:
                if abs(t - current_time) < 1/fps:
                    cv2.circle(frame, (20, 20), 4, (255,255,255), -1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Annotated video saved to: {output_path}")

#%%
trialno = 26
lever_down_secs = np.hstack(bhvdf.query(f'trialno == {trialno}').lever_rel.values)/1000
lever_up_secs = np.hstack(bhvdf.query(f'trialno == {trialno}').leverup_rel.values)/1000

lever_intervals = list(zip(lever_down_secs, lever_up_secs))

video_path = rf"D:\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\Data\FIClickRwd\video\Zirconium\Zirconium_250305_side_trials\Zirconium_250305_side_trialno_0{trialno}.mp4"
#video_path = "/path/to/video.mp4"

# Example: these are TRIAL START frame indices
#ttl_timestamps = [0]

event_tracks = {
    "lever_intervals": lever_intervals,  # our new continuous event
}

static_text = {"animal":"Zirconium", "date":"2025-03-05", "view":"side"}

annotate_video(
    video_path,
    ttl_timestamps=[0], #ttl_timestamps,
    event_tracks=event_tracks,
    static_text=static_text,
    #roi_rects=[(50, 60, 120, 90)],       # optional
    output_path=None                      # auto: <video>_annotated.mp4
)


# %%
cap = cv2.VideoCapture(video_path)
duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
cap.release()
print(f"Duration: {duration:.2f} seconds")


# %%
bhvdf.query('trialno == 137').trial_duration
# %%

"""
.##.....##....###....########.########..####.##.....##.....#######..########....##.....##.####.########..########..#######...######.
.###...###...##.##......##....##.....##..##...##...##.....##.....##.##..........##.....##..##..##.....##.##.......##.....##.##....##
.####.####..##...##.....##....##.....##..##....##.##......##.....##.##..........##.....##..##..##.....##.##.......##.....##.##......
.##.###.##.##.....##....##....########...##.....###.......##.....##.######......##.....##..##..##.....##.######...##.....##..######.
.##.....##.#########....##....##...##....##....##.##......##.....##.##...........##...##...##..##.....##.##.......##.....##.......##
.##.....##.##.....##....##....##....##...##...##...##.....##.....##.##............##.##....##..##.....##.##.......##.....##.##....##
.##.....##.##.....##....##....##.....##.####.##.....##.....#######..##.............###....####.########..########..#######...######.
"""

import cv2
import os
import numpy as np
from math import ceil

def video_grid(
    video_paths,
    rows=5,
    cols=4,
    output_path="grid.mp4",
    tile_size=None,            # (width, height); if None, use the first video’s size
    target_fps=None,           # if None, use min FPS across inputs
    layout_order="row",        # "row" or "col" fill order
    duration_mode="pad",       # "pad" (freeze last frame) or "cut" (stop at shortest)
    codec="mp4v",              # e.g., "mp4v" or "avc1"
):
    """
    Create a grid (rows x cols) mosaic from video_paths.
    - video_paths: list of paths (<= rows*cols). If fewer, remaining tiles are black.
    - tile_size: per-tile (W,H); all tiles are resized to this. If None, inferred from first video.
    - target_fps: output fps; default=min input fps.
    - duration_mode: "pad" to extend shorter videos by freezing last frame; "cut" to stop at shortest.
    - layout_order: "row" fills left-to-right, top-to-bottom; "col" fills top-to-bottom, left-to-right.
    """
    assert rows > 0 and cols > 0, "rows/cols must be positive"
    grid_slots = rows * cols

    # Open caps (some slots may be dummy black)
    caps = []
    meta = []
    for p in video_paths[:grid_slots]:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {p}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        meta.append({"path": p, "fps": fps, "size": (w,h), "frames": n})
        caps.append(cap)

    # Fill remaining slots with None (will render black)
    while len(caps) < grid_slots:
        caps.append(None)
        meta.append({"path": None, "fps": np.inf, "size": (0,0), "frames": 0})

    # Decide tile size
    if tile_size is None:
        # choose size of first real video, or fallback
        for m in meta:
            if m["path"] is not None:
                tile_size = m["size"]
                break
        if tile_size is None:
            tile_size = (320, 240)
    tile_w, tile_h = tile_size

    # Decide fps
    if target_fps is None:
        real_fps = [m["fps"] for m in meta if m["path"] is not None and m["fps"] > 0]
        target_fps = min(real_fps) if real_fps else 30.0

    # Determine per-source frame stepping relative to target_fps
    # We'll pull one frame from each cap per output frame; if a source is "slower", we occasionally repeat;
    # if a source is "faster", we occasionally skip.
    steps = []
    for m in meta:
        src_fps = m["fps"] if m["fps"] and np.isfinite(m["fps"]) else target_fps
        steps.append(src_fps / target_fps)  # avg source frames to advance per output frame

    # Duration
    # Compute how many OUTPUT frames to write
    # If "cut": stop at the earliest source end when aligned; if "pad": go until the longest.
    # Convert each source length to output-frame equivalents:
    out_frames_per_src = []
    for m, step in zip(meta, steps):
        if m["path"] is None or m["frames"] == 0 or not np.isfinite(step) or step <= 0:
            out_frames_per_src.append(0)
        else:
            out_frames_per_src.append(int(np.floor(m["frames"] / step)))

    if duration_mode == "cut":
        total_out_frames = min([f for f in out_frames_per_src if f > 0] or [0])
    else:
        total_out_frames = max(out_frames_per_src) if out_frames_per_src else 0

    # Prepare writer
    grid_w, grid_h = cols * tile_w, rows * tile_h
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (grid_w, grid_h))

    # For each source, maintain a floating index into its frames
    float_idx = [0.0] * grid_slots
    ended = [False] * grid_slots
    last_frame_cache = [None] * grid_slots

    def read_resized(cap, desired_size, cache_idx):
        if cap is None:
            # black tile
            return np.zeros((desired_size[1], desired_size[0], 3), dtype=np.uint8)
        ret, fr = cap.read()
        if not ret:
            # no new frame: return last cached or black
            if last_frame_cache[cache_idx] is not None:
                return last_frame_cache[cache_idx]
            else:
                return np.zeros((desired_size[1], desired_size[0], 3), dtype=np.uint8)
        fr = cv2.resize(fr, desired_size, interpolation=cv2.INTER_AREA)
        last_frame_cache[cache_idx] = fr
        return fr

    # Main loop
    for of in range(total_out_frames):
        tiles = []
        # advance sources as needed for this output frame
        for i, cap in enumerate(caps):
            if cap is None:
                # black tile
                tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
                continue

            if ended[i]:
                # stick to last cached frame (pad mode) or black (cut mode)
                if duration_mode == "pad":
                    if last_frame_cache[i] is None:
                        tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
                    else:
                        tiles.append(last_frame_cache[i])
                else:  # cut
                    tiles.append(np.zeros((tile_h, tile_w, 3), dtype=np.uint8))
                continue

            # Determine how many frames to advance since last output frame
            need_advance = 1 if of == 0 else 0
            # accumulate fractional advance
            float_idx[i] += steps[i]
            # Consume floor(steps) frames; handle fractional by accumulating
            advance = int(np.floor(float_idx[i])) - (0 if of == 0 else int(np.floor(float_idx[i-1])) if False else 0)
            # Simpler & stable approach: keep a residual accumulator:
        # Rethink stepping: use a residual per source
        break  # (we'll re-implement stepping cleanly below)

    # --- Clean stepping implementation with residuals ---
def video_grid(
    video_paths,
    rows=5,
    cols=4,
    output_path="grid.mp4",
    tile_size=None,
    target_fps=None,
    layout_order="row",
    duration_mode="pad",
    codec="mp4v",
):
    import cv2, os, numpy as np

    grid_slots = rows * cols

    # Open caps and meta
    caps, meta = [], []
    for p in video_paths[:grid_slots]:
        cap = cv2.VideoCapture(p)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open: {p}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        meta.append({"path": p, "fps": fps, "size": (w,h), "frames": n})
        caps.append(cap)
    while len(caps) < grid_slots:
        caps.append(None)
        meta.append({"path": None, "fps": np.inf, "size": (0,0), "frames": 0})

    # Tile size
    if tile_size is None:
        tile_size = next((m["size"] for m in meta if m["path"]), (320,240))
    tile_w, tile_h = tile_size

    # FPS
    if target_fps is None:
        real_fps = [m["fps"] for m in meta if m["path"] and m["fps"] > 0]
        target_fps = min(real_fps) if real_fps else 30.0

    # Duration in output frames
    def out_frames_for(m):
        if not m["path"] or m["fps"] <= 0 or m["frames"] <= 0:
            return 0
        return int(np.floor(m["frames"] * (target_fps / m["fps"])))
    out_frames_per_src = [out_frames_for(m) for m in meta]
    total_out = (max if duration_mode=="pad" else min)([f for f in out_frames_per_src if f>0] or [0])

    # Writer
    grid_w, grid_h = cols*tile_w, rows*tile_h
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, target_fps, (grid_w, grid_h))

    # Per-source residual stepping
    residual = [0.0]*grid_slots
    ended = [False]*grid_slots
    last_frame = [None]*grid_slots

    def read_frame_i(i):
        cap = caps[i]
        if cap is None:
            return np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

        if ended[i]:
            return last_frame[i] if last_frame[i] is not None else np.zeros((tile_h,tile_w,3), np.uint8)

        # determine how many source frames to consume this output tick
        # advance by src_fps/target_fps frames on average
        step = meta[i]["fps"] / target_fps if np.isfinite(meta[i]["fps"]) and meta[i]["fps"]>0 else 1.0
        residual[i] += step
        to_read = int(residual[i])
        residual[i] -= to_read

        if to_read < 1:
            to_read = 1  # ensure progress at least 1 per tick to avoid stalls

        fr = None
        for _ in range(to_read):
            ret, fr = cap.read()
            if not ret:
                ended[i] = True
                break

        if fr is None:
            # no fresh frame: pad or cut
            if duration_mode == "pad" and last_frame[i] is not None:
                fr = last_frame[i]
            else:
                return np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

        fr = cv2.resize(fr, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        last_frame[i] = fr
        return fr

    # Grid placement order
    def index_for_slot(r, c):
        if layout_order == "col":
            return c*rows + r
        return r*cols + c

    for _ in range(total_out):
        row_imgs = []
        for r in range(rows):
            tiles = []
            for c in range(cols):
                idx = index_for_slot(r, c)
                tiles.append(read_frame_i(idx))
            row_imgs.append(np.hstack(tiles))
        grid_frame = np.vstack(row_imgs)
        out.write(grid_frame)

    # cleanup
    out.release()
    for cap in caps:
        if cap is not None:
            cap.release()

    print(f"[ok] wrote {output_path} ({rows}x{cols} @ {target_fps:.2f} fps)")

#%%
def reorder_vertical(items, rows, cols, fill=None):
    """
    Reorder a flat list that’s in row-major order into column-major order.
    items: flat list, length <= rows*cols
    fill: optional placeholder if items < rows*cols (kept in the tail)
    """
    # pad if needed
    total = rows * cols
    if len(items) < total and fill is not None:
        items = items + [fill] * (total - len(items))
    elif len(items) < total:
        # no fill -> just proceed; missing cells will be skipped
        pass

    # build row-major grid
    grid = [items[r*cols:(r+1)*cols] for r in range(rows)]

    # flatten column-first
    out = []
    for c in range(cols):
        for r in range(rows):
            if r < len(grid) and c < len(grid[r]):
                out.append(grid[r][c])
    return out

#%%
# --------- Example usage ---------
videos = sorted(glob.glob(r"D:\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\Data\FIClickRwd\video\Zirconium\Zirconium_250305_side_trials\trials_for_matrix\*.mp4"))[:20]  # 5x4

n_rows = 4
n_cols = 5

videos_vertical = reorder_vertical(videos, rows=n_cols, cols=n_rows)


video_grid(videos_vertical, rows=n_rows, cols=n_cols, output_path="wall_4x5.mp4", tile_size=(1920//n_cols,1080//n_rows), target_fps=150, duration_mode="pad")
#%%
videos_vertical


#%%
len(videos)

# %%
cap = cv2.VideoCapture(video_path)
w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

print(f"Resolution: {w}x{h} | FPS: {fps:.2f} | Frames: {n}")
# %%

videos = sorted(glob.glob(r"D:\Learning Lab Dropbox\Learning Lab Team Folder\Patlab protocols\Data\FIClickRwd\video\Zirconium\Zirconium_250305_side_trials\3FIs\*.mp4"))  # 5x4
n_rows = 1
n_cols = 3
video_grid(videos, rows=n_rows, cols=n_cols, output_path="wall_3FIs.mp4", tile_size=(1920//5,1080//4), target_fps=150*5, duration_mode="pad")

# %%
videos
# %%
