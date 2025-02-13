import pandas as pd
import numpy as np

import cv2
from tqdm import trange

from .utils import plot_image, IMAGE_SIZE


def get_ball_locations(yolo, capture, batch_size = 500):
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_num = 1
    ball_locations = pd.DataFrame()

    frames = []
    pbar = trange(num_frames)
    pbar.set_description("Reading Frames")

    while capture.isOpened():
        pbar.update(1)

        ret, frame = capture.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

        if len(frames) < batch_size:
            continue

        results = yolo(frames).pandas().xyxy
        for i, res_df in enumerate(results):
            if "sports ball" not in res_df["name"].values:
                continue
            
            curr_frame_num = frame_num + i
            res_df["frame_num"] = curr_frame_num
            ball_loc = res_df[res_df["name"] == "sports ball"]
            ball_locations = pd.concat([ball_locations, ball_loc])
    
        frame_num += batch_size
        frames = []

    ball_locations = ball_locations.reset_index(drop = True)
    return ball_locations