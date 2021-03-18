import numpy as np
import av
from sklearn.utils import extmath
import matplotlib.pyplot as plt


def open_video(vid_path):
    """Loads the video matrices

    Args:
        vid_path (str): Path to the video file

    Returns:
        ndarray: X_1^M-1
        ndarray: X_2^M
    """
    v = av.open(vid_path)
    total_frames = v.streams.video[0].frames
    pixels = (v.streams.video[0].format.height // 3) * \
             (v.streams.video[0].format.width // 3)

    X = np.zeros((total_frames, pixels))
    i = 0
    for packet in v.demux():
        for frame in packet.decode():
            img = np.array(frame.to_image(), dtype=np.float64)[2::3, 2::3].mean(2)
            X[i] = img.flat
            i += 1

    return X[:-1].T.copy(), X[1:].T.copy()
