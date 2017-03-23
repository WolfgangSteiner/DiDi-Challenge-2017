import pykitti
import numpy as np
import cv2
import moviepy.editor as mpy
import imageutils
from Utils import progress_bar
basedir = './kitti'
date = '2011_09_26'
drive = '0001'

# The range argument is optional - default is None, which loads the whole dataset
data = pykitti.raw(basedir, date, drive)
data.load_velo()

def create_birds_eye_view(velo):
    w,h = 800,700
    output = np.zeros((h,w))
    y = 0
    iy = h - 1
    delta = 0.1

    sorted_idx = velo[:,0].argsort()
    d = velo[sorted_idx]

    current_idx = 0;

    while y < 70:
        row = [];
        while d[current_idx, 0] < y + delta:
            row.append(d[current_idx,:]);
            current_idx += 1

        if len(row):
            row = np.stack(row, axis=0)
            sorted_idx = row[:,1].argsort()
            row = row[sorted_idx]

            x = -40.0
            ix = w - 1
            row_idx = 0;
            while x < 40.0:
                window = [];
                while row_idx < len(row) and row[row_idx,1] < x + delta:
                    window.append(row[row_idx])
                    row_idx +=1

                if len(window):
                    window = np.stack(window, axis=0)
                    output[iy,ix] = window[:,3].max()

                x += delta
                ix -= 1

        y += delta
        iy -= 1

    output = (output * 255).astype(np.uint8)
    output = imageutils.expand_channel(output)
    return output

frames = []

for i,d in enumerate(data.velo):
    progress_bar(i, len(data.velo))
    frame = create_birds_eye_view(d)
    frames.append(frame)

if len(frames) > 1:
    print("Creating video...")
    clip = mpy.ImageSequenceClip(frames, fps=10)
    clip.write_videofile("test.mp4")
