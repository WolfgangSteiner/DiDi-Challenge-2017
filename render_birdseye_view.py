import pykitti
import numpy as np
import cv2
import moviepy.editor as mpy
import imageutils
import argparse
from Utils import progress_bar
# The range argument is optional - default is None, which loads the whole dataset

parser = argparse.ArgumentParser()
parser.add_argument('date', help="Date of the drive, format : YYYY_MM_DD.")
parser.add_argument('drive', help="Number of the drive.", type=int)
parser.add_argument('-d', dest="basedir", default="./kitti", help="Name of the data directory.")
args = parser.parse_args()

data = pykitti.raw(args.basedir, args.date, "%04d" % args.drive)
data.load_velo()


def coord_for_point(point):
    x = point[0]
    y = point[1]

    if x < 0 or x >= 70 or y < -40 or y >= 40:
        return None
    else:
        return 800 - 1 - int((y + 40) * 10), 700 - 1 - int(x * 10)


def create_birds_eye_view(velo):
    w,h = 800,700
    output = np.zeros((h,w))

    for p in velo:
        c = coord_for_point(p)

        if not c is None:
            x, y = c
            output[y,x] = max(output[y,x], p[3])

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
else:
    cv2.imwrite("out.png", frames[0])
