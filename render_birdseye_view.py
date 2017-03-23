import pykitti
import numpy as np
import cv2
import moviepy.editor as mpy
import imageutils
import argparse
from Utils import progress_bar
from drawing import TextRenderer
# The range argument is optional - default is None, which loads the whole dataset

parser = argparse.ArgumentParser()
parser.add_argument('date', help="Date of the drive, format : YYYY_MM_DD.")
parser.add_argument('drive', help="Number of the drive.", type=int)
parser.add_argument('-d', dest="basedir", default="./kitti", help="Name of the data directory.")
parser.add_argument('-1', dest="single_frame", action="store_true")
args = parser.parse_args()

r = range(1) if args.single_frame else None
data = pykitti.raw(args.basedir, args.date, "%04d" % args.drive, r)
data.load_velo()
data.load_rgb(format='cv2')


def coord_for_point(point):
    x = point[0]
    y = point[1]

    if x < 0 or x >= 70 or y < -40 or y >= 40:
        return None
    else:
        return 800 - 1 - int((y + 40) * 10), 700 - 1 - int(x * 10)


def image_from_map(map):
    return imageutils.expand_channel((map * 255).astype(np.uint8))


def create_birds_eye_view(velo):
    w,h = 800,700
    density_map = np.zeros((h,w))
    height_map = np.ones((h,w)) * -1.0e6
    intensity_map = np.zeros((h,w))

    for p in velo:
        c = coord_for_point(p)

        if not c is None:
            x, y = c
            if p[2] > height_map[y,x]:
                height_map[y,x] = p[2]
                intensity_map[y,x] = p[3]
            density_map[y,x] += 1

    density_map = np.log(density_map + 1) / np.log(64)
    height_map = (height_map - height_map.min()) / (height_map.max() - height_map.min())
    return intensity_map, density_map, height_map


frames = []

for i,(velo,stereo_pair) in enumerate(zip(data.velo,data.rgb)):
    progress_bar(i, len(data.velo))
    intensity_map, density_map, height_map = create_birds_eye_view(velo)
    img = stereo_pair.right
    im_height,im_width = img.shape[0:2]
    im_offset = (2400 - im_width) // 2
    frame = np.zeros((700 + im_height,2400,3))
    frame[0:im_height,im_offset:im_offset+im_width,:] = img
    frame[im_height:im_height+700,0:800,:] = image_from_map(intensity_map)
    frame[im_height:im_height+700,800:1600,:] = image_from_map(density_map)
    frame[im_height:im_height+700,1600:2400,:] = image_from_map(height_map)
    tr = TextRenderer(frame)
    text_offset = 20
    tr.text_at("Intensity", (400, im_height+text_offset), horizontal_align="center")
    tr.text_at("Density", (1200, im_height+text_offset), horizontal_align="center")
    tr.text_at("Height", (2000, im_height+text_offset), horizontal_align="center")

    frames.append(frame)


if len(frames) > 1:
    print("Creating video...")
    clip = mpy.ImageSequenceClip(frames, fps=10)
    clip.write_videofile("test.mp4")
else:
    cv2.imwrite("out.png", frames[0])
