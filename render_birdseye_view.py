import pykitti
import numpy as np
import cv2
import moviepy.editor as mpy
import imageutils
import argparse
from MV3DFeatures import create_birds_eye_view
from Utils import progress_bar
from drawing import TextRenderer
import time

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

def image_from_map(map):
    return imageutils.expand_channel((map * 255).astype(np.uint8))

frames = []

start_time = time.time()
w,h = 256,256
src_x_range = [-25.6, 25.6]
y_min = 3.2
src_y_range = [y_min, y_min + 51.2]

for i,(velo,stereo_pair) in enumerate(zip(data.velo,data.rgb)):
    progress_bar(i, len(data.velo))
    intensity_map, density_map, height_map = create_birds_eye_view(velo, src_x_range, src_y_range, [w,h])
    img = stereo_pair.right
    im_height,im_width = img.shape[0:2]
    frame_width = max(im_width, 3*w)
    frame_height = im_height + h
    im_offset = (frame_width - im_width) // 2
    frame = np.zeros((frame_height,frame_width,3))
    frame[0:im_height,im_offset:im_offset+im_width,:] = imageutils.bgr2rgb(img)
    frame[im_height:frame_height,0:w,:] = image_from_map(intensity_map)
    frame[im_height:frame_height,w:2*w,:] = image_from_map(density_map)
    frame[im_height:frame_height,2*w:3*w,:] = image_from_map(height_map)
    tr = TextRenderer(frame)
    text_offset = 20
    tr.text_at("Intensity", (w//2, im_height+text_offset), horizontal_align="center")
    tr.text_at("Density", (3*w//2, im_height+text_offset), horizontal_align="center")
    tr.text_at("Height", (5*w//2, im_height+text_offset), horizontal_align="center")

    frames.append(frame)

total_time = time.time() - start_time
time_per_frame = total_time / len(frames)
print("Total:     %.4fs" % total_time)
print("Per frame: %.4fs" % time_per_frame)


if len(frames) > 1:
    print("Creating video...")
    clip = mpy.ImageSequenceClip(frames, fps=10)
    clip.write_videofile("test.mp4")
else:
    cv2.imwrite("out.png", frames[0])
