import pykitti
import numpy as np
import cv2
import moviepy.editor as mpy
import imageutils
import argparse
from MV3DFeatures import create_birds_eye_view, create_front_view
from Utils import progress_bar
import drawing
import time
from Tracklet import parse_tracklets, bounding_boxes_for_frame
from VectorMath import *
import Calibration
from renderutils import image_from_map, normalize_and_render_map

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


def transform_bounding_box_bv(bbox, img_size, x_range, y_range):
    w,h = img_size
    x1,x2 = x_range
    y1,y2 = y_range
    factor = w / (x2 - x1)
    t = Transformation()
    t.flip_xy()
    t.mirror_xy()
    t.translate(-x1,0,0)
    t.scale(factor)
    t.translate(0,h,0)
    return t.transform(bbox)


def draw_bounding_box_bv(image, bbox):
    color = (255,0,0)
    drawing.draw_line(image, bbox[0,0:2], bbox[1,0:2], color)
    drawing.draw_line(image, bbox[1,0:2], bbox[2,0:2], color)
    drawing.draw_line(image, bbox[2,0:2], bbox[3,0:2], color)
    drawing.draw_line(image, bbox[3,0:2], bbox[0,0:2], color)


def draw_bounding_box_image(image, bbox):
    color = (0,0,255)
    drawing.draw_line(image, bbox[0,0:2], bbox[1,0:2], color)
    drawing.draw_line(image, bbox[1,0:2], bbox[2,0:2], color)
    drawing.draw_line(image, bbox[2,0:2], bbox[3,0:2], color)
    drawing.draw_line(image, bbox[3,0:2], bbox[0,0:2], color)

    drawing.draw_line(image, bbox[4,0:2], bbox[5,0:2], color)
    drawing.draw_line(image, bbox[5,0:2], bbox[6,0:2], color)
    drawing.draw_line(image, bbox[6,0:2], bbox[7,0:2], color)
    drawing.draw_line(image, bbox[7,0:2], bbox[4,0:2], color)

    drawing.draw_line(image, bbox[0,0:2], bbox[4,0:2], color)
    drawing.draw_line(image, bbox[1,0:2], bbox[5,0:2], color)
    drawing.draw_line(image, bbox[2,0:2], bbox[6,0:2], color)
    drawing.draw_line(image, bbox[3,0:2], bbox[7,0:2], color)



def draw_bounding_boxes_bv(image, tracklets, frame_idx, x_range, y_range):
    img_size = imageutils.img_size(image)
    for bbox in bounding_boxes_for_frame(tracklets, frame_idx):
        bbox = transform_bounding_box_bv(bbox, img_size, x_range, y_range)
        draw_bounding_box_bv(image, bbox)


def project_bbox(bbox):
    projected_bbox = np.zeros((8,2))
    w = 1392
    for i,v in enumerate(bbox):
        projected_bbox[i,0] = bbox[i,0] / bbox[i,2] / bbox[i,3]
        projected_bbox[i,1] = bbox[i,1] / bbox[i,2] / bbox[i,3]

    return projected_bbox


def draw_bounding_boxes_image(image, tracklets, frame_idx, transformation):
    for bbox in bounding_boxes_for_frame(tracklets, frame_idx):
        bbox = transformation.transform(bbox)
        draw_bounding_box_image(image, project_bbox(bbox))



frames = []
tracklets = parse_tracklets("%s/%s/%s_drive_%04d_sync/tracklet_labels.xml" %(args.basedir, args.date, args.date, args.drive))
T_velo_cam = Calibration.read_calibration_matrix_velo_to_cam("%s/%s/calib_velo_to_cam.txt" % (args.basedir, args.date))

cam_to_cam_file = "%s/%s/calib_cam_to_cam.txt" % (args.basedir, args.date)
P_rect = Calibration.read_calibration_matrix(cam_to_cam_file, "P_rect_03", 3, 4)
R_rect = Calibration.read_calibration_matrix(cam_to_cam_file, "R_rect_00", 3, 3)

view_transformation = Transformation()
view_transformation.add_transformation(T_velo_cam)
view_transformation.add_transformation(R_rect)
view_transformation.add_transformation(P_rect)

inv_view_transformation = view_transformation.inverse()

start_time = time.time()
bv_w,bv_h = 512,512
fv_w,fv_h = 512,64
src_x_range = [-25.6, 25.6]
z_min = -1.5
y_min = 3.2
src_y_range = [y_min, y_min + 51.2]
src_z_range = [-1.74, 0.0]

for i,(velo,stereo_pair) in enumerate(zip(data.velo,data.rgb)):
    progress_bar(i, len(data.velo))
    bv_intensity, bv_density, bv_height = create_birds_eye_view(velo, src_x_range, src_y_range, src_z_range, [bv_w,bv_h])
    fv_intensity, fv_distance, fv_height = create_front_view(velo, [fv_w,fv_h], -1.5, 1.0, 0.08, 0.2)
    img = np.array(stereo_pair.right)
    draw_bounding_boxes_image(img, tracklets, i, view_transformation)


    im_height,im_width = img.shape[0:2]
    text_height = 40
    text_offset = 10
    frame_width = max(im_width, 3*bv_w, 3*fv_w)
    frame_height = im_height + bv_h + fv_h + 2 * text_height
    frame = np.zeros((frame_height,frame_width,3))

    im_offset = (frame_width - im_width) // 2
    imageutils.paste_img(frame, imageutils.bgr2rgb(img), [im_offset, 0])

    bv_intensity = image_from_map(bv_intensity)
    draw_bounding_boxes_bv(bv_intensity, tracklets, i, src_x_range, src_y_range)

    bv_density = image_from_map(bv_density)
    draw_bounding_boxes_bv(bv_density, tracklets, i, src_x_range, src_y_range)

    bv_height = image_from_map(bv_height)
    draw_bounding_boxes_bv(bv_height, tracklets, i, src_x_range, src_y_range)


    y = im_height + text_height
    imageutils.paste_img(frame, bv_intensity, [0,y])
    imageutils.paste_img(frame, bv_density, [bv_w,y])
    imageutils.paste_img(frame, bv_height, [2*bv_w,y])

    y += bv_h + text_height
    imageutils.paste_img(frame, image_from_map(fv_intensity), [0,y])
    imageutils.paste_img(frame, normalize_and_render_map(fv_distance), [fv_w,y])
    imageutils.paste_img(frame, normalize_and_render_map(fv_height), [2*fv_w,y])


    tr = drawing.TextRenderer(frame)
    for i,s in enumerate(("Intensity", "Density", "Height")):
        x = bv_w // 2 + i * bv_w
        y = im_height + text_offset
        tr.text_at(s, (x,y), horizontal_align="center")

    for i,s in enumerate(("Intensity", "Distance", "Height")):
        x = fv_w // 2 + i * fv_w
        y = im_height + bv_h + text_height + text_offset
        tr.text_at(s, (x,y), horizontal_align="center")


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
