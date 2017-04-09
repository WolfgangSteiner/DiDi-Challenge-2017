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
from KittiTracklet import parse_tracklets, tracklets_for_frame
from VectorMath import *
import Calibration
import renderutils
import keras
from lidar_bv_loss import multitask_loss
import cvcolor
from Tracklet import Tracklet, bounding_box_for_tracklet

# The range argument is optional - default is None, which loads the whole dataset

parser = argparse.ArgumentParser()
parser.add_argument('date', help="Date of the drive, format : YYYY_MM_DD.")
parser.add_argument('drive', help="Number of the drive.", type=int)
parser.add_argument('model', help="Keras model to render.")
parser.add_argument('-d', dest="basedir", default="./kitti", help="Name of the data directory.")
parser.add_argument('-1', dest="single_frame", action="store_true")
parser.add_argument('--threshold', dest="threshold", type=float, default=0.95)
parser.add_argument('--fps', dest="fps", type=int, default=10)
args = parser.parse_args()

r = range(1) if args.single_frame else None
data = pykitti.raw(args.basedir, args.date, "%04d" % args.drive, r)
data.load_velo()
data.load_rgb(format='cv2')


frames = []
tracklets_kitti = parse_tracklets("%s/%s/%s_drive_%04d_sync/tracklet_labels.xml" %(args.basedir, args.date, args.date, args.drive))
T_velo_cam = Calibration.read_calibration_matrix_velo_to_cam("%s/%s/calib_velo_to_cam.txt" % (args.basedir, args.date))

cam_to_cam_file = "%s/%s/calib_cam_to_cam.txt" % (args.basedir, args.date)
P_rect = Calibration.read_calibration_matrix(cam_to_cam_file, "P_rect_03", 3, 4)
R_rect = Calibration.read_calibration_matrix(cam_to_cam_file, "R_rect_00", 3, 3)

view_transformation = Transformation()
view_transformation.add_transformation(T_velo_cam)
view_transformation.add_transformation(R_rect)
view_transformation.add_transformation(P_rect)

start_time = time.time()
bv_w,bv_h = 512,512
fv_w,fv_h = 512,64
src_x_range = [-25.6, 25.6]
z_min = -1.5
y_min = 0.0
src_y_range = [y_min, y_min + 51.2]
src_z_range = [-1.74, 0.0]


def transform_bounding_box_bv(bbox, img_size, x_range, y_range):
    t = renderutils.transformation_velo_to_bv(img_size, x_range, y_range)
    return t.transform(bbox)


def draw_bounding_boxes_bv(image, tracklets, frame_idx, x_range, y_range, color):
    img_size = imageutils.img_size(image)
    for bbox in bounding_boxes_for_frame(tracklets, frame_idx):
        bbox = transform_bounding_box_bv(bbox, img_size, x_range, y_range)
        renderutils.draw_bounding_box_bv(image, bbox, color)


def draw_tracklets_bv(image, tracklets, x_range, y_range, color):
    img_size = imageutils.img_size(image)
    for t in tracklets:
        bbox = bounding_box_for_tracklet(t)
        bbox = transform_bounding_box_bv(bbox, img_size, x_range, y_range)
        renderutils.draw_bounding_box_bv(image, bbox, color)



def project_bbox(bbox):
    projected_bbox = np.zeros((bbox.shape[0],2))
    w = 1392
    for i,v in enumerate(bbox):
        x, y, z, w = bbox[i]
        z = max(z, 0.1)
        projected_bbox[i,0] = x / z / w
        projected_bbox[i,1] = y / z / w

    return projected_bbox


def draw_bounding_boxes_image(image, tracklets, frame_idx, transformation, color):
    for bbox in bounding_boxes_for_frame(tracklets, frame_idx):
        bbox = transformation.transform(bbox)
        renderutils.draw_bounding_box_image(image, project_bbox(bbox), color)


def draw_tracklets_image(image, tracklets, transformation, color):
    for t in tracklets:
        bbox = bounding_box_for_tracklet(t)
        assert bbox.shape[0] == 9
        bbox = transformation.transform(bbox)
        renderutils.draw_bounding_box_image(image, project_bbox(bbox), color)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def get_tracklets_from_prediction(y_pred):
    tracklets = []
    delta_x = 51.2 / 32
    delta_y = 51.2 / 32
    xmin_lidar = 0
    ymin_lidar = -51.2 / 2
    for i,yi in enumerate(y_pred.reshape((-1,7))):
        confidence = sigmoid(yi[0])
        if confidence > args.threshold:
            idx_bv = i % 32
            idy_bv = i // 32
            idx_lidar = idy_bv
            idy_lidar = 32 - 1 - idx_bv
            prior_x = idx_lidar * delta_x + xmin_lidar
            prior_y = idy_lidar * delta_y + ymin_lidar
            dx,dy = yi[1:3]
            px = prior_x + dx
            py = prior_y + dy
            l,w,h = yi[3:6]
            rotation_z = yi[6]
            tracklets.append(Tracklet((px,py,-1.73),(l,w,h), rotation_z))

    print("found %d tracklets" % len(tracklets))
    return tracklets


model = keras.models.load_model(args.model, custom_objects={'multitask_loss': multitask_loss})

for i,(velo,stereo_pair) in enumerate(zip(data.velo,data.rgb)):
    progress_bar(i, len(data.velo))
    lidar_bv = create_birds_eye_view(velo, src_x_range, src_y_range, src_z_range, [bv_w,bv_h])
    bv_intensity = lidar_bv[:,:,0]
    bv_density = lidar_bv[:,:,1]
    bv_height = lidar_bv[:,:,2]

    y_pred = model.predict(lidar_bv.reshape((1,512,512,3)))[0]
    tracklets_pred = get_tracklets_from_prediction(y_pred)
    tracklets_true = tracklets_for_frame(tracklets_kitti, i)

    lidar_fv = create_front_view(velo, [fv_w,fv_h], -1.5, 1.0, 0.08, 0.2)
    fv_intensity = lidar_fv[:,:,0]
    fv_distance = lidar_fv[:,:,1]
    fv_height = lidar_fv[:,:,2]
    img = np.array(stereo_pair.right)
    draw_tracklets_image(img, tracklets_true, view_transformation, cvcolor.red)
    draw_tracklets_image(img, tracklets_pred, view_transformation, cvcolor.green)


    im_height,im_width = img.shape[0:2]
    text_height = 40
    text_offset = 10
    frame_width = max(im_width, 3*bv_w, 3*fv_w)
    frame_height = im_height + bv_h + fv_h + 2 * text_height
    frame = imageutils.new_img((frame_width,frame_height))

    im_offset = (frame_width - im_width) // 2
    imageutils.paste_img(frame, np.array(img), [im_offset, 0])

    bv_intensity = renderutils.image_from_map(bv_intensity)
    draw_tracklets_bv(bv_intensity, tracklets_true, src_x_range, src_y_range, cvcolor.red)
    draw_tracklets_bv(bv_intensity, tracklets_pred, src_x_range, src_y_range, cvcolor.green)

    bv_density = renderutils.image_from_map(bv_density)
    draw_tracklets_bv(bv_density, tracklets_true, src_x_range, src_y_range, cvcolor.red)
    draw_tracklets_bv(bv_density, tracklets_pred, src_x_range, src_y_range, cvcolor.green)

    bv_height = renderutils.image_from_map(bv_height)
    draw_tracklets_bv(bv_height, tracklets_true, src_x_range, src_y_range, cvcolor.red)
    draw_tracklets_bv(bv_height, tracklets_pred, src_x_range, src_y_range, cvcolor.green)


    y = im_height + text_height
    imageutils.paste_img(frame, imageutils.flip_img_y(bv_intensity), [0,y])
    imageutils.paste_img(frame, imageutils.flip_img_y(bv_density), [bv_w,y])
    imageutils.paste_img(frame, imageutils.flip_img_y(bv_height), [2*bv_w,y])

    y += bv_h + text_height
    imageutils.paste_img(frame, renderutils.image_from_map(fv_intensity), [0,y])
    imageutils.paste_img(frame, renderutils.normalize_and_render_map(fv_distance), [fv_w,y])
    imageutils.paste_img(frame, renderutils.normalize_and_render_map(fv_height), [2*fv_w,y])


    tr = drawing.TextRenderer(frame)
    for i,s in enumerate(("Intensity", "Density", "Height")):
        x = bv_w // 2 + i * bv_w
        y = im_height + text_offset
        tr.text_at(s, (x,y), horizontal_align="center")

    for i,s in enumerate(("Intensity", "Distance", "Height")):
        x = fv_w // 2 + i * fv_w
        y = im_height + bv_h + text_height + text_offset
        tr.text_at(s, (x,y), horizontal_align="center")


    frames.append(imageutils.bgr2rgb(frame))

total_time = time.time() - start_time
time_per_frame = total_time / len(frames)
print("Total:     %.4fs" % total_time)
print("Per frame: %.4fs" % time_per_frame)


if len(frames) > 1:
    print("Creating video...")
    clip = mpy.ImageSequenceClip(frames, fps=args.fps)
    clip.write_videofile("test.mp4")
else:
    cv2.imwrite("out.png", frames[0])
