import Calibration
import Tracklet
import VectorMath
from MV3DFeatures import create_birds_eye_view
import glob, os
import random
import numpy as np
import cv2
import imageutils
import renderutils
from cv2grid import CV2Grid
import KittiLabel
from Tracklet import Tracklet, bounding_box_for_tracklet
from FeatureVectorEncoderLidarBV import FeatureVectorEncoderLidarBV


def get_stem(path):
    return os.path.splitext(os.path.split(path)[1])[0]


def get_file_stems(category="training"):
    file_names = glob.glob("kitti/%s/velodyne/*.bin" % category)
    return [get_stem(f) for f in file_names]


def split_test_set(file_stems, validation_cut=0.2, seed=None):
    random.seed = seed
    shuffled_stems = list(file_stems)
    random.shuffle(shuffled_stems)
    num_test = int(len(shuffled_stems) * (1.0 - validation_cut))
    return shuffled_stems[0:num_test], shuffled_stems[num_test:]


def read_velodyne_data(stem, category="training"):
    path = "kitti/%s/velodyne/%s.bin" % (category, stem)
    velo = np.fromfile(path, dtype=np.float32)
    velo = velo.reshape((-1, 4))
    return velo


def read_labels(stem, category="training"):
    path = "kitti/%s/label_2/%s.txt" % (category, stem)
    return KittiLabel.read_labels(path)


def read_transforms(stem, category="training"):
    path = "kitti/%s/calib/%s.txt" % (category, stem)
    P2 = Calibration.read_calibration_matrix(path, "P2", 3, 4)
    R0_rect = Calibration.read_calibration_matrix(path, "R0_rect", 3, 3)
    Tr_velo_to_cam = Calibration.read_calibration_matrix(path, "Tr_velo_to_cam", 3, 4)
    t = VectorMath.Transformation()
    t.add_transformation(Tr_velo_to_cam)
    #t.add_transformation(R0_rect)
    #t.add_transformation(P2)
    return t.invert()


def Generator(file_stems, category="training", batchsize=32, draw_ground_truth=False):
    idx = 0
    num_examples = len(file_stems)
    bv_size = [512,512]
    src_x_range = [-25.6,25.6]
    y_min = 0.0
    src_y_range = [y_min, y_min + 51.2]
    src_z_range = [-10.0, 10.0]
    T_velo_to_bv = renderutils.transformation_velo_to_bv(bv_size, src_x_range, src_y_range)

    # src_x_range, src_y_range are defined in bv coordinate system
    # The resulting tracklets will be encoded in the lidar corrdinate system, thus
    # the axes are swapped:
    encoder = FeatureVectorEncoderLidarBV(src_y_range, src_x_range, [32,32])

    while True:
        X = []
        y = []
        images = []
        for i in range(batchsize):
            idx = (idx + 1) % num_examples
            stem = file_stems[idx]
            velo = read_velodyne_data(stem, category)
            labels = read_labels(stem, category)
            T_cam_to_velo = read_transforms(stem, category)

            tracklets = [KittiLabel.tracklet_for_label(l, T_cam_to_velo) for l in labels]

            if draw_ground_truth:
                bv = create_birds_eye_view(velo, src_x_range, src_y_range, src_z_range, bv_size)
                img_bv = np.stack((bv*255),axis=2).astype(np.uint8)

                for t in tracklets:
                    bbox = bounding_box_for_tracklet(t)
                    bbox = T_velo_to_bv.transform(bbox)
                    renderutils.draw_bounding_box_bv(img_bv, bbox)

                images.append(img_bv)

            X.append(create_birds_eye_view(velo, src_x_range, src_y_range, src_z_range, bv_size))
            y.append(encoder.encode_tracklets(tracklets))

        X = np.stack(X, axis=0)
        y = np.stack(y, axis=0)

        if draw_ground_truth:
            yield X, y, images

        else:
            yield X,y


if __name__ == "__main__":
    stems = get_file_stems()
    X_in_test, X_in_val = split_test_set(stems, seed=42)
    gen = Generator(X_in_test, batchsize=6, draw_ground_truth=True)
    num_rows = 2
    num_cols = 3
    grid = np.array([3,2])
    img_size = grid * 512
    g = CV2Grid(img_size, grid)
    X,y,images = gen.next()
    print(X.shape)

    for i in range(X.shape[0]):
        row = i // num_cols
        col = i % num_cols
        img = images[i]
        g.paste_img(imageutils.flip_img_y(img), (row,col))

    g.save("overview.png")
