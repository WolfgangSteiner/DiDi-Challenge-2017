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


def get_file_stems(category="training"):
    file_names = glob.glob("kitti/%s/velodyne/*.bin" % category)
    return [os.path.splitext(f)[0] for f in file_names]


def split_test_set(file_stems, validation_cut=0.2, seed=None):
    random.seed = seed
    shuffled_stems = list(file_stems)
    random.shuffle(shuffled_stems)
    num_test = int(len(shuffled_stems) * (1.0 - validation_cut))
    return shuffled_stems[0:num_test], shuffled_stems[num_test:]


def read_velodyne_data(stem):
    velo = np.fromfile(stem + ".bin", dtype=np.float32)
    velo = velo.reshape((-1, 4))
    return velo


def Generator(file_stems, batchsize=32):
    idx = 0
    num_examples = len(file_stems)
    bv_size = [512,512]
    src_x_range = [-25.6,25.6]
    y_min = 0.0
    src_y_range = [y_min, y_min + 51.2]
    src_z_range = [-10.0, 10.0]
    labels_path =

    while True:
        X = []
        for i in range(batchsize):
            idx = (idx + 1) % num_examples
            stem = file_stems[idx]
            velo = read_velodyne_data(stem)
            X.append(create_birds_eye_view(velo, src_x_range, src_y_range, src_z_range, bv_size))

        X = np.stack(X, axis=0)
        yield X


if __name__ == "__main__":
    stems = get_file_stems()
    X_in_test, X_in_val = split_test_set(stems, seed=42)
    gen = Generator(X_in_test, batchsize=6)
    num_rows = 2
    num_cols = 3
    grid = np.array([3,2])
    img_size = grid * 512
    g = CV2Grid(img_size, grid)
    X = gen.next()
    print(X.shape)

    for i in range(X.shape[0]):
        row = i // num_cols
        col = i % num_cols
        img = renderutils.image_from_map(X[i,1])
        g.paste_img(img, (row,col))

    g.save("overview.png")
