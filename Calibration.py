import numpy as np
from VectorMath import Transformation


def read_vector(file_name, vector_name):
    """Read vector from KITTI calibration file.
    Arguments:
    file_name:   name of the KITTI calibration file
    vector_name: name of the vector to read

    Returns:
    numpy array containing the vector.
    """

    with open(file_name) as f:
        for l in f.readlines():
            if l.startswith(vector_name):
                return np.array(map(float, l.split(' ')[1:]), np.float32)


def read_matrix(file_name, matrix_name, nrows, ncols):
    vector = read_vector(file_name, matrix_name)
    return vector.reshape((nrows,ncols))


def read_calibration_matrix(file_name, matrix_name, nrows, ncols):
    t = Transformation()
    t.m[0:nrows, 0:ncols] = read_matrix(file_name, matrix_name, nrows, ncols)
    return t


def read_calibration_matrix_velo_to_cam(file_name):
    R = read_matrix(file_name, "R", 3, 3)
    T = read_vector(file_name, "T")
    t = Transformation()
    t.m[0:3, 0:3] = R
    t.m[0:3, 3] = T
    return t


if __name__ == "__main__":
    print(read_vector("kitti/2011_09_26/calib_cam_to_cam.txt", "P_rect_01"))
