import numpy as np
import math

class Transformation(object):
    def __init__(self):
        self.m = matrix_identity()


    def add_transformation(self, t):
        if (type(t) == Transformation):
            self.add_transformation(t.m)
        else:
            self.m = np.matmul(t, self.m)


    def transform(self, coords):
        return np.matmul(coords, self.m.transpose())


    def flip_xy(self):
        self.add_transformation(matrix_flip_xy())


    def mirror_x(self):
        self.add_transformation(matrix_mirror_x())


    def mirror_y(self):
        self.add_transformation(matrix_mirror_y())


    def mirror_xy(self):
        self.add_transformation(matrix_mirror_x())
        self.add_transformation(matrix_mirror_y())


    def translate(self, *args):
        self.add_transformation(matrix_translate(*args))


    def scale(self, s):
        self.add_transformation(matrix_scale(s))


    def rotate_y(self, phi):
        self.add_transformation(matrix_rotation_y(phi))


    def rotate_z(self, phi):
        self.add_transformation(matrix_rotation_z(phi))



def matrix_identity():
    m = np.zeros((4,4), np.float32)
    np.fill_diagonal(m, 1.0)
    return m


def matrix_zero():
    m = np.zeros((4,4), np.float32)
    return m


def matrix_rotation_y(phi):
    m = matrix_identity()
    m[0,0] = math.cos(phi)
    m[0,2] = -math.sin(phi)
    m[2,0] = math.sin(phi)
    m[2,2] = math.cos(phi)
    return m


def matrix_rotation_z(rz):
    m = matrix_identity()
    m[0:2,0] = [math.cos(rz), -math.sin(rz)]
    m[0:2,1] = [math.sin(rz), math.cos(rz)]
    return m


def matrix_translate(*args):
    if len(args) == 1:
        tx, ty, tz = args[0]
    else:
        tx, ty, tz = args

    m = matrix_identity()
    m[0,3] = tx
    m[1,3] = ty
    m[2,3] = tz
    return m


def matrix_flip_xy():
    m = matrix_zero()
    m[0,1] = 1.0
    m[1,0] = 1.0
    m[2,2] = 1.0
    m[3,3] = 1.0
    return m


def matrix_mirror_y():
    m = matrix_identity()
    m[1,1] = -1.0
    return m


def matrix_mirror_x():
    m = matrix_identity()
    m[0,0] = -1.0
    return m


def matrix_scale(s):
    m = matrix_identity()
    m[0,0] = s
    m[1,1] = s
    m[2,2] = s
    return m


def combine_tranformations(*args):
    m = matrix_identity();
    for t in args:
        m = np.matmul(t, m)
    return m


def assert_equal(a, b):
    assert np.all(np.isclose(a,b))


if __name__ == "__main__":
    v = np.ones(4)
    t = Transformation()
    t.add_transformation(matrix_translate(1,2,3))
    assert_equal(t.transform(v), [2,3,4,1])
    t.add_transformation(matrix_scale(0.1))
    assert_equal(t.transform(v), [0.2, 0.3, 0.4, 1.0])
