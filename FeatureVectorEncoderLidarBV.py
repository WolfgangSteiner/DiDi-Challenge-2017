from Tracklet import Tracklet
import numpy as np
import math

class FeatureVectorEncoderLidarBV(object):
    '''
        Class to encode/decode tracklets into feature vectors for the lidar bird's eye
        network. The vector will contain the following fields:
          - confidence : Is there an object at this point in the prior grid? [0.0,1.0]
          - px, py     : Position of the object in KITTI lidar coordinates.
          - l, w, h    : Dimensions of the bounding box of the object.
          - rotation_z : Rotation of the bounding box around the z axis.
    '''


    def __init__(self, src_x_range, src_y_range, prior_grid):
        self.xmin, self.xmax = src_x_range
        self.ymin, self.ymax = src_y_range
        self.nx, self.ny = prior_grid
        self.delta_x = (self.xmax - self.xmin) / self.nx
        self.delta_y = (self.ymax - self.ymin) / self.ny


    def num_priors(self):
        return self.nx, self.ny


    def num_features(self):
        return 7;


    def normalize_angle(self, phi):
        while phi > math.pi:
            phi -= 2 * math.pi

        while phi < -math.pi:
            phi += 2 * math.pi

        return phi



    def encode_tracklets(self, tracklets):
        result = np.zeros((self.ny, self.nx, self.num_features()), np.float32)

        for t in tracklets:
            px, py = t.position[0:2]

            if px < self.xmin or px >= self.xmax or py < self.ymin or py >= self.ymax:
                continue

            l, w, h = t.size
            ix = int((px - self.xmin) / self.delta_x)
            #ix = max(0, min(self.nx - 1, ix))

            iy = int((py - self.ymin) / self.delta_y)
            #iy = max(0, min(self.ny - 1, iy))
            phi = self.normalize_angle(t.rotation_z)

            result[iy,ix] = [1.0, px, py, abs(l), abs(w), abs(h), phi]

        return result.reshape((self.ny*self.nx*self.num_features(),))
