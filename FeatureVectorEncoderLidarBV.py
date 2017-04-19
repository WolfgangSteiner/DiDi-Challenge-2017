from Tracklet import Tracklet
import numpy as np
import math

class FeatureVectorEncoderLidarBV(object):
    '''
        Class to encode/decode tracklets into feature vectors for the lidar bird's eye
        network. The vector will contain the following fields:
          - confidence         : Is there an object at this point in the prior grid? [0.0,1.0]
          - lidar_px, lidar_py : Position of the object in KITTI lidar coordinates.
          - l, w, h            : Dimensions of the bounding box of the object.
          - rotation_z         : Rotation of the bounding box around the z axis.
    '''


    def __init__(self, lidar_x_range, lidar_y_range, prior_grid):
        self.lidar_xmin, self.lidar_xmax = lidar_x_range
        self.lidar_ymin, self.lidar_ymax = lidar_y_range
        self.nx, self.ny = prior_grid
        self.delta_x = (self.lidar_xmax - self.lidar_xmin) / self.nx
        self.delta_y = (self.lidar_ymax - self.lidar_ymin) / self.ny


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
            lidar_px, lidar_py = t.position[0:2]

            if lidar_px < self.lidar_xmin or lidar_px >= self.lidar_xmax or lidar_py < self.lidar_ymin or lidar_py >= self.lidar_ymax:
                continue

            l, w, h = t.size
            lidar_ix = int((lidar_px - self.lidar_xmin) / self.delta_x)
            lidar_prior_x = lidar_ix * self.delta_x + self.lidar_xmin
            dx = lidar_px - lidar_prior_x
            #ix = max(0, min(self.nx - 1, ix))

            lidar_iy = int((lidar_py - self.lidar_ymin) / self.delta_y)
            lidar_prior_y = lidar_iy * self.delta_y + self.lidar_ymin
            dy = lidar_py - lidar_prior_y
            #iy = max(0, min(self.ny - 1, iy))

            phi = self.normalize_angle(t.rotation_z)

            # The bird's eye view feature map has a different coordinate system than the
            # lidar coordinate system.
            ix_bv = 32 - 1 - lidar_iy
            iy_bv = lidar_ix


            result[iy_bv,ix_bv] = [1.0, dx, dy, abs(l), abs(w), abs(h), phi]

        return result.reshape((self.ny*self.nx*self.num_features(),))
