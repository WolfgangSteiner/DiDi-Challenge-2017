from _MV3DFeatures import _create_birds_eye_view, _create_front_view
import numpy as np

def create_birds_eye_view(velo, src_x_range, src_y_range, src_z_range, dst_size):
    '''
    Create bird's eye feature maps from velodyne lidar point cloud data.

    Arguments:
        velo:
            np.array of point cloud data points, shape (-1,4)

        src_x_range, src_y_range, src_z_range:
            coordinate ranges in KITTI lidar coordinates specifying the points that should be
            included in the feature maps.

        dst_size:
            size of the resulting feature maps in pixels.
    '''

    src_x_range = np.array(src_x_range, np.float32)
    src_y_range = np.array(src_y_range, np.float32)
    src_z_range = np.array(src_z_range, np.float32)
    w,h = dst_size
    feature_map = np.zeros((h,w,3), np.float32)
    feature_map[:,:,2:] = src_z_range[0]

    _create_birds_eye_view(velo, feature_map, src_x_range, src_y_range, src_z_range)

    min_height = src_z_range[0]
    max_height = src_z_range[1]
    feature_map[:,:,2:] = (feature_map[:,:,2:] - min_height) / (max_height - min_height)
    feature_map[:,:,1] = np.log(feature_map[:,:,1] + 1) / np.log(64)

    return feature_map


def create_front_view(velo, dst_size, min_z, max_z, delta_theta=0.08, delta_phi=0.4):
    w,h = dst_size
    feature_map = np.zeros((h,w,3), np.float32)
    feature_map[:,:,2] = min_z
    _create_front_view(velo, feature_map, min_z, max_z, delta_theta, delta_phi)

    return feature_map
