from _MV3DFeatures import _create_birds_eye_view, _create_front_view
import numpy as np
import cv2

def create_birds_eye_view(velo, image, src_x_range, src_y_range, src_z_range, dst_size, T_lidar_to_camera):
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
    feature_map_lidar = np.zeros((h,w,3), np.float32)
    feature_map_lidar[:,:,2] = src_z_range[0]
    feature_map_image = np.zeros((h,w,3), np.float32)

    _create_birds_eye_view(velo, feature_map_lidar, src_x_range, src_y_range, src_z_range)
    _create_birds_eye_view_from_camera(image, feature_map_image, src_x_range, src_y_range, T_lidar_to_camera)

    min_height = src_z_range[0]
    max_height = src_z_range[1]
    feature_map_lidar[:,:,2] = (feature_map_lidar[:,:,2] - min_height) / (max_height - min_height)
    feature_map_lidar[:,:,1] = np.log(feature_map_lidar[:,:,1] + 1) / np.log(64)

    return np.concatenate((feature_map_lidar, feature_map_image), axis=2)


def _create_birds_eye_view_from_camera(image, feature_map, src_x_range, src_y_range, T_lidar_to_front_view):
    h,w = feature_map.shape[0:2]
    lidar_x1, lidar_x2 = src_x_range
    lidar_y1, lidar_y2 = src_y_range
    lidar_z = -1.74

    dst_points_lidar = \
        [[lidar_x1, lidar_y2, lidar_z, 1.0],
         [lidar_x1, lidar_y1, lidar_z, 1.0],
         [lidar_x2, lidar_y2, lidar_z, 1.0],
         [lidar_x2, lidar_y1, lidar_z, 1.0]]

    dst_points_bv = np.array([[0,0], [w,0], [0,h], [w,h]], np.float32)
    src_points_camera_3d = T_lidar_to_front_view.transform(dst_points_lidar)

    src_points = []
    for p in src_points_camera_3d:
        src_points.append(p[0:2] / p[2])
    src_points = np.array(src_points, np.float32)

    T = cv2.getPerspectiveTransform(src_points, dst_points_bv)
    bv_img = cv2.warpPerspective(image, T, feature_map.shape[0:2])
    feature_map[:,:,:] = bv_img.astype(np.float32) / 255.0


def bv_to_lidar(p, src_x_range, src_y_range, bv_size):
    h,w = bv_size
    x,y = p
    px_per_meter = w / (src_x_range[1] - src_x_range[0])
    lidar_x = y / px_per_meter
    lidar_y = -(x * px_per_meter + src_y_range[0])
    lidar_z = -1.73
    return np.array([lidar_x,lidar_y,lidar_z])


def _interpolate_pixel(image, x, y):
    h,w = image.shape[0:2]
    if x < 0 or x >= w:
        return np.zeros(3, np.uint)

    if y < 0 or y >= h:
        return np.zeros(3, np.uint)

    return image[int(y), int(x), :]



def create_front_view(velo, dst_size, min_z, max_z, delta_theta=0.08, delta_phi=0.4):
    w,h = dst_size
    feature_map = np.zeros((h,w,3), np.float32)
    feature_map[:,:,2] = min_z
    _create_front_view(velo, feature_map, min_z, max_z, delta_theta, delta_phi)

    return feature_map
