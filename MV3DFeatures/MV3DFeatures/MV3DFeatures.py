from _MV3DFeatures import _create_birds_eye_view, _create_front_view
import numpy as np

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
    feature_map = np.zeros((h,w,3), np.float32)
    feature_map[:,:,2:] = src_z_range[0]

    _create_birds_eye_view(velo, feature_map, src_x_range, src_y_range, src_z_range)

    feature_map_image = np.zeros((h,w,3), np.float32)
    _create_birds_eye_view_from_camera(image, feature_map_image, src_x_range, src_y_range, T_lidar_to_camera)

    min_height = src_z_range[0]
    max_height = src_z_range[1]
    feature_map[:,:,2:] = (feature_map[:,:,2:] - min_height) / (max_height - min_height)
    feature_map[:,:,1] = np.log(feature_map[:,:,1] + 1) / np.log(64)

    result = np.concatenate((feature_map, feature_map_image), axis=2)
    assert(result.shape == (h,w,6))

    return result


def _create_birds_eye_view_from_camera(image, feature_map, src_x_range, src_y_range, T_lidar_to_camera):
    h,w = feature_map.shape[0:2]

    dst_points = [[]]

    p_lidar = np.zeros(4)
    p_lidar[2] = -1.73  # z-value of ground plane
    p_lidar[3] = 1.0

    dst_points = np.array([])


    for y in range(h):
        p_lidar[0] = y / px_per_meter + src_x_range[0]
        for x in range(w):
            p_lidar[1] = (w - x - 1) / px_per_meter + src_y_range[0]
            p_camera = T_lidar_to_camera.transform(p_lidar)
            p_camera /= p_camera[2]  # perspective projection
            pixel_values = _interpolate_pixel(image, p_camera[0], p_camera[1])
            feature_map[y,x,0:3] = pixel_values.astype(np.float32) / 255.0


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
