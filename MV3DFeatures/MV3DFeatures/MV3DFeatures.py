from _MV3DFeatures import _create_birds_eye_view
import numpy as np

def create_birds_eye_view(velo, src_x_range, src_y_range, dst_size):
    src_x_range = np.array(src_x_range, np.float32)
    src_y_range = np.array(src_y_range, np.float32)
    w,h = dst_size
    feature_map = np.zeros((3,h,w), np.float32)
    feature_map[2:,:,:] = -1.0e6

    _create_birds_eye_view(velo, feature_map, src_x_range, src_y_range)

    min_height = feature_map[2,:,:].min()
    max_height = feature_map[2,:,:].max()
    feature_map[2,:,:] = (feature_map[2,:,:] - min_height) / (max_height - min_height)
    feature_map[1,:,:] = np.log(feature_map[1,:,:] + 1) / np.log(64)

    return feature_map[0,:,:], feature_map[1,:,:], feature_map[2,:,:]
