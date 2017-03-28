from _MV3DFeatures import _create_birds_eye_view
import numpy as np

def create_birds_eye_view(velo):
    w,h = 800,700
    density_map = np.zeros((h,w), np.float32)
    height_map = np.ones((h,w), np.float32) * -1.0e6
    intensity_map = np.zeros((h,w), np.float32)

    _create_birds_eye_view(velo, density_map, height_map, intensity_map)

    height_map = (height_map - height_map.min()) / (height_map.max() - height_map.min())
    density_map = np.log(density_map + 1) / np.log(64)
    return intensity_map, density_map, height_map
