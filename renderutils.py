import imageutils
import numpy as np

def image_from_map(map):
    return imageutils.expand_channel((map * 255).astype(np.uint8))


def normalize_and_render_map(map):
    min_value = map.min()
    max_value = map.max()

    normalized_map = (map - min_value) / (max_value - min_value)
    return image_from_map(normalized_map)
