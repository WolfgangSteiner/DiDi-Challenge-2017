import imageutils
import numpy as np
import drawing
from VectorMath import Transformation

def image_from_map(map):
    return imageutils.expand_channel((map * 255).astype(np.uint8))


def normalize_and_render_map(map):
    min_value = map.min()
    max_value = map.max()

    normalized_map = (map - min_value) / (max_value - min_value)
    return image_from_map(normalized_map)


def transformation_velo_to_bv(img_size, x_range, y_range):
    w,h = img_size
    x1,x2 = x_range
    y1,y2 = y_range
    factor = w / (x2 - x1)
    t = Transformation()
    t.flip_xy()
    t.mirror_x()
    t.translate(-x1,0,0)
    t.scale(factor)
    return t


def orientation_indicator(bbox):
    orientation_vector = bbox[4] - bbox[0]
    orientation_vector /= max(1.0, np.linalg.norm(orientation_vector))
    anchor_point = 0.25 * (bbox[4] + bbox[5] + bbox[6] + bbox[7])
    tip_length = max(1.0, np.linalg.norm(bbox[4] - bbox[5]))
    tip = anchor_point + tip_length * orientation_vector
    return np.stack([tip, bbox[4], bbox[5], bbox[6], bbox[7]], axis=0)


def draw_bounding_box_bv(image, bbox, color=(0,0,255)):
    drawing.draw_line(image, bbox[0,0:2], bbox[1,0:2], color)
    drawing.draw_line(image, bbox[1,0:2], bbox[5,0:2], color)
    drawing.draw_line(image, bbox[5,0:2], bbox[4,0:2], color)
    drawing.draw_line(image, bbox[4,0:2], bbox[0,0:2], color)
    tip = bbox[8]
    for i in range(4,6):
        drawing.draw_line(image, tip[0:2], bbox[i,0:2], color)


def draw_bounding_box_image(image, bbox, color=(0,0,255)):
    drawing.draw_line(image, bbox[0,0:2], bbox[1,0:2], color)
    drawing.draw_line(image, bbox[1,0:2], bbox[2,0:2], color)
    drawing.draw_line(image, bbox[2,0:2], bbox[3,0:2], color)
    drawing.draw_line(image, bbox[3,0:2], bbox[0,0:2], color)

    drawing.draw_line(image, bbox[4,0:2], bbox[5,0:2], color)
    drawing.draw_line(image, bbox[5,0:2], bbox[6,0:2], color)
    drawing.draw_line(image, bbox[6,0:2], bbox[7,0:2], color)
    drawing.draw_line(image, bbox[7,0:2], bbox[4,0:2], color)

    drawing.draw_line(image, bbox[0,0:2], bbox[4,0:2], color)
    drawing.draw_line(image, bbox[1,0:2], bbox[5,0:2], color)
    drawing.draw_line(image, bbox[2,0:2], bbox[6,0:2], color)
    drawing.draw_line(image, bbox[3,0:2], bbox[7,0:2], color)

    tip = bbox[8]
    for i in range(4,8):
        drawing.draw_line(image, tip[0:2], bbox[i,0:2], color)
