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


def transformation_velo_to_bv(img_size, lidar_x_range, lidar_y_range):
    '''
    Construct transformation matrix from KITTI velodyne coordinates to birs's eye feature map
    pixel coordinates.

    Arguments:
    img_size:
        size of the feature map in pixels [w,h]
    lidar_x_range, lidar_y_range:
        x,y range of the lidar points that are mapped to the feature map, in KITTI Lidar coordinates.

    Returns:
    An object of class Transformation
    '''

    w,h = img_size
    x1,x2 = lidar_x_range
    y1,y2 = lidar_y_range
    factor = h / (x2 - x1)
    t = Transformation()
    t.flip_xy()
    t.mirror_x()
    t.translate(-y1,0,0)
    t.scale(factor)
    return t


def draw_bounding_box_bv(image, bbox, color=(0,0,255)):
    '''
    Draw bounding box into an image of a bird's eye feature map.

    Arguments:
        image:
            Destination bird's eye image.

        bbox:
            Bounding box with eight corner points and one point indicating the orientation.

        color:
            Color of the resulting bounding box image.
    '''
    drawing.draw_line(image, bbox[0,0:2], bbox[1,0:2], color)
    drawing.draw_line(image, bbox[1,0:2], bbox[5,0:2], color)
    drawing.draw_line(image, bbox[5,0:2], bbox[4,0:2], color)
    drawing.draw_line(image, bbox[4,0:2], bbox[0,0:2], color)
    tip = bbox[8]
    for i in range(4,6):
        drawing.draw_line(image, tip[0:2], bbox[i,0:2], color)


def draw_bounding_box_image(image, bbox, color=(0,0,255)):
    '''
    Draw bounding box into a camera image.

    Arguments:
        image:
            Destination camera image.

        bbox:
            Bounding box with eight corner points and one point indicating the orientation.
            The coordinates are supposed to be projected into 2D space.

        color:
            Color of the resulting bounding box image.
    '''
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
