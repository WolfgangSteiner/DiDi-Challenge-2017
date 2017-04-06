import numpy as np
from collections import namedtuple
import VectorMath
from Tracklet import Tracklet
import math

Label = namedtuple("Label", "type,truncated,occluded,alpha,bbox,dimensions,location,rotation_y")

def read_labels(file_name, types=("Car","Van")):
    result = []
    with open(file_name) as f:
        for l in f.readlines():
            arr = l.split(' ')
            type = arr[0]
            truncated = float(arr[1])
            occluded = int(arr[2])
            alpha = float(arr[3])
            bbox = np.array(arr[4:8], np.float32)
            dimensions = np.array(arr[8:11], np.float32)
            location = np.array(arr[11:14], np.float32)
            rotation_y = float(arr[14])
            if type in types:
                result.append(Label(type, truncated, occluded, alpha, bbox, dimensions, location, rotation_y))

    return result


def tracklet_for_label(label, t_cam_to_velo):
    rotation_z = label.rotation_y + math.pi / 2
    px,py,pz = label.location
    h,w,l = label.dimensions
    position = t_cam_to_velo.transform(np.array([px,py,pz,1.0]))
    size = t_cam_to_velo.transform(np.array([w,h,l,0.0]))
    return Tracklet(position[0:3],size[0:3],rotation_z)


if __name__ == "__main__":
    labels = read_labels("kitti/training/label_2/002777.txt")
    print(labels)
