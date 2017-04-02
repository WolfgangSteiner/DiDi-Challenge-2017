import numpy as np
from collections import namedtuple

Label = namedtuple("Label", "type,truncated,occluded,alpha,bbox,dimensions,location,rotation_y")

def read_labels(file_name):
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
            if type in ("Car", "Van"):
                result.append(Label(type, truncated, occluded, alpha, bbox, dimensions, location, rotation_y))

    return result



if __name__ == "__main__":
    labels = read_labels("kitti/training/label_2/002777.txt")
    print(labels)