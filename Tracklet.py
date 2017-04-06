from collections import namedtuple
import numpy as np
import VectorMath

Tracklet = namedtuple("Tracklet", "position,size,rotation_z")

def bounding_box_for_tracklet(tracklet):
    l,w,h = tracklet.size
    bbox = np.zeros((8,4))
    bbox[0] = [-l/2, -w/2, 0.0, 1.0]
    bbox[1] = [-l/2,  w/2, 0.0, 1.0]
    bbox[2] = [-l/2,  w/2,   h, 1.0]
    bbox[3] = [-l/2, -w/2,   h, 1.0]
    bbox[4:8,:] = bbox[0:4,:] + [l, 0.0, 0.0, 0.0]
    t = VectorMath.Transformation()
    t.rotate_z(tracklet.rotation_z)
    t.translate(tracklet.position)
    return t.transform(bbox)
