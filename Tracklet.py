import xml.etree.ElementTree as ElementTree
import numpy as np
from math import cos, sin
from VectorMath import *

class Pose(object):
    def __init__(self):
        self.translation = None
        self.rotation = None
        self.state = None
        self.occlusion = None
        self.occlusion_kf = None
        self.truncation = None
        self.amt_occlusion = None
        self.amt_occlusion_kf = None
        self.amt_border_l = None
        self.amt_border_r = None
        self.amt_border_kf = None


class TrackletObject(object):
    def __init__(self):
        self.type = None
        self.size = None
        self.first_frame = None
        self.poses = []


    def __repr__(self):
        return "TrackletObject: Type: %s Size: %s first_frame: %d, poses: %s" % (self.type, self.size, self.first_frame, self.poses)


    def pose_for_frame(self, frame_idx):
        if frame_idx < self.first_frame or frame_idx >= self.first_frame + len(self.poses):
            return None
        else:
            return self.poses[frame_idx - self.first_frame]


    def has_pose_for_frame(self, frame_idx):
        return not self.pose_for_frame(frame_idx) is None


    def bounding_box_for_frame(self, frame_idx):
        pose = self.pose_for_frame(frame_idx)
        bbox = np.zeros((8,4))
        h, w, l = self.size
        bbox[0,:] = [-l/2, -w/2, 0.0, 1.0]
        bbox[1,:] = [-l/2,  w/2, 0.0, 1.0]
        bbox[2,:] = [-l/2,  w/2,   h, 1.0]
        bbox[3,:] = [-l/2, -w/2,   h, 1.0]
        bbox[4:8,:] = bbox[0:4,:] + [ l, 0.0, 0.0, 0.0]
        rz = pose.rotation[2]
        t = Transformation()
        t.rotate_z(rz)
        t.translate(pose.translation)
        return t.transform(bbox)


def get_float(node, tag):
    return float(node.find(tag).text)


def get_int(node, tag):
    return int(node.find(tag).text)


def parse_tracklets(file_name):
    tree = ElementTree.parse(file_name)
    root = tree.getroot()
    tracklets_node = root.find("tracklets")
    tracklets = []

    for n in tracklets_node.findall("item"):
        tracklet = TrackletObject()
        tracklet.type = n.find("objectType").text
        tracklet.size = np.array((get_float(n, "h"), get_float(n, "w"), get_float(n, "l")), np.float32)
        tracklet.first_frame = get_int(n, "first_frame")
        tracklets.append(tracklet)

        for p in n.findall("poses/item"):
            pose = Pose()
            pose.translation = np.array((get_float(p,"tx"), get_float(p,"ty"), get_float(p,"tz")), np.float32)
            pose.rotation = np.array((get_float(p,"rx"), get_float(p,"ry"), get_float(p,"rz")), np.float32)
            pose.state = get_int(p,"state")
            pose.occlusion = get_int(p,"occlusion")
            pose.occlusion_kf = get_int(p,"occlusion_kf")
            pose.amt_occlusion = get_float(p,"amt_occlusion")
            pose.amt_occlusion_kf = get_float(p,"amt_occlusion_kf")
            pose.amt_border_l = get_float(p,"amt_border_l")
            pose.amt_border_r = get_float(p,"amt_border_r")
            pose.amt_border_kf = get_float(p,"amt_border_kf")
            tracklet.poses.append(pose)

    return tracklets


def bounding_boxes_for_frame(tracklets, frame_idx):
    bboxes = []

    for t in tracklets:
        if t.has_pose_for_frame(frame_idx):
            bboxes.append(t.bounding_box_for_frame(frame_idx))

    return bboxes
