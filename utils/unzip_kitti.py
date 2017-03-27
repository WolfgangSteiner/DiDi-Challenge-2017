import glob
import os.path
import sys

for d in glob.glob("kitti/2011_*_*"):
    stem = os.path.split(d)[1].split(".")[0]
    date = "_".join(stem.split("_")[0:3])
    path = os.path.join("kitti", date, stem)
    if not os.path.exists(path):
        os.system("unzip %s -d kitti" % d)
    else:
        print("skipping %s" % d)

        
