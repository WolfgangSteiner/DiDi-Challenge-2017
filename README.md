## Example Video
https://www.youtube.com/watch?v=6exLh35T2GE

## Instructions
### To visualize KITTI driving:
* Download data from http://www.cvlibs.net/datasets/kitti/raw_data.php, e.g.   
  - http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip
  - http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_calib.zip
  - http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0001/2011_09_26_drive_0001_tracklets.zip
* unzip into a `kitti` subdirectory.
* `python2 render_features.py 2011_09_26 1` -> test.mp4

### To work with the Generator.py:
* Download:
  - http://kitti.is.tue.mpg.de/kitti/data_object_velodyne.zip
  - http://kitti.is.tue.mpg.de/kitti/data_object_image_2.zip
  - http://kitti.is.tue.mpg.de/kitti/data_object_calib.zip
  - http://kitti.is.tue.mpg.de/kitti/data_object_label_2.zip
* unzip into a `kitti` subdirectory.
* `python2 Generator.py` -> overview.png

## Dependencies
* python2 (because ros-indigo is python2 :( )
* pykitti
* opencv3
* numpy
