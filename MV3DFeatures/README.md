# Install instructions
```
python2 setup.py build
sudo python2 setup.py install
```

# Dependencies
* boost-python

# Usage
```
from MV3DFeatures import create_birds_eye_view
intensity_map, density_map, height_map = create_birds_eye_view(velodyne_point_cloud)
```
