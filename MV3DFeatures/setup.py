from distutils.core import setup, Extension
import numpy as np

print(np.get_include())


ext = Extension(
    'MV3DFeatures._MV3DFeatures',
    sources = ['src/_MV3DFeatures.cpp'],
    include_dirs=[np.get_include()],
    libraries=["boost_python"])

setup(
    name='MV3DFeatures',
    description = "Extract lidar features for MV3D network.",
    version='0.1.0',
    packages=['MV3DFeatures'],
    ext_modules = [ext])
