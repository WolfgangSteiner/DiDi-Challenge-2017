cmake_minimum_required(VERSION 2.8.0)

set(CMAKE_CXX_STANDARD 11)

find_package( PythonLibs 2.7 REQUIRED )
find_package( PythonInterp 2.7 REQUIRED )
find_package (Boost 1.54.0 REQUIRED COMPONENTS system filesystem)
include_directories( ${PYTHON_INCLUDE_DIRS} )

add_library (MV3DFeatures SHARED _MV3DFeatures.cpp)
