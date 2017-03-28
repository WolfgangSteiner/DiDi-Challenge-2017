//==============================================================================================
#include <boost/python.hpp>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
//==============================================================================================

int offset_for_point(float x, float y)
{
  if (x < 0.0f || x >= 70.0f || y < -40.0f || y >=40.0f)
  {
    return -1;
  }
  else
  {
    const int px = 800 - 1 - int((y + 40.0f) * 10.0f);
    const int py = 700 - 1 - int(x * 10.0f);

    return py * 800 + px;
  }
}


//----------------------------------------------------------------------------------------------

void _create_birds_eye_view(
  PyObject* apPointCloud,
  PyObject* apDensityMap,
  PyObject* apHeightMap,
  PyObject* apIntensityMap)
{
  auto pPointCloud = reinterpret_cast<PyArrayObject*>(apPointCloud);
  const int num_dimensions_point_cloud = PyArray_NDIM(pPointCloud);
  assert(num_dimensions_point_cloud == 2);
  const int num_lidar_points = PyArray_DIM(pPointCloud,0);

  const size_t kPointSize = 4;
  const float* pData = reinterpret_cast<float*>(PyArray_DATA(pPointCloud));

  auto pHeightMap = reinterpret_cast<PyArrayObject*>(apHeightMap);
  float* pHeightMapPtr = reinterpret_cast<float*>(PyArray_DATA(pHeightMap));

  auto pIntensityMap = reinterpret_cast<PyArrayObject*>(apIntensityMap);
  float* pIntensityMapPtr = reinterpret_cast<float*>(PyArray_DATA(pIntensityMap));

  auto pDensityMap = reinterpret_cast<PyArrayObject*>(apDensityMap);
  float* pDensityMapPtr = reinterpret_cast<float*>(PyArray_DATA(pDensityMap));

  for (int i = 0; i < num_lidar_points; ++i)
  {
    const float* pPoint = &pData[i * kPointSize];
    const float x = pPoint[0];
    const float y = pPoint[1];
    const float z = pPoint[2];
    const float r = pPoint[3];
    const int kOffset = offset_for_point(x,y);

    if (kOffset > -1)
    {
      if (z > pHeightMapPtr[kOffset])
      {
        assert(kOffset < 700 * 800);
        pHeightMapPtr[kOffset] = z;
	      pIntensityMapPtr[kOffset] = r;
      }
      pDensityMapPtr[kOffset]++;
    }
  }
}


//----------------------------------------------------------------------------------------------

BOOST_PYTHON_MODULE(_MV3DFeatures)
{
  boost::python::def("_create_birds_eye_view", _create_birds_eye_view);
}


//==============================================================================================
