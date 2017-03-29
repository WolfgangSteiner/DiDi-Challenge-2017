//==============================================================================================
#include <boost/python.hpp>
#include <utility>
#include <algorithm>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
//==============================================================================================

int offset_for_point(
  float x, float y,
  const std::pair<float,float>& aXRange, const std::pair<float,float>& aYRange,
  float aDeltaX, float aDeltaY,
  int aWidth, int aHeight)
{
  if (x < aYRange.first || x >= aYRange.second || y < aXRange.first || y >=aXRange.second)
  {
    return -1;
  }
  else
  {
    const int px = aWidth - 1 - int((y - aXRange.first) / aDeltaX);
    const int py = aHeight - 1 - int(x / aDeltaY);

//    px = std::max(0, std::min(px, aWidth - 1));
//    py = std::max(0, std::min(py, aHeight - 1));

    return py * aWidth + px;
  }
}

//----------------------------------------------------------------------------------------------

std::pair<float,float> unpack_range(PyObject* apRange)
{
  auto pRange = reinterpret_cast<PyArrayObject*>(apRange);
  assert(PyArray_NDIM(pRange) == 1);
  assert(PyArray_DIM(pRange, 0) == 2);
  const float* pRangeData = reinterpret_cast<const float*>(PyArray_DATA(pRange));
  return std::make_pair(pRangeData[0],pRangeData[1]);
}


//----------------------------------------------------------------------------------------------

void _create_birds_eye_view(
  PyObject* apPointCloud,
  PyObject* apFeatureMap,
  PyObject* apSrcXRange,
  PyObject* apSrcYRange)
{
  auto pPointCloud = reinterpret_cast<PyArrayObject*>(apPointCloud);
  const int num_dimensions_point_cloud = PyArray_NDIM(pPointCloud);
  assert(num_dimensions_point_cloud == 2);
  const int num_lidar_points = PyArray_DIM(pPointCloud,0);

  const size_t kPointSize = 4;
  const float* pData = reinterpret_cast<float*>(PyArray_DATA(pPointCloud));

  auto pFeatureMap = reinterpret_cast<PyArrayObject*>(apFeatureMap);
  float* pFeatureMapPtr = reinterpret_cast<float*>(PyArray_DATA(pFeatureMap));
  const int kNumDimensionsFeatureMap = PyArray_NDIM(pFeatureMap);
  assert(kNumDimensionsFeatureMap == 3);
  const int kNumFeatureMaps = PyArray_DIM(pFeatureMap,0);
  assert(kNumFeatureMaps >= 3);

  const int h = PyArray_DIM(pFeatureMap, 1);
  const int w = PyArray_DIM(pFeatureMap, 2);
  const size_t kFeatureMapSize = w * h;

  float* pIntensityMapPtr = pFeatureMapPtr;
  float* pDensityMapPtr = pFeatureMapPtr + kFeatureMapSize;
  float* pHeightMapPtr = pFeatureMapPtr + 2 * kFeatureMapSize;

  const auto kSrcXRange = unpack_range(apSrcXRange);
  const auto kSrcYRange = unpack_range(apSrcYRange);

  const float kDeltaX = (kSrcXRange.second - kSrcXRange.first) / w;
  const float kDeltaY = (kSrcYRange.second - kSrcYRange.first) / h;

  for (int i = 0; i < num_lidar_points; ++i)
  {
    const float* pPoint = &pData[i * kPointSize];
    const float x = pPoint[0];
    const float y = pPoint[1];
    const float z = pPoint[2];
    const float r = pPoint[3];
    const int kOffset = offset_for_point(x,y, kSrcXRange, kSrcYRange, kDeltaX, kDeltaY, w, h);

    if (kOffset > -1)
    {
      if (z > pHeightMapPtr[kOffset])
      {
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
