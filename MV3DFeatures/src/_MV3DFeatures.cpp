//==============================================================================================
#include <boost/python.hpp>
#include <utility>
#include <algorithm>
#include <vector>
#include <cmath>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
//==============================================================================================

std::pair<float,float> unpack_range(PyObject* apRange)
{
  auto pRange = reinterpret_cast<PyArrayObject*>(apRange);
  assert(PyArray_NDIM(pRange) == 1);
  assert(PyArray_DIM(pRange, 0) == 2);
  const float* pRangeData = reinterpret_cast<const float*>(PyArray_DATA(pRange));
  return std::make_pair(pRangeData[0],pRangeData[1]);
}


//----------------------------------------------------------------------------------------------

std::vector<int> get_shape(PyObject* apArray)
{
  auto pArray = reinterpret_cast<PyArrayObject*>(apArray);
  const int kNumDimensions = PyArray_NDIM(pArray);
  std::vector<int> shape;
  for (int i=0; i < kNumDimensions; ++i)
  {
    shape.push_back(PyArray_DIM(pArray, i));
  }

  return shape;
}


//----------------------------------------------------------------------------------------------

float* get_data(PyObject* apArray)
{
  auto pArray = reinterpret_cast<PyArrayObject*>(apArray);
  return reinterpret_cast<float*>(PyArray_DATA(pArray));
}

//----------------------------------------------------------------------------------------------

float get_float(PyObject* apFloat)
{
  return PyFloat_AsDouble(apFloat);
}


//----------------------------------------------------------------------------------------------

int calc_offset_bv(
  float x, float y,
  const std::pair<float,float>& aXRange, const std::pair<float,float>& aYRange,
  float aDeltaX, float aDeltaY,
  int aWidth,
  int aNumFeatureMaps)
{
  if (x < aXRange.first || x >= aXRange.second || y < aYRange.first || y >=aYRange.second
      || std::abs(x) > y)
  {
    return -1;
  }
  else
  {
    const int px = (x - aXRange.first) / aDeltaX;
    const int py = (y - aYRange.first) / aDeltaY;

    return (py * aWidth + px) * aNumFeatureMaps;
  }
}

//----------------------------------------------------------------------------------------------

void _create_birds_eye_view(
  PyObject* apPointCloud,
  PyObject* apFeatureMap,
  PyObject* apSrcXRange,
  PyObject* apSrcYRange,
  PyObject* apSrcZRange)
{
  const size_t kPointSize = 4;
  const auto kPointCloudShape = get_shape(apPointCloud);
  assert(kPointCloudShape.size() == 2);
  const int kNumLidarPoints = kPointCloudShape[0];
  const float* pPointData = get_data(apPointCloud);

  const auto kFeatureMapShape = get_shape(apFeatureMap);
  float* pFeatureMapPtr = get_data(apFeatureMap);
  assert(kFeatureMapShape.size() == 3);
  const int kNumFeatureMaps = kFeatureMapShape[2];
  assert(kNumFeatureMaps >= 3);

  const int h = kFeatureMapShape[0];
  const int w = kFeatureMapShape[1];
  const size_t kFeatureMapSize = w * h;

  const auto kSrcXRange = unpack_range(apSrcXRange);
  const auto kSrcYRange = unpack_range(apSrcYRange);
  const auto kSrcZRange = unpack_range(apSrcZRange);
  const float kZMin = kSrcZRange.first;
  const float kZMax = kSrcZRange.second;

  const float kDeltaX = (kSrcXRange.second - kSrcXRange.first) / w;
  const float kDeltaY = (kSrcYRange.second - kSrcYRange.first) / h;

  for (int i = 0; i < kNumLidarPoints; ++i)
  {
    const float* pPoint = &pPointData[i * kPointSize];
    const float x = -pPoint[1];
    const float y = pPoint[0];
    const float z = pPoint[2];
    const float r = pPoint[3];

    if (z > kZMax || z < kZMin)
    {
      continue;
    }

    const int kOffset = calc_offset_bv(x,y, kSrcXRange, kSrcYRange, kDeltaX, kDeltaY, w, kNumFeatureMaps);

    if (kOffset > -1)
    {
      if (z > pFeatureMapPtr[kOffset + 2])
      {
        pFeatureMapPtr[kOffset+2] = z; // height
	      pFeatureMapPtr[kOffset] = r;  // intensity
      }
      pFeatureMapPtr[kOffset+1]++;  // density
    }
  }
}


//----------------------------------------------------------------------------------------------

float calc_distance(float x, float y)
{
  return std::sqrt(std::pow(x,2) + std::pow(y,2));
}


//----------------------------------------------------------------------------------------------

int calc_offset_fv(
  float x,
  float y,
  float z,
  float aDeltaTheta,
  float aDeltaPhi,
  float z_min,
  float z_max,
  int aWidth,
  int aHeight,
  int aNumFeatureMaps)
{
  const float c = std::atan2(y, x) / aDeltaTheta;
  const float r = std::atan2(z - z_min, calc_distance(x,y)) / std::atan2(z_max - z_min, 10.0f);

  const int px = aWidth - 1 - (c + aWidth / 2);
  const int py = aHeight - 1 - aHeight * r;

  if (px < 0 || px >= aWidth || py < 0 || py >= aHeight)
  {
    return -1;
  }
  else
  {
    return (py * aWidth + px) * aNumFeatureMaps;
  }
}


//----------------------------------------------------------------------------------------------

void _create_front_view(
  PyObject* apPointCloud,
  PyObject* apFeatureMap,
  PyObject* apMinZ,
  PyObject* apMaxZ,
  PyObject* apDeltaTheta,
  PyObject* apDeltaPhi)
{
  const size_t kPointSize = 4;
  const auto kPointCloudShape = get_shape(apPointCloud);
  assert(kPointCloudShape.size() == 2);
  const int kNumLidarPoints = kPointCloudShape[0];
  const float* pPointData = get_data(apPointCloud);

  const auto kFeatureMapShape = get_shape(apFeatureMap);
  float* pFeatureMapPtr = get_data(apFeatureMap);
  assert(kFeatureMapShape.size() == 3);
  const int kNumFeatureMaps = kFeatureMapShape[2];
  assert(kNumFeatureMaps == 3);

  const int h = kFeatureMapShape[0];
  const int w = kFeatureMapShape[1];
  const size_t kFeatureMapSize = w * h;

  float* pIntensityMapPtr = pFeatureMapPtr;
  float* pDistanceMapPtr = pFeatureMapPtr + kFeatureMapSize;
  float* pHeightMapPtr = pFeatureMapPtr + 2 * kFeatureMapSize;

  const float kDeltaTheta = get_float(apDeltaTheta) * M_PI / 180.0f;
  const float kDeltaPhi = get_float(apDeltaPhi) * M_PI / 180.0f;
  const float kMinZ = get_float(apMinZ);
  const float kMaxZ = get_float(apMaxZ);

  for (int i = 0; i < kNumLidarPoints; ++i)
  {
    const float* pPoint = &pPointData[i * kPointSize];
    const float x = pPoint[0];
    const float y = pPoint[1];
    const float z = pPoint[2];
    const float r = pPoint[3];

    const int kOffset = calc_offset_fv(x, y, z, kDeltaTheta, kDeltaPhi, kMinZ, kMaxZ, w, h, kNumFeatureMaps);

    if (kOffset >= 0)
    {
      pFeatureMapPtr[kOffset + 0] = r;  // Intensity
      pFeatureMapPtr[kOffset + 1] = calc_distance(x,y);  // Distance
      pFeatureMapPtr[kOffset + 2] = z;  // Height
    }
  }
}

//----------------------------------------------------------------------------------------------

BOOST_PYTHON_MODULE(_MV3DFeatures)
{
  boost::python::def("_create_birds_eye_view", _create_birds_eye_view);
  boost::python::def("_create_front_view", _create_front_view);
}


//==============================================================================================
