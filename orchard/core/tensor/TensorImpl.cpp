#include "TensorImpl.h"
#include <sstream>

namespace orchard::core::tensor {

TensorImpl::TensorImpl(Storage&& s, DType dt, DeviceType dev, std::span<const int64_t> shp)
    : storage(std::move(s)), ndim(static_cast<int64_t>(shp.size())), dtype(dt), device(dev) {
  for (int64_t i = 0; i < ndim && i < 8; ++i) {
    shape[i] = shp[i];
  }
  int64_t stride_val = 1;
  for (int64_t i = ndim - 1; i >= 0; --i) {
    stride[i] = stride_val;
    stride_val *= shape[i];
  }
}

bool TensorImpl::isContiguous() const {
  int64_t expected = 1;
  for (int64_t i = ndim - 1; i >= 0; --i) {
    if (stride[i] != expected) return false;
    expected *= shape[i];
  }
  return true;
}

std::string TensorImpl::toString() const {
  std::ostringstream oss;
  oss << "TensorImpl(shape=[";
  for (int64_t i = 0; i < ndim; ++i) {
    if (i) oss << ",";
    oss << shape[i];
  }
  oss << "], dtype=" << toString(dtype)
      << ", device=" << (device == DeviceType::MPS ? "mps" : "cpu") << ")";
  return oss.str();
}

} // namespace orchard::core::tensor
