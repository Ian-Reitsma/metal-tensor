#include "Tensor.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace orchard::core::tensor {

Tensor::Tensor() = default;

Tensor::~Tensor() {
  if (impl_ && --impl_->refcount == 0) {
    std::free(impl_->storage.data);
    delete impl_;
  }
}

Tensor::Tensor(Tensor&& other) noexcept { impl_ = other.impl_; other.impl_ = nullptr; }

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    this->~Tensor();
    impl_ = other.impl_;
    other.impl_ = nullptr;
  }
  return *this;
}

Tensor Tensor::empty(std::span<const int64_t> shape, DType dtype, DeviceType dev) {
  Tensor t;
  t.impl_ = new TensorImpl();
  t.impl_->dtype = dtype;
  t.impl_->device = dev;
  std::copy(shape.begin(), shape.end(), t.impl_->shape.begin());
  int64_t stride = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    t.impl_->stride[i] = stride;
    stride *= shape[i];
  }
  t.impl_->storage.bytes = stride * elementSize(dtype);
  t.impl_->storage.data = std::aligned_alloc(64, t.impl_->storage.bytes);
  return t;
}

Tensor Tensor::clone() const {
  Tensor t = Tensor::empty(shape(), dtype(), device());
  std::memcpy(t.impl_->storage.data, impl_->storage.data, impl_->storage.bytes);
  return t;
}

void Tensor::to(DeviceType newDev) { impl_->device = newDev; }

Tensor Tensor::view(std::span<const int64_t> newShape) const {
  Tensor t = *this;
  t.impl_->refcount++;
  t.impl_->is_view = true;
  std::copy(newShape.begin(), newShape.end(), t.impl_->shape.begin());
  // recompute strides
  int64_t stride = 1;
  for (int i = static_cast<int>(newShape.size()) - 1; i >= 0; --i) {
    t.impl_->stride[i] = stride;
    stride *= newShape[i];
  }
  return t;
}

Tensor Tensor::slice(int dim, int start, int end, int step) const {
  Tensor t = *this;
  t.impl_->refcount++;
  t.impl_->offset += start * impl_->stride[dim] * elementSize(dtype());
  t.impl_->shape[dim] = (end - start + step - 1) / step;
  t.impl_->stride[dim] *= step;
  t.impl_->is_view = true;
  return t;
}

Tensor Tensor::contiguous() const {
  // simplistic check: assume already contiguous
  return *this;
}

std::string Tensor::toString() const {
  std::ostringstream oss;
  oss << "Tensor(shape=[";
  bool first = true;
  for (int64_t dim : impl_->shape) {
    if (dim == 0) break;
    if (!first) oss << ",";
    oss << dim;
    first = false;
  }
  oss << "], dtype=" << toString(dtype())
      << ", device=" << (device() == DeviceType::MPS ? "mps" : "cpu") << ")";
  return oss.str();
}

} // namespace orchard::core::tensor
