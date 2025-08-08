#include "Tensor.h"
#include <cstring>

namespace orchard::core::tensor {

Tensor::Tensor() = default;

Tensor::Tensor(const Tensor& other) : impl_(other.impl_) {
  if (impl_) impl_->refcount.fetch_add(1);
}

Tensor& Tensor::operator=(const Tensor& other) {
  if (this != &other) {
    this->~Tensor();
    impl_ = other.impl_;
    if (impl_) impl_->refcount.fetch_add(1);
  }
  return *this;
}

Tensor::Tensor(Tensor&& other) noexcept : impl_(other.impl_) {
  other.impl_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
  if (this != &other) {
    this->~Tensor();
    impl_ = other.impl_;
    other.impl_ = nullptr;
  }
  return *this;
}

Tensor::~Tensor() {
  if (impl_ && impl_->refcount.fetch_sub(1) == 1) {
    delete impl_;
  }
}

Tensor Tensor::empty(std::span<const int64_t> shape, DType dtype, DeviceType dev) {
  size_t elems = 1;
  for (auto s : shape) elems *= s;
  size_t bytes = elems * elementSize(dtype);
  Storage storage(dev, bytes);
  auto impl = new TensorImpl(std::move(storage), dtype, dev, shape);
  return Tensor{impl};
}

Tensor Tensor::clone() const {
  Tensor t = Tensor::empty(shape(), dtype(), device());
  std::memcpy(t.impl_->storage.data, impl_->storage.data, impl_->storage.bytes);
  return t;
}

void Tensor::to(DeviceType newDev) {
  if (!impl_ || impl_->device == newDev) return;
  Storage newStorage(newDev, impl_->storage.bytes);
  std::memcpy(newStorage.data, impl_->storage.data, impl_->storage.bytes);
  impl_->storage = std::move(newStorage);
  impl_->device = newDev;
}

Tensor Tensor::view(std::span<const int64_t> newShape) const {
  Tensor t(*this);
  if (!t.impl_) return t;
  t.impl_->is_view = true;
  t.impl_->ndim = static_cast<int64_t>(newShape.size());
  for (size_t i = 0; i < newShape.size() && i < 8; ++i) {
    t.impl_->shape[i] = newShape[i];
  }
  int64_t stride = 1;
  for (int64_t i = t.impl_->ndim - 1; i >= 0; --i) {
    t.impl_->stride[i] = stride;
    stride *= t.impl_->shape[i];
  }
  return t;
}

Tensor Tensor::slice(int dim, int start, int end, int step) const {
  Tensor t(*this);
  if (!t.impl_) return t;
  t.impl_->is_view = true;
  t.impl_->offset += start * t.impl_->stride[dim] * elementSize(dtype());
  t.impl_->shape[dim] = (end - start + step - 1) / step;
  t.impl_->stride[dim] *= step;
  return t;
}

Tensor Tensor::contiguous() const {
  if (!impl_ || impl_->isContiguous()) return *this;
  Tensor out = Tensor::empty(shape(), dtype(), device());
  std::memcpy(out.impl_->storage.data, impl_->storage.data, impl_->storage.bytes);
  return out;
}

std::string Tensor::toString() const {
  return impl_ ? impl_->toString() : "Tensor(nullptr)";
}

} // namespace orchard::core::tensor
