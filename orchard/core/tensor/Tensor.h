#pragma once
#include "TensorImpl.h"
#include <span>
#include <string>

namespace orchard::core::tensor {

class Tensor {
 public:
  Tensor();
  Tensor(const Tensor&) = delete;
  Tensor& operator=(const Tensor&) = delete;
  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(Tensor&& other) noexcept;
  ~Tensor();

  static Tensor empty(std::span<const int64_t> shape, DType dtype, DeviceType dev = DeviceType::MPS);
  Tensor clone() const;
  void to(DeviceType newDev);
  Tensor view(std::span<const int64_t> newShape) const;
  Tensor slice(int dim, int start, int end, int step=1) const;
  Tensor contiguous() const;

  DType dtype() const { return impl_->dtype; }
  DeviceType device() const { return impl_->device; }
  std::span<const int64_t> shape() const { return std::span<const int64_t>(impl_->shape.data(), 8); }
  size_t nbytes() const { return impl_->storage.bytes; }
  std::string toString() const;

 private:
  TensorImpl* impl_ = nullptr;
};

} // namespace orchard::core::tensor
