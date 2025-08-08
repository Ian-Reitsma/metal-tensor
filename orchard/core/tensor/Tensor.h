#pragma once
#include "TensorImpl.h"
#include <span>
#include <string>

namespace orchard::core::tensor {

class Tensor {
 public:
  Tensor();
  Tensor(const Tensor& other);
  Tensor& operator=(const Tensor& other);
  Tensor(Tensor&& other) noexcept;
  Tensor& operator=(Tensor&& other) noexcept;
  ~Tensor();

  [[nodiscard]] static Tensor empty(std::span<const int64_t> shape, DType dtype, DeviceType dev = DeviceType::MPS);
  [[nodiscard]] static Tensor zerosLike(const Tensor& t);
  [[nodiscard]] static Tensor fromData(void* src, std::span<const int64_t> shape, DType dtype, DeviceType dev);
  [[nodiscard]] Tensor clone() const;
  void to(DeviceType newDev);
  Tensor view(std::span<const int64_t> newShape) const;
  Tensor slice(int dim, int start, int end, int step=1) const;
  [[nodiscard]] bool is_contiguous() const { return impl_ && impl_->isContiguous(); }
  Tensor contiguous() const;

  [[nodiscard]] DType dtype() const { return impl_->dtype; }
  [[nodiscard]] DeviceType device() const { return impl_->device; }
  [[nodiscard]] std::span<const int64_t> shape() const { return std::span<const int64_t>(impl_->shape.data(), static_cast<size_t>(impl_->ndim)); }
  [[nodiscard]] void* data() const { return impl_->storage.data; }
  [[nodiscard]] size_t nbytes() const { return impl_->storage.bytes; }
  [[nodiscard]] std::string toString() const;

 private:
  explicit Tensor(TensorImpl* impl) : impl_(impl) {}
  TensorImpl* impl_ = nullptr;
};

} // namespace orchard::core::tensor
