#pragma once
#include "DType.h"
#include "Storage.h"
#include <array>
#include <atomic>

namespace orchard::core::tensor {

struct TensorImpl {
  Storage storage;
  std::array<int64_t,8> shape{};
  std::array<int64_t,8> stride{};
  std::atomic<int64_t> refcount{1};
  void* grad_fn = nullptr;
  void* grad_ctx = nullptr;
  DType dtype = DType::kFloat32;
  DeviceType device = DeviceType::MPS;
  size_t offset = 0;
  bool is_view = false;
};

} // namespace orchard::core::tensor
