#pragma once
#include "DType.h"
#include "Storage.h"
#include <array>
#include <atomic>
#include <string>
#ifdef __APPLE__
#include <os/lock.h>
#else
#include <mutex>
#endif

namespace orchard::core::tensor {

struct TensorImpl {
  Storage storage;
  std::array<int64_t,8> shape{};
  std::array<int64_t,8> stride{};
  int64_t ndim = 0;
  std::atomic<int64_t> refcount{1};
#ifdef __APPLE__
  os_unfair_lock lock = OS_UNFAIR_LOCK_INIT;
#else
  std::mutex lock;
#endif
  void* grad_fn = nullptr;
  void* grad_ctx = nullptr;
  DType dtype = DType::kFloat32;
  DeviceType device = DeviceType::MPS;
  size_t offset = 0;
  bool is_view = false;

  TensorImpl(Storage&& storage, DType dtype, DeviceType device, std::span<const int64_t> shape);
  bool isContiguous() const;
  std::string toString() const;
};

} // namespace orchard::core::tensor
