#pragma once
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#ifdef __APPLE__
#import <Metal/Metal.h>
#endif

namespace orchard::core::tensor {

enum class DeviceType { CPU, MPS };

struct Storage {
  DeviceType device = DeviceType::CPU;
  size_t bytes = 0;
#ifdef __APPLE__
  id<MTLBuffer> buffer = nil;
#endif
  void* data = nullptr;

  Storage() = default;
  Storage(DeviceType dev, size_t bytes);
  ~Storage();

  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
  Storage(Storage&& other) noexcept;
  Storage& operator=(Storage&& other) noexcept;
};

} // namespace orchard::core::tensor
