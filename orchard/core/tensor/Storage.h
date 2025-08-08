#pragma once
#include <cstddef>
#include <cstdint>

namespace orchard::core::tensor {

enum class DeviceType { CPU, MPS };

struct Storage {
  void* data = nullptr;
  size_t bytes = 0;
};

} // namespace orchard::core::tensor
