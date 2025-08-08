#pragma once
#include <cstddef>
#include "../core/tensor/Storage.h"

namespace orchard::runtime {

class Allocator {
public:
  virtual ~Allocator() = default;
  virtual orchard::core::tensor::Storage allocate(size_t bytes) = 0;
  virtual void deallocate(orchard::core::tensor::Storage& storage) = 0;
};

} // namespace orchard::runtime
