#pragma once
#include <cstddef>
#include "../core/tensor/Storage.h"
#include <mutex>
#include <cstdio>

namespace orchard::runtime {

class Allocator {
public:
  virtual ~Allocator() = default;
  virtual orchard::core::tensor::Storage allocate(size_t bytes) = 0;
  virtual void deallocate(orchard::core::tensor::Storage& storage) = 0;
};

class ArenaAllocator : public Allocator {
public:
  explicit ArenaAllocator(orchard::core::tensor::DeviceType dev);
  orchard::core::tensor::Storage allocate(size_t bytes) override;
  void deallocate(orchard::core::tensor::Storage& storage) override;
private:
  orchard::core::tensor::DeviceType device_;
};

} // namespace orchard::runtime
