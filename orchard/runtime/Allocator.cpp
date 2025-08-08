#include "Allocator.h"

namespace orchard::runtime {

ArenaAllocator::ArenaAllocator(orchard::core::tensor::DeviceType dev)
    : device_(dev) {}

orchard::core::tensor::Storage ArenaAllocator::allocate(size_t bytes) {
  orchard::core::tensor::Storage s(device_, bytes);
#ifdef ORCHARD_PROFILE_ALLOC
  static std::mutex log_mu;
  std::lock_guard<std::mutex> guard(log_mu);
  if (FILE* f = std::fopen("/tmp/orchard_tensor_profile.log", "a")) {
    std::fprintf(f, "alloc %zu %d\n", bytes, static_cast<int>(device_));
    std::fclose(f);
  }
#endif
  return s;
}

void ArenaAllocator::deallocate(orchard::core::tensor::Storage& storage) {
#ifdef ORCHARD_PROFILE_ALLOC
  static std::mutex log_mu;
  std::lock_guard<std::mutex> guard(log_mu);
  if (FILE* f = std::fopen("/tmp/orchard_tensor_profile.log", "a")) {
    std::fprintf(f, "free %zu %d\n", storage.bytes, static_cast<int>(device_));
    std::fclose(f);
  }
#endif
  storage = orchard::core::tensor::Storage();
}

} // namespace orchard::runtime
