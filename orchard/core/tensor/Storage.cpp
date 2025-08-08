#include "Storage.h"
#include <new>

namespace orchard::core::tensor {

Storage::Storage(DeviceType dev, size_t b) : device(dev), bytes(b) {
#ifdef __APPLE__
  id<MTLDevice> d = MTLCreateSystemDefaultDevice();
  buffer = [d newBufferWithLength:b options:MTLResourceStorageModeShared];
  data = [buffer contents];
#else
  if (posix_memalign(&data, 64, b) != 0) {
    throw std::bad_alloc();
  }
#endif
}

Storage::~Storage() {
#ifdef __APPLE__
  if (buffer) {
    [buffer release];
    buffer = nil;
  }
#else
  if (data) {
    std::free(data);
  }
#endif
  data = nullptr;
  bytes = 0;
}

Storage::Storage(Storage&& other) noexcept {
  device = other.device;
  bytes = other.bytes;
#ifdef __APPLE__
  buffer = other.buffer;
  other.buffer = nil;
#endif
  data = other.data;
  other.data = nullptr;
  other.bytes = 0;
}

Storage& Storage::operator=(Storage&& other) noexcept {
  if (this != &other) {
    this->~Storage();
    device = other.device;
    bytes = other.bytes;
#ifdef __APPLE__
    buffer = other.buffer;
    other.buffer = nil;
#endif
    data = other.data;
    other.data = nullptr;
    other.bytes = 0;
  }
  return *this;
}

} // namespace orchard::core::tensor
