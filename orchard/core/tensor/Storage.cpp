#include "Storage.h"

namespace orchard::core::tensor {

Storage::Storage(DeviceType dev, size_t b) : device(dev), bytes(b) {
#ifdef __APPLE__
  if (device == DeviceType::MPS) {
    id<MTLDevice> d = MTLCreateSystemDefaultDevice();
    buffer = [d newBufferWithLength:b options:MTLResourceStorageModeShared];
    data = [buffer contents];
    return;
  }
#endif
  if (posix_memalign(&data, 64, b) != 0) {
    throw std::bad_alloc();
  }
}

Storage::~Storage() {
#ifdef __APPLE__
  if (buffer) {
    [buffer release];
    buffer = nil;
  }
#endif
  if (data && device == DeviceType::CPU) {
    std::free(data);
  }
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
