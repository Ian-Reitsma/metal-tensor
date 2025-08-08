#pragma once

namespace orchard::runtime {

#ifdef __APPLE__
#import <Metal/Metal.h>
#endif

struct MetalContext {
#ifdef __APPLE__
  id<MTLDevice> device{MTLCreateSystemDefaultDevice()};
#endif
};

} // namespace orchard::runtime
