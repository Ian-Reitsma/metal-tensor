cmake_minimum_required(VERSION 3.27)
project(metal_tensor LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
add_compile_options(-std=gnu++20 -ObjC++)
set(CMAKE_OSX_ARCHITECTURES "arm64;arm64e" CACHE STRING "" FORCE)

find_package(Metal REQUIRED)

set(CORE_SOURCES
  orchard/core/tensor/Tensor.cpp
  orchard/core/tensor/TensorImpl.cpp
)
add_library(orchard_core STATIC ${CORE_SOURCES})
target_include_directories(orchard_core PUBLIC orchard)

set(METAL_SOURCES
  orchard/core/tensor/Storage.cpp
  orchard/runtime/Allocator.cpp
  orchard/runtime/CpuContext.cpp
  orchard/runtime/MetalContext.mm
)
add_library(orchard_metal STATIC ${METAL_SOURCES})
target_include_directories(orchard_metal PUBLIC orchard)
target_link_libraries(orchard_metal PUBLIC orchard_core Metal::Metal Metal::MetalKit Metal::Foundation)
target_compile_options(orchard_metal PRIVATE -fno-objc-arc)
