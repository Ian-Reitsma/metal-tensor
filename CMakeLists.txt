cmake_minimum_required(VERSION 3.27)
if(APPLE)
project(metal_tensor LANGUAGES CXX OBJCXX)
else()
project(metal_tensor LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(APPLE)
  set(CMAKE_OBJCXX_STANDARD 20)
  set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)
  add_compile_options(-std=gnu++20 -ObjC++)
  set(CMAKE_OSX_ARCHITECTURES "arm64;arm64e" CACHE STRING "" FORCE)
else()
  add_compile_options(-std=gnu++20)
endif()

if(APPLE)
  find_package(Metal REQUIRED)
endif()

set(CORE_SOURCES
  orchard/core/tensor/Tensor.cpp
  orchard/core/tensor/TensorImpl.cpp
  orchard/core/tensor/Storage.cpp
)
add_library(orchard_core STATIC ${CORE_SOURCES})
target_include_directories(orchard_core PUBLIC .)

set(RUNTIME_SOURCES
  orchard/runtime/Allocator.cpp
  orchard/runtime/CpuContext.cpp
)
if(APPLE)
  list(APPEND RUNTIME_SOURCES orchard/runtime/MetalContext.mm)
endif()
add_library(orchard_metal STATIC ${RUNTIME_SOURCES})
target_include_directories(orchard_metal PUBLIC .)
if(APPLE)
  target_link_libraries(orchard_metal PUBLIC orchard_core Metal::Metal Metal::MetalKit Metal::Foundation)
  target_compile_options(orchard_metal PRIVATE -fno-objc-arc)
else()
  target_link_libraries(orchard_metal PUBLIC orchard_core)
endif()

enable_testing()
find_package(GTest REQUIRED)
add_executable(tensor_test tests/TensorTest.cpp)
target_link_libraries(tensor_test PRIVATE orchard_metal GTest::gtest_main)
add_test(NAME tensor_test COMMAND tensor_test)
