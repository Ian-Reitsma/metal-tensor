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
)
add_library(orchard_core STATIC ${CORE_SOURCES})
target_include_directories(orchard_core PUBLIC orchard)

set(METAL_SOURCES
  orchard/core/tensor/Storage.cpp
  orchard/runtime/Allocator.cpp
  orchard/runtime/CpuContext.cpp
)
if(APPLE)
  list(APPEND METAL_SOURCES orchard/runtime/MetalContext.mm)
endif()
add_library(orchard_metal STATIC ${METAL_SOURCES})
target_include_directories(orchard_metal PUBLIC orchard)
if(APPLE)
  target_link_libraries(orchard_metal PUBLIC orchard_core Metal::Metal Metal::MetalKit Metal::Foundation)
  target_compile_options(orchard_metal PRIVATE -fno-objc-arc)
else()
  target_link_libraries(orchard_metal PUBLIC orchard_core)
endif()

include(CTest)
if(BUILD_TESTING)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  )
  FetchContent_MakeAvailable(googletest)
  add_executable(tensor_tests tests/tensor_test.cpp)
  target_include_directories(tensor_tests PRIVATE orchard)
  target_link_libraries(tensor_tests PRIVATE orchard_core orchard_metal GTest::gtest_main)
  add_test(NAME tensor_tests COMMAND tensor_tests)
endif()
