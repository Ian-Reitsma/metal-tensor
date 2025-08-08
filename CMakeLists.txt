cmake_minimum_required(VERSION 3.27)
project(metal_tensor LANGUAGES CXX OBJCXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
set(CMAKE_OBJCXX_STANDARD_REQUIRED ON)

find_package(Metal REQUIRED)

add_library(orchard_core STATIC
  orchard/core/tensor/Tensor.cpp
)

target_include_directories(orchard_core PUBLIC orchard)

add_library(orchard_metal STATIC)

target_link_libraries(orchard_metal PUBLIC Metal::Metal Metal::MetalKit Metal::Foundation)
