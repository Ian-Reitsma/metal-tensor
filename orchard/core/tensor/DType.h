#pragma once
#include <cstddef>
#include <string>

namespace orchard::core::tensor {

enum class DType {
  kFloat32,
  kBFloat16,
  kFloat16,
  kUInt8,
  kInt32
};

inline size_t elementSize(DType t) {
  switch (t) {
    case DType::kFloat32: return 4;
    case DType::kBFloat16: return 2;
    case DType::kFloat16: return 2;
    case DType::kUInt8: return 1;
    case DType::kInt32: return 4;
  }
  return 0;
}

inline std::string toString(DType t) {
  switch (t) {
    case DType::kFloat32: return "f32";
    case DType::kBFloat16: return "bf16";
    case DType::kFloat16: return "f16";
    case DType::kUInt8: return "u8";
    case DType::kInt32: return "i32";
  }
  return "unknown";
}

} // namespace orchard::core::tensor
