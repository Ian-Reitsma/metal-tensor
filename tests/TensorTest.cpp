#include <gtest/gtest.h>
#include "orchard/core/tensor/Tensor.h"

using namespace orchard::core::tensor;

TEST(TensorTest, AlignmentAndClone) {
  std::array<int64_t,2> sh{2,3};
  auto t = Tensor::empty(sh, DType::kFloat32, DeviceType::CPU);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(t.data()) % 64, 0u);
  auto c = t.clone();
  EXPECT_NE(c.data(), t.data());
}

#ifdef __APPLE__
TEST(TensorTest, ZeroCopyTo) {
  std::array<int64_t,1> sh2{4};
  auto t = Tensor::empty(sh2, DType::kFloat32, DeviceType::MPS);
  void* ptr = t.data();
  t.to(DeviceType::CPU);
  EXPECT_EQ(ptr, t.data());
}
#endif
