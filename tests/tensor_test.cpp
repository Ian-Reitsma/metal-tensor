#include "core/tensor/Tensor.h"
#include <gtest/gtest.h>
#include <array>
using namespace orchard::core::tensor;

TEST(TensorBasic, Alignment) {
  std::array<int64_t,2> shape{2,3};
  Tensor t = Tensor::empty(shape, DType::kFloat32, DeviceType::CPU);
  uintptr_t addr = reinterpret_cast<uintptr_t>(t.data());
  ASSERT_EQ(addr % 64, 0);
}

TEST(TensorBasic, CloneDistinct) {
  std::array<int64_t,2> shape{2,3};
  Tensor t = Tensor::empty(shape, DType::kFloat32, DeviceType::CPU);
  Tensor c = t.clone();
  ASSERT_NE(c.data(), t.data());
}

#ifdef __APPLE__
TEST(TensorBasic, ZeroCopyTo) {
  std::array<int64_t,2> shape{2,3};
  Tensor t = Tensor::empty(shape, DType::kFloat32, DeviceType::MPS);
  void* p = t.data();
  t.to(DeviceType::CPU);
  EXPECT_EQ(p, t.data());
}
#endif

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
