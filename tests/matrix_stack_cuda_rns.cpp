#include "../src/matrix_stack.hpp"
#include <gtest/gtest.h>

template <size_t N>
void AssertMatrixEqual(const MatrixStack<N> &a, const MatrixStack<N> &b) {
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      ASSERT_EQ(a(i, j), b(i, j)) << "Mismatch at (" << i << "," << j << ")";
    }
  }
}

TEST(MatrixStackCUDA_RNS, IdentityMultiply) {
  constexpr size_t N = 16;
  auto A = MatrixStack<N>::create_identity();
  auto B = MatrixStack<N>::create_identity();
  AssertMatrixEqual(A * B, A.multiply_cuda_rns(B));
}

TEST(MatrixStackCUDA_RNS, ZeroMatrix) {
  constexpr size_t N = 16;
  MatrixStack<N> A, B;
  AssertMatrixEqual(A * B, A.multiply_cuda_rns(B));
}

TEST(MatrixStackCUDA_RNS, RandomMatrices) {
  constexpr size_t N = 16;
  auto A = MatrixStack<N>::generate_random(-100, 100);
  auto B = MatrixStack<N>::generate_random(-100, 100);
  AssertMatrixEqual(A * B, A.multiply_cuda_rns(B));
}

TEST(MatrixStackCUDA_RNS, RandomMatricesBigValues) {
  constexpr size_t N = 16;
  auto A = MatrixStack<N>::generate_random(-1000, 1000);
  auto B = MatrixStack<N>::generate_random(-1000, 1000);
  AssertMatrixEqual(A * B, A.multiply_cuda_rns(B));
}

TEST(MatrixStackCUDA_RNS, NonSquareGrid) {
  constexpr size_t N = 15;
  auto A = MatrixStack<N>::generate_random(-100, 100);
  auto B = MatrixStack<N>::generate_random(-100, 100);
  AssertMatrixEqual(A * B, A.multiply_cuda_rns(B));
}

TEST(MatrixStackCUDA_RNS, SingleElement) {
  constexpr size_t N = 1;
  auto A = MatrixStack<N>::generate_random(-100, 100);
  auto B = MatrixStack<N>::generate_random(-100, 100);
  AssertMatrixEqual(A * B, A.multiply_cuda_rns(B));
}
