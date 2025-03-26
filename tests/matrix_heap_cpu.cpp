#include "../src/matrix_heap.hpp"
#include <cstdint>
#include <gtest/gtest.h>
#include <ostream>

template <class T>
std::ostream &operator<<(std::ostream &os, const MatrixHeap<T> &obj) {
  for (auto const &el : obj) {
    os << el << ' ';
  }
  os << '\n';

  return os;
}

TEST(MatrixHeapTest, MultiplyByIdentityReturnsOriginal) {
  auto mat = MatrixHeap<int32_t>::generate_random(3);
  auto identity = MatrixHeap<int32_t>::create_identity(3);
  auto result = mat * identity;

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_EQ(result(i, j), mat(i, j));
    }
  }
}

TEST(MatrixHeapTest, MultiplyTwoMatrices2x2) {
  MatrixHeap<int32_t> A{1, 2, 3, 4};
  MatrixHeap<int32_t> B{5, 6, 7, 8};
  MatrixHeap<int32_t> expected{19, 22, 43, 50};
  auto result = A * B;

  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      EXPECT_EQ(result(i, j), expected(i, j));
    }
  }
}

TEST(MatrixHeapTest, ScalarMultiplication) {
  MatrixHeap<int32_t> mat{1, 2, 3, 4};
  MatrixHeap<int32_t> expected_positive{2, 4, 6, 8};
  MatrixHeap<int32_t> expected_zero{0, 0, 0, 0};

  auto result1 = 2 * mat;
  auto result2 = mat * 0;

  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      EXPECT_EQ(result1(i, j), expected_positive(i, j));
      EXPECT_EQ(result2(i, j), expected_zero(i, j));
    }
  }
}

TEST(MatrixHeapTest, 1x1Multiplication) {
  MatrixHeap<int32_t> A{5};
  MatrixHeap<int32_t> B{7};
  auto result = A * B;
  EXPECT_EQ(result(0, 0), 35);
}

TEST(MatrixHeapTest, MultiplyByZeroMatrixGivesZero) {
  MatrixHeap<int32_t> zero(2);
  MatrixHeap<int32_t> mat{1, 2, 3, 4};

  auto result1 = zero * mat;
  auto result2 = mat * zero;

  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      EXPECT_EQ(result1.size(), result2.size());

      EXPECT_EQ(result1(i, j), 0);
      EXPECT_EQ(result2(i, j), 0);
    }
  }
}

TEST(MatrixHeapTest, NonCommutativeMultiplication) {
  MatrixHeap<int32_t> A{1, 2, 3, 4};
  MatrixHeap<int32_t> B{5, 6, 7, 8};
  auto AB = A * B;
  auto BA = B * A;

  EXPECT_NE(AB(0, 0), BA(0, 0));
  EXPECT_NE(AB(1, 1), BA(1, 1));
}

TEST(MatrixHeapTest, TransposeMultiplicationSymmetry) {
  MatrixHeap<int32_t> A{1, 2, 3, 4};
  auto AT = A.transpose();
  auto product = A * AT;

  EXPECT_EQ(product(0, 1), product(1, 0));
  EXPECT_EQ(product(0, 0), product(0, 0));
  EXPECT_EQ(product(1, 1), product(1, 1));
}

TEST(MatrixHeapTest, 3x3MatrixMultiplication) {
  MatrixHeap<int32_t> A{1, 2, 3, 4, 5, 6, 7, 8, 9};
  MatrixHeap<int32_t> B{9, 8, 7, 6, 5, 4, 3, 2, 1};
  MatrixHeap<int32_t> expected{30, 24, 18, 84, 69, 54, 138, 114, 90};

  auto result = A * B;

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      EXPECT_EQ(result(i, j), expected(i, j));
    }
  }
}
