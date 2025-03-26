#include "../src/matrix_stack.hpp"
#include <gtest/gtest.h>

TEST(MatrixStackTest, MultiplyByIdentityReturnsOriginal) {
  MatrixStack<2> mat{1, 2, 3, 4};
  auto identity = MatrixStack<2>::create_identity();
  auto result = mat * identity;

  for (size_t i = 0; i < mat.size(); ++i) {
    for (size_t j = 0; j < mat.size(); ++j) {
      EXPECT_EQ(result(i, j), mat(i, j));
    }
  }
}

TEST(MatrixStackTest, MultiplyTwoMatrices2x2) {
  MatrixStack<2> A{1, 2, 3, 4};
  MatrixStack<2> B{5, 6, 7, 8};
  MatrixStack<2> expected{19, 22, 43, 50};
  auto result = A * B;

  for (size_t i = 0; i < expected.size(); ++i) {
    for (size_t j = 0; j < expected.size(); ++j) {
      EXPECT_EQ(result(i, j), expected(i, j));
    }
  }
}

TEST(MatrixStackTest, ScalarMultiplication) {
  MatrixStack<2> mat{1, 2, 3, 4};
  int32_t scalar = 2;
  MatrixStack<2> expected{2, 4, 6, 8};

  auto result1 = scalar * mat;
  auto result2 = mat * scalar;

  for (size_t i = 0; i < expected.size(); ++i) {
    for (size_t j = 0; j < expected.size(); ++j) {
      EXPECT_EQ(result1(i, j), expected(i, j));
      EXPECT_EQ(result2(i, j), expected(i, j));
    }
  }
}

TEST(MatrixStackTest, 1x1Multiplication) {
  MatrixStack<1> A{5};
  MatrixStack<1> B{7};
  auto result = A * B;

  EXPECT_EQ(result(0, 0), 35);
}

TEST(MatrixStackTest, MultiplyByZeroMatrixGivesZero) {
  MatrixStack<2> zero{};
  MatrixStack<2> mat{1, 2, 3, 4};

  auto result1 = zero * mat;
  auto result2 = mat * zero;

  for (size_t i = 0; i < mat.size(); ++i) {
    for (size_t j = 0; j < mat.size(); ++j) {
      EXPECT_EQ(result1(i, j), 0);
      EXPECT_EQ(result2(i, j), 0);
    }
  }
}
