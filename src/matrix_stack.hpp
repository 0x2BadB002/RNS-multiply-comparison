#ifndef MATRIX
#define MATRIX

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <initializer_list>
#include <random>

#include "multiply_int32.hpp"

template <size_t N> class MatrixStack {
public:
  MatrixStack();
  explicit MatrixStack(std::initializer_list<int32_t> init);

  int32_t &operator()(size_t row, size_t col);
  const int32_t &operator()(size_t row, size_t col) const;

  MatrixStack &operator+=(const MatrixStack &rhs);
  MatrixStack &operator-=(const MatrixStack &rhs);

  MatrixStack transpose() const;
  static MatrixStack create_identity();
  static MatrixStack generate_random(int32_t min = -10'000,
                                     int32_t max = 10'000);

  MatrixStack<N> multiply_cuda(const MatrixStack<N> &rhs) const;
  MatrixStack<N> multiply_cuda_rns(const MatrixStack<N> &rhs) const;

  auto begin() { return data_.begin(); }
  auto end() { return data_.end(); }
  auto begin() const { return data_.begin(); }
  auto end() const { return data_.end(); }
  auto cbegin() const { return data_.cbegin(); }
  auto cend() const { return data_.cend(); }
  constexpr size_t size() const { return N; }

private:
  std::array<int32_t, N * N> data_;
};

template <size_t N>
inline MatrixStack<N> operator+(MatrixStack<N> lhs, const MatrixStack<N> &rhs) {
  lhs += rhs;
  return lhs;
}

template <size_t N>
inline MatrixStack<N> operator-(MatrixStack<N> lhs, const MatrixStack<N> &rhs) {
  lhs -= rhs;
  return lhs;
}

template <size_t N>
MatrixStack<N> operator*(const MatrixStack<N> &lhs, const MatrixStack<N> &rhs) {
  MatrixStack<N> result;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      int32_t sum{};
      for (size_t k = 0; k < N; ++k)
        sum += lhs(i, k) * rhs(k, j);
      result(i, j) = sum;
    }
  }
  return result;
}

template <size_t N>
MatrixStack<N> MatrixStack<N>::multiply_cuda(const MatrixStack<N> &rhs) const {
  MatrixStack<N> result;

  matrixMultiplyCUDA(data_.data(), rhs.data_.data(), result.data_.data(), N);

  return result;
}

template <size_t N>
MatrixStack<N>
MatrixStack<N>::multiply_cuda_rns(const MatrixStack<N> &rhs) const {
  MatrixStack<N> result;

  rnsMatrixMultiply(data_.data(), rhs.data_.data(), result.data_.data(), N);

  return result;
}

template <size_t N>
MatrixStack<N> operator*(int32_t scalar, MatrixStack<N> mat) {
  for (auto &elem : mat)
    elem *= scalar;
  return mat;
}

template <size_t N>
MatrixStack<N> operator*(MatrixStack<N> mat, int32_t scalar) {
  return scalar * mat;
}

template <size_t N> MatrixStack<N>::MatrixStack() : data_{} {}

template <size_t N>
MatrixStack<N>::MatrixStack(std::initializer_list<int32_t> init) {
  assert(init.size() == N * N && "Initializer list size mismatch");
  std::copy(init.begin(), init.end(), data_.begin());
}

template <size_t N>
int32_t &MatrixStack<N>::operator()(size_t row, size_t col) {
  assert(row < N && col < N && "Index out of bounds");
  return data_[row * N + col];
}

template <size_t N>
const int32_t &MatrixStack<N>::operator()(size_t row, size_t col) const {
  assert(row < N && col < N && "Index out of bounds");
  return data_[row * N + col];
}

template <size_t N>
MatrixStack<N> &MatrixStack<N>::operator+=(const MatrixStack<N> &rhs) {
  for (size_t i = 0; i < N * N; ++i)
    data_[i] += rhs.data_[i];
  return *this;
}

template <size_t N>
MatrixStack<N> &MatrixStack<N>::operator-=(const MatrixStack<N> &rhs) {
  for (size_t i = 0; i < N * N; ++i)
    data_[i] -= rhs.data_[i];
  return *this;
}

template <size_t N> MatrixStack<N> MatrixStack<N>::transpose() const {
  MatrixStack<N> result;
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      result(j, i) = (*this)(i, j);
  return result;
}

template <size_t N> MatrixStack<N> MatrixStack<N>::create_identity() {
  MatrixStack<N> result;
  for (size_t i = 0; i < N; ++i)
    result(i, i) = 1;
  return result;
}

template <size_t N>
MatrixStack<N> MatrixStack<N>::generate_random(int32_t min, int32_t max) {
  assert(min <= max && "Invalid range for random matrix");

  MatrixStack<N> result;
  std::random_device rd;
  std::mt19937 gen(rd());

  const float stddev = (max - min) / 6.0f;
  std::normal_distribution<float> dist(0.0f, stddev);

  for (auto &elem : result.data_) {
    float sample = dist(gen);
    int32_t val = static_cast<int32_t>(std::round(sample));
    elem = val < min ? min : (val > max ? max : val);
  }

  return result;
}

#endif // !MATRIX
