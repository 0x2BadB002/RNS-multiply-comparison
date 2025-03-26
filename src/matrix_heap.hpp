#ifndef MATRIX_HEAP
#define MATRIX_HEAP

#include <algorithm>
#include <cassert>
#include <cmath>
#include <initializer_list>
#include <random>
#include <vector>

template <typename T> class MatrixHeap {
public:
  MatrixHeap(size_t dim) : size_(dim), data_(dim * dim, 0) {}
  MatrixHeap(std::initializer_list<T> init)
      : size_(static_cast<size_t>(std::sqrt(init.size()))), data_(init) {
    assert(init.size() == size_ * size_);
    assert(init.size() != 0 &&
           "Used empty initializer list, instead use constructor");
  }

  T &operator()(size_t row, size_t col) {
    assert(row < size_ && col < size_);
    return data_[row * size_ + col];
  }

  const T &operator()(size_t row, size_t col) const {
    assert(row < size_ && col < size_);
    return data_[row * size_ + col];
  }

  MatrixHeap &operator+=(const MatrixHeap &rhs) {
    assert(size_ == rhs.size_);
    std::transform(data_.begin(), data_.end(), rhs.data_.begin(), data_.begin(),
                   std::plus<>());
    return *this;
  }

  MatrixHeap &operator-=(const MatrixHeap &rhs) {
    assert(size_ == rhs.size_);
    std::transform(data_.begin(), data_.end(), rhs.data_.begin(), data_.begin(),
                   std::minus<>());
    return *this;
  }

  MatrixHeap transpose() const {
    MatrixHeap result(size_);
    for (size_t i = 0; i < size_; ++i)
      for (size_t j = 0; j < size_; ++j)
        result(j, i) = (*this)(i, j);
    return result;
  }

  static MatrixHeap create_identity(size_t n) {
    MatrixHeap mat(n);
    for (size_t i = 0; i < n; ++i)
      mat(i, i) = 1;

    return mat;
  }

  static MatrixHeap generate_random(size_t n, T min = -10000, T max = 10000) {
    assert(min <= max);
    MatrixHeap mat(n);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<T> dist(min, max);

    for (auto &elem : mat.data_)
      elem = dist(gen);

    return mat;
  }

  MatrixHeap<T> multiply_cuda(const MatrixHeap<T> &rhs) const {
    MatrixHeap<T> result;

    matrixMultiplyCUDA(data_.data(), rhs.data_.data(), result.data_.data(),
                       size_);

    return result;
  }

  size_t size() const { return size_; }

  auto begin() { return data_.begin(); }
  auto end() { return data_.end(); }
  auto begin() const { return data_.begin(); }
  auto end() const { return data_.end(); }
  auto cbegin() const { return data_.cbegin(); }
  auto cend() const { return data_.cend(); }

private:
  size_t size_;
  std::vector<T> data_;
};

template <typename T>
inline MatrixHeap<T> operator+(MatrixHeap<T> lhs, const MatrixHeap<T> &rhs) {
  return lhs += rhs;
}

template <typename T>
inline MatrixHeap<T> operator-(MatrixHeap<T> lhs, const MatrixHeap<T> &rhs) {
  return lhs -= rhs;
}

template <typename T>
MatrixHeap<T> operator*(const MatrixHeap<T> &lhs, const MatrixHeap<T> &rhs) {
  assert(lhs.size() == rhs.size());
  auto n = lhs.size();

  MatrixHeap<T> result(n);
  for (size_t i = 0; i < n; ++i)
    for (size_t j = 0; j < n; ++j) {
      auto sum{0};
      for (size_t k = 0; k < n; ++k)
        sum += lhs(i, k) * rhs(k, j);
      result(i, j) = sum;
    }

  return result;
}

template <typename T> MatrixHeap<T> operator*(T scalar, MatrixHeap<T> mat) {
  for (auto &el : mat) {
    el *= scalar;
  }

  return mat;
}

template <typename T> MatrixHeap<T> operator*(MatrixHeap<T> mat, T scalar) {
  return scalar * mat;
}

#endif // !MATRIX_HEAP
