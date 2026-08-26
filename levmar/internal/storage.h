#pragma once

#include <array>
#include <vector>

#include <levmar/internal/core.h>

namespace levmar::detail {

template <Index Extent, class Scalar = double> struct VectorStorage {
  std::array<Scalar, Extent> storage{};

  static constexpr Index extent = Extent;
  constexpr Index size() const noexcept { return Extent; }
  constexpr bool empty() const noexcept { return Extent == 0; }

  Scalar *data() { return storage.data(); }
  const Scalar *data() const { return storage.data(); }

  VectorView<Extent, Scalar> view() {
    return VectorView<Extent, Scalar>(storage.data(), Extent);
  }
  ConstVectorView<Extent, Scalar> view() const {
    return ConstVectorView<Extent, Scalar>(storage.data(), Extent);
  }

  Scalar &operator[](Index i) { return storage[i]; }

  const Scalar &operator[](Index i) const { return storage[i]; }

  void fill(const Scalar &value) { storage.fill(value); }
};

template <class Scalar> struct VectorStorage<std::dynamic_extent, Scalar> {
  std::vector<Scalar> storage;

  static constexpr Index extent = std::dynamic_extent;
  constexpr Index size() const noexcept { return storage.size(); }
  constexpr bool empty() const noexcept { return storage.empty(); }

  Scalar *data() { return storage.data(); }
  const Scalar *data() const { return storage.data(); }

  VectorView<std::dynamic_extent, Scalar> view() {
    return VectorView<std::dynamic_extent, Scalar>(storage.data(),
                                                   storage.size());
  }
  ConstVectorView<std::dynamic_extent, Scalar> view() const {
    return ConstVectorView<std::dynamic_extent, Scalar>(storage.data(),
                                                        storage.size());
  }

  Scalar &operator[](Index i) noexcept { return storage[i]; }

  const Scalar &operator[](Index i) const noexcept { return storage[i]; }

  void resize(Index n) { storage.resize(n); }
  void assign(Index n, const Scalar &value) { storage.assign(n, value); }
  void fill(const Scalar &value) { std::ranges::fill(storage, value); }
};

template <Index Rows, Index Cols, class Scalar = double> struct MatrixStorage {
  std::array<Scalar, Rows * Cols> storage{};

  static constexpr Index rows_extent = Rows;
  static constexpr Index cols_extent = Cols;

  constexpr Index rows() const noexcept { return Rows; }
  constexpr Index cols() const noexcept { return Cols; }
  constexpr Index leading_dim() const noexcept { return Rows; }
  constexpr Index size() const noexcept { return rows() * cols(); }
  constexpr bool empty() const noexcept { return size() == 0; }

  Scalar *data() { return storage.data(); }
  const Scalar *data() const { return storage.data(); }

  MatrixView<Rows, Cols, Scalar> view() {
    return MatrixView<Rows, Cols, Scalar>(storage.data());
  }
  ConstMatrixView<Rows, Cols, Scalar> view() const {
    return ConstMatrixView<Rows, Cols, Scalar>(storage.data());
  }

  Scalar &operator()(Index i, Index j) noexcept {
    return storage[i + j * leading_dim()];
  }

  const Scalar &operator()(Index i, Index j) const noexcept {
    return storage[i + j * leading_dim()];
  }

  Scalar &operator[](Index i, Index j) noexcept { return (*this)(i, j); }

  const Scalar &operator[](Index i, Index j) const noexcept {
    return (*this)(i, j);
  }

  void fill(const Scalar &value) { storage.fill(value); }
};

template <Index Cols, class Scalar>
struct MatrixStorage<std::dynamic_extent, Cols, Scalar> {
  Index rows_ = 0;
  std::vector<Scalar> storage;

  static constexpr Index rows_extent = std::dynamic_extent;
  static constexpr Index cols_extent = Cols;

  constexpr Index rows() const noexcept { return rows_; }
  constexpr Index cols() const noexcept { return Cols; }
  constexpr Index leading_dim() const noexcept { return rows_; }
  constexpr Index size() const noexcept { return rows() * cols(); }
  constexpr bool empty() const noexcept { return size() == 0; }

  Scalar *data() { return storage.data(); }
  const Scalar *data() const { return storage.data(); }

  MatrixView<std::dynamic_extent, Cols, Scalar> view() {
    return MatrixView<std::dynamic_extent, Cols, Scalar>(storage.data(), rows_);
  }

  ConstMatrixView<std::dynamic_extent, Cols, Scalar> view() const {
    return ConstMatrixView<std::dynamic_extent, Cols, Scalar>(storage.data(),
                                                              rows_);
  }

  Scalar &operator()(Index i, Index j) noexcept {
    return storage[i + j * leading_dim()];
  }

  const Scalar &operator()(Index i, Index j) const noexcept {
    return storage[i + j * leading_dim()];
  }

  Scalar &operator[](Index i, Index j) noexcept { return (*this)(i, j); }

  const Scalar &operator[](Index i, Index j) const noexcept {
    return (*this)(i, j);
  }

  void resize(Index rows) {
    rows_ = rows;
    storage.assign(rows_ * Cols, Scalar{});
  }

  void fill(const Scalar &value) { std::ranges::fill(storage, value); }
};

template <Index Rows, class Scalar>
struct MatrixStorage<Rows, std::dynamic_extent, Scalar> {
  Index cols_ = 0;
  std::vector<Scalar> storage;

  static constexpr Index rows_extent = Rows;
  static constexpr Index cols_extent = std::dynamic_extent;

  constexpr Index rows() const noexcept { return Rows; }
  constexpr Index cols() const noexcept { return cols_; }
  constexpr Index leading_dim() const noexcept { return Rows; }
  constexpr Index size() const noexcept { return rows() * cols(); }
  constexpr bool empty() const noexcept { return size() == 0; }

  Scalar *data() { return storage.data(); }
  const Scalar *data() const { return storage.data(); }

  MatrixView<Rows, std::dynamic_extent, Scalar> view() {
    return MatrixView<Rows, std::dynamic_extent, Scalar>(storage.data(), cols_);
  }

  ConstMatrixView<Rows, std::dynamic_extent, Scalar> view() const {
    return ConstMatrixView<Rows, std::dynamic_extent, Scalar>(storage.data(),
                                                              cols_);
  }

  Scalar &operator()(Index i, Index j) noexcept {
    return storage[i + j * leading_dim()];
  }

  const Scalar &operator()(Index i, Index j) const noexcept {
    return storage[i + j * leading_dim()];
  }

  Scalar &operator[](Index i, Index j) noexcept { return (*this)(i, j); }

  const Scalar &operator[](Index i, Index j) const noexcept {
    return (*this)(i, j);
  }

  void resize(Index cols) {
    cols_ = cols;
    storage.assign(Rows * cols_, Scalar{});
  }

  void fill(const Scalar &value) { std::ranges::fill(storage, value); }
};

template <class Scalar>
struct MatrixStorage<std::dynamic_extent, std::dynamic_extent, Scalar> {
  Index rows_ = 0;
  Index cols_ = 0;
  std::vector<Scalar> storage{};

  static constexpr Index rows_extent = std::dynamic_extent;
  static constexpr Index cols_extent = std::dynamic_extent;

  constexpr Index rows() const noexcept { return rows_; }
  constexpr Index cols() const noexcept { return cols_; }
  constexpr Index leading_dim() const noexcept { return rows_; }
  constexpr Index size() const noexcept { return rows() * cols(); }
  constexpr bool empty() const noexcept { return size() == 0; }

  Scalar *data() { return storage.data(); }
  const Scalar *data() const { return storage.data(); }

  MatrixView<std::dynamic_extent, std::dynamic_extent, Scalar> view() {
    return MatrixView<std::dynamic_extent, std::dynamic_extent, Scalar>(
        storage.data(), rows_, cols_);
  }

  ConstMatrixView<std::dynamic_extent, std::dynamic_extent, Scalar>
  view() const {
    return ConstMatrixView<std::dynamic_extent, std::dynamic_extent, Scalar>(
        storage.data(), rows_, cols_);
  }

  Scalar &operator()(Index i, Index j) noexcept {
    return storage[i + j * leading_dim()];
  }

  const Scalar &operator()(Index i, Index j) const noexcept {
    return storage[i + j * leading_dim()];
  }

  Scalar &operator[](Index i, Index j) noexcept { return (*this)(i, j); }

  const Scalar &operator[](Index i, Index j) const noexcept {
    return (*this)(i, j);
  }

  void resize(Index rows, Index cols) {
    cols_ = cols;
    rows_ = rows;
    storage.assign(rows_ * cols_, Scalar{});
  }

  void fill(const Scalar &value) { std::ranges::fill(storage, value); }
};

} // namespace levmar::detail
