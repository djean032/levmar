#pragma once

#include "levmar/internal/evaluation_state.h"
#include "levmar/internal/storage.h"
#include <levmar/internal/core.h>
#include <levmar/internal/solver_policy.h>

namespace levmar::detail {

template <Index M, Index N> struct DampedQrWorkspace {
  MatrixStorage<N, N> qr_ut;
  VectorStorage<N> rhs;
  VectorStorage<N> row;

  void resize(Index, Index n) {
    if constexpr (N == std::dynamic_extent) {
      if (qr_ut.rows() == n) {
        return;
      }
      qr_ut.resize(n, n);
      rhs.resize(n);
      row.resize(n);
    }
  }
};

template <Index M, Index N> struct PivotedHouseholderQrWorkspace {
  MatrixStorage<M, N> packed_qr;
  MatrixStorage<N, N> r_base;
  MatrixStorage<N, N> r_work;

  VectorStorage<M> transformed_rhs;
  VectorStorage<N> tau;
  VectorStorage<N, Index> permutation;
  VectorStorage<N> rhs;
  VectorStorage<N> row;
  VectorStorage<N> damping_diagonal;

  Index numerical_rank = 0;
  double rank_threshold = 0.0;
  bool factor_valid = false;

  void invalidate() noexcept {
    numerical_rank = 0;
    rank_threshold = 0.0;
    factor_valid = false;
  }

  void resize(Index m, Index n) {
    bool same_shape = true;

    if constexpr (M == std::dynamic_extent) {
      same_shape = same_shape && packed_qr.rows() == m;
    }

    if constexpr (N == std::dynamic_extent) {
      same_shape = same_shape && packed_qr.cols() == n;
    }

    if (same_shape) {
      invalidate();
      return;
    }

    if constexpr (M == std::dynamic_extent && N == std::dynamic_extent) {
      packed_qr.resize(m, n);
      transformed_rhs.resize(m);
    } else if constexpr (M == std::dynamic_extent) {
      packed_qr.resize(m);
      transformed_rhs.resize(m);
    } else if constexpr (N == std::dynamic_extent) {
      packed_qr.resize(n);
    }

    if constexpr (N == std::dynamic_extent) {
      r_base.resize(n, n);
      r_work.resize(n, n);
      tau.resize(n);
      permutation.resize(n);
      rhs.resize(n);
      row.resize(n);
      damping_diagonal.resize(n);
    }

    invalidate();
  }
};

template <class LinearAlgebra, Index M, Index N> struct LinearAlgebraWorkspace;

template <Index M, Index N>
struct LinearAlgebraWorkspace<PivotedHouseholderQr, M, N> {
  using Type = PivotedHouseholderQrWorkspace<M, N>;
};

template <Index M, Index N> struct LinearAlgebraWorkspace<DampedQr, M, N> {
  using Type = DampedQrWorkspace<M, N>;
};

template <class Policy, Index M, Index N> struct SolverWorkspace {
  LMWorkspace<M, N> work;
  typename LinearAlgebraWorkspace<typename Policy::LinearAlgebraPolicy, M,
                                  N>::Type linear;

  void resize(Index m, Index n) {
    work.resize(m, n);
    linear.resize(m, n);
  }
};

} // namespace levmar::detail
