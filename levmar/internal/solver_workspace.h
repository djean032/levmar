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

template <class LinearAlgebra, Index M, Index N> struct LinearAlgebraWorkspace;

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
