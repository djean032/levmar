#pragma once

#include "levmar/internal/core.h"
#include <array>
#include <cassert>
#include <string>
#include <string_view>

#include <levmar/internal/autodiff/dual.h>
#include <levmar/internal/problem.h>
#include <levmar/internal/storage.h>

namespace levmar::detail {

template <Index M, Index N> struct DualEvalContext {};

template <Index M, Index N>
  requires(N != std::dynamic_extent)
struct DualEvalContext<M, N> {
  std::array<Dual<N>, N> parameters;
  VectorStorage<M, Dual<N>> residuals;
  DualEvaluationState state;

  void prepare(Index m) {
    state = {};

    if constexpr (M == std::dynamic_extent) {
      residuals.resize(m);
    } else {
      assert(m == M);
    }
  }
};

template <class Residual, Index M, Index N>
concept DualResidualCallable =
    N != std::dynamic_extent &&
    ResidualCallableOn<Residual, ConstVectorView<N, Dual<N>>,
                       VectorView<M, Dual<N>>>;

template <Index M, Index N, class Residual>
  requires(N != std::dynamic_extent &&
           ResidualCallableOn<Residual, ConstVectorView<N, Dual<N>>,
                              VectorView<M, Dual<N>>>)
[[nodiscard]] inline ErrorOrVoid evaluate_residual_dual(
    const Residual &residual, ConstVectorView<N> parameters, VectorView<M> r,
    MatrixView<M, N> J, DualEvalContext<M, N> &context,
    std::string_view what = "Dual AutoDiff residual/jacobian") {
  const Index m = r.size();

  assert(parameters.size() == N);
  assert(J.extent(0) == m);
  assert(J.extent(1) == N);

  context.prepare(m);

  for (Index j = 0; j < N; ++j) {
    Dual<N> &x = context.parameters[j];
    x = Dual<N>{parameters[j]};
    x.derivatives[j] = 1.0;
    x.state = &context.state;
  }

  ConstVectorView<N, Dual<N>> dual_parameters(context.parameters.data(), N);
  VectorView<M, Dual<N>> dual_residuals(context.residuals.data(), m);

  if (auto result = residual(dual_parameters, dual_residuals); !result) {
    return std::unexpected(
        Error{result.error().code,
              std::string(what) + " failed: " + result.error().message});
  }

  if (context.state.pow_domain_error) {
    return std::unexpected(Error{ErrorCode::NumericalFailure,
                                 "AutoDiff pow requires a positive base"});
  }

  for (Index i = 0; i < m; ++i) {
    r[i] = context.residuals[i].value;
  }

  for (Index j = 0; j < N; ++j) {
    for (Index i = 0; i < m; ++i) {
      J[i, j] = context.residuals[i].derivatives[j];
    }
  }

  return {};
}

} // namespace levmar::detail
