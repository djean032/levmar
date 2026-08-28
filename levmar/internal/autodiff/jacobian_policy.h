#pragma once

#include "levmar/internal/core.h"
#include "levmar/internal/evaluation_state.h"
#include "levmar/internal/problem.h"
#include <expected>
#include <levmar/internal/autodiff/dual_forward.h>
#include <levmar/internal/autodiff/graph_bridge.h>

namespace levmar::detail {

template <Index N>
inline constexpr bool kUsesDirectDualAutoDiff =
    N != std::dynamic_extent;

struct GraphAutoDiffJacobian {
  template <class Context>
  static ErrorOrVoid activate(Context &context, std::string_view what) {
    if (context.autodiff_cache.recorded) {
      return {};
    }
    return record_autodiff_graph(context, what);
  }

  template <class Context>
  static ErrorOrVoid evaluate(Context &context, std::string_view what) {
    return evaluate_cached_autodiff_graph(context);
  }
};

struct DirectDualJacobian {
  template <Index M, Index N, ResidualCallable<M, N> Residual, class Jacobian>
    requires OptionalJacobianCallable<Jacobian, M, N>
  static ErrorOrVoid evaluate(LMSolveContext<M, N, Residual, Jacobian> &context,
                              std::string_view what) {
    if constexpr (!DualResidualCallable<Residual, M, N>) {
      return std::unexpected(Error{
          ErrorCode::InvalidProblem,
          "JacobianMode::AutoDiff requires a scalar-generic residual callback "
          "compatible with Dual<N>"});
    } else {

      auto &work = context.work;
      return evaluate_residual_dual(
          context.problem.residual,
          ConstVectorView<N>(work.x_current.data(), work.x_current.size()),
          work.r.view(), work.J.view(), work.dual_eval, what);
    }
  }
};

template <Index M, Index N, ResidualCallable<M, N> Residual, class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
[[nodiscard]] inline ErrorOrVoid evaluate_autodiff_residual_and_jacobian(
    LMSolveContext<M, N, Residual, Jacobian> &context,
    std::string_view what = "AutoDiff residual/jacobian") {
  ErrorOrVoid evaluation;

  if constexpr (kUsesDirectDualAutoDiff<N>) {
    evaluation = DirectDualJacobian::evaluate(context, what);
  } else {
    if (auto activation = GraphAutoDiffJacobian::activate(context, what);
        !activation) {
      return activation;
    }
    evaluation = GraphAutoDiffJacobian::evaluate(context, what);
  }

  if (!evaluation) {
    return evaluation;
  }

  ++context.result.function_evaluations;
  ++context.result.jacobian_evaluations;
  return {};
}

} // namespace levmar::detail
