#pragma once

#include "levmar/internal/core.h"
#include "levmar/internal/problem.h"
#include "levmar/internal/solver_policy.h"
#include <string>
#include <string_view>

#include <levmar/internal/autodiff/jacobian_policy.h>
#include <levmar/internal/finite_difference.h>

template <class JacobianPolicy, Index M, Index N,
          ResidualCallable<M, N> Residual, class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
[[nodiscard]] inline ErrorOrVoid
evaluate_jacobian(LMSolveContext<M, N, Residual, Jacobian> &context,
                  std::string_view what = "Jacobian Evaluation") {
  const auto &problem = context.problem;
  auto &work = context.work;

  if constexpr (std::same_as<JacobianPolicy, UserJacobian>) {
    if constexpr (std::same_as<Jacobian, NoJacobian>) {
      return std::unexpected(
          Error{ErrorCode::InvalidProblem,
                std::string(what) + " failed: missing jacobian function"});
    } else {
      if (auto jacobian_result =
              problem.jacobian(work.x_current.view(), work.J.view());
          !jacobian_result) {
        return std::unexpected(Error{
            jacobian_result.error().code,
            std::string(what) + " failed: " + jacobian_result.error().message});
      }
    }
    ++context.result.jacobian_evaluations;
    return {};
  } else if constexpr (std::same_as<JacobianPolicy,
                                    ForwardDifferenceJacobian>) {
    return evaluate_forward_difference_jacobian(context, what);
  } else if constexpr (std::same_as<JacobianPolicy,
                                    CentralDifferenceJacobian>) {
    return evaluate_central_difference_jacobian(context, what);
  } else {
    static_assert(std::same_as<JacobianPolicy, AutoDiffJacobian>,
                  "Unsupported Jacobian policy");
    return evaluate_autodiff_residual_and_jacobian(context, what);
  }
}
