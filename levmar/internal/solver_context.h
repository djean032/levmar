#pragma once

#include "levmar/internal/core.h"
#include "levmar/internal/evaluation_state.h"
#include "levmar/internal/problem.h"
#include "levmar/internal/solver_workspace.h"
#include <utility>

namespace levmar::detail {
template <class Policy, Index M, Index N, ResidualCallable<M, N> Residual,
          class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
struct SolverContext {
  using ProblemType = Problem<M, N, Residual, Jacobian>;
  using Workspace = SolverWorkspace<Policy, M, N>;
  using EvaluationContext = LMSolveContext<M, N, Residual, Jacobian>;

  Workspace &workspace;
  Result result;
  EvaluationContext evaluation_context;

  template <class X>
  SolverContext(const ProblemType &problem, const Options &options,
                Workspace &workspace_, X &&x)
      : workspace(workspace_),
        evaluation_context(problem, options, result, workspace_.work,
                           std::forward<X>(x)) {
    workspace.resize(problem.num_residuals, problem.num_parameters);
  }
};
} // namespace levmar::detail
