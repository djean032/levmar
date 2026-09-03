#pragma once

#include "levmar/internal/core.h"
#include "levmar/internal/evaluation_state.h"
#include "levmar/internal/problem.h"
#include "levmar/internal/solver_workspace.h"
#include <limits>
#include <utility>
#include <vector>

namespace levmar::detail {

enum class TrialDecision {
  Accepted,
  LinearSolveFailure,
  NonFiniteTrialParameter,
  DampingLimit,
  SmallStep,
  FunctionEvaluationLimit,
  NonFiniteTrialCost,
  NonFinitePredictedReduction,
  NonPositivePredictedReduction,
  SmallCostReduction,
  NonFiniteRho,
  LowRho,
};

enum class LambdaPath {
  None,
  HouseholderGn,
  More,
  MoreSafeguard,
  LegacyBisection
};

struct LmTrialTrace {
  Index inner_linear_solves = 0;
  Index lmpar_iterations = 0;
  Index lmpar_safeguarded_refinements = 0;
  Index bisection_bracket_expansions = 0;
  Index bisection_refinements = 0;
  Index max_correlation_col_i = 0;
  Index max_correlation_col_j = 0;
  Index bisection_calls = 0;
  Index more_calls = 0;
  Index gn_calls = 0;
  bool radius_bound_active = false;
  bool lmpar_fallback = false;
  bool lmpar_bounds_valid = false;
  bool lmpar_has_feasible_step = false;
  double cost_before = std::numeric_limits<double>::quiet_NaN();
  double trial_cost = std::numeric_limits<double>::quiet_NaN();
  double actual_reduction = std::numeric_limits<double>::quiet_NaN();
  double predicted_reduction = std::numeric_limits<double>::quiet_NaN();
  double rho = std::numeric_limits<double>::quiet_NaN();
  double lambda_before = std::numeric_limits<double>::quiet_NaN();
  double lambda_after = std::numeric_limits<double>::quiet_NaN();
  double last_evaluated_lambda = std::numeric_limits<double>::quiet_NaN();
  double lmpar_parl = std::numeric_limits<double>::quiet_NaN();
  double lmpar_paru = std::numeric_limits<double>::quiet_NaN();
  double lmpar_feasible_lambda = std::numeric_limits<double>::quiet_NaN();
  double gradient_inf_norm = std::numeric_limits<double>::quiet_NaN();
  double raw_step_norm = std::numeric_limits<double>::quiet_NaN();
  double scaled_step_norm = std::numeric_limits<double>::quiet_NaN();
  double trust_radius_before = std::numeric_limits<double>::quiet_NaN();
  double trust_radius_after = std::numeric_limits<double>::quiet_NaN();
  double selected_lambda = std::numeric_limits<double>::quiet_NaN();
  double max_abs_column_correlation = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> current_parameters;
  std::vector<double> trial_parameters;
  std::vector<double> step;
  std::vector<double> gradient;
  std::vector<double> jacobian_column_norms;
  std::vector<double> parameter_scales;
  std::vector<double> effective_damping_diagonal;
  TrialDecision decision = TrialDecision::LinearSolveFailure;
  TerminationReason termination = TerminationReason::NotTerminated;
  LambdaPath lambda_path = LambdaPath::None;
};

template <class Policy, Index M, Index N, ResidualCallable<M, N> Residual,
          class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
struct SolverContext {
  using ProblemType = Problem<M, N, Residual, Jacobian>;
  using Workspace = SolverWorkspace<Policy, M, N>;
  using EvaluationContext = LMSolveContext<M, N, Residual, Jacobian>;

  Workspace &workspace;
  Result result;
  double trust_radius = 0.0;
  double selected_lambda = 0.0; // previous accepted lambda
  Index lmpar_iteration_limit= 10;
  Index lmpar_inner_solve_limit = 30;
  std::vector<LmTrialTrace> *trial_trace = nullptr;
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
