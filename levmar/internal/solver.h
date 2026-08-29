#pragma once

#include "levmar/internal/core.h"
#include "levmar/internal/evaluation_state.h"
#include "levmar/internal/problem.h"
#include "levmar/internal/solver_policy.h"
#include "levmar/internal/solver_workspace.h"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <functional>
#include <utility>

#include <expected>
#include <levmar/internal/solver_context.h>
#include <levmar/internal/solver_validation.h>
#include <numeric>

namespace levmar::detail {

inline constexpr double kMaxLMDampingDiagonal = 1e32;

template <Index M, Index N>
void reset_parameter_scaling(NoScaling, LMWorkspace<M, N> &work) {
  work.damping_scale.fill(1.0);
};

template <Index M, Index N>
void reset_parameter_scaling(JacobianColumnScaling, LMWorkspace<M, N> &work) {
  work.scale.fill(0.0);
  work.damping_scale.fill(0.0);
};

template <class Policy, Index M, Index N, ResidualCallable<M, N> Residual,
          class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
[[nodiscard]] ErrorOrVoid
initialize_solver(SolverContext<Policy, M, N, Residual, Jacobian> &context) {
  validate_static_solver_configuration<
      Policy, SolverContext<Policy, M, N, Residual, Jacobian>>();

  context.result = Result{};
  reset_parameter_scaling(typename Policy::ScalingPolicy{},
                          context.evaluation_context.work);

  auto &evaluation = context.evaluation_context;
  const auto &problem = evaluation.problem;

  if (auto validation = validate_context(evaluation); !validation) {
    return validation;
  }

  if (auto validation = validate_runtime_options<Policy>(evaluation.options);
      !validation) {
    return validation;
  }

  context.trust_radius = evaluation.options.lm.initial_trust_region_radius;
  context.selected_lambda = evaluation.options.lm.initial_lambda;
  context.result.lambda = context.selected_lambda;

  if constexpr (std::same_as<typename Policy::LinearAlgebraPolicy, DampedQr>) {
    if (problem.num_residuals < problem.num_parameters) {
      return std::unexpected(
          Error{ErrorCode::InvalidProblem,
                "DampedQr requires residuals >= parameters"});
    }
  }

  std::ranges::copy(evaluation.x,
                    context.evaluation_context.work.x_current.view().begin());

  context.result.lambda = evaluation.options.lm.initial_lambda;
  return {};
}

template <class Policy, Index M, Index N, ResidualCallable<M, N> Residual,
          class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
[[nodiscard]] bool model_evaluation_budget_available(
    SolverContext<Policy, M, N, Residual, Jacobian> &context) {
  const auto &options = context.evaluation_context.options;
  const Index used = context.result.function_evaluations;

  Index needed = 1;
  if constexpr (std::same_as<typename Policy::JacobianPolicy,
                             ForwardDifferenceJacobian>) {
    needed += context.evaluation_context.work.n;
  } else if constexpr (std::same_as<typename Policy::JacobianPolicy,
                                    CentralDifferenceJacobian>) {
    needed += 2 * context.evaluation_context.work.n;
  }

  if (used > options.max_function_evaluations ||
      needed > options.max_function_evaluations - used) {
    context.result.termination = TerminationReason::MaxFunctionEvaluations;
    context.result.message = "Maximum function evaluations reached";
    return false;
  }

  return true;
}

template <class Policy, Index M, Index N, ResidualCallable<M, N> Residual,
          class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
[[nodiscard]] ErrorOrVoid evaluate_current_model(
    SolverContext<Policy, M, N, Residual, Jacobian> &context) {
  if (!model_evaluation_budget_available<Policy>(context)) {
    return {};
  }
  auto &evaluation = context.evaluation_context;

  if constexpr (std::same_as<typename Policy::JacobianPolicy,
                             AutoDiffJacobian>) {
    return evaluate_jacobian<AutoDiffJacobian>(evaluation);
  } else {
    if (auto residual = evaluate_residual(evaluation); !residual) {
      return residual;
    }
    return evaluate_jacobian<typename Policy::JacobianPolicy>(evaluation);
  }
}

template <class Policy, Index M, Index N, ResidualCallable<M, N> Residual,
          class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
[[nodiscard]] ErrorOrVoid update_cost_and_gradient(
    SolverContext<Policy, M, N, Residual, Jacobian> &context) {
  auto &work = context.evaluation_context.work;

  const double cost =
      0.5 * std::transform_reduce(work.r.view().begin(), work.r.view().end(),
                                  0.0, std::plus<>{},
                                  [](double ri) { return ri * ri; });

  if (!std::isfinite(cost)) {
    context.result.termination = TerminationReason::NumericalFailure;
    context.result.message = "Non-finite residual or accumulated cost";
    return {};
  }

  double gradient_inf_norm = 0.0;
  for (Index j = 0; j < work.n; ++j) {
    double gj = 0.0;
    for (Index i = 0; i < work.m; ++i) {
      gj += work.J(i, j) * work.r[i];
    }

    if (!std::isfinite(gj)) {
      context.result.termination = TerminationReason::NumericalFailure;
      context.result.message =
          "Non-finite Jacobian/residual contribution or overflow";
      return {};
    }

    work.g[j] = gj;
    gradient_inf_norm = std::max(gradient_inf_norm, std::abs(gj));
  }

  context.result.final_cost = cost;
  context.result.gradient_inf_norm = gradient_inf_norm;
  return {};
}

template <class Policy, Index M, Index N, ResidualCallable<M, N> Residual,
          class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
[[nodiscard]] Result
finish_solver(SolverContext<Policy, M, N, Residual, Jacobian> &context) {
  auto &work = context.evaluation_context.work;
  context.result.parameters.assign(work.x_current.view().begin(),
                                   work.x_current.view().end());
  return std::move(context.result);
}

template <Index M, Index N>
[[nodiscard]] bool append_row_with_givens(DampedQrWorkspace<M, N> &qr, Index n,
                                          double &incoming_rhs) {
  for (Index k = 0; k < n; ++k) {
    const double diagonal = qr.qr_ut[k, k];
    const double incoming = qr.row[k];
    const double magnitude = std::hypot(diagonal, incoming);

    if (!std::isfinite(magnitude)) {
      return false;
    }

    if (magnitude == 0.0) {
      continue;
    }

    const double c = diagonal / magnitude;
    const double s = incoming / magnitude;

    for (Index j = k; j < n; ++j) {
      const double upper = qr.qr_ut[k, j];
      const double row_value = qr.row[j];

      qr.qr_ut[k, j] = c * upper + s * row_value;
      qr.row[j] = -s * upper + c * row_value;
    }

    const double upper_rhs = qr.rhs[k];
    qr.rhs[k] = c * upper_rhs + s * incoming_rhs;
    incoming_rhs = -s * upper_rhs + c * incoming_rhs;
  }
  return true;
}

template <class ScalingPolicy, Index M, Index N>
[[nodiscard]] bool solve_damped_qr(ScalingPolicy, LMWorkspace<M, N> &work,
                                   DampedQrWorkspace<M, N> &qr, double lambda) {
  if (!std::isfinite(lambda) || lambda <= 0.0) {
    return false;
  }
  const Index n = work.n;
  const double sqrt_lambda = std::sqrt(lambda);
  if (!std::isfinite(sqrt_lambda)) {
    return false;
  }

  qr.qr_ut.fill(0.0);
  qr.rhs.fill(0.0);

  for (Index i = 0; i < work.m; ++i) {
    for (Index j = 0; j < work.n; ++j) {
      if constexpr (kUsesJacobianColumnScaling<ScalingPolicy>) {
        qr.row[j] = work.J[i, j] / work.scale[j];
      } else {
        qr.row[j] = work.J[i, j];
      }
    }

    double incoming_rhs = -work.r[i];
    if (!append_row_with_givens(qr, n, incoming_rhs)) {
      return false;
    }
  }

  for (Index i = 0; i < n; ++i) {
    qr.row.fill(0.0);
    if constexpr (kUsesJacobianColumnScaling<ScalingPolicy>) {
      const double current_diagonal =
          work.damping_scale[i] * work.damping_scale[i];
      const double damping_diagonal =
          std::min(current_diagonal, kMaxLMDampingDiagonal);
      qr.row[i] = sqrt_lambda * std::sqrt(damping_diagonal) / work.scale[i];
    } else {
      qr.row[i] = sqrt_lambda;
    }

    double incoming_rhs = 0.0;
    if (!append_row_with_givens(qr, n, incoming_rhs)) {
      return false;
    }
  }
  for (Index j = 0; j < n; ++j) {
    for (Index i = 0; i <= j; ++i) {
      if (!std::isfinite(qr.qr_ut[i, j])) {
        return false;
      }
    }

    if (!std::isfinite(qr.rhs[j])) {
      return false;
    }
  }

  for (Index i = n; i-- > 0;) {
    const double diagonal = qr.qr_ut[i, i];
    if (!std::isfinite(diagonal) || diagonal == 0.0) {
      return false;
    }

    double sum = qr.rhs[i];
    for (Index j = i + 1; j < n; ++j) {
      sum -= qr.qr_ut[i, j] * work.step[j];
    }

    work.step[i] = sum / diagonal;
    if (!std::isfinite(work.step[i])) {
      return false;
    }
  }

  if constexpr (kUsesJacobianColumnScaling<ScalingPolicy>) {
    for (Index j = 0; j < n; ++j) {
      work.step[j] /= work.scale[j];
    }
  }
  return true;
}

template <Index M, Index N>
[[nodiscard]] bool update_parameter_scaling(NoScaling,
                                            LMWorkspace<M, N> &work) {
  return true;
}

template <Index M, Index N>
[[nodiscard]] bool update_parameter_scaling(JacobianColumnScaling,
                                            LMWorkspace<M, N> &work) {
  for (Index j = 0; j < work.n; ++j) {
    double column_norm = 0.0;
    for (Index i = 0; i < work.m; ++i) {
      column_norm = std::hypot(column_norm, work.J(i, j));
      if (!std::isfinite(column_norm)) {
        return false;
      }
    }
    if (work.scale[j] == 0.0) {
      work.scale[j] = column_norm == 0.0 ? 1.0 : column_norm;
    } /* else {
       work.scale[j] = std::max(work.scale[j], column_norm);
     }
     */
    work.damping_scale[j] = std::max(work.scale[j], column_norm);
  }
  return true;
}

template <class ScalingPolicy, Index M, Index N>
[[nodiscard]] bool
solve_linear_step(DampedQr, ScalingPolicy, LMWorkspace<M, N> &work,
                  DampedQrWorkspace<M, N> &linear, double lambda) {
  return solve_damped_qr(ScalingPolicy{}, work, linear, lambda);
}

template <class Policy, Index M, Index N, ResidualCallable<M, N> Residual,
          class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
[[nodiscard]] ErrorOr<bool>
try_lm_step(SolverContext<Policy, M, N, Residual, Jacobian> &context) {
  using LinearAlgebra = typename Policy::LinearAlgebraPolicy;
  using Scaling = typename Policy::ScalingPolicy;
  auto &work = context.evaluation_context.work;
  auto &linear = context.workspace.linear;
  auto &lm_opts = context.evaluation_context.options.lm;
  LmTrialTrace trace;
  trace.cost_before = context.result.final_cost;
  trace.lambda_before = context.result.lambda;
  trace.gradient_inf_norm = context.result.gradient_inf_norm;
  trace.trust_radius_before = context.trust_radius;
  Index inner_linear_solves = 0;
  bool radius_bound_active = false;

  const auto record_trace = [&](TrialDecision decision) {
    if (context.trial_trace == nullptr) {
      return;
    }
    trace.lambda_after = context.result.lambda;
    trace.decision = decision;
    trace.cost_before = context.result.final_cost;
    trace.trust_radius_after = context.trust_radius;
    trace.selected_lambda = context.selected_lambda;
    trace.inner_linear_solves = inner_linear_solves;
    trace.radius_bound_active = radius_bound_active;
    trace.termination = context.result.termination;
    context.trial_trace->push_back(std::move(trace));
  };

  const auto solve_at_lambda = [&](double lambda) -> bool {
    ++context.result.linear_solves;
    ++inner_linear_solves;
    return solve_linear_step(LinearAlgebra{}, Scaling{}, work, linear, lambda);
  };

  const auto step_norms = [&]() -> std::pair<double, double> {
    double raw_squared_norm = 0.0;
    double scaled_squared_norm = 0.0;

    for (Index j = 0; j < work.n; ++j) {
      const double step = work.step[j];
      double scale = 1.0;
      if constexpr (kUsesJacobianColumnScaling<Scaling>) {
        scale = work.scale[j];
      }
      raw_squared_norm += step * step;
      scaled_squared_norm += (scale * step) * (scale * step);
    }

    return {std::sqrt(raw_squared_norm), std::sqrt(scaled_squared_norm)};
  };

  const auto reject_and_shrink_radius =
      [&](double trial_scaled_step_norm) -> bool {
    const double next_radius = 0.25 * trial_scaled_step_norm;
    if (next_radius < lm_opts.min_trust_region_radius) {
      context.result.termination = TerminationReason::DampingLimit;
      return false;
    }

    context.trust_radius =
        std::max(lm_opts.min_trust_region_radius, next_radius);
    return true;
  };

  if (!solve_at_lambda(lm_opts.min_lambda)) {
    context.result.termination = TerminationReason::NumericalFailure;
    context.result.message = "Linear solve failed";
    record_trace(TrialDecision::LinearSolveFailure);
    return false;
  }

  auto [raw_step_norm, scaled_step_norm] = step_norms();

  double selected_lambda = lm_opts.min_lambda;

  if (scaled_step_norm > context.trust_radius) {
    radius_bound_active = true;
    double lambda_low = lm_opts.min_lambda;
    double lambda_high = std::max(context.selected_lambda, lambda_low);

    if (lambda_high == lambda_low) {
      if (lambda_high == lm_opts.max_lambda) {
        context.result.termination = TerminationReason::DampingLimit;
        record_trace(TrialDecision::DampingLimit);
        return false;
      }
      lambda_high = std::min(2.0 * lambda_high, lm_opts.max_lambda);
    }

    if (!solve_at_lambda(lambda_high)) {
      context.result.termination = TerminationReason::NumericalFailure;
      context.result.message = "Linear solve failed";
      record_trace(TrialDecision::LinearSolveFailure);
      return false;
    }

    std::tie(raw_step_norm, scaled_step_norm) = step_norms();

    while (scaled_step_norm > context.trust_radius) {
      if (lambda_high >= lm_opts.max_lambda) {
        context.result.termination = TerminationReason::DampingLimit;
        record_trace(TrialDecision::DampingLimit);
        return false;
      }

      lambda_low = lambda_high;
      lambda_high = std::min(2.0 * lambda_high, lm_opts.max_lambda);

      if (!solve_at_lambda(lambda_high)) {
        context.result.termination = TerminationReason::NumericalFailure;
        context.result.message = "Linear solve failed";
        record_trace(TrialDecision::LinearSolveFailure);
        return false;
      }
      std::tie(raw_step_norm, scaled_step_norm) = step_norms();
    }

    while (scaled_step_norm < 0.9 * context.trust_radius) {
      const double lambda_mid =
          std::exp(0.5 * (std::log(lambda_low) + std::log(lambda_high)));

      if (lambda_mid == lambda_low || lambda_mid == lambda_high) {
        break;
      }

      if (!solve_at_lambda(lambda_mid)) {
        context.result.termination = TerminationReason::NumericalFailure;
        context.result.message = "Linear solve failed";
        record_trace(TrialDecision::LinearSolveFailure);
        return false;
      }

      const auto [raw_mid_norm, scaled_mid_norm] = step_norms();
      if (scaled_mid_norm > context.trust_radius) {
        lambda_low = lambda_mid;
      } else {
        lambda_high = lambda_mid;
        raw_step_norm = raw_mid_norm;
        scaled_step_norm = scaled_mid_norm;
      }
    }

    if (!solve_at_lambda(lambda_high)) {
      context.result.termination = TerminationReason::NumericalFailure;
      context.result.message = "Linear solve failed";
      record_trace(TrialDecision::LinearSolveFailure);
      return false;
    }

    std::tie(raw_step_norm, scaled_step_norm) = step_norms();
    selected_lambda = lambda_high;
  }

  context.selected_lambda = selected_lambda;
  context.result.lambda = selected_lambda;
  context.result.step_norm = raw_step_norm;
  trace.raw_step_norm = raw_step_norm;
  trace.scaled_step_norm = scaled_step_norm;

  for (Index j = 0; j < work.n; ++j) {
    const double step = work.step[j];
    work.x_trial[j] = work.x_current[j] + step;

    if (!std::isfinite(work.x_trial[j])) {
      context.result.termination = TerminationReason::NumericalFailure;
      context.result.message = "Non-finite trial parameter";
      record_trace(TrialDecision::NonFiniteTrialParameter);
      return false;
    }
  }

  if (context.trial_trace != nullptr) {
    trace.current_parameters.reserve(work.n);
    trace.trial_parameters.reserve(work.n);
    trace.step.reserve(work.n);
    trace.gradient.reserve(work.n);
    trace.jacobian_column_norms.reserve(work.n);
    trace.parameter_scales.reserve(work.n);
    trace.effective_damping_diagonal.reserve(work.n);
    for (Index j = 0; j < work.n; ++j) {
      double scale = 1.0;
      if constexpr (kUsesJacobianColumnScaling<Scaling>) {
        scale = work.scale[j];
      }
      double column_squared_norm = 0.0;
      for (Index i = 0; i < work.m; ++i) {
        column_squared_norm += work.J[i, j] * work.J[i, j];
      }
      trace.current_parameters.push_back(work.x_current[j]);
      trace.trial_parameters.push_back(work.x_trial[j]);
      trace.step.push_back(work.step[j]);
      trace.gradient.push_back(work.g[j]);
      trace.jacobian_column_norms.push_back(std::sqrt(column_squared_norm));
      trace.parameter_scales.push_back(scale);
      double damping_diagonal = 1.0;
      if constexpr (kUsesJacobianColumnScaling<Scaling>) {
        damping_diagonal =
            std::min(work.damping_scale[j] * work.damping_scale[j],
                     kMaxLMDampingDiagonal);
      }
      trace.effective_damping_diagonal.push_back(context.result.lambda *
                                                 damping_diagonal);
    }
  }
  if (!std::isfinite(raw_step_norm) || !std::isfinite(scaled_step_norm)) {
    context.result.termination = TerminationReason::NumericalFailure;
    context.result.message = "Non-finite step norm";
    record_trace(TrialDecision::NonFiniteTrialParameter);
    return false;
  }

  if (context.result.step_norm <=
      context.evaluation_context.options.step_tolerance) {
    context.result.termination = TerminationReason::SmallStep;
    record_trace(TrialDecision::SmallStep);
    return false;
  }

  if (context.result.function_evaluations >=
      context.evaluation_context.options.max_function_evaluations) {
    context.result.termination = TerminationReason::MaxFunctionEvaluations;
    context.result.message = "Maximum function evaluations reached";
    record_trace(TrialDecision::FunctionEvaluationLimit);
    return false;
  }

  if (auto trial = evaluate_residual_at(
          context.evaluation_context, work.x_trial.view(), work.r_trial.view(),
          "Trial residual evaluation");
      !trial) {
    return std::unexpected(trial.error());
  }

  const double trial_cost =
      0.5 * std::transform_reduce(work.r_trial.view().begin(),
                                  work.r_trial.view().end(), 0.0, std::plus<>{},
                                  [](double ri) { return ri * ri; });
  trace.trial_cost = trial_cost;
  if (!std::isfinite(trial_cost)) {
    reject_and_shrink_radius(scaled_step_norm);
    record_trace(TrialDecision::NonFiniteTrialCost);
    return false;
  }

  double model_squared_norm = 0.0;

  for (Index i = 0; i < work.m; ++i) {
    double model_residual = work.r[i];

    for (Index j = 0; j < work.n; ++j) {
      model_residual += work.J[i, j] * work.step[j];
    }

    model_squared_norm += model_residual * model_residual;
  }

  const double model_cost = 0.5 * model_squared_norm;
  /*
  double predicted_reduction = 0.0;
  for (Index j = 0; j < work.n; ++j) {
    double scaled_step = work.step[j];
    if constexpr (kUsesJacobianColumnScaling<Scaling>) {
      scaled_step *= work.scale[j];
    }

    const double contribution =
        std::fma(context.result.lambda * scaled_step, scaled_step,
                 -work.step[j] * work.g[j]);
    if (!std::isfinite(contribution)) {
      context.result.termination = TerminationReason::NumericalFailure;
      context.result.message = "Non-finite predicted reduction";
      return false;
    }
    predicted_reduction += 0.5 * contribution;
  }
  if (!std::isfinite(predicted_reduction)) {
    context.result.termination = TerminationReason::NumericalFailure;
    context.result.message = "Non-finite predicted reduction";
    return false;
  }
  */
  const double actual_reduction = context.result.final_cost - trial_cost;
  const double predicted_reduction = context.result.final_cost - model_cost;
  trace.actual_reduction = actual_reduction;
  trace.predicted_reduction = predicted_reduction;
  if (!std::isfinite(model_cost)) {
    context.result.termination = TerminationReason::NumericalFailure;
    context.result.message = "Non-finite values in model cost analysis";
    record_trace(TrialDecision::NonFinitePredictedReduction);
    return false;
  }

  if (predicted_reduction <= 0.0) {
    const auto &options = context.evaluation_context.options;
    const double cost_change = std::abs(actual_reduction);
    const bool small_abs_change = cost_change <= options.cost_tolerance;
    const bool small_rel_change =
        options.relative_cost_tolerance > 0.0 &&
        context.result.final_cost > 0.0 &&
        cost_change <=
            options.relative_cost_tolerance * context.result.final_cost;

    if (small_abs_change || small_rel_change) {
      context.result.termination = TerminationReason::SmallCostReduction;
      record_trace(TrialDecision::SmallCostReduction);
      return false;
    }

    reject_and_shrink_radius(scaled_step_norm);
    record_trace(TrialDecision::NonPositivePredictedReduction);
    return false;
  }

  const double rho = actual_reduction / predicted_reduction;
  trace.rho = rho;
  if (!std::isfinite(rho)) {
    context.result.termination = TerminationReason::NumericalFailure;
    context.result.message = "Non-finite rho";
    record_trace(TrialDecision::NonFiniteRho);
    return false;
  }

  if (rho <= 1e-3) {
    reject_and_shrink_radius(scaled_step_norm);
    record_trace(TrialDecision::LowRho);
    return false;
  } else {
    if (rho > 0.75 && scaled_step_norm >= 0.9 * context.trust_radius) {
      context.trust_radius =
          std::min(lm_opts.max_trust_region_radius, 2.0 * context.trust_radius);
    }
  }

  record_trace(TrialDecision::Accepted);

  std::ranges::copy(work.x_trial.view().begin(), work.x_trial.view().end(),
                    work.x_current.view().begin());
  std::ranges::copy(work.r_trial.view().begin(), work.r_trial.view().end(),
                    work.r.view().begin());
  return true;
}

} // namespace levmar::detail

namespace levmar {

template <class Policy, Index M, Index N, ResidualCallable<M, N> Residual,
          class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
[[nodiscard]] ErrorOr<Result>
solve(detail::SolverContext<Policy, M, N, Residual, Jacobian> &context) {
  if (auto initialization = detail::initialize_solver<Policy>(context);
      !initialization) {
    return std::unexpected(initialization.error());
  }

  if (auto evaluation = detail::evaluate_current_model<Policy>(context);
      !evaluation) {
    return std::unexpected(evaluation.error());
  }

  if (context.result.termination != TerminationReason::NotTerminated) {
    return detail::finish_solver(context);
  }

  if (!detail::update_parameter_scaling(typename Policy::ScalingPolicy{},
                                        context.evaluation_context.work)) {
    context.result.termination = TerminationReason::NumericalFailure;
    context.result.message = "Non-finite Jacobian column scale";
    return detail::finish_solver(context);
  }

  if (auto update = detail::update_cost_and_gradient(context); !update) {
    return std::unexpected(update.error());
  }

  if (context.result.termination != TerminationReason::NotTerminated) {
    return detail::finish_solver(context);
  }

  context.result.initial_cost = context.result.final_cost;

  if (context.result.gradient_inf_norm <=
      context.evaluation_context.options.gradient_tolerance) {
    context.result.termination = TerminationReason::SmallGradient;
    return detail::finish_solver(context);
  }

  for (Index iteration = 0;
       iteration < context.evaluation_context.options.max_iterations;
       iteration++) {
    ++context.result.iterations;
    const double previous_cost = context.result.final_cost;

    auto accepted = detail::try_lm_step(context);

    if (!accepted) {
      return std::unexpected(accepted.error());
    }

    if (context.result.termination != TerminationReason::NotTerminated) {
      return detail::finish_solver(context);
    }

    if (!*accepted) {
      ++context.result.rejected_steps;
      continue;
    }
    ++context.result.accepted_steps;

    if (auto evaluation = detail::evaluate_current_model<Policy>(context);
        !evaluation) {
      return std::unexpected(evaluation.error());
    }

    if (context.result.termination != TerminationReason::NotTerminated) {
      return detail::finish_solver(context);
    }

    if (!detail::update_parameter_scaling(typename Policy::ScalingPolicy{},
                                          context.evaluation_context.work)) {
      context.result.termination = TerminationReason::NumericalFailure;
      context.result.message = "Non-finite Jacobian column scale";
      return detail::finish_solver(context);
    }

    if (auto update = detail::update_cost_and_gradient(context); !update) {
      return std::unexpected(update.error());
    }

    if (context.result.termination != TerminationReason::NotTerminated) {
      return detail::finish_solver(context);
    }

    const double cost_reduction = previous_cost - context.result.final_cost;
    const bool small_abs_reduction =
        cost_reduction <= context.evaluation_context.options.cost_tolerance;
    const bool small_rel_reduction =
        context.evaluation_context.options.relative_cost_tolerance > 0.0 &&
        previous_cost > 0.0 &&
        cost_reduction <=
            context.evaluation_context.options.relative_cost_tolerance *
                previous_cost;

    if (small_abs_reduction || small_rel_reduction) {
      context.result.termination = TerminationReason::SmallCostReduction;
      return detail::finish_solver(context);
    }

    if (context.result.gradient_inf_norm <=
        context.evaluation_context.options.gradient_tolerance) {
      context.result.termination = TerminationReason::SmallGradient;
      return detail::finish_solver(context);
    }
  }

  context.result.termination = TerminationReason::MaxIterations;
  return detail::finish_solver(context);
}

} // namespace levmar
