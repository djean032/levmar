#pragma once

#include "levmar/internal/core.h"
#include "levmar/internal/evaluation_state.h"
#include "levmar/internal/problem.h"
#include "levmar/internal/solver_policy.h"
#include "levmar/internal/solver_workspace.h"
#include "levmar/internal/storage.h"
#include <algorithm>
#include <cmath>
#include <concepts>
#include <functional>
#include <limits>
#include <span>
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

  context.trust_radius = 0.0;
  context.selected_lambda = 0.0;
  context.result.lambda = 0.0;
  if constexpr (std::same_as<typename Policy::LinearAlgebraPolicy,
                             PivotedHouseholderQr>) {
    context.workspace.linear.invalidate();
  }

  if constexpr (kRequiresTallQr<typename Policy::LinearAlgebraPolicy>) {
    if (problem.num_residuals < problem.num_parameters) {
      return std::unexpected(Error{ErrorCode::InvalidProblem,
                                   "QR requires residuals >= parameters"});
    }
  }

  std::ranges::copy(evaluation.x,
                    context.evaluation_context.work.x_current.view().begin());

  return {};
}

template <Index Rows, Index Cols>
[[nodiscard]] bool
matrix_column_tail_l2_norm(const MatrixStorage<Rows, Cols> &matrix,
                           Index first_row, Index column, double &norm) {
  double max_abs = 0.0;
  const Index rows = matrix.rows();

  for (Index i = first_row; i < rows; ++i) {
    const double value = std::abs(matrix[i, column]);
    if (!std::isfinite(value)) {
      return false;
    }

    max_abs = std::max(max_abs, value);
  }

  if (max_abs == 0.0) {
    norm = 0.0;
    return true;
  }

  double sum0 = 0.0;
  double sum1 = 0.0;
  double sum2 = 0.0;
  double sum3 = 0.0;
  Index i = first_row;

  for (; i + 3 < rows; i += 4) {
    const double x0 = matrix[i, column] / max_abs;
    const double x1 = matrix[i + 1, column] / max_abs;
    const double x2 = matrix[i + 2, column] / max_abs;
    const double x3 = matrix[i + 3, column] / max_abs;

    sum0 = std::fma(x0, x0, sum0);
    sum1 = std::fma(x1, x1, sum1);
    sum2 = std::fma(x2, x2, sum2);
    sum3 = std::fma(x3, x3, sum3);
  }

  double sum = (sum0 + sum1) + (sum2 + sum3);
  for (; i < rows; ++i) {
    const double scaled = matrix[i, column] / max_abs;
    sum = std::fma(scaled, scaled, sum);
  }

  norm = max_abs * std::sqrt(sum);
  return std::isfinite(norm);
}

template <Index M, Index N>
[[nodiscard]] bool update_parameter_scaling(NoScaling,
                                            LMWorkspace<M, N> &work) {
  return true;
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
    double sum0 = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;
    Index i = 0;

    for (; i + 3 < work.m; i += 4) {
      sum0 = std::fma(work.J[i, j], work.r[i], sum0);
      sum1 = std::fma(work.J[i + 1, j], work.r[i + 1], sum1);
      sum2 = std::fma(work.J[i + 2, j], work.r[i + 2], sum2);
      sum3 = std::fma(work.J[i + 3, j], work.r[i + 3], sum3);
    }

    double gj = (sum0 + sum1) + (sum2 + sum3);
    for (; i < work.m; ++i) {
      gj = std::fma(work.J[i, j], work.r[i], gj);
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

[[nodiscard]] inline bool
apply_householder_reflector(ConstVectorView<std::dynamic_extent> reflector_tail,
                            double tau,
                            VectorView<std::dynamic_extent> target) {
  double sum0 = 0.0;
  double sum1 = 0.0;
  double sum2 = 0.0;
  double sum3 = 0.0;
  Index i = 0;

  for (; i + 3 < reflector_tail.size(); i += 4) {
    sum0 = std::fma(reflector_tail[i], target[i + 1], sum0);
    sum1 = std::fma(reflector_tail[i + 1], target[i + 2], sum1);
    sum2 = std::fma(reflector_tail[i + 2], target[i + 3], sum2);
    sum3 = std::fma(reflector_tail[i + 3], target[i + 4], sum3);
  }

  double dot = target[0] + (sum0 + sum1) + (sum2 + sum3);
  for (; i < reflector_tail.size(); ++i) {
    dot = std::fma(reflector_tail[i], target[i + 1], dot);
  }

  if (!std::isfinite(dot)) {
    return false;
  }

  const double coefficient = tau * dot;
  if (!std::isfinite(coefficient)) {
    return false;
  }

  target[0] -= coefficient;
  for (i = 0; i < reflector_tail.size(); ++i) {
    target[i + 1] = std::fma(-reflector_tail[i], coefficient, target[i + 1]);
  }

  return true;
}

template <class ScalingPolicy, Index M, Index N>
[[nodiscard]] bool
prepare_pivoted_householder_qr(ScalingPolicy, const LMWorkspace<M, N> &work,
                               PivotedHouseholderQrWorkspace<M, N> &qr,
                               double rank_tolerance_multiplier) {
  qr.invalidate();
  for (Index i = 0; i < work.m; ++i) {
    qr.transformed_rhs[i] = -work.r[i];
  }
  for (Index j = 0; j < work.n; ++j) {
    qr.permutation[j] = j;
    for (Index i = 0; i < work.m; ++i) {
      if constexpr (kUsesJacobianColumnScaling<ScalingPolicy>) {
        qr.packed_qr[i, j] = work.J[i, j] / work.scale[j];
      } else {
        qr.packed_qr[i, j] = work.J[i, j];
      }
    }
  }

  for (Index k = 0; k < work.n; ++k) {
    Index pivot = k;
    double pivot_norm = 0.0;

    for (Index j = k; j < work.n; ++j) {
      double norm = 0.0;
      if (!matrix_column_tail_l2_norm(qr.packed_qr, k, j, norm)) {
        return false;
      }

      if (norm > pivot_norm) {
        pivot = j;
        pivot_norm = norm;
      }
    }

    if (pivot != k) {
      for (Index i = 0; i < work.m; ++i) {
        std::swap(qr.packed_qr[i, k], qr.packed_qr[i, pivot]);
      }
      std::swap(qr.permutation[k], qr.permutation[pivot]);
    }

    if (pivot_norm == 0.0) {
      qr.tau[k] = 0.0;
      continue;
    }

    const double x0 = qr.packed_qr[k, k];
    const double alpha = -std::copysign(pivot_norm, x0);
    const double leading_value = x0 - alpha;

    if (!std::isfinite(alpha) || !std::isfinite(leading_value) ||
        leading_value == 0.0) {
      return false;
    }

    qr.packed_qr[k, k] = alpha;
    qr.tau[k] = (alpha - x0) / alpha;

    if (!std::isfinite(qr.tau[k])) {
      return false;
    }

    for (Index i = k + 1; i < work.m; ++i) {
      qr.packed_qr[i, k] /= leading_value;

      if (!std::isfinite(qr.packed_qr[i, k])) {
        return false;
      }
    }

    const auto reflector_tail = ConstVectorView<std::dynamic_extent>(
        qr.packed_qr.data() + (k + 1) + k * work.m, work.m - k - 1);

    for (Index j = k + 1; j < work.n; ++j) {
      auto target = VectorView<std::dynamic_extent>(
          qr.packed_qr.data() + k + j * work.m, work.m - k);

      if (!apply_householder_reflector(reflector_tail, qr.tau[k], target)) {
        return false;
      }
    }
    auto rhs_target = VectorView<std::dynamic_extent>(
        qr.transformed_rhs.data() + k, work.m - k);

    if (!apply_householder_reflector(reflector_tail, qr.tau[k], rhs_target)) {
      return false;
    }
  }

  for (Index j = 0; j < work.n; ++j) {
    for (Index i = 0; i <= j; ++i) {
      const double value = qr.packed_qr[i, j];
      if (!std::isfinite(value)) {
        return false;
      }
      qr.r_base[i, j] = value;
    }
  }

  for (Index k = 0; k < work.n; ++k) {
    const Index original_column = qr.permutation[k];

    if constexpr (kUsesJacobianColumnScaling<ScalingPolicy>) {
      const double scale = work.damping_scale[original_column];
      const double diagonal = std::min(scale * scale, kMaxLMDampingDiagonal);

      if (!std::isfinite(diagonal)) {
        return false;
      }
      qr.damping_diagonal[k] = diagonal;
    } else {
      qr.damping_diagonal[k] = 1.0;
    }
  }

  const double leading_diagonal = std::abs(qr.r_base[0, 0]);
  qr.rank_threshold =
      rank_tolerance_multiplier * std::numeric_limits<double>::epsilon() *
      static_cast<double>(std::max(work.m, work.n)) * leading_diagonal;

  if (!std::isfinite(qr.rank_threshold)) {
    return false;
  }

  qr.numerical_rank = 0;
  for (Index k = 0; k < work.n; ++k) {
    if (std::abs(qr.r_base[k, k]) > qr.rank_threshold) {
      ++qr.numerical_rank;
    }
  }
  qr.factor_valid = true;
  return true;
}

template <Index M, Index N>
[[nodiscard]] bool compute_pivoted_householder_gauss_newton_direction(
    const LMWorkspace<M, N> &work, PivotedHouseholderQrWorkspace<M, N> &qr) {
  const Index rank = qr.numerical_rank;

  for (Index k = 0; k < work.n; ++k) {
    qr.rhs[k] = k < rank ? qr.transformed_rhs[k] : 0.0;
  }
  for (Index i = rank; i-- > 0;) {
    const double diagonal = qr.r_base[i, i];
    if (!std::isfinite(diagonal) || diagonal == 0.0) {
      return false;
    }

    double sum = qr.rhs[i];
    for (Index j = i + 1; j < rank; ++j) {
      sum -= qr.r_base[i, j] * qr.rhs[j];
    }

    qr.rhs[i] = sum / diagonal;
    if (!std::isfinite(qr.rhs[i])) {
      return false;
    }
  }
  return true;
}

template <Index N>
[[nodiscard]] bool
append_row_with_givens(MatrixStorage<N, N> &upper, VectorStorage<N> &rhs,
                       VectorStorage<N> &row, Index n, double &incoming_rhs) {
  for (Index k = 0; k < n; ++k) {
    const double diagonal = upper[k, k];
    const double incoming = row[k];
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
      const double upper_value = upper[k, j];
      const double row_value = row[j];

      upper[k, j] = c * upper_value + s * row_value;
      row[j] = -s * upper_value + c * row_value;
    }

    const double upper_rhs = rhs[k];
    rhs[k] = c * upper_rhs + s * incoming_rhs;
    incoming_rhs = -s * upper_rhs + c * incoming_rhs;
  }
  return true;
}

template <class ScalingPolicy, Index M, Index N>
[[nodiscard]] bool
solve_pivoted_householder_qr(ScalingPolicy, LMWorkspace<M, N> &work,
                             PivotedHouseholderQrWorkspace<M, N> &qr,
                             double lambda) {
  if (!qr.factor_valid || !std::isfinite(lambda) || lambda <= 0.0) {
    return false;
  }

  const Index n = work.n;
  const double sqrt_lambda = std::sqrt(lambda);
  if (!std::isfinite(sqrt_lambda)) {
    return false;
  }

  for (Index j = 0; j < n; ++j) {
    qr.rhs[j] = qr.transformed_rhs[j];

    for (Index i = 0; i <= j; ++i) {
      qr.r_work[i, j] = qr.r_base[i, j];
    }
  }

  for (Index k = 0; k < n; ++k) {
    qr.row.fill(0.0);

    if constexpr (kUsesJacobianColumnScaling<ScalingPolicy>) {
      const Index original_column = qr.permutation[k];
      qr.row[k] = sqrt_lambda * std::sqrt(qr.damping_diagonal[k]) /
                  work.scale[original_column];
    } else {
      qr.row[k] = sqrt_lambda;
    }

    if (!std::isfinite(qr.row[k])) {
      return false;
    }

    double incoming_rhs = 0.0;
    if (!append_row_with_givens(qr.r_work, qr.rhs, qr.row, n, incoming_rhs)) {
      return false;
    }
  }

  for (Index i = n; i-- > 0;) {
    const double diagonal = qr.r_work[i, i];
    if (!std::isfinite(diagonal) || diagonal == 0.0) {
      return false;
    }

    double sum = qr.rhs[i];
    for (Index j = i + 1; j < n; ++j) {
      sum -= qr.r_work[i, j] * qr.rhs[j];
    }

    qr.rhs[i] = sum / diagonal;

    if (!std::isfinite(qr.rhs[i])) {
      return false;
    }
  }

  for (Index k = 0; k < n; ++k) {
    const Index original_column = qr.permutation[k];

    if constexpr (kUsesJacobianColumnScaling<ScalingPolicy>) {
      work.step[original_column] = qr.rhs[k] / work.scale[original_column];
    } else {
      work.step[original_column] = qr.rhs[k];
    }
    if (!std::isfinite(work.step[original_column])) {
      return false;
    }
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
    if (!append_row_with_givens(qr.qr_ut, qr.rhs, qr.row, n, incoming_rhs)) {
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
    if (!append_row_with_givens(qr.qr_ut, qr.rhs, qr.row, n, incoming_rhs)) {
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
[[nodiscard]] bool update_parameter_scaling(JacobianColumnScaling,
                                            LMWorkspace<M, N> &work) {
  for (Index j = 0; j < work.n; ++j) {
    double column_norm = 0.0;
    if (!matrix_column_tail_l2_norm(work.J, 0, j, column_norm)) {
      return false;
    }

    if (work.scale[j] == 0.0) {
      work.scale[j] = column_norm == 0.0 ? 1.0 : column_norm;
    }
    work.damping_scale[j] = std::max(work.scale[j], column_norm);
  }

  return true;
}

template <Index M, Index N>
[[nodiscard]] double
initial_scaled_parameter_norm(NoScaling, const LMWorkspace<M, N> &work) {
  double squared_norm = 0.0;

  for (Index j = 0; j < work.n; ++j) {
    const double value = work.x_current[j];
    squared_norm = std::fma(value, value, squared_norm);
  }
  return std::sqrt(squared_norm);
}

template <Index M, Index N>
[[nodiscard]] double
initial_scaled_parameter_norm(JacobianColumnScaling,
                              const LMWorkspace<M, N> &work) {
  double squared_norm = 0.0;

  for (Index j = 0; j < work.n; ++j) {
    const double scaled = work.scale[j] * work.x_current[j];
    squared_norm = std::fma(scaled, scaled, squared_norm);
  }

  return std::sqrt(squared_norm);
}

template <class ScalingPolicy, Index M, Index N>
[[nodiscard]] bool
solve_linear_step(DampedQr, ScalingPolicy, LMWorkspace<M, N> &work,
                  DampedQrWorkspace<M, N> &linear, double lambda) {
  return solve_damped_qr(ScalingPolicy{}, work, linear, lambda);
}

template <class ScalingPolicy, Index M, Index N>
[[nodiscard]] bool
solve_linear_step(PivotedHouseholderQr, ScalingPolicy, LMWorkspace<M, N> &work,
                  PivotedHouseholderQrWorkspace<M, N> &linear, double lambda) {
  return solve_pivoted_householder_qr(ScalingPolicy{}, work, linear, lambda);
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
  Index lmpar_iterations = 0;
  Index lmpar_safeguarded_refinements = 0;
  Index bisection_bracket_expansions = 0;
  Index bisection_refinements = 0;
  bool radius_bound_active = false;
  bool lmpar_fallback = false;
  double selected_lambda = lm_opts.min_lambda;
  double last_evaluated_lambda = std::numeric_limits<double>::quiet_NaN();

  const auto record_trace = [&](TrialDecision decision) {
    if (context.trial_trace == nullptr) {
      return;
    }
    trace.lambda_after = context.result.lambda;
    trace.decision = decision;
    trace.cost_before = context.result.final_cost;
    trace.trust_radius_after = context.trust_radius;
    trace.selected_lambda = selected_lambda;
    trace.last_evaluated_lambda = last_evaluated_lambda;
    trace.inner_linear_solves = inner_linear_solves;
    trace.lmpar_iterations = lmpar_iterations;
    trace.lmpar_safeguarded_refinements = lmpar_safeguarded_refinements;
    trace.bisection_bracket_expansions = bisection_bracket_expansions;
    trace.bisection_refinements = bisection_refinements;
    trace.radius_bound_active = radius_bound_active;
    trace.lmpar_fallback = lmpar_fallback;
    trace.termination = context.result.termination;
    context.trial_trace->push_back(std::move(trace));
  };

  const auto solve_at_lambda = [&](double lambda) -> bool {
    last_evaluated_lambda = lambda;
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

  double raw_step_norm = 0.0;
  double scaled_step_norm = 0.0;

  const auto select_lambda_by_bisection = [&]() -> bool {
    double lambda_low = lm_opts.min_lambda;
    double lambda_high = std::max(context.selected_lambda, lambda_low);
    if (lambda_high <= lambda_low) {
      lambda_high = std::min(2.0 * lambda_low, lm_opts.max_lambda);
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
      ++bisection_bracket_expansions;

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
      ++bisection_refinements;

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
    return true;
  };

  if constexpr (std::same_as<LinearAlgebra, PivotedHouseholderQr>) {
    ++context.result.linear_solves;
    ++inner_linear_solves;
    last_evaluated_lambda = 0.0;

    if (!compute_pivoted_householder_gauss_newton_direction(work, linear)) {
      context.result.termination = TerminationReason::NumericalFailure;
      context.result.message = "Gauss-Newton direction computation failure";
      record_trace(TrialDecision::LinearSolveFailure);
      return false;
    }

    for (Index k = 0; k < work.n; ++k) {
      const Index original_column = linear.permutation[k];

      if constexpr (kUsesJacobianColumnScaling<Scaling>) {
        work.step[original_column] =
            linear.rhs[k] / work.scale[original_column];
      } else {
        work.step[original_column] = linear.rhs[k];
      }

      if (!std::isfinite(work.step[original_column])) {
        context.result.termination = TerminationReason::NumericalFailure;
        context.result.message = "Non-finite Gauss-Newton step";
        record_trace(TrialDecision::LinearSolveFailure);
        return false;
      }
    }

    std::tie(raw_step_norm, scaled_step_norm) = step_norms();

    if (scaled_step_norm > context.trust_radius) {
      radius_bound_active = true;
      const auto damping_diagonal = [&](Index k) {
        if constexpr (kUsesJacobianColumnScaling<Scaling>) {
          const Index original_column = linear.permutation[k];
          return std::sqrt(linear.damping_diagonal[k]) /
                 work.scale[original_column];
        } else {
          return 1.0;
        }
      };

      double dxnorm_squared = 0.0;
      for (Index k = 0; k < work.n; ++k) {
        dxnorm_squared = std::fma(linear.rhs[k], linear.rhs[k], dxnorm_squared);
      }
      const double dxnorm = std::sqrt(dxnorm_squared);
      const double delta = context.trust_radius;
      const double fp = dxnorm - context.trust_radius;
      bool lmpar_bounds_valid = std::isfinite(dxnorm) && std::isfinite(fp);

      double parl = lm_opts.min_lambda;

      if (lmpar_bounds_valid && linear.numerical_rank == work.n &&
          dxnorm > 0.0) {
        bool parl_valid = true;

        for (Index k = 0; k < work.n; ++k) {
          const double damping = damping_diagonal(k);
          if (!std::isfinite(damping) || damping <= 0.0) {
            parl_valid = false;
            break;
          }

          linear.row[k] = damping * (damping * linear.rhs[k] / dxnorm);
          if (!std::isfinite(linear.row[k])) {
            parl_valid = false;
            break;
          }
        }

        if (parl_valid) {
          for (Index j = 0; j < work.n; ++j) {
            double sum = linear.row[j];

            for (Index i = 0; i < j; ++i) {
              sum -= linear.r_base[i, j] * linear.row[i];
            }

            const double diagonal = linear.r_base[j, j];
            if (!std::isfinite(sum) || !std::isfinite(diagonal) ||
                diagonal == 0.0) {
              parl_valid = false;
              break;
            }

            linear.row[j] = sum / diagonal;
            if (!std::isfinite(linear.row[j])) {
              parl_valid = false;
              break;
            }
          }
        }

        if (parl_valid) {
          double squared_norm = 0.0;

          for (Index k = 0; k < work.n; ++k) {
            squared_norm = std::fma(linear.row[k], linear.row[k], squared_norm);
          }

          if (std::isfinite(squared_norm) && squared_norm > 0.0) {
            parl = std::clamp((fp / delta) / squared_norm, lm_opts.min_lambda,
                              lm_opts.max_lambda);
          } else {
            parl_valid = false;
          }
        }
        lmpar_bounds_valid = parl_valid;
      }

      double gnorm_squared = 0.0;
      bool paru_valid = lmpar_bounds_valid;

      if (paru_valid) {
        for (Index k = 0; k < work.n; ++k) {
          double sum = 0.0;

          for (Index i = 0; i <= k; ++i) {
            sum = std::fma(linear.r_base[i, k], linear.transformed_rhs[i], sum);
          }

          const double damping = damping_diagonal(k);
          if (!std::isfinite(sum) || !std::isfinite(damping) ||
              damping <= 0.0) {
            paru_valid = false;
            break;
          }

          linear.row[k] = sum / damping;
          if (!std::isfinite(linear.row[k])) {
            paru_valid = false;
            break;
          }

          gnorm_squared = std::fma(linear.row[k], linear.row[k], gnorm_squared);
        }
      }

      double gnorm = 0.0;
      double paru = lm_opts.max_lambda;

      if (paru_valid && std::isfinite(gnorm_squared) && gnorm_squared > 0.0) {
        gnorm = std::sqrt(gnorm_squared);

        if (std::isfinite(gnorm)) {
          paru = std::clamp(gnorm / delta, parl, lm_opts.max_lambda);
        } else {
          paru_valid = false;
        }
      } else {
        paru_valid = false;
      }

      double feasible_lambda = std::numeric_limits<double>::quiet_NaN();
      double feasible_raw_norm = std::numeric_limits<double>::quiet_NaN();
      double feasible_scaled_norm = std::numeric_limits<double>::quiet_NaN();
      bool has_feasible_step = false;

      const auto remember_feasible_step = [&](double lambda) {
        if (scaled_step_norm > delta ||
            (has_feasible_step && scaled_step_norm <= feasible_scaled_norm)) {
          return;
        }

        has_feasible_step = true;
        feasible_lambda = lambda;
        feasible_raw_norm = raw_step_norm;
        feasible_scaled_norm = scaled_step_norm;
        trace.lmpar_has_feasible_step = true;
        trace.lmpar_feasible_lambda = feasible_lambda;
        std::ranges::copy(work.step.view(),
                          linear.feasible_step.view().begin());
      };

      const auto solve_pivoted_lambda = [&](double lambda) -> bool {
        if (!solve_at_lambda(lambda)) {
          context.result.termination = TerminationReason::NumericalFailure;
          context.result.message = "Linear solve failed";
          record_trace(TrialDecision::LinearSolveFailure);
          return false;
        }

        std::tie(raw_step_norm, scaled_step_norm) = step_norms();
        remember_feasible_step(lambda);
        return true;
      };

      lmpar_bounds_valid = lmpar_bounds_valid && paru_valid;
      trace.lmpar_bounds_valid = lmpar_bounds_valid;
      if (lmpar_bounds_valid) {
        trace.lmpar_parl = parl;
        trace.lmpar_paru = paru;
      }

      double lambda = lm_opts.min_lambda;

      if (lmpar_bounds_valid) {
        lambda = std::clamp(context.selected_lambda, parl, paru);

        if (context.selected_lambda <= 0.0) {
          lambda = std::clamp(gnorm / dxnorm, parl, paru);
        }
      }

      bool lambda_selected = false;

      if (lmpar_bounds_valid) {
        for (Index iteration = 0;
             iteration < context.lmpar_iteration_limit &&
             inner_linear_solves - 1 < context.lmpar_inner_solve_limit;
             ++iteration) {
          ++lmpar_iterations;
          if (!solve_pivoted_lambda(lambda)) {
            return false;
          }

          std::tie(raw_step_norm, scaled_step_norm) = step_norms();
          const double current_fp = scaled_step_norm - delta;

          if (!std::isfinite(current_fp)) {
            break;
          }

          if (std::abs(current_fp) <= 0.1 * delta) {
            lambda_selected = true;
            break;
          }

          if (current_fp > 0.0) {
            parl = std::max(parl, lambda);
          } else {
            paru = std::min(paru, lambda);
          }
          trace.lmpar_parl = parl;
          trace.lmpar_paru = paru;

          if (!(parl < paru) || scaled_step_norm <= 0.0) {
            break;
          }

          bool correction_valid = true;
          for (Index k = 0; k < work.n; ++k) {
            const double damping = damping_diagonal(k);
            linear.row[k] = damping * linear.rhs[k] / scaled_step_norm;

            if (!std::isfinite(linear.row[k])) {
              correction_valid = false;
              break;
            }
          }

          if (correction_valid) {
            for (Index j = 0; j < work.n; ++j) {
              const double diagonal = linear.r_work[j, j];
              if (!std::isfinite(diagonal) || diagonal == 0.0) {
                correction_valid = false;
                break;
              }

              linear.row[j] /= diagonal;
              if (!std::isfinite(linear.row[j])) {
                correction_valid = false;
                break;
              }

              for (Index i = j + 1; i < work.n; ++i) {
                linear.row[i] -= linear.r_work[j, i] * linear.row[j];
              }
            }
          }

          double correction_norm_squared = 0.0;
          if (correction_valid) {
            for (Index k = 0; k < work.n; ++k) {
              if (!std::isfinite(linear.row[k])) {
                correction_valid = false;
                break;
              }
              correction_norm_squared = std::fma(linear.row[k], linear.row[k],
                                                 correction_norm_squared);
            }
          }

          double candidate = std::numeric_limits<double>::quiet_NaN();
          if (correction_valid && std::isfinite(correction_norm_squared) &&
              correction_norm_squared > 0.0) {
            const double correction =
                (current_fp / delta) / correction_norm_squared;
            candidate = lambda + correction;
          }

          if (!std::isfinite(candidate) || candidate <= parl ||
              candidate >= paru) {
            candidate = std::exp(0.5 * (std::log(parl) + std::log(paru)));
          }

          if (!std::isfinite(candidate) || candidate <= parl ||
              candidate >= paru) {
            break;
          }

          lambda = candidate;
        }
        if (lambda_selected) {
          selected_lambda = lambda;
        }

        if (!lambda_selected) {
          lmpar_fallback = true;
          if (!has_feasible_step && parl < paru) {
            if (!solve_pivoted_lambda(paru)) {
              return false;
            }
          }
        }

        while (!lambda_selected &&
               inner_linear_solves - 1 < context.lmpar_inner_solve_limit &&
               parl < paru) {
          const double midpoint =
              std::exp(0.5 * (std::log(parl) + std::log(paru)));

          if (!std::isfinite(midpoint) || midpoint == parl ||
              midpoint == paru) {
            break;
          }

          if (!solve_pivoted_lambda(midpoint)) {
            return false;
          }

          ++lmpar_safeguarded_refinements;
          const double fp = scaled_step_norm - delta;

          if (std::abs(fp) <= 0.1 * delta) {
            lambda_selected = true;
            selected_lambda = midpoint;
            break;
          }

          if (fp > 0.0) {
            parl = midpoint;
          } else {
            paru = midpoint;
          }
          trace.lmpar_parl = parl;
          trace.lmpar_paru = paru;
        }

        if (!lambda_selected && has_feasible_step) {
          selected_lambda = feasible_lambda;
          raw_step_norm = feasible_raw_norm;
          scaled_step_norm = feasible_scaled_norm;
          std::ranges::copy(linear.feasible_step.view(),
                            work.step.view().begin());
          lambda_selected = true;
        }

        if (!lambda_selected) {
          context.result.termination = TerminationReason::DampingLimit;
          record_trace(TrialDecision::DampingLimit);
          return false;
        }
      } else if (!select_lambda_by_bisection()) {
        return false;
      }
    } else {
      selected_lambda = 0.0;
    }
  } else {
    if (!solve_at_lambda(lm_opts.min_lambda)) {
      context.result.termination = TerminationReason::NumericalFailure;
      context.result.message = "Linear solve failed";
      record_trace(TrialDecision::LinearSolveFailure);
      return false;
    }

    std::tie(raw_step_norm, scaled_step_norm) = step_norms();

    if (scaled_step_norm > context.trust_radius) {
      radius_bound_active = true;
      if (!select_lambda_by_bisection()) {
        return false;
      }
    }
  }

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

    trace.max_abs_column_correlation = 0.0;

    for (Index j = 0; j < work.n; ++j) {
      for (Index k = j + 1; k < work.n; ++k) {
        const double norm_j = trace.jacobian_column_norms[j];
        const double norm_k = trace.jacobian_column_norms[k];

        if (norm_j == 0.0 || norm_k == 0.0) {
          continue;
        }

        double dot = 0.0;

        for (Index i = 0; i < work.m; ++i) {
          dot += work.J[i, j] * work.J[i, k];
        }

        const double abs_corr = std::abs(dot / (norm_j * norm_k));

        if (abs_corr > trace.max_abs_column_correlation) {
          trace.max_abs_column_correlation = abs_corr;
          trace.max_correlation_col_i = j;
          trace.max_correlation_col_j = k;
        }
      }
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

  std::ranges::copy(work.r.view().begin(), work.r.view().end(),
                    work.r_trial_minus.view().begin());

  double *model_residual = work.r_trial_minus.data();
  for (Index j = 0; j < work.n; ++j) {
    const double step = work.step[j];

    for (Index i = 0; i < work.m; ++i) {
      model_residual[i] = std::fma(work.J[i, j], step, model_residual[i]);
    }
  }

  const double model_squared_norm = std::transform_reduce(
      model_residual, model_residual + work.m, 0.0, std::plus<>{},
      [](double value) { return value * value; });

  const double model_cost = 0.5 * model_squared_norm;

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

  if (rho <= 0.25) {
    context.trust_radius =
        std::max(lm_opts.min_trust_region_radius, 0.25 * context.trust_radius);
  } else {
    if (rho > 0.75 && scaled_step_norm >= 0.9 * context.trust_radius) {
      context.trust_radius =
          std::min(lm_opts.max_trust_region_radius, 2.0 * context.trust_radius);
    }
  }

  if (rho <= 0.1) {
    reject_and_shrink_radius(scaled_step_norm);
    record_trace(TrialDecision::LowRho);
    return false;
  }

  context.selected_lambda = selected_lambda;
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

  const double scaled_x_norm = detail::initial_scaled_parameter_norm(
      typename Policy::ScalingPolicy{}, context.workspace.work);
  const double factor =
      context.evaluation_context.options.lm.initial_trust_region_factor;

  context.trust_radius = factor * (scaled_x_norm > 0.0 ? scaled_x_norm : 1.0);

  context.trust_radius =
      std::clamp(context.trust_radius,
                 context.evaluation_context.options.lm.min_trust_region_radius,
                 context.evaluation_context.options.lm.max_trust_region_radius);

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

    if constexpr (std::same_as<typename Policy::LinearAlgebraPolicy,
                               PivotedHouseholderQr>) {
      detail::PivotedHouseholderQrWorkspace<M, N> &linear =
          context.workspace.linear;

      if (!linear.factor_valid) {
        if (!detail::prepare_pivoted_householder_qr(
                typename Policy::ScalingPolicy{}, context.workspace.work,
                linear,
                context.evaluation_context.options.lm
                    .rank_tolerance_multiplier)) {
          context.result.termination = TerminationReason::NumericalFailure;
          context.result.message =
              "Pivoted Householder QR factorization failed";
          return detail::finish_solver(context);
        }
        ++context.result.factorization_count;
      }
    }

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

    if constexpr (std::same_as<typename Policy::LinearAlgebraPolicy,
                               PivotedHouseholderQr>) {
      context.workspace.linear.invalidate();
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
