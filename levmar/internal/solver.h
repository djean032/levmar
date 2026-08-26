#pragma once

#include "levmar/internal/core.h"
#include "levmar/internal/evaluation_state.h"
#include "levmar/internal/problem.h"
#include "levmar/internal/solver_policy.h"
#include "levmar/internal/solver_workspace.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <utility>

#include <expected>
#include <levmar/internal/solver_context.h>
#include <levmar/internal/solver_validation.h>
#include <numeric>

namespace levmar::detail {

template <class Policy, Index M, Index N, ResidualCallable<M, N> Residual,
          class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
[[nodiscard]] ErrorOrVoid
initialize_solver(SolverContext<Policy, M, N, Residual, Jacobian> &context) {
  validate_static_solver_configuration<
      Policy, SolverContext<Policy, M, N, Residual, Jacobian>>();

  context.result = Result{};

  auto &evaluation = context.evaluation_context;
  const auto &problem = evaluation.problem;

  if (auto validation = validate_context(evaluation); !validation) {
    return validation;
  }

  if (auto validation = validate_runtime_options<Policy>(evaluation.options);
      !validation) {
    return validation;
  }

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
[[nodiscard]] ErrorOrVoid evaluate_current_model(
    SolverContext<Policy, M, N, Residual, Jacobian> &context) {
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

template <Index M, Index N>
[[nodiscard]] bool solve_damped_qr(LMWorkspace<M, N> &work,
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
      qr.row[j] = work.J(i, j);
    }

    double incoming_rhs = -work.r[i];
    if (!append_row_with_givens(qr, n, incoming_rhs)) {
      return false;
    }
  }

  for (Index i = 0; i < n; ++i) {
    qr.row.fill(0.0);
    qr.row[i] = sqrt_lambda;

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
  return detail::finish_solver(context);
}

} // namespace levmar
