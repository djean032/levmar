#pragma once

#include <algorithm>
#include <ranges>
#include <string_view>

#include <levmar/internal/evaluation_state.h>

template <Index M, Index N, ResidualCallable<M, N> Residual, class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
[[nodiscard]] inline ErrorOrVoid evaluate_forward_difference_jacobian(
    LMSolveContext<M, N, Residual, Jacobian> &context,
    std::string_view what = "Forward Diff Jacobian") {
  auto &result = context.result;
  const auto &options = context.options;
  auto &work = context.work;

  const double rel_step =
      resolved_finite_difference_step<ForwardDifferenceJacobian>(options);
  auto x_current = work.x_current.view();
  auto x_trial = work.x_trial.view();
  auto r = work.r.view();
  auto r_trial = work.r_trial.view();

  std::ranges::copy(x_current, x_trial.begin());

  // RESIDUAL VEC MUST BE FILLED PRIOR TO CALL
  const auto process_column = [&](Index j) -> ErrorOrVoid {
    const double xj = x_current[j];
    const double h = finite_difference_perturbation(xj, rel_step);
    x_trial[j] = xj + h;
    if (auto residual_result =
            evaluate_residual_at(context, x_trial, r_trial, what);
        !residual_result) {
      return residual_result;
    }

    if constexpr (M != std::dynamic_extent) {
      for (Index i = 0; i < M; ++i) {
        work.J(i, j) = (r_trial[i] - r[i]) / h;
      }
    } else {
      for (Index i = 0; i < work.m; ++i) {
        work.J(i, j) = (r_trial[i] - r[i]) / h;
      }
    }

    x_trial[j] = xj;
    return {};
  };

  if constexpr (N != std::dynamic_extent) {
    for (Index j = 0; j < N; ++j) {
      if (auto column_result = process_column(j); !column_result) {
        return column_result;
      }
    }
  } else {
    for (Index j = 0; j < work.n; ++j) {
      if (auto column_result = process_column(j); !column_result) {
        return column_result;
      }
    }
  }

  ++result.jacobian_evaluations;
  return {};
}

template <Index M, Index N, ResidualCallable<M, N> Residual, class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
[[nodiscard]] inline ErrorOrVoid evaluate_central_difference_jacobian(
    LMSolveContext<M, N, Residual, Jacobian> &context,
    std::string_view what = "Central Diff Jacobian") {
  auto &result = context.result;
  const auto &options = context.options;
  auto &work = context.work;

  const double rel_step =
      resolved_finite_difference_step<CentralDifferenceJacobian>(options);
  auto x_current = work.x_current.view();
  auto x_trial = work.x_trial.view();
  auto r_trial = work.r_trial.view();
  auto r_trial_minus = work.r_trial_minus.view();

  std::ranges::copy(x_current, x_trial.begin());

  // RESIDUAL VEC MUST BE FILLED PRIOR TO CALL
  const auto process_column = [&](Index j) -> ErrorOrVoid {
    const double xj = x_current[j];
    const double h = finite_difference_perturbation(xj, rel_step);
    x_trial[j] = xj + h;
    if (auto residual_result =
            evaluate_residual_at(context, x_trial, r_trial, what);
        !residual_result) {
      return residual_result;
    }
    x_trial[j] = xj - h;
    if (auto residual_result =
            evaluate_residual_at(context, x_trial, r_trial_minus, what);
        !residual_result) {
      return residual_result;
    }

    if constexpr (M != std::dynamic_extent) {
      for (Index i = 0; i < M; ++i) {
        work.J(i, j) = (r_trial[i] - r_trial_minus[i]) / (2.0 * h);
      }
    } else {
      for (Index i = 0; i < work.m; ++i) {
        work.J(i, j) = (r_trial[i] - r_trial_minus[i]) / (2.0 * h);
      }
    }

    x_trial[j] = xj;
    return {};
  };

  if constexpr (N != std::dynamic_extent) {
    for (Index j = 0; j < N; ++j) {
      if (auto column_result = process_column(j); !column_result) {
        return column_result;
      }
    }
  } else {
    for (Index j = 0; j < work.n; ++j) {
      if (auto column_result = process_column(j); !column_result) {
        return column_result;
      }
    }
  }

  ++result.jacobian_evaluations;
  return {};
}
