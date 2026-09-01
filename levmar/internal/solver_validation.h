#pragma once

#include "levmar/internal/autodiff/dual_forward.h"
#include "levmar/internal/core.h"
#include "levmar/internal/problem.h"
#include <cmath>
#include <concepts>

#include <levmar/internal/autodiff/jacobian_policy.h>
#include <levmar/internal/evaluation_state.h>
#include <levmar/internal/solver_policy.h>

namespace levmar::detail {

template <class JacobianPolicy>
inline constexpr bool kInitialJacobianPolicy =
    std::same_as<JacobianPolicy, UserJacobian> ||
    std::same_as<JacobianPolicy, ForwardDifferenceJacobian> ||
    std::same_as<JacobianPolicy, CentralDifferenceJacobian> ||
    std::same_as<JacobianPolicy, AutoDiffJacobian>;

template <class ScalingPolicy>
inline constexpr bool kInitialScalingPolicy =
    std::same_as<ScalingPolicy, NoScaling> ||
    kUsesJacobianColumnScaling<ScalingPolicy>;

template <class Policy, class Context>
consteval bool has_supported_autodiff_callback() {
  using Problem = typename Context::ProblemType;
  using Residual = typename Problem::ResidualType;

  constexpr Index M = Problem::residual_extent;
  constexpr Index N = Problem::parameter_extent;

  if constexpr (!std::same_as<typename Policy::JacobianPolicy,
                              AutoDiffJacobian>) {
    return true;
  } else if constexpr (!AutoDiffResidualCallable<Residual, M, N>) {
    return false;
  } else if constexpr (kUsesDirectDualAutoDiff<N>) {
    return DualResidualCallable<Residual, M, N>;
  } else {
    return true;
  }
}

template <class Policy, class Context>
inline constexpr bool kStaticSolverConfigurationValid = [] {
  using Problem = typename Context::ProblemType;
  using Jacobian = typename Problem::JacobianType;

  constexpr Index M = Problem::residual_extent;
  constexpr Index N = Problem::parameter_extent;

  using JacobianPolicy = typename Policy::JacobianPolicy;
  using StrategyPolicy = typename Policy::StrategyPolicy;
  using LinearAlgebraPolicy = typename Policy::LinearAlgebraPolicy;
  using LossPolicy = typename Policy::LossPolicy;
  using ScalingPolicy = typename Policy::ScalingPolicy;

  constexpr bool user_jacobian_is_valid =
      !std::same_as<JacobianPolicy, UserJacobian> ||
      !std::same_as<Jacobian, NoJacobian>;

  constexpr bool shape_is_valid = !kRequiresTallQr<LinearAlgebraPolicy> ||
                                  M == std::dynamic_extent ||
                                  N == std::dynamic_extent || M >= N;

  return kInitialJacobianPolicy<JacobianPolicy> &&
         std::same_as<StrategyPolicy, LevenbergMarquardt> &&
         kSupportedLinearAlgebra<LinearAlgebraPolicy> &&
         std::same_as<LossPolicy, SquaredLoss> &&
         kInitialScalingPolicy<ScalingPolicy> && user_jacobian_is_valid &&
         has_supported_autodiff_callback<Policy, Context>() && shape_is_valid;
}();

template <class Policy, class Context>
consteval void validate_static_solver_configuration() {
  using Problem = typename Context::ProblemType;
  using Residual = typename Problem::ResidualType;
  using Jacobian = typename Problem::JacobianType;

  constexpr Index M = Problem::residual_extent;
  constexpr Index N = Problem::parameter_extent;

  using JacobianPolicy = typename Policy::JacobianPolicy;
  using StrategyPolicy = typename Policy::StrategyPolicy;
  using LinearAlgebraPolicy = typename Policy::LinearAlgebraPolicy;
  using LossPolicy = typename Policy::LossPolicy;
  using ScalingPolicy = typename Policy::ScalingPolicy;

  static_assert(kInitialJacobianPolicy<JacobianPolicy>,
                "Unsupported Jacobian policy");
  static_assert(std::same_as<StrategyPolicy, LevenbergMarquardt>,
                "Only LevenbergMarquardt is currently implemented");
  static_assert(kSupportedLinearAlgebra<LinearAlgebraPolicy>,
                "Unsupported linear-algebra policy");
  static_assert(std::same_as<LossPolicy, SquaredLoss>,
                "Only SquaredLoss is currently implemented");
  static_assert(kInitialScalingPolicy<ScalingPolicy>,
                "Unsupported Scaling Policy");

  if constexpr (std::same_as<JacobianPolicy, UserJacobian>) {
    static_assert(!std::same_as<Jacobian, NoJacobian>,
                  "UserJacobian requires a Jacobian callback");
  }

  if constexpr (std::same_as<JacobianPolicy, AutoDiffJacobian>) {
    static_assert(AutoDiffResidualCallable<Residual, M, N>,
                  "AutoDiffJacobian requires a scalar-generic residual "
                  "compatible with graph AutoDiff");

    if constexpr (kUsesDirectDualAutoDiff<N>) {
      static_assert(DualResidualCallable<Residual, M, N>,
                    "AutoDiffJacobian for fixed-size parameters requires a "
                    "residual compatible with Dual<N>");
    }
  }

  if constexpr (kRequiresTallQr<LinearAlgebraPolicy> &&
                M != std::dynamic_extent && N != std::dynamic_extent) {
    static_assert(M >= N,
                  "QR requires residuals >= parameters for static problems");
  }
}

template <class Policy>
ErrorOrVoid validate_runtime_options(const Options &options) {
  using JacobianPolicy = typename Policy::JacobianPolicy;

  if (!std::isfinite(options.gradient_tolerance) ||
      options.gradient_tolerance < 0.0) {
    return std::unexpected(
        Error{ErrorCode::InvalidProblem,
              "gradient_tolerance must be finite and non-negative"});
  }

  if constexpr (std::same_as<JacobianPolicy, ForwardDifferenceJacobian> ||
                std::same_as<JacobianPolicy, CentralDifferenceJacobian>) {
    if (!std::isfinite(options.finite_difference_step) ||
        options.finite_difference_step < 0.0) {
      return std::unexpected(
          Error{ErrorCode::InvalidProblem,
                "finite_difference_step must be finite and non-negative"});
    }
  }

  if (!std::isfinite(options.step_tolerance) || options.step_tolerance < 0.0) {
    return std::unexpected(
        Error{ErrorCode::InvalidProblem,
              "step_tolerance must be non-negative and finite"});
  }

  if (!std::isfinite(options.cost_tolerance) || options.cost_tolerance < 0.0) {
    return std::unexpected(
        Error{ErrorCode::InvalidProblem,
              "cost_tolerance must be non-negative and finite"});
  }

  if (!std::isfinite(options.relative_cost_tolerance) ||
      options.relative_cost_tolerance < 0.0) {
    return std::unexpected(
        Error{ErrorCode::InvalidProblem,
              "relative_cost_tolerance must be non-negative and finite"});
  }

  if (options.max_function_evaluations == 0) {
    return std::unexpected(Error{ErrorCode::InvalidProblem,
                                 "max_function_evaluations must be positive"});
  }

  if (!std::isfinite(options.lm.min_lambda) ||
      !std::isfinite(options.lm.max_lambda) || options.lm.min_lambda <= 0.0 ||
      options.lm.max_lambda <= 0.0 ||
      options.lm.min_lambda > options.lm.max_lambda) {
    return std::unexpected(
        Error{ErrorCode::InvalidProblem, "Invalid lambda configuration"});
  }

  if (!std::isfinite(options.lm.initial_trust_region_factor) ||
      options.lm.initial_trust_region_factor <= 0.0) {
    return std::unexpected(
        Error{ErrorCode::InvalidProblem,
              "initial_trust_region_factor must be finite and positive"});
  }

  if (!std::isfinite(options.lm.initial_trust_region_radius) ||
      !std::isfinite(options.lm.min_trust_region_radius) ||
      !std::isfinite(options.lm.max_trust_region_radius) ||
      options.lm.min_trust_region_radius <= 0.0 ||
      options.lm.min_trust_region_radius >
          options.lm.initial_trust_region_radius ||
      options.lm.initial_trust_region_radius >
          options.lm.max_trust_region_radius) {
    return std::unexpected(Error{ErrorCode::InvalidProblem,
                                 "Invalid trust-region radius configuration"});
  }

  if (!std::isfinite(options.lm.rank_tolerance_multiplier) ||
      options.lm.rank_tolerance_multiplier <= 0.0) {
    return std::unexpected(
        Error{ErrorCode::InvalidProblem,
              "rank_tolerance_multiplier must be finite and positive"});
  }

  return {};
}

} // namespace levmar::detail
