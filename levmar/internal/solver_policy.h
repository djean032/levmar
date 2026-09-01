#pragma once

#include <concepts>

namespace levmar {

struct UserJacobian {};
struct ForwardDifferenceJacobian {};
struct CentralDifferenceJacobian {};
struct AutoDiffJacobian {};

struct LevenbergMarquardt {};

struct DampedQr {};
struct PivotedHouseholderQr {};

template <class LinearAlgebra>
inline constexpr bool kSupportedLinearAlgebra =
    std::same_as<LinearAlgebra, DampedQr> ||
    std::same_as<LinearAlgebra, PivotedHouseholderQr>;

template <class LinearAlgebra>
inline constexpr bool kRequiresTallQr =
    std::same_as<LinearAlgebra, DampedQr> ||
    std::same_as<LinearAlgebra, PivotedHouseholderQr>;

struct SquaredLoss {};

struct NoScaling {};

struct JacobianColumnScaling {};

template <class ScalingPolicy>
inline constexpr bool kUsesJacobianColumnScaling =
    std::same_as<ScalingPolicy, JacobianColumnScaling>;

template <class Jacobian, class Strategy, class LinearAlgebra, class Loss,
          class Scaling>
struct SolverPolicy {
  using JacobianPolicy = Jacobian;
  using StrategyPolicy = Strategy;
  using LinearAlgebraPolicy = LinearAlgebra;
  using LossPolicy = Loss;
  using ScalingPolicy = Scaling;
};

using DefaultSolverPolicy = SolverPolicy<UserJacobian, LevenbergMarquardt,
                                         DampedQr, SquaredLoss, NoScaling>;

} // namespace levmar
