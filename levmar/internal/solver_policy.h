#pragma once

namespace levmar {

struct UserJacobian {};
struct ForwardDifferenceJacobian {};
struct CentralDifferenceJacobian {};
struct AutoDiffJacobian {};

struct LevenbergMarquardt {};

struct DampedQr {};

struct SquaredLoss {};

struct NoScaling {};

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
