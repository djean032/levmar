#pragma once

#include <algorithm>
#include <cmath>
#include <concepts>
#include <functional>
#include <limits>
#include <type_traits>

#include <levmar/internal/autodiff/graph.h>

template <Index M, Index N>
using ResidualSignature = ErrorOrVoid(ConstVectorView<N> x, VectorView<M> r);

template <Index M, Index N>
using JacobianSignature = ErrorOrVoid(ConstVectorView<N> x, MatrixView<M, N> J);

template <class Residual, class XView, class RView>
concept ResidualCallableOn =
    requires(const Residual residual, XView x, RView r) {
      { residual(x, r) } -> std::same_as<ErrorOrVoid>;
    };

template <class Residual, Index M, Index N>
concept ResidualCallable =
    ResidualCallableOn<Residual, ConstVectorView<N>, VectorView<M>>;

template <class Residual, Index M, Index N>
concept AutoDiffResidualCallable =
    ResidualCallableOn<Residual, ConstVectorView<N, AdExprRef>,
                       VectorView<M, AdExprRef>>;

template <class Jacobian, class XView, class JView>
concept JacobianCallableOn = requires(Jacobian jacobian, XView x, JView J) {
  { jacobian(x, J) } -> std::same_as<ErrorOrVoid>;
};

template <class Jacobian, Index M, Index N>
concept JacobianCallable =
    JacobianCallableOn<Jacobian, ConstVectorView<N>, MatrixView<M, N>>;

template <class Jacobian, Index M, Index N>
concept OptionalJacobianCallable =
    std::same_as<Jacobian, NoJacobian> || JacobianCallable<Jacobian, M, N>;

enum class JacobianMode {
  User,
  ForwardDifference,
  CentralDifference,
  AutoDiff
};

enum class Strategy { GaussNewton, LevenbergMarquardt, TrustRegionLM, DogLeg };

enum class LinearSolver { NormalEquationsCholesky, QR, SVD };

enum class LossKind { Squared, Huber, Cauchy, SoftL1, User };

enum class ScalingMode { None, JacobianColumnNorm, User };

template <Index M, Index N, ResidualCallable<M, N> Residual,
          class Jacobian = NoJacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
struct [[nodiscard]] Problem {
  static constexpr Index residual_extent = M;
  static constexpr Index parameter_extent = N;

  Index num_residuals = (M == std::dynamic_extent ? 0 : M);
  Index num_parameters = (N == std::dynamic_extent ? 0 : N);

  Residual residual;
  Jacobian jacobian;

  Problem(Residual residual_, Jacobian jacobian_ = {})
    requires(M != std::dynamic_extent && N != std::dynamic_extent)
      : residual(residual_), jacobian(jacobian_) {}

  Problem(Index m, Index n, Residual residual_, Jacobian jacobian_ = {})
    requires(M == std::dynamic_extent || N == std::dynamic_extent)
      : num_residuals(M == std::dynamic_extent ? m : M),
        num_parameters(N == std::dynamic_extent ? n : N), residual(residual_),
        jacobian(jacobian_) {}

  bool has_user_jacobian() const {
    if constexpr (std::is_same_v<Jacobian, NoJacobian>) {
      return false;
    } else {
      return true;
    }
  }
};

template <Index M, Index N, ResidualCallable<M, N> Residual,
          class Jacobian = NoJacobian>
  requires(M != std::dynamic_extent && N != std::dynamic_extent) &&
          OptionalJacobianCallable<Jacobian, M, N>
[[nodiscard]] auto make_problem(Residual residual, Jacobian jacobian = {}) {
  return Problem<M, N, Residual, Jacobian>(residual, jacobian);
}

template <ResidualCallable<std::dynamic_extent, std::dynamic_extent> Residual,
          class Jacobian = NoJacobian>
  requires OptionalJacobianCallable<Jacobian, std::dynamic_extent,
                                    std::dynamic_extent>
[[nodiscard]] auto make_dynamic_problem(Index m, Index n, Residual residual,
                                        Jacobian jacobian = {}) {
  return Problem<std::dynamic_extent, std::dynamic_extent, Residual, Jacobian>(
      m, n, residual, jacobian);
}

template <Index N, ResidualCallable<std::dynamic_extent, N> Residual,
          class Jacobian = NoJacobian>
  requires OptionalJacobianCallable<Jacobian, std::dynamic_extent, N>
[[nodiscard]] auto make_problem_dynamic_residuals(Index m, Residual residual,
                                                  Jacobian jacobian = {}) {
  return Problem<std::dynamic_extent, N, Residual, Jacobian>(m, N, residual,
                                                             jacobian);
}

template <Index M, ResidualCallable<M, std::dynamic_extent> Residual,
          class Jacobian = NoJacobian>
  requires OptionalJacobianCallable<Jacobian, M, std::dynamic_extent>
[[nodiscard]] auto make_problem_dynamic_parameters(Index n, Residual residual,
                                                   Jacobian jacobian = {}) {
  return Problem<M, std::dynamic_extent, Residual, Jacobian>(M, n, residual,
                                                             jacobian);
}

struct LossOptions {
  LossKind kind = LossKind::Squared;

  double scale = 1.0;

  std::function<bool(double s, double &rho0, double &rho1, double &rho2)>
      user_loss;
};

struct LMOptions {
  double initial_lambda = 1e-3;
  double min_lambda = 1e-15;
  double max_lambda = 1e15;
};

struct Options {
  Strategy strategy = Strategy::LevenbergMarquardt;
  LinearSolver linear_solver = LinearSolver::NormalEquationsCholesky;
  JacobianMode jacobian_mode = JacobianMode::User;
  ScalingMode scaling = ScalingMode::None;

  LossOptions loss;

  Index max_iterations = 100;
  Index max_function_evaluations = 1000;

  double gradient_tolerance = 1e-8;
  double step_tolerance = 1e-12;
  double cost_tolerance = 1e-12;

  double finite_difference_step = 0.0;

  LMOptions lm;
};

template <Index M, Index N, ResidualCallable<M, N> Residual, class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
[[nodiscard]] inline ErrorOrVoid
validate_problem(const Problem<M, N, Residual, Jacobian> &problem,
                 const Options &options) {
  if (problem.num_residuals < 1) {
    return std::unexpected(Error{ErrorCode::InvalidProblem,
                                 "Problem must have at least one residual"});
  }

  if (problem.num_parameters < 1) {
    return std::unexpected(Error{ErrorCode::InvalidProblem,
                                 "Problem must have at least one parameter"});
  }

  if constexpr (M != std::dynamic_extent) {
    if (problem.num_residuals != M) {
      return std::unexpected(Error{ErrorCode::InvalidProblem,
                                   "Problem residual count does not "
                                   "match static residual extent"});
    }
  }

  if constexpr (N != std::dynamic_extent) {
    if (problem.num_parameters != N) {
      return std::unexpected(Error{ErrorCode::InvalidProblem,
                                   "Problem parameter count does not match "
                                   "static parameter extent"});
    }
  }

  if (options.jacobian_mode == JacobianMode::User) {
    if (!problem.has_user_jacobian()) {
      return std::unexpected(Error{ErrorCode::InvalidProblem,
                                   "JacobianMode::User requires a jacobian "
                                   "function"});
    }
  }

  if (options.jacobian_mode == JacobianMode::AutoDiff) {
    if constexpr (!AutoDiffResidualCallable<Residual, M, N>) {
      return std::unexpected(Error{
          ErrorCode::InvalidProblem,
          "JacobianMode::AutoDiff requires a scalar-generic Residual callback "
          "using ConstVectorView<N, Scalar> and VectorView<M, Scalar>"});
    }
  }

  return {};
}

inline double resolved_finite_difference_step(const Options &options) {
  if (options.finite_difference_step > 0.0) {
    return options.finite_difference_step;
  }

  switch (options.jacobian_mode) {
  case JacobianMode::ForwardDifference:
    return std::sqrt(std::numeric_limits<double>::epsilon());
  case JacobianMode::CentralDifference:
    return std::cbrt(std::numeric_limits<double>::epsilon());
  case JacobianMode::User:
    return 0.0;
  case JacobianMode::AutoDiff:
    return 0.0;
  }
  return std::sqrt(std::numeric_limits<double>::epsilon());
}

inline double finite_difference_perturbation(double xj, double rel_step) {
  return rel_step * std::max(double{1.0}, std::abs(xj));
}
