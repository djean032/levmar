#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <mdspan>
#include <span>
#include <string>
#include <vector>

using index = std::size_t;

using VectorView = std::span<double>;
using ConstVectorView = std::span<const double>;

using MatrixView =
    std::mdspan<double, std::dextents<index, 2>, std::layout_left>;
using ConstMatrixView =
    std::mdspan<const double, std::dextents<index, 2>, std::layout_left>;

enum class Status {
  Success,
  MaxIterations,
  SmallStep,
  SmallGradient,
  SmallCostReduction,
  UserFunctionError,
  NumericalFailure,
  InvalidProblem
};

enum class JacobianMode { User, ForwardDifference, CentralDifference };

enum class Strategy { GaussNewton, LevenbergMarquardt, TrustRegionLM, DogLeg };

enum class LinearSolver { NormalEquationsCholesky, QR, SVD };

enum class LossKind { Squared, Huber, Cauchy, SoftL1, User };

enum class ScalingMode { None, JacobianColumnNorm, User };

using ResidualFunction = std::function<Status(ConstVectorView x, VectorView r)>;
using JacobianFunction = std::function<Status(ConstVectorView x, MatrixView J)>;

struct Problem {
  index num_residuals = 0;
  index num_parameters = 0;

  ResidualFunction residual;
  JacobianFunction jacobian;

  bool has_user_jacobian() const { return static_cast<bool>(jacobian); }

  bool valid() const {
    return num_residuals > 0 && num_parameters > 0 &&
           static_cast<bool>(residual);
  }
};

struct LossOptions {
  LossKind kind = LossKind::Squared;

  double scale = 1.0;

  std::function<Status(double s, double &rho0, double &rho1, double &rho2)>
      user_loss;
};

struct Options {
  Strategy strategy = Strategy::LevenbergMarquardt;
  LinearSolver linear_solver = LinearSolver::NormalEquationsCholesky;
  JacobianMode jacobian_mode = JacobianMode::User;
  ScalingMode scaling = ScalingMode::None;

  LossOptions loss;

  index max_iterations = 100;
  index max_function_evaluations = 1000;

  double initial_lambda = 1e-3;
  double min_lambda = 1e-15;
  double max_lambda = 1e15;

  double gradient_tolerance = 1e-8;
  double step_tolerance = 1e-12;
  double cost_tolerance = 1e-12;

  double finite_difference_step = 0.0;

  bool verbose = false;
};

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
  }
  return std::sqrt(std::numeric_limits<double>::epsilon());
}

inline double finite_difference_perturbation(double xj, double rel_step) {
  return rel_step * std::max(double{1.0}, std::abs(xj));
}

struct Result {
  Status status = Status::InvalidProblem;

  index iterations = 0;
  index function_evalutions = 0;
  index jacobian_evaluations = 0;
  index linear_solves = 0;

  double initial_cost = std::numeric_limits<double>::quiet_NaN();
  double final_cost = std::numeric_limits<double>::quiet_NaN();

  double gradient_inf_norm = std::numeric_limits<double>::quiet_NaN();
  double step_norm = std::numeric_limits<double>::quiet_NaN();

  double lambda = std::numeric_limits<double>::quiet_NaN();

  std::string message;
};

struct DenseMatrix {
  index rows = 0;
  index cols = 0;
  std::vector<double> storage;

  DenseMatrix() = default;

  DenseMatrix(index rows_, index cols_)
      : rows(rows_), cols(cols_), storage(rows_ * cols_, 0.0) {}

  void resize(index rows_, index cols_) {
    rows = rows_;
    cols = cols_;
  }

  MatrixView view() { return MatrixView(storage.data(), rows, cols); }

  ConstMatrixView view() const {
    return ConstMatrixView(storage.data(), rows, cols);
  }

  double *data() { return storage.data(); }

  const double *data() const { return storage.data(); }

  index leading_dim() const { return rows; }

  double &operator()(index i, index j) { return view()[i, j]; }

  double operator()(index i, index j) const { return view()[i, j]; }
};

struct Workspace {
  index m = 0;
  index n = 0;

  std::vector<double> x_trial;

  std::vector<double> r_trial;
  std::vector<double> r;

  DenseMatrix J;
  DenseMatrix A;

  std::vector<double> g;
  std::vector<double> step;

  std::vector<double> scale;
  std::vector<double> weights;

  Workspace() = default;

  Workspace(index m_, index n_) { resize(m_, n_); }

  void resize(index m_, index n_) {
    m = m_;
    n = n_;

    x_trial.assign(n, 0.0);

    r.assign(m, 0.0);
    r_trial.assign(m, 0.0);

    J.resize(m, n);
    A.resize(m, n);

    g.assign(n, 0.0);
    step.assign(n, 0.0);

    scale.assign(n, 1.0);
    weights.assign(m, 1.0);
  }

  VectorView x_trial_view() { return VectorView(x_trial); }
  VectorView r_view() { return VectorView(r); }
  VectorView r_trial_view() { return VectorView(r_trial); }
  VectorView g_view() { return VectorView(g); }
  VectorView step_view() { return VectorView(step); }
};

struct SolveContext {
  const Problem *problem = nullptr;
  const Options *options = nullptr;
  Result *result = nullptr;
  Workspace *work = nullptr;

  ConstVectorView x;
};
