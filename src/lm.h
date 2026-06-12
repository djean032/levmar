#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <mdspan>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using Index = std::size_t;

using VectorView = std::span<double>;
using ConstVectorView = std::span<const double>;

using MatrixView =
    std::mdspan<double, std::dextents<Index, 2>, std::layout_left>;
using ConstMatrixView =
    std::mdspan<const double, std::dextents<Index, 2>, std::layout_left>;

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

using ResidualFunction = std::function<bool(ConstVectorView x, VectorView r)>;
using JacobianFunction = std::function<bool(ConstVectorView x, MatrixView J)>;

struct Problem {
  Index num_residuals = 0;
  Index num_parameters = 0;

  ResidualFunction residual;
  JacobianFunction jacobian;

  bool has_user_jacobian() const { return static_cast<bool>(jacobian); }
};

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

inline Status validate_problem(const Problem &problem, const Options &options,
                               std::string &message) {
  if (problem.num_residuals == 0) {
    message = "Problem must have at least one residual";
    return Status::InvalidProblem;
  }

  if (problem.num_parameters == 0) {
    message = "Problem must have at least one parameter";
    return Status::InvalidProblem;
  }

  if (!problem.residual) {
    message = "Problem requires a residual function";
    return Status::InvalidProblem;
  }

  if (options.jacobian_mode == JacobianMode::User &&
      !problem.has_user_jacobian()) {
    message = "JacobianMode::User requires a jacobian function";
    return Status::InvalidProblem;
  }

  return Status::Success;
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
  }
  return std::sqrt(std::numeric_limits<double>::epsilon());
}

inline double finite_difference_perturbation(double xj, double rel_step) {
  return rel_step * std::max(double{1.0}, std::abs(xj));
}

struct Result {
  Status status = Status::InvalidProblem;

  Index iterations = 0;
  Index function_evaluations = 0;
  Index jacobian_evaluations = 0;
  Index linear_solves = 0;

  double initial_cost = std::numeric_limits<double>::quiet_NaN();
  double final_cost = std::numeric_limits<double>::quiet_NaN();

  double gradient_inf_norm = std::numeric_limits<double>::quiet_NaN();
  double step_norm = std::numeric_limits<double>::quiet_NaN();

  double lambda = std::numeric_limits<double>::quiet_NaN();

  std::string message;
};

struct DenseMatrix {
  Index rows = 0;
  Index cols = 0;
  std::vector<double> storage;

  DenseMatrix() = default;

  DenseMatrix(Index rows_, Index cols_)
      : rows(rows_), cols(cols_), storage(rows_ * cols_, 0.0) {}

  void resize(Index rows_, Index cols_) {
    rows = rows_;
    cols = cols_;
    storage.assign(rows * cols, 0.0);
  }

  MatrixView view() { return MatrixView(storage.data(), rows, cols); }

  ConstMatrixView view() const {
    return ConstMatrixView(storage.data(), rows, cols);
  }

  double *data() { return storage.data(); }

  const double *data() const { return storage.data(); }

  Index leading_dim() const { return rows; }

  double &operator()(Index i, Index j) { return view()[i, j]; }

  double operator()(Index i, Index j) const { return view()[i, j]; }
};

struct LMWorkspace {
  Index m = 0;
  Index n = 0;

  std::vector<double> x_trial;

  std::vector<double> r_trial;
  std::vector<double> r_trial_minus;
  std::vector<double> r;

  DenseMatrix J;
  DenseMatrix JTJ;

  std::vector<double> g;
  std::vector<double> step;

  std::vector<double> scale;
  std::vector<double> weights;

  LMWorkspace() = default;

  LMWorkspace(Index m_, Index n_) { resize(m_, n_); }

  void resize(Index m_, Index n_) {
    m = m_;
    n = n_;

    x_trial.assign(n, 0.0);

    r.assign(m, 0.0);
    r_trial.assign(m, 0.0);
    r_trial_minus.assign(m, 0.0);

    J.resize(m, n);
    JTJ.resize(n, n);

    g.assign(n, 0.0);
    step.assign(n, 0.0);

    scale.assign(n, 1.0);
    weights.assign(m, 1.0);
  }

  VectorView x_trial_view() { return VectorView(x_trial); }
  VectorView r_view() { return VectorView(r); }
  VectorView r_trial_view() { return VectorView(r_trial); }
  VectorView r_trial_minus_view() { return VectorView(r_trial_minus); }
  VectorView g_view() { return VectorView(g); }
  VectorView step_view() { return VectorView(step); }
};

struct LMSolveContext {
  const Problem *problem = nullptr;
  const Options *options = nullptr;
  Result *result = nullptr;
  LMWorkspace *work = nullptr;
  ConstVectorView x;
};

inline bool
evaluate_residual_at(LMSolveContext &context, ConstVectorView x, VectorView r,
                     std::string_view what = "Residual Evaluation") {
  Result &result = *context.result;
  if (!context.problem->residual(x, r)) {
    result.status = Status::UserFunctionError;
    result.message = std::string(what) + " failed";
    return false;
  }
  ++result.function_evaluations;
  return true;
}

inline bool evaluate_residual(LMSolveContext &context,
                              std::string_view what = "Residual Evaluation") {
  LMWorkspace &work = *context.work;
  return evaluate_residual_at(context, context.x, work.r_view(), what);
}

inline bool evaluate_forward_difference_jacobian(
    LMSolveContext &context, std::string_view what = "Forward Diff Jacobian") {
  const Problem &problem = *context.problem;
  Result &result = *context.result;
  const Options &options = *context.options;
  LMWorkspace &work = *context.work;

  const Index row = work.m;
  const Index col = work.n;
  const double rel_step = resolved_finite_difference_step(options);

  std::ranges::copy(context.x, work.x_trial.begin());

  // RESIDUAL VEC MUST BE FILLED PRIOR TO CALL
  for (Index j = 0; j < col; ++j) {
    const double xj = context.x[j];
    const double h = finite_difference_perturbation(xj, rel_step);
    work.x_trial[j] = xj + h;
    if (!evaluate_residual_at(context, work.x_trial_view(),
                              work.r_trial_view(), what)) {
      return false;
    }
    for (Index i = 0; i < row; ++i) {
      work.J(i, j) = (work.r_trial[i] - work.r[i]) / h;
    }
    work.x_trial[j] = xj;
  }
  ++result.jacobian_evaluations;
  return true;
}

inline bool evaluate_central_difference_jacobian(
    LMSolveContext &context, std::string_view what = "Central Diff Jacobian") {
  Result &result = *context.result;
  const Options &options = *context.options;
  LMWorkspace &work = *context.work;

  const Index row = work.m;
  const Index col = work.n;
  const double rel_step = resolved_finite_difference_step(options);

  std::ranges::copy(context.x, work.x_trial.begin());

  // RESIDUAL VEC MUST BE FILLED PRIOR TO CALL
  for (Index j = 0; j < col; ++j) {
    const double xj = context.x[j];
    const double h = finite_difference_perturbation(xj, rel_step);
    work.x_trial[j] = xj + h;
    if (!evaluate_residual_at(context, work.x_trial_view(),
                              work.r_trial_view(), what)) {
      return false;
    }
    work.x_trial[j] = xj - h;
    if (!evaluate_residual_at(context, work.x_trial_view(),
                              work.r_trial_minus_view(), what)) {
      return false;
    }
    for (Index i = 0; i < row; ++i) {
      work.J(i, j) =
          (work.r_trial[i] - work.r_trial_minus[i]) / (2.0 * h);
    }
    work.x_trial[j] = xj;
  }
  ++result.jacobian_evaluations;
  return true;
}

inline bool evaluate_jacobian(LMSolveContext &context,
                              std::string_view what = "Jacobian Evaluation") {
  const Problem &problem = *context.problem;
  Result &result = *context.result;
  const Options &options = *context.options;
  LMWorkspace &work = *context.work;
  switch (options.jacobian_mode) {
  case JacobianMode::User:
    if (!problem.has_user_jacobian() ||
        !problem.jacobian(context.x, work.J.view())) {
      result.status = Status::UserFunctionError;
      result.message = std::string(what) + " failed";
      return false;
    }
    ++result.jacobian_evaluations;
    return true;
  case JacobianMode::ForwardDifference:
    return evaluate_forward_difference_jacobian(context, what);
  case JacobianMode::CentralDifference:
    return evaluate_central_difference_jacobian(context, what);
  }

  result.status = Status::InvalidProblem;
  result.message = std::string(what) + " failed.";
  return false;
}
