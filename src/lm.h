#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <mdspan>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

using Index = std::size_t;

struct NoJacobian {};

// Views for vectors and matrices using span and mdspan
template <Index N> using VectorView = std::span<double, N>;

template <Index N> using ConstVectorView = std::span<const double, N>;

template <Index M, Index N>
using MatrixView =
    std::mdspan<double, std::extents<Index, M, N>, std::layout_left>;

template <Index M, Index N>
using ConstMatrixView =
    std::mdspan<const double, std::extents<Index, M, N>, std::layout_left>;

// Vector storage: both dynamic and static
/*
template <class Storage> struct vector_storage_traits {
  static constexpr Index extent = Storage::extent;
  static constexpr bool dynamic_extent = extent == std::dynamic_extent;
  static constexpr bool static_extent = !dynamic_extent;
};
*/

template <Index Extent> struct VectorStorage {
  std::array<double, Extent> storage{};

  static constexpr Index extent = Extent;
  constexpr Index size() const { return Extent; }
  double *data() { return storage.data(); }
  const double *data() const { return storage.data(); }

  VectorView<Extent> view() {
    return VectorView<Extent>(storage.data(), Extent);
  }
  ConstVectorView<Extent> view() const {
    return ConstVectorView<Extent>(storage.data(), Extent);
  }

  void fill(double value) { storage.fill(value); }
};

template <> struct VectorStorage<std::dynamic_extent> {
  std::vector<double> storage;

  static constexpr Index extent = std::dynamic_extent;
  Index size() const { return storage.size(); }

  double *data() { return storage.data(); }
  const double *data() const { return storage.data(); }

  VectorView<std::dynamic_extent> view() {
    return VectorView<std::dynamic_extent>(storage.data(), storage.size());
  }
  ConstVectorView<std::dynamic_extent> view() const {
    return ConstVectorView<std::dynamic_extent>(storage.data(), storage.size());
  }

  void resize(Index n) { storage.resize(n); }
  void assign(Index n, double value) { storage.assign(n, value); }
  void fill(double value) { std::ranges::fill(storage, value); }
};

// Matrix storage: both dynamic and static
/*
template <class Storage> struct matrix_storage_traits {
  static constexpr Index rows_extent = Storage::rows_extent;
  static constexpr Index cols_extent = Storage::cols_extent;

  static constexpr bool dynamic_rows = rows_extent == std::dynamic_extent;
  static constexpr bool dynamic_cols = cols_extent == std::dynamic_extent;

  static constexpr bool fully_static = !dynamic_rows && !dynamic_cols;
  static constexpr bool fully_dynamic = dynamic_rows && dynamic_cols;
  static constexpr bool mixed = dynamic_rows != dynamic_cols;
};
*/

template <Index Rows, Index Cols> struct MatrixStorage {
  std::array<double, Rows * Cols> storage{};

  static constexpr Index rows_extent = Rows;
  static constexpr Index cols_extent = Cols;

  constexpr Index rows() const { return Rows; }
  constexpr Index cols() const { return Cols; }
  constexpr Index leading_dim() const { return Rows; }

  double *data() { return storage.data(); }
  const double *data() const { return storage.data(); }

  MatrixView<Rows, Cols> view() {
    return MatrixView<Rows, Cols>(storage.data());
  }
  ConstMatrixView<Rows, Cols> view() const {
    return ConstMatrixView<Rows, Cols>(storage.data());
  }

  double &operator()(Index i, Index j) { return view()[i, j]; }

  double operator()(Index i, Index j) const { return view()[i, j]; }

  void fill(double value) { storage.fill(value); }
};

template <Index Cols> struct MatrixStorage<std::dynamic_extent, Cols> {
  Index rows_ = 0;
  std::vector<double> storage;

  static constexpr Index rows_extent = std::dynamic_extent;
  static constexpr Index cols_extent = Cols;

  Index rows() const { return rows_; }
  constexpr Index cols() const { return Cols; }
  Index leading_dim() const { return rows_; }

  double *data() { return storage.data(); }
  const double *data() const { return storage.data(); }

  MatrixView<std::dynamic_extent, Cols> view() {
    return MatrixView<std::dynamic_extent, Cols>(storage.data(), rows_);
  }

  ConstMatrixView<std::dynamic_extent, Cols> view() const {
    return ConstMatrixView<std::dynamic_extent, Cols>(storage.data(), rows_);
  }

  double &operator()(Index i, Index j) { return view()[i, j]; }

  double operator()(Index i, Index j) const { return view()[i, j]; }

  void resize(Index rows) {
    rows_ = rows;
    storage.assign(rows_ * Cols, 0.0);
  }

  void fill(double value) { std::ranges::fill(storage, value); }
};

template <Index Rows> struct MatrixStorage<Rows, std::dynamic_extent> {
  Index cols_ = 0;
  std::vector<double> storage;

  static constexpr Index rows_extent = Rows;
  static constexpr Index cols_extent = std::dynamic_extent;

  constexpr Index rows() const { return Rows; }
  Index cols() const { return cols_; }
  constexpr Index leading_dim() const { return Rows; }

  double *data() { return storage.data(); }
  const double *data() const { return storage.data(); }

  MatrixView<Rows, std::dynamic_extent> view() {
    return MatrixView<Rows, std::dynamic_extent>(storage.data(), cols_);
  }

  ConstMatrixView<Rows, std::dynamic_extent> view() const {
    return ConstMatrixView<Rows, std::dynamic_extent>(storage.data(), cols_);
  }

  double &operator()(Index i, Index j) { return view()[i, j]; }

  double operator()(Index i, Index j) const { return view()[i, j]; }

  void resize(Index cols) {
    cols_ = cols;
    storage.assign(Rows * cols_, 0.0);
  }

  void fill(double value) { std::ranges::fill(storage, value); }
};

template <> struct MatrixStorage<std::dynamic_extent, std::dynamic_extent> {
  Index rows_ = 0;
  Index cols_ = 0;
  std::vector<double> storage{};

  static constexpr Index rows_extent = std::dynamic_extent;
  static constexpr Index cols_extent = std::dynamic_extent;

  Index rows() const { return rows_; }
  Index cols() const { return cols_; }
  Index leading_dim() const { return rows_; }

  double *data() { return storage.data(); }
  const double *data() const { return storage.data(); }

  MatrixView<std::dynamic_extent, std::dynamic_extent> view() {
    return MatrixView<std::dynamic_extent, std::dynamic_extent>(storage.data(),
                                                                rows_, cols_);
  }

  ConstMatrixView<std::dynamic_extent, std::dynamic_extent> view() const {
    return ConstMatrixView<std::dynamic_extent, std::dynamic_extent>(
        storage.data(), rows_, cols_);
  }

  double &operator()(Index i, Index j) { return view()[i, j]; }

  double operator()(Index i, Index j) const { return view()[i, j]; }

  void resize(Index rows, Index cols) {
    cols_ = cols;
    rows_ = rows;
    storage.assign(rows_ * cols_, 0.0);
  }

  void fill(double value) { std::ranges::fill(storage, value); }
};

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

// TODO: CHANGE THIS
/*
using ResidualFunction = std::function<bool(ConstVectorView x, VectorView r)>;
using JacobianFunction = std::function<bool(ConstVectorView x, MatrixView J)>;
*/

template <Index M, Index N, class Residual, class Jacobian = NoJacobian>
struct Problem {
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

template <Index M, Index N, class Residual, class Jacobian = NoJacobian>
auto make_problem(Residual residual, Jacobian jacobian = {}) {
  static_assert(M != std::dynamic_extent && N != std::dynamic_extent,
                "make_problem requires static residual and parameter extents");
  return Problem<M, N, Residual, Jacobian>(residual, jacobian);
}

template <class Residual, class Jacobian = NoJacobian>
auto make_dynamic_problem(Index m, Index n, Residual residual,
                          Jacobian jacobian = {}) {
  return Problem<std::dynamic_extent, std::dynamic_extent, Residual, Jacobian>(
      m, n, residual, jacobian);
}

template <Index N, class Residual, class Jacobian = NoJacobian>
auto make_problem_dynamic_residuals(Index m, Residual residual,
                                    Jacobian jacobian = {}) {
  return Problem<std::dynamic_extent, N, Residual, Jacobian>(m, N, residual,
                                                             jacobian);
}

template <Index M, class Residual, class Jacobian = NoJacobian>
auto make_problem_dynamic_parameters(Index n, Residual residual,
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

/*
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
*/

template <Index M, Index N> struct LMWorkspace {
  static constexpr Index residual_extent = M;
  static constexpr Index parameter_extent = N;

  Index m = (M == std::dynamic_extent ? 0 : M);
  Index n = (N == std::dynamic_extent ? 0 : N);

  VectorStorage<N> x_current;
  VectorStorage<N> x_trial;

  VectorStorage<M> r_trial;
  VectorStorage<M> r_trial_minus;
  VectorStorage<M> r;

  MatrixStorage<M, N> J;
  MatrixStorage<N, N> JTJ;

  VectorStorage<N> g;
  VectorStorage<N> step;

  VectorStorage<N> scale;
  VectorStorage<M> weights;

  LMWorkspace() = default;

  LMWorkspace(Index m_runtime, Index n_runtime) {
    resize(m_runtime, n_runtime);
  }

  void resize(Index m_runtime, Index n_runtime) {
    if constexpr (M == std::dynamic_extent && N == std::dynamic_extent) {
      m = m_runtime;
      n = n_runtime;
      J.resize(m, n);
      JTJ.resize(n, n);
      r.resize(m);
      r_trial.resize(m);
      r_trial_minus.resize(m);
      weights.resize(m);
      x_current.resize(n);
      x_trial.resize(n);
      g.resize(n);
      step.resize(n);
      scale.resize(n);
    } else if constexpr (M == std::dynamic_extent) {
      m = m_runtime;
      n = N;
      J.resize(m);
      r.resize(m);
      r_trial.resize(m);
      r_trial_minus.resize(m);
      weights.resize(m);
    } else if constexpr (N == std::dynamic_extent) {
      n = n_runtime;
      m = M;
      J.resize(n);
      JTJ.resize(n, n);
      x_current.resize(n);
      x_trial.resize(n);
      g.resize(n);
      step.resize(n);
      scale.resize(n);
    } else {
      n = N;
      m = M;
    }

    scale.fill(1.0);
    weights.fill(1.0);
  }
};

template <class ProblemT> struct LMSolveContext {
  using Workspace =
      LMWorkspace<ProblemT::residual_extent, ProblemT::parameter_extent>;
  const ProblemT &problem;
  const Options &options;
  Result &result;
  Workspace &work;
  ConstVectorView<ProblemT::parameter_extent> x;

  LMSolveContext(const ProblemT &problem_, const Options &options_,
                 Result &result_, Workspace &work_,
                 ConstVectorView<ProblemT::parameter_extent> x_)
      : problem(problem_), options(options_), result(result_), work(work_),
        x(x_) {}

  LMSolveContext(const ProblemT &problem_, const Options &options_,
                 Result &result_, Workspace &work_,
                 const std::array<double, ProblemT::parameter_extent> &x_)
    requires(ProblemT::parameter_extent != std::dynamic_extent)
      : problem(problem_), options(options_), result(result_), work(work_),
        x(x_.data(), x_.size()) {}

  LMSolveContext(const ProblemT &problem_, const Options &options_,
                 Result &result_, Workspace &work_,
                 const std::vector<double> &x_)
    requires(ProblemT::parameter_extent == std::dynamic_extent)
      : problem(problem_), options(options_), result(result_), work(work_),
        x(x_.data(), x_.size()) {}
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
    if (!evaluate_residual_at(context, work.x_trial_view(), work.r_trial_view(),
                              what)) {
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
    if (!evaluate_residual_at(context, work.x_trial_view(), work.r_trial_view(),
                              what)) {
      return false;
    }
    work.x_trial[j] = xj - h;
    if (!evaluate_residual_at(context, work.x_trial_view(),
                              work.r_trial_minus_view(), what)) {
      return false;
    }
    for (Index i = 0; i < row; ++i) {
      work.J(i, j) = (work.r_trial[i] - work.r_trial_minus[i]) / (2.0 * h);
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
