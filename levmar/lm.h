#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <expected>
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

enum class ErrorCode { InvalidProblem, UserFunctionError, NumericalFailure };

enum class TerminationReason {
  NotTerminated,
  SmallStep,
  SmallGradient,
  SmallCostReduction,
  MaxIterations
};

struct Error {
  ErrorCode code;
  std::string message;
};

template <class T> using ErrorOr = std::expected<T, Error>;

using ErrorOrVoid = ErrorOr<void>;

// Views for vectors and matrices using span and mdspan
template <Index N, class Scalar = double>
using VectorView = std::span<Scalar, N>;

template <Index N, class Scalar = double>
using ConstVectorView = std::span<const Scalar, N>;

template <Index M, Index N, class Scalar = double>
using MatrixView =
    std::mdspan<Scalar, std::extents<Index, M, N>, std::layout_left>;

template <Index M, Index N, class Scalar = double>
using ConstMatrixView =
    std::mdspan<const Scalar, std::extents<Index, M, N>, std::layout_left>;

template <Index Extent, class Scalar = double> struct VectorStorage {
  std::array<Scalar, Extent> storage{};

  static constexpr Index extent = Extent;
  constexpr Index size() const noexcept { return Extent; }
  constexpr bool empty() const noexcept { return Extent == 0; }

  Scalar *data() { return storage.data(); }
  const Scalar *data() const { return storage.data(); }

  VectorView<Extent, Scalar> view() {
    return VectorView<Extent, Scalar>(storage.data(), Extent);
  }
  ConstVectorView<Extent, Scalar> view() const {
    return ConstVectorView<Extent, Scalar>(storage.data(), Extent);
  }

  Scalar &operator[](Index i) { return storage[i]; }

  const Scalar &operator[](Index i) const { return storage[i]; }

  void fill(const Scalar &value) { storage.fill(value); }
};

template <class Scalar> struct VectorStorage<std::dynamic_extent, Scalar> {
  std::vector<Scalar> storage;

  static constexpr Index extent = std::dynamic_extent;
  constexpr Index size() const noexcept { return storage.size(); }
  constexpr bool empty() const noexcept { return storage.empty(); }

  Scalar *data() { return storage.data(); }
  const Scalar *data() const { return storage.data(); }

  VectorView<std::dynamic_extent, Scalar> view() {
    return VectorView<std::dynamic_extent, Scalar>(storage.data(),
                                                   storage.size());
  }
  ConstVectorView<std::dynamic_extent, Scalar> view() const {
    return ConstVectorView<std::dynamic_extent, Scalar>(storage.data(),
                                                        storage.size());
  }

  Scalar &operator[](Index i) noexcept { return storage[i]; }

  const Scalar &operator[](Index i) const noexcept { return storage[i]; }

  void resize(Index n) { storage.resize(n); }
  void assign(Index n, const Scalar &value) { storage.assign(n, value); }
  void fill(const Scalar &value) { std::ranges::fill(storage, value); }
};

template <Index Rows, Index Cols, class Scalar = double> struct MatrixStorage {
  std::array<Scalar, Rows * Cols> storage{};

  static constexpr Index rows_extent = Rows;
  static constexpr Index cols_extent = Cols;

  constexpr Index rows() const noexcept { return Rows; }
  constexpr Index cols() const noexcept { return Cols; }
  constexpr Index leading_dim() const noexcept { return Rows; }
  constexpr Index size() const noexcept { return rows() * cols(); }
  constexpr bool empty() const noexcept { return size() == 0; }

  Scalar *data() { return storage.data(); }
  const Scalar *data() const { return storage.data(); }

  MatrixView<Rows, Cols, Scalar> view() {
    return MatrixView<Rows, Cols, Scalar>(storage.data());
  }
  ConstMatrixView<Rows, Cols, Scalar> view() const {
    return ConstMatrixView<Rows, Cols, Scalar>(storage.data());
  }

  Scalar &operator()(Index i, Index j) noexcept {
    return storage[i + j * leading_dim()];
  }

  const Scalar &operator()(Index i, Index j) const noexcept {
    return storage[i + j * leading_dim()];
  }

  Scalar &operator[](Index i, Index j) noexcept { return (*this)(i, j); }

  const Scalar &operator[](Index i, Index j) const noexcept {
    return (*this)(i, j);
  }

  void fill(const Scalar &value) { storage.fill(value); }
};

template <Index Cols, class Scalar>
struct MatrixStorage<std::dynamic_extent, Cols, Scalar> {
  Index rows_ = 0;
  std::vector<Scalar> storage;

  static constexpr Index rows_extent = std::dynamic_extent;
  static constexpr Index cols_extent = Cols;

  constexpr Index rows() const noexcept { return rows_; }
  constexpr Index cols() const noexcept { return Cols; }
  constexpr Index leading_dim() const noexcept { return rows_; }
  constexpr Index size() const noexcept { return rows() * cols(); }
  constexpr bool empty() const noexcept { return size() == 0; }

  Scalar *data() { return storage.data(); }
  const Scalar *data() const { return storage.data(); }

  MatrixView<std::dynamic_extent, Cols, Scalar> view() {
    return MatrixView<std::dynamic_extent, Cols, Scalar>(storage.data(), rows_);
  }

  ConstMatrixView<std::dynamic_extent, Cols, Scalar> view() const {
    return ConstMatrixView<std::dynamic_extent, Cols, Scalar>(storage.data(),
                                                              rows_);
  }

  Scalar &operator()(Index i, Index j) noexcept {
    return storage[i + j * leading_dim()];
  }

  const Scalar &operator()(Index i, Index j) const noexcept {
    return storage[i + j * leading_dim()];
  }

  Scalar &operator[](Index i, Index j) noexcept { return (*this)(i, j); }

  const Scalar &operator[](Index i, Index j) const noexcept {
    return (*this)(i, j);
  }

  void resize(Index rows) {
    rows_ = rows;
    storage.assign(rows_ * Cols, Scalar{});
  }

  void fill(const Scalar &value) { std::ranges::fill(storage, value); }
};

template <Index Rows, class Scalar>
struct MatrixStorage<Rows, std::dynamic_extent, Scalar> {
  Index cols_ = 0;
  std::vector<Scalar> storage;

  static constexpr Index rows_extent = Rows;
  static constexpr Index cols_extent = std::dynamic_extent;

  constexpr Index rows() const noexcept { return Rows; }
  constexpr Index cols() const noexcept { return cols_; }
  constexpr Index leading_dim() const noexcept { return Rows; }
  constexpr Index size() const noexcept { return rows() * cols(); }
  constexpr bool empty() const noexcept { return size() == 0; }

  Scalar *data() { return storage.data(); }
  const Scalar *data() const { return storage.data(); }

  MatrixView<Rows, std::dynamic_extent, Scalar> view() {
    return MatrixView<Rows, std::dynamic_extent, Scalar>(storage.data(), cols_);
  }

  ConstMatrixView<Rows, std::dynamic_extent, Scalar> view() const {
    return ConstMatrixView<Rows, std::dynamic_extent, Scalar>(storage.data(),
                                                              cols_);
  }

  Scalar &operator()(Index i, Index j) noexcept {
    return storage[i + j * leading_dim()];
  }

  const Scalar &operator()(Index i, Index j) const noexcept {
    return storage[i + j * leading_dim()];
  }

  Scalar &operator[](Index i, Index j) noexcept { return (*this)(i, j); }

  const Scalar &operator[](Index i, Index j) const noexcept {
    return (*this)(i, j);
  }

  void resize(Index cols) {
    cols_ = cols;
    storage.assign(Rows * cols_, Scalar{});
  }

  void fill(const Scalar &value) { std::ranges::fill(storage, value); }
};

template <class Scalar>
struct MatrixStorage<std::dynamic_extent, std::dynamic_extent, Scalar> {
  Index rows_ = 0;
  Index cols_ = 0;
  std::vector<Scalar> storage{};

  static constexpr Index rows_extent = std::dynamic_extent;
  static constexpr Index cols_extent = std::dynamic_extent;

  constexpr Index rows() const noexcept { return rows_; }
  constexpr Index cols() const noexcept { return cols_; }
  constexpr Index leading_dim() const noexcept { return rows_; }
  constexpr Index size() const noexcept { return rows() * cols(); }
  constexpr bool empty() const noexcept { return size() == 0; }

  Scalar *data() { return storage.data(); }
  const Scalar *data() const { return storage.data(); }

  MatrixView<std::dynamic_extent, std::dynamic_extent, Scalar> view() {
    return MatrixView<std::dynamic_extent, std::dynamic_extent, Scalar>(
        storage.data(), rows_, cols_);
  }

  ConstMatrixView<std::dynamic_extent, std::dynamic_extent, Scalar>
  view() const {
    return ConstMatrixView<std::dynamic_extent, std::dynamic_extent, Scalar>(
        storage.data(), rows_, cols_);
  }

  Scalar &operator()(Index i, Index j) noexcept {
    return storage[i + j * leading_dim()];
  }

  const Scalar &operator()(Index i, Index j) const noexcept {
    return storage[i + j * leading_dim()];
  }

  Scalar &operator[](Index i, Index j) noexcept { return (*this)(i, j); }

  const Scalar &operator[](Index i, Index j) const noexcept {
    return (*this)(i, j);
  }

  void resize(Index rows, Index cols) {
    cols_ = cols;
    rows_ = rows;
    storage.assign(rows_ * cols_, Scalar{});
  }

  void fill(const Scalar &value) { std::ranges::fill(storage, value); }
};

template <Index M, Index N>
using ResidualSignature = ErrorOrVoid(ConstVectorView<N> x, VectorView<M> r);

template <Index M, Index N>
using JacobianSignature = ErrorOrVoid(ConstVectorView<N> x, MatrixView<M, N> J);

template <class Residual, Index M, Index N>
concept ResidualCallable =
    requires(Residual residual, ConstVectorView<N> x, VectorView<M> r) {
      { residual(x, r) } -> std::same_as<ErrorOrVoid>;
    };

template <class Jacobian, Index M, Index N>
concept JacobianCallable =
    requires(Jacobian jacobian, ConstVectorView<N> x, MatrixView<M, N> J) {
      { jacobian(x, J) } -> std::same_as<ErrorOrVoid>;
    };

template <class Jacobian, Index M, Index N>
concept OptionalJacobianCallable =
    std::same_as<Jacobian, NoJacobian> || JacobianCallable<Jacobian, M, N>;

enum class JacobianMode { User, ForwardDifference, CentralDifference };

enum class Strategy { GaussNewton, LevenbergMarquardt, TrustRegionLM, DogLeg };

enum class LinearSolver { NormalEquationsCholesky, QR, SVD };

enum class LossKind { Squared, Huber, Cauchy, SoftL1, User };

enum class ScalingMode { None, JacobianColumnNorm, User };

template <Index M, Index N, ResidualCallable<M, N> Residual,
          class Jacobian = NoJacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
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

template <Index M, Index N, ResidualCallable<M, N> Residual,
          class Jacobian = NoJacobian>
  requires(M != std::dynamic_extent && N != std::dynamic_extent) &&
          OptionalJacobianCallable<Jacobian, M, N>
auto make_problem(Residual residual, Jacobian jacobian = {}) {
  return Problem<M, N, Residual, Jacobian>(residual, jacobian);
}

template <ResidualCallable<std::dynamic_extent, std::dynamic_extent> Residual,
          class Jacobian = NoJacobian>
  requires OptionalJacobianCallable<Jacobian, std::dynamic_extent,
                                    std::dynamic_extent>
auto make_dynamic_problem(Index m, Index n, Residual residual,
                          Jacobian jacobian = {}) {
  return Problem<std::dynamic_extent, std::dynamic_extent, Residual, Jacobian>(
      m, n, residual, jacobian);
}

template <Index N, ResidualCallable<std::dynamic_extent, N> Residual,
          class Jacobian = NoJacobian>
  requires OptionalJacobianCallable<Jacobian, std::dynamic_extent, N>
auto make_problem_dynamic_residuals(Index m, Residual residual,
                                    Jacobian jacobian = {}) {
  return Problem<std::dynamic_extent, N, Residual, Jacobian>(m, N, residual,
                                                             jacobian);
}

template <Index M, ResidualCallable<M, std::dynamic_extent> Residual,
          class Jacobian = NoJacobian>
  requires OptionalJacobianCallable<Jacobian, M, std::dynamic_extent>
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

template <Index M, Index N, ResidualCallable<M, N> Residual, class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
inline ErrorOrVoid
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
      return std::unexpected(Error{
          ErrorCode::InvalidProblem,
          "Problem residual count does not match static residual extent"});
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
  }
  return std::sqrt(std::numeric_limits<double>::epsilon());
}

inline double finite_difference_perturbation(double xj, double rel_step) {
  return rel_step * std::max(double{1.0}, std::abs(xj));
}

struct Result {
  TerminationReason termination = TerminationReason::NotTerminated;

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

template <Index M, Index N, ResidualCallable<M, N> Residual, class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
struct LMSolveContext {
  using ProblemType = Problem<M, N, Residual, Jacobian>;
  using Workspace = LMWorkspace<M, N>;
  const ProblemType &problem;
  const Options &options;
  Result &result;
  Workspace &work;
  ConstVectorView<N> x;

  LMSolveContext(const ProblemType &problem_, const Options &options_,
                 Result &result_, Workspace &work_, ConstVectorView<N> x_)
    requires(N != std::dynamic_extent)
      : problem(problem_), options(options_), result(result_), work(work_),
        x(x_) {
    ensure_workspace_shape();
  }

  LMSolveContext(const ProblemType &problem_, const Options &options_,
                 Result &result_, Workspace &work_, ConstVectorView<N> x_)
    requires(N == std::dynamic_extent)
      : problem(problem_), options(options_), result(result_), work(work_),
        x() {
    ensure_workspace_shape();
    x = x_;
  }

  LMSolveContext(const ProblemType &problem_, const Options &options_,
                 Result &result_, Workspace &work_,
                 const std::array<double, N> &x_)
    requires(N != std::dynamic_extent)
      : problem(problem_), options(options_), result(result_), work(work_),
        x(x_.data(), x_.size()) {
    ensure_workspace_shape();
  }

  LMSolveContext(const ProblemType &problem_, const Options &options_,
                 Result &result_, Workspace &work_,
                 const std::vector<double> &x_)
    requires(N == std::dynamic_extent)
      : problem(problem_), options(options_), result(result_), work(work_) {
    ensure_workspace_shape();
    x = ConstVectorView<std::dynamic_extent>(x_.data(), x_.size());
  }

private:
  void ensure_workspace_shape() {
    if constexpr (N == std::dynamic_extent || M == std::dynamic_extent) {
      work.resize(problem.num_residuals, problem.num_parameters);
    }
  }
};

template <Index M, Index N, ResidualCallable<M, N> Residual, class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
inline ErrorOrVoid
validate_context(const LMSolveContext<M, N, Residual, Jacobian> &context) {
  if (auto problem_result = validate_problem(context.problem, context.options);
      !problem_result) {
    return problem_result;
  }

  if (context.x.size() != context.problem.num_parameters) {
    return std::unexpected(Error{ErrorCode::InvalidProblem,
                                 "Initial parameter vector size does not match "
                                 "problem.num_parameters"});
  }

  if (context.work.m != context.problem.num_residuals) {
    return std::unexpected(Error{ErrorCode::InvalidProblem,
                                 "Workspace residual dimension does not match "
                                 "problem.num_residuals"});
  }

  if (context.work.n != context.problem.num_parameters) {
    return std::unexpected(
        Error{ErrorCode::InvalidProblem,
              "Workspace parameters dimension does not match "
              "problem.num_parameters"});
  }

  return {};
}

template <Index M, Index N, ResidualCallable<M, N> Residual, class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
inline ErrorOrVoid
evaluate_residual_at(LMSolveContext<M, N, Residual, Jacobian> &context,
                     ConstVectorView<N> x, VectorView<M> r,
                     std::string_view what = "Residual Evaluation") {
  if (auto residual_result = context.problem.residual(x, r); !residual_result) {
    return std::unexpected(Error{
        residual_result.error().code,
        std::string(what) + " failed: " + residual_result.error().message});
  }
  ++context.result.function_evaluations;
  return {};
}

template <Index M, Index N, ResidualCallable<M, N> Residual, class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
inline ErrorOrVoid
evaluate_residual_at(LMSolveContext<M, N, Residual, Jacobian> &context,
                     VectorView<N> x, VectorView<M> r,
                     std::string_view what = "Residual Evaluation") {
  return evaluate_residual_at(context, ConstVectorView<N>(x.data(), x.size()),
                              r, what);
}

template <Index M, Index N, ResidualCallable<M, N> Residual, class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
inline ErrorOrVoid
evaluate_residual(LMSolveContext<M, N, Residual, Jacobian> &context,
                  std::string_view what = "Residual Evaluation") {
  return evaluate_residual_at(context, context.work.x_current.view(),
                              context.work.r.view(), what);
}

template <Index M, Index N, ResidualCallable<M, N> Residual, class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
inline ErrorOrVoid evaluate_forward_difference_jacobian(
    LMSolveContext<M, N, Residual, Jacobian> &context,
    std::string_view what = "Forward Diff Jacobian") {
  auto &result = context.result;
  const auto &options = context.options;
  auto &work = context.work;

  const double rel_step = resolved_finite_difference_step(options);
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
inline ErrorOrVoid evaluate_central_difference_jacobian(
    LMSolveContext<M, N, Residual, Jacobian> &context,
    std::string_view what = "Central Diff Jacobian") {
  auto &result = context.result;
  const auto &options = context.options;
  auto &work = context.work;

  const double rel_step = resolved_finite_difference_step(options);
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

template <Index M, Index N, ResidualCallable<M, N> Residual, class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
inline ErrorOrVoid
evaluate_jacobian(LMSolveContext<M, N, Residual, Jacobian> &context,
                  std::string_view what = "Jacobian Evaluation") {
  const auto &problem = context.problem;
  const auto &options = context.options;
  auto &work = context.work;
  switch (options.jacobian_mode) {
  case JacobianMode::User:
    if (!problem.has_user_jacobian()) {
      return std::unexpected(
          Error{ErrorCode::InvalidProblem,
                std::string(what) + " failed: missing jacobian function"});
    }
    if (auto jacobian_result =
            problem.jacobian(work.x_current.view(), work.J.view());
        !jacobian_result) {
      return std::unexpected(Error{
          jacobian_result.error().code,
          std::string(what) + " failed: " + jacobian_result.error().message});
    }
    ++context.result.jacobian_evaluations;
    return {};
  case JacobianMode::ForwardDifference:
    return evaluate_forward_difference_jacobian(context, what);
  case JacobianMode::CentralDifference:
    return evaluate_central_difference_jacobian(context, what);
  }

  return std::unexpected(
      Error{ErrorCode::InvalidProblem, std::string(what) + " failed."});
}
