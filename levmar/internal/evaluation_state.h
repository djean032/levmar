#pragma once

#include <array>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <levmar/internal/autodiff/dual_forward.h>
#include <levmar/internal/autodiff/graph_forward.h>
#include <levmar/internal/core.h>
#include <levmar/internal/problem.h>
#include <levmar/internal/storage.h>

namespace levmar {

struct Result {
  TerminationReason termination = TerminationReason::NotTerminated;

  Index iterations = 0;
  Index function_evaluations = 0;
  Index jacobian_evaluations = 0;
  Index linear_solves = 0;
  Index accepted_steps = 0;
  Index rejected_steps = 0;

  double initial_cost = std::numeric_limits<double>::quiet_NaN();
  double final_cost = std::numeric_limits<double>::quiet_NaN();

  double gradient_inf_norm = std::numeric_limits<double>::quiet_NaN();
  double step_norm = std::numeric_limits<double>::quiet_NaN();

  double lambda = std::numeric_limits<double>::quiet_NaN();

  std::vector<double> parameters;

  std::string message;
};

} // namespace levmar

namespace levmar::detail {

template <Index M, Index N> struct LMWorkspace {
  static constexpr Index residual_extent = M;
  static constexpr Index parameter_extent = N;

  Index m = (M == std::dynamic_extent ? 0 : M);
  Index n = (N == std::dynamic_extent ? 0 : N);

  AdEvalContext graph_eval;
  DualEvalContext<M, N> dual_eval;

  VectorStorage<N> x_current;
  VectorStorage<N> x_trial;

  VectorStorage<M> r_trial;
  VectorStorage<M> r_trial_minus;
  VectorStorage<M> r;

  MatrixStorage<M, N> J;

  VectorStorage<N> g;
  VectorStorage<N> step;

  VectorStorage<N> scale;
  VectorStorage<N> damping_scale;
  VectorStorage<M> weights;

  LMWorkspace() {
    scale.fill(1.0);
    weights.fill(1.0);
  }

  LMWorkspace(Index m_runtime, Index n_runtime) : LMWorkspace() {
    resize(m_runtime, n_runtime);
  }

  void resize(Index m_runtime, Index n_runtime) {
    const Index requested_m = M == std::dynamic_extent ? m_runtime : M;
    const Index requested_n = N == std::dynamic_extent ? n_runtime : N;

    if (m == requested_m && n == requested_n) {
      return;
    }

    m = requested_m;
    n = requested_n;
    if constexpr (M == std::dynamic_extent && N == std::dynamic_extent) {
      J.resize(m, n);
      r.resize(m);
      r_trial.resize(m);
      r_trial_minus.resize(m);
      weights.resize(m);
      x_current.resize(n);
      x_trial.resize(n);
      g.resize(n);
      step.resize(n);
      scale.resize(n);
      damping_scale.resize(n);
    } else if constexpr (M == std::dynamic_extent) {
      J.resize(m);
      r.resize(m);
      r_trial.resize(m);
      r_trial_minus.resize(m);
      weights.resize(m);
    } else if constexpr (N == std::dynamic_extent) {
      J.resize(n);
      x_current.resize(n);
      x_trial.resize(n);
      g.resize(n);
      step.resize(n);
      scale.resize(n);
      damping_scale.resize(n);
    }

    scale.fill(1.0);
    damping_scale.fill(1.0);
    weights.fill(1.0);
  }
};

template <Index M> struct AutoDiffGraphCache {
  AdGraph graph;
  VectorStorage<M, NodeId> roots;
  bool recorded = false;
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
  AutoDiffGraphCache<M> autodiff_cache;

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
[[nodiscard]] inline ErrorOrVoid
validate_context(const LMSolveContext<M, N, Residual, Jacobian> &context) {
  if (auto problem_result = validate_problem(context.problem);
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
[[nodiscard]] inline ErrorOrVoid
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
[[nodiscard]] inline ErrorOrVoid
evaluate_residual_at(LMSolveContext<M, N, Residual, Jacobian> &context,
                     VectorView<N> x, VectorView<M> r,
                     std::string_view what = "Residual Evaluation") {
  return evaluate_residual_at(context, ConstVectorView<N>(x.data(), x.size()),
                              r, what);
}

template <Index M, Index N, ResidualCallable<M, N> Residual, class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
[[nodiscard]] inline ErrorOrVoid
evaluate_residual(LMSolveContext<M, N, Residual, Jacobian> &context,
                  std::string_view what = "Residual Evaluation") {
  return evaluate_residual_at(context, context.work.x_current.view(),
                              context.work.r.view(), what);
}

} // namespace levmar::detail
