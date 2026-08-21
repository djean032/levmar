#pragma once

#include <algorithm>
#include <array>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include <levmar/internal/evaluation_state.h>

template <Index M, Index N, class Residual>
ErrorOrVoid invoke_residual_autodiff(const Residual &residual, AdGraph &graph,
                                     std::array<AdExprRef, N> &x_exprs,
                                     std::array<AdExprRef, M> &r_exprs,
                                     std::array<NodeId, M> &roots) {
  if constexpr (AutoDiffResidualCallable<Residual, M, N>) {

    for (Index j = 0; j < N; ++j) {
      x_exprs[j] = variable(graph, j);
    }

    r_exprs.fill(AdExprRef{&graph, kInvalidNode});

    ConstVectorView<N, AdExprRef> x_view(x_exprs.data(), x_exprs.size());
    VectorView<M, AdExprRef> r_view(r_exprs.data(), r_exprs.size());

    if (auto result = residual(x_view, r_view); !result) {
      return result;
    }

    for (Index i = 0; i < M; ++i) {
      const AdExprRef &expr = r_exprs[i];
      if (expr.graph != &graph || expr.id == kInvalidNode ||
          expr.id >= graph.nodes.size()) {
        return std::unexpected(Error{
            ErrorCode::UserFunctionError,
            "AutoDiff residual did not assign a valid expression to residual " +
                std::to_string(i)});
      }
      roots[i] = r_exprs[i].id;
    }

    return {};
  } else {
    return std::unexpected(Error{
        ErrorCode::InvalidProblem,
        "JacobianMode::AutoDiff requires a scalar-generic residual callback"});
  }
}

template <Index M, Index N, class Residual>
ErrorOrVoid invoke_residual_autodiff(const Residual &residual, Index m, Index n,
                                     AdGraph &graph,
                                     std::vector<AdExprRef> &x_exprs,
                                     std::vector<AdExprRef> &r_exprs,
                                     std::vector<NodeId> &roots) {
  if constexpr (AutoDiffResidualCallable<Residual, M, N>) {
    x_exprs.resize(n);
    for (Index j = 0; j < n; ++j) {
      x_exprs[j] = variable(graph, j);
    }

    r_exprs.assign(m, AdExprRef{&graph, kInvalidNode});
    roots.resize(m);

    ConstVectorView<N, AdExprRef> x_view(x_exprs.data(), x_exprs.size());
    VectorView<M, AdExprRef> r_view(r_exprs.data(), r_exprs.size());

    if (auto result = residual(x_view, r_view); !result) {
      return result;
    }

    for (Index i = 0; i < m; ++i) {
      const AdExprRef &expr = r_exprs[i];
      if (expr.graph != &graph || expr.id == kInvalidNode ||
          expr.id >= graph.nodes.size()) {
        return std::unexpected(Error{
            ErrorCode::UserFunctionError,
            "AutoDiff residual did not assign a valid expression to residual " +
                std::to_string(i)});
      }
      roots[i] = r_exprs[i].id;
    }
    return {};
  } else {
    return std::unexpected(Error{
        ErrorCode::InvalidProblem,
        "JacobianMode::AutoDiff requires a scalar-generic residual callback"});
  }
}

template <Index M, Index N, ResidualCallable<M, N> Residual, class Jacobian>
  requires OptionalJacobianCallable<Jacobian, M, N>
[[nodiscard]] inline ErrorOrVoid
record_autodiff_graph(LMSolveContext<M, N, Residual, Jacobian> &context,
                      std::string_view what = "AutoDiff residual/jacobian") {
  auto &cache = context.autodiff_cache;

  AdGraph graph;

  const auto commit = [&](const auto &roots) {
    if constexpr (M == std::dynamic_extent) {
      cache.roots.resize(context.problem.num_residuals);
    }

    std::ranges::copy(roots, cache.roots.view().begin());
    cache.graph = std::move(graph);
    cache.recorded = true;
  };

  if constexpr (M != std::dynamic_extent && N != std::dynamic_extent) {
    std::array<AdExprRef, N> x_exprs;
    std::array<AdExprRef, M> r_exprs;
    std::array<NodeId, M> roots;

    if (auto residual_result = invoke_residual_autodiff<M, N>(
            context.problem.residual, graph, x_exprs, r_exprs, roots);
        !residual_result) {
      return std::unexpected(Error{
          residual_result.error().code,
          std::string(what) + " failed: " + residual_result.error().message});
    }

    commit(roots);
  } else {
    std::vector<AdExprRef> x_exprs;
    std::vector<AdExprRef> r_exprs;
    std::vector<NodeId> roots;

    if (auto residual_result = invoke_residual_autodiff<M, N>(
            context.problem.residual, context.problem.num_residuals,
            context.problem.num_parameters, graph, x_exprs, r_exprs, roots);
        !residual_result) {
      return std::unexpected(Error{
          residual_result.error().code,
          std::string(what) + " failed: " + residual_result.error().message});
    }

    commit(roots);
  }

  return {};
}

template <Index M, Index N, class Residual, class Jacobian>
[[nodiscard]] inline ErrorOrVoid evaluate_cached_autodiff_graph(
    LMSolveContext<M, N, Residual, Jacobian> &context) {
  auto &cache = context.autodiff_cache;
  auto &work = context.work;

  return evaluate_roots_forward(
      cache.graph,
      std::span<const NodeId>(cache.roots.data(), cache.roots.size()),
      ConstVectorView<N>(work.x_current.data(), work.x_current.size()),
      work.r.view(), work.J.view(), work.ad_eval);
}

template <Index M, Index N, class Residual, class Jacobian>
[[nodiscard]] inline ErrorOrVoid evaluate_autodiff_residual_and_jacobian(
    LMSolveContext<M, N, Residual, Jacobian> &context,
    std::string_view what = "AutoDiff residual/jacobian") {
  if (!context.autodiff_cache.recorded) {
    if (auto result = record_autodiff_graph(context, what); !result) {
      return result;
    }
  }

  if (auto result = evaluate_cached_autodiff_graph(context); !result) {
    return result;
  }

  ++context.result.jacobian_evaluations;
  ++context.result.function_evaluations;
  return {};
}
