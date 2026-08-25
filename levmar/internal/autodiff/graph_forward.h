#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <span>
#include <vector>

#include <levmar/internal/autodiff/graph.h>

inline constexpr std::size_t kTangentBlockWidth = 8;

template <std::size_t W> struct alignas(64) TangentBlock {
  std::array<double, W> lanes{};

  double &operator[](std::size_t lane) { return lanes[lane]; }
  const double &operator[](std::size_t lane) const { return lanes[lane]; }

  void fill(double value) { lanes.fill(value); }
};

struct AdEvalContext {
  std::vector<double> values;
  std::vector<double> tangents;
  std::vector<TangentBlock<kTangentBlockWidth>> tangent_blocks;

  void ensure_value_capacity(Index n) {
    if (values.size() < n) {
      values.resize(n);
    }
  }
  void ensure_tangent_capacity(Index n) {
    if (tangents.size() < n) {
      tangents.resize(n);
    }
  }
  void ensure_tangent_block_capacity(Index n) {
    if (tangent_blocks.size() < n) {
      tangent_blocks.resize(n);
    }
  }

  void reset_tangents(Index n) {
    std::fill(tangents.begin(), tangents.begin() + n, 0.0);
  }
};

template <Index N>
ErrorOrVoid forward_pass(const AdGraph &graph, ConstVectorView<N> parameters,
                         AdEvalContext &ctx) {

  ctx.ensure_value_capacity(graph.nodes.size());

  for (NodeId id = 0; id < graph.nodes.size(); ++id) {
    const Node &node = graph.nodes[id];
    switch (node.kind) {
    case NodeKind::Constant:
      ctx.values[id] = std::bit_cast<double>(node.literal_bits);
      break;
    case NodeKind::Variable:
      ctx.values[id] = parameters[node.parameter_index];
      break;
    case NodeKind::Neg:
      ctx.values[id] = -ctx.values[node.a];
      break;
    case NodeKind::Add:
      ctx.values[id] = ctx.values[node.a] + ctx.values[node.b];
      break;
    case NodeKind::Sub:
      ctx.values[id] = ctx.values[node.a] - ctx.values[node.b];
      break;
    case NodeKind::Mul:
      ctx.values[id] = ctx.values[node.a] * ctx.values[node.b];
      break;
    case NodeKind::Exp:
      ctx.values[id] = std::exp(ctx.values[node.a]);
      break;
    case NodeKind::Div:
      ctx.values[id] = ctx.values[node.a] / ctx.values[node.b];
      break;
    case NodeKind::Atan2:
      ctx.values[id] = std::atan2(ctx.values[node.a], ctx.values[node.b]);
      break;
    case NodeKind::Log:
      ctx.values[id] = std::log(ctx.values[node.a]);
      break;
    case NodeKind::Log1p:
      ctx.values[id] = std::log1p(ctx.values[node.a]);
      break;
    case NodeKind::Sin:
      ctx.values[id] = std::sin(ctx.values[node.a]);
      break;
    case NodeKind::Cos:
      ctx.values[id] = std::cos(ctx.values[node.a]);
      break;
    case NodeKind::Tan:
      ctx.values[id] = std::tan(ctx.values[node.a]);
      break;
    case NodeKind::Sqrt:
      ctx.values[id] = std::sqrt(ctx.values[node.a]);
      break;
    case NodeKind::Expm1:
      ctx.values[id] = std::expm1(ctx.values[node.a]);
      break;
    case NodeKind::Pow:
      if (ctx.values[node.a] <= 0.0) {
        return std::unexpected(Error{ErrorCode::NumericalFailure,
                                     "AutoDiff pow requires a positive base"});
      }
      ctx.values[id] = std::pow(ctx.values[node.a], ctx.values[node.b]);
      break;
    default:
      assert(false);
    }
  }
  return {};
}

inline void forward_tangent_pass(const AdGraph &graph, Index parameter_index,
                                 AdEvalContext &ctx) {
  ctx.ensure_tangent_capacity(graph.nodes.size());
  ctx.reset_tangents(graph.nodes.size());

  for (NodeId id = 0; id < graph.nodes.size(); ++id) {
    const Node &node = graph.nodes[id];
    switch (node.kind) {
    case NodeKind::Constant:
      ctx.tangents[id] = 0.0;
      break;
    case NodeKind::Variable:
      ctx.tangents[id] = node.parameter_index == parameter_index ? 1.0 : 0.0;
      break;
    case NodeKind::Neg:
      ctx.tangents[id] = -ctx.tangents[node.a];
      break;
    case NodeKind::Add:
      ctx.tangents[id] = ctx.tangents[node.a] + ctx.tangents[node.b];
      break;
    case NodeKind::Sub:
      ctx.tangents[id] = ctx.tangents[node.a] - ctx.tangents[node.b];
      break;
    case NodeKind::Mul:
      ctx.tangents[id] = ctx.tangents[node.a] * ctx.values[node.b] +
                         ctx.values[node.a] * ctx.tangents[node.b];
      break;
    case NodeKind::Exp:
      ctx.tangents[id] = ctx.values[id] * ctx.tangents[node.a];
      break;
    case NodeKind::Div:
      ctx.tangents[id] = (ctx.tangents[node.a] * ctx.values[node.b] -
                          ctx.values[node.a] * ctx.tangents[node.b]) /
                         (ctx.values[node.b] * ctx.values[node.b]);
      break;
    case NodeKind::Log:
      ctx.tangents[id] = ctx.tangents[node.a] / ctx.values[node.a];
      break;
    case NodeKind::Sqrt:
      ctx.tangents[id] =
          ctx.tangents[node.a] / (2 * std::sqrt(ctx.values[node.a]));
      break;
    case NodeKind::Sin:
      ctx.tangents[id] = std::cos(ctx.values[node.a]) * ctx.tangents[node.a];
      break;
    case NodeKind::Cos:
      ctx.tangents[id] = -std::sin(ctx.values[node.a]) * ctx.tangents[node.a];
      break;
    case NodeKind::Tan: {
      const double cosine = std::cos(ctx.values[node.a]);
      ctx.tangents[id] = ctx.tangents[node.a] / (cosine * cosine);
      break;
    }
    case NodeKind::Log1p:
      ctx.tangents[id] = ctx.tangents[node.a] / (1 + ctx.values[node.a]);
      break;
    case NodeKind::Expm1:
      ctx.tangents[id] = std::exp(ctx.values[node.a]) * ctx.tangents[node.a];
      break;
    case NodeKind::Atan2:
      ctx.tangents[id] = (ctx.values[node.b] * ctx.tangents[node.a] -
                          ctx.tangents[node.b] * ctx.values[node.a]) /
                         (ctx.values[node.b] * ctx.values[node.b] +
                          ctx.values[node.a] * ctx.values[node.a]);
      break;
    case NodeKind::Pow:
      assert(ctx.values[node.a] > 0.0);
      ctx.tangents[id] =
          ctx.values[id] *
          (ctx.values[node.b] * ctx.tangents[node.a] / ctx.values[node.a] +
           std::log(ctx.values[node.a]) * ctx.tangents[node.b]);
      break;
    default:
      assert(false);
    }
  }
}

inline void forward_tangent_block(const AdGraph &graph, Index first_parameter,
                                  Index active_lanes, AdEvalContext &ctx) {
  constexpr Index W = kTangentBlockWidth;
  assert(active_lanes <= W);

  ctx.ensure_tangent_block_capacity(graph.nodes.size());

  for (NodeId id = 0; id < graph.nodes.size(); ++id) {
    const Node &node = graph.nodes[id];
    auto &out = ctx.tangent_blocks[id];

    switch (node.kind) {
    case NodeKind::Constant:
      out.fill(0.0);
      break;
    case NodeKind::Variable:
      for (Index lane = 0; lane < W; ++lane) {
        out[lane] = lane < active_lanes &&
                            node.parameter_index == first_parameter + lane
                        ? 1.0
                        : 0.0;
      }
      break;
    case NodeKind::Neg: {
      const auto &a = ctx.tangent_blocks[node.a];
      for (Index lane = 0; lane < W; ++lane) {
        out[lane] = -a[lane];
      }
      break;
    }
    case NodeKind::Add: {
      const auto &a = ctx.tangent_blocks[node.a];
      const auto &b = ctx.tangent_blocks[node.b];
      for (Index lane = 0; lane < W; ++lane) {
        out[lane] = a[lane] + b[lane];
      }
      break;
    }
    case NodeKind::Sub: {
      const auto &a = ctx.tangent_blocks[node.a];
      const auto &b = ctx.tangent_blocks[node.b];
      for (Index lane = 0; lane < W; ++lane) {
        out[lane] = a[lane] - b[lane];
      }
      break;
    }
    case NodeKind::Mul: {
      const auto &da = ctx.tangent_blocks[node.a];
      const auto &db = ctx.tangent_blocks[node.b];
      const double a = ctx.values[node.a];
      const double b = ctx.values[node.b];
      for (Index lane = 0; lane < W; ++lane) {
        out[lane] = da[lane] * b + a * db[lane];
      }
      break;
    }
    case NodeKind::Exp: {
      const auto &a = ctx.tangent_blocks[node.a];
      const double scale = ctx.values[id];
      for (Index lane = 0; lane < W; ++lane) {
        out[lane] = scale * a[lane];
      }
      break;
    }
    case NodeKind::Div: {
      const auto &da = ctx.tangent_blocks[node.a];
      const auto &db = ctx.tangent_blocks[node.b];
      const double a = ctx.values[node.a];
      const double b = ctx.values[node.b];
      const double denominator = b * b;
      for (Index lane = 0; lane < W; ++lane) {
        out[lane] = (da[lane] * b - a * db[lane]) / denominator;
      }
      break;
    }
    case NodeKind::Log: {
      const auto &a = ctx.tangent_blocks[node.a];
      const double scale = 1.0 / ctx.values[node.a];
      for (Index lane = 0; lane < W; ++lane) {
        out[lane] = scale * a[lane];
      }
      break;
    }
    case NodeKind::Sqrt: {
      const auto &a = ctx.tangent_blocks[node.a];
      const double denominator = 2.0 * ctx.values[id];
      for (Index lane = 0; lane < W; ++lane) {
        out[lane] = a[lane] / denominator;
      }
      break;
    }
    case NodeKind::Sin: {
      const auto &a = ctx.tangent_blocks[node.a];
      const double scale = std::cos(ctx.values[node.a]);
      for (Index lane = 0; lane < W; ++lane) {
        out[lane] = scale * a[lane];
      }
      break;
    }
    case NodeKind::Cos: {
      const auto &a = ctx.tangent_blocks[node.a];
      const double scale = -std::sin(ctx.values[node.a]);
      for (Index lane = 0; lane < W; ++lane) {
        out[lane] = scale * a[lane];
      }
      break;
    }
    case NodeKind::Tan: {
      const auto &a = ctx.tangent_blocks[node.a];
      const double cosine = std::cos(ctx.values[node.a]);
      const double denominator = cosine * cosine;
      for (Index lane = 0; lane < W; ++lane) {
        out[lane] = a[lane] / denominator;
      }
      break;
    }
    case NodeKind::Log1p: {
      const auto &a = ctx.tangent_blocks[node.a];
      const double scale = 1.0 / (1.0 + ctx.values[node.a]);
      for (Index lane = 0; lane < W; ++lane) {
        out[lane] = scale * a[lane];
      }
      break;
    }
    case NodeKind::Expm1: {
      const auto &a = ctx.tangent_blocks[node.a];
      const double scale = std::exp(ctx.values[node.a]);
      for (Index lane = 0; lane < W; ++lane) {
        out[lane] = scale * a[lane];
      }
      break;
    }
    case NodeKind::Atan2: {
      const auto &da = ctx.tangent_blocks[node.a];
      const auto &db = ctx.tangent_blocks[node.b];
      const double a = ctx.values[node.a];
      const double b = ctx.values[node.b];
      const double denominator = b * b + a * a;
      for (Index lane = 0; lane < W; ++lane) {
        out[lane] = (b * da[lane] - db[lane] * a) / denominator;
      }
      break;
    }
    case NodeKind::Pow: {
      assert(ctx.values[node.a] > 0.0);
      const auto &da = ctx.tangent_blocks[node.a];
      const auto &db = ctx.tangent_blocks[node.b];
      const double a = ctx.values[node.a];
      const double b = ctx.values[node.b];
      const double value = ctx.values[id];
      const double log_a = std::log(a);
      for (Index lane = 0; lane < W; ++lane) {
        out[lane] = value * (b * da[lane] / a + log_a * db[lane]);
      }
      break;
    }
    default:
      assert(false);
    }
  }
}

template <Index M, Index N>
ErrorOrVoid
evaluate_roots_forward(const AdGraph &graph, std::span<const NodeId> roots,
                       ConstVectorView<N> parameters, VectorView<M> r,
                       MatrixView<M, N> J, AdEvalContext &ctx) {
  const Index m = roots.size();
  const Index n = parameters.size();

  assert(r.size() == m);
  if constexpr (M == std::dynamic_extent) {
    assert(J.extent(0) == m);
  } else {
    assert(m == M);
  }

  if constexpr (N == std::dynamic_extent) {
    assert(J.extent(1) == n);
  } else {
    assert(n == N);
  }

  // fill residuals
  ErrorOrVoid result = forward_pass(graph, parameters, ctx);
  if (!result) {
    return result;
  }
  for (Index i = 0; i < m; ++i) {
    r[i] = ctx.values[roots[i]];
  }

  // Fill Jacobian columns in SIMD-sized tangent blocks.
  for (Index first_parameter = 0; first_parameter < n;
       first_parameter += kTangentBlockWidth) {
    const Index active_lanes =
        std::min(kTangentBlockWidth, n - first_parameter);
    forward_tangent_block(graph, first_parameter, active_lanes, ctx);

    for (Index lane = 0; lane < active_lanes; ++lane) {
      const Index j = first_parameter + lane;
      for (Index i = 0; i < m; ++i) {
        J[i, j] = ctx.tangent_blocks[roots[i]][lane];
      }
    }
  }
  return {};
}
