#pragma once

#include <algorithm>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

#include <levmar/internal/core.h>

namespace levmar::detail {

using NodeId = Index;
constexpr NodeId kInvalidNode = std::numeric_limits<Index>::max();

enum class NodeKind {
  Constant,
  Variable,

  // Unary Ops
  Neg,
  Exp,
  Log,
  Sqrt,
  Sin,
  Cos,
  Tan,
  Log1p,
  Expm1,

  // Binary Ops
  Add,
  Sub,
  Mul,
  Div,
  Pow,
  Atan2
};

constexpr bool is_unary_kind(NodeKind kind) {
  return kind >= NodeKind::Neg && kind <= NodeKind::Expm1;
}

constexpr bool is_binary_kind(NodeKind kind) {
  return kind >= NodeKind::Add && kind <= NodeKind::Atan2;
}

struct Node {
  NodeKind kind{};
  NodeId a = kInvalidNode;
  NodeId b = kInvalidNode;
  Index parameter_index = 0;
  std::uint64_t literal_bits = 0;
  Index value_slot = kInvalidNode;
};

struct NodeKey {
  NodeKind kind{};
  NodeId a = kInvalidNode;
  NodeId b = kInvalidNode;
  Index parameter_index = 0;
  std::uint64_t literal_bits = 0;

  bool operator==(const NodeKey &) const = default;
};

struct NodeKeyHash {
  std::size_t operator()(const NodeKey &key) const noexcept {
    std::size_t h = 0;

    auto mix = [&](std::size_t v) {
      h ^= v + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
    };

    mix(static_cast<std::size_t>(key.kind));
    mix(static_cast<std::size_t>(key.a));
    mix(static_cast<std::size_t>(key.b));
    mix(static_cast<std::size_t>(key.parameter_index));
    mix(static_cast<std::size_t>(key.literal_bits));

    return h;
  }
};

struct AdGraph {
  std::vector<Node> nodes;
  std::unordered_map<NodeKey, NodeId, NodeKeyHash> interned;

  NodeId intern_node(const NodeKey &key, Node node) {
    if (auto it = interned.find(key); it != interned.end()) {
      return it->second;
    }

    const NodeId id = nodes.size();
    node.value_slot = id;
    nodes.push_back(node);
    interned.emplace(key, id);
    return id;
  }

  NodeId make_constant(double v) {
    const auto bits = std::bit_cast<std::uint64_t>(v);

    NodeKey key{.kind = NodeKind::Constant,
                .a = kInvalidNode,
                .b = kInvalidNode,
                .parameter_index = 0,
                .literal_bits = bits};

    Node node{.kind = NodeKind::Constant,
              .a = kInvalidNode,
              .b = kInvalidNode,
              .parameter_index = 0,
              .literal_bits = bits,
              .value_slot = kInvalidNode};

    return intern_node(key, node);
  }

  NodeId make_variable(Index parameter_index) {
    NodeKey key{.kind = NodeKind::Variable,
                .a = kInvalidNode,
                .b = kInvalidNode,
                .parameter_index = parameter_index,
                .literal_bits = 0};

    Node node{.kind = NodeKind::Variable,
              .a = kInvalidNode,
              .b = kInvalidNode,
              .parameter_index = parameter_index,
              .literal_bits = 0,
              .value_slot = kInvalidNode};

    return intern_node(key, node);
  }

  NodeId make_unary(NodeKind kind, NodeId child) {
    assert(child != kInvalidNode);
    assert(child < nodes.size());
    assert(is_unary_kind(kind));
    NodeKey key{.kind = kind,
                .a = child,
                .b = kInvalidNode,
                .parameter_index = 0,
                .literal_bits = 0};

    Node node{.kind = kind,
              .a = child,
              .b = kInvalidNode,
              .parameter_index = 0,
              .literal_bits = 0,
              .value_slot = kInvalidNode};

    return intern_node(key, node);
  }

  NodeId make_binary(NodeKind kind, NodeId left, NodeId right) {
    assert(left != kInvalidNode);
    assert(right != kInvalidNode);
    assert(left < nodes.size());
    assert(right < nodes.size());
    assert(is_binary_kind(kind));

    if (kind == NodeKind::Mul || kind == NodeKind::Add) {
      if (right < left) {
        std::swap(left, right);
      }
    }

    NodeKey key{.kind = kind,
                .a = left,
                .b = right,
                .parameter_index = 0,
                .literal_bits = 0};

    Node node{.kind = kind,
              .a = left,
              .b = right,
              .parameter_index = 0,
              .literal_bits = 0,
              .value_slot = kInvalidNode};

    return intern_node(key, node);
  }
};

struct AdExprRef {
  AdGraph *graph = nullptr;
  NodeId id = kInvalidNode;

  AdExprRef &operator=(double value) {
    assert(graph != nullptr);
    id = graph->make_constant(value);
    return *this;
  }
};

inline AdExprRef constant(AdGraph &graph, double value) {
  return AdExprRef{&graph, graph.make_constant(value)};
}

inline AdExprRef variable(AdGraph &graph, Index parameter_index) {
  return AdExprRef{&graph, graph.make_variable(parameter_index)};
}

inline AdExprRef exp(const AdExprRef &expr) {
  assert(expr.graph != nullptr);
  return AdExprRef{expr.graph, expr.graph->make_unary(NodeKind::Exp, expr.id)};
}

inline AdExprRef sqrt(const AdExprRef &expr) {
  assert(expr.graph != nullptr);
  return AdExprRef{expr.graph, expr.graph->make_unary(NodeKind::Sqrt, expr.id)};
}

inline AdExprRef sin(const AdExprRef &expr) {
  assert(expr.graph != nullptr);
  return AdExprRef{expr.graph, expr.graph->make_unary(NodeKind::Sin, expr.id)};
}

inline AdExprRef cos(const AdExprRef &expr) {
  assert(expr.graph != nullptr);
  return AdExprRef{expr.graph, expr.graph->make_unary(NodeKind::Cos, expr.id)};
}

inline AdExprRef tan(const AdExprRef &expr) {
  assert(expr.graph != nullptr);
  return AdExprRef{expr.graph, expr.graph->make_unary(NodeKind::Tan, expr.id)};
}

inline AdExprRef log1p(const AdExprRef &expr) {
  assert(expr.graph != nullptr);
  return AdExprRef{expr.graph,
                   expr.graph->make_unary(NodeKind::Log1p, expr.id)};
}

inline AdExprRef log(const AdExprRef &expr) {
  assert(expr.graph != nullptr);
  return AdExprRef{expr.graph, expr.graph->make_unary(NodeKind::Log, expr.id)};
}

inline AdExprRef expm1(const AdExprRef &expr) {
  assert(expr.graph != nullptr);
  return AdExprRef{expr.graph,
                   expr.graph->make_unary(NodeKind::Expm1, expr.id)};
}

inline AdExprRef pow(const AdExprRef &lhs, const AdExprRef &rhs) {
  assert(lhs.graph != nullptr);
  assert(rhs.graph != nullptr);
  assert(lhs.graph == rhs.graph);
  return AdExprRef{lhs.graph,
                   lhs.graph->make_binary(NodeKind::Pow, lhs.id, rhs.id)};
}

template <typename T>
  requires std::convertible_to<T, double>
inline AdExprRef pow(const AdExprRef &lhs, T rhs) {
  assert(lhs.graph != nullptr);
  return pow(lhs, constant(*lhs.graph, static_cast<double>(rhs)));
}

template <typename T>
  requires std::convertible_to<T, double>
inline AdExprRef pow(T lhs, const AdExprRef &rhs) {
  assert(rhs.graph != nullptr);
  return pow(constant(*rhs.graph, static_cast<double>(lhs)), rhs);
}

inline AdExprRef atan2(const AdExprRef &lhs, const AdExprRef &rhs) {
  assert(lhs.graph != nullptr);
  assert(rhs.graph != nullptr);
  assert(lhs.graph == rhs.graph);
  return AdExprRef{lhs.graph,
                   lhs.graph->make_binary(NodeKind::Atan2, lhs.id, rhs.id)};
}

template <typename T>
  requires std::convertible_to<T, double>
inline AdExprRef atan2(const AdExprRef &lhs, T rhs) {
  assert(lhs.graph != nullptr);
  return atan2(lhs, constant(*lhs.graph, static_cast<double>(rhs)));
}

template <typename T>
  requires std::convertible_to<T, double>
inline AdExprRef atan2(T lhs, const AdExprRef &rhs) {
  assert(rhs.graph != nullptr);
  return atan2(constant(*rhs.graph, static_cast<double>(lhs)), rhs);
}

inline AdExprRef operator-(const AdExprRef &expr) {
  assert(expr.graph != nullptr);
  return AdExprRef{expr.graph, expr.graph->make_unary(NodeKind::Neg, expr.id)};
}

inline AdExprRef operator-(const AdExprRef &lhs, const AdExprRef &rhs) {
  assert(lhs.graph != nullptr);
  assert(rhs.graph != nullptr);
  assert(lhs.graph == rhs.graph);
  return AdExprRef{lhs.graph,
                   lhs.graph->make_binary(NodeKind::Sub, lhs.id, rhs.id)};
}

template <typename T>
  requires std::convertible_to<T, double>
inline AdExprRef operator-(const AdExprRef &lhs, T rhs) {
  assert(lhs.graph != nullptr);
  return lhs - constant(*lhs.graph, static_cast<double>(rhs));
}

template <typename T>
  requires std::convertible_to<T, double>
inline AdExprRef operator-(T lhs, const AdExprRef &rhs) {
  assert(rhs.graph != nullptr);
  return constant(*rhs.graph, static_cast<double>(lhs)) - rhs;
}

inline AdExprRef operator+(const AdExprRef &lhs, const AdExprRef &rhs) {
  assert(lhs.graph != nullptr);
  assert(rhs.graph != nullptr);
  assert(lhs.graph == rhs.graph);
  return AdExprRef{lhs.graph,
                   lhs.graph->make_binary(NodeKind::Add, lhs.id, rhs.id)};
}

template <typename T>
  requires std::convertible_to<T, double>
inline AdExprRef operator+(const AdExprRef &lhs, T rhs) {
  assert(lhs.graph != nullptr);
  return lhs + constant(*lhs.graph, static_cast<double>(rhs));
}

template <typename T>
  requires std::convertible_to<T, double>
inline AdExprRef operator+(T lhs, const AdExprRef &rhs) {
  assert(rhs.graph != nullptr);
  return constant(*rhs.graph, static_cast<double>(lhs)) + rhs;
}

inline AdExprRef operator*(const AdExprRef &lhs, const AdExprRef &rhs) {
  assert(lhs.graph != nullptr);
  assert(rhs.graph != nullptr);
  assert(lhs.graph == rhs.graph);
  return AdExprRef{lhs.graph,
                   lhs.graph->make_binary(NodeKind::Mul, lhs.id, rhs.id)};
}

template <typename T>
  requires std::convertible_to<T, double>
inline AdExprRef operator*(const AdExprRef &lhs, T rhs) {
  assert(lhs.graph != nullptr);
  return lhs * constant(*lhs.graph, static_cast<double>(rhs));
}

template <typename T>
  requires std::convertible_to<T, double>
inline AdExprRef operator*(T lhs, const AdExprRef &rhs) {
  assert(rhs.graph != nullptr);
  return constant(*rhs.graph, static_cast<double>(lhs)) * rhs;
}

inline AdExprRef operator/(const AdExprRef &lhs, const AdExprRef &rhs) {
  assert(lhs.graph != nullptr);
  assert(rhs.graph != nullptr);
  assert(lhs.graph == rhs.graph);
  return AdExprRef{lhs.graph,
                   lhs.graph->make_binary(NodeKind::Div, lhs.id, rhs.id)};
}

template <typename T>
  requires std::convertible_to<T, double>
inline AdExprRef operator/(const AdExprRef &lhs, T rhs) {
  assert(lhs.graph != nullptr);
  return lhs / constant(*lhs.graph, static_cast<double>(rhs));
}

template <typename T>
  requires std::convertible_to<T, double>
inline AdExprRef operator/(T lhs, const AdExprRef &rhs) {
  assert(rhs.graph != nullptr);
  return constant(*rhs.graph, static_cast<double>(lhs)) / rhs;
}

} // namespace levmar::detail
