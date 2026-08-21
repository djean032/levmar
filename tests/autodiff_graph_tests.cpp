#include <levmar/internal/evaluation.h>

#include <bit>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

void fail(const std::string &message) { throw std::runtime_error(message); }

void expect_true(bool condition, const std::string &what) {
  if (!condition) {
    fail(what);
  }
}

void expect_success(const ErrorOrVoid &result, const std::string &what) {
  if (!result) {
    fail(what + ": " + result.error().message);
  }
}

template <class T>
void expect_equal(const T &actual, const T &expected, const std::string &what) {
  if (!(actual == expected)) {
    std::ostringstream message;
    message << what;
    fail(message.str());
  }
}

void expect_close(double actual, double expected, double atol, double rtol,
                  const std::string &what) {
  if (std::abs(actual - expected) > atol + rtol * std::abs(expected)) {
    std::ostringstream message;
    message << std::setprecision(17) << what << ": got " << actual
            << ", expected " << expected;
    fail(message.str());
  }
}

void test_node_key_equality_and_hash() {
  NodeKey a{.kind = NodeKind::Mul,
            .a = 1,
            .b = 2,
            .parameter_index = 0,
            .literal_bits = 0};
  NodeKey b = a;
  NodeKey c{.kind = NodeKind::Add,
            .a = 1,
            .b = 2,
            .parameter_index = 0,
            .literal_bits = 0};

  NodeKeyHash hash;
  expect_true(a == b, "equal NodeKeys should compare equal");
  expect_true(!(a == c), "different NodeKinds should compare unequal");
  expect_equal(hash(a), hash(b), "equal NodeKeys should hash equally");
}

void test_constant_interning() {
  AdGraph graph;

  const NodeId one_a = graph.make_constant(1.0);
  const NodeId one_b = graph.make_constant(1.0);
  const NodeId two = graph.make_constant(2.0);
  const NodeId plus_zero = graph.make_constant(0.0);
  const NodeId minus_zero = graph.make_constant(-0.0);

  expect_equal(one_a, one_b, "same constant should intern to same node");
  expect_true(one_a != two, "different constants should not intern together");
  expect_true(plus_zero != minus_zero,
              "signed zero constants should remain distinct");

  expect_equal(graph.nodes[one_a].literal_bits,
               std::bit_cast<std::uint64_t>(1.0),
               "constant node should store exact literal bits");
}

void test_variable_interning() {
  AdGraph graph;

  const NodeId x0_a = graph.make_variable(0);
  const NodeId x0_b = graph.make_variable(0);
  const NodeId x1 = graph.make_variable(1);

  expect_equal(x0_a, x0_b, "same variable index should intern to same node");
  expect_true(x0_a != x1,
              "different variable indices should not intern together");
  expect_equal(graph.nodes[x1].parameter_index, Index{1},
               "variable node should store parameter index");
}

void test_unary_interning() {
  AdGraph graph;
  const NodeId x0 = graph.make_variable(0);

  const NodeId neg_a = graph.make_unary(NodeKind::Neg, x0);
  const NodeId neg_b = graph.make_unary(NodeKind::Neg, x0);
  const NodeId exp_a = graph.make_unary(NodeKind::Exp, x0);

  expect_equal(neg_a, neg_b, "same unary node should intern to same node");
  expect_true(neg_a != exp_a,
              "different unary kinds should not intern together");
  expect_equal(graph.nodes[neg_a].a, x0, "unary node should store child in a");
  expect_equal(graph.nodes[neg_a].b, kInvalidNode,
               "unary node should keep b invalid");
}

void test_binary_interning_and_commutative_normalization() {
  AdGraph graph;
  const NodeId x0 = graph.make_variable(0);
  const NodeId x1 = graph.make_variable(1);

  const NodeId add_ab = graph.make_binary(NodeKind::Add, x0, x1);
  const NodeId add_ba = graph.make_binary(NodeKind::Add, x1, x0);
  const NodeId mul_ab = graph.make_binary(NodeKind::Mul, x0, x1);
  const NodeId mul_ba = graph.make_binary(NodeKind::Mul, x1, x0);
  const NodeId sub_ab = graph.make_binary(NodeKind::Sub, x0, x1);
  const NodeId sub_ba = graph.make_binary(NodeKind::Sub, x1, x0);

  expect_equal(add_ab, add_ba,
               "Add should normalize child order and intern commutatively");
  expect_equal(mul_ab, mul_ba,
               "Mul should normalize child order and intern commutatively");
  expect_true(sub_ab != sub_ba,
              "Sub should preserve child order and not intern commutatively");

  expect_equal(graph.nodes[add_ab].a, std::min(x0, x1),
               "commutative binary node should store normalized left child");
  expect_equal(graph.nodes[add_ab].b, std::max(x0, x1),
               "commutative binary node should store normalized right child");
}

void test_topological_and_slot_invariants() {
  AdGraph graph;
  const NodeId x0 = graph.make_variable(0);
  const NodeId x1 = graph.make_variable(1);
  const NodeId sum = graph.make_binary(NodeKind::Add, x0, x1);
  const NodeId neg = graph.make_unary(NodeKind::Neg, sum);
  const NodeId ex = graph.make_unary(NodeKind::Exp, neg);

  expect_true(graph.nodes[sum].a < sum && graph.nodes[sum].b < sum,
              "binary nodes should be created after their children");
  expect_true(graph.nodes[neg].a < neg,
              "unary nodes should be created after their child");
  expect_true(graph.nodes[ex].a < ex,
              "nested unary nodes should be created after their child");

  for (NodeId id = 0; id < graph.nodes.size(); ++id) {
    expect_equal(graph.nodes[id].value_slot, id,
                 "value_slot should equal node id in the first implementation");
  }
}

void test_expr_ref_helpers() {
  AdGraph graph;

  const auto c = constant(graph, 1.0);
  const auto x = variable(graph, 2);

  expect_true(c.graph == &graph,
              "constant helper should preserve graph pointer");
  expect_true(x.graph == &graph,
              "variable helper should preserve graph pointer");
  expect_equal(c.id, graph.make_constant(1.0),
               "constant helper should reuse interned constant node");
  expect_equal(x.id, graph.make_variable(2),
               "variable helper should reuse interned variable node");
}

void test_unary_operator_builders() {
  AdGraph graph;
  const auto x = variable(graph, 0);

  const auto neg = -x;
  const auto ex = exp(x);

  expect_true(neg.graph == &graph, "unary minus should preserve graph pointer");
  expect_true(ex.graph == &graph, "exp should preserve graph pointer");
  expect_equal(graph.nodes[neg.id].kind, NodeKind::Neg,
               "unary minus should build Neg node");
  expect_equal(graph.nodes[ex.id].kind, NodeKind::Exp,
               "exp should build Exp node");
  expect_equal(graph.nodes[neg.id].a, x.id,
               "unary minus should use expression as child");
  expect_equal(graph.nodes[ex.id].a, x.id,
               "exp should use expression as child");
}

void test_binary_operator_builders() {
  AdGraph graph;
  const auto x0 = variable(graph, 0);
  const auto x1 = variable(graph, 1);

  const auto add = x0 + x1;
  const auto sub = x0 - x1;
  const auto mul = x0 * x1;

  expect_true(add.graph == &graph, "addition should preserve graph pointer");
  expect_true(sub.graph == &graph, "subtraction should preserve graph pointer");
  expect_true(mul.graph == &graph,
              "multiplication should preserve graph pointer");

  expect_equal(graph.nodes[add.id].kind, NodeKind::Add,
               "addition should build Add node");
  expect_equal(graph.nodes[sub.id].kind, NodeKind::Sub,
               "subtraction should build Sub node");
  expect_equal(graph.nodes[mul.id].kind, NodeKind::Mul,
               "multiplication should build Mul node");
}

void test_scalar_lifting() {
  AdGraph graph;
  const auto x = variable(graph, 0);

  const auto add_rhs = x + 1.0;
  const auto add_lhs = 1.0 + x;
  const auto sub_rhs = x - 2.0;
  const auto sub_lhs = 2.0 - x;
  const auto mul_rhs = x * 3.0;
  const auto mul_lhs = 3.0 * x;

  expect_true(add_rhs.graph == &graph,
              "rhs scalar lifting should preserve graph pointer for add");
  expect_true(add_lhs.graph == &graph,
              "lhs scalar lifting should preserve graph pointer for add");
  expect_true(sub_rhs.graph == &graph,
              "rhs scalar lifting should preserve graph pointer for sub");
  expect_true(sub_lhs.graph == &graph,
              "lhs scalar lifting should preserve graph pointer for sub");
  expect_true(mul_rhs.graph == &graph,
              "rhs scalar lifting should preserve graph pointer for mul");
  expect_true(mul_lhs.graph == &graph,
              "lhs scalar lifting should preserve graph pointer for mul");

  const NodeId one = graph.make_constant(1.0);
  const NodeId two = graph.make_constant(2.0);
  const NodeId three = graph.make_constant(3.0);

  expect_true(graph.nodes[add_rhs.id].a == std::min(x.id, one) &&
                  graph.nodes[add_rhs.id].b == std::max(x.id, one),
              "add with lifted rhs constant should normalize commutatively");
  expect_true(graph.nodes[add_lhs.id].a == std::min(one, x.id) &&
                  graph.nodes[add_lhs.id].b == std::max(one, x.id),
              "add with lifted lhs constant should normalize commutatively");

  expect_equal(graph.nodes[sub_rhs.id].b, two,
               "rhs scalar lifting for subtraction should use lifted constant "
               "as rhs child");
  expect_equal(graph.nodes[sub_lhs.id].a, two,
               "lhs scalar lifting for subtraction should use lifted constant "
               "as lhs child");

  expect_true(graph.nodes[mul_rhs.id].a == std::min(x.id, three) &&
                  graph.nodes[mul_rhs.id].b == std::max(x.id, three),
              "mul with lifted rhs constant should normalize commutatively");
  expect_true(graph.nodes[mul_lhs.id].a == std::min(three, x.id) &&
                  graph.nodes[mul_lhs.id].b == std::max(three, x.id),
              "mul with lifted lhs constant should normalize commutatively");
}

void test_operator_interning() {
  AdGraph graph;
  const auto x0 = variable(graph, 0);
  const auto x1 = variable(graph, 1);

  const auto add_ab = x0 + x1;
  const auto add_ba = x1 + x0;
  const auto mul_ab = x0 * x1;
  const auto mul_ba = x1 * x0;
  const auto sub_ab = x0 - x1;
  const auto sub_ba = x1 - x0;
  const auto repeated = x0 * (1.0 - exp(-x1 * 2.0)) - 3.0;
  const auto repeated_again = x0 * (1.0 - exp(-x1 * 2.0)) - 3.0;

  expect_equal(add_ab.id, add_ba.id,
               "operator+ should reuse interned commutative Add nodes");
  expect_equal(mul_ab.id, mul_ba.id,
               "operator* should reuse interned commutative Mul nodes");
  expect_true(sub_ab.id != sub_ba.id,
              "operator- should preserve operand order");
  expect_equal(
      repeated.id, repeated_again.id,
      "rebuilding the same expression should intern to the same root node");
}

void test_forward_pass_single_root_value() {
  AdGraph graph;
  const auto x0 = variable(graph, 0);
  const auto x1 = variable(graph, 1);
  constexpr double t = 2.5;
  constexpr double y = 0.75;

  const auto root = x0 * (1.0 - exp(-x1 * t)) - y;

  const std::array<double, 2> parameters{1.25, 0.4};
  AdEvalContext ctx;
  expect_success(
      forward_pass(
          graph, ConstVectorView<2>(parameters.data(), parameters.size()), ctx),
      "forward pass should succeed for a single root");

  const double expected =
      parameters[0] * (1.0 - std::exp(-parameters[1] * t)) - y;
  expect_close(ctx.values[root.id], expected, 1e-12, 1e-12,
               "forward pass should compute the correct single-root value");
}

void test_forward_pass_multi_root_values() {
  AdGraph graph;
  const auto x0 = variable(graph, 0);
  const auto x1 = variable(graph, 1);
  constexpr double t = 1.75;
  constexpr double y = 0.2;

  const auto e = exp(-x1 * t);
  const auto r0 = x0 * e;
  const auto r1 = 1.0 - e;
  const auto r2 = x0 * e - y;

  const std::array<double, 2> parameters{2.0, 0.3};
  AdEvalContext ctx;
  expect_success(
      forward_pass(
          graph, ConstVectorView<2>(parameters.data(), parameters.size()), ctx),
      "forward pass should succeed for multiple roots");

  const double expected_e = std::exp(-parameters[1] * t);
  const double expected_r0 = parameters[0] * expected_e;
  const double expected_r1 = 1.0 - expected_e;
  const double expected_r2 = expected_r0 - y;

  expect_close(ctx.values[e.id], expected_e, 1e-12, 1e-12,
               "forward pass should compute shared subexpression value once");
  expect_close(ctx.values[r0.id], expected_r0, 1e-12, 1e-12,
               "forward pass should compute the first root value");
  expect_close(ctx.values[r1.id], expected_r1, 1e-12, 1e-12,
               "forward pass should compute the second root value");
  expect_close(ctx.values[r2.id], expected_r2, 1e-12, 1e-12,
               "forward pass should compute the third root value");
}

void test_forward_tangent_pass_single_root() {
  AdGraph graph;
  const auto x0 = variable(graph, 0);
  const auto x1 = variable(graph, 1);
  constexpr double t = 2.5;
  constexpr double y = 0.75;

  const auto root = x0 * (1.0 - exp(-x1 * t)) - y;

  const std::array<double, 2> parameters{1.25, 0.4};
  AdEvalContext ctx;
  expect_success(
      forward_pass(
          graph, ConstVectorView<2>(parameters.data(), parameters.size()), ctx),
      "forward pass should succeed before tangent evaluation");

  const double e = std::exp(-parameters[1] * t);

  forward_tangent_pass(graph, 0, ctx);
  expect_close(ctx.tangents[root.id], 1.0 - e, 1e-12, 1e-12,
               "forward tangent pass should compute df/dx0");

  forward_tangent_pass(graph, 1, ctx);
  expect_close(ctx.tangents[root.id], parameters[0] * t * e, 1e-12, 1e-12,
               "forward tangent pass should compute df/dx1");
}

void test_forward_tangent_pass_multi_root() {
  AdGraph graph;
  const auto x0 = variable(graph, 0);
  const auto x1 = variable(graph, 1);
  constexpr double t = 1.75;
  constexpr double y = 0.2;

  const auto e = exp(-x1 * t);
  const auto r0 = x0 * e;
  const auto r1 = 1.0 - e;
  const auto r2 = x0 * e - y;

  const std::array<double, 2> parameters{2.0, 0.3};
  AdEvalContext ctx;
  expect_success(
      forward_pass(
          graph, ConstVectorView<2>(parameters.data(), parameters.size()), ctx),
      "forward pass should succeed before multi-root tangent evaluation");

  const double ev = std::exp(-parameters[1] * t);

  forward_tangent_pass(graph, 0, ctx);
  expect_close(ctx.tangents[r0.id], ev, 1e-12, 1e-12,
               "forward tangent pass should compute dr0/dx0");
  expect_close(ctx.tangents[r1.id], 0.0, 1e-12, 1e-12,
               "forward tangent pass should compute dr1/dx0");
  expect_close(ctx.tangents[r2.id], ev, 1e-12, 1e-12,
               "forward tangent pass should compute dr2/dx0");

  forward_tangent_pass(graph, 1, ctx);
  expect_close(ctx.tangents[r0.id], -parameters[0] * t * ev, 1e-12, 1e-12,
               "forward tangent pass should compute dr0/dx1");
  expect_close(ctx.tangents[r1.id], t * ev, 1e-12, 1e-12,
               "forward tangent pass should compute dr1/dx1");
  expect_close(ctx.tangents[r2.id], -parameters[0] * t * ev, 1e-12, 1e-12,
               "forward tangent pass should compute dr2/dx1");
}

void test_evaluate_roots_forward_single_root() {
  AdGraph graph;
  const auto x0 = variable(graph, 0);
  const auto x1 = variable(graph, 1);
  constexpr double t = 2.5;
  constexpr double y = 0.75;

  const auto root = x0 * (1.0 - exp(-x1 * t)) - y;
  const std::array<NodeId, 1> roots{root.id};

  const std::array<double, 2> parameters{1.25, 0.4};
  VectorStorage<1> residuals;
  MatrixStorage<1, 2> jacobian;
  AdEvalContext ctx;

  expect_success(evaluate_roots_forward(
                     graph, std::span<const NodeId>(roots.data(), roots.size()),
                     ConstVectorView<2>(parameters.data(), parameters.size()),
                     residuals.view(), jacobian.view(), ctx),
                 "single-root forward evaluation should succeed");

  const double e = std::exp(-parameters[1] * t);
  const double expected_value = parameters[0] * (1.0 - e) - y;

  expect_close(residuals[0], expected_value, 1e-12, 1e-12,
               "evaluate_roots_forward should write the single residual value");
  expect_close(jacobian(0, 0), 1.0 - e, 1e-12, 1e-12,
               "evaluate_roots_forward should write df/dx0 into J");
  expect_close(jacobian(0, 1), parameters[0] * t * e, 1e-12, 1e-12,
               "evaluate_roots_forward should write df/dx1 into J");
}

void test_evaluate_roots_forward_multi_root() {
  AdGraph graph;
  const auto x0 = variable(graph, 0);
  const auto x1 = variable(graph, 1);
  constexpr double t = 1.75;
  constexpr double y = 0.2;

  const auto e = exp(-x1 * t);
  const auto r0 = x0 * e;
  const auto r1 = 1.0 - e;
  const auto r2 = x0 * e - y;
  const std::array<NodeId, 3> roots{r0.id, r1.id, r2.id};

  const std::array<double, 2> parameters{2.0, 0.3};
  VectorStorage<3> residuals;
  MatrixStorage<3, 2> jacobian;
  AdEvalContext ctx;

  expect_success(evaluate_roots_forward(
                     graph, std::span<const NodeId>(roots.data(), roots.size()),
                     ConstVectorView<2>(parameters.data(), parameters.size()),
                     residuals.view(), jacobian.view(), ctx),
                 "multi-root forward evaluation should succeed");

  const double ev = std::exp(-parameters[1] * t);
  const double expected_r0 = parameters[0] * ev;
  const double expected_r1 = 1.0 - ev;
  const double expected_r2 = expected_r0 - y;

  expect_close(residuals[0], expected_r0, 1e-12, 1e-12,
               "evaluate_roots_forward should write r0");
  expect_close(residuals[1], expected_r1, 1e-12, 1e-12,
               "evaluate_roots_forward should write r1");
  expect_close(residuals[2], expected_r2, 1e-12, 1e-12,
               "evaluate_roots_forward should write r2");

  expect_close(jacobian(0, 0), ev, 1e-12, 1e-12,
               "evaluate_roots_forward should write dr0/dx0");
  expect_close(jacobian(1, 0), 0.0, 1e-12, 1e-12,
               "evaluate_roots_forward should write dr1/dx0");
  expect_close(jacobian(2, 0), ev, 1e-12, 1e-12,
               "evaluate_roots_forward should write dr2/dx0");

  expect_close(jacobian(0, 1), -parameters[0] * t * ev, 1e-12, 1e-12,
               "evaluate_roots_forward should write dr0/dx1");
  expect_close(jacobian(1, 1), t * ev, 1e-12, 1e-12,
               "evaluate_roots_forward should write dr1/dx1");
  expect_close(jacobian(2, 1), -parameters[0] * t * ev, 1e-12, 1e-12,
               "evaluate_roots_forward should write dr2/dx1");
}

void test_evaluate_roots_forward_dynamic_tail_block() {
  AdGraph graph;
  AdExprRef root = variable(graph, 0);
  for (Index j = 1; j < 9; ++j) {
    root = root + variable(graph, j);
  }

  const std::array<NodeId, 1> roots{root.id};
  VectorStorage<std::dynamic_extent> parameters;
  parameters.resize(9);
  for (Index j = 0; j < parameters.size(); ++j) {
    parameters[j] = static_cast<double>(j + 1);
  }

  VectorStorage<1> residuals;
  MatrixStorage<1, std::dynamic_extent> jacobian;
  jacobian.resize(parameters.size());
  AdEvalContext ctx;

  expect_success(
      evaluate_roots_forward(
          graph, std::span<const NodeId>(roots.data(), roots.size()),
          ConstVectorView<std::dynamic_extent>(parameters.data(),
                                               parameters.size()),
          residuals.view(), jacobian.view(), ctx),
      "dynamic tangent blocks should evaluate a final partial block");

  expect_close(residuals[0], 45.0, 1e-12, 1e-12,
               "dynamic tangent blocks should evaluate the residual");
  for (Index j = 0; j < parameters.size(); ++j) {
    expect_close(
        jacobian(0, j), 1.0, 1e-12, 1e-12,
        "dynamic tangent blocks should evaluate every Jacobian column");
  }

  forward_tangent_block(graph, 8, 1, ctx);
  expect_close(ctx.tangent_blocks[root.id][0], 1.0, 1e-12, 1e-12,
               "final block should seed its active lane");
  for (Index lane = 1; lane < kTangentBlockWidth; ++lane) {
    expect_close(ctx.tangent_blocks[root.id][lane], 0.0, 1e-12, 1e-12,
                 "final block should clear inactive lanes");
  }
}

void test_evaluate_jacobian_autodiff_static_problem() {
  constexpr double t0 = 2.5;
  constexpr double y0 = 0.75;
  constexpr double t1 = 1.25;
  constexpr double y1 = 0.10;

  auto residual = [=]<class Scalar>(ConstVectorView<2, Scalar> x,
                                    VectorView<2, Scalar> r) -> ErrorOrVoid {
    using std::exp;
    r[0] = x[0] * (1.0 - exp(-x[1] * t0)) - y0;
    r[1] = x[0] * (1.0 - exp(-x[1] * t1)) - y1;
    return {};
  };
  auto problem = make_problem<2, 2>(residual);
  Options options;
  options.jacobian_mode = JacobianMode::AutoDiff;
  Result result;
  LMWorkspace<2, 2> workspace;
  const std::array<double, 2> x0{1.25, 0.4};
  LMSolveContext context(problem, options, result, workspace, x0);

  std::ranges::copy(context.x, workspace.x_current.view().begin());
  workspace.r.fill(-123.0);
  workspace.J.fill(-456.0);

  if (auto jacobian_result =
          evaluate_jacobian(context, "autodiff static jacobian");
      !jacobian_result) {
    fail(jacobian_result.error().message);
  }

  const double e0 = std::exp(-x0[1] * t0);
  const double e1 = std::exp(-x0[1] * t1);

  expect_close(
      workspace.r[0], x0[0] * (1.0 - e0) - y0, 1e-12, 1e-12,
      "AutoDiff jacobian evaluation should also fill static residual row 0");
  expect_close(
      workspace.r[1], x0[0] * (1.0 - e1) - y1, 1e-12, 1e-12,
      "AutoDiff jacobian evaluation should also fill static residual row 1");

  expect_close(workspace.J(0, 0), 1.0 - e0, 1e-12, 1e-12,
               "AutoDiff jacobian evaluation should fill static d r0 / d x0");
  expect_close(workspace.J(1, 0), 1.0 - e1, 1e-12, 1e-12,
               "AutoDiff jacobian evaluation should fill static d r1 / d x0");
  expect_close(workspace.J(0, 1), x0[0] * t0 * e0, 1e-12, 1e-12,
               "AutoDiff jacobian evaluation should fill static d r0 / d x1");
  expect_close(workspace.J(1, 1), x0[0] * t1 * e1, 1e-12, 1e-12,
               "AutoDiff jacobian evaluation should fill static d r1 / d x1");

  expect_equal(result.function_evaluations, Index{1},
               "AutoDiff jacobian evaluation should increment "
               "function_evaluations once for static problems");
  expect_equal(result.jacobian_evaluations, Index{1},
               "AutoDiff jacobian evaluation should increment "
               "jacobian_evaluations once for static problems");
}

void test_evaluate_jacobian_autodiff_dynamic_problem() {
  constexpr double t0 = 2.5;
  constexpr double y0 = 0.75;
  constexpr double t1 = 1.25;
  constexpr double y1 = 0.10;
  constexpr double t2 = 0.50;
  constexpr double y2 = -0.20;

  auto residual =
      [=]<class Scalar>(
          ConstVectorView<std::dynamic_extent, Scalar> x,
          VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
    using std::exp;
    expect_equal(x.size(), Index{2},
                 "dynamic autodiff residual should receive two parameters");
    expect_equal(
        r.size(), Index{3},
        "dynamic autodiff residual should receive three residual slots");

    r[0] = x[0] * (1.0 - exp(-x[1] * t0)) - y0;
    r[1] = x[0] * (1.0 - exp(-x[1] * t1)) - y1;
    r[2] = x[0] * (1.0 - exp(-x[1] * t2)) - y2;
    return {};
  };
  auto problem = make_dynamic_problem(3, 2, residual);
  Options options;
  options.jacobian_mode = JacobianMode::AutoDiff;
  Result result;
  LMWorkspace<std::dynamic_extent, std::dynamic_extent> workspace;
  const std::vector<double> x0{1.25, 0.4};
  LMSolveContext context(problem, options, result, workspace, x0);

  std::ranges::copy(context.x, workspace.x_current.view().begin());
  workspace.r.fill(-123.0);
  workspace.J.fill(-456.0);

  if (auto jacobian_result =
          evaluate_jacobian(context, "autodiff dynamic jacobian");
      !jacobian_result) {
    fail(jacobian_result.error().message);
  }

  const double e0 = std::exp(-x0[1] * t0);
  const double e1 = std::exp(-x0[1] * t1);
  const double e2 = std::exp(-x0[1] * t2);

  expect_equal(workspace.r.size(), Index{3},
               "dynamic AutoDiff jacobian evaluation should preserve residual "
               "workspace size");
  expect_equal(workspace.J.rows(), Index{3},
               "dynamic AutoDiff jacobian evaluation should preserve Jacobian "
               "row count");
  expect_equal(workspace.J.cols(), Index{2},
               "dynamic AutoDiff jacobian evaluation should preserve Jacobian "
               "column count");

  expect_close(
      workspace.r[0], x0[0] * (1.0 - e0) - y0, 1e-12, 1e-12,
      "AutoDiff jacobian evaluation should also fill dynamic residual row 0");
  expect_close(
      workspace.r[1], x0[0] * (1.0 - e1) - y1, 1e-12, 1e-12,
      "AutoDiff jacobian evaluation should also fill dynamic residual row 1");
  expect_close(
      workspace.r[2], x0[0] * (1.0 - e2) - y2, 1e-12, 1e-12,
      "AutoDiff jacobian evaluation should also fill dynamic residual row 2");

  expect_close(workspace.J(0, 0), 1.0 - e0, 1e-12, 1e-12,
               "AutoDiff jacobian evaluation should fill dynamic d r0 / d x0");
  expect_close(workspace.J(1, 0), 1.0 - e1, 1e-12, 1e-12,
               "AutoDiff jacobian evaluation should fill dynamic d r1 / d x0");
  expect_close(workspace.J(2, 0), 1.0 - e2, 1e-12, 1e-12,
               "AutoDiff jacobian evaluation should fill dynamic d r2 / d x0");
  expect_close(workspace.J(0, 1), x0[0] * t0 * e0, 1e-12, 1e-12,
               "AutoDiff jacobian evaluation should fill dynamic d r0 / d x1");
  expect_close(workspace.J(1, 1), x0[0] * t1 * e1, 1e-12, 1e-12,
               "AutoDiff jacobian evaluation should fill dynamic d r1 / d x1");
  expect_close(workspace.J(2, 1), x0[0] * t2 * e2, 1e-12, 1e-12,
               "AutoDiff jacobian evaluation should fill dynamic d r2 / d x1");

  expect_equal(result.function_evaluations, Index{1},
               "AutoDiff jacobian evaluation should increment "
               "function_evaluations once for dynamic problems");
  expect_equal(result.jacobian_evaluations, Index{1},
               "AutoDiff jacobian evaluation should increment "
               "jacobian_evaluations once for dynamic problems");
}

void test_evaluate_jacobian_autodiff_static_cache() {
  Index recordings = 0;
  auto residual =
      [&recordings]<class Scalar>(ConstVectorView<2, Scalar> x,
                                  VectorView<2, Scalar> r) -> ErrorOrVoid {
    ++recordings;
    r[0] = x[0] * x[0] + 2.0 * x[1];
    r[1] = x[0] * x[1];
    return {};
  };
  auto problem = make_problem<2, 2>(residual);
  Options options;
  options.jacobian_mode = JacobianMode::AutoDiff;
  Result result;
  LMWorkspace<2, 2> workspace;
  const std::array<double, 2> x0{2.0, 3.0};
  LMSolveContext context(problem, options, result, workspace, x0);

  std::ranges::copy(context.x, workspace.x_current.view().begin());
  if (auto jacobian_result = evaluate_jacobian(context); !jacobian_result) {
    fail(jacobian_result.error().message);
  }

  workspace.x_current[0] = 4.0;
  workspace.x_current[1] = 5.0;
  if (auto jacobian_result = evaluate_jacobian(context); !jacobian_result) {
    fail(jacobian_result.error().message);
  }

  expect_close(workspace.r[0], 26.0, 1e-12, 1e-12,
               "cached static AutoDiff should evaluate the new residual");
  expect_close(workspace.r[1], 20.0, 1e-12, 1e-12,
               "cached static AutoDiff should evaluate the new residual");
  expect_close(workspace.J(0, 0), 8.0, 1e-12, 1e-12,
               "cached static AutoDiff should update dr0/dx0");
  expect_close(workspace.J(1, 0), 5.0, 1e-12, 1e-12,
               "cached static AutoDiff should update dr1/dx0");
  expect_close(workspace.J(0, 1), 2.0, 1e-12, 1e-12,
               "cached static AutoDiff should update dr0/dx1");
  expect_close(workspace.J(1, 1), 4.0, 1e-12, 1e-12,
               "cached static AutoDiff should update dr1/dx1");
  expect_equal(recordings, Index{1},
               "cached static AutoDiff should record the graph once");
  expect_equal(result.function_evaluations, Index{2},
               "cached static AutoDiff should count both evaluations");
  expect_equal(result.jacobian_evaluations, Index{2},
               "cached static AutoDiff should count both Jacobians");
}

void test_evaluate_jacobian_autodiff_dynamic_cache() {
  Index recordings = 0;
  auto residual =
      [&recordings]<class Scalar>(
          ConstVectorView<std::dynamic_extent, Scalar> x,
          VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
    ++recordings;
    r[0] = x[0] * x[0];
    r[1] = x[0] * x[1];
    r[2] = x[1] + 1.0;
    return {};
  };
  auto problem = make_dynamic_problem(3, 2, residual);
  Options options;
  options.jacobian_mode = JacobianMode::AutoDiff;
  Result result;
  LMWorkspace<std::dynamic_extent, std::dynamic_extent> workspace;
  const std::vector<double> x0{2.0, 3.0};
  LMSolveContext context(problem, options, result, workspace, x0);

  std::ranges::copy(context.x, workspace.x_current.view().begin());
  if (auto jacobian_result = evaluate_jacobian(context); !jacobian_result) {
    fail(jacobian_result.error().message);
  }

  workspace.x_current[0] = 4.0;
  workspace.x_current[1] = 5.0;
  if (auto jacobian_result = evaluate_jacobian(context); !jacobian_result) {
    fail(jacobian_result.error().message);
  }

  expect_close(workspace.r[0], 16.0, 1e-12, 1e-12,
               "cached dynamic AutoDiff should evaluate the new residual");
  expect_close(workspace.r[1], 20.0, 1e-12, 1e-12,
               "cached dynamic AutoDiff should evaluate the new residual");
  expect_close(workspace.r[2], 6.0, 1e-12, 1e-12,
               "cached dynamic AutoDiff should evaluate the new residual");
  expect_close(workspace.J(0, 0), 8.0, 1e-12, 1e-12,
               "cached dynamic AutoDiff should update dr0/dx0");
  expect_close(workspace.J(1, 0), 5.0, 1e-12, 1e-12,
               "cached dynamic AutoDiff should update dr1/dx0");
  expect_close(workspace.J(2, 0), 0.0, 1e-12, 1e-12,
               "cached dynamic AutoDiff should update dr2/dx0");
  expect_close(workspace.J(0, 1), 0.0, 1e-12, 1e-12,
               "cached dynamic AutoDiff should update dr0/dx1");
  expect_close(workspace.J(1, 1), 4.0, 1e-12, 1e-12,
               "cached dynamic AutoDiff should update dr1/dx1");
  expect_close(workspace.J(2, 1), 1.0, 1e-12, 1e-12,
               "cached dynamic AutoDiff should update dr2/dx1");
  expect_equal(recordings, Index{1},
               "cached dynamic AutoDiff should record the graph once");
  expect_equal(result.function_evaluations, Index{2},
               "cached dynamic AutoDiff should count both evaluations");
  expect_equal(result.jacobian_evaluations, Index{2},
               "cached dynamic AutoDiff should count both Jacobians");
}

void test_evaluate_jacobian_autodiff_literal_residual() {
  auto residual = []<class Scalar>(ConstVectorView<1, Scalar> x,
                                   VectorView<2, Scalar> r) -> ErrorOrVoid {
    r[0] = 0.0;
    r[1] = x[0];
    return {};
  };
  auto problem = make_problem<2, 1>(residual);
  Options options;
  options.jacobian_mode = JacobianMode::AutoDiff;
  Result result;
  LMWorkspace<2, 1> workspace;
  const std::array<double, 1> x0{1.25};
  LMSolveContext context(problem, options, result, workspace, x0);

  std::ranges::copy(context.x, workspace.x_current.view().begin());
  if (auto jacobian_result = evaluate_jacobian(context); !jacobian_result) {
    fail(jacobian_result.error().message);
  }

  expect_close(workspace.r[0], 0.0, 1e-12, 1e-12,
               "AutoDiff should lift literal residuals into the graph");
  expect_close(workspace.r[1], x0[0], 1e-12, 1e-12,
               "AutoDiff should evaluate expression residuals");
  expect_close(workspace.J(0, 0), 0.0, 1e-12, 1e-12,
               "literal residual derivative should be zero");
  expect_close(workspace.J(1, 0), 1.0, 1e-12, 1e-12,
               "identity residual derivative should be one");
}

void test_evaluate_jacobian_autodiff_double_only_residual() {
  auto residual = [](ConstVectorView<1> x, VectorView<1> r) -> ErrorOrVoid {
    r[0] = x[0];
    return {};
  };
  auto problem = make_problem<1, 1>(residual);
  Options options;
  options.jacobian_mode = JacobianMode::AutoDiff;
  Result result;
  LMWorkspace<1, 1> workspace;
  const std::array<double, 1> x0{1.25};
  LMSolveContext context(problem, options, result, workspace, x0);

  std::ranges::copy(context.x, workspace.x_current.view().begin());
  const auto jacobian_result = evaluate_jacobian(context);

  expect_true(!jacobian_result,
              "AutoDiff should reject a double-only residual callback");
  expect_equal(jacobian_result.error().code, ErrorCode::InvalidProblem,
               "double-only AutoDiff error should identify an invalid problem");
  expect_true(
      jacobian_result.error().message.find("scalar-generic") !=
          std::string::npos,
      "double-only AutoDiff error should explain the callback requirement");
}

void test_evaluate_jacobian_autodiff_mixed_extents() {
  auto residual =
      []<class Scalar>(
          ConstVectorView<1, Scalar> x,
          VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
    r[0] = x[0];
    r[1] = 2.0 * x[0];
    return {};
  };
  auto problem = make_problem_dynamic_residuals<1>(2, residual);
  Options options;
  options.jacobian_mode = JacobianMode::AutoDiff;
  Result result;
  LMWorkspace<std::dynamic_extent, 1> workspace;
  const std::array<double, 1> x0{1.25};
  LMSolveContext context(problem, options, result, workspace, x0);

  std::ranges::copy(context.x, workspace.x_current.view().begin());
  if (auto jacobian_result = evaluate_jacobian(context); !jacobian_result) {
    fail(jacobian_result.error().message);
  }

  expect_close(workspace.r[0], x0[0], 1e-12, 1e-12,
               "mixed-extent AutoDiff should evaluate the first residual");
  expect_close(workspace.r[1], 2.0 * x0[0], 1e-12, 1e-12,
               "mixed-extent AutoDiff should evaluate the second residual");
  expect_close(workspace.J(0, 0), 1.0, 1e-12, 1e-12,
               "mixed-extent AutoDiff should differentiate the first residual");
  expect_close(
      workspace.J(1, 0), 2.0, 1e-12, 1e-12,
      "mixed-extent AutoDiff should differentiate the second residual");
}

void test_extended_primitives_forward() {
  AdGraph graph;
  const auto x = variable(graph, 0);
  const auto y = variable(graph, 1);
  const auto div = x / y;
  const auto logarithm = log(x);
  const auto square_root = sqrt(x);
  const auto sine = sin(x);
  const auto cosine = cos(x);
  const auto tangent = tan(y);
  const auto logarithm_one_plus = log1p(x);
  const auto exponential_minus_one = expm1(y);
  const auto angle = atan2(y, x);
  const auto power = pow(x, y);

  const std::array<NodeId, 10> roots{
      div.id,    logarithm.id, square_root.id,        sine.id,
      cosine.id, tangent.id,   logarithm_one_plus.id, exponential_minus_one.id,
      angle.id,  power.id};
  const std::array<double, 2> parameters{1.5, 0.75};
  VectorStorage<10> residuals;
  MatrixStorage<10, 2> jacobian;
  AdEvalContext ctx;

  if (auto result = evaluate_roots_forward(
          graph, std::span<const NodeId>(roots.data(), roots.size()),
          ConstVectorView<2>(parameters.data(), parameters.size()),
          residuals.view(), jacobian.view(), ctx);
      !result) {
    fail(result.error().message);
  }

  const double x_value = parameters[0];
  const double y_value = parameters[1];
  const double atan2_denominator = x_value * x_value + y_value * y_value;
  const double power_value = std::pow(x_value, y_value);

  expect_close(residuals[0], x_value / y_value, 1e-12, 1e-12,
               "division primal value");
  expect_close(jacobian(0, 0), 1.0 / y_value, 1e-12, 1e-12,
               "division derivative with respect to numerator");
  expect_close(jacobian(0, 1), -x_value / (y_value * y_value), 1e-12, 1e-12,
               "division derivative with respect to denominator");
  expect_close(residuals[1], std::log(x_value), 1e-12, 1e-12,
               "log primal value");
  expect_close(jacobian(1, 0), 1.0 / x_value, 1e-12, 1e-12, "log derivative");
  expect_close(residuals[2], std::sqrt(x_value), 1e-12, 1e-12,
               "sqrt primal value");
  expect_close(jacobian(2, 0), 1.0 / (2.0 * std::sqrt(x_value)), 1e-12, 1e-12,
               "sqrt derivative");
  expect_close(residuals[3], std::sin(x_value), 1e-12, 1e-12,
               "sin primal value");
  expect_close(jacobian(3, 0), std::cos(x_value), 1e-12, 1e-12,
               "sin derivative");
  expect_close(residuals[4], std::cos(x_value), 1e-12, 1e-12,
               "cos primal value");
  expect_close(jacobian(4, 0), -std::sin(x_value), 1e-12, 1e-12,
               "cos derivative");
  expect_close(residuals[5], std::tan(y_value), 1e-12, 1e-12,
               "tan primal value");
  expect_close(jacobian(5, 1), 1.0 / std::pow(std::cos(y_value), 2), 1e-12,
               1e-12, "tan derivative");
  expect_close(residuals[6], std::log1p(x_value), 1e-12, 1e-12,
               "log1p primal value");
  expect_close(jacobian(6, 0), 1.0 / (1.0 + x_value), 1e-12, 1e-12,
               "log1p derivative");
  expect_close(residuals[7], std::expm1(y_value), 1e-12, 1e-12,
               "expm1 primal value");
  expect_close(jacobian(7, 1), std::exp(y_value), 1e-12, 1e-12,
               "expm1 derivative");
  expect_close(residuals[8], std::atan2(y_value, x_value), 1e-12, 1e-12,
               "atan2 primal value");
  expect_close(jacobian(8, 0), -y_value / atan2_denominator, 1e-12, 1e-12,
               "atan2 derivative with respect to x");
  expect_close(jacobian(8, 1), x_value / atan2_denominator, 1e-12, 1e-12,
               "atan2 derivative with respect to y");
  expect_close(residuals[9], power_value, 1e-12, 1e-12, "pow primal value");
  expect_close(jacobian(9, 0), power_value * y_value / x_value, 1e-12, 1e-12,
               "pow derivative with respect to base");
  expect_close(jacobian(9, 1), power_value * std::log(x_value), 1e-12, 1e-12,
               "pow derivative with respect to exponent");
}

void test_extended_primitives_scalar_generic_residual() {
  auto residual = []<class Scalar>(ConstVectorView<2, Scalar> x,
                                   VectorView<3, Scalar> r) -> ErrorOrVoid {
    using std::atan2;
    using std::cos;
    using std::expm1;
    using std::log;
    using std::log1p;
    using std::pow;
    using std::sin;
    using std::sqrt;
    using std::tan;

    r[0] = x[0] / x[1] + log(x[0]) + sqrt(x[0]);
    r[1] = sin(x[0]) + cos(x[0]) + tan(x[1]) + log1p(x[0]) + expm1(x[1]);
    r[2] =
        atan2(x[1], 2.0) + atan2(2.0, x[1]) + pow(x[0], 2.0) + pow(2.0, x[1]);
    return {};
  };
  auto problem = make_problem<3, 2>(residual);
  Options options;
  options.jacobian_mode = JacobianMode::AutoDiff;
  Result result;
  LMWorkspace<3, 2> workspace;
  const std::array<double, 2> x0{1.5, 0.75};
  LMSolveContext context(problem, options, result, workspace, x0);

  std::ranges::copy(context.x, workspace.x_current.view().begin());
  if (auto jacobian_result = evaluate_jacobian(context); !jacobian_result) {
    fail(jacobian_result.error().message);
  }

  expect_close(workspace.r[2],
               std::atan2(x0[1], 2.0) + std::atan2(2.0, x0[1]) +
                   std::pow(x0[0], 2.0) + std::pow(2.0, x0[1]),
               1e-12, 1e-12,
               "scalar-generic residual should use lifted atan2 and pow");
}

void test_pow_domain_errors() {
  AdGraph graph;
  const auto base = variable(graph, 0);
  const auto root = pow(base, 2.0);
  const std::array<double, 1> parameters{0.0};
  AdEvalContext ctx;

  const auto forward_result = forward_pass(
      graph, ConstVectorView<1>(parameters.data(), parameters.size()), ctx);
  expect_true(!forward_result,
              "pow with a zero base should fail in the primal pass");
  expect_equal(forward_result.error().code, ErrorCode::NumericalFailure,
               "pow primal failure should be numerical");

  const std::array<NodeId, 1> roots{root.id};
  VectorStorage<1> residuals;
  MatrixStorage<1, 1> jacobian;
  const auto roots_result = evaluate_roots_forward(
      graph, std::span<const NodeId>(roots.data(), roots.size()),
      ConstVectorView<1>(parameters.data(), parameters.size()),
      residuals.view(), jacobian.view(), ctx);
  expect_true(!roots_result,
              "pow domain failure should propagate through root evaluation");
  expect_equal(roots_result.error().code, ErrorCode::NumericalFailure,
               "root evaluation should preserve pow numerical failure");

  auto residual = []<class Scalar>(ConstVectorView<1, Scalar> x,
                                   VectorView<1, Scalar> r) -> ErrorOrVoid {
    using std::pow;
    r[0] = pow(x[0], 2.0);
    return {};
  };
  auto problem = make_problem<1, 1>(residual);
  Options options;
  options.jacobian_mode = JacobianMode::AutoDiff;
  Result result;
  LMWorkspace<1, 1> workspace;
  LMSolveContext context(problem, options, result, workspace, parameters);

  std::ranges::copy(context.x, workspace.x_current.view().begin());
  const auto jacobian_result = evaluate_jacobian(context);
  expect_true(
      !jacobian_result,
      "pow domain failure should propagate through Jacobian evaluation");
  expect_equal(jacobian_result.error().code, ErrorCode::NumericalFailure,
               "Jacobian evaluation should preserve pow numerical failure");
}

} // namespace

int main() {
  test_node_key_equality_and_hash();
  test_constant_interning();
  test_variable_interning();
  test_unary_interning();
  test_binary_interning_and_commutative_normalization();
  test_topological_and_slot_invariants();
  test_expr_ref_helpers();
  test_unary_operator_builders();
  test_binary_operator_builders();
  test_scalar_lifting();
  test_operator_interning();
  test_forward_pass_single_root_value();
  test_forward_pass_multi_root_values();
  test_forward_tangent_pass_single_root();
  test_forward_tangent_pass_multi_root();
  test_evaluate_roots_forward_single_root();
  test_evaluate_roots_forward_multi_root();
  test_evaluate_roots_forward_dynamic_tail_block();
  test_evaluate_jacobian_autodiff_static_problem();
  test_evaluate_jacobian_autodiff_dynamic_problem();
  test_evaluate_jacobian_autodiff_static_cache();
  test_evaluate_jacobian_autodiff_dynamic_cache();
  test_evaluate_jacobian_autodiff_literal_residual();
  test_evaluate_jacobian_autodiff_double_only_residual();
  test_evaluate_jacobian_autodiff_mixed_extents();
  test_extended_primitives_forward();
  test_extended_primitives_scalar_generic_residual();
  test_pow_domain_errors();
  return 0;
}
