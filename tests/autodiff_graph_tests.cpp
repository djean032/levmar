#include <levmar/internal/autodiff/dual.h>
#include <levmar/internal/autodiff/dual_forward.h>
#include <levmar/internal/evaluation.h>
#include <levmar/internal/solver.h>
#include <levmar/internal/solver_validation.h>

#include <array>
#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace levmar;
using namespace levmar::detail;

struct StaticResidual {
  template <class Scalar>
  ErrorOrVoid operator()(ConstVectorView<1, Scalar>,
                         VectorView<2, Scalar>) const {
    return {};
  }
};

struct StaticJacobian {
  ErrorOrVoid operator()(ConstVectorView<1>, MatrixView<2, 1>) const {
    return {};
  }
};

struct DoubleOnlyResidual {
  ErrorOrVoid operator()(ConstVectorView<1>, VectorView<2>) const { return {}; }
};

struct UnderdeterminedResidual {
  template <class Scalar>
  ErrorOrVoid operator()(ConstVectorView<2, Scalar>,
                         VectorView<1, Scalar>) const {
    return {};
  }
};

struct UnderdeterminedJacobian {
  ErrorOrVoid operator()(ConstVectorView<2>, MatrixView<1, 2>) const {
    return {};
  }
};

using ValidUserContext = LMSolveContext<2, 1, StaticResidual, StaticJacobian>;
using MissingUserJacobianContext =
    LMSolveContext<2, 1, StaticResidual, NoJacobian>;
using DoubleOnlyAutoDiffContext =
    LMSolveContext<2, 1, DoubleOnlyResidual, NoJacobian>;
using UnderdeterminedContext =
    LMSolveContext<1, 2, UnderdeterminedResidual, UnderdeterminedJacobian>;

using AutoDiffSolverPolicy = SolverPolicy<AutoDiffJacobian, LevenbergMarquardt,
                                          DampedQr, SquaredLoss, NoScaling>;
using ForwardDifferenceSolverPolicy =
    SolverPolicy<ForwardDifferenceJacobian, LevenbergMarquardt, DampedQr,
                 SquaredLoss, NoScaling>;
using CentralDifferenceSolverPolicy =
    SolverPolicy<CentralDifferenceJacobian, LevenbergMarquardt, DampedQr,
                 SquaredLoss, NoScaling>;
using PivotedQrSolverPolicy =
    SolverPolicy<UserJacobian, LevenbergMarquardt, PivotedHouseholderQr,
                 SquaredLoss, NoScaling>;

static_assert(
    kStaticSolverConfigurationValid<DefaultSolverPolicy, ValidUserContext>);
static_assert(
    kStaticSolverConfigurationValid<AutoDiffSolverPolicy, ValidUserContext>);
static_assert(!kStaticSolverConfigurationValid<DefaultSolverPolicy,
                                               MissingUserJacobianContext>);
static_assert(!kStaticSolverConfigurationValid<AutoDiffSolverPolicy,
                                               DoubleOnlyAutoDiffContext>);
static_assert(!kStaticSolverConfigurationValid<DefaultSolverPolicy,
                                               UnderdeterminedContext>);
static_assert([] {
  validate_static_solver_configuration<DefaultSolverPolicy, ValidUserContext>();
  validate_static_solver_configuration<AutoDiffSolverPolicy,
                                       ValidUserContext>();
  return true;
}());

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

void test_runtime_option_validation() {
  const auto expect_valid = []<class Policy>(Options options,
                                             const std::string &what) {
    const auto result = validate_runtime_options<Policy>(options);
    expect_true(result.has_value(), what + " should succeed");
  };

  const auto expect_invalid = []<class Policy>(Options options,
                                               const std::string &what) {
    const auto result = validate_runtime_options<Policy>(options);
    expect_true(!result, what + " should fail");
    expect_equal(result.error().code, ErrorCode::InvalidProblem,
                 what + " should report an invalid problem");
  };

  const std::array<double, 3> invalid_tolerances{
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity(), -1.0};
  const std::array<double, 4> invalid_lambdas{
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity(), 0.0, -1.0};
  const std::array<double, 4> invalid_rank_multipliers{
      std::numeric_limits<double>::quiet_NaN(),
      std::numeric_limits<double>::infinity(), 0.0, -1.0};

  expect_valid.template operator()<DefaultSolverPolicy>(
      Options{}, "default user-Jacobian options");
  expect_valid.template operator()<AutoDiffSolverPolicy>(
      Options{}, "default AutoDiff options");

  Options zero_iterations;
  zero_iterations.max_iterations = 0;
  expect_valid.template operator()<DefaultSolverPolicy>(
      zero_iterations, "zero iteration budget");

  Options zero_function_evaluations;
  zero_function_evaluations.max_function_evaluations = 0;
  expect_invalid.template operator()<DefaultSolverPolicy>(
      zero_function_evaluations, "zero function-evaluation budget");

  for (const double value : invalid_tolerances) {
    Options options;
    options.gradient_tolerance = value;
    expect_invalid.template operator()<DefaultSolverPolicy>(
        options, "invalid gradient tolerance");

    options = {};
    options.step_tolerance = value;
    expect_invalid.template operator()<DefaultSolverPolicy>(
        options, "invalid step tolerance");

    options = {};
    options.cost_tolerance = value;
    expect_invalid.template operator()<DefaultSolverPolicy>(
        options, "invalid cost tolerance");

    options = {};
    options.relative_cost_tolerance = value;
    expect_invalid.template operator()<DefaultSolverPolicy>(
        options, "invalid relative cost tolerance");
  }

  for (const double value : invalid_tolerances) {
    Options options;
    options.finite_difference_step = value;
    expect_invalid.template operator()<ForwardDifferenceSolverPolicy>(
        options, "invalid forward-difference step");
    expect_invalid.template operator()<CentralDifferenceSolverPolicy>(
        options, "invalid central-difference step");
  }

  Options unused_finite_difference_step;
  unused_finite_difference_step.finite_difference_step =
      std::numeric_limits<double>::quiet_NaN();
  expect_valid.template operator()<DefaultSolverPolicy>(
      unused_finite_difference_step,
      "unused user-Jacobian finite-difference step");
  expect_valid.template operator()<AutoDiffSolverPolicy>(
      unused_finite_difference_step, "unused AutoDiff finite-difference step");

  for (const double value : invalid_lambdas) {
    Options options;
    options.lm.min_lambda = value;
    expect_invalid.template operator()<DefaultSolverPolicy>(
        options, "invalid minimum lambda");

    options = {};
    options.lm.max_lambda = value;
    expect_invalid.template operator()<DefaultSolverPolicy>(
        options, "invalid maximum lambda");
  }

  Options unordered_lambdas;
  unordered_lambdas.lm.min_lambda = 2.0;
  unordered_lambdas.lm.max_lambda = 1.0;
  expect_invalid.template operator()<DefaultSolverPolicy>(unordered_lambdas,
                                                          "unordered lambdas");

  for (const double value : invalid_rank_multipliers) {
    Options options;
    options.lm.rank_tolerance_multiplier = value;
    expect_invalid.template operator()<DefaultSolverPolicy>(
        options, "invalid rank tolerance multiplier");
  }

  for (const double value : invalid_lambdas) {
    Options options;
    options.lm.min_trust_region_radius = value;
    expect_invalid.template operator()<DefaultSolverPolicy>(
        options, "invalid minimum trust-region radius");

    options = {};
    options.lm.initial_trust_region_radius = value;
    expect_invalid.template operator()<DefaultSolverPolicy>(
        options, "invalid initial trust-region radius");

    options = {};
    options.lm.max_trust_region_radius = value;
    expect_invalid.template operator()<DefaultSolverPolicy>(
        options, "invalid maximum trust-region radius");
  }

  Options minimum_above_initial_radius;
  minimum_above_initial_radius.lm.min_trust_region_radius = 2.0;
  minimum_above_initial_radius.lm.initial_trust_region_radius = 1.0;
  minimum_above_initial_radius.lm.max_trust_region_radius = 3.0;
  expect_invalid.template operator()<DefaultSolverPolicy>(
      minimum_above_initial_radius,
      "minimum trust-region radius above initial radius");

  Options initial_above_maximum_radius;
  initial_above_maximum_radius.lm.min_trust_region_radius = 1.0;
  initial_above_maximum_radius.lm.initial_trust_region_radius = 3.0;
  initial_above_maximum_radius.lm.max_trust_region_radius = 2.0;
  expect_invalid.template operator()<DefaultSolverPolicy>(
      initial_above_maximum_radius,
      "initial trust-region radius above maximum radius");

  Options valid_trust_region_radii;
  valid_trust_region_radii.lm.min_trust_region_radius = 1e-6;
  valid_trust_region_radii.lm.initial_trust_region_radius = 1e2;
  valid_trust_region_radii.lm.max_trust_region_radius = 1e8;
  expect_valid.template operator()<DefaultSolverPolicy>(
      valid_trust_region_radii, "ordered trust-region radii");
}

void test_dual_arithmetic_and_math() {
  DualEvaluationState state;
  Dual<2> a{1.5};
  a.derivatives[0] = 1.0;
  a.state = &state;
  Dual<2> b{0.75};
  b.derivatives[1] = 1.0;
  b.state = &state;

  const auto expect_dual = [&](const Dual<2> &actual, double value,
                               double derivative_a, double derivative_b,
                               const std::string &what) {
    expect_close(actual.value, value, 1e-12, 1e-12, what + " value");
    expect_close(actual.derivatives[0], derivative_a, 1e-12, 1e-12,
                 what + " derivative 0");
    expect_close(actual.derivatives[1], derivative_b, 1e-12, 1e-12,
                 what + " derivative 1");
    expect_equal(actual.state, &state, what + " state");
  };

  expect_dual(a + b, 2.25, 1.0, 1.0, "dual addition");
  expect_dual(a - b, 0.75, 1.0, -1.0, "dual subtraction");
  expect_dual(-a, -1.5, -1.0, 0.0, "dual negation");
  expect_dual(a * b, 1.125, 0.75, 1.5, "dual multiplication");
  expect_dual(a / b, 2.0, 1.0 / b.value, -a.value / (b.value * b.value),
              "dual division");

  expect_dual(exp(a), std::exp(a.value), std::exp(a.value), 0.0, "dual exp");
  expect_dual(log(a), std::log(a.value), 1.0 / a.value, 0.0, "dual log");
  expect_dual(log1p(a), std::log1p(a.value), 1.0 / (1.0 + a.value), 0.0,
              "dual log1p");
  expect_dual(expm1(a), std::expm1(a.value), std::exp(a.value), 0.0,
              "dual expm1");
  expect_dual(sin(a), std::sin(a.value), std::cos(a.value), 0.0, "dual sin");
  expect_dual(cos(a), std::cos(a.value), -std::sin(a.value), 0.0, "dual cos");
  expect_dual(tan(a), std::tan(a.value),
              1.0 / (std::cos(a.value) * std::cos(a.value)), 0.0, "dual tan");
  expect_dual(sqrt(a), std::sqrt(a.value), 1.0 / (2.0 * std::sqrt(a.value)),
              0.0, "dual sqrt");

  const double atan2_denominator = a.value * a.value + b.value * b.value;
  expect_dual(atan2(a, b), std::atan2(a.value, b.value),
              b.value / atan2_denominator, -a.value / atan2_denominator,
              "dual atan2");

  const double power = std::pow(a.value, b.value);
  expect_dual(pow(a, b), power, power * b.value / a.value,
              power * std::log(a.value), "dual pow");
}

void test_dual_pow_domain_error() {
  DualEvaluationState state;
  Dual<1> base{-2.0};
  base.state = &state;
  Dual<1> exponent{0.5};
  exponent.state = &state;

  const Dual<1> result = pow(base, exponent);

  expect_true(state.pow_domain_error,
              "dual pow should record a nonpositive base");
  expect_equal(result.state, &state,
               "invalid dual pow should preserve evaluation state");
}

void test_evaluate_residual_dual_static() {
  auto residual = []<class Scalar>(ConstVectorView<2, Scalar> x,
                                   VectorView<2, Scalar> r) -> ErrorOrVoid {
    using std::exp;
    r[0] = x[0] * (1.0 - exp(-x[1]));
    r[1] = x[0] / x[1];
    return {};
  };
  const std::array<double, 2> parameters{1.5, 0.75};
  VectorStorage<2> residuals;
  MatrixStorage<2, 2> jacobian;
  DualEvalContext<2, 2> context;

  expect_success(
      evaluate_residual_dual(
          residual, ConstVectorView<2>(parameters.data(), parameters.size()),
          residuals.view(), jacobian.view(), context, "static dual evaluation"),
      "direct dual static evaluation should succeed");

  const double exponential = std::exp(-parameters[1]);
  expect_close(residuals[0], parameters[0] * (1.0 - exponential), 1e-12, 1e-12,
               "direct dual static residual 0");
  expect_close(residuals[1], parameters[0] / parameters[1], 1e-12, 1e-12,
               "direct dual static residual 1");
  expect_close(jacobian(0, 0), 1.0 - exponential, 1e-12, 1e-12,
               "direct dual static dr0/dx0");
  expect_close(jacobian(1, 0), 1.0 / parameters[1], 1e-12, 1e-12,
               "direct dual static dr1/dx0");
  expect_close(jacobian(0, 1), parameters[0] * exponential, 1e-12, 1e-12,
               "direct dual static dr0/dx1");
  expect_close(jacobian(1, 1), -parameters[0] / (parameters[1] * parameters[1]),
               1e-12, 1e-12, "direct dual static dr1/dx1");

  const std::array<double, 2> changed_parameters{2.0, 0.5};
  expect_success(
      evaluate_residual_dual(residual,
                             ConstVectorView<2>(changed_parameters.data(),
                                                changed_parameters.size()),
                             residuals.view(), jacobian.view(), context),
      "direct dual static evaluation should reuse context");
  expect_close(residuals[1], changed_parameters[0] / changed_parameters[1],
               1e-12, 1e-12,
               "direct dual static context reuse should update residuals");
  expect_close(
      jacobian(1, 1),
      -changed_parameters[0] / (changed_parameters[1] * changed_parameters[1]),
      1e-12, 1e-12, "direct dual static context reuse should update Jacobian");
}

void test_evaluate_residual_dual_dynamic_residuals() {
  auto residual =
      []<class Scalar>(
          ConstVectorView<2, Scalar> x,
          VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
    using std::sin;
    r[0] = x[0] + x[1];
    r[1] = x[0] * x[1];
    r[2] = sin(x[0]);
    return {};
  };
  const std::array<double, 2> parameters{1.5, 0.75};
  VectorStorage<std::dynamic_extent> residuals;
  residuals.resize(3);
  MatrixStorage<std::dynamic_extent, 2> jacobian;
  jacobian.resize(3);
  DualEvalContext<std::dynamic_extent, 2> context;

  expect_success(evaluate_residual_dual(
                     residual,
                     ConstVectorView<2>(parameters.data(), parameters.size()),
                     residuals.view(), jacobian.view(), context),
                 "direct dual dynamic-residual evaluation should succeed");

  expect_equal(context.residuals.size(), Index{3},
               "direct dual context should resize dynamic residual storage");
  expect_close(residuals[0], parameters[0] + parameters[1], 1e-12, 1e-12,
               "direct dual dynamic residual 0");
  expect_close(residuals[1], parameters[0] * parameters[1], 1e-12, 1e-12,
               "direct dual dynamic residual 1");
  expect_close(residuals[2], std::sin(parameters[0]), 1e-12, 1e-12,
               "direct dual dynamic residual 2");
  expect_close(jacobian(0, 0), 1.0, 1e-12, 1e-12,
               "direct dual dynamic dr0/dx0");
  expect_close(jacobian(1, 0), parameters[1], 1e-12, 1e-12,
               "direct dual dynamic dr1/dx0");
  expect_close(jacobian(2, 0), std::cos(parameters[0]), 1e-12, 1e-12,
               "direct dual dynamic dr2/dx0");
  expect_close(jacobian(0, 1), 1.0, 1e-12, 1e-12,
               "direct dual dynamic dr0/dx1");
  expect_close(jacobian(1, 1), parameters[0], 1e-12, 1e-12,
               "direct dual dynamic dr1/dx1");
  expect_close(jacobian(2, 1), 0.0, 1e-12, 1e-12,
               "direct dual dynamic dr2/dx1");
}

void test_evaluate_residual_dual_errors() {
  auto callback_error = [](ConstVectorView<1, Dual<1>>,
                           VectorView<1, Dual<1>>) -> ErrorOrVoid {
    return std::unexpected(
        Error{ErrorCode::UserFunctionError, "callback error"});
  };
  auto pow_error = []<class Scalar>(ConstVectorView<1, Scalar> x,
                                    VectorView<1, Scalar> r) -> ErrorOrVoid {
    using std::pow;
    r[0] = pow(-2.0, x[0]);
    return {};
  };
  auto identity = []<class Scalar>(ConstVectorView<1, Scalar> x,
                                   VectorView<1, Scalar> r) -> ErrorOrVoid {
    r[0] = x[0];
    return {};
  };
  const std::array<double, 1> parameters{0.5};
  VectorStorage<1> residuals;
  MatrixStorage<1, 1> jacobian;
  DualEvalContext<1, 1> context;
  const auto parameter_view =
      ConstVectorView<1>(parameters.data(), parameters.size());

  const auto callback_result =
      evaluate_residual_dual(callback_error, parameter_view, residuals.view(),
                             jacobian.view(), context, "direct dual callback");
  expect_true(!callback_result,
              "direct dual should propagate callback failures");
  expect_equal(callback_result.error().code, ErrorCode::UserFunctionError,
               "direct dual callback error should preserve its code");
  expect_equal(callback_result.error().message,
               std::string("direct dual callback failed: callback error"),
               "direct dual callback error should include its context");

  const auto pow_result = evaluate_residual_dual(
      pow_error, parameter_view, residuals.view(), jacobian.view(), context);
  expect_true(!pow_result, "direct dual should reject invalid pow bases");
  expect_equal(pow_result.error().code, ErrorCode::NumericalFailure,
               "direct dual invalid pow should be numerical");

  expect_success(evaluate_residual_dual(identity, parameter_view,
                                        residuals.view(), jacobian.view(),
                                        context),
                 "direct dual context should reset pow errors");
  expect_close(residuals[0], parameters[0], 1e-12, 1e-12,
               "direct dual context reuse should evaluate after pow failure");
  expect_close(jacobian(0, 0), 1.0, 1e-12, 1e-12,
               "direct dual context reuse should reset derivative state");
}

void test_graph_and_direct_dual_policies_agree() {
  Index recordings = 0;
  auto residual =
      [&recordings]<class Scalar>(ConstVectorView<2, Scalar> x,
                                  VectorView<3, Scalar> r) -> ErrorOrVoid {
    if constexpr (std::same_as<Scalar, AdExprRef>) {
      ++recordings;
    }
    using std::exp;
    using std::log1p;
    using std::sin;
    r[0] = x[0] * exp(-x[1]);
    r[1] = log1p(x[0] * x[0]) + sin(x[1]);
    r[2] = x[0] / x[1];
    return {};
  };
  auto problem = make_problem<3, 2>(residual);
  Options options;
  const std::array<double, 2> x0{1.5, 0.75};

  Result graph_result;
  LMWorkspace<3, 2> graph_workspace;
  LMSolveContext graph_context(problem, options, graph_result, graph_workspace,
                               x0);
  std::ranges::copy(graph_context.x, graph_workspace.x_current.view().begin());
  expect_success(GraphAutoDiffJacobian::activate(graph_context, "graph policy"),
                 "graph policy activation should succeed");
  expect_success(GraphAutoDiffJacobian::evaluate(graph_context, "graph policy"),
                 "graph policy evaluation should succeed");

  Result dual_result;
  LMWorkspace<3, 2> dual_workspace;
  LMSolveContext dual_context(problem, options, dual_result, dual_workspace,
                              x0);
  std::ranges::copy(dual_context.x, dual_workspace.x_current.view().begin());
  expect_success(DirectDualJacobian::evaluate(dual_context, "dual policy"),
                 "direct-dual policy evaluation should succeed");

  for (Index i = 0; i < 3; ++i) {
    expect_close(graph_workspace.r[i], dual_workspace.r[i], 1e-12, 1e-12,
                 "graph and direct-dual residuals should agree");
    for (Index j = 0; j < 2; ++j) {
      expect_close(graph_workspace.J(i, j), dual_workspace.J(i, j), 1e-12,
                   1e-12, "graph and direct-dual Jacobians should agree");
    }
  }

  graph_workspace.x_current[0] = 2.0;
  graph_workspace.x_current[1] = 0.5;
  dual_workspace.x_current[0] = 2.0;
  dual_workspace.x_current[1] = 0.5;
  expect_success(GraphAutoDiffJacobian::evaluate(graph_context, "graph policy"),
                 "cached graph policy evaluation should succeed");
  expect_success(DirectDualJacobian::evaluate(dual_context, "dual policy"),
                 "reused direct-dual policy evaluation should succeed");

  expect_equal(recordings, Index{1},
               "graph policy should record the residual exactly once");
  expect_true(graph_context.autodiff_cache.recorded,
              "graph policy should retain its recorded graph");
  expect_equal(dual_context.autodiff_cache.graph.nodes.size(), Index{0},
               "direct-dual policy should not record a graph");
  for (Index i = 0; i < 3; ++i) {
    expect_close(graph_workspace.r[i], dual_workspace.r[i], 1e-12, 1e-12,
                 "reused graph and direct-dual residuals should agree");
    for (Index j = 0; j < 2; ++j) {
      expect_close(graph_workspace.J(i, j), dual_workspace.J(i, j), 1e-12,
                   1e-12,
                   "reused graph and direct-dual Jacobians should agree");
    }
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
  Result result;
  LMWorkspace<2, 2> workspace;
  const std::array<double, 2> x0{1.25, 0.4};
  LMSolveContext context(problem, options, result, workspace, x0);

  std::ranges::copy(context.x, workspace.x_current.view().begin());
  workspace.r.fill(-123.0);
  workspace.J.fill(-456.0);

  if (auto jacobian_result = evaluate_jacobian<AutoDiffJacobian>(
          context, "autodiff static jacobian");
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
  Result result;
  LMWorkspace<std::dynamic_extent, std::dynamic_extent> workspace;
  const std::vector<double> x0{1.25, 0.4};
  LMSolveContext context(problem, options, result, workspace, x0);

  std::ranges::copy(context.x, workspace.x_current.view().begin());
  workspace.r.fill(-123.0);
  workspace.J.fill(-456.0);

  if (auto jacobian_result = evaluate_jacobian<AutoDiffJacobian>(
          context, "autodiff dynamic jacobian");
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

void test_evaluate_jacobian_autodiff_static_direct_dual() {
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
  Result result;
  LMWorkspace<2, 2> workspace;
  const std::array<double, 2> x0{2.0, 3.0};
  LMSolveContext context(problem, options, result, workspace, x0);

  std::ranges::copy(context.x, workspace.x_current.view().begin());
  if (auto jacobian_result = evaluate_jacobian<AutoDiffJacobian>(context);
      !jacobian_result) {
    fail(jacobian_result.error().message);
  }

  workspace.x_current[0] = 4.0;
  workspace.x_current[1] = 5.0;
  if (auto jacobian_result = evaluate_jacobian<AutoDiffJacobian>(context);
      !jacobian_result) {
    fail(jacobian_result.error().message);
  }

  expect_close(workspace.r[0], 26.0, 1e-12, 1e-12,
               "direct-dual AutoDiff should evaluate the new residual");
  expect_close(workspace.r[1], 20.0, 1e-12, 1e-12,
               "direct-dual AutoDiff should evaluate the new residual");
  expect_close(workspace.J(0, 0), 8.0, 1e-12, 1e-12,
               "direct-dual AutoDiff should update dr0/dx0");
  expect_close(workspace.J(1, 0), 5.0, 1e-12, 1e-12,
               "direct-dual AutoDiff should update dr1/dx0");
  expect_close(workspace.J(0, 1), 2.0, 1e-12, 1e-12,
               "direct-dual AutoDiff should update dr0/dx1");
  expect_close(workspace.J(1, 1), 4.0, 1e-12, 1e-12,
               "direct-dual AutoDiff should update dr1/dx1");
  expect_equal(
      recordings, Index{2},
      "direct-dual AutoDiff should invoke the residual per evaluation");
  expect_equal(context.autodiff_cache.graph.nodes.size(), Index{0},
               "direct-dual AutoDiff should not record a graph");
  expect_equal(result.function_evaluations, Index{2},
               "direct-dual AutoDiff should count both evaluations");
  expect_equal(result.jacobian_evaluations, Index{2},
               "direct-dual AutoDiff should count both Jacobians");
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
  Result result;
  LMWorkspace<std::dynamic_extent, std::dynamic_extent> workspace;
  const std::vector<double> x0{2.0, 3.0};
  LMSolveContext context(problem, options, result, workspace, x0);

  std::ranges::copy(context.x, workspace.x_current.view().begin());
  if (auto jacobian_result = evaluate_jacobian<AutoDiffJacobian>(context);
      !jacobian_result) {
    fail(jacobian_result.error().message);
  }

  workspace.x_current[0] = 4.0;
  workspace.x_current[1] = 5.0;
  if (auto jacobian_result = evaluate_jacobian<AutoDiffJacobian>(context);
      !jacobian_result) {
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
  Result result;
  LMWorkspace<2, 1> workspace;
  const std::array<double, 1> x0{1.25};
  LMSolveContext context(problem, options, result, workspace, x0);

  std::ranges::copy(context.x, workspace.x_current.view().begin());
  if (auto jacobian_result = evaluate_jacobian<AutoDiffJacobian>(context);
      !jacobian_result) {
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
  Result result;
  LMWorkspace<1, 1> workspace;
  const std::array<double, 1> x0{1.25};
  LMSolveContext context(problem, options, result, workspace, x0);

  std::ranges::copy(context.x, workspace.x_current.view().begin());
  const auto jacobian_result = evaluate_jacobian<AutoDiffJacobian>(context);

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
  Result result;
  LMWorkspace<std::dynamic_extent, 1> workspace;
  const std::array<double, 1> x0{1.25};
  LMSolveContext context(problem, options, result, workspace, x0);

  std::ranges::copy(context.x, workspace.x_current.view().begin());
  if (auto jacobian_result = evaluate_jacobian<AutoDiffJacobian>(context);
      !jacobian_result) {
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
  Result result;
  LMWorkspace<3, 2> workspace;
  const std::array<double, 2> x0{1.5, 0.75};
  LMSolveContext context(problem, options, result, workspace, x0);

  std::ranges::copy(context.x, workspace.x_current.view().begin());
  if (auto jacobian_result = evaluate_jacobian<AutoDiffJacobian>(context);
      !jacobian_result) {
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
  Result result;
  LMWorkspace<1, 1> workspace;
  LMSolveContext context(problem, options, result, workspace, parameters);

  std::ranges::copy(context.x, workspace.x_current.view().begin());
  const auto jacobian_result = evaluate_jacobian<AutoDiffJacobian>(context);
  expect_true(
      !jacobian_result,
      "pow domain failure should propagate through Jacobian evaluation");
  expect_equal(jacobian_result.error().code, ErrorCode::NumericalFailure,
               "Jacobian evaluation should preserve pow numerical failure");
}

template <Index M, Index N>
double augmented_damped_qr_objective(const LMWorkspace<M, N> &work,
                                     double lambda) {
  double objective = 0.0;
  for (Index i = 0; i < work.m; ++i) {
    double residual = work.r[i];
    for (Index j = 0; j < work.n; ++j) {
      residual += work.J(i, j) * work.step[j];
    }
    objective += residual * residual;
  }

  for (Index j = 0; j < work.n; ++j) {
    objective += lambda * work.step[j] * work.step[j];
  }
  return objective;
}

void test_damped_qr_static() {
  LMWorkspace<3, 2> work;
  DampedQrWorkspace<3, 2> qr;

  work.J(0, 0) = 1.0;
  work.J(1, 0) = 0.0;
  work.J(2, 0) = 1.0;
  work.J(0, 1) = 0.0;
  work.J(1, 1) = 1.0;
  work.J(2, 1) = 1.0;
  work.r[0] = 1.0;
  work.r[1] = 2.0;
  work.r[2] = 3.0;

  constexpr double lambda = 1.0;
  expect_true(solve_damped_qr(NoScaling{}, work, qr, lambda),
              "static damped QR should succeed");
  expect_close(work.step[0], -0.875, 1e-12, 1e-12,
               "static damped QR first step component");
  expect_close(work.step[1], -1.375, 1e-12, 1e-12,
               "static damped QR second step component");
  expect_true(augmented_damped_qr_objective(work, lambda) < 14.0,
              "static damped QR should lower the augmented objective");
}

void test_pivoted_householder_qr_static() {
  LMWorkspace<3, 2> work;
  PivotedHouseholderQrWorkspace<3, 2> qr;

  work.J(0, 0) = 1.0;
  work.J(1, 0) = 0.0;
  work.J(2, 0) = 1.0;
  work.J(0, 1) = 0.0;
  work.J(1, 1) = 1.0;
  work.J(2, 1) = 1.0;
  work.r[0] = 1.0;
  work.r[1] = 2.0;
  work.r[2] = 3.0;

  expect_true(prepare_pivoted_householder_qr(NoScaling{}, work, qr, 32.0),
              "static pivoted Householder QR preparation should succeed");
  expect_true(qr.factor_valid,
              "static pivoted Householder QR should cache its factor");
  expect_equal(qr.numerical_rank, Index{2},
               "static pivoted Householder QR rank");

  constexpr double lambda = 1.0;
  expect_true(solve_pivoted_householder_qr(NoScaling{}, work, qr, lambda),
              "static pivoted Householder QR solve should succeed");
  expect_close(work.step[0], -0.875, 1e-12, 1e-12,
               "static pivoted Householder QR first step component");
  expect_close(work.step[1], -1.375, 1e-12, 1e-12,
               "static pivoted Householder QR second step component");

  expect_true(solve_pivoted_householder_qr(NoScaling{}, work, qr, 2.0),
              "static pivoted Householder QR cached solve should succeed");
  expect_true(qr.factor_valid,
              "static pivoted Householder QR should retain its factor");
}

void test_damped_qr_dynamic() {
  LMWorkspace<std::dynamic_extent, std::dynamic_extent> work(3, 2);
  DampedQrWorkspace<std::dynamic_extent, std::dynamic_extent> qr;
  qr.resize(3, 2);

  work.J(0, 0) = 1.0;
  work.J(1, 0) = 0.0;
  work.J(2, 0) = 1.0;
  work.J(0, 1) = 0.0;
  work.J(1, 1) = 1.0;
  work.J(2, 1) = 1.0;
  work.r[0] = 1.0;
  work.r[1] = 2.0;
  work.r[2] = 3.0;

  constexpr double lambda = 1.0;
  expect_true(solve_damped_qr(NoScaling{}, work, qr, lambda),
              "dynamic damped QR should succeed");
  expect_close(work.step[0], -0.875, 1e-12, 1e-12,
               "dynamic damped QR first step component");
  expect_close(work.step[1], -1.375, 1e-12, 1e-12,
               "dynamic damped QR second step component");
  expect_true(augmented_damped_qr_objective(work, lambda) < 14.0,
              "dynamic damped QR should lower the augmented objective");
}

void test_damped_qr_column_scaling_static() {
  LMWorkspace<3, 2> work;
  DampedQrWorkspace<3, 2> qr;

  work.J(0, 0) = 1.0;
  work.J(1, 0) = 0.0;
  work.J(2, 0) = 1.0;
  work.J(0, 1) = 0.0;
  work.J(1, 1) = 1.0;
  work.J(2, 1) = 1.0;
  work.r[0] = 1.0;
  work.r[1] = 2.0;
  work.r[2] = 3.0;
  work.scale[0] = 2.0;
  work.scale[1] = 4.0;
  work.damping_scale[0] = 2.0;
  work.damping_scale[1] = 4.0;

  constexpr double lambda = 1.0;
  expect_true(solve_damped_qr(JacobianColumnScaling{}, work, qr, lambda),
              "scaled damped QR should succeed");
  expect_close(work.step[0], -67.0 / 107.0, 1e-12, 1e-12,
               "scaled damped QR first step component");
  expect_close(work.step[1], -26.0 / 107.0, 1e-12, 1e-12,
               "scaled damped QR second step component");
}

void test_pivoted_householder_qr_column_scaling_static() {
  LMWorkspace<3, 2> work;
  PivotedHouseholderQrWorkspace<3, 2> qr;

  work.J(0, 0) = 1.0;
  work.J(1, 0) = 0.0;
  work.J(2, 0) = 1.0;
  work.J(0, 1) = 0.0;
  work.J(1, 1) = 1.0;
  work.J(2, 1) = 1.0;
  work.r[0] = 1.0;
  work.r[1] = 2.0;
  work.r[2] = 3.0;
  work.scale[0] = 2.0;
  work.scale[1] = 4.0;
  work.damping_scale[0] = 2.0;
  work.damping_scale[1] = 4.0;

  expect_true(
      prepare_pivoted_householder_qr(JacobianColumnScaling{}, work, qr, 32.0),
      "scaled pivoted Householder QR preparation should succeed");
  expect_equal(qr.numerical_rank, Index{2},
               "scaled pivoted Householder QR rank");

  constexpr double lambda = 1.0;
  expect_true(
      solve_pivoted_householder_qr(JacobianColumnScaling{}, work, qr, lambda),
      "scaled pivoted Householder QR solve should succeed");
  expect_close(work.step[0], -67.0 / 107.0, 1e-12, 1e-12,
               "scaled pivoted Householder QR first step component");
  expect_close(work.step[1], -26.0 / 107.0, 1e-12, 1e-12,
               "scaled pivoted Householder QR second step component");
}

void test_pivoted_householder_qr_unpermutation() {
  LMWorkspace<3, 2> damped_work;
  LMWorkspace<3, 2> pivoted_work;
  DampedQrWorkspace<3, 2> damped_qr;
  PivotedHouseholderQrWorkspace<3, 2> pivoted_qr;

  for (auto *work : {&damped_work, &pivoted_work}) {
    work->J[0, 0] = 1.0;
    work->J[1, 0] = 0.0;
    work->J[2, 0] = 1.0;
    work->J[0, 1] = 4.0;
    work->J[1, 1] = 1.0;
    work->J[2, 1] = 5.0;
    work->r[0] = 1.0;
    work->r[1] = 2.0;
    work->r[2] = 3.0;
  }

  constexpr double lambda = 1.0;
  expect_true(solve_damped_qr(NoScaling{}, damped_work, damped_qr, lambda),
              "pivot reference damped QR solve should succeed");
  expect_true(prepare_pivoted_householder_qr(NoScaling{}, pivoted_work,
                                             pivoted_qr, 32.0),
              "pivoted Householder QR preparation should succeed");
  expect_equal(pivoted_qr.permutation[0], Index{1},
               "larger second column should be selected first");
  expect_true(solve_pivoted_householder_qr(NoScaling{}, pivoted_work,
                                           pivoted_qr, lambda),
              "pivoted Householder QR solve should succeed");
  expect_close(pivoted_work.step[0], damped_work.step[0], 1e-12, 1e-12,
               "pivoted Householder QR should unpermute first step component");
  expect_close(pivoted_work.step[1], damped_work.step[1], 1e-12, 1e-12,
               "pivoted Householder QR should unpermute second step component");
}

void test_pivoted_householder_qr_scaled_unpermutation() {
  LMWorkspace<3, 2> damped_work;
  LMWorkspace<3, 2> pivoted_work;
  DampedQrWorkspace<3, 2> damped_qr;
  PivotedHouseholderQrWorkspace<3, 2> pivoted_qr;

  for (auto *work : {&damped_work, &pivoted_work}) {
    work->J[0, 0] = 1.0;
    work->J[1, 0] = 0.0;
    work->J[2, 0] = 1.0;
    work->J[0, 1] = 4.0;
    work->J[1, 1] = 1.0;
    work->J[2, 1] = 5.0;
    work->r[0] = 1.0;
    work->r[1] = 2.0;
    work->r[2] = 3.0;
    work->scale[0] = 2.0;
    work->scale[1] = 4.0;
    work->damping_scale[0] = 2.0;
    work->damping_scale[1] = 4.0;
  }

  constexpr double lambda = 1.0;
  expect_true(
      solve_damped_qr(JacobianColumnScaling{}, damped_work, damped_qr, lambda),
      "scaled pivot reference damped QR solve should succeed");
  expect_true(prepare_pivoted_householder_qr(JacobianColumnScaling{},
                                             pivoted_work, pivoted_qr, 32.0),
              "scaled pivoted Householder QR preparation should succeed");
  expect_equal(pivoted_qr.permutation[0], Index{1},
               "scaled larger second column should be selected first");
  expect_true(solve_pivoted_householder_qr(JacobianColumnScaling{},
                                           pivoted_work, pivoted_qr, lambda),
              "scaled pivoted Householder QR solve should succeed");
  expect_close(pivoted_work.step[0], damped_work.step[0], 1e-12, 1e-12,
               "scaled pivoted QR should unpermute first step component");
  expect_close(pivoted_work.step[1], damped_work.step[1], 1e-12, 1e-12,
               "scaled pivoted QR should unpermute second step component");
}

void test_pivoted_householder_qr_scaled_factor_invariants() {
  LMWorkspace<3, 2> damped_work;
  LMWorkspace<3, 2> pivoted_work;
  DampedQrWorkspace<3, 2> damped_qr;
  PivotedHouseholderQrWorkspace<3, 2> pivoted_qr;

  for (auto *work : {&damped_work, &pivoted_work}) {
    work->J[0, 0] = 1.0;
    work->J[1, 0] = 0.0;
    work->J[2, 0] = 1.0;
    work->J[0, 1] = 4.0;
    work->J[1, 1] = 1.0;
    work->J[2, 1] = 5.0;
    work->r[0] = 1.0;
    work->r[1] = -2.0;
    work->r[2] = 3.0;
    work->scale[0] = 2.0;
    work->scale[1] = 7.0;
    work->damping_scale[0] = 11.0;
    work->damping_scale[1] = 3.0;
  }

  expect_true(prepare_pivoted_householder_qr(JacobianColumnScaling{},
                                             pivoted_work, pivoted_qr, 32.0),
              "scaled pivoted factor invariant preparation should succeed");
  expect_equal(pivoted_qr.permutation[0], Index{1},
               "scaled pivoted factor invariant should force a pivot");

  for (Index k = 0; k < pivoted_work.n; ++k) {
    const Index original_column = pivoted_qr.permutation[k];
    const double expected = pivoted_work.damping_scale[original_column] *
                            pivoted_work.damping_scale[original_column];
    expect_close(pivoted_qr.damping_diagonal[k], expected, 1e-12, 1e-12,
                 "scaled pivoted damping diagonal should follow permutation");
  }

  MatrixStorage<3, 3> q_transpose;
  q_transpose.fill(0.0);
  for (Index column = 0; column < pivoted_work.m; ++column) {
    q_transpose[column, column] = 1.0;
    for (Index k = 0; k < pivoted_work.n; ++k) {
      const auto reflector_tail = ConstVectorView<std::dynamic_extent>(
          pivoted_qr.packed_qr.data() + (k + 1) + k * pivoted_work.m,
          pivoted_work.m - k - 1);
      auto target = VectorView<std::dynamic_extent>(
          q_transpose.data() + k + column * pivoted_work.m, pivoted_work.m - k);
      expect_true(apply_householder_reflector(reflector_tail, pivoted_qr.tau[k],
                                              target),
                  "stored Householder reflector should apply to identity");
    }
  }

  for (Index i = 0; i < pivoted_work.m; ++i) {
    for (Index j = 0; j < pivoted_work.n; ++j) {
      double value = 0.0;
      for (Index k = 0; k < pivoted_work.m; ++k) {
        value += q_transpose[i, k] *
                 pivoted_work.J[k, pivoted_qr.permutation[j]] /
                 pivoted_work.scale[pivoted_qr.permutation[j]];
      }
      const double expected = i <= j ? pivoted_qr.r_base[i, j] : 0.0;
      expect_close(value, expected, 1e-12, 1e-12,
                   "stored Householder factors should reconstruct scaled R");
    }

    double transformed_rhs = 0.0;
    for (Index k = 0; k < pivoted_work.m; ++k) {
      transformed_rhs -= q_transpose[i, k] * pivoted_work.r[k];
    }
    expect_close(transformed_rhs, pivoted_qr.transformed_rhs[i], 1e-12, 1e-12,
                 "stored Householder factors should reconstruct cached RHS");
  }

  const auto r_base = pivoted_qr.r_base;
  const auto transformed_rhs = pivoted_qr.transformed_rhs;
  const auto permutation = pivoted_qr.permutation;
  const auto damping_diagonal = pivoted_qr.damping_diagonal;
  for (double lambda : {1e-6, 1e-2, 1.0, 1e4}) {
    expect_true(
        solve_damped_qr(JacobianColumnScaling{}, damped_work, damped_qr,
                        lambda),
        "scaled pivoted factor invariant reference solve should succeed");
    expect_true(solve_pivoted_householder_qr(JacobianColumnScaling{},
                                             pivoted_work, pivoted_qr, lambda),
                "scaled pivoted factor invariant cached solve should succeed");
    for (Index j = 0; j < pivoted_work.n; ++j) {
      expect_close(pivoted_work.step[j], damped_work.step[j], 1e-11, 1e-11,
                   "scaled pivoted lambda sweep should match streamed QR step");
      expect_close(pivoted_qr.permutation[j], permutation[j], 0.0, 0.0,
                   "cached solve should not mutate permutation");
      expect_close(pivoted_qr.damping_diagonal[j], damping_diagonal[j], 0.0,
                   0.0, "cached solve should not mutate damping diagonal");
      for (Index i = 0; i <= j; ++i) {
        expect_close(pivoted_qr.r_base[i, j], r_base[i, j], 0.0, 0.0,
                     "cached solve should not mutate base R");
      }
    }
    for (Index i = 0; i < pivoted_work.m; ++i) {
      expect_close(pivoted_qr.transformed_rhs[i], transformed_rhs[i], 0.0, 0.0,
                   "cached solve should not mutate transformed RHS");
    }
  }
}

void test_pivoted_householder_qr_frozen_ill_conditioned_system() {
  LMWorkspace<5, 3> damped_work;
  LMWorkspace<5, 3> pivoted_work;
  DampedQrWorkspace<5, 3> damped_qr;
  PivotedHouseholderQrWorkspace<5, 3> pivoted_qr;

  constexpr std::array<std::array<double, 3>, 5> jacobian{{
      {1.0, 1.0e8, 1.0e-4},
      {2.0, 2.0e8 + 1.0, -2.0e-4},
      {-1.0, -1.0e8 + 1.0, 3.0e-4},
      {3.0, 3.0e8 - 1.0, -1.0e-4},
      {0.5, 5.0e7 + 2.0, 4.0e-4},
  }};
  constexpr std::array<double, 5> residual{1.0, -2.0, 0.5, 3.0, -1.0};

  for (auto *work : {&damped_work, &pivoted_work}) {
    for (Index i = 0; i < work->m; ++i) {
      work->r[i] = residual[i];
      for (Index j = 0; j < work->n; ++j) {
        work->J[i, j] = jacobian[i][j];
      }
    }
    work->scale[0] = 2.0;
    work->scale[1] = 1.0e8;
    work->scale[2] = 1.0e-4;
    work->damping_scale[0] = 5.0;
    work->damping_scale[1] = 7.0;
    work->damping_scale[2] = 11.0;
  }

  expect_true(prepare_pivoted_householder_qr(JacobianColumnScaling{},
                                             pivoted_work, pivoted_qr, 32.0),
              "frozen ill-conditioned pivoted preparation should succeed");
  expect_true(pivoted_qr.permutation[0] != Index{0},
              "frozen ill-conditioned system should exercise pivoting");

  for (double lambda : {1e-6, 1e-2, 1.0, 1e4}) {
    expect_true(solve_damped_qr(JacobianColumnScaling{}, damped_work, damped_qr,
                                lambda),
                "frozen ill-conditioned streamed solve should succeed");
    expect_true(solve_pivoted_householder_qr(JacobianColumnScaling{},
                                             pivoted_work, pivoted_qr, lambda),
                "frozen ill-conditioned cached solve should succeed");
    for (Index j = 0; j < pivoted_work.n; ++j) {
      expect_close(
          pivoted_work.step[j], damped_work.step[j], 1e-7, 1e-10,
          "frozen ill-conditioned cached step should match streamed QR");
    }
  }
}

void test_pivoted_householder_qr_rank_deficient() {
  LMWorkspace<3, 2> damped_work;
  LMWorkspace<3, 2> pivoted_work;
  DampedQrWorkspace<3, 2> damped_qr;
  PivotedHouseholderQrWorkspace<3, 2> pivoted_qr;

  for (auto *work : {&damped_work, &pivoted_work}) {
    work->J[0, 0] = 1.0;
    work->J[1, 0] = 2.0;
    work->J[2, 0] = 3.0;
    work->J[0, 1] = 1.0;
    work->J[1, 1] = 2.0;
    work->J[2, 1] = 3.0;
    work->r[0] = 1.0;
    work->r[1] = 2.0;
    work->r[2] = 3.0;
  }

  constexpr double lambda = 1.0;
  expect_true(solve_damped_qr(NoScaling{}, damped_work, damped_qr, lambda),
              "rank-deficient reference damped QR solve should succeed");
  expect_true(
      prepare_pivoted_householder_qr(NoScaling{}, pivoted_work, pivoted_qr,
                                     32.0),
      "rank-deficient pivoted Householder QR preparation should succeed");
  expect_equal(pivoted_qr.numerical_rank, Index{1},
               "rank-deficient pivoted Householder QR rank");
  expect_true(solve_pivoted_householder_qr(NoScaling{}, pivoted_work,
                                           pivoted_qr, lambda),
              "rank-deficient damped pivoted QR solve should succeed");
  expect_close(pivoted_work.step[0], damped_work.step[0], 1e-12, 1e-12,
               "rank-deficient pivoted QR first step component");
  expect_close(pivoted_work.step[1], damped_work.step[1], 1e-12, 1e-12,
               "rank-deficient pivoted QR second step component");
}

void test_pivoted_householder_qr_dynamic_and_mixed() {
  const auto run = []<Index M, Index N>(LMWorkspace<M, N> &work,
                                        PivotedHouseholderQrWorkspace<M, N> &qr,
                                        const std::string &what) {
    work.J[0, 0] = 1.0;
    work.J[1, 0] = 0.0;
    work.J[2, 0] = 1.0;
    work.J[0, 1] = 0.0;
    work.J[1, 1] = 1.0;
    work.J[2, 1] = 1.0;
    work.r[0] = 1.0;
    work.r[1] = 2.0;
    work.r[2] = 3.0;

    expect_true(prepare_pivoted_householder_qr(NoScaling{}, work, qr, 32.0),
                what + " preparation should succeed");
    expect_true(solve_pivoted_householder_qr(NoScaling{}, work, qr, 1.0),
                what + " solve should succeed");
    expect_close(work.step[0], -0.875, 1e-12, 1e-12,
                 what + " first step component");
    expect_close(work.step[1], -1.375, 1e-12, 1e-12,
                 what + " second step component");
  };

  LMWorkspace<std::dynamic_extent, std::dynamic_extent> dynamic_work(3, 2);
  PivotedHouseholderQrWorkspace<std::dynamic_extent, std::dynamic_extent>
      dynamic_qr;
  dynamic_qr.resize(3, 2);
  run(dynamic_work, dynamic_qr, "dynamic pivoted Householder QR");

  LMWorkspace<std::dynamic_extent, 2> mixed_work(3, 2);
  PivotedHouseholderQrWorkspace<std::dynamic_extent, 2> mixed_qr;
  mixed_qr.resize(3, 2);
  run(mixed_work, mixed_qr, "mixed pivoted Householder QR");
}

void test_solve_pivoted_householder_qr_policy() {
  auto residual = [](ConstVectorView<2> x, VectorView<3> r) {
    r[0] = x[0] - 1.0;
    r[1] = x[1] - 2.0;
    r[2] = x[0] + x[1] - 3.0;
    return ErrorOrVoid{};
  };
  auto jacobian = [](ConstVectorView<2>, MatrixView<3, 2> J) {
    J[0, 0] = 1.0;
    J[1, 0] = 0.0;
    J[2, 0] = 1.0;
    J[0, 1] = 0.0;
    J[1, 1] = 1.0;
    J[2, 1] = 1.0;
    return ErrorOrVoid{};
  };
  const auto problem = make_problem<3, 2>(residual, jacobian);
  Options options;
  options.max_iterations = 50;
  SolverWorkspace<PivotedQrSolverPolicy, 3, 2> workspace;
  const std::array<double, 2> x0{0.0, 0.0};
  SolverContext<PivotedQrSolverPolicy, 3, 2, decltype(residual),
                decltype(jacobian)>
      context(problem, options, workspace, x0);

  const auto solved = solve<PivotedQrSolverPolicy>(context);
  expect_true(solved.has_value(), "pivoted Householder QR policy should solve");
  expect_close(solved->parameters[0], 1.0, 1e-8, 1e-8,
               "pivoted Householder QR policy first parameter");
  expect_close(solved->parameters[1], 2.0, 1e-8, 1e-8,
               "pivoted Householder QR policy second parameter");
  expect_true(solved->factorization_count > 0,
              "pivoted Householder QR policy should factorize");
  expect_true(solved->linear_solves > solved->factorization_count,
              "pivoted Householder QR policy should reuse a factorization");
}

void test_pivoted_householder_qr_factor_lifecycle() {
  auto residual = [](ConstVectorView<1> x, VectorView<1> r) {
    r[0] = x[0];
    return ErrorOrVoid{};
  };
  auto jacobian = [](ConstVectorView<1>, MatrixView<1, 1> J) {
    J[0, 0] = 1.0;
    return ErrorOrVoid{};
  };
  const auto problem = make_problem<1, 1>(residual, jacobian);
  Options options;
  options.max_iterations = 2;
  options.lm.min_lambda = 0.1;
  options.lm.initial_trust_region_factor = 0.5;
  SolverWorkspace<PivotedQrSolverPolicy, 1, 1> workspace;
  const std::array<double, 1> x0{1.0};
  SolverContext<PivotedQrSolverPolicy, 1, 1, decltype(residual),
                decltype(jacobian)>
      context(problem, options, workspace, x0);
  std::vector<LmTrialTrace> trace;
  context.trial_trace = &trace;

  const auto solved = solve<PivotedQrSolverPolicy>(context);
  expect_true(solved.has_value(),
              "pivoted factor lifecycle solve should succeed");
  expect_equal(trace.size(), std::size_t{2},
               "pivoted factor lifecycle should accept two model steps");
  expect_true(
      trace.front().inner_linear_solves > 1,
      "pivoted factor lifecycle should select lambda without refactoring");
  expect_equal(solved->factorization_count, trace.size(),
               "pivoted factor lifecycle should rebuild only after acceptance");
  expect_true(
      solved->linear_solves > solved->factorization_count,
      "pivoted factor lifecycle should reuse each accepted factorization");
}

void expect_bisection_band(const std::vector<LmTrialTrace> &trace,
                           const std::string &what) {
  expect_equal(trace.size(), std::size_t{1}, what + " should record one trial");
  const auto &trial = trace.front();
  expect_true(trial.inner_linear_solves > 5,
              what + " should refine the lambda bracket");
  expect_true(trial.radius_bound_active,
              what + " should require lambda escalation");
  expect_equal(trial.lambda_path, LambdaPath::LegacyBisection,
               what + " should trace the legacy bisection path");
  expect_equal(trial.bisection_calls, Index{1},
               what + " should enter bisection once");
  expect_equal(trial.more_calls, Index{0},
               what + " should not enter More selection");
  expect_equal(trial.gn_calls, Index{0},
               what + " should not compute a Householder GN step");
  expect_true(trial.scaled_step_norm <= 0.5,
              what + " should stay inside the trust radius");
  expect_true(trial.scaled_step_norm >= 0.45,
              what + " should land in the bisection boundary band");
}

void test_bisection_lambda_selection_static() {
  auto residual = [](ConstVectorView<1> x, VectorView<1> r) {
    r[0] = x[0];
    return ErrorOrVoid{};
  };
  auto jacobian = [](ConstVectorView<1>, MatrixView<1, 1> J) {
    J[0, 0] = 1.0;
    return ErrorOrVoid{};
  };
  const auto problem = make_problem<1, 1>(residual, jacobian);
  Options options;
  options.max_iterations = 1;
  options.lm.min_lambda = 0.1;
  options.lm.initial_trust_region_factor = 0.5;
  SolverWorkspace<DefaultSolverPolicy, 1, 1> workspace;
  const std::array<double, 1> x0{1.0};
  SolverContext<DefaultSolverPolicy, 1, 1, decltype(residual),
                decltype(jacobian)>
      context(problem, options, workspace, x0);
  std::vector<LmTrialTrace> trace;
  context.trial_trace = &trace;

  const auto solved = solve<DefaultSolverPolicy>(context);
  expect_true(solved.has_value(), "static bisection solve should succeed");
  expect_bisection_band(trace, "static bisection solve");
}

void test_bisection_lambda_selection_dynamic() {
  auto residual = [](ConstVectorView<std::dynamic_extent> x,
                     VectorView<std::dynamic_extent> r) {
    r[0] = x[0];
    return ErrorOrVoid{};
  };
  auto jacobian = [](ConstVectorView<std::dynamic_extent>,
                     MatrixView<std::dynamic_extent, std::dynamic_extent> J) {
    J[0, 0] = 1.0;
    return ErrorOrVoid{};
  };
  const auto problem = make_dynamic_problem(1, 1, residual, jacobian);
  Options options;
  options.max_iterations = 1;
  options.lm.min_lambda = 0.1;
  options.lm.initial_trust_region_factor = 0.5;
  SolverWorkspace<DefaultSolverPolicy, std::dynamic_extent, std::dynamic_extent>
      workspace;
  const std::vector<double> x0{1.0};
  SolverContext<DefaultSolverPolicy, std::dynamic_extent, std::dynamic_extent,
                decltype(residual), decltype(jacobian)>
      context(problem, options, workspace, x0);
  std::vector<LmTrialTrace> trace;
  context.trial_trace = &trace;

  const auto solved = solve<DefaultSolverPolicy>(context);
  expect_true(solved.has_value(), "dynamic bisection solve should succeed");
  expect_bisection_band(trace, "dynamic bisection solve");
}

void expect_pivoted_lmpar_band(const Result &result,
                               const std::vector<LmTrialTrace> &trace,
                               const std::string &what,
                               double trust_radius = 0.5,
                               double max_step_ratio = 1.0) {
  expect_equal(trace.size(), std::size_t{1}, what + " should record one trial");
  const auto &trial = trace.front();
  expect_true(trial.radius_bound_active,
              what + " should require lambda selection");
  expect_true(trial.scaled_step_norm <= max_step_ratio * trust_radius,
              what + " should stay within the lambda-selection band");
  expect_true(trial.scaled_step_norm >= 0.9 * trust_radius,
              what + " should approach the trust-radius boundary");
  expect_true(trial.selected_lambda > 0.1,
              what + " should select damping above the lower bound");
  expect_close(result.lambda, trial.selected_lambda, 0.0, 0.0,
               what + " should report the selected lambda");
  expect_equal(result.factorization_count, Index{1},
               what + " should reuse one cached factorization");
  expect_true(result.linear_solves > result.factorization_count,
              what + " should reuse the cached factorization across lambdas");
  expect_true(trial.inner_linear_solves < 6,
              what + " should improve on bisection lambda selection");
}

void expect_pivoted_lmpar_correction(const Result &result,
                                     const std::vector<LmTrialTrace> &trace,
                                     const std::string &what) {
  expect_pivoted_lmpar_band(result, trace, what, 0.25 * std::sqrt(2.0), 1.1);
  const auto &trial = trace.front();
  expect_true(trial.lmpar_iterations > 1,
              what + " should perform a More correction iteration");
  expect_equal(trial.gn_calls, Index{1},
               what + " should compute one Householder GN step");
  expect_equal(trial.more_calls, Index{1},
               what + " should enter More selection once");
  expect_equal(trial.bisection_calls, Index{0},
               what + " should not enter legacy bisection");
  expect_true(!trial.lmpar_fallback,
              what + " should not fall back to bisection");
  expect_equal(trial.bisection_bracket_expansions, Index{0},
               what + " should not expand a bisection bracket");
  expect_equal(trial.bisection_refinements, Index{0},
               what + " should not refine a bisection bracket");
}

void expect_pivoted_interior_gauss_newton_step(
    const Result &result, const std::vector<LmTrialTrace> &trace,
    const std::string &what) {
  expect_equal(trace.size(), std::size_t{1}, what + " should record one trial");
  const auto &trial = trace.front();
  expect_true(!trial.radius_bound_active,
              what + " should keep the trust-region constraint inactive");
  expect_close(result.lambda, 0.0, 0.0, 0.0,
               what + " should use the undamped Gauss-Newton step");
  expect_close(trial.selected_lambda, 0.0, 0.0, 0.0,
               what + " should trace the undamped selected lambda");
  expect_close(trial.last_evaluated_lambda, 0.0, 0.0, 0.0,
               what + " should trace the Gauss-Newton solve");
  expect_equal(trial.inner_linear_solves, Index{1},
               what + " should perform only the Gauss-Newton solve");
  expect_equal(trial.lambda_path, LambdaPath::HouseholderGn,
               what + " should trace the Householder GN path");
  expect_equal(trial.gn_calls, Index{1},
               what + " should compute one Householder GN step");
  expect_equal(trial.more_calls, Index{0},
               what + " should skip More selection");
  expect_equal(trial.bisection_calls, Index{0},
               what + " should skip bisection selection");
  expect_equal(trial.lmpar_iterations, Index{0},
               what + " should skip Moré iteration");
  expect_equal(trial.bisection_refinements, Index{0},
               what + " should skip bisection refinement");
  expect_close(trial.step[0], -1.0, 1e-12, 1e-12,
               what + " first Gauss-Newton component");
  expect_close(trial.step[1], 0.0, 1e-12, 1e-12,
               what + " rank-deficient Gauss-Newton component");
}

void test_pivoted_lmpar_lambda_selection_static() {
  auto residual = [](ConstVectorView<1> x, VectorView<1> r) {
    r[0] = x[0];
    return ErrorOrVoid{};
  };
  auto jacobian = [](ConstVectorView<1>, MatrixView<1, 1> J) {
    J[0, 0] = 1.0;
    return ErrorOrVoid{};
  };
  const auto problem = make_problem<1, 1>(residual, jacobian);
  Options options;
  options.max_iterations = 1;
  options.lm.min_lambda = 0.1;
  options.lm.initial_trust_region_factor = 0.5;
  SolverWorkspace<PivotedQrSolverPolicy, 1, 1> workspace;
  const std::array<double, 1> x0{1.0};
  SolverContext<PivotedQrSolverPolicy, 1, 1, decltype(residual),
                decltype(jacobian)>
      context(problem, options, workspace, x0);
  std::vector<LmTrialTrace> trace;
  context.trial_trace = &trace;

  const auto solved = solve<PivotedQrSolverPolicy>(context);
  expect_true(solved.has_value(), "static pivoted lmpar solve should succeed");
  expect_pivoted_lmpar_band(*solved, trace, "static pivoted lmpar solve");
}

void test_pivoted_lmpar_lambda_selection_dynamic() {
  auto residual = [](ConstVectorView<std::dynamic_extent> x,
                     VectorView<std::dynamic_extent> r) {
    r[0] = x[0];
    return ErrorOrVoid{};
  };
  auto jacobian = [](ConstVectorView<std::dynamic_extent>,
                     MatrixView<std::dynamic_extent, std::dynamic_extent> J) {
    J[0, 0] = 1.0;
    return ErrorOrVoid{};
  };
  const auto problem = make_dynamic_problem(1, 1, residual, jacobian);
  Options options;
  options.max_iterations = 1;
  options.lm.min_lambda = 0.1;
  options.lm.initial_trust_region_factor = 0.5;
  SolverWorkspace<PivotedQrSolverPolicy, std::dynamic_extent,
                  std::dynamic_extent>
      workspace;
  const std::vector<double> x0{1.0};
  SolverContext<PivotedQrSolverPolicy, std::dynamic_extent, std::dynamic_extent,
                decltype(residual), decltype(jacobian)>
      context(problem, options, workspace, x0);
  std::vector<LmTrialTrace> trace;
  context.trial_trace = &trace;

  const auto solved = solve<PivotedQrSolverPolicy>(context);
  expect_true(solved.has_value(), "dynamic pivoted lmpar solve should succeed");
  expect_pivoted_lmpar_band(*solved, trace, "dynamic pivoted lmpar solve");
}

void test_pivoted_interior_gauss_newton_step_static() {
  auto residual = [](ConstVectorView<2> x, VectorView<2> r) {
    r[0] = x[0];
    r[1] = 2.0 * x[0];
    return ErrorOrVoid{};
  };
  auto jacobian = [](ConstVectorView<2>, MatrixView<2, 2> J) {
    J[0, 0] = 1.0;
    J[0, 1] = 0.0;
    J[1, 0] = 2.0;
    J[1, 1] = 0.0;
    return ErrorOrVoid{};
  };
  const auto problem = make_problem<2, 2>(residual, jacobian);
  Options options;
  options.max_iterations = 1;
  options.lm.min_lambda = 0.1;
  options.lm.initial_trust_region_factor = 1.0;
  SolverWorkspace<PivotedQrSolverPolicy, 2, 2> workspace;
  const std::array<double, 2> x0{1.0, 2.0};
  SolverContext<PivotedQrSolverPolicy, 2, 2, decltype(residual),
                decltype(jacobian)>
      context(problem, options, workspace, x0);
  std::vector<LmTrialTrace> trace;
  context.trial_trace = &trace;

  const auto solved = solve<PivotedQrSolverPolicy>(context);
  expect_true(solved.has_value(), "static interior Gauss-Newton solve");
  expect_pivoted_interior_gauss_newton_step(
      *solved, trace, "static interior Gauss-Newton solve");
}

void test_pivoted_interior_gauss_newton_step_dynamic() {
  auto residual = [](ConstVectorView<std::dynamic_extent> x,
                     VectorView<std::dynamic_extent> r) {
    r[0] = x[0];
    r[1] = 2.0 * x[0];
    return ErrorOrVoid{};
  };
  auto jacobian = [](ConstVectorView<std::dynamic_extent>,
                     MatrixView<std::dynamic_extent, std::dynamic_extent> J) {
    J[0, 0] = 1.0;
    J[0, 1] = 0.0;
    J[1, 0] = 2.0;
    J[1, 1] = 0.0;
    return ErrorOrVoid{};
  };
  const auto problem = make_dynamic_problem(2, 2, residual, jacobian);
  Options options;
  options.max_iterations = 1;
  options.lm.min_lambda = 0.1;
  options.lm.initial_trust_region_factor = 1.0;
  SolverWorkspace<PivotedQrSolverPolicy, std::dynamic_extent,
                  std::dynamic_extent>
      workspace;
  const std::vector<double> x0{1.0, 2.0};
  SolverContext<PivotedQrSolverPolicy, std::dynamic_extent, std::dynamic_extent,
                decltype(residual), decltype(jacobian)>
      context(problem, options, workspace, x0);
  std::vector<LmTrialTrace> trace;
  context.trial_trace = &trace;

  const auto solved = solve<PivotedQrSolverPolicy>(context);
  expect_true(solved.has_value(), "dynamic interior Gauss-Newton solve");
  expect_pivoted_interior_gauss_newton_step(
      *solved, trace, "dynamic interior Gauss-Newton solve");
}

void test_pivoted_lmpar_correction_static() {
  auto residual = [](ConstVectorView<2> x, VectorView<2> r) {
    r[0] = x[0];
    r[1] = x[1];
    return ErrorOrVoid{};
  };
  auto jacobian = [](ConstVectorView<2>, MatrixView<2, 2> J) {
    J[0, 0] = 1.0;
    J[0, 1] = 0.0;
    J[1, 0] = 0.0;
    J[1, 1] = 2.0;
    return ErrorOrVoid{};
  };
  const auto problem = make_problem<2, 2>(residual, jacobian);
  Options options;
  options.max_iterations = 1;
  options.lm.min_lambda = 0.1;
  options.lm.initial_trust_region_factor = 0.25;
  SolverWorkspace<PivotedQrSolverPolicy, 2, 2> workspace;
  const std::array<double, 2> x0{1.0, 1.0};
  SolverContext<PivotedQrSolverPolicy, 2, 2, decltype(residual),
                decltype(jacobian)>
      context(problem, options, workspace, x0);
  std::vector<LmTrialTrace> trace;
  context.trial_trace = &trace;

  const auto solved = solve<PivotedQrSolverPolicy>(context);
  expect_true(solved.has_value(),
              "static pivoted lmpar correction should succeed");
  expect_pivoted_lmpar_correction(*solved, trace,
                                  "static pivoted lmpar correction");
}

void test_pivoted_lmpar_correction_dynamic() {
  auto residual = [](ConstVectorView<std::dynamic_extent> x,
                     VectorView<std::dynamic_extent> r) {
    r[0] = x[0];
    r[1] = x[1];
    return ErrorOrVoid{};
  };
  auto jacobian = [](ConstVectorView<std::dynamic_extent>,
                     MatrixView<std::dynamic_extent, std::dynamic_extent> J) {
    J[0, 0] = 1.0;
    J[0, 1] = 0.0;
    J[1, 0] = 0.0;
    J[1, 1] = 2.0;
    return ErrorOrVoid{};
  };
  const auto problem = make_dynamic_problem(2, 2, residual, jacobian);
  Options options;
  options.max_iterations = 1;
  options.lm.min_lambda = 0.1;
  options.lm.initial_trust_region_factor = 0.25;
  SolverWorkspace<PivotedQrSolverPolicy, std::dynamic_extent,
                  std::dynamic_extent>
      workspace;
  const std::vector<double> x0{1.0, 1.0};
  SolverContext<PivotedQrSolverPolicy, std::dynamic_extent, std::dynamic_extent,
                decltype(residual), decltype(jacobian)>
      context(problem, options, workspace, x0);
  std::vector<LmTrialTrace> trace;
  context.trial_trace = &trace;

  const auto solved = solve<PivotedQrSolverPolicy>(context);
  expect_true(solved.has_value(),
              "dynamic pivoted lmpar correction should succeed");
  expect_pivoted_lmpar_correction(*solved, trace,
                                  "dynamic pivoted lmpar correction");
}

void test_pivoted_lmpar_rejected_trial_preserves_warm_start() {
  auto residual = [](ConstVectorView<1> x, VectorView<1> r) {
    r[0] = x[0] - 3.0;
    return ErrorOrVoid{};
  };
  auto jacobian = [](ConstVectorView<1> x, MatrixView<1, 1> J) {
    J[0, 0] = x[0] < 2.0 ? 1.0 : -1.0;
    return ErrorOrVoid{};
  };
  const auto problem = make_problem<1, 1>(residual, jacobian);
  Options options;
  options.max_iterations = 2;
  options.lm.min_lambda = 0.1;
  SolverWorkspace<PivotedQrSolverPolicy, 1, 1> workspace;
  const std::array<double, 1> x0{1.0};
  SolverContext<PivotedQrSolverPolicy, 1, 1, decltype(residual),
                decltype(jacobian)>
      context(problem, options, workspace, x0);
  std::vector<LmTrialTrace> trace;
  context.trial_trace = &trace;

  const auto solved = solve<PivotedQrSolverPolicy>(context);
  expect_true(solved.has_value(), "warm-start rejection solve should succeed");
  expect_equal(trace.size(), std::size_t{2},
               "warm-start rejection should record two trials");
  expect_equal(trace[0].decision, TrialDecision::Accepted,
               "warm-start rejection first trial should be accepted");
  expect_equal(trace[1].decision, TrialDecision::LowRho,
               "warm-start rejection second trial should be rejected");
  expect_true(trace[0].selected_lambda != trace[1].selected_lambda,
              "warm-start rejection should select a different rejected lambda");
  expect_close(context.selected_lambda, trace[0].selected_lambda, 0.0, 0.0,
               "rejected trial should preserve the accepted warm-start lambda");
}

void test_model_cost_cancellation() {
  auto residual = [](ConstVectorView<1> x, VectorView<1> r) {
    r[0] = 1e16 + x[0];
    return ErrorOrVoid{};
  };
  auto jacobian = [](ConstVectorView<1>, MatrixView<1, 1> J) {
    J[0, 0] = 1.0;
    return ErrorOrVoid{};
  };
  const auto problem = make_problem<1, 1>(residual, jacobian);
  Options options;
  options.max_iterations = 1;
  options.lm.min_lambda = 1e16;
  options.lm.max_lambda = 1e16;
  options.lm.initial_trust_region_radius = 2.0;
  SolverWorkspace<DefaultSolverPolicy, 1, 1> workspace;
  const std::array<double, 1> x0{0.0};
  SolverContext<DefaultSolverPolicy, 1, 1, decltype(residual),
                decltype(jacobian)>
      context(problem, options, workspace, x0);
  std::vector<LmTrialTrace> trace;
  context.trial_trace = &trace;

  const auto solved = solve<DefaultSolverPolicy>(context);
  expect_true(solved.has_value(),
              "model-cost cancellation solve should succeed");
  expect_equal(trace.size(), std::size_t{1},
               "model-cost cancellation solve should record one trial");
  expect_equal(trace.front().actual_reduction, 0.0,
               "trial cost should round to the current cost");
  expect_equal(trace.front().predicted_reduction, 0.0,
               "model cost should round to the current cost");
  expect_equal(trace.front().decision,
               TrialDecision::NonPositivePredictedReduction,
               "zero model reduction should reject the invalid model step");
  expect_equal(solved->termination, TerminationReason::MaxIterations,
               "rejected invalid model step should not report convergence");
}

void test_solve_static_user_jacobian() {
  auto residual = [](ConstVectorView<2> x, VectorView<3> r) {
    r[0] = x[0] - 1.0;
    r[1] = x[1] - 2.0;
    r[2] = x[0] + x[1] - 3.0;
    return ErrorOrVoid{};
  };
  auto jacobian = [](ConstVectorView<2>, MatrixView<3, 2> J) {
    J[0, 0] = 1.0;
    J[1, 0] = 0.0;
    J[2, 0] = 1.0;
    J[0, 1] = 0.0;
    J[1, 1] = 1.0;
    J[2, 1] = 1.0;
    return ErrorOrVoid{};
  };
  const auto problem = make_problem<3, 2>(residual, jacobian);
  Options options;
  options.max_iterations = 50;
  options.max_function_evaluations = 100;
  SolverWorkspace<DefaultSolverPolicy, 3, 2> workspace;
  const std::array<double, 2> x0{0.0, 0.0};
  SolverContext<DefaultSolverPolicy, 3, 2, decltype(residual),
                decltype(jacobian)>
      context(problem, options, workspace, x0);

  const auto solved = solve<DefaultSolverPolicy>(context);
  expect_true(solved.has_value(), "static solve should succeed");
  const Result &result = *solved;
  expect_close(result.parameters[0], 1.0, 1e-8, 1e-8,
               "static solve first parameter");
  expect_close(result.parameters[1], 2.0, 1e-8, 1e-8,
               "static solve second parameter");
  expect_true(result.final_cost < 1e-14,
              "static solve should reach near-zero cost");
  expect_true(result.iterations > 0,
              "static solve should perform an LM attempt");
  expect_true(result.linear_solves > 0,
              "static solve should perform a linear solve");
  expect_true(result.termination != TerminationReason::NotTerminated,
              "static solve should return a terminal result");
}

void test_solve_dynamic_user_jacobian() {
  auto residual = [](ConstVectorView<std::dynamic_extent> x,
                     VectorView<std::dynamic_extent> r) {
    r[0] = x[0] - 1.0;
    r[1] = x[1] - 2.0;
    r[2] = x[0] + x[1] - 3.0;
    return ErrorOrVoid{};
  };
  auto jacobian = [](ConstVectorView<std::dynamic_extent>,
                     MatrixView<std::dynamic_extent, std::dynamic_extent> J) {
    J[0, 0] = 1.0;
    J[1, 0] = 0.0;
    J[2, 0] = 1.0;
    J[0, 1] = 0.0;
    J[1, 1] = 1.0;
    J[2, 1] = 1.0;
    return ErrorOrVoid{};
  };
  const auto problem = make_dynamic_problem(3, 2, residual, jacobian);
  Options options;
  options.max_iterations = 50;
  options.max_function_evaluations = 100;
  SolverWorkspace<DefaultSolverPolicy, std::dynamic_extent, std::dynamic_extent>
      workspace;
  const std::vector<double> x0{0.0, 0.0};
  SolverContext<DefaultSolverPolicy, std::dynamic_extent, std::dynamic_extent,
                decltype(residual), decltype(jacobian)>
      context(problem, options, workspace, x0);

  const auto solved = solve<DefaultSolverPolicy>(context);
  expect_true(solved.has_value(), "dynamic solve should succeed");
  const Result &result = *solved;
  expect_close(result.parameters[0], 1.0, 1e-8, 1e-8,
               "dynamic solve first parameter");
  expect_close(result.parameters[1], 2.0, 1e-8, 1e-8,
               "dynamic solve second parameter");
  expect_true(result.final_cost < 1e-14,
              "dynamic solve should reach near-zero cost");
  expect_true(result.termination != TerminationReason::NotTerminated,
              "dynamic solve should return a terminal result");
}

void test_solve_forward_difference_evaluation_budget() {
  Index residual_calls = 0;
  auto residual = [&residual_calls](ConstVectorView<2> x, VectorView<3> r) {
    ++residual_calls;
    r[0] = x[0] - 1.0;
    r[1] = x[1] - 2.0;
    r[2] = x[0] + x[1] - 3.0;
    return ErrorOrVoid{};
  };
  const auto problem = make_problem<3, 2>(residual);
  Options options;
  options.max_function_evaluations = 2;
  SolverWorkspace<ForwardDifferenceSolverPolicy, 3, 2> workspace;
  const std::array<double, 2> x0{0.0, 0.0};
  SolverContext<ForwardDifferenceSolverPolicy, 3, 2, decltype(residual),
                NoJacobian>
      context(problem, options, workspace, x0);

  const auto solved = solve<ForwardDifferenceSolverPolicy>(context);
  expect_true(solved.has_value(), "budget termination should be a result");
  expect_equal(solved->termination, TerminationReason::MaxFunctionEvaluations,
               "forward difference solve should stop before budget overrun");
  expect_equal(residual_calls, Index{0},
               "budget preflight should avoid residual callbacks");
}

void test_solve_numerical_termination() {
  auto residual = [](ConstVectorView<1>, VectorView<2> r) {
    r[0] = std::numeric_limits<double>::infinity();
    r[1] = 0.0;
    return ErrorOrVoid{};
  };
  auto jacobian = [](ConstVectorView<1>, MatrixView<2, 1> J) {
    J[0, 0] = 1.0;
    J[1, 0] = 0.0;
    return ErrorOrVoid{};
  };
  const auto problem = make_problem<2, 1>(residual, jacobian);
  Options options;
  SolverWorkspace<DefaultSolverPolicy, 2, 1> workspace;
  const std::array<double, 1> x0{0.0};
  SolverContext<DefaultSolverPolicy, 2, 1, decltype(residual),
                decltype(jacobian)>
      context(problem, options, workspace, x0);

  const auto solved = solve<DefaultSolverPolicy>(context);
  expect_true(solved.has_value(), "numerical failure should be a result");
  expect_equal(solved->termination, TerminationReason::NumericalFailure,
               "non-finite residual should terminate numerically");
}

void test_solve_callback_error() {
  auto residual = [](ConstVectorView<1>, VectorView<2>) -> ErrorOrVoid {
    return std::unexpected(
        Error{ErrorCode::UserFunctionError, "expected residual failure"});
  };
  auto jacobian = [](ConstVectorView<1>, MatrixView<2, 1>) {
    return ErrorOrVoid{};
  };
  const auto problem = make_problem<2, 1>(residual, jacobian);
  Options options;
  SolverWorkspace<DefaultSolverPolicy, 2, 1> workspace;
  const std::array<double, 1> x0{0.0};
  SolverContext<DefaultSolverPolicy, 2, 1, decltype(residual),
                decltype(jacobian)>
      context(problem, options, workspace, x0);

  const auto solved = solve<DefaultSolverPolicy>(context);
  expect_true(!solved, "residual callback error should propagate");
  expect_equal(solved.error().code, ErrorCode::UserFunctionError,
               "residual callback error code should be preserved");
}

void test_solve_zero_iteration_budget() {
  auto residual = [](ConstVectorView<1> x, VectorView<2> r) {
    r[0] = x[0] - 1.0;
    r[1] = x[0] - 1.0;
    return ErrorOrVoid{};
  };
  auto jacobian = [](ConstVectorView<1>, MatrixView<2, 1> J) {
    J[0, 0] = 1.0;
    J[1, 0] = 1.0;
    return ErrorOrVoid{};
  };
  const auto problem = make_problem<2, 1>(residual, jacobian);
  Options options;
  options.max_iterations = 0;
  SolverWorkspace<DefaultSolverPolicy, 2, 1> workspace;
  const std::array<double, 1> x0{0.0};
  SolverContext<DefaultSolverPolicy, 2, 1, decltype(residual),
                decltype(jacobian)>
      context(problem, options, workspace, x0);

  const auto solved = solve<DefaultSolverPolicy>(context);
  expect_true(solved.has_value(), "zero iteration limit should be a result");
  expect_equal(solved->termination, TerminationReason::MaxIterations,
               "zero iteration limit should terminate immediately");
}

void test_solve_finite_difference_policies() {
  auto residual = [](ConstVectorView<2> x, VectorView<3> r) {
    r[0] = x[0] - 1.0;
    r[1] = x[1] - 2.0;
    r[2] = x[0] + x[1] - 3.0;
    return ErrorOrVoid{};
  };
  const auto problem = make_problem<3, 2>(residual);
  const std::array<double, 2> x0{0.0, 0.0};

  const auto run = [&]<class Policy>(const std::string &name) {
    Options options;
    options.max_iterations = 50;
    options.max_function_evaluations = 500;
    SolverWorkspace<Policy, 3, 2> workspace;
    SolverContext<Policy, 3, 2, decltype(residual), NoJacobian> context(
        problem, options, workspace, x0);

    const auto solved = solve<Policy>(context);
    expect_true(solved.has_value(), name + " solve should succeed");
    expect_close(solved->parameters[0], 1.0, 1e-6, 1e-6,
                 name + " first parameter");
    expect_close(solved->parameters[1], 2.0, 1e-6, 1e-6,
                 name + " second parameter");
    expect_true(solved->final_cost < 1e-12,
                name + " solve should reach near-zero cost");
  };

  run.template operator()<ForwardDifferenceSolverPolicy>("forward difference");
  run.template operator()<CentralDifferenceSolverPolicy>("central difference");
}

void test_solve_autodiff_policies() {
  auto static_residual = []<class Scalar>(ConstVectorView<2, Scalar> x,
                                          VectorView<3, Scalar> r) {
    r[0] = x[0] - 1.0;
    r[1] = x[1] - 2.0;
    r[2] = x[0] + x[1] - 3.0;
    return ErrorOrVoid{};
  };
  const auto static_problem = make_problem<3, 2>(static_residual);
  Options static_options;
  static_options.max_iterations = 50;
  SolverWorkspace<AutoDiffSolverPolicy, 3, 2> static_workspace;
  const std::array<double, 2> static_x0{0.0, 0.0};
  SolverContext<AutoDiffSolverPolicy, 3, 2, decltype(static_residual),
                NoJacobian>
      static_context(static_problem, static_options, static_workspace,
                     static_x0);

  const auto static_solved = solve<AutoDiffSolverPolicy>(static_context);
  expect_true(static_solved.has_value(), "direct dual solve should succeed");
  expect_close(static_solved->parameters[0], 1.0, 1e-8, 1e-8,
               "direct dual solve first parameter");
  expect_close(static_solved->parameters[1], 2.0, 1e-8, 1e-8,
               "direct dual solve second parameter");

  auto dynamic_residual =
      []<class Scalar>(ConstVectorView<std::dynamic_extent, Scalar> x,
                       VectorView<std::dynamic_extent, Scalar> r) {
        r[0] = x[0] - 1.0;
        r[1] = x[1] - 2.0;
        r[2] = x[0] + x[1] - 3.0;
        return ErrorOrVoid{};
      };
  const auto dynamic_problem = make_dynamic_problem(3, 2, dynamic_residual);
  Options dynamic_options;
  dynamic_options.max_iterations = 50;
  SolverWorkspace<AutoDiffSolverPolicy, std::dynamic_extent,
                  std::dynamic_extent>
      dynamic_workspace;
  const std::vector<double> dynamic_x0{0.0, 0.0};
  SolverContext<AutoDiffSolverPolicy, std::dynamic_extent, std::dynamic_extent,
                decltype(dynamic_residual), NoJacobian>
      dynamic_context(dynamic_problem, dynamic_options, dynamic_workspace,
                      dynamic_x0);

  const auto dynamic_solved = solve<AutoDiffSolverPolicy>(dynamic_context);
  expect_true(dynamic_solved.has_value(), "graph solve should succeed");
  expect_close(dynamic_solved->parameters[0], 1.0, 1e-8, 1e-8,
               "graph solve first parameter");
  expect_close(dynamic_solved->parameters[1], 2.0, 1e-8, 1e-8,
               "graph solve second parameter");
}

void test_solve_termination_paths() {
  auto residual = [](ConstVectorView<1> x, VectorView<1> r) {
    r[0] = x[0] - 1.0;
    return ErrorOrVoid{};
  };
  auto incorrect_jacobian = [](ConstVectorView<1>, MatrixView<1, 1> jacobian) {
    jacobian[0, 0] = -1.0;
    return ErrorOrVoid{};
  };
  auto jacobian = [](ConstVectorView<1>, MatrixView<1, 1> jacobian) {
    jacobian[0, 0] = 1.0;
    return ErrorOrVoid{};
  };
  const auto incorrect_problem =
      make_problem<1, 1>(residual, incorrect_jacobian);
  const std::array<double, 1> x0{0.0};

  Options damping_options;
  damping_options.max_iterations = 20;
  damping_options.lm.max_lambda = 1e-2;
  SolverWorkspace<DefaultSolverPolicy, 1, 1> damping_workspace;
  SolverContext<DefaultSolverPolicy, 1, 1, decltype(residual),
                decltype(incorrect_jacobian)>
      damping_context(incorrect_problem, damping_options, damping_workspace,
                      x0);
  std::vector<LmTrialTrace> damping_trace;
  damping_context.trial_trace = &damping_trace;
  const auto damping_solved = solve<DefaultSolverPolicy>(damping_context);
  expect_true(damping_solved.has_value(), "damping-limit solve should succeed");
  expect_equal(damping_solved->termination, TerminationReason::DampingLimit,
               "rejected trials should reach damping limit");
  expect_true(damping_solved->linear_solves > 1,
              "rejected trials should grow lambda across multiple solves");
  expect_true(!damping_trace.empty(),
              "damping-limit solve should record trials");
  expect_close(damping_trace.back().last_evaluated_lambda,
               damping_options.lm.max_lambda, 0.0, 0.0,
               "damping-limit trace should report the final attempted lambda");

  Options rejected_step_options = damping_options;
  rejected_step_options.step_tolerance = 2.0;
  SolverWorkspace<DefaultSolverPolicy, 1, 1> rejected_step_workspace;
  SolverContext<DefaultSolverPolicy, 1, 1, decltype(residual),
                decltype(incorrect_jacobian)>
      rejected_step_context(incorrect_problem, rejected_step_options,
                            rejected_step_workspace, x0);
  const auto rejected_step_solved =
      solve<DefaultSolverPolicy>(rejected_step_context);
  expect_true(rejected_step_solved.has_value(),
              "rejected small raw step solve should succeed");
  expect_equal(rejected_step_solved->termination,
               TerminationReason::DampingLimit,
               "rejected small raw step must not report SmallStep");

  const auto problem = make_problem<1, 1>(residual, jacobian);
  Options step_options;
  step_options.step_tolerance = 2.0;
  step_options.gradient_tolerance = 0.0;
  SolverWorkspace<DefaultSolverPolicy, 1, 1> step_workspace;
  SolverContext<DefaultSolverPolicy, 1, 1, decltype(residual),
                decltype(jacobian)>
      step_context(problem, step_options, step_workspace, x0);
  const auto step_solved = solve<DefaultSolverPolicy>(step_context);
  expect_true(step_solved.has_value(), "small-step solve should succeed");
  expect_equal(step_solved->termination, TerminationReason::SmallStep,
               "large step tolerance should terminate with SmallStep");

  using ScaledSolverPolicy =
      SolverPolicy<UserJacobian, LevenbergMarquardt, DampedQr, SquaredLoss,
                   JacobianColumnScaling>;
  auto scaled_residual = [](ConstVectorView<1> x, VectorView<1> r) {
    r[0] = 10.0 * (x[0] - 101.0);
    return ErrorOrVoid{};
  };
  auto scaled_jacobian = [](ConstVectorView<1>, MatrixView<1, 1> jacobian) {
    jacobian[0, 0] = 10.0;
    return ErrorOrVoid{};
  };
  const auto scaled_problem =
      make_problem<1, 1>(scaled_residual, scaled_jacobian);
  Options scaled_step_options;
  scaled_step_options.max_iterations = 1;
  scaled_step_options.step_tolerance = 0.02;
  scaled_step_options.gradient_tolerance = 0.0;
  scaled_step_options.cost_tolerance = 0.0;
  scaled_step_options.lm.min_lambda = 1e-4;
  scaled_step_options.lm.max_lambda = 1e-4;
  SolverWorkspace<ScaledSolverPolicy, 1, 1> scaled_step_workspace;
  const std::array<double, 1> scaled_x0{100.0};
  SolverContext<ScaledSolverPolicy, 1, 1, decltype(scaled_residual),
                decltype(scaled_jacobian)>
      scaled_step_context(scaled_problem, scaled_step_options,
                          scaled_step_workspace, scaled_x0);
  const auto scaled_step_solved =
      solve<ScaledSolverPolicy>(scaled_step_context);
  expect_true(scaled_step_solved.has_value(),
              "scaled relative small-step solve should succeed");
  expect_true(scaled_step_solved->step_norm >
                  scaled_step_options.step_tolerance,
              "scaled relative small-step test requires a raw step above "
              "tolerance");
  expect_equal(scaled_step_solved->termination, TerminationReason::SmallStep,
               "scaled relative step should terminate with SmallStep");

  Options cost_options;
  cost_options.cost_tolerance = 1.0;
  cost_options.gradient_tolerance = 0.0;
  SolverWorkspace<DefaultSolverPolicy, 1, 1> cost_workspace;
  SolverContext<DefaultSolverPolicy, 1, 1, decltype(residual),
                decltype(jacobian)>
      cost_context(problem, cost_options, cost_workspace, x0);
  const auto cost_solved = solve<DefaultSolverPolicy>(cost_context);
  expect_true(cost_solved.has_value(), "small-cost solve should succeed");
  expect_equal(cost_solved->termination, TerminationReason::SmallCostReduction,
               "large cost tolerance should terminate with SmallCostReduction");

  Options relative_cost_options;
  relative_cost_options.cost_tolerance = 0.0;
  relative_cost_options.relative_cost_tolerance = 1.0;
  relative_cost_options.gradient_tolerance = 0.0;
  SolverWorkspace<DefaultSolverPolicy, 1, 1> relative_cost_workspace;
  SolverContext<DefaultSolverPolicy, 1, 1, decltype(residual),
                decltype(jacobian)>
      relative_cost_context(problem, relative_cost_options,
                            relative_cost_workspace, x0);
  const auto relative_cost_solved =
      solve<DefaultSolverPolicy>(relative_cost_context);
  expect_true(relative_cost_solved.has_value(),
              "relative small-cost solve should succeed");
  expect_equal(relative_cost_solved->termination,
               TerminationReason::SmallCostReduction,
               "large relative cost tolerance should terminate with "
               "SmallCostReduction");

  auto mismatched_reduction_residual = [](ConstVectorView<1> x,
                                          VectorView<1> r) {
    r[0] = x[0] - 0.95 * x[0] * x[0] - 1.0;
    return ErrorOrVoid{};
  };
  auto mismatched_reduction_jacobian = [](ConstVectorView<1> x,
                                          MatrixView<1, 1> jacobian) {
    jacobian[0, 0] = 1.0 - 1.9 * x[0];
    return ErrorOrVoid{};
  };
  const auto mismatched_reduction_problem = make_problem<1, 1>(
      mismatched_reduction_residual, mismatched_reduction_jacobian);
  Options mismatched_reduction_options;
  mismatched_reduction_options.max_iterations = 1;
  mismatched_reduction_options.cost_tolerance = 0.1;
  mismatched_reduction_options.gradient_tolerance = 0.0;
  SolverWorkspace<DefaultSolverPolicy, 1, 1> mismatched_reduction_workspace;
  SolverContext<DefaultSolverPolicy, 1, 1,
                decltype(mismatched_reduction_residual),
                decltype(mismatched_reduction_jacobian)>
      mismatched_reduction_context(mismatched_reduction_problem,
                                   mismatched_reduction_options,
                                   mismatched_reduction_workspace, x0);
  const auto mismatched_reduction_solved =
      solve<DefaultSolverPolicy>(mismatched_reduction_context);
  expect_true(mismatched_reduction_solved.has_value(),
              "mismatched reduction solve should succeed");
  expect_equal(mismatched_reduction_solved->accepted_steps, Index{1},
               "mismatched reduction trial should be accepted");
  expect_equal(mismatched_reduction_solved->termination,
               TerminationReason::MaxIterations,
               "small actual reduction with substantial predicted reduction "
               "must not report SmallCostReduction");

  auto nonlinear_residual = [](ConstVectorView<1> x, VectorView<1> r) {
    r[0] = x[0] * x[0] - 1.0;
    return ErrorOrVoid{};
  };
  auto nonlinear_jacobian = [](ConstVectorView<1> x,
                               MatrixView<1, 1> jacobian) {
    jacobian[0, 0] = 2.0 * x[0];
    return ErrorOrVoid{};
  };
  const auto nonlinear_problem =
      make_problem<1, 1>(nonlinear_residual, nonlinear_jacobian);
  const std::array<double, 1> nonlinear_x0{2.0};
  Options budget_options;
  budget_options.max_iterations = 20;
  budget_options.max_function_evaluations = 3;
  SolverWorkspace<DefaultSolverPolicy, 1, 1> budget_workspace;
  SolverContext<DefaultSolverPolicy, 1, 1, decltype(nonlinear_residual),
                decltype(nonlinear_jacobian)>
      budget_context(nonlinear_problem, budget_options, budget_workspace,
                     nonlinear_x0);
  const auto budget_solved = solve<DefaultSolverPolicy>(budget_context);
  expect_true(budget_solved.has_value(), "budget solve should succeed");
  expect_equal(budget_solved->termination,
               TerminationReason::MaxFunctionEvaluations,
               "accepted relinearization should exhaust the exact budget");
  expect_equal(budget_solved->function_evaluations, Index{3},
               "accepted relinearization should use all allowed evaluations");
}

} // namespace

int main() {
  test_runtime_option_validation();
  test_dual_arithmetic_and_math();
  test_dual_pow_domain_error();
  test_evaluate_residual_dual_static();
  test_evaluate_residual_dual_dynamic_residuals();
  test_evaluate_residual_dual_errors();
  test_graph_and_direct_dual_policies_agree();
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
  test_evaluate_jacobian_autodiff_static_direct_dual();
  test_evaluate_jacobian_autodiff_dynamic_cache();
  test_evaluate_jacobian_autodiff_literal_residual();
  test_evaluate_jacobian_autodiff_double_only_residual();
  test_evaluate_jacobian_autodiff_mixed_extents();
  test_extended_primitives_forward();
  test_extended_primitives_scalar_generic_residual();
  test_pow_domain_errors();
  test_damped_qr_static();
  test_pivoted_householder_qr_static();
  test_damped_qr_dynamic();
  test_damped_qr_column_scaling_static();
  test_pivoted_householder_qr_column_scaling_static();
  test_pivoted_householder_qr_unpermutation();
  test_pivoted_householder_qr_scaled_unpermutation();
  test_pivoted_householder_qr_scaled_factor_invariants();
  test_pivoted_householder_qr_frozen_ill_conditioned_system();
  test_pivoted_householder_qr_rank_deficient();
  test_pivoted_householder_qr_dynamic_and_mixed();
  test_solve_pivoted_householder_qr_policy();
  test_pivoted_householder_qr_factor_lifecycle();
  test_bisection_lambda_selection_static();
  test_bisection_lambda_selection_dynamic();
  test_pivoted_lmpar_lambda_selection_static();
  test_pivoted_lmpar_lambda_selection_dynamic();
  test_pivoted_interior_gauss_newton_step_static();
  test_pivoted_interior_gauss_newton_step_dynamic();
  test_pivoted_lmpar_correction_static();
  test_pivoted_lmpar_correction_dynamic();
  test_pivoted_lmpar_rejected_trial_preserves_warm_start();
  test_model_cost_cancellation();
  test_solve_static_user_jacobian();
  test_solve_dynamic_user_jacobian();
  test_solve_forward_difference_evaluation_budget();
  test_solve_numerical_termination();
  test_solve_callback_error();
  test_solve_zero_iteration_budget();
  test_solve_finite_difference_policies();
  test_solve_autodiff_policies();
  test_solve_termination_paths();
  return 0;
}
