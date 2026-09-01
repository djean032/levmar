#include <levmar/internal/solver.h>
#include <levmar/lm.h>

#ifdef LEVMAR_HAVE_CERES
#include <ceres/autodiff_cost_function.h>
#include <ceres/ceres.h>
#endif

#ifdef LEVMAR_HAVE_CMINPACK
#include <cminpack.h>
#endif

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace levmar;
using namespace levmar::detail;
using std::exp;

using NistAnalyticSolverPolicy =
    SolverPolicy<UserJacobian, LevenbergMarquardt, DampedQr, SquaredLoss,
                 JacobianColumnScaling>;
using NistAutoDiffSolverPolicy =
    SolverPolicy<AutoDiffJacobian, LevenbergMarquardt, DampedQr, SquaredLoss,
                 JacobianColumnScaling>;

template <class LinearAlgebra>
using NistAnalyticSolverPolicyFor =
    SolverPolicy<UserJacobian, LevenbergMarquardt, LinearAlgebra, SquaredLoss,
                 JacobianColumnScaling>;
template <class LinearAlgebra>
using NistAutoDiffSolverPolicyFor =
    SolverPolicy<AutoDiffJacobian, LevenbergMarquardt, LinearAlgebra,
                 SquaredLoss, JacobianColumnScaling>;

struct CorpusProblem {
  std::string name;
  std::string model_id;
  std::string model_class;
  std::string difficulty;
  Index m = 0;
  Index n = 0;
  Index predictor_count = 0;
  bool numerical_derivatives_recommended = true;
  std::vector<std::vector<double>> data;
  std::vector<std::pair<std::string, std::vector<double>>> params;
};

struct ComparisonStats {
  std::uint64_t count = 0;
  double max_abs_error = 0.0;
  double max_rel_error = 0.0;
  double sum_sq_abs_error = 0.0;
  double sum_sq_rel_error = 0.0;

  void add(double actual, double expected) {
    const double abs_error = std::abs(actual - expected);
    const double rel_error = abs_error / std::max(std::abs(expected), 1e-16);
    ++count;
    max_abs_error = std::max(max_abs_error, abs_error);
    max_rel_error = std::max(max_rel_error, rel_error);
    sum_sq_abs_error += abs_error * abs_error;
    sum_sq_rel_error += rel_error * rel_error;
  }

  double rms_abs_error() const {
    return count == 0 ? 0.0 : std::sqrt(sum_sq_abs_error / count);
  }

  double rms_rel_error() const {
    return count == 0 ? 0.0 : std::sqrt(sum_sq_rel_error / count);
  }
};

struct TimingStats {
  struct ModeComparison {
    double analytic_residual_and_jacobian_seconds = 0.0;
    double autodiff_residual_and_jacobian_seconds = 0.0;
    double graph_residual_and_jacobian_seconds = 0.0;
    double direct_dual_residual_and_jacobian_seconds = 0.0;
    bool graph_available = false;
    bool direct_dual_available = false;
    double forward_difference_residual_and_jacobian_seconds = 0.0;
    double central_difference_residual_and_jacobian_seconds = 0.0;
  };

  double residual_seconds = 0.0;
  double analytic_jacobian_seconds = 0.0;
  double analytic_residual_and_jacobian_seconds = 0.0;
  double autodiff_jacobian_seconds = 0.0;
  double graph_jacobian_seconds = 0.0;
  double direct_dual_jacobian_seconds = 0.0;
  bool graph_jacobian_available = false;
  bool direct_dual_jacobian_available = false;
  double forward_difference_seconds = 0.0;
  double forward_difference_residual_and_jacobian_seconds = 0.0;
  double central_difference_seconds = 0.0;
  double central_difference_residual_and_jacobian_seconds = 0.0;
  double total_seconds = 0.0;
  ModeComparison numerical_subset;
};

struct ScalarMoments {
  std::uint64_t count = 0;
  double sum = 0.0;
  double sum_sq = 0.0;

  void add(double value) {
    ++count;
    sum += value;
    sum_sq += value * value;
  }

  double mean() const { return count == 0 ? 0.0 : sum / count; }

  double stddev() const {
    if (count == 0) {
      return 0.0;
    }
    const double mean_value = mean();
    const double variance =
        std::max(0.0, sum_sq / count - mean_value * mean_value);
    return std::sqrt(variance);
  }
};

struct TimingMoments {
  ScalarMoments residual;
  ScalarMoments analytic;
  ScalarMoments analytic_residual_and_jacobian;
  ScalarMoments autodiff;
  ScalarMoments graph;
  ScalarMoments direct_dual;
  ScalarMoments forward_difference;
  ScalarMoments forward_difference_residual_and_jacobian;
  ScalarMoments central_difference;
  ScalarMoments central_difference_residual_and_jacobian;
  ScalarMoments total;
  struct ModeComparison {
    ScalarMoments analytic;
    ScalarMoments autodiff;
    ScalarMoments graph;
    ScalarMoments direct_dual;
    ScalarMoments forward_difference;
    ScalarMoments central_difference;

    void add(const TimingStats::ModeComparison &timing) {
      analytic.add(timing.analytic_residual_and_jacobian_seconds);
      autodiff.add(timing.autodiff_residual_and_jacobian_seconds);
      if (timing.graph_available) {
        graph.add(timing.graph_residual_and_jacobian_seconds);
      }
      if (timing.direct_dual_available) {
        direct_dual.add(timing.direct_dual_residual_and_jacobian_seconds);
      }
      forward_difference.add(
          timing.forward_difference_residual_and_jacobian_seconds);
      central_difference.add(
          timing.central_difference_residual_and_jacobian_seconds);
    }
  } numerical_subset;

  void add(const TimingStats &timing) {
    residual.add(timing.residual_seconds);
    analytic.add(timing.analytic_jacobian_seconds);
    analytic_residual_and_jacobian.add(
        timing.analytic_residual_and_jacobian_seconds);
    autodiff.add(timing.autodiff_jacobian_seconds);
    if (timing.graph_jacobian_available) {
      graph.add(timing.graph_jacobian_seconds);
    }
    if (timing.direct_dual_jacobian_available) {
      direct_dual.add(timing.direct_dual_jacobian_seconds);
    }
    forward_difference.add(timing.forward_difference_seconds);
    forward_difference_residual_and_jacobian.add(
        timing.forward_difference_residual_and_jacobian_seconds);
    central_difference.add(timing.central_difference_seconds);
    central_difference_residual_and_jacobian.add(
        timing.central_difference_residual_and_jacobian_seconds);
    total.add(timing.total_seconds);
    numerical_subset.add(timing.numerical_subset);
  }
};

struct ProblemBenchmark {
  std::string name;
  TimingMoments dynamic_timing;
  TimingMoments static_timing;
};

struct SummaryBenchmark {
  TimingMoments dynamic_timing;
  TimingMoments static_timing;
};

struct SolverBenchmark {
  ScalarMoments dynamic_total;
  ScalarMoments static_total;
};

struct ExternalSolverBenchmark {
  // One sample per complete NIST sweep, matching SolverBenchmark semantics.
  ScalarMoments seconds;

  // Conformance/work statistics are collected once per (problem, start) case,
  // not once per timing repetition.
  ScalarMoments lre;
  ScalarMoments iterations;
  ScalarMoments function_evaluations;
  ScalarMoments jacobian_evaluations;
  ScalarMoments linear_solves;
  ScalarMoments accepted_steps;
  ScalarMoments rejected_steps;
  std::uint64_t usable = 0;
  std::uint64_t cases = 0;
};

struct ExternalSolverResult {
  std::string solver;
  std::string start_label;
  Index m = 0;
  Index n = 0;
  double seconds = 0.0;
  double lre = 0.0;
  std::uint64_t iterations = 0;
  std::uint64_t function_evaluations = 0;
  std::uint64_t jacobian_evaluations = 0;
  std::uint64_t linear_solves = 0;
  std::uint64_t accepted_steps = 0;
  std::uint64_t rejected_steps = 0;
  bool has_jacobian_evaluations = false;
  bool has_linear_solves = false;
  bool has_accepted_steps = false;
  bool has_rejected_steps = false;
  bool usable = false;
};

struct SolverWorkRow {
  std::string start_label;
  std::string extent;
  Index m = 0;
  Index n = 0;
  double elapsed_seconds = 0.0;
  double lre = 0.0;
  Result result;
};

struct ControllerTraceRecord {
  std::string problem;
  std::string start;
  std::string derivative;
  std::string extent;
  std::vector<LmTrialTrace> trials;
};

bool is_controller_trace_target(std::string_view problem,
                                std::string_view start,
                                std::string_view derivative,
                                std::string_view extent) {
  return (problem == "MGH10" && start == "start1" &&
          ((derivative == "analytic" && extent == "dynamic") ||
           (derivative == "autodiff" && extent == "static"))) ||
         (problem == "Bennett5" && start == "start2" &&
          ((derivative == "analytic" && extent == "dynamic") ||
           (derivative == "autodiff" && extent == "static"))) ||
         (derivative == "analytic" && extent == "dynamic" &&
          ((problem == "Lanczos1" && start == "start1") ||
           (problem == "MGH09" && start == "start1") ||
           (problem == "Roszman1" && start == "start1") ||
           (problem == "Gauss1" && start == "start2") ||
           (problem == "MGH17" && start == "start1") ||
           (problem == "Hahn1" && start == "start1") ||
           (problem == "Thurber" && start == "start1"))) ||
         (derivative == "autodiff" && extent == "static" &&
          ((problem == "Rat43" && start == "start1") ||
           (problem == "MGH10" && start == "start2") ||
           (problem == "Kirby2" && start == "start1")));
}

std::vector<LmTrialTrace> *
controller_trace_trials(std::vector<ControllerTraceRecord> *records,
                        std::string_view problem, std::string_view start,
                        std::string_view derivative, std::string_view extent) {
  if (records == nullptr ||
      !is_controller_trace_target(problem, start, derivative, extent)) {
    return nullptr;
  }
  const auto existing =
      std::ranges::find_if(*records, [&](const ControllerTraceRecord &record) {
        return record.problem == problem && record.start == start &&
               record.derivative == derivative && record.extent == extent;
      });
  if (existing != records->end()) {
    return &existing->trials;
  }
  records->push_back({std::string(problem),
                      std::string(start),
                      std::string(derivative),
                      std::string(extent),
                      {}});
  return &records->back().trials;
}

struct ProblemReport {
  std::string name;
  bool numerical_derivatives_skipped = false;
  Index solver_start_cases = 0;
  Index solver_start_case_passes = 0;
  double solver_dynamic_lre_sum = 0.0;
  double solver_static_lre_sum = 0.0;
  double solver_dynamic_seconds = 0.0;
  double solver_static_seconds = 0.0;
  Index solver_dynamic_iterations = 0;
  Index solver_static_iterations = 0;
  Index solver_dynamic_function_evaluations = 0;
  Index solver_static_function_evaluations = 0;
  Index solver_dynamic_jacobian_evaluations = 0;
  Index solver_static_jacobian_evaluations = 0;
  Index solver_dynamic_linear_solves = 0;
  Index solver_static_linear_solves = 0;
  Index solver_dynamic_accepted_steps = 0;
  Index solver_static_accepted_steps = 0;
  Index solver_dynamic_rejected_steps = 0;
  Index solver_static_rejected_steps = 0;
  std::vector<std::string> solver_failures;
  std::vector<SolverWorkRow> solver_work_rows;
  std::vector<ExternalSolverResult> external_solver_results;
  TimingStats dynamic_timing;
  TimingStats static_timing;
  ComparisonStats residual_stats;
  ComparisonStats analytic_stats;
  ComparisonStats autodiff_stats;
  ComparisonStats autodiff_vs_analytic_stats;
  ComparisonStats graph_autodiff_stats;
  ComparisonStats graph_autodiff_vs_analytic_stats;
  ComparisonStats direct_dual_stats;
  ComparisonStats direct_dual_vs_analytic_stats;
  ComparisonStats forward_difference_stats;
  ComparisonStats central_difference_stats;
};

struct SummaryStats {
  std::uint64_t problems = 0;
  std::uint64_t numerical_derivative_skips = 0;
  Index solver_start_cases = 0;
  Index solver_start_case_passes = 0;
  double solver_dynamic_lre_sum = 0.0;
  double solver_static_lre_sum = 0.0;
  double solver_dynamic_seconds = 0.0;
  double solver_static_seconds = 0.0;
  Index solver_dynamic_iterations = 0;
  Index solver_static_iterations = 0;
  Index solver_dynamic_function_evaluations = 0;
  Index solver_static_function_evaluations = 0;
  Index solver_dynamic_jacobian_evaluations = 0;
  Index solver_static_jacobian_evaluations = 0;
  Index solver_dynamic_linear_solves = 0;
  Index solver_static_linear_solves = 0;
  Index solver_dynamic_accepted_steps = 0;
  Index solver_static_accepted_steps = 0;
  Index solver_dynamic_rejected_steps = 0;
  Index solver_static_rejected_steps = 0;
  std::vector<std::string> solver_failures;
  std::vector<ExternalSolverResult> external_solver_results;
  TimingStats dynamic_timing;
  TimingStats static_timing;
  ComparisonStats residual_stats;
  ComparisonStats analytic_stats;
  ComparisonStats autodiff_stats;
  ComparisonStats autodiff_vs_analytic_stats;
  ComparisonStats graph_autodiff_stats;
  ComparisonStats graph_autodiff_vs_analytic_stats;
  ComparisonStats direct_dual_stats;
  ComparisonStats direct_dual_vs_analytic_stats;
  ComparisonStats forward_difference_stats;
  ComparisonStats central_difference_stats;
};

using Clock = std::chrono::steady_clock;
constexpr std::uint64_t kKernelTimingRepeats = 64;

double milliseconds(double seconds) { return seconds * 1000.0; }

double elapsed_seconds(Clock::time_point start, Clock::time_point end);

std::string solver_diagnostics(const Result &result) {
  std::ostringstream message;
  message << std::scientific << "termination "
          << static_cast<int>(result.termination) << ", iterations "
          << result.iterations << ", function evaluations "
          << result.function_evaluations << ", lambda " << result.lambda
          << ", gradient " << result.gradient_inf_norm;
  return message.str();
}

std::string_view termination_reason_name(TerminationReason termination) {
  switch (termination) {
  case TerminationReason::NotTerminated:
    return "not_terminated";
  case TerminationReason::SmallStep:
    return "small_step";
  case TerminationReason::SmallGradient:
    return "small_gradient";
  case TerminationReason::SmallCostReduction:
    return "small_cost_reduction";
  case TerminationReason::MaxIterations:
    return "max_iterations";
  case TerminationReason::MaxFunctionEvaluations:
    return "max_function_evaluations";
  case TerminationReason::NumericalFailure:
    return "numerical_failure";
  case TerminationReason::DampingLimit:
    return "damping_limit";
  }
  return "unknown";
}

double log_relative_error(const std::vector<double> &parameters,
                          const std::vector<double> &certified) {
  constexpr double kMaxDigits = 11.0;
  double lre = kMaxDigits;
  for (Index j = 0; j < parameters.size(); ++j) {
    const double relative_error =
        std::abs(certified[j] - parameters[j]) / std::abs(certified[j]);
    const double parameter_lre = -std::log10(relative_error);
    lre = std::min(lre, std::clamp(parameter_lre, 0.0, kMaxDigits));
  }
  return lre;
}

#include "external_benchmark_adapters.h"

template <class Fn>
double measure_average_seconds(std::uint64_t repeats, Fn &&fn) {
  const auto start = Clock::now();
  for (std::uint64_t i = 0; i < repeats; ++i) {
    fn();
  }
  return std::chrono::duration<double>(Clock::now() - start).count() /
         static_cast<double>(repeats);
}

std::vector<std::string> split_csv_line(const std::string &line) {
  std::vector<std::string> parts;
  std::stringstream stream(line);
  std::string cell;
  while (std::getline(stream, cell, ',')) {
    parts.push_back(cell);
  }
  return parts;
}

std::map<std::string, std::string>
read_meta(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to open " + path.string());
  }

  std::map<std::string, std::string> meta;
  std::string line;
  while (std::getline(file, line)) {
    const auto pos = line.find('=');
    if (pos == std::string::npos) {
      throw std::runtime_error("Invalid meta line in " + path.string());
    }
    meta.emplace(line.substr(0, pos), line.substr(pos + 1));
  }
  return meta;
}

std::vector<std::vector<double>>
read_numeric_csv(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to open " + path.string());
  }

  std::string line;
  if (!std::getline(file, line)) {
    throw std::runtime_error("Missing CSV header in " + path.string());
  }

  std::vector<std::vector<double>> rows;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }
    std::vector<double> row;
    for (const auto &cell : split_csv_line(line)) {
      row.push_back(std::stod(cell));
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

std::vector<std::pair<std::string, std::vector<double>>>
read_params_csv(const std::filesystem::path &path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to open " + path.string());
  }

  std::string line;
  if (!std::getline(file, line)) {
    throw std::runtime_error("Missing CSV header in " + path.string());
  }

  std::vector<std::pair<std::string, std::vector<double>>> params;
  while (std::getline(file, line)) {
    if (line.empty()) {
      continue;
    }
    const auto parts = split_csv_line(line);
    std::vector<double> beta;
    for (Index i = 1; i < parts.size(); ++i) {
      beta.push_back(std::stod(parts[i]));
    }
    params.emplace_back(parts[0], std::move(beta));
  }
  return params;
}

CorpusProblem load_problem(const std::filesystem::path &path) {
  const auto meta = read_meta(path / "meta.txt");

  CorpusProblem problem;
  problem.name = meta.at("name");
  problem.model_id = meta.at("model_id");
  problem.model_class = meta.at("model_class");
  problem.difficulty = meta.at("difficulty");
  problem.m = static_cast<Index>(std::stoull(meta.at("m")));
  problem.n = static_cast<Index>(std::stoull(meta.at("n")));
  problem.predictor_count =
      static_cast<Index>(std::stoull(meta.at("predictor_count")));
  problem.numerical_derivatives_recommended =
      meta.at("numerical_derivatives_recommended") == "true";
  problem.data = read_numeric_csv(path / "data.csv");
  problem.params = read_params_csv(path / "params.csv");

  if (problem.data.size() != problem.m) {
    throw std::runtime_error("Observation count mismatch in " + path.string());
  }
  for (const auto &row : problem.data) {
    if (row.size() != problem.predictor_count + 1) {
      throw std::runtime_error("Predictor count mismatch in " + path.string());
    }
  }
  return problem;
}

bool nearly_equal(double actual, double expected, double atol, double rtol) {
  return std::abs(actual - expected) <= atol + rtol * std::abs(expected);
}

void expect_close(double actual, double expected, double atol, double rtol,
                  const std::string &what) {
  if (!nearly_equal(actual, expected, atol, rtol)) {
    std::ostringstream message;
    message << std::scientific;
    message << what << ": got " << actual << ", expected " << expected;
    throw std::runtime_error(message.str());
  }
}

double elapsed_seconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

void merge_stats(ComparisonStats &into, const ComparisonStats &from) {
  into.count += from.count;
  into.max_abs_error = std::max(into.max_abs_error, from.max_abs_error);
  into.max_rel_error = std::max(into.max_rel_error, from.max_rel_error);
  into.sum_sq_abs_error += from.sum_sq_abs_error;
  into.sum_sq_rel_error += from.sum_sq_rel_error;
}

void merge_summary(SummaryStats &summary, const ProblemReport &report) {
  ++summary.problems;
  if (report.numerical_derivatives_skipped) {
    ++summary.numerical_derivative_skips;
  }
  summary.solver_start_cases += report.solver_start_cases;
  summary.solver_start_case_passes += report.solver_start_case_passes;
  summary.solver_dynamic_lre_sum += report.solver_dynamic_lre_sum;
  summary.solver_static_lre_sum += report.solver_static_lre_sum;
  summary.solver_dynamic_seconds += report.solver_dynamic_seconds;
  summary.solver_static_seconds += report.solver_static_seconds;
  summary.solver_dynamic_iterations += report.solver_dynamic_iterations;
  summary.solver_static_iterations += report.solver_static_iterations;
  summary.solver_dynamic_function_evaluations +=
      report.solver_dynamic_function_evaluations;
  summary.solver_static_function_evaluations +=
      report.solver_static_function_evaluations;
  summary.solver_dynamic_jacobian_evaluations +=
      report.solver_dynamic_jacobian_evaluations;
  summary.solver_static_jacobian_evaluations +=
      report.solver_static_jacobian_evaluations;
  summary.solver_dynamic_linear_solves += report.solver_dynamic_linear_solves;
  summary.solver_static_linear_solves += report.solver_static_linear_solves;
  summary.solver_dynamic_accepted_steps += report.solver_dynamic_accepted_steps;
  summary.solver_static_accepted_steps += report.solver_static_accepted_steps;
  summary.solver_dynamic_rejected_steps += report.solver_dynamic_rejected_steps;
  summary.solver_static_rejected_steps += report.solver_static_rejected_steps;
  summary.solver_failures.insert(summary.solver_failures.end(),
                                 report.solver_failures.begin(),
                                 report.solver_failures.end());
  summary.external_solver_results.insert(summary.external_solver_results.end(),
                                         report.external_solver_results.begin(),
                                         report.external_solver_results.end());
  summary.dynamic_timing.residual_seconds +=
      report.dynamic_timing.residual_seconds;
  summary.dynamic_timing.analytic_jacobian_seconds +=
      report.dynamic_timing.analytic_jacobian_seconds;
  summary.dynamic_timing.analytic_residual_and_jacobian_seconds +=
      report.dynamic_timing.analytic_residual_and_jacobian_seconds;
  summary.dynamic_timing.graph_jacobian_seconds +=
      report.dynamic_timing.graph_jacobian_seconds;
  summary.dynamic_timing.direct_dual_jacobian_seconds +=
      report.dynamic_timing.direct_dual_jacobian_seconds;
  summary.dynamic_timing.graph_jacobian_available =
      summary.dynamic_timing.graph_jacobian_available ||
      report.dynamic_timing.graph_jacobian_available;
  summary.dynamic_timing.direct_dual_jacobian_available =
      summary.dynamic_timing.direct_dual_jacobian_available ||
      report.dynamic_timing.direct_dual_jacobian_available;
  summary.dynamic_timing.forward_difference_seconds +=
      report.dynamic_timing.forward_difference_seconds;
  summary.dynamic_timing.forward_difference_residual_and_jacobian_seconds +=
      report.dynamic_timing.forward_difference_residual_and_jacobian_seconds;
  summary.dynamic_timing.central_difference_seconds +=
      report.dynamic_timing.central_difference_seconds;
  summary.dynamic_timing.central_difference_residual_and_jacobian_seconds +=
      report.dynamic_timing.central_difference_residual_and_jacobian_seconds;
  summary.dynamic_timing.total_seconds += report.dynamic_timing.total_seconds;
  summary.dynamic_timing.numerical_subset
      .analytic_residual_and_jacobian_seconds +=
      report.dynamic_timing.numerical_subset
          .analytic_residual_and_jacobian_seconds;
  summary.dynamic_timing.numerical_subset.graph_residual_and_jacobian_seconds +=
      report.dynamic_timing.numerical_subset
          .graph_residual_and_jacobian_seconds;
  summary.dynamic_timing.numerical_subset
      .direct_dual_residual_and_jacobian_seconds +=
      report.dynamic_timing.numerical_subset
          .direct_dual_residual_and_jacobian_seconds;
  summary.dynamic_timing.numerical_subset.graph_available =
      summary.dynamic_timing.numerical_subset.graph_available ||
      report.dynamic_timing.numerical_subset.graph_available;
  summary.dynamic_timing.numerical_subset.direct_dual_available =
      summary.dynamic_timing.numerical_subset.direct_dual_available ||
      report.dynamic_timing.numerical_subset.direct_dual_available;
  summary.dynamic_timing.numerical_subset
      .forward_difference_residual_and_jacobian_seconds +=
      report.dynamic_timing.numerical_subset
          .forward_difference_residual_and_jacobian_seconds;
  summary.dynamic_timing.numerical_subset
      .central_difference_residual_and_jacobian_seconds +=
      report.dynamic_timing.numerical_subset
          .central_difference_residual_and_jacobian_seconds;
  summary.static_timing.residual_seconds +=
      report.static_timing.residual_seconds;
  summary.static_timing.analytic_jacobian_seconds +=
      report.static_timing.analytic_jacobian_seconds;
  summary.static_timing.analytic_residual_and_jacobian_seconds +=
      report.static_timing.analytic_residual_and_jacobian_seconds;
  summary.static_timing.graph_jacobian_seconds +=
      report.static_timing.graph_jacobian_seconds;
  summary.static_timing.direct_dual_jacobian_seconds +=
      report.static_timing.direct_dual_jacobian_seconds;
  summary.static_timing.graph_jacobian_available =
      summary.static_timing.graph_jacobian_available ||
      report.static_timing.graph_jacobian_available;
  summary.static_timing.direct_dual_jacobian_available =
      summary.static_timing.direct_dual_jacobian_available ||
      report.static_timing.direct_dual_jacobian_available;
  summary.static_timing.forward_difference_seconds +=
      report.static_timing.forward_difference_seconds;
  summary.static_timing.forward_difference_residual_and_jacobian_seconds +=
      report.static_timing.forward_difference_residual_and_jacobian_seconds;
  summary.static_timing.central_difference_seconds +=
      report.static_timing.central_difference_seconds;
  summary.static_timing.central_difference_residual_and_jacobian_seconds +=
      report.static_timing.central_difference_residual_and_jacobian_seconds;
  summary.static_timing.total_seconds += report.static_timing.total_seconds;
  summary.static_timing.numerical_subset
      .analytic_residual_and_jacobian_seconds +=
      report.static_timing.numerical_subset
          .analytic_residual_and_jacobian_seconds;
  summary.static_timing.numerical_subset.graph_residual_and_jacobian_seconds +=
      report.static_timing.numerical_subset.graph_residual_and_jacobian_seconds;
  summary.static_timing.numerical_subset
      .direct_dual_residual_and_jacobian_seconds +=
      report.static_timing.numerical_subset
          .direct_dual_residual_and_jacobian_seconds;
  summary.static_timing.numerical_subset.graph_available =
      summary.static_timing.numerical_subset.graph_available ||
      report.static_timing.numerical_subset.graph_available;
  summary.static_timing.numerical_subset.direct_dual_available =
      summary.static_timing.numerical_subset.direct_dual_available ||
      report.static_timing.numerical_subset.direct_dual_available;
  summary.static_timing.numerical_subset
      .forward_difference_residual_and_jacobian_seconds +=
      report.static_timing.numerical_subset
          .forward_difference_residual_and_jacobian_seconds;
  summary.static_timing.numerical_subset
      .central_difference_residual_and_jacobian_seconds +=
      report.static_timing.numerical_subset
          .central_difference_residual_and_jacobian_seconds;
  merge_stats(summary.residual_stats, report.residual_stats);
  merge_stats(summary.analytic_stats, report.analytic_stats);
  merge_stats(summary.graph_autodiff_stats, report.graph_autodiff_stats);
  merge_stats(summary.graph_autodiff_vs_analytic_stats,
              report.graph_autodiff_vs_analytic_stats);
  merge_stats(summary.direct_dual_stats, report.direct_dual_stats);
  merge_stats(summary.direct_dual_vs_analytic_stats,
              report.direct_dual_vs_analytic_stats);
  merge_stats(summary.forward_difference_stats,
              report.forward_difference_stats);
  merge_stats(summary.central_difference_stats,
              report.central_difference_stats);
}

template <Index M, Index N, class ResidualFn, class JacobianFn>
void run_kernel_variant(
    const CorpusProblem &corpus, ResidualFn residual, JacobianFn jacobian,
    ConstVectorView<N> beta,
    const std::vector<std::vector<double>> &expected_residuals,
    const std::vector<std::vector<double>> &expected_jacobian,
    bool run_numerical_derivatives, TimingStats &timing,
    ComparisonStats *residual_stats, ComparisonStats *analytic_stats,
    ComparisonStats *graph_autodiff_stats,
    ComparisonStats *graph_autodiff_vs_analytic_stats,
    ComparisonStats *direct_dual_stats,
    ComparisonStats *direct_dual_vs_analytic_stats,
    ComparisonStats *forward_difference_stats,
    ComparisonStats *central_difference_stats, const std::string &what_prefix) {
  auto problem = [&]() {
    if constexpr (M == std::dynamic_extent && N == std::dynamic_extent) {
      return make_dynamic_problem(corpus.m, corpus.n, residual, jacobian);
    } else if constexpr (M == std::dynamic_extent) {
      return make_problem_dynamic_residuals<N>(corpus.m, residual, jacobian);
    } else if constexpr (N == std::dynamic_extent) {
      return make_problem_dynamic_parameters<M>(corpus.n, residual, jacobian);
    } else {
      return make_problem<M, N>(residual, jacobian);
    }
  }();

  using ContextType =
      LMSolveContext<M, N, decltype(residual), decltype(jacobian)>;

  Options options;
  Result result;
  LMWorkspace<M, N> workspace;
  ContextType context(problem, options, result, workspace, beta);

  std::ranges::copy(context.x, workspace.x_current.view().begin());
  timing.residual_seconds += measure_average_seconds(kKernelTimingRepeats, [&] {
    std::ranges::copy(context.x, workspace.x_current.view().begin());
    if (auto residual_result =
            evaluate_residual(context, what_prefix + " residual");
        !residual_result) {
      throw std::runtime_error(residual_result.error().message);
    }
  });
  for (Index i = 0; i < corpus.m; ++i) {
    if (residual_stats != nullptr) {
      residual_stats->add(workspace.r.view()[i], expected_residuals[i][0]);
    }
    expect_close(workspace.r.view()[i], expected_residuals[i][0], 1e-12, 1e-10,
                 what_prefix + " residual row " + std::to_string(i));
  }

  std::vector<double> analytic_jacobian(corpus.m * corpus.n);
  timing.analytic_jacobian_seconds +=
      measure_average_seconds(kKernelTimingRepeats, [&] {
        std::ranges::copy(context.x, workspace.x_current.view().begin());
        if (auto jacobian_result = evaluate_jacobian<UserJacobian>(
                context, what_prefix + " analytic jacobian");
            !jacobian_result) {
          throw std::runtime_error(jacobian_result.error().message);
        }
      });
  for (Index j = 0; j < corpus.n; ++j) {
    for (Index i = 0; i < corpus.m; ++i) {
      if (analytic_stats != nullptr) {
        analytic_stats->add(workspace.J(i, j), expected_jacobian[i][j]);
      }
      analytic_jacobian[i + j * corpus.m] = workspace.J(i, j);
      expect_close(workspace.J(i, j), expected_jacobian[i][j], 1e-12, 1e-10,
                   what_prefix + " analytic jacobian row " + std::to_string(i) +
                       " col " + std::to_string(j));
    }
  }
  timing.analytic_residual_and_jacobian_seconds +=
      measure_average_seconds(kKernelTimingRepeats, [&] {
        std::ranges::copy(context.x, workspace.x_current.view().begin());
        if (auto residual_result =
                evaluate_residual(context, what_prefix + " analytic residual");
            !residual_result) {
          throw std::runtime_error(residual_result.error().message);
        }
        if (auto jacobian_result = evaluate_jacobian<UserJacobian>(
                context, what_prefix + " analytic residual/jacobian");
            !jacobian_result) {
          throw std::runtime_error(jacobian_result.error().message);
        }
      });
  if (run_numerical_derivatives) {
    timing.numerical_subset.analytic_residual_and_jacobian_seconds +=
        measure_average_seconds(kKernelTimingRepeats, [&] {
          std::ranges::copy(context.x, workspace.x_current.view().begin());
          if (auto residual_result = evaluate_residual(
                  context, what_prefix + " subset analytic residual");
              !residual_result) {
            throw std::runtime_error(residual_result.error().message);
          }
          if (auto jacobian_result = evaluate_jacobian<UserJacobian>(
                  context, what_prefix + " subset analytic residual/jacobian");
              !jacobian_result) {
            throw std::runtime_error(jacobian_result.error().message);
          }
        });
  }

  if constexpr (AutoDiffResidualCallable<ResidualFn, M, N>) {
    if (auto activation = GraphAutoDiffJacobian::activate(
            context, what_prefix + " graph autodiff activation");
        !activation) {
      throw std::runtime_error(activation.error().message);
    }
    timing.graph_jacobian_available = true;
    timing.graph_jacobian_seconds +=
        measure_average_seconds(kKernelTimingRepeats, [&] {
          std::ranges::copy(context.x, workspace.x_current.view().begin());
          if (auto graph_result = GraphAutoDiffJacobian::evaluate(
                  context, what_prefix + " graph autodiff jacobian");
              !graph_result) {
            throw std::runtime_error(graph_result.error().message);
          }
        });
    if (run_numerical_derivatives) {
      timing.numerical_subset.graph_available = true;
      timing.numerical_subset.graph_residual_and_jacobian_seconds +=
          measure_average_seconds(kKernelTimingRepeats, [&] {
            std::ranges::copy(context.x, workspace.x_current.view().begin());
            if (auto graph_result = GraphAutoDiffJacobian::evaluate(
                    context,
                    what_prefix + " subset graph autodiff residual/jacobian");
                !graph_result) {
              throw std::runtime_error(graph_result.error().message);
            }
          });
    }
    for (Index j = 0; j < corpus.n; ++j) {
      for (Index i = 0; i < corpus.m; ++i) {
        if (graph_autodiff_stats != nullptr) {
          graph_autodiff_stats->add(workspace.J(i, j), expected_jacobian[i][j]);
        }
        if (graph_autodiff_vs_analytic_stats != nullptr) {
          graph_autodiff_vs_analytic_stats->add(
              workspace.J(i, j), analytic_jacobian[i + j * corpus.m]);
        }
        expect_close(workspace.J(i, j), expected_jacobian[i][j], 1e-12, 1e-10,
                     what_prefix + " graph autodiff jacobian row " +
                         std::to_string(i) + " col " + std::to_string(j));
      }
    }

    if constexpr (kUsesDirectDualAutoDiff<N>) {
      timing.direct_dual_jacobian_available = true;
      timing.direct_dual_jacobian_seconds +=
          measure_average_seconds(kKernelTimingRepeats, [&] {
            std::ranges::copy(context.x, workspace.x_current.view().begin());
            if (auto dual_result = DirectDualJacobian::evaluate(
                    context, what_prefix + " direct-dual jacobian");
                !dual_result) {
              throw std::runtime_error(dual_result.error().message);
            }
          });
      if (run_numerical_derivatives) {
        timing.numerical_subset.direct_dual_available = true;
        timing.numerical_subset.direct_dual_residual_and_jacobian_seconds +=
            measure_average_seconds(kKernelTimingRepeats, [&] {
              std::ranges::copy(context.x, workspace.x_current.view().begin());
              if (auto dual_result = DirectDualJacobian::evaluate(
                      context,
                      what_prefix + " subset direct-dual residual/jacobian");
                  !dual_result) {
                throw std::runtime_error(dual_result.error().message);
              }
            });
      }
      for (Index j = 0; j < corpus.n; ++j) {
        for (Index i = 0; i < corpus.m; ++i) {
          if (direct_dual_stats != nullptr) {
            direct_dual_stats->add(workspace.J(i, j), expected_jacobian[i][j]);
          }
          if (direct_dual_vs_analytic_stats != nullptr) {
            direct_dual_vs_analytic_stats->add(
                workspace.J(i, j), analytic_jacobian[i + j * corpus.m]);
          }
          expect_close(workspace.J(i, j), expected_jacobian[i][j], 1e-12, 1e-10,
                       what_prefix + " direct-dual jacobian row " +
                           std::to_string(i) + " col " + std::to_string(j));
        }
      }
    }
  }

  timing.forward_difference_seconds +=
      measure_average_seconds(kKernelTimingRepeats, [&] {
        std::ranges::copy(context.x, workspace.x_current.view().begin());
        if (auto jacobian_result = evaluate_jacobian<ForwardDifferenceJacobian>(
                context, what_prefix + " fd jacobian");
            !jacobian_result) {
          throw std::runtime_error(jacobian_result.error().message);
        }
      });
  timing.forward_difference_residual_and_jacobian_seconds +=
      measure_average_seconds(kKernelTimingRepeats, [&] {
        std::ranges::copy(context.x, workspace.x_current.view().begin());
        if (auto residual_result =
                evaluate_residual(context, what_prefix + " fd residual");
            !residual_result) {
          throw std::runtime_error(residual_result.error().message);
        }
        if (auto jacobian_result = evaluate_jacobian<ForwardDifferenceJacobian>(
                context, what_prefix + " fd residual/jacobian");
            !jacobian_result) {
          throw std::runtime_error(jacobian_result.error().message);
        }
      });
  if (run_numerical_derivatives) {
    timing.numerical_subset.forward_difference_residual_and_jacobian_seconds +=
        measure_average_seconds(kKernelTimingRepeats, [&] {
          std::ranges::copy(context.x, workspace.x_current.view().begin());
          if (auto residual_result = evaluate_residual(
                  context, what_prefix + " subset fd residual");
              !residual_result) {
            throw std::runtime_error(residual_result.error().message);
          }
          if (auto jacobian_result =
                  evaluate_jacobian<ForwardDifferenceJacobian>(
                      context, what_prefix + " subset fd residual/jacobian");
              !jacobian_result) {
            throw std::runtime_error(jacobian_result.error().message);
          }
        });
  }
  if (run_numerical_derivatives) {
    for (Index j = 0; j < corpus.n; ++j) {
      for (Index i = 0; i < corpus.m; ++i) {
        if (forward_difference_stats != nullptr) {
          forward_difference_stats->add(workspace.J(i, j),
                                        expected_jacobian[i][j]);
        }
        expect_close(workspace.J(i, j), expected_jacobian[i][j], 1e-7, 2e-3,
                     what_prefix + " fd jacobian row " + std::to_string(i) +
                         " col " + std::to_string(j));
      }
    }
  }

  timing.central_difference_seconds +=
      measure_average_seconds(kKernelTimingRepeats, [&] {
        std::ranges::copy(context.x, workspace.x_current.view().begin());
        if (auto jacobian_result = evaluate_jacobian<CentralDifferenceJacobian>(
                context, what_prefix + " central jacobian");
            !jacobian_result) {
          throw std::runtime_error(jacobian_result.error().message);
        }
      });
  timing.central_difference_residual_and_jacobian_seconds +=
      measure_average_seconds(kKernelTimingRepeats, [&] {
        std::ranges::copy(context.x, workspace.x_current.view().begin());
        if (auto residual_result =
                evaluate_residual(context, what_prefix + " central residual");
            !residual_result) {
          throw std::runtime_error(residual_result.error().message);
        }
        if (auto jacobian_result = evaluate_jacobian<CentralDifferenceJacobian>(
                context, what_prefix + " central residual/jacobian");
            !jacobian_result) {
          throw std::runtime_error(jacobian_result.error().message);
        }
      });
  if (run_numerical_derivatives) {
    timing.numerical_subset.central_difference_residual_and_jacobian_seconds +=
        measure_average_seconds(kKernelTimingRepeats, [&] {
          std::ranges::copy(context.x, workspace.x_current.view().begin());
          if (auto residual_result = evaluate_residual(
                  context, what_prefix + " subset central residual");
              !residual_result) {
            throw std::runtime_error(residual_result.error().message);
          }
          if (auto jacobian_result =
                  evaluate_jacobian<CentralDifferenceJacobian>(
                      context,
                      what_prefix + " subset central residual/jacobian");
              !jacobian_result) {
            throw std::runtime_error(jacobian_result.error().message);
          }
        });
  }
  if (run_numerical_derivatives) {
    for (Index j = 0; j < corpus.n; ++j) {
      for (Index i = 0; i < corpus.m; ++i) {
        if (central_difference_stats != nullptr) {
          central_difference_stats->add(workspace.J(i, j),
                                        expected_jacobian[i][j]);
        }
        expect_close(workspace.J(i, j), expected_jacobian[i][j], 1e-8, 5e-4,
                     what_prefix + " central jacobian row " +
                         std::to_string(i) + " col " + std::to_string(j));
      }
    }
  }

  timing.total_seconds =
      timing.residual_seconds + timing.analytic_jacobian_seconds +
      timing.graph_jacobian_seconds + timing.forward_difference_seconds +
      timing.central_difference_seconds;
}

template <class Policy, Index M, Index N, class ResidualFn, class JacobianFn>
Result solve_nist_variant(const CorpusProblem &corpus, ResidualFn residual,
                          JacobianFn jacobian, ConstVectorView<N> beta,
                          const std::string &what,
                          bool require_convergence = true,
                          bool enable_relative_cost_termination = true,
                          std::vector<LmTrialTrace> *trial_trace = nullptr) {
  auto problem = [&]() {
    if constexpr (M == std::dynamic_extent && N == std::dynamic_extent) {
      return make_dynamic_problem(corpus.m, corpus.n, residual, jacobian);
    } else if constexpr (M == std::dynamic_extent) {
      return make_problem_dynamic_residuals<N>(corpus.m, residual, jacobian);
    } else if constexpr (N == std::dynamic_extent) {
      return make_problem_dynamic_parameters<M>(corpus.n, residual, jacobian);
    } else {
      return make_problem<M, N>(residual, jacobian);
    }
  }();

  Options options;
  constexpr double tolerance = std::numeric_limits<double>::epsilon();
  options.max_iterations = 10'000;
  options.max_function_evaluations = 100'000;
  options.gradient_tolerance = tolerance;
  options.step_tolerance = tolerance;
  // Ceres and MINPACK use relative cost tolerances; levmar's is absolute.
  options.cost_tolerance = 0.0;
  options.relative_cost_tolerance =
      enable_relative_cost_termination ? tolerance : 0.0;
  options.lm.min_lambda = 1e-16;
  options.lm.max_lambda = 1e32;

  SolverWorkspace<Policy, M, N> workspace;
  SolverContext<Policy, M, N, decltype(residual), decltype(jacobian)> context(
      problem, options, workspace, beta);
  context.trial_trace = trial_trace;
  auto solved = solve<Policy>(context);
  if (!solved) {
    throw std::runtime_error(what + ": " + solved.error().message);
  }
  if (require_convergence &&
      (solved->termination == TerminationReason::MaxIterations ||
       solved->termination == TerminationReason::MaxFunctionEvaluations ||
       solved->termination == TerminationReason::NumericalFailure ||
       solved->termination == TerminationReason::DampingLimit)) {
    throw std::runtime_error(what + ": solver did not converge (" +
                             solver_diagnostics(*solved) +
                             "): " + solved->message);
  }
  return std::move(*solved);
}

template <class Fn>
void dispatch_static_problem(const CorpusProblem &corpus, Fn &&fn) {
#define LEVMAR_STATIC_CASE(M, N)                                               \
  if (corpus.m == M && corpus.n == N) {                                        \
    fn.template operator()<M, N>();                                            \
    return;                                                                    \
  }
  LEVMAR_STATIC_CASE(154, 3)
  LEVMAR_STATIC_CASE(6, 2)
  LEVMAR_STATIC_CASE(214, 3)
  LEVMAR_STATIC_CASE(54, 3)
  LEVMAR_STATIC_CASE(35, 3)
  LEVMAR_STATIC_CASE(168, 9)
  LEVMAR_STATIC_CASE(250, 8)
  LEVMAR_STATIC_CASE(236, 7)
  LEVMAR_STATIC_CASE(151, 5)
  LEVMAR_STATIC_CASE(24, 6)
  LEVMAR_STATIC_CASE(11, 4)
  LEVMAR_STATIC_CASE(16, 3)
  LEVMAR_STATIC_CASE(33, 5)
  LEVMAR_STATIC_CASE(14, 2)
  LEVMAR_STATIC_CASE(128, 3)
  LEVMAR_STATIC_CASE(9, 3)
  LEVMAR_STATIC_CASE(15, 4)
  LEVMAR_STATIC_CASE(25, 4)
  LEVMAR_STATIC_CASE(37, 7)
#undef LEVMAR_STATIC_CASE
  throw std::runtime_error(
      "No fully static dispatch available for problem dimensions " +
      std::to_string(corpus.m) + "x" + std::to_string(corpus.n));
}

template <class Policy>
ProblemReport
run_problem(const std::filesystem::path &problem_dir,
            bool solve_all_starts = false, bool time_solvers = false,
            std::uint64_t solver_order_offset = 0,
            bool run_external_solvers = false, bool run_validation = true,
            bool enable_relative_cost_termination = true,
            std::vector<ControllerTraceRecord> *controller_traces = nullptr) {
  const auto corpus = load_problem(problem_dir);
  ProblemReport report;
  report.name = corpus.name;
  report.numerical_derivatives_skipped =
      !corpus.numerical_derivatives_recommended;

  std::vector<double> beta_storage(corpus.n, 0.0);
  const auto certified_residuals =
      read_numeric_csv(problem_dir / "residuals_certified.csv");
  double certified_cost = 0.0;
  for (const auto &row : certified_residuals) {
    certified_cost += 0.5 * row[0] * row[0];
  }
  const auto certified_parameters =
      std::ranges::find_if(corpus.params, [](const auto &entry) {
        return entry.first == "certified";
      });
  if (certified_parameters == corpus.params.end()) {
    throw std::runtime_error(corpus.name + " has no certified parameters");
  }

  for (const auto &[label, params] : corpus.params) {
    beta_storage = params;
    const auto expected_residuals =
        read_numeric_csv(problem_dir / ("residuals_" + label + ".csv"));
    const auto expected_jacobian =
        read_numeric_csv(problem_dir / ("jacobian_" + label + ".csv"));

    const bool run_numerical_derivatives =
        corpus.numerical_derivatives_recommended &&
        (label == "certified" || label == "benchmark");

    auto run_pair = [&]<Index SM, Index SN>(
                        auto dynamic_residual, auto dynamic_jacobian,
                        auto static_residual, auto static_jacobian) {
#ifndef LEVMAR_NIST_EXTERNAL_ONLY
      if (run_validation) {
        run_kernel_variant<std::dynamic_extent, std::dynamic_extent>(
            corpus, dynamic_residual, dynamic_jacobian,
            ConstVectorView<std::dynamic_extent>(beta_storage.data(),
                                                 beta_storage.size()),
            expected_residuals, expected_jacobian, run_numerical_derivatives,
            report.dynamic_timing, &report.residual_stats,
            &report.analytic_stats, &report.graph_autodiff_stats,
            &report.graph_autodiff_vs_analytic_stats, &report.direct_dual_stats,
            &report.direct_dual_vs_analytic_stats,
            &report.forward_difference_stats, &report.central_difference_stats,
            corpus.name + " dynamic " + label);

        run_kernel_variant<SM, SN>(
            corpus, static_residual, static_jacobian,
            ConstVectorView<SN>(beta_storage.data(), beta_storage.size()),
            expected_residuals, expected_jacobian, run_numerical_derivatives,
            report.static_timing, nullptr, nullptr,
            &report.graph_autodiff_stats,
            &report.graph_autodiff_vs_analytic_stats, &report.direct_dual_stats,
            &report.direct_dual_vs_analytic_stats, nullptr, nullptr,
            corpus.name + " static " + label);
      }
#endif

      const bool solve_start =
          label.starts_with("start") &&
          (corpus.difficulty == "lower" || solve_all_starts);
#ifdef LEVMAR_NIST_EXTERNAL_ONLY
      if (solve_start && run_external_solvers) {
        ++report.solver_start_cases;
        auto run_external = [&](std::string_view solver_name, auto &&fn) {
          try {
            auto result = fn();
            result.start_label = label;
            result.m = corpus.m;
            result.n = corpus.n;
            report.external_solver_results.push_back(std::move(result));
          } catch (const std::exception &error) {
            ExternalSolverResult failed;
            failed.solver = std::string(solver_name);
            failed.start_label = label;
            failed.m = corpus.m;
            failed.n = corpus.n;
            failed.lre = 0.0;
            failed.usable = false;
            report.external_solver_results.push_back(std::move(failed));
            report.solver_failures.push_back(corpus.name + " " + label + " " +
                                             std::string(solver_name) + ": " +
                                             error.what());
          }
        };
#ifdef LEVMAR_HAVE_CMINPACK
        run_external("minpack_lmder", [&] {
          return run_minpack_solver("minpack_lmder", corpus, dynamic_residual,
                                    dynamic_jacobian, beta_storage,
                                    certified_parameters->second, true);
        });
        run_external("minpack_lmdif", [&] {
          return run_minpack_solver("minpack_lmdif", corpus, dynamic_residual,
                                    dynamic_jacobian, beta_storage,
                                    certified_parameters->second, false);
        });
#endif
#ifdef LEVMAR_HAVE_CERES
        run_external("ceres_analytic", [&] {
          return run_ceres_analytic_solver(corpus, dynamic_residual,
                                           dynamic_jacobian, beta_storage,
                                           certified_parameters->second);
        });
        run_external("ceres_autodiff", [&] {
          return run_ceres_autodiff_solver<SM, SN>(
              static_residual, beta_storage, certified_parameters->second);
        });
#endif
      }
#else
      if (solve_start) {
        auto solve_pair = [&](bool require_convergence, double *dynamic_seconds,
                              double *static_seconds, bool static_first) {
          constexpr std::string_view derivative =
              std::same_as<typename Policy::JacobianPolicy, UserJacobian>
                  ? "analytic"
                  : "autodiff";
          auto solve_dynamic = [&] {
            const auto start = Clock::now();
            auto solution = solve_nist_variant<Policy, std::dynamic_extent,
                                               std::dynamic_extent>(
                corpus, dynamic_residual, dynamic_jacobian,
                ConstVectorView<std::dynamic_extent>(beta_storage.data(),
                                                     beta_storage.size()),
                corpus.name + " dynamic " + label + " solve",
                require_convergence, enable_relative_cost_termination,
                controller_trace_trials(controller_traces, corpus.name, label,
                                        derivative, "dynamic"));
            if (dynamic_seconds != nullptr) {
              *dynamic_seconds += elapsed_seconds(start, Clock::now());
            }
            return solution;
          };
          auto solve_static = [&] {
            const auto start = Clock::now();
            auto solution = solve_nist_variant<Policy, SM, SN>(
                corpus, static_residual, static_jacobian,
                ConstVectorView<SN>(beta_storage.data(), beta_storage.size()),
                corpus.name + " static " + label + " solve",
                require_convergence, enable_relative_cost_termination,
                controller_trace_trials(controller_traces, corpus.name, label,
                                        derivative, "static"));
            if (static_seconds != nullptr) {
              *static_seconds += elapsed_seconds(start, Clock::now());
            }
            return solution;
          };
          if (static_first) {
            auto static_solution = solve_static();
            auto dynamic_solution = solve_dynamic();
            return std::pair{std::move(dynamic_solution),
                             std::move(static_solution)};
          }
          auto dynamic_solution = solve_dynamic();
          auto static_solution = solve_static();
          return std::pair{std::move(dynamic_solution),
                           std::move(static_solution)};
        };
        auto verify_solution = [&] {
          const auto [dynamic_solution, static_solution] =
              solve_pair(true, nullptr, nullptr, false);
          for (Index j = 0; j < corpus.n; ++j) {
            expect_close(static_solution.parameters[j],
                         dynamic_solution.parameters[j], 1e-10, 1e-8,
                         corpus.name + " static/dynamic " + label +
                             " parameter " + std::to_string(j));
          }
          expect_close(dynamic_solution.final_cost, certified_cost, 1e-8, 1e-6,
                       corpus.name + " dynamic " + label + " cost (" +
                           solver_diagnostics(dynamic_solution) + ")");
          expect_close(static_solution.final_cost, certified_cost, 1e-8, 1e-6,
                       corpus.name + " static " + label + " cost (" +
                           solver_diagnostics(static_solution) + ")");
        };

        if (!solve_all_starts) {
          verify_solution();
        } else {
          ++report.solver_start_cases;

          // External solvers are independent competitors.  Run them outside
          // the levmar try block so a levmar failure cannot suppress a Ceres
          // or Minpack test case (and vice versa).
          if (run_external_solvers) {
            auto run_external = [&](std::string_view solver_name, auto &&fn) {
              try {
                auto result = fn();
                result.start_label = label;
                result.m = corpus.m;
                result.n = corpus.n;
                report.external_solver_results.push_back(std::move(result));
              } catch (const std::exception &error) {
                ExternalSolverResult failed;
                failed.solver = std::string(solver_name);
                failed.start_label = label;
                failed.m = corpus.m;
                failed.n = corpus.n;
                failed.lre = 0.0;
                failed.usable = false;
                report.external_solver_results.push_back(std::move(failed));
                report.solver_failures.push_back(
                    corpus.name + " " + label + " " + std::string(solver_name) +
                    ": " + error.what());
              }
            };
#ifdef LEVMAR_HAVE_CMINPACK
            run_external("minpack_lmder", [&] {
              return run_minpack_solver(
                  "minpack_lmder", corpus, dynamic_residual, dynamic_jacobian,
                  beta_storage, certified_parameters->second, true);
            });
            run_external("minpack_lmdif", [&] {
              return run_minpack_solver(
                  "minpack_lmdif", corpus, dynamic_residual, dynamic_jacobian,
                  beta_storage, certified_parameters->second, false);
            });
#endif
#ifdef LEVMAR_HAVE_CERES
            run_external("ceres_analytic", [&] {
              return run_ceres_analytic_solver(corpus, dynamic_residual,
                                               dynamic_jacobian, beta_storage,
                                               certified_parameters->second);
            });
            run_external("ceres_autodiff", [&] {
              return run_ceres_autodiff_solver<SM, SN>(
                  static_residual, beta_storage, certified_parameters->second);
            });
#endif
          }

          try {
            const bool static_first =
                (solver_order_offset + report.solver_start_cases) % 2 != 0;
            double dynamic_seconds = 0.0;
            double static_seconds = 0.0;
            const auto [dynamic_solution, static_solution] = solve_pair(
                false, &dynamic_seconds, &static_seconds, static_first);
            if (time_solvers) {
              report.solver_dynamic_seconds += dynamic_seconds;
              report.solver_static_seconds += static_seconds;
            }
            report.solver_dynamic_iterations += dynamic_solution.iterations;
            report.solver_static_iterations += static_solution.iterations;
            report.solver_dynamic_function_evaluations +=
                dynamic_solution.function_evaluations;
            report.solver_static_function_evaluations +=
                static_solution.function_evaluations;
            report.solver_dynamic_jacobian_evaluations +=
                dynamic_solution.jacobian_evaluations;
            report.solver_static_jacobian_evaluations +=
                static_solution.jacobian_evaluations;
            report.solver_dynamic_linear_solves +=
                dynamic_solution.linear_solves;
            report.solver_static_linear_solves += static_solution.linear_solves;
            report.solver_dynamic_accepted_steps +=
                dynamic_solution.accepted_steps;
            report.solver_static_accepted_steps +=
                static_solution.accepted_steps;
            report.solver_dynamic_rejected_steps +=
                dynamic_solution.rejected_steps;
            report.solver_static_rejected_steps +=
                static_solution.rejected_steps;
            const double dynamic_lre = log_relative_error(
                dynamic_solution.parameters, certified_parameters->second);
            const double static_lre = log_relative_error(
                static_solution.parameters, certified_parameters->second);
            report.solver_work_rows.push_back({label, "dynamic", corpus.m,
                                               corpus.n, dynamic_seconds,
                                               dynamic_lre, dynamic_solution});
            report.solver_work_rows.push_back({label, "static", corpus.m,
                                               corpus.n, static_seconds,
                                               static_lre, static_solution});
            report.solver_dynamic_lre_sum += dynamic_lre;
            report.solver_static_lre_sum += static_lre;
            if (dynamic_lre >= 4.0) {
              ++report.solver_start_case_passes;
            } else {
              std::ostringstream message;
              message << corpus.name << ' ' << label
                      << ": dynamic_lre=" << dynamic_lre
                      << ", static_lre=" << static_lre << ", "
                      << solver_diagnostics(dynamic_solution);
              report.solver_failures.push_back(message.str());
            }
          } catch (const std::exception &error) {
            report.solver_failures.push_back(corpus.name + " " + label + ": " +
                                             error.what());
          }
        }
      }
#endif
    };

    if (corpus.model_id == "bennett5") {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        using std::pow;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          r[i] = x[0] * pow(x[1] + xv, -1.0 / x[2]) - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double t = x[1] + xv;
          const double p = std::pow(t, -1.0 / x[2]);
          J[i, 0] = p;
          J[i, 1] = -x[0] * p / (x[2] * t);
          J[i, 2] = x[0] * p * std::log(t) / (x[2] * x[2]);
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<3, Scalar> x,
                            VectorView<154, Scalar> r) -> ErrorOrVoid {
        using std::pow;
        for (Index i = 0; i < 154; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          r[i] = x[0] * pow(x[1] + xv, -1.0 / x[2]) - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<3> x,
                                 MatrixView<154, 3> J) -> ErrorOrVoid {
        for (Index i = 0; i < 154; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double t = x[1] + xv;
          const double p = std::pow(t, -1.0 / x[2]);
          J[i, 0] = p;
          J[i, 1] = -x[0] * p / (x[2] * t);
          J[i, 2] = x[0] * p * std::log(t) / (x[2] * x[2]);
        }
        return {};
      };
      run_pair.template operator()<154, 3>(residual_dynamic, jacobian_dynamic,
                                           residual_static, jacobian_static);
    } else if (corpus.model_id == "monomolecular" && corpus.m == 6) {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        using std::exp;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto e = exp(-x[1] * xv);
          r[i] = x[0] * (1.0 - e) - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(-x[1] * xv);
          J[i, 0] = 1.0 - e;
          J[i, 1] = x[0] * xv * e;
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<2, Scalar> x,
                            VectorView<6, Scalar> r) -> ErrorOrVoid {
        using std::exp;
        for (Index i = 0; i < 6; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto e = exp(-x[1] * xv);
          r[i] = x[0] * (1.0 - e) - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<2> x,
                                 MatrixView<6, 2> J) -> ErrorOrVoid {
        for (Index i = 0; i < 6; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(-x[1] * xv);
          J[i, 0] = 1.0 - e;
          J[i, 1] = x[0] * xv * e;
        }
        return {};
      };
      run_pair.template operator()<6, 2>(residual_dynamic, jacobian_dynamic,
                                         residual_static, jacobian_static);
    } else if (corpus.model_id == "monomolecular" && corpus.m == 14) {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        using std::exp;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto e = exp(-x[1] * xv);
          r[i] = x[0] * (1.0 - e) - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(-x[1] * xv);
          J[i, 0] = 1.0 - e;
          J[i, 1] = x[0] * xv * e;
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<2, Scalar> x,
                            VectorView<14, Scalar> r) -> ErrorOrVoid {
        using std::exp;
        for (Index i = 0; i < 14; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto e = exp(-x[1] * xv);
          r[i] = x[0] * (1.0 - e) - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<2> x,
                                 MatrixView<14, 2> J) -> ErrorOrVoid {
        for (Index i = 0; i < 14; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(-x[1] * xv);
          J[i, 0] = 1.0 - e;
          J[i, 1] = x[0] * xv * e;
        }
        return {};
      };
      run_pair.template operator()<14, 2>(residual_dynamic, jacobian_dynamic,
                                          residual_static, jacobian_static);
    } else if (corpus.model_id == "chwirut" && corpus.m == 214) {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        using std::exp;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto e = exp(-x[0] * xv);
          const auto d = x[1] + x[2] * xv;
          r[i] = e / d - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(-x[0] * xv);
          const double d = x[1] + x[2] * xv;
          const double d2 = d * d;
          J[i, 0] = -xv * e / d;
          J[i, 1] = -e / d2;
          J[i, 2] = -xv * e / d2;
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<3, Scalar> x,
                            VectorView<214, Scalar> r) -> ErrorOrVoid {
        using std::exp;
        for (Index i = 0; i < 214; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto e = exp(-x[0] * xv);
          const auto d = x[1] + x[2] * xv;
          r[i] = e / d - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<3> x,
                                 MatrixView<214, 3> J) -> ErrorOrVoid {
        for (Index i = 0; i < 214; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(-x[0] * xv);
          const double d = x[1] + x[2] * xv;
          const double d2 = d * d;
          J[i, 0] = -xv * e / d;
          J[i, 1] = -e / d2;
          J[i, 2] = -xv * e / d2;
        }
        return {};
      };
      run_pair.template operator()<214, 3>(residual_dynamic, jacobian_dynamic,
                                           residual_static, jacobian_static);
    } else if (corpus.model_id == "chwirut" && corpus.m == 54) {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        using std::exp;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto e = exp(-x[0] * xv);
          const auto d = x[1] + x[2] * xv;
          r[i] = e / d - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(-x[0] * xv);
          const double d = x[1] + x[2] * xv;
          const double d2 = d * d;
          J[i, 0] = -xv * e / d;
          J[i, 1] = -e / d2;
          J[i, 2] = -xv * e / d2;
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<3, Scalar> x,
                            VectorView<54, Scalar> r) -> ErrorOrVoid {
        using std::exp;
        for (Index i = 0; i < 54; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto e = exp(-x[0] * xv);
          const auto d = x[1] + x[2] * xv;
          r[i] = e / d - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<3> x,
                                 MatrixView<54, 3> J) -> ErrorOrVoid {
        for (Index i = 0; i < 54; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(-x[0] * xv);
          const double d = x[1] + x[2] * xv;
          const double d2 = d * d;
          J[i, 0] = -xv * e / d;
          J[i, 1] = -e / d2;
          J[i, 2] = -xv * e / d2;
        }
        return {};
      };
      run_pair.template operator()<54, 3>(residual_dynamic, jacobian_dynamic,
                                          residual_static, jacobian_static);
    } else if (corpus.model_id == "triple_exponential") {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        using std::exp;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          r[i] = x[0] * 0.0;
          for (Index j = 0; j < 6; j += 2) {
            r[i] = r[i] + x[j] * exp(-x[j + 1] * xv);
          }
          r[i] = r[i] - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          for (Index j = 0; j < 6; j += 2) {
            const double e = std::exp(-x[j + 1] * xv);
            J[i, j] = e;
            J[i, j + 1] = -x[j] * xv * e;
          }
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<6, Scalar> x,
                            VectorView<24, Scalar> r) -> ErrorOrVoid {
        using std::exp;
        for (Index i = 0; i < 24; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          r[i] = x[0] * 0.0;
          for (Index j = 0; j < 6; j += 2) {
            r[i] = r[i] + x[j] * exp(-x[j + 1] * xv);
          }
          r[i] = r[i] - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<6> x,
                                 MatrixView<24, 6> J) -> ErrorOrVoid {
        for (Index i = 0; i < 24; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          for (Index j = 0; j < 6; j += 2) {
            const double e = std::exp(-x[j + 1] * xv);
            J[i, j] = e;
            J[i, j + 1] = -x[j] * xv * e;
          }
        }
        return {};
      };
      run_pair.template operator()<24, 6>(residual_dynamic, jacobian_dynamic,
                                          residual_static, jacobian_static);
    } else if (corpus.model_id == "gauss_mixture") {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        using std::exp;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto d1 = xv - x[3];
          const auto d2 = xv - x[6];
          r[i] = x[0] * exp(-x[1] * xv) +
                 x[2] * exp(-(d1 * d1) / (x[4] * x[4])) +
                 x[5] * exp(-(d2 * d2) / (x[7] * x[7])) - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e1 = std::exp(-x[1] * xv);
          const double d1 = xv - x[3];
          const double d2 = xv - x[6];
          const double s1 = x[4] * x[4];
          const double s2 = x[7] * x[7];
          const double e2 = std::exp(-(d1 * d1) / s1);
          const double e3 = std::exp(-(d2 * d2) / s2);
          J[i, 0] = e1;
          J[i, 1] = -x[0] * xv * e1;
          J[i, 2] = e2;
          J[i, 3] = x[2] * e2 * 2.0 * d1 / s1;
          J[i, 4] = x[2] * e2 * 2.0 * d1 * d1 / (x[4] * x[4] * x[4]);
          J[i, 5] = e3;
          J[i, 6] = x[5] * e3 * 2.0 * d2 / s2;
          J[i, 7] = x[5] * e3 * 2.0 * d2 * d2 / (x[7] * x[7] * x[7]);
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<8, Scalar> x,
                            VectorView<250, Scalar> r) -> ErrorOrVoid {
        using std::exp;
        for (Index i = 0; i < 250; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto d1 = xv - x[3];
          const auto d2 = xv - x[6];
          r[i] = x[0] * exp(-x[1] * xv) +
                 x[2] * exp(-(d1 * d1) / (x[4] * x[4])) +
                 x[5] * exp(-(d2 * d2) / (x[7] * x[7])) - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<8> x,
                                 MatrixView<250, 8> J) -> ErrorOrVoid {
        for (Index i = 0; i < 250; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e1 = std::exp(-x[1] * xv);
          const double d1 = xv - x[3];
          const double d2 = xv - x[6];
          const double s1 = x[4] * x[4];
          const double s2 = x[7] * x[7];
          const double e2 = std::exp(-(d1 * d1) / s1);
          const double e3 = std::exp(-(d2 * d2) / s2);
          J[i, 0] = e1;
          J[i, 1] = -x[0] * xv * e1;
          J[i, 2] = e2;
          J[i, 3] = x[2] * e2 * 2.0 * d1 / s1;
          J[i, 4] = x[2] * e2 * 2.0 * d1 * d1 / (x[4] * x[4] * x[4]);
          J[i, 5] = e3;
          J[i, 6] = x[5] * e3 * 2.0 * d2 / s2;
          J[i, 7] = x[5] * e3 * 2.0 * d2 * d2 / (x[7] * x[7] * x[7]);
        }
        return {};
      };
      run_pair.template operator()<250, 8>(residual_dynamic, jacobian_dynamic,
                                           residual_static, jacobian_static);
    } else if (corpus.model_id == "danwood") {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        using std::pow;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          r[i] = x[0] * pow(row[0], x[1]) - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double x_pow = std::pow(xv, x[1]);
          J[i, 0] = x_pow;
          J[i, 1] = x[0] * x_pow * std::log(xv);
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<2, Scalar> x,
                            VectorView<6, Scalar> r) -> ErrorOrVoid {
        using std::pow;
        for (Index i = 0; i < 6; ++i) {
          const auto &row = corpus.data[i];
          r[i] = x[0] * pow(row[0], x[1]) - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<2> x,
                                 MatrixView<6, 2> J) -> ErrorOrVoid {
        for (Index i = 0; i < 6; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double x_pow = std::pow(xv, x[1]);
          J[i, 0] = x_pow;
          J[i, 1] = x[0] * x_pow * std::log(xv);
        }
        return {};
      };
      run_pair.template operator()<6, 2>(residual_dynamic, jacobian_dynamic,
                                         residual_static, jacobian_static);
    } else if (corpus.model_id == "misra1b") {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        using std::pow;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto t = 1.0 + x[1] * xv / 2.0;
          r[i] = x[0] * (1.0 - pow(t, -2.0)) - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double t = 1.0 + x[1] * xv / 2.0;
          J[i, 0] = 1.0 - std::pow(t, -2.0);
          J[i, 1] = x[0] * xv * std::pow(t, -3.0);
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<2, Scalar> x,
                            VectorView<14, Scalar> r) -> ErrorOrVoid {
        using std::pow;
        for (Index i = 0; i < 14; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto t = 1.0 + x[1] * xv / 2.0;
          r[i] = x[0] * (1.0 - pow(t, -2.0)) - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<2> x,
                                 MatrixView<14, 2> J) -> ErrorOrVoid {
        for (Index i = 0; i < 14; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double t = 1.0 + x[1] * xv / 2.0;
          J[i, 0] = 1.0 - std::pow(t, -2.0);
          J[i, 1] = x[0] * xv * std::pow(t, -3.0);
        }
        return {};
      };
      run_pair.template operator()<14, 2>(residual_dynamic, jacobian_dynamic,
                                          residual_static, jacobian_static);
    } else if (corpus.model_id == "rational_quadratic") {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double x2 = xv * xv;
          const auto n = x[0] + x[1] * xv + x[2] * x2;
          const auto d = 1.0 + x[3] * xv + x[4] * x2;
          r[i] = n / d - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double x2 = xv * xv;
          const double n = x[0] + x[1] * xv + x[2] * x2;
          const double d = 1.0 + x[3] * xv + x[4] * x2;
          const double d2 = d * d;
          J[i, 0] = 1.0 / d;
          J[i, 1] = xv / d;
          J[i, 2] = x2 / d;
          J[i, 3] = -n * xv / d2;
          J[i, 4] = -n * x2 / d2;
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<5, Scalar> x,
                            VectorView<151, Scalar> r) -> ErrorOrVoid {
        for (Index i = 0; i < 151; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double x2 = xv * xv;
          const auto n = x[0] + x[1] * xv + x[2] * x2;
          const auto d = 1.0 + x[3] * xv + x[4] * x2;
          r[i] = n / d - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<5> x,
                                 MatrixView<151, 5> J) -> ErrorOrVoid {
        for (Index i = 0; i < 151; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double x2 = xv * xv;
          const double n = x[0] + x[1] * xv + x[2] * x2;
          const double d = 1.0 + x[3] * xv + x[4] * x2;
          const double d2 = d * d;
          J[i, 0] = 1.0 / d;
          J[i, 1] = xv / d;
          J[i, 2] = x2 / d;
          J[i, 3] = -n * xv / d2;
          J[i, 4] = -n * x2 / d2;
        }
        return {};
      };
      run_pair.template operator()<151, 5>(residual_dynamic, jacobian_dynamic,
                                           residual_static, jacobian_static);
    } else if (corpus.model_id == "rational_cubic" && corpus.m == 236) {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double x2 = xv * xv;
          const double x3 = x2 * xv;
          const auto n = x[0] + x[1] * xv + x[2] * x2 + x[3] * x3;
          const auto d = 1.0 + x[4] * xv + x[5] * x2 + x[6] * x3;
          r[i] = n / d - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double x2 = xv * xv;
          const double x3 = x2 * xv;
          const double n = x[0] + x[1] * xv + x[2] * x2 + x[3] * x3;
          const double d = 1.0 + x[4] * xv + x[5] * x2 + x[6] * x3;
          const double d2 = d * d;
          J[i, 0] = 1.0 / d;
          J[i, 1] = xv / d;
          J[i, 2] = x2 / d;
          J[i, 3] = x3 / d;
          J[i, 4] = -n * xv / d2;
          J[i, 5] = -n * x2 / d2;
          J[i, 6] = -n * x3 / d2;
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<7, Scalar> x,
                            VectorView<236, Scalar> r) -> ErrorOrVoid {
        for (Index i = 0; i < 236; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double x2 = xv * xv;
          const double x3 = x2 * xv;
          const auto n = x[0] + x[1] * xv + x[2] * x2 + x[3] * x3;
          const auto d = 1.0 + x[4] * xv + x[5] * x2 + x[6] * x3;
          r[i] = n / d - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<7> x,
                                 MatrixView<236, 7> J) -> ErrorOrVoid {
        for (Index i = 0; i < 236; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double x2 = xv * xv;
          const double x3 = x2 * xv;
          const double n = x[0] + x[1] * xv + x[2] * x2 + x[3] * x3;
          const double d = 1.0 + x[4] * xv + x[5] * x2 + x[6] * x3;
          const double d2 = d * d;
          J[i, 0] = 1.0 / d;
          J[i, 1] = xv / d;
          J[i, 2] = x2 / d;
          J[i, 3] = x3 / d;
          J[i, 4] = -n * xv / d2;
          J[i, 5] = -n * x2 / d2;
          J[i, 6] = -n * x3 / d2;
        }
        return {};
      };
      run_pair.template operator()<236, 7>(residual_dynamic, jacobian_dynamic,
                                           residual_static, jacobian_static);
    } else if (corpus.model_id == "rational_cubic" && corpus.m == 37) {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double x2 = xv * xv;
          const double x3 = x2 * xv;
          const auto n = x[0] + x[1] * xv + x[2] * x2 + x[3] * x3;
          const auto d = 1.0 + x[4] * xv + x[5] * x2 + x[6] * x3;
          r[i] = n / d - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double x2 = xv * xv;
          const double x3 = x2 * xv;
          const double n = x[0] + x[1] * xv + x[2] * x2 + x[3] * x3;
          const double d = 1.0 + x[4] * xv + x[5] * x2 + x[6] * x3;
          const double d2 = d * d;
          J[i, 0] = 1.0 / d;
          J[i, 1] = xv / d;
          J[i, 2] = x2 / d;
          J[i, 3] = x3 / d;
          J[i, 4] = -n * xv / d2;
          J[i, 5] = -n * x2 / d2;
          J[i, 6] = -n * x3 / d2;
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<7, Scalar> x,
                            VectorView<37, Scalar> r) -> ErrorOrVoid {
        for (Index i = 0; i < 37; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double x2 = xv * xv;
          const double x3 = x2 * xv;
          const auto n = x[0] + x[1] * xv + x[2] * x2 + x[3] * x3;
          const auto d = 1.0 + x[4] * xv + x[5] * x2 + x[6] * x3;
          r[i] = n / d - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<7> x,
                                 MatrixView<37, 7> J) -> ErrorOrVoid {
        for (Index i = 0; i < 37; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double x2 = xv * xv;
          const double x3 = x2 * xv;
          const double n = x[0] + x[1] * xv + x[2] * x2 + x[3] * x3;
          const double d = 1.0 + x[4] * xv + x[5] * x2 + x[6] * x3;
          const double d2 = d * d;
          J[i, 0] = 1.0 / d;
          J[i, 1] = xv / d;
          J[i, 2] = x2 / d;
          J[i, 3] = x3 / d;
          J[i, 4] = -n * xv / d2;
          J[i, 5] = -n * x2 / d2;
          J[i, 6] = -n * x3 / d2;
        }
        return {};
      };
      run_pair.template operator()<37, 7>(residual_dynamic, jacobian_dynamic,
                                          residual_static, jacobian_static);
    } else if (corpus.model_id == "nelson_log") {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        using std::exp;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double x1 = row[0];
          const double x2 = row[1];
          r[i] = x[0] - x[1] * x1 * exp(-x[2] * x2) - std::log(row.back());
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double x1 = row[0];
          const double x2 = row[1];
          const double e = std::exp(-x[2] * x2);
          J[i, 0] = 1.0;
          J[i, 1] = -x1 * e;
          J[i, 2] = x[1] * x1 * x2 * e;
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<3, Scalar> x,
                            VectorView<128, Scalar> r) -> ErrorOrVoid {
        using std::exp;
        for (Index i = 0; i < 128; ++i) {
          const auto &row = corpus.data[i];
          const double x1 = row[0];
          const double x2 = row[1];
          r[i] = x[0] - x[1] * x1 * exp(-x[2] * x2) - std::log(row.back());
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<3> x,
                                 MatrixView<128, 3> J) -> ErrorOrVoid {
        for (Index i = 0; i < 128; ++i) {
          const auto &row = corpus.data[i];
          const double x1 = row[0];
          const double x2 = row[1];
          const double e = std::exp(-x[2] * x2);
          J[i, 0] = 1.0;
          J[i, 1] = -x1 * e;
          J[i, 2] = x[1] * x1 * x2 * e;
        }
        return {};
      };
      run_pair.template operator()<128, 3>(residual_dynamic, jacobian_dynamic,
                                           residual_static, jacobian_static);
    } else if (corpus.model_id == "mgh17") {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        using std::exp;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          r[i] = x[0] + x[1] * exp(-x[3] * xv) + x[2] * exp(-x[4] * xv) -
                 row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e1 = std::exp(-x[3] * xv);
          const double e2 = std::exp(-x[4] * xv);
          J[i, 0] = 1.0;
          J[i, 1] = e1;
          J[i, 2] = e2;
          J[i, 3] = -x[1] * xv * e1;
          J[i, 4] = -x[2] * xv * e2;
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<5, Scalar> x,
                            VectorView<33, Scalar> r) -> ErrorOrVoid {
        using std::exp;
        for (Index i = 0; i < 33; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          r[i] = x[0] + x[1] * exp(-x[3] * xv) + x[2] * exp(-x[4] * xv) -
                 row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<5> x,
                                 MatrixView<33, 5> J) -> ErrorOrVoid {
        for (Index i = 0; i < 33; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e1 = std::exp(-x[3] * xv);
          const double e2 = std::exp(-x[4] * xv);
          J[i, 0] = 1.0;
          J[i, 1] = e1;
          J[i, 2] = e2;
          J[i, 3] = -x[1] * xv * e1;
          J[i, 4] = -x[2] * xv * e2;
        }
        return {};
      };
      run_pair.template operator()<33, 5>(residual_dynamic, jacobian_dynamic,
                                          residual_static, jacobian_static);
    } else if (corpus.model_id == "misra1c") {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        using std::sqrt;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto t = 1.0 + 2.0 * x[1] * xv;
          r[i] = x[0] * (1.0 - 1.0 / sqrt(t)) - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double t = 1.0 + 2.0 * x[1] * xv;
          J[i, 0] = 1.0 - 1.0 / std::sqrt(t);
          J[i, 1] = x[0] * xv * std::pow(t, -1.5);
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<2, Scalar> x,
                            VectorView<14, Scalar> r) -> ErrorOrVoid {
        using std::sqrt;
        for (Index i = 0; i < 14; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto t = 1.0 + 2.0 * x[1] * xv;
          r[i] = x[0] * (1.0 - 1.0 / sqrt(t)) - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<2> x,
                                 MatrixView<14, 2> J) -> ErrorOrVoid {
        for (Index i = 0; i < 14; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double t = 1.0 + 2.0 * x[1] * xv;
          J[i, 0] = 1.0 - 1.0 / std::sqrt(t);
          J[i, 1] = x[0] * xv * std::pow(t, -1.5);
        }
        return {};
      };
      run_pair.template operator()<14, 2>(residual_dynamic, jacobian_dynamic,
                                          residual_static, jacobian_static);
    } else if (corpus.model_id == "misra1d") {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto d = 1.0 + x[1] * xv;
          r[i] = (x[0] * x[1] * xv) / d - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double d = 1.0 + x[1] * xv;
          const double d2 = d * d;
          J[i, 0] = x[1] * xv / d;
          J[i, 1] = x[0] * xv / d2;
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<2, Scalar> x,
                            VectorView<14, Scalar> r) -> ErrorOrVoid {
        for (Index i = 0; i < 14; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto d = 1.0 + x[1] * xv;
          r[i] = (x[0] * x[1] * xv) / d - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<2> x,
                                 MatrixView<14, 2> J) -> ErrorOrVoid {
        for (Index i = 0; i < 14; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double d = 1.0 + x[1] * xv;
          const double d2 = d * d;
          J[i, 0] = x[1] * xv / d;
          J[i, 1] = x[0] * xv / d2;
        }
        return {};
      };
      run_pair.template operator()<14, 2>(residual_dynamic, jacobian_dynamic,
                                          residual_static, jacobian_static);
    } else if (corpus.model_id == "roszman1") {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        using std::atan2;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          r[i] =
              x[0] - x[1] * xv -
              atan2(x[2] / (xv - x[3]), x[0] * 0.0 + 1.0) / std::numbers::pi -
              row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double u = x[2] / (xv - x[3]);
          const double common = -1.0 / (std::numbers::pi * (1.0 + u * u));
          J[i, 0] = 1.0;
          J[i, 1] = -xv;
          J[i, 2] = common / (xv - x[3]);
          J[i, 3] = common * x[2] / ((xv - x[3]) * (xv - x[3]));
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<4, Scalar> x,
                            VectorView<25, Scalar> r) -> ErrorOrVoid {
        using std::atan2;
        for (Index i = 0; i < 25; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          r[i] =
              x[0] - x[1] * xv -
              atan2(x[2] / (xv - x[3]), x[0] * 0.0 + 1.0) / std::numbers::pi -
              row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<4> x,
                                 MatrixView<25, 4> J) -> ErrorOrVoid {
        for (Index i = 0; i < 25; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double u = x[2] / (xv - x[3]);
          const double common = -1.0 / (std::numbers::pi * (1.0 + u * u));
          J[i, 0] = 1.0;
          J[i, 1] = -xv;
          J[i, 2] = common / (xv - x[3]);
          J[i, 3] = common * x[2] / ((xv - x[3]) * (xv - x[3]));
        }
        return {};
      };
      run_pair.template operator()<25, 4>(residual_dynamic, jacobian_dynamic,
                                          residual_static, jacobian_static);
    } else if (corpus.model_id == "enso") {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        using std::cos;
        using std::sin;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double annual = 2.0 * std::numbers::pi * xv / 12.0;
          const auto p4 = 2.0 * std::numbers::pi * xv / x[3];
          const auto p7 = 2.0 * std::numbers::pi * xv / x[6];
          const auto value = x[0] + x[1] * cos(annual) + x[2] * sin(annual) +
                             x[4] * cos(p4) + x[5] * sin(p4) + x[7] * cos(p7) +
                             x[8] * sin(p7);
          r[i] = value - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double annual = 2.0 * std::numbers::pi * xv / 12.0;
          const double p4 = 2.0 * std::numbers::pi * xv / x[3];
          const double p7 = 2.0 * std::numbers::pi * xv / x[6];
          const double d4 = 2.0 * std::numbers::pi * xv / (x[3] * x[3]);
          const double d7 = 2.0 * std::numbers::pi * xv / (x[6] * x[6]);
          J[i, 0] = 1.0;
          J[i, 1] = std::cos(annual);
          J[i, 2] = std::sin(annual);
          J[i, 3] = d4 * (x[4] * std::sin(p4) - x[5] * std::cos(p4));
          J[i, 4] = std::cos(p4);
          J[i, 5] = std::sin(p4);
          J[i, 6] = d7 * (x[7] * std::sin(p7) - x[8] * std::cos(p7));
          J[i, 7] = std::cos(p7);
          J[i, 8] = std::sin(p7);
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<9, Scalar> x,
                            VectorView<168, Scalar> r) -> ErrorOrVoid {
        using std::cos;
        using std::sin;
        for (Index i = 0; i < 168; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double annual = 2.0 * std::numbers::pi * xv / 12.0;
          const auto p4 = 2.0 * std::numbers::pi * xv / x[3];
          const auto p7 = 2.0 * std::numbers::pi * xv / x[6];
          const auto value = x[0] + x[1] * cos(annual) + x[2] * sin(annual) +
                             x[4] * cos(p4) + x[5] * sin(p4) + x[7] * cos(p7) +
                             x[8] * sin(p7);
          r[i] = value - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<9> x,
                                 MatrixView<168, 9> J) -> ErrorOrVoid {
        for (Index i = 0; i < 168; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double annual = 2.0 * std::numbers::pi * xv / 12.0;
          const double p4 = 2.0 * std::numbers::pi * xv / x[3];
          const double p7 = 2.0 * std::numbers::pi * xv / x[6];
          const double d4 = 2.0 * std::numbers::pi * xv / (x[3] * x[3]);
          const double d7 = 2.0 * std::numbers::pi * xv / (x[6] * x[6]);
          J[i, 0] = 1.0;
          J[i, 1] = std::cos(annual);
          J[i, 2] = std::sin(annual);
          J[i, 3] = d4 * (x[4] * std::sin(p4) - x[5] * std::cos(p4));
          J[i, 4] = std::cos(p4);
          J[i, 5] = std::sin(p4);
          J[i, 6] = d7 * (x[7] * std::sin(p7) - x[8] * std::cos(p7));
          J[i, 7] = std::cos(p7);
          J[i, 8] = std::sin(p7);
        }
        return {};
      };
      run_pair.template operator()<168, 9>(residual_dynamic, jacobian_dynamic,
                                           residual_static, jacobian_static);
    } else if (corpus.model_id == "mgh09") {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto num = xv * xv + x[1] * xv;
          const auto den = xv * xv + x[2] * xv + x[3];
          r[i] = x[0] * num / den - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double num = xv * xv + x[1] * xv;
          const double den = xv * xv + x[2] * xv + x[3];
          const double den2 = den * den;
          J[i, 0] = num / den;
          J[i, 1] = x[0] * xv / den;
          J[i, 2] = -x[0] * num * xv / den2;
          J[i, 3] = -x[0] * num / den2;
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<4, Scalar> x,
                            VectorView<11, Scalar> r) -> ErrorOrVoid {
        for (Index i = 0; i < 11; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto num = xv * xv + x[1] * xv;
          const auto den = xv * xv + x[2] * xv + x[3];
          r[i] = x[0] * num / den - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<4> x,
                                 MatrixView<11, 4> J) -> ErrorOrVoid {
        for (Index i = 0; i < 11; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double num = xv * xv + x[1] * xv;
          const double den = xv * xv + x[2] * xv + x[3];
          const double den2 = den * den;
          J[i, 0] = num / den;
          J[i, 1] = x[0] * xv / den;
          J[i, 2] = -x[0] * num * xv / den2;
          J[i, 3] = -x[0] * num / den2;
        }
        return {};
      };
      run_pair.template operator()<11, 4>(residual_dynamic, jacobian_dynamic,
                                          residual_static, jacobian_static);
    } else if (corpus.model_id == "rat42") {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        using std::exp;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto e = exp(x[1] - x[2] * xv);
          r[i] = x[0] / (1.0 + e) - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(x[1] - x[2] * xv);
          const double d = 1.0 + e;
          const double d2 = d * d;
          J[i, 0] = 1.0 / d;
          J[i, 1] = -x[0] * e / d2;
          J[i, 2] = x[0] * xv * e / d2;
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<3, Scalar> x,
                            VectorView<9, Scalar> r) -> ErrorOrVoid {
        using std::exp;
        for (Index i = 0; i < 9; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto e = exp(x[1] - x[2] * xv);
          r[i] = x[0] / (1.0 + e) - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<3> x,
                                 MatrixView<9, 3> J) -> ErrorOrVoid {
        for (Index i = 0; i < 9; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(x[1] - x[2] * xv);
          const double d = 1.0 + e;
          const double d2 = d * d;
          J[i, 0] = 1.0 / d;
          J[i, 1] = -x[0] * e / d2;
          J[i, 2] = x[0] * xv * e / d2;
        }
        return {};
      };
      run_pair.template operator()<9, 3>(residual_dynamic, jacobian_dynamic,
                                         residual_static, jacobian_static);
    } else if (corpus.model_id == "mgh10") {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        using std::exp;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto e = exp(x[1] / (xv + x[2]));
          r[i] = x[0] * e - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double q = xv + x[2];
          const double e = std::exp(x[1] / q);
          J[i, 0] = e;
          J[i, 1] = x[0] * e / q;
          J[i, 2] = -x[0] * e * x[1] / (q * q);
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<3, Scalar> x,
                            VectorView<16, Scalar> r) -> ErrorOrVoid {
        using std::exp;
        for (Index i = 0; i < 16; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto e = exp(x[1] / (xv + x[2]));
          r[i] = x[0] * e - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<3> x,
                                 MatrixView<16, 3> J) -> ErrorOrVoid {
        for (Index i = 0; i < 16; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double q = xv + x[2];
          const double e = std::exp(x[1] / q);
          J[i, 0] = e;
          J[i, 1] = x[0] * e / q;
          J[i, 2] = -x[0] * e * x[1] / (q * q);
        }
        return {};
      };
      run_pair.template operator()<16, 3>(residual_dynamic, jacobian_dynamic,
                                          residual_static, jacobian_static);
    } else if (corpus.model_id == "eckerle4") {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        using std::exp;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto delta = xv - x[2];
          const auto e = exp(-(delta * delta) / (2.0 * x[1] * x[1]));
          r[i] = (x[0] / x[1]) * e - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double b1 = x[0];
          const double b2 = x[1];
          const double b3 = x[2];
          const double delta = xv - b3;
          const double e = std::exp(-(delta * delta) / (2.0 * b2 * b2));
          J[i, 0] = e / b2;
          J[i, 1] =
              b1 * e * (delta * delta / (b2 * b2 * b2 * b2) - 1.0 / (b2 * b2));
          J[i, 2] = b1 * e * delta / (b2 * b2 * b2);
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<3, Scalar> x,
                            VectorView<35, Scalar> r) -> ErrorOrVoid {
        using std::exp;
        for (Index i = 0; i < 35; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto delta = xv - x[2];
          const auto e = exp(-(delta * delta) / (2.0 * x[1] * x[1]));
          r[i] = (x[0] / x[1]) * e - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<3> x,
                                 MatrixView<35, 3> J) -> ErrorOrVoid {
        for (Index i = 0; i < 35; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double b1 = x[0];
          const double b2 = x[1];
          const double b3 = x[2];
          const double delta = xv - b3;
          const double e = std::exp(-(delta * delta) / (2.0 * b2 * b2));
          J[i, 0] = e / b2;
          J[i, 1] =
              b1 * e * (delta * delta / (b2 * b2 * b2 * b2) - 1.0 / (b2 * b2));
          J[i, 2] = b1 * e * delta / (b2 * b2 * b2);
        }
        return {};
      };
      run_pair.template operator()<35, 3>(residual_dynamic, jacobian_dynamic,
                                          residual_static, jacobian_static);
    } else if (corpus.model_id == "rat43") {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        using std::exp;
        using std::pow;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto e = exp(x[1] - x[2] * xv);
          const auto t = 1.0 + e;
          const auto p = pow(t, -1.0 / x[3]);
          r[i] = x[0] * p - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double b1 = x[0];
          const double b2 = x[1];
          const double b3 = x[2];
          const double b4 = x[3];
          const double e = std::exp(b2 - b3 * xv);
          const double t = 1.0 + e;
          const double p = std::pow(t, -1.0 / b4);
          const double yhat = b1 * p;
          J[i, 0] = p;
          J[i, 1] = -yhat * e / (b4 * t);
          J[i, 2] = yhat * xv * e / (b4 * t);
          J[i, 3] = yhat * std::log(t) / (b4 * b4);
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<4, Scalar> x,
                            VectorView<15, Scalar> r) -> ErrorOrVoid {
        using std::exp;
        using std::pow;
        for (Index i = 0; i < 15; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const auto e = exp(x[1] - x[2] * xv);
          const auto t = 1.0 + e;
          const auto p = pow(t, -1.0 / x[3]);
          r[i] = x[0] * p - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<4> x,
                                 MatrixView<15, 4> J) -> ErrorOrVoid {
        for (Index i = 0; i < 15; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double b1 = x[0];
          const double b2 = x[1];
          const double b3 = x[2];
          const double b4 = x[3];
          const double e = std::exp(b2 - b3 * xv);
          const double t = 1.0 + e;
          const double p = std::pow(t, -1.0 / b4);
          const double yhat = b1 * p;
          J[i, 0] = p;
          J[i, 1] = -yhat * e / (b4 * t);
          J[i, 2] = yhat * xv * e / (b4 * t);
          J[i, 3] = yhat * std::log(t) / (b4 * b4);
        }
        return {};
      };
      run_pair.template operator()<15, 4>(residual_dynamic, jacobian_dynamic,
                                          residual_static, jacobian_static);
    } else if (corpus.model_id == "linear_dense" && corpus.m == 1000 &&
               corpus.n == 4) {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          auto value = x[0] * 0.0;
          for (Index j = 0; j < 4; ++j) {
            value = value + x[j] * row[j];
          }
          r[i] = value - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent>,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          for (Index j = 0; j < 4; ++j) {
            J[i, j] = row[j];
          }
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<4, Scalar> x,
                            VectorView<1000, Scalar> r) -> ErrorOrVoid {
        for (Index i = 0; i < 1000; ++i) {
          const auto &row = corpus.data[i];
          auto value = x[0] * 0.0;
          for (Index j = 0; j < 4; ++j) {
            value = value + x[j] * row[j];
          }
          r[i] = value - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<4>,
                                 MatrixView<1000, 4> J) -> ErrorOrVoid {
        for (Index i = 0; i < 1000; ++i) {
          const auto &row = corpus.data[i];
          for (Index j = 0; j < 4; ++j) {
            J[i, j] = row[j];
          }
        }
        return {};
      };
      run_pair.template operator()<1000, 4>(residual_dynamic, jacobian_dynamic,
                                            residual_static, jacobian_static);
    } else if (corpus.model_id == "linear_dense" && corpus.m == 10000 &&
               corpus.n == 4) {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          auto value = x[0] * 0.0;
          for (Index j = 0; j < 4; ++j) {
            value = value + x[j] * row[j];
          }
          r[i] = value - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent>,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          for (Index j = 0; j < 4; ++j) {
            J[i, j] = row[j];
          }
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<4, Scalar> x,
                            VectorView<10000, Scalar> r) -> ErrorOrVoid {
        for (Index i = 0; i < 10000; ++i) {
          const auto &row = corpus.data[i];
          auto value = x[0] * 0.0;
          for (Index j = 0; j < 4; ++j) {
            value = value + x[j] * row[j];
          }
          r[i] = value - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<4>,
                                 MatrixView<10000, 4> J) -> ErrorOrVoid {
        for (Index i = 0; i < 10000; ++i) {
          const auto &row = corpus.data[i];
          for (Index j = 0; j < 4; ++j) {
            J[i, j] = row[j];
          }
        }
        return {};
      };
      run_pair.template operator()<10000, 4>(residual_dynamic, jacobian_dynamic,
                                             residual_static, jacobian_static);
    } else if (corpus.model_id == "rational_dense" && corpus.m == 32 &&
               corpus.n == 32) {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        constexpr Index Half = 16;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          double powers[17];
          powers[0] = 1.0;
          for (Index j = 1; j <= Half; ++j)
            powers[j] = powers[j - 1] * xv;
          auto num = x[0] * 0.0;
          auto den = x[0] * 0.0 + 1.0;
          for (Index j = 0; j < Half; ++j) {
            num = num + x[j] * powers[j];
            den = den + x[Half + j] * powers[j + 1];
          }
          r[i] = num / den - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        constexpr Index Half = 16;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          double powers[17];
          powers[0] = 1.0;
          for (Index j = 1; j <= Half; ++j)
            powers[j] = powers[j - 1] * xv;
          double num = 0.0;
          double den = 1.0;
          for (Index j = 0; j < Half; ++j) {
            num = num + x[j] * powers[j];
            den = den + x[Half + j] * powers[j + 1];
          }
          const double den2 = den * den;
          for (Index j = 0; j < Half; ++j) {
            J[i, j] = powers[j] / den;
            J[i, Half + j] = -num * powers[j + 1] / den2;
          }
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<32, Scalar> x,
                            VectorView<32, Scalar> r) -> ErrorOrVoid {
        constexpr Index Half = 16;
        for (Index i = 0; i < 32; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          double powers[17];
          powers[0] = 1.0;
          for (Index j = 1; j <= Half; ++j)
            powers[j] = powers[j - 1] * xv;
          auto num = x[0] * 0.0;
          auto den = x[0] * 0.0 + 1.0;
          for (Index j = 0; j < Half; ++j) {
            num = num + x[j] * powers[j];
            den = den + x[Half + j] * powers[j + 1];
          }
          r[i] = num / den - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<32> x,
                                 MatrixView<32, 32> J) -> ErrorOrVoid {
        constexpr Index Half = 16;
        for (Index i = 0; i < 32; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          double powers[17];
          powers[0] = 1.0;
          for (Index j = 1; j <= Half; ++j)
            powers[j] = powers[j - 1] * xv;
          double num = 0.0;
          double den = 1.0;
          for (Index j = 0; j < Half; ++j) {
            num = num + x[j] * powers[j];
            den = den + x[Half + j] * powers[j + 1];
          }
          const double den2 = den * den;
          for (Index j = 0; j < Half; ++j) {
            J[i, j] = powers[j] / den;
            J[i, Half + j] = -num * powers[j + 1] / den2;
          }
        }
        return {};
      };
      run_pair.template operator()<32, 32>(residual_dynamic, jacobian_dynamic,
                                           residual_static, jacobian_static);
    } else if (corpus.model_id == "rational_dense" && corpus.m == 64 &&
               corpus.n == 64) {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        constexpr Index Half = 32;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          double powers[33];
          powers[0] = 1.0;
          for (Index j = 1; j <= Half; ++j)
            powers[j] = powers[j - 1] * xv;
          auto num = x[0] * 0.0;
          auto den = x[0] * 0.0 + 1.0;
          for (Index j = 0; j < Half; ++j) {
            num = num + x[j] * powers[j];
            den = den + x[Half + j] * powers[j + 1];
          }
          r[i] = num / den - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        constexpr Index Half = 32;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          double powers[33];
          powers[0] = 1.0;
          for (Index j = 1; j <= Half; ++j)
            powers[j] = powers[j - 1] * xv;
          double num = 0.0;
          double den = 1.0;
          for (Index j = 0; j < Half; ++j) {
            num = num + x[j] * powers[j];
            den = den + x[Half + j] * powers[j + 1];
          }
          const double den2 = den * den;
          for (Index j = 0; j < Half; ++j) {
            J[i, j] = powers[j] / den;
            J[i, Half + j] = -num * powers[j + 1] / den2;
          }
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<64, Scalar> x,
                            VectorView<64, Scalar> r) -> ErrorOrVoid {
        constexpr Index Half = 32;
        for (Index i = 0; i < 64; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          double powers[33];
          powers[0] = 1.0;
          for (Index j = 1; j <= Half; ++j)
            powers[j] = powers[j - 1] * xv;
          auto num = x[0] * 0.0;
          auto den = x[0] * 0.0 + 1.0;
          for (Index j = 0; j < Half; ++j) {
            num = num + x[j] * powers[j];
            den = den + x[Half + j] * powers[j + 1];
          }
          r[i] = num / den - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<64> x,
                                 MatrixView<64, 64> J) -> ErrorOrVoid {
        constexpr Index Half = 32;
        for (Index i = 0; i < 64; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          double powers[33];
          powers[0] = 1.0;
          for (Index j = 1; j <= Half; ++j)
            powers[j] = powers[j - 1] * xv;
          double num = 0.0;
          double den = 1.0;
          for (Index j = 0; j < Half; ++j) {
            num += x[j] * powers[j];
            den += x[Half + j] * powers[j + 1];
          }
          const double den2 = den * den;
          for (Index j = 0; j < Half; ++j) {
            J[i, j] = powers[j] / den;
            J[i, Half + j] = -num * powers[j + 1] / den2;
          }
        }
        return {};
      };
      run_pair.template operator()<64, 64>(residual_dynamic, jacobian_dynamic,
                                           residual_static, jacobian_static);
    } else if (corpus.model_id == "exp_sum" && corpus.m == 512 &&
               corpus.n == 32) {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          auto value = x[0] * 0.0;
          for (Index j = 0; j < corpus.n; j += 2) {
            value = value + x[j] * exp(-x[j + 1] * xv);
          }
          r[i] = value - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          for (Index j = 0; j < corpus.n; j += 2) {
            const double e = std::exp(-x[j + 1] * xv);
            J[i, j] = e;
            J[i, j + 1] = -x[j] * xv * e;
          }
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<32, Scalar> x,
                            VectorView<512, Scalar> r) -> ErrorOrVoid {
        for (Index i = 0; i < 512; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          auto value = x[0] * 0.0;
          for (Index j = 0; j < 32; j += 2) {
            value = value + x[j] * exp(-x[j + 1] * xv);
          }
          r[i] = value - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<32> x,
                                 MatrixView<512, 32> J) -> ErrorOrVoid {
        for (Index i = 0; i < 512; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          for (Index j = 0; j < 32; j += 2) {
            const double e = std::exp(-x[j + 1] * xv);
            J[i, j] = e;
            J[i, j + 1] = -x[j] * xv * e;
          }
        }
        return {};
      };
      run_pair.template operator()<512, 32>(residual_dynamic, jacobian_dynamic,
                                            residual_static, jacobian_static);
    } else if (corpus.model_id == "exp_sum" && corpus.m == 1024 &&
               corpus.n == 64) {
      auto residual_dynamic =
          [&]<class Scalar>(
              ConstVectorView<std::dynamic_extent, Scalar> x,
              VectorView<std::dynamic_extent, Scalar> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          auto value = x[0] * 0.0;
          for (Index j = 0; j < corpus.n; j += 2) {
            value = value + x[j] * exp(-x[j + 1] * xv);
          }
          r[i] = value - row.back();
        }
        return {};
      };
      auto jacobian_dynamic =
          [&](ConstVectorView<std::dynamic_extent> x,
              MatrixView<std::dynamic_extent, std::dynamic_extent> J)
          -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          for (Index j = 0; j < corpus.n; j += 2) {
            const double e = std::exp(-x[j + 1] * xv);
            J[i, j] = e;
            J[i, j + 1] = -x[j] * xv * e;
          }
        }
        return {};
      };
      auto residual_static =
          [&]<class Scalar>(ConstVectorView<64, Scalar> x,
                            VectorView<1024, Scalar> r) -> ErrorOrVoid {
        for (Index i = 0; i < 1024; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          auto value = x[0] * 0.0;
          for (Index j = 0; j < 64; j += 2) {
            value = value + x[j] * exp(-x[j + 1] * xv);
          }
          r[i] = value - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<64> x,
                                 MatrixView<1024, 64> J) -> ErrorOrVoid {
        for (Index i = 0; i < 1024; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          for (Index j = 0; j < 64; j += 2) {
            const double e = std::exp(-x[j + 1] * xv);
            J[i, j] = e;
            J[i, j + 1] = -x[j] * xv * e;
          }
        }
        return {};
      };
      run_pair.template operator()<1024, 64>(residual_dynamic, jacobian_dynamic,
                                             residual_static, jacobian_static);
    } else {
      throw std::runtime_error(
          "Explicit callback dispatch not yet implemented for " +
          corpus.model_id + " / " + corpus.name);
    }
  }
  return report;
}

void print_stats_line(const std::string &label, const ComparisonStats &stats) {
  std::cout << "  " << label << ": count=" << stats.count
            << " max_abs=" << std::scientific << stats.max_abs_error
            << " max_rel=" << stats.max_rel_error
            << " rms_abs=" << stats.rms_abs_error()
            << " rms_rel=" << stats.rms_rel_error() << '\n';
}

void print_problem_report(const ProblemReport &report) {
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "passed " << report.name << " dynamic_total="
            << milliseconds(report.dynamic_timing.total_seconds)
            << "ms static_total="
            << milliseconds(report.static_timing.total_seconds)
            << "ms dynamic_residual="
            << milliseconds(report.dynamic_timing.residual_seconds)
            << "ms static_residual="
            << milliseconds(report.static_timing.residual_seconds)
            << "ms dynamic_analytic="
            << milliseconds(report.dynamic_timing.analytic_jacobian_seconds)
            << "ms static_analytic="
            << milliseconds(report.static_timing.analytic_jacobian_seconds)
            << "ms dynamic_autodiff="
            << milliseconds(report.dynamic_timing.autodiff_jacobian_seconds)
            << "ms static_autodiff="
            << milliseconds(report.static_timing.autodiff_jacobian_seconds)
            << "ms dynamic_fd="
            << milliseconds(report.dynamic_timing.forward_difference_seconds)
            << "ms static_fd="
            << milliseconds(report.static_timing.forward_difference_seconds)
            << "ms dynamic_central="
            << milliseconds(report.dynamic_timing.central_difference_seconds)
            << "ms static_central="
            << milliseconds(report.static_timing.central_difference_seconds)
            << "ms";
  if (report.numerical_derivatives_skipped) {
    std::cout << " numerical_derivative_checks_skipped=true";
  }
  std::cout << '\n';
  std::cout
      << "  residual_plus_jacobian: dynamic_analytic="
      << milliseconds(
             report.dynamic_timing.analytic_residual_and_jacobian_seconds)
      << "ms dynamic_autodiff="
      << milliseconds(report.dynamic_timing.autodiff_jacobian_seconds)
      << "ms dynamic_fd="
      << milliseconds(report.dynamic_timing
                          .forward_difference_residual_and_jacobian_seconds)
      << "ms dynamic_central="
      << milliseconds(report.dynamic_timing
                          .central_difference_residual_and_jacobian_seconds)
      << "ms static_analytic="
      << milliseconds(
             report.static_timing.analytic_residual_and_jacobian_seconds)
      << "ms static_autodiff="
      << milliseconds(report.static_timing.autodiff_jacobian_seconds)
      << "ms static_fd="
      << milliseconds(report.static_timing
                          .forward_difference_residual_and_jacobian_seconds)
      << "ms static_central="
      << milliseconds(report.static_timing
                          .central_difference_residual_and_jacobian_seconds)
      << "ms\n";
  print_stats_line("residual", report.residual_stats);
  print_stats_line("analytic", report.analytic_stats);
  if (report.autodiff_stats.count > 0) {
    print_stats_line("autodiff", report.autodiff_stats);
  }
  if (report.autodiff_vs_analytic_stats.count > 0) {
    print_stats_line("autodiff_vs_analytic", report.autodiff_vs_analytic_stats);
  }
  if (report.forward_difference_stats.count > 0) {
    print_stats_line("forward_diff", report.forward_difference_stats);
  }
  if (report.central_difference_stats.count > 0) {
    print_stats_line("central_diff", report.central_difference_stats);
  }
  for (const auto &failure : report.solver_failures) {
    std::cout << "  solver_failure: " << failure << '\n';
  }
}

void print_summary(const SummaryStats &summary) {
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "summary problems=" << summary.problems
            << " numerical_derivative_check_skips="
            << summary.numerical_derivative_skips << " dynamic_total="
            << milliseconds(summary.dynamic_timing.total_seconds)
            << "ms static_total="
            << milliseconds(summary.static_timing.total_seconds)
            << "ms dynamic_residual="
            << milliseconds(summary.dynamic_timing.residual_seconds)
            << "ms static_residual="
            << milliseconds(summary.static_timing.residual_seconds)
            << "ms dynamic_analytic="
            << milliseconds(summary.dynamic_timing.analytic_jacobian_seconds)
            << "ms static_analytic="
            << milliseconds(summary.static_timing.analytic_jacobian_seconds)
            << "ms dynamic_autodiff="
            << milliseconds(summary.dynamic_timing.autodiff_jacobian_seconds)
            << "ms static_autodiff="
            << milliseconds(summary.static_timing.autodiff_jacobian_seconds)
            << "ms dynamic_fd="
            << milliseconds(summary.dynamic_timing.forward_difference_seconds)
            << "ms static_fd="
            << milliseconds(summary.static_timing.forward_difference_seconds)
            << "ms dynamic_central="
            << milliseconds(summary.dynamic_timing.central_difference_seconds)
            << "ms static_central="
            << milliseconds(summary.static_timing.central_difference_seconds)
            << "ms\n";
  std::cout
      << "  residual_plus_jacobian: dynamic_analytic="
      << milliseconds(
             summary.dynamic_timing.analytic_residual_and_jacobian_seconds)
      << "ms dynamic_autodiff="
      << milliseconds(summary.dynamic_timing.autodiff_jacobian_seconds)
      << "ms dynamic_fd="
      << milliseconds(summary.dynamic_timing
                          .forward_difference_residual_and_jacobian_seconds)
      << "ms dynamic_central="
      << milliseconds(summary.dynamic_timing
                          .central_difference_residual_and_jacobian_seconds)
      << "ms static_analytic="
      << milliseconds(
             summary.static_timing.analytic_residual_and_jacobian_seconds)
      << "ms static_autodiff="
      << milliseconds(summary.static_timing.autodiff_jacobian_seconds)
      << "ms static_fd="
      << milliseconds(summary.static_timing
                          .forward_difference_residual_and_jacobian_seconds)
      << "ms static_central="
      << milliseconds(summary.static_timing
                          .central_difference_residual_and_jacobian_seconds)
      << "ms\n";
  std::cout
      << "  numerical_validation_subset_residual_plus_jacobian: "
         "dynamic_analytic="
      << milliseconds(summary.dynamic_timing.numerical_subset
                          .analytic_residual_and_jacobian_seconds)
      << "ms dynamic_autodiff="
      << milliseconds(summary.dynamic_timing.numerical_subset
                          .autodiff_residual_and_jacobian_seconds)
      << "ms dynamic_fd="
      << milliseconds(summary.dynamic_timing.numerical_subset
                          .forward_difference_residual_and_jacobian_seconds)
      << "ms dynamic_central="
      << milliseconds(summary.dynamic_timing.numerical_subset
                          .central_difference_residual_and_jacobian_seconds)
      << "ms static_analytic="
      << milliseconds(summary.static_timing.numerical_subset
                          .analytic_residual_and_jacobian_seconds)
      << "ms static_autodiff="
      << milliseconds(summary.static_timing.numerical_subset
                          .autodiff_residual_and_jacobian_seconds)
      << "ms static_fd="
      << milliseconds(summary.static_timing.numerical_subset
                          .forward_difference_residual_and_jacobian_seconds)
      << "ms static_central="
      << milliseconds(summary.static_timing.numerical_subset
                          .central_difference_residual_and_jacobian_seconds)
      << "ms\n";
  print_stats_line("residual", summary.residual_stats);
  print_stats_line("analytic", summary.analytic_stats);
  if (summary.autodiff_stats.count > 0) {
    print_stats_line("autodiff", summary.autodiff_stats);
  }
  if (summary.autodiff_vs_analytic_stats.count > 0) {
    print_stats_line("autodiff_vs_analytic",
                     summary.autodiff_vs_analytic_stats);
  }
  if (summary.forward_difference_stats.count > 0) {
    print_stats_line("forward_diff", summary.forward_difference_stats);
  }
  if (summary.central_difference_stats.count > 0) {
    print_stats_line("central_diff", summary.central_difference_stats);
  }
  if (summary.solver_start_cases > 0) {
    std::cout << "  solver_start_cases: passed="
              << summary.solver_start_case_passes
              << " failed=" << summary.solver_failures.size()
              << " total=" << summary.solver_start_cases
              << " dynamic_average_lre="
              << summary.solver_dynamic_lre_sum / summary.solver_start_cases
              << " static_average_lre="
              << summary.solver_static_lre_sum / summary.solver_start_cases
              << '\n';
    if (summary.solver_dynamic_seconds > 0.0 ||
        summary.solver_static_seconds > 0.0) {
      std::cout << "  solver_seconds: dynamic_total="
                << summary.solver_dynamic_seconds << " dynamic_mean="
                << summary.solver_dynamic_seconds / summary.solver_start_cases
                << " static_total=" << summary.solver_static_seconds
                << " static_mean="
                << summary.solver_static_seconds / summary.solver_start_cases
                << '\n';
    }
  }
}

void write_problem_csv_row(std::ofstream &file, const ProblemReport &report) {
  file << report.name << ",problem,"
       << (report.numerical_derivatives_skipped ? "true" : "false") << ','
       << std::setprecision(17)
       << milliseconds(report.dynamic_timing.total_seconds) << ','
       << milliseconds(report.dynamic_timing.residual_seconds) << ','
       << milliseconds(report.dynamic_timing.analytic_jacobian_seconds) << ','
       << milliseconds(
              report.dynamic_timing.analytic_residual_and_jacobian_seconds)
       << ',' << milliseconds(report.dynamic_timing.autodiff_jacobian_seconds)
       << ',' << milliseconds(report.dynamic_timing.forward_difference_seconds)
       << ','
       << milliseconds(report.dynamic_timing
                           .forward_difference_residual_and_jacobian_seconds)
       << ',' << milliseconds(report.dynamic_timing.central_difference_seconds)
       << ','
       << milliseconds(report.dynamic_timing
                           .central_difference_residual_and_jacobian_seconds)
       << ',' << milliseconds(report.static_timing.total_seconds) << ','
       << milliseconds(report.static_timing.residual_seconds) << ','
       << milliseconds(report.static_timing.analytic_jacobian_seconds) << ','
       << milliseconds(
              report.static_timing.analytic_residual_and_jacobian_seconds)
       << ',' << milliseconds(report.static_timing.autodiff_jacobian_seconds)
       << ',' << milliseconds(report.static_timing.forward_difference_seconds)
       << ','
       << milliseconds(report.static_timing
                           .forward_difference_residual_and_jacobian_seconds)
       << ',' << milliseconds(report.static_timing.central_difference_seconds)
       << ','
       << milliseconds(report.static_timing
                           .central_difference_residual_and_jacobian_seconds)
       << ','
       << milliseconds(report.dynamic_timing.numerical_subset
                           .analytic_residual_and_jacobian_seconds)
       << ','
       << milliseconds(report.dynamic_timing.numerical_subset
                           .autodiff_residual_and_jacobian_seconds)
       << ','
       << milliseconds(report.dynamic_timing.numerical_subset
                           .forward_difference_residual_and_jacobian_seconds)
       << ','
       << milliseconds(report.dynamic_timing.numerical_subset
                           .central_difference_residual_and_jacobian_seconds)
       << ','
       << milliseconds(report.static_timing.numerical_subset
                           .analytic_residual_and_jacobian_seconds)
       << ','
       << milliseconds(report.static_timing.numerical_subset
                           .autodiff_residual_and_jacobian_seconds)
       << ','
       << milliseconds(report.static_timing.numerical_subset
                           .forward_difference_residual_and_jacobian_seconds)
       << ','
       << milliseconds(report.static_timing.numerical_subset
                           .central_difference_residual_and_jacobian_seconds)
       << ',' << report.residual_stats.count << ','
       << report.residual_stats.max_abs_error << ','
       << report.residual_stats.max_rel_error << ','
       << report.residual_stats.rms_abs_error() << ','
       << report.residual_stats.rms_rel_error() << ','
       << report.analytic_stats.count << ','
       << report.analytic_stats.max_abs_error << ','
       << report.analytic_stats.max_rel_error << ','
       << report.analytic_stats.rms_abs_error() << ','
       << report.analytic_stats.rms_rel_error() << ','
       << report.autodiff_stats.count << ','
       << report.autodiff_stats.max_abs_error << ','
       << report.autodiff_stats.max_rel_error << ','
       << report.autodiff_stats.rms_abs_error() << ','
       << report.autodiff_stats.rms_rel_error() << ','
       << report.autodiff_vs_analytic_stats.count << ','
       << report.autodiff_vs_analytic_stats.max_abs_error << ','
       << report.autodiff_vs_analytic_stats.max_rel_error << ','
       << report.autodiff_vs_analytic_stats.rms_abs_error() << ','
       << report.autodiff_vs_analytic_stats.rms_rel_error() << ','
       << report.forward_difference_stats.count << ','
       << report.forward_difference_stats.max_abs_error << ','
       << report.forward_difference_stats.max_rel_error << ','
       << report.forward_difference_stats.rms_abs_error() << ','
       << report.forward_difference_stats.rms_rel_error() << ','
       << report.central_difference_stats.count << ','
       << report.central_difference_stats.max_abs_error << ','
       << report.central_difference_stats.max_rel_error << ','
       << report.central_difference_stats.rms_abs_error() << ','
       << report.central_difference_stats.rms_rel_error() << '\n';
}

void write_summary_csv_row(std::ofstream &file, const SummaryStats &summary) {
  file << "summary,summary,false," << std::setprecision(17)
       << milliseconds(summary.dynamic_timing.total_seconds) << ','
       << milliseconds(summary.dynamic_timing.residual_seconds) << ','
       << milliseconds(summary.dynamic_timing.analytic_jacobian_seconds) << ','
       << milliseconds(
              summary.dynamic_timing.analytic_residual_and_jacobian_seconds)
       << ',' << milliseconds(summary.dynamic_timing.autodiff_jacobian_seconds)
       << ',' << milliseconds(summary.dynamic_timing.forward_difference_seconds)
       << ','
       << milliseconds(summary.dynamic_timing
                           .forward_difference_residual_and_jacobian_seconds)
       << ',' << milliseconds(summary.dynamic_timing.central_difference_seconds)
       << ','
       << milliseconds(summary.dynamic_timing
                           .central_difference_residual_and_jacobian_seconds)
       << ',' << milliseconds(summary.static_timing.total_seconds) << ','
       << milliseconds(summary.static_timing.residual_seconds) << ','
       << milliseconds(summary.static_timing.analytic_jacobian_seconds) << ','
       << milliseconds(
              summary.static_timing.analytic_residual_and_jacobian_seconds)
       << ',' << milliseconds(summary.static_timing.autodiff_jacobian_seconds)
       << ',' << milliseconds(summary.static_timing.forward_difference_seconds)
       << ','
       << milliseconds(summary.static_timing
                           .forward_difference_residual_and_jacobian_seconds)
       << ',' << milliseconds(summary.static_timing.central_difference_seconds)
       << ','
       << milliseconds(summary.static_timing
                           .central_difference_residual_and_jacobian_seconds)
       << ','
       << milliseconds(summary.dynamic_timing.numerical_subset
                           .analytic_residual_and_jacobian_seconds)
       << ','
       << milliseconds(summary.dynamic_timing.numerical_subset
                           .autodiff_residual_and_jacobian_seconds)
       << ','
       << milliseconds(summary.dynamic_timing.numerical_subset
                           .forward_difference_residual_and_jacobian_seconds)
       << ','
       << milliseconds(summary.dynamic_timing.numerical_subset
                           .central_difference_residual_and_jacobian_seconds)
       << ','
       << milliseconds(summary.static_timing.numerical_subset
                           .analytic_residual_and_jacobian_seconds)
       << ','
       << milliseconds(summary.static_timing.numerical_subset
                           .autodiff_residual_and_jacobian_seconds)
       << ','
       << milliseconds(summary.static_timing.numerical_subset
                           .forward_difference_residual_and_jacobian_seconds)
       << ','
       << milliseconds(summary.static_timing.numerical_subset
                           .central_difference_residual_and_jacobian_seconds)
       << ',' << summary.residual_stats.count << ','
       << summary.residual_stats.max_abs_error << ','
       << summary.residual_stats.max_rel_error << ','
       << summary.residual_stats.rms_abs_error() << ','
       << summary.residual_stats.rms_rel_error() << ','
       << summary.analytic_stats.count << ','
       << summary.analytic_stats.max_abs_error << ','
       << summary.analytic_stats.max_rel_error << ','
       << summary.analytic_stats.rms_abs_error() << ','
       << summary.analytic_stats.rms_rel_error() << ','
       << summary.autodiff_stats.count << ','
       << summary.autodiff_stats.max_abs_error << ','
       << summary.autodiff_stats.max_rel_error << ','
       << summary.autodiff_stats.rms_abs_error() << ','
       << summary.autodiff_stats.rms_rel_error() << ','
       << summary.autodiff_vs_analytic_stats.count << ','
       << summary.autodiff_vs_analytic_stats.max_abs_error << ','
       << summary.autodiff_vs_analytic_stats.max_rel_error << ','
       << summary.autodiff_vs_analytic_stats.rms_abs_error() << ','
       << summary.autodiff_vs_analytic_stats.rms_rel_error() << ','
       << summary.forward_difference_stats.count << ','
       << summary.forward_difference_stats.max_abs_error << ','
       << summary.forward_difference_stats.max_rel_error << ','
       << summary.forward_difference_stats.rms_abs_error() << ','
       << summary.forward_difference_stats.rms_rel_error() << ','
       << summary.central_difference_stats.count << ','
       << summary.central_difference_stats.max_abs_error << ','
       << summary.central_difference_stats.max_rel_error << ','
       << summary.central_difference_stats.rms_abs_error() << ','
       << summary.central_difference_stats.rms_rel_error() << '\n';
}

void write_csv_report(const std::filesystem::path &path,
                      const std::vector<ProblemReport> &reports,
                      const SummaryStats &summary) {
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to open CSV report path " + path.string());
  }

  const auto write_stats = [&](const ComparisonStats &stats) {
    file << stats.count << ',' << stats.max_abs_error << ','
         << stats.max_rel_error << ',' << stats.rms_abs_error() << ','
         << stats.rms_rel_error();
  };
  const auto write_row =
      [&](const std::string &name, const std::string &row_type, bool skipped,
          const ComparisonStats &residual, const ComparisonStats &analytic,
          const ComparisonStats &graph,
          const ComparisonStats &graph_vs_analytic,
          const ComparisonStats &direct_dual,
          const ComparisonStats &direct_dual_vs_analytic,
          const ComparisonStats &forward_difference,
          const ComparisonStats &central_difference) {
        file << name << ',' << row_type << ',' << (skipped ? "true" : "false")
             << ',' << std::setprecision(17);
        write_stats(residual);
        file << ',';
        write_stats(analytic);
        file << ',';
        write_stats(graph);
        file << ',';
        write_stats(graph_vs_analytic);
        file << ',';
        write_stats(direct_dual);
        file << ',';
        write_stats(direct_dual_vs_analytic);
        file << ',';
        write_stats(forward_difference);
        file << ',';
        write_stats(central_difference);
        file << '\n';
      };

  file << "name,row_type,numerical_derivative_checks_skipped,"
       << "residual_count,residual_max_abs,residual_max_rel,residual_rms_abs,"
          "residual_rms_rel,"
       << "analytic_count,analytic_max_abs,analytic_max_rel,analytic_rms_abs,"
          "analytic_rms_rel,"
       << "graph_autodiff_count,graph_autodiff_max_abs,"
          "graph_autodiff_max_rel,graph_autodiff_rms_abs,"
          "graph_autodiff_rms_rel,"
       << "graph_autodiff_vs_analytic_count,"
          "graph_autodiff_vs_analytic_max_abs,"
          "graph_autodiff_vs_analytic_max_rel,"
          "graph_autodiff_vs_analytic_rms_abs,"
          "graph_autodiff_vs_analytic_rms_rel,"
       << "direct_dual_count,direct_dual_max_abs,direct_dual_max_rel,"
          "direct_dual_rms_abs,direct_dual_rms_rel,"
       << "direct_dual_vs_analytic_count,direct_dual_vs_analytic_max_abs,"
          "direct_dual_vs_analytic_max_rel,direct_dual_vs_analytic_rms_abs,"
          "direct_dual_vs_analytic_rms_rel,"
       << "forward_difference_count,forward_difference_max_abs,"
          "forward_difference_max_rel,forward_difference_rms_abs,"
          "forward_difference_rms_rel,"
       << "central_difference_count,central_difference_max_abs,"
          "central_difference_max_rel,central_difference_rms_abs,"
          "central_difference_rms_rel\n";

  for (const auto &report : reports) {
    write_row(report.name, "problem", report.numerical_derivatives_skipped,
              report.residual_stats, report.analytic_stats,
              report.graph_autodiff_stats,
              report.graph_autodiff_vs_analytic_stats, report.direct_dual_stats,
              report.direct_dual_vs_analytic_stats,
              report.forward_difference_stats, report.central_difference_stats);
  }
  write_row("summary", "summary", false, summary.residual_stats,
            summary.analytic_stats, summary.graph_autodiff_stats,
            summary.graph_autodiff_vs_analytic_stats, summary.direct_dual_stats,
            summary.direct_dual_vs_analytic_stats,
            summary.forward_difference_stats, summary.central_difference_stats);
}

void write_solver_work_csv(const std::filesystem::path &path,
                           const std::vector<ProblemReport> &analytic_reports,
                           const std::vector<ProblemReport> &autodiff_reports,
                           bool include_levmar,
                           std::string_view linear_algebra = "damped_qr") {
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to open solver work CSV path " +
                             path.string());
  }
  file << "solver,linear_algebra,derivative,problem,start,extent,m,n,elapsed_"
          "ms,termination,"
          "lre,"
          "final_cost,lambda,gradient_inf_norm,step_norm,iterations,"
          "function_evaluations,jacobian_evaluations,linear_solves,"
          "factorization_count,accepted_steps,rejected_steps\n";
  const auto write_levmar_reports =
      [&](std::string_view derivative,
          const std::vector<ProblemReport> &reports) {
        for (const auto &report : reports) {
          for (const auto &row : report.solver_work_rows) {
            const auto &result = row.result;
            const Index factorization_count = linear_algebra == "damped_qr"
                                                  ? result.linear_solves
                                                  : result.factorization_count;
            file << "levmar," << linear_algebra << ',' << derivative << ','
                 << report.name << ',' << row.start_label << ',' << row.extent
                 << ',' << row.m << ',' << row.n << ',' << std::setprecision(17)
                 << milliseconds(row.elapsed_seconds) << ','
                 << termination_reason_name(result.termination) << ','
                 << row.lre << ',' << result.final_cost << ',' << result.lambda
                 << ',' << result.gradient_inf_norm << ',' << result.step_norm
                 << ',' << result.iterations << ','
                 << result.function_evaluations << ','
                 << result.jacobian_evaluations << ',' << result.linear_solves
                 << ',' << factorization_count << ',' << result.accepted_steps
                 << ',' << result.rejected_steps << '\n';
          }
        }
      };
  if (include_levmar) {
    write_levmar_reports("analytic", analytic_reports);
    write_levmar_reports("autodiff", autodiff_reports);
  }
  const auto write_external_reports =
      [&](const std::vector<ProblemReport> &reports) {
        for (const auto &report : reports) {
          for (const auto &result : report.external_solver_results) {
            const bool finite_difference = result.solver.ends_with("lmdif");
            const bool autodiff = result.solver.ends_with("autodiff");
            file << result.solver << ",external,"
                 << (finite_difference ? "finite_difference"
                     : autodiff        ? "autodiff"
                                       : "analytic")
                 << ',' << report.name << ',' << result.start_label << ','
                 << (autodiff ? "static" : "dynamic") << ',' << result.m << ','
                 << result.n << ',' << std::setprecision(17)
                 << milliseconds(result.seconds) << ','
                 << (result.usable ? "converged" : "failed") << ','
                 << result.lre << ",nan,nan,nan,nan," << result.iterations
                 << ',' << result.function_evaluations << ','
                 << (result.has_jacobian_evaluations
                         ? std::to_string(result.jacobian_evaluations)
                         : "nan")
                 << ','
                 << (result.has_linear_solves
                         ? std::to_string(result.linear_solves)
                         : "nan")
                 << ",nan,"
                 << (result.has_accepted_steps
                         ? std::to_string(result.accepted_steps)
                         : "nan")
                 << ','
                 << (result.has_rejected_steps
                         ? std::to_string(result.rejected_steps)
                         : "nan")
                 << '\n';
          }
        }
      };
  write_external_reports(analytic_reports);
  write_external_reports(autodiff_reports);
}

std::string_view trial_decision_name(TrialDecision decision) {
  switch (decision) {
  case TrialDecision::Accepted:
    return "accepted";
  case TrialDecision::LinearSolveFailure:
    return "linear_solve_failure";
  case TrialDecision::DampingLimit:
    return "damping_limit";
  case TrialDecision::NonFiniteTrialParameter:
    return "nonfinite_trial_parameter";
  case TrialDecision::SmallStep:
    return "small_step";
  case TrialDecision::FunctionEvaluationLimit:
    return "function_evaluation_limit";
  case TrialDecision::NonFiniteTrialCost:
    return "nonfinite_trial_cost";
  case TrialDecision::NonFinitePredictedReduction:
    return "nonfinite_predicted_reduction";
  case TrialDecision::NonPositivePredictedReduction:
    return "nonpositive_predicted_reduction";
  case TrialDecision::SmallCostReduction:
    return "small_cost_reduction";
  case TrialDecision::NonFiniteRho:
    return "nonfinite_rho";
  case TrialDecision::LowRho:
    return "low_rho";
  }
  return "unknown";
}

void write_controller_trace_csv(
    const std::filesystem::path &path,
    const std::vector<ControllerTraceRecord> &records) {
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to open controller trace CSV path " +
                             path.string());
  }
  file << "problem,start,derivative,extent,attempt,cost_before,trial_cost,"
          "actual_reduction,predicted_reduction,rho,lambda_before,lambda_after,"
          "trust_radius_before,trust_radius_after,selected_lambda,inner_linear_"
          "solves,radius_bound_active,gradient_inf_norm,"
          "raw_step_norm,scaled_step_norm,current_parameters,trial_parameters,"
          "step,gradient,jacobian_column_norms,parameter_scales,"
          "effective_damping_diagonal,max_abs_column_correlation,"
          "max_correlation_col_i,max_correlation_col_j,"
          "decision,termination\n";
  const auto write_vector = [&](const std::vector<double> &values) {
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (i != 0) {
        file << ';';
      }
      file << values[i];
    }
  };
  file << std::setprecision(17);
  for (const auto &record : records) {
    for (std::size_t attempt = 0; attempt < record.trials.size(); ++attempt) {
      const auto &trial = record.trials[attempt];
      file << record.problem << ',' << record.start << ',' << record.derivative
           << ',' << record.extent << ',' << attempt + 1 << ','
           << trial.cost_before << ',' << trial.trial_cost << ','
           << trial.actual_reduction << ',' << trial.predicted_reduction << ','
           << trial.rho << ',' << trial.lambda_before << ','
           << trial.lambda_after << ',' << trial.trust_radius_before << ','
           << trial.trust_radius_after << ',' << trial.selected_lambda << ','
           << trial.inner_linear_solves << ',' << trial.radius_bound_active
           << ',' << trial.gradient_inf_norm << ',' << trial.raw_step_norm
           << ',' << trial.scaled_step_norm << ',';
      write_vector(trial.current_parameters);
      file << ',';
      write_vector(trial.trial_parameters);
      file << ',';
      write_vector(trial.step);
      file << ',';
      write_vector(trial.gradient);
      file << ',';
      write_vector(trial.jacobian_column_norms);
      file << ',';
      write_vector(trial.parameter_scales);
      file << ',';
      write_vector(trial.effective_damping_diagonal);
      file << ',' << trial.max_abs_column_correlation << ','
           << trial.max_correlation_col_i << ',' << trial.max_correlation_col_j
           << ',' << trial_decision_name(trial.decision) << ','
           << termination_reason_name(trial.termination) << '\n';
    }
  }
}

void write_benchmark_csv_rows(std::ofstream &file, const std::string &scope,
                              const std::string &name, std::uint64_t iterations,
                              const TimingMoments &dynamic,
                              const TimingMoments &statik) {
  const auto write_row = [&](const std::string &metric,
                             const ScalarMoments &dynamic_values,
                             const ScalarMoments &static_values) {
    const auto mean_milliseconds = [](const ScalarMoments &values) {
      return values.count == 0 ? std::numeric_limits<double>::quiet_NaN()
                               : milliseconds(values.mean());
    };
    const auto stddev_milliseconds = [](const ScalarMoments &values) {
      return values.count == 0 ? std::numeric_limits<double>::quiet_NaN()
                               : milliseconds(values.stddev());
    };
    file << scope << ',' << name << ',' << iterations << ',' << metric << ','
         << std::setprecision(17) << mean_milliseconds(dynamic_values) << ','
         << stddev_milliseconds(dynamic_values) << ','
         << mean_milliseconds(static_values) << ','
         << stddev_milliseconds(static_values) << '\n';
  };

  write_row("analytic_residual_and_jacobian",
            dynamic.analytic_residual_and_jacobian,
            statik.analytic_residual_and_jacobian);
  write_row("graph_autodiff_residual_and_jacobian", dynamic.graph,
            statik.graph);
  write_row("direct_dual_residual_and_jacobian", dynamic.direct_dual,
            statik.direct_dual);
  write_row("forward_difference_residual_and_jacobian",
            dynamic.forward_difference_residual_and_jacobian,
            statik.forward_difference_residual_and_jacobian);
  write_row("central_difference_residual_and_jacobian",
            dynamic.central_difference_residual_and_jacobian,
            statik.central_difference_residual_and_jacobian);
  write_row("subset_analytic_residual_and_jacobian",
            dynamic.numerical_subset.analytic,
            statik.numerical_subset.analytic);
  write_row("subset_graph_autodiff_residual_and_jacobian",
            dynamic.numerical_subset.graph, statik.numerical_subset.graph);
  write_row("subset_direct_dual_residual_and_jacobian",
            dynamic.numerical_subset.direct_dual,
            statik.numerical_subset.direct_dual);
  write_row("subset_forward_difference_residual_and_jacobian",
            dynamic.numerical_subset.forward_difference,
            statik.numerical_subset.forward_difference);
  write_row("subset_central_difference_residual_and_jacobian",
            dynamic.numerical_subset.central_difference,
            statik.numerical_subset.central_difference);
}

void write_benchmark_csv(const std::filesystem::path &path,
                         const SummaryBenchmark &summary,
                         const std::vector<ProblemBenchmark> &problems,
                         std::uint64_t iterations) {
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to open benchmark CSV path " +
                             path.string());
  }

  file << "scope,name,iterations,metric,dynamic_mean_ms,dynamic_stddev_ms,"
          "static_mean_ms,static_stddev_ms\n";
  write_benchmark_csv_rows(file, "summary", "summary", iterations,
                           summary.dynamic_timing, summary.static_timing);
  for (const auto &problem : problems) {
    write_benchmark_csv_rows(file, "problem", problem.name, iterations,
                             problem.dynamic_timing, problem.static_timing);
  }
}

void print_timing_moment_line(const std::string &label,
                              const ScalarMoments &dynamic,
                              const ScalarMoments &statik) {
  std::cout << "  " << label << ": dynamic_total=" << milliseconds(dynamic.sum)
            << "ms dynamic_mean=" << milliseconds(dynamic.mean())
            << "ms dynamic_stddev=" << milliseconds(dynamic.stddev())
            << "ms static_total=" << milliseconds(statik.sum)
            << "ms static_mean=" << milliseconds(statik.mean())
            << "ms static_stddev=" << milliseconds(statik.stddev()) << "ms\n";
}

void print_problem_benchmark(const ProblemBenchmark &benchmark) {
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "benchmark " << benchmark.name << '\n';
  print_timing_moment_line("total", benchmark.dynamic_timing.total,
                           benchmark.static_timing.total);
  print_timing_moment_line("residual", benchmark.dynamic_timing.residual,
                           benchmark.static_timing.residual);
  print_timing_moment_line("analytic", benchmark.dynamic_timing.analytic,
                           benchmark.static_timing.analytic);
  print_timing_moment_line(
      "analytic_residual_plus_jacobian",
      benchmark.dynamic_timing.analytic_residual_and_jacobian,
      benchmark.static_timing.analytic_residual_and_jacobian);
  print_timing_moment_line("autodiff", benchmark.dynamic_timing.autodiff,
                           benchmark.static_timing.autodiff);
  print_timing_moment_line("forward_diff",
                           benchmark.dynamic_timing.forward_difference,
                           benchmark.static_timing.forward_difference);
  print_timing_moment_line(
      "forward_diff_residual_plus_jacobian",
      benchmark.dynamic_timing.forward_difference_residual_and_jacobian,
      benchmark.static_timing.forward_difference_residual_and_jacobian);
  print_timing_moment_line("central_diff",
                           benchmark.dynamic_timing.central_difference,
                           benchmark.static_timing.central_difference);
  print_timing_moment_line(
      "central_diff_residual_plus_jacobian",
      benchmark.dynamic_timing.central_difference_residual_and_jacobian,
      benchmark.static_timing.central_difference_residual_and_jacobian);
}

void print_summary_benchmark(const SummaryBenchmark &benchmark,
                             std::uint64_t iterations) {
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "benchmark_summary iterations=" << iterations << '\n';
  print_timing_moment_line("total", benchmark.dynamic_timing.total,
                           benchmark.static_timing.total);
  print_timing_moment_line("residual", benchmark.dynamic_timing.residual,
                           benchmark.static_timing.residual);
  print_timing_moment_line("analytic", benchmark.dynamic_timing.analytic,
                           benchmark.static_timing.analytic);
  print_timing_moment_line(
      "analytic_residual_plus_jacobian",
      benchmark.dynamic_timing.analytic_residual_and_jacobian,
      benchmark.static_timing.analytic_residual_and_jacobian);
  print_timing_moment_line("autodiff", benchmark.dynamic_timing.autodiff,
                           benchmark.static_timing.autodiff);
  print_timing_moment_line("forward_diff",
                           benchmark.dynamic_timing.forward_difference,
                           benchmark.static_timing.forward_difference);
  print_timing_moment_line(
      "forward_diff_residual_plus_jacobian",
      benchmark.dynamic_timing.forward_difference_residual_and_jacobian,
      benchmark.static_timing.forward_difference_residual_and_jacobian);
  print_timing_moment_line("central_diff",
                           benchmark.dynamic_timing.central_difference,
                           benchmark.static_timing.central_difference);
  print_timing_moment_line(
      "central_diff_residual_plus_jacobian",
      benchmark.dynamic_timing.central_difference_residual_and_jacobian,
      benchmark.static_timing.central_difference_residual_and_jacobian);
  print_timing_moment_line("subset_analytic_residual_plus_jacobian",
                           benchmark.dynamic_timing.numerical_subset.analytic,
                           benchmark.static_timing.numerical_subset.analytic);
  print_timing_moment_line("subset_autodiff_residual_plus_jacobian",
                           benchmark.dynamic_timing.numerical_subset.autodiff,
                           benchmark.static_timing.numerical_subset.autodiff);
  print_timing_moment_line(
      "subset_forward_diff_residual_plus_jacobian",
      benchmark.dynamic_timing.numerical_subset.forward_difference,
      benchmark.static_timing.numerical_subset.forward_difference);
  print_timing_moment_line(
      "subset_central_diff_residual_plus_jacobian",
      benchmark.dynamic_timing.numerical_subset.central_difference,
      benchmark.static_timing.numerical_subset.central_difference);
}

SummaryBenchmark
benchmark_summary(const std::vector<std::filesystem::path> &problem_dirs,
                  std::uint64_t iterations,
                  std::vector<ProblemBenchmark> &problem_benchmarks) {
  SummaryBenchmark summary_benchmark;
  problem_benchmarks.clear();
  for (const auto &path : problem_dirs) {
    ProblemBenchmark benchmark;
    benchmark.name = path.filename().string();
    problem_benchmarks.push_back(std::move(benchmark));
  }

  for (std::uint64_t iteration = 0; iteration < iterations; ++iteration) {
    SummaryStats iteration_summary;
    for (Index i = 0; i < problem_dirs.size(); ++i) {
      const auto report =
          run_problem<NistAnalyticSolverPolicy>(problem_dirs[i]);
      problem_benchmarks[i].name = report.name;
      problem_benchmarks[i].dynamic_timing.add(report.dynamic_timing);
      problem_benchmarks[i].static_timing.add(report.static_timing);
      merge_summary(iteration_summary, report);
    }
    summary_benchmark.dynamic_timing.add(iteration_summary.dynamic_timing);
    summary_benchmark.static_timing.add(iteration_summary.static_timing);
  }

  return summary_benchmark;
}

template <class Policy>
SolverBenchmark
benchmark_solver(const std::vector<std::filesystem::path> &problem_dirs,
                 std::uint64_t iterations,
                 bool enable_relative_cost_termination = true) {
  SolverBenchmark benchmark;
  for (std::uint64_t iteration = 0; iteration < iterations; ++iteration) {
    SummaryStats summary;
    for (const auto &path : problem_dirs) {
      merge_summary(summary, run_problem<Policy>(
                                 path, true, true, iteration, false, false,
                                 enable_relative_cost_termination));
    }
    benchmark.dynamic_total.add(summary.solver_dynamic_seconds);
    benchmark.static_total.add(summary.solver_static_seconds);
  }
  return benchmark;
}

std::map<std::string, ExternalSolverBenchmark> benchmark_external_solvers(
    const std::vector<std::filesystem::path> &problem_dirs,
    std::uint64_t iterations) {
  std::map<std::string, ExternalSolverBenchmark> benchmarks;

  for (std::uint64_t iteration = 0; iteration < iterations; ++iteration) {
    // Accumulate a complete-suite time for each solver.  This makes one
    // ExternalSolverBenchmark::seconds sample directly comparable to one
    // SolverBenchmark dynamic_total/static_total sample.
    std::map<std::string, double> sweep_seconds;

    for (const auto &path : problem_dirs) {
      const auto report = run_problem<NistAnalyticSolverPolicy>(
          path, true, false, iteration, true, false);
      for (const auto &result : report.external_solver_results) {
        sweep_seconds[result.solver] += result.seconds;

        // Case-level statistics do not depend on timing repetition.  Record
        // them only on the first sweep so passed/total is the actual NIST case
        // count rather than case_count * benchmark_iterations.
        if (iteration == 0) {
          auto &benchmark = benchmarks[result.solver];
          ++benchmark.cases;
          benchmark.lre.add(result.lre);
          benchmark.iterations.add(result.iterations);
          benchmark.function_evaluations.add(result.function_evaluations);
          if (result.has_jacobian_evaluations) {
            benchmark.jacobian_evaluations.add(result.jacobian_evaluations);
          }
          if (result.has_linear_solves) {
            benchmark.linear_solves.add(result.linear_solves);
          }
          if (result.has_accepted_steps) {
            benchmark.accepted_steps.add(result.accepted_steps);
          }
          if (result.has_rejected_steps) {
            benchmark.rejected_steps.add(result.rejected_steps);
          }
          if (result.usable) {
            ++benchmark.usable;
          }
        }
      }
    }

    for (const auto &[solver, seconds] : sweep_seconds) {
      benchmarks[solver].seconds.add(seconds);
    }
  }
  return benchmarks;
}

void print_solver_benchmark(const SolverBenchmark &benchmark,
                            std::uint64_t iterations, Index start_cases) {
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "solver_benchmark iterations=" << iterations
            << " start_cases=" << start_cases << '\n';
  print_timing_moment_line("solver_sweep", benchmark.dynamic_total,
                           benchmark.static_total);
}

void print_solver_comparison(const SummaryStats &analytic_summary,
                             const SolverBenchmark &analytic_benchmark,
                             const SummaryStats &autodiff_summary,
                             const SolverBenchmark &autodiff_benchmark,
                             std::uint64_t iterations) {
  const auto print_row = [](std::string_view name, std::string_view dynamic,
                            std::string_view statik,
                            const SummaryStats &summary,
                            const SolverBenchmark &benchmark) {
    const double cases = static_cast<double>(summary.solver_start_cases);
    std::cout << "  " << name << " (" << dynamic << " / " << statik
              << "): dynamic=" << milliseconds(benchmark.dynamic_total.mean())
              << " +/- " << milliseconds(benchmark.dynamic_total.stddev())
              << "ms static=" << milliseconds(benchmark.static_total.mean())
              << " +/- " << milliseconds(benchmark.static_total.stddev())
              << "ms dynamic_lre="
              << summary.solver_dynamic_lre_sum / summary.solver_start_cases
              << " static_lre="
              << summary.solver_static_lre_sum / summary.solver_start_cases
              << " passed=" << summary.solver_start_case_passes << '/'
              << summary.solver_start_cases << " dynamic_function_evaluations="
              << summary.solver_dynamic_function_evaluations / cases
              << " static_function_evaluations="
              << summary.solver_static_function_evaluations / cases
              << " dynamic_jacobian_evaluations="
              << summary.solver_dynamic_jacobian_evaluations / cases
              << " static_jacobian_evaluations="
              << summary.solver_static_jacobian_evaluations / cases
              << " dynamic_linear_solves="
              << summary.solver_dynamic_linear_solves / cases
              << " static_linear_solves="
              << summary.solver_static_linear_solves / cases
              << " dynamic_accepted_steps="
              << summary.solver_dynamic_accepted_steps / cases
              << " static_accepted_steps="
              << summary.solver_static_accepted_steps / cases
              << " dynamic_rejected_steps="
              << summary.solver_dynamic_rejected_steps / cases
              << " static_rejected_steps="
              << summary.solver_static_rejected_steps / cases << '\n';
  };

  std::cout << std::fixed << std::setprecision(4);
  std::cout << "solver_comparison iterations=" << iterations << '\n';
  print_row("analytic", "analytic", "analytic", analytic_summary,
            analytic_benchmark);
  print_row("autodiff", "graph", "dual", autodiff_summary, autodiff_benchmark);
}

void print_external_solver_benchmarks(
    const std::map<std::string, ExternalSolverBenchmark> &benchmarks) {
  const auto mean_or_nan = [](const ScalarMoments &moments) {
    return moments.count == 0 ? std::numeric_limits<double>::quiet_NaN()
                              : moments.mean();
  };
  std::cout << std::fixed << std::setprecision(4);
  for (const auto &[name, benchmark] : benchmarks) {
    std::cout << "  " << name
              << ": mean=" << milliseconds(benchmark.seconds.mean()) << " +/- "
              << milliseconds(benchmark.seconds.stddev())
              << "ms lre=" << benchmark.lre.mean()
              << " passed=" << benchmark.usable << '/' << benchmark.cases
              << " timing_samples=" << benchmark.seconds.count
              << " function_evaluations="
              << benchmark.function_evaluations.mean()
              << " jacobian_evaluations="
              << mean_or_nan(benchmark.jacobian_evaluations)
              << " linear_solves=" << mean_or_nan(benchmark.linear_solves)
              << " accepted_steps=" << mean_or_nan(benchmark.accepted_steps)
              << " rejected_steps=" << mean_or_nan(benchmark.rejected_steps)
              << '\n';
  }
}

void write_solver_benchmark_csv(const std::filesystem::path &path,
                                const SolverBenchmark &benchmark,
                                std::uint64_t iterations) {
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to open benchmark CSV path " +
                             path.string());
  }
  file << "scope,name,iterations,metric,dynamic_mean_ms,dynamic_stddev_ms,"
          "static_mean_ms,static_stddev_ms\n";
  file << "summary,summary," << iterations << ",solver_sweep,"
       << std::setprecision(17) << milliseconds(benchmark.dynamic_total.mean())
       << ',' << milliseconds(benchmark.dynamic_total.stddev()) << ','
       << milliseconds(benchmark.static_total.mean()) << ','
       << milliseconds(benchmark.static_total.stddev()) << '\n';
}

void write_external_solver_benchmark_csv(
    const std::filesystem::path &path,
    const std::map<std::string, ExternalSolverBenchmark> &benchmarks,
    std::uint64_t iterations) {
  std::ofstream file(path, std::ios::app);
  if (!file) {
    throw std::runtime_error("Failed to open benchmark CSV path " +
                             path.string());
  }
  for (const auto &[name, benchmark] : benchmarks) {
    file << "external," << name << ',' << iterations << ",solver_sweep,"
         << std::setprecision(17) << milliseconds(benchmark.seconds.mean())
         << ',' << milliseconds(benchmark.seconds.stddev()) << ",nan,nan\n";
  }
}

void write_solver_comparison_csv(
    const std::filesystem::path &path, const SummaryStats &analytic_summary,
    const SolverBenchmark &analytic_benchmark,
    const SummaryStats &autodiff_summary,
    const SolverBenchmark &autodiff_benchmark,
    const std::map<std::string, ExternalSolverBenchmark> &external_benchmarks,
    std::uint64_t iterations) {
  std::ofstream file(path);
  if (!file) {
    throw std::runtime_error("Failed to open benchmark CSV path " +
                             path.string());
  }
  file << "solver,derivative,iterations,samples,mean_ms,stddev_ms,mean_lre,"
          "passed,total,mean_function_evaluations,"
          "mean_jacobian_evaluations,mean_linear_solves,mean_accepted_steps,"
          "mean_rejected_steps\n";
  const auto write_levmar =
      [&](std::string_view derivative, const ScalarMoments &timing, double lre,
          Index passed, Index total, double mean_function_evaluations,
          double mean_jacobian_evaluations, double mean_linear_solves,
          double mean_accepted_steps, double mean_rejected_steps) {
        file << "levmar," << derivative << ',' << iterations << ','
             << timing.count << ',' << std::setprecision(17)
             << milliseconds(timing.mean()) << ','
             << milliseconds(timing.stddev()) << ',' << lre << ',' << passed
             << ',' << total << ',' << mean_function_evaluations << ','
             << mean_jacobian_evaluations << ',' << mean_linear_solves << ','
             << mean_accepted_steps << ',' << mean_rejected_steps << '\n';
      };
  write_levmar(
      "dynamic_analytic", analytic_benchmark.dynamic_total,
      analytic_summary.solver_dynamic_lre_sum /
          analytic_summary.solver_start_cases,
      analytic_summary.solver_start_case_passes,
      analytic_summary.solver_start_cases,
      static_cast<double>(
          analytic_summary.solver_dynamic_function_evaluations) /
          analytic_summary.solver_start_cases,
      static_cast<double>(
          analytic_summary.solver_dynamic_jacobian_evaluations) /
          analytic_summary.solver_start_cases,
      static_cast<double>(analytic_summary.solver_dynamic_linear_solves) /
          analytic_summary.solver_start_cases,
      static_cast<double>(analytic_summary.solver_dynamic_accepted_steps) /
          analytic_summary.solver_start_cases,
      static_cast<double>(analytic_summary.solver_dynamic_rejected_steps) /
          analytic_summary.solver_start_cases);
  write_levmar(
      "static_analytic", analytic_benchmark.static_total,
      analytic_summary.solver_static_lre_sum /
          analytic_summary.solver_start_cases,
      analytic_summary.solver_start_case_passes,
      analytic_summary.solver_start_cases,
      static_cast<double>(analytic_summary.solver_static_function_evaluations) /
          analytic_summary.solver_start_cases,
      static_cast<double>(analytic_summary.solver_static_jacobian_evaluations) /
          analytic_summary.solver_start_cases,
      static_cast<double>(analytic_summary.solver_static_linear_solves) /
          analytic_summary.solver_start_cases,
      static_cast<double>(analytic_summary.solver_static_accepted_steps) /
          analytic_summary.solver_start_cases,
      static_cast<double>(analytic_summary.solver_static_rejected_steps) /
          analytic_summary.solver_start_cases);
  write_levmar(
      "dynamic_graph_autodiff", autodiff_benchmark.dynamic_total,
      autodiff_summary.solver_dynamic_lre_sum /
          autodiff_summary.solver_start_cases,
      autodiff_summary.solver_start_case_passes,
      autodiff_summary.solver_start_cases,
      static_cast<double>(
          autodiff_summary.solver_dynamic_function_evaluations) /
          autodiff_summary.solver_start_cases,
      static_cast<double>(
          autodiff_summary.solver_dynamic_jacobian_evaluations) /
          autodiff_summary.solver_start_cases,
      static_cast<double>(autodiff_summary.solver_dynamic_linear_solves) /
          autodiff_summary.solver_start_cases,
      static_cast<double>(autodiff_summary.solver_dynamic_accepted_steps) /
          autodiff_summary.solver_start_cases,
      static_cast<double>(autodiff_summary.solver_dynamic_rejected_steps) /
          autodiff_summary.solver_start_cases);
  write_levmar(
      "static_dual_autodiff", autodiff_benchmark.static_total,
      autodiff_summary.solver_static_lre_sum /
          autodiff_summary.solver_start_cases,
      autodiff_summary.solver_start_case_passes,
      autodiff_summary.solver_start_cases,
      static_cast<double>(autodiff_summary.solver_static_function_evaluations) /
          autodiff_summary.solver_start_cases,
      static_cast<double>(autodiff_summary.solver_static_jacobian_evaluations) /
          autodiff_summary.solver_start_cases,
      static_cast<double>(autodiff_summary.solver_static_linear_solves) /
          autodiff_summary.solver_start_cases,
      static_cast<double>(autodiff_summary.solver_static_accepted_steps) /
          autodiff_summary.solver_start_cases,
      static_cast<double>(autodiff_summary.solver_static_rejected_steps) /
          autodiff_summary.solver_start_cases);
  const auto mean_or_nan = [](const ScalarMoments &moments) {
    return moments.count == 0 ? std::numeric_limits<double>::quiet_NaN()
                              : moments.mean();
  };
  for (const auto &[solver, benchmark] : external_benchmarks) {
    file << solver << ','
         << (solver.ends_with("lmdif")      ? "finite_difference"
             : solver.ends_with("lmder")    ? "analytic"
             : solver.ends_with("autodiff") ? "autodiff"
                                            : "analytic")
         << ',' << iterations << ',' << benchmark.seconds.count << ','
         << std::setprecision(17) << milliseconds(benchmark.seconds.mean())
         << ',' << milliseconds(benchmark.seconds.stddev()) << ','
         << benchmark.lre.mean() << ',' << benchmark.usable << ','
         << benchmark.cases << ',' << benchmark.function_evaluations.mean()
         << ',' << mean_or_nan(benchmark.jacobian_evaluations) << ','
         << mean_or_nan(benchmark.linear_solves) << ','
         << mean_or_nan(benchmark.accepted_steps) << ','
         << mean_or_nan(benchmark.rejected_steps) << '\n';
  }
}

double frozen_linearization_objective(
    const LMWorkspace<std::dynamic_extent, std::dynamic_extent> &work,
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
    const double damping = std::min(
        work.damping_scale[j] * work.damping_scale[j], kMaxLMDampingDiagonal);
    objective += lambda * damping * work.step[j] * work.step[j];
  }
  return objective;
}

void run_frozen_linearization(const std::filesystem::path &problem_dir) {
  const auto corpus = load_problem(problem_dir);
  const auto jacobian = read_numeric_csv(problem_dir / "jacobian_start1.csv");
  const auto residuals = read_numeric_csv(problem_dir / "residuals_start1.csv");
  if (jacobian.size() != corpus.m || residuals.size() != corpus.m) {
    throw std::runtime_error("frozen linearization fixture shape mismatch");
  }

  LMWorkspace<std::dynamic_extent, std::dynamic_extent> damped_work(corpus.m,
                                                                    corpus.n);
  LMWorkspace<std::dynamic_extent, std::dynamic_extent> pivoted_work(corpus.m,
                                                                     corpus.n);
  DampedQrWorkspace<std::dynamic_extent, std::dynamic_extent> damped_qr;
  PivotedHouseholderQrWorkspace<std::dynamic_extent, std::dynamic_extent>
      pivoted_qr;
  damped_qr.resize(corpus.m, corpus.n);
  pivoted_qr.resize(corpus.m, corpus.n);

  for (auto *work : {&damped_work, &pivoted_work}) {
    for (Index i = 0; i < corpus.m; ++i) {
      if (jacobian[i].size() != corpus.n || residuals[i].size() != 1) {
        throw std::runtime_error("frozen linearization CSV row shape mismatch");
      }
      work->r[i] = residuals[i][0];
      for (Index j = 0; j < corpus.n; ++j) {
        work->J(i, j) = jacobian[i][j];
      }
    }
    if (!update_parameter_scaling(JacobianColumnScaling{}, *work)) {
      throw std::runtime_error("frozen linearization scaling failed");
    }
  }

  if (!prepare_pivoted_householder_qr(JacobianColumnScaling{}, pivoted_work,
                                      pivoted_qr, 32.0)) {
    throw std::runtime_error(
        "frozen linearization cached factorization failed");
  }

  std::cout << "problem,lambda,max_abs_step_error,max_rel_step_error,"
               "objective_abs_error\n";
  for (double lambda : {1e-16, 1e-12, 1e-8, 1e-4, 1.0}) {
    if (!solve_damped_qr(JacobianColumnScaling{}, damped_work, damped_qr,
                         lambda) ||
        !solve_pivoted_householder_qr(JacobianColumnScaling{}, pivoted_work,
                                      pivoted_qr, lambda)) {
      throw std::runtime_error("frozen linearization solve failed");
    }

    double max_abs_step_error = 0.0;
    double max_rel_step_error = 0.0;
    for (Index j = 0; j < corpus.n; ++j) {
      const double abs_error =
          std::abs(pivoted_work.step[j] - damped_work.step[j]);
      max_abs_step_error = std::max(max_abs_step_error, abs_error);
      max_rel_step_error =
          std::max(max_rel_step_error,
                   abs_error / std::max(std::abs(damped_work.step[j]), 1e-16));
    }
    const double objective_abs_error =
        std::abs(frozen_linearization_objective(pivoted_work, lambda) -
                 frozen_linearization_objective(damped_work, lambda));
    std::cout << corpus.name << ',' << std::setprecision(17) << lambda << ','
              << max_abs_step_error << ',' << max_rel_step_error << ','
              << objective_abs_error << '\n';
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::filesystem::path corpus_dir;
    std::filesystem::path csv_path;
    std::filesystem::path benchmark_csv_path;
    std::filesystem::path solver_work_csv_path;
    std::filesystem::path controller_trace_csv_path;
    if (argc > 1) {
      corpus_dir = argv[1];
    } else {
      corpus_dir = std::filesystem::path(__FILE__).parent_path() / "corpus";
    }
    if (argc > 2) {
      csv_path = argv[2];
    } else {
      csv_path = std::filesystem::path(__FILE__).parent_path() /
                 "cpp_runner_results.csv";
    }
    std::uint64_t benchmark_iterations = 40;
    if (argc > 3) {
      benchmark_iterations = static_cast<std::uint64_t>(std::stoull(argv[3]));
    }
    bool solve_all_starts = false;
    bool solve_all_autodiff = false;
    bool solve_all_external = false;
    bool benchmark_solvers = false;
    bool validation_only = false;
    bool solver_work_external = false;
    bool relative_cost_termination = true;
    std::filesystem::path frozen_linearization_problem_dir;
#ifdef LEVMAR_NIST_DEFAULT_PIVOTED_HOUSEHOLDER_QR
    std::string_view linear_algebra = "pivoted-householder-qr";
#else
    std::string_view linear_algebra = "damped-qr";
#endif
    for (int i = 4; i < argc; ++i) {
      if (std::string_view(argv[i]) == "--solve-all") {
        solve_all_starts = true;
      } else if (std::string_view(argv[i]) == "--solve-all-autodiff") {
        solve_all_starts = true;
        solve_all_autodiff = true;
      } else if (std::string_view(argv[i]) == "--solve-all-external") {
        solve_all_starts = true;
        solve_all_external = true;
      } else if (std::string_view(argv[i]) == "--benchmark-solvers") {
        benchmark_solvers = true;
        solve_all_autodiff = true;
        solve_all_external = true;
      } else if (std::string_view(argv[i]) == "--validate") {
        validation_only = true;
      } else if (std::string_view(argv[i]) == "--solver-work-csv") {
        if (++i == argc) {
          throw std::runtime_error("--solver-work-csv requires a path");
        }
        solver_work_csv_path = argv[i];
      } else if (std::string_view(argv[i]) == "--solver-work-external") {
        solver_work_external = true;
      } else if (std::string_view(argv[i]) == "--controller-trace") {
        if (++i == argc) {
          throw std::runtime_error("--controller-trace requires a path");
        }
        controller_trace_csv_path = argv[i];
      } else if (std::string_view(argv[i]) == "--frozen-linearization") {
        if (++i == argc) {
          throw std::runtime_error(
              "--frozen-linearization requires a problem directory");
        }
        frozen_linearization_problem_dir = argv[i];
      } else if (std::string_view(argv[i]) == "--linear-algebra") {
        if (++i == argc) {
          throw std::runtime_error("--linear-algebra requires a backend");
        }
        linear_algebra = argv[i];
        if (linear_algebra != "damped-qr" &&
            linear_algebra != "pivoted-householder-qr") {
          throw std::runtime_error(
              "--linear-algebra must be damped-qr or pivoted-householder-qr");
        }
      } else if (std::string_view(argv[i]) ==
                 "--disable-relative-cost-termination") {
        relative_cost_termination = false;
      } else if (std::string_view(argv[i]).starts_with("--")) {
        throw std::runtime_error("Unknown option " + std::string(argv[i]));
      } else {
        benchmark_csv_path = argv[i];
      }
    }

    std::vector<std::filesystem::path> problem_dirs;
    for (const auto &entry : std::filesystem::directory_iterator(corpus_dir)) {
      if (entry.is_directory()) {
        problem_dirs.push_back(entry.path());
      }
    }

    std::ranges::sort(problem_dirs);

#ifdef LEVMAR_NIST_EXTERNAL_ONLY
    if (linear_algebra != "damped-qr") {
      throw std::runtime_error(
          "--linear-algebra is available only in levmar_nist_runner");
    }
    if (!controller_trace_csv_path.empty()) {
      throw std::runtime_error(
          "Controller traces are available only in levmar_nist_runner");
    }
    if (solver_work_csv_path.empty()) {
      throw std::runtime_error(
          "External runners support only --solver-work-csv export");
    }
    std::vector<ProblemReport> external_reports;
    for (const auto &path : problem_dirs) {
      external_reports.push_back(run_problem<NistAnalyticSolverPolicy>(
          path, true, false, 0, true, false, relative_cost_termination));
    }
    write_solver_work_csv(solver_work_csv_path, external_reports, {}, false);
    return 0;
#else
    if (!frozen_linearization_problem_dir.empty()) {
      run_frozen_linearization(frozen_linearization_problem_dir);
      return 0;
    }

    if (!controller_trace_csv_path.empty()) {
      std::vector<ControllerTraceRecord> controller_traces;
      const auto run_controller_trace = [&]<class LinearAlgebra>() {
        for (const auto &path : problem_dirs) {
          run_problem<NistAnalyticSolverPolicyFor<LinearAlgebra>>(
              path, true, false, 0, false, false, relative_cost_termination,
              &controller_traces);
          run_problem<NistAutoDiffSolverPolicyFor<LinearAlgebra>>(
              path, true, false, 0, false, false, relative_cost_termination,
              &controller_traces);
        }
      };
      if (linear_algebra == "pivoted-householder-qr") {
        run_controller_trace.template operator()<PivotedHouseholderQr>();
      } else {
        run_controller_trace.template operator()<DampedQr>();
      }
      write_controller_trace_csv(controller_trace_csv_path, controller_traces);
      return 0;
    }

    if (!solver_work_csv_path.empty()) {
      std::vector<ProblemReport> analytic_reports;
      std::vector<ProblemReport> autodiff_reports;
      const auto run_solver_work = [&]<class LinearAlgebra>() {
        for (const auto &path : problem_dirs) {
          auto analytic_report =
              run_problem<NistAnalyticSolverPolicyFor<LinearAlgebra>>(
                  path, true, false, 0, solver_work_external, false,
                  relative_cost_termination);
          analytic_reports.push_back(std::move(analytic_report));
          if (!solver_work_external) {
            auto autodiff_report =
                run_problem<NistAutoDiffSolverPolicyFor<LinearAlgebra>>(
                    path, true, false, 0, false, false,
                    relative_cost_termination);
            autodiff_reports.push_back(std::move(autodiff_report));
          }
        }
      };
      if (linear_algebra == "pivoted-householder-qr") {
        run_solver_work.template operator()<PivotedHouseholderQr>();
      } else {
        run_solver_work.template operator()<DampedQr>();
      }
      write_solver_work_csv(solver_work_csv_path, analytic_reports,
                            autodiff_reports, !solver_work_external,
                            linear_algebra == "pivoted-householder-qr"
                                ? "pivoted_householder_qr"
                                : "damped_qr");
      return 0;
    }

    if (benchmark_solvers) {
      SummaryStats analytic_summary;
      SummaryStats autodiff_summary;
      for (const auto &path : problem_dirs) {
        merge_summary(analytic_summary, run_problem<NistAnalyticSolverPolicy>(
                                            path, true, false, 0, false, false,
                                            relative_cost_termination));
        merge_summary(autodiff_summary, run_problem<NistAutoDiffSolverPolicy>(
                                            path, true, false, 0, false, false,
                                            relative_cost_termination));
      }
      const auto analytic_benchmark =
          benchmark_solver<NistAnalyticSolverPolicy>(
              problem_dirs, benchmark_iterations, relative_cost_termination);
      const auto autodiff_benchmark =
          benchmark_solver<NistAutoDiffSolverPolicy>(
              problem_dirs, benchmark_iterations, relative_cost_termination);
      print_solver_comparison(analytic_summary, analytic_benchmark,
                              autodiff_summary, autodiff_benchmark,
                              benchmark_iterations);
      const auto external_benchmarks =
          benchmark_external_solvers(problem_dirs, benchmark_iterations);
      if (external_benchmarks.empty()) {
        throw std::runtime_error(
            "External solver benchmarks are unavailable; configure with "
            "-DLEVMAR_BUILD_EXTERNAL_BENCHMARKS=ON and install Ceres or "
            "CMinpack");
      }
      print_external_solver_benchmarks(external_benchmarks);
      if (!benchmark_csv_path.empty()) {
        write_solver_comparison_csv(benchmark_csv_path, analytic_summary,
                                    analytic_benchmark, autodiff_summary,
                                    autodiff_benchmark, external_benchmarks,
                                    benchmark_iterations);
      }
      return 0;
    }

    SummaryStats summary;
    std::vector<ProblemReport> reports;
    for (const auto &path : problem_dirs) {
      auto report = run_problem<NistAnalyticSolverPolicy>(
          path, solve_all_starts, false, 0, false, true,
          relative_cost_termination);
      print_problem_report(report);
      merge_summary(summary, report);
      reports.push_back(report);
    }
    print_summary(summary);
    write_csv_report(csv_path, reports, summary);
    if (validation_only) {
      std::cout << "all corpus checks passed\n";
      return 0;
    }
    if (solve_all_starts) {
      const auto solver_benchmark = benchmark_solver<NistAnalyticSolverPolicy>(
          problem_dirs, benchmark_iterations);
      print_solver_benchmark(solver_benchmark, benchmark_iterations,
                             summary.solver_start_cases);
      if (!benchmark_csv_path.empty()) {
        write_solver_benchmark_csv(benchmark_csv_path, solver_benchmark,
                                   benchmark_iterations);
      }
      if (solve_all_autodiff) {
        SummaryStats autodiff_summary;
        for (const auto &path : problem_dirs) {
          merge_summary(autodiff_summary,
                        run_problem<NistAutoDiffSolverPolicy>(path, true));
        }
        const auto autodiff_benchmark =
            benchmark_solver<NistAutoDiffSolverPolicy>(problem_dirs,
                                                       benchmark_iterations);
        print_solver_comparison(summary, solver_benchmark, autodiff_summary,
                                autodiff_benchmark, benchmark_iterations);
        if (!autodiff_summary.solver_failures.empty()) {
          return 1;
        }
      }
      if (solve_all_external) {
        const auto external_benchmarks =
            benchmark_external_solvers(problem_dirs, benchmark_iterations);
        if (external_benchmarks.empty()) {
          throw std::runtime_error(
              "External solver benchmarks are unavailable; configure with "
              "-DLEVMAR_BUILD_EXTERNAL_BENCHMARKS=ON and install Ceres or "
              "CMinpack");
        }
        print_external_solver_benchmarks(external_benchmarks);
        if (!benchmark_csv_path.empty()) {
          write_external_solver_benchmark_csv(
              benchmark_csv_path, external_benchmarks, benchmark_iterations);
        }
      }
      if (!summary.solver_failures.empty()) {
        return 1;
      }
      return 0;
    }

    std::vector<ProblemBenchmark> problem_benchmarks;
    const auto summary_benchmark = benchmark_summary(
        problem_dirs, benchmark_iterations, problem_benchmarks);
    for (const auto &benchmark : problem_benchmarks) {
      print_problem_benchmark(benchmark);
    }
    print_summary_benchmark(summary_benchmark, benchmark_iterations);
    if (!benchmark_csv_path.empty()) {
      write_benchmark_csv(benchmark_csv_path, summary_benchmark,
                          problem_benchmarks, benchmark_iterations);
    }
#endif
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }

  std::cout << "all corpus checks passed\n";
  return 0;
}
