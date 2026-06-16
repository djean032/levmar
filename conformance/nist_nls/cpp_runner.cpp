#include <levmar/lm.h>

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
#include <vector>

namespace {

struct CorpusProblem {
  std::string name;
  std::string model_id;
  std::string model_class;
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
  double residual_seconds = 0.0;
  double analytic_jacobian_seconds = 0.0;
  double forward_difference_seconds = 0.0;
  double central_difference_seconds = 0.0;
  double total_seconds = 0.0;
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
    const double variance = std::max(0.0, sum_sq / count - mean_value * mean_value);
    return std::sqrt(variance);
  }
};

struct TimingMoments {
  ScalarMoments residual;
  ScalarMoments analytic;
  ScalarMoments forward_difference;
  ScalarMoments central_difference;
  ScalarMoments total;

  void add(const TimingStats &timing) {
    residual.add(timing.residual_seconds);
    analytic.add(timing.analytic_jacobian_seconds);
    forward_difference.add(timing.forward_difference_seconds);
    central_difference.add(timing.central_difference_seconds);
    total.add(timing.total_seconds);
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

struct ProblemReport {
  std::string name;
  bool numerical_derivatives_skipped = false;
  TimingStats dynamic_timing;
  TimingStats static_timing;
  ComparisonStats residual_stats;
  ComparisonStats analytic_stats;
  ComparisonStats forward_difference_stats;
  ComparisonStats central_difference_stats;
};

struct SummaryStats {
  std::uint64_t problems = 0;
  std::uint64_t numerical_derivative_skips = 0;
  TimingStats dynamic_timing;
  TimingStats static_timing;
  ComparisonStats residual_stats;
  ComparisonStats analytic_stats;
  ComparisonStats forward_difference_stats;
  ComparisonStats central_difference_stats;
};

using Clock = std::chrono::steady_clock;
constexpr std::uint64_t kKernelTimingRepeats = 64;

double milliseconds(double seconds) { return seconds * 1000.0; }

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

std::map<std::string, std::string> read_meta(const std::filesystem::path &path) {
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

std::vector<std::vector<double>> read_numeric_csv(const std::filesystem::path &path) {
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
  problem.m = static_cast<Index>(std::stoull(meta.at("m")));
  problem.n = static_cast<Index>(std::stoull(meta.at("n")));
  problem.predictor_count = static_cast<Index>(std::stoull(meta.at("predictor_count")));
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
  summary.dynamic_timing.residual_seconds += report.dynamic_timing.residual_seconds;
  summary.dynamic_timing.analytic_jacobian_seconds +=
      report.dynamic_timing.analytic_jacobian_seconds;
  summary.dynamic_timing.forward_difference_seconds +=
      report.dynamic_timing.forward_difference_seconds;
  summary.dynamic_timing.central_difference_seconds +=
      report.dynamic_timing.central_difference_seconds;
  summary.dynamic_timing.total_seconds += report.dynamic_timing.total_seconds;
  summary.static_timing.residual_seconds += report.static_timing.residual_seconds;
  summary.static_timing.analytic_jacobian_seconds +=
      report.static_timing.analytic_jacobian_seconds;
  summary.static_timing.forward_difference_seconds +=
      report.static_timing.forward_difference_seconds;
  summary.static_timing.central_difference_seconds +=
      report.static_timing.central_difference_seconds;
  summary.static_timing.total_seconds += report.static_timing.total_seconds;
  merge_stats(summary.residual_stats, report.residual_stats);
  merge_stats(summary.analytic_stats, report.analytic_stats);
  merge_stats(summary.forward_difference_stats, report.forward_difference_stats);
  merge_stats(summary.central_difference_stats, report.central_difference_stats);
}

template <Index M, Index N, class ResidualFn, class JacobianFn>
void run_kernel_variant(const CorpusProblem &corpus,
                        ResidualFn residual,
                        JacobianFn jacobian,
                        ConstVectorView<N> beta,
                        const std::vector<std::vector<double>> &expected_residuals,
                        const std::vector<std::vector<double>> &expected_jacobian,
                        bool run_numerical_derivatives,
                        TimingStats &timing,
                        ComparisonStats *residual_stats,
                        ComparisonStats *analytic_stats,
                        ComparisonStats *forward_difference_stats,
                        ComparisonStats *central_difference_stats,
                        const std::string &what_prefix) {
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

  using ContextType = LMSolveContext<M, N, decltype(residual), decltype(jacobian)>;

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

  options.jacobian_mode = JacobianMode::User;
  timing.analytic_jacobian_seconds += measure_average_seconds(
      kKernelTimingRepeats, [&] {
        std::ranges::copy(context.x, workspace.x_current.view().begin());
        if (auto jacobian_result =
                evaluate_jacobian(context, what_prefix + " analytic jacobian");
            !jacobian_result) {
          throw std::runtime_error(jacobian_result.error().message);
        }
      });
  for (Index j = 0; j < corpus.n; ++j) {
    for (Index i = 0; i < corpus.m; ++i) {
      if (analytic_stats != nullptr) {
        analytic_stats->add(workspace.J(i, j), expected_jacobian[i][j]);
      }
      expect_close(workspace.J(i, j), expected_jacobian[i][j], 1e-12, 1e-10,
                   what_prefix + " analytic jacobian row " +
                       std::to_string(i) + " col " + std::to_string(j));
    }
  }

  if (!run_numerical_derivatives) {
    timing.total_seconds = timing.residual_seconds + timing.analytic_jacobian_seconds +
                           timing.forward_difference_seconds + timing.central_difference_seconds;
    return;
  }

  options.jacobian_mode = JacobianMode::ForwardDifference;
  timing.forward_difference_seconds += measure_average_seconds(
      kKernelTimingRepeats, [&] {
        std::ranges::copy(context.x, workspace.x_current.view().begin());
        if (auto jacobian_result =
                evaluate_jacobian(context, what_prefix + " fd jacobian");
            !jacobian_result) {
          throw std::runtime_error(jacobian_result.error().message);
        }
      });
  for (Index j = 0; j < corpus.n; ++j) {
    for (Index i = 0; i < corpus.m; ++i) {
      if (forward_difference_stats != nullptr) {
        forward_difference_stats->add(workspace.J(i, j), expected_jacobian[i][j]);
      }
      expect_close(workspace.J(i, j), expected_jacobian[i][j], 1e-7, 2e-3,
                   what_prefix + " fd jacobian row " + std::to_string(i) +
                       " col " + std::to_string(j));
    }
  }

  options.jacobian_mode = JacobianMode::CentralDifference;
  timing.central_difference_seconds += measure_average_seconds(
      kKernelTimingRepeats, [&] {
        std::ranges::copy(context.x, workspace.x_current.view().begin());
        if (auto jacobian_result =
                evaluate_jacobian(context, what_prefix + " central jacobian");
            !jacobian_result) {
          throw std::runtime_error(jacobian_result.error().message);
        }
      });
  for (Index j = 0; j < corpus.n; ++j) {
    for (Index i = 0; i < corpus.m; ++i) {
      if (central_difference_stats != nullptr) {
        central_difference_stats->add(workspace.J(i, j), expected_jacobian[i][j]);
      }
      expect_close(workspace.J(i, j), expected_jacobian[i][j], 1e-8, 5e-4,
                   what_prefix + " central jacobian row " + std::to_string(i) +
                       " col " + std::to_string(j));
    }
  }

  timing.total_seconds = timing.residual_seconds + timing.analytic_jacobian_seconds +
                         timing.forward_difference_seconds + timing.central_difference_seconds;
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
  throw std::runtime_error("No fully static dispatch available for problem dimensions " +
                           std::to_string(corpus.m) + "x" + std::to_string(corpus.n));
}

ProblemReport run_problem(const std::filesystem::path &problem_dir) {
  const auto corpus = load_problem(problem_dir);
  ProblemReport report;
  report.name = corpus.name;
  report.numerical_derivatives_skipped = !corpus.numerical_derivatives_recommended;

  std::vector<double> beta_storage(corpus.n, 0.0);

  for (const auto &[label, params] : corpus.params) {
    beta_storage = params;
    const auto expected_residuals =
        read_numeric_csv(problem_dir / ("residuals_" + label + ".csv"));
    const auto expected_jacobian =
        read_numeric_csv(problem_dir / ("jacobian_" + label + ".csv"));

    const bool run_numerical_derivatives =
        corpus.numerical_derivatives_recommended &&
        (label == "certified" || label == "benchmark");

    auto run_pair = [&]<Index SM, Index SN>(auto dynamic_residual,
                                            auto dynamic_jacobian,
                                            auto static_residual,
                                            auto static_jacobian) {
      run_kernel_variant<std::dynamic_extent, std::dynamic_extent>(
          corpus, dynamic_residual, dynamic_jacobian,
          ConstVectorView<std::dynamic_extent>(beta_storage.data(),
                                               beta_storage.size()),
          expected_residuals, expected_jacobian, run_numerical_derivatives,
          report.dynamic_timing, &report.residual_stats, &report.analytic_stats,
          &report.forward_difference_stats, &report.central_difference_stats,
          corpus.name + " dynamic " + label);

      run_kernel_variant<SM, SN>(
          corpus, static_residual, static_jacobian,
          ConstVectorView<SN>(beta_storage.data(), beta_storage.size()),
          expected_residuals, expected_jacobian, run_numerical_derivatives,
          report.static_timing, nullptr, nullptr, nullptr, nullptr,
          corpus.name + " static " + label);
    };

    if (corpus.model_id == "bennett5") {
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          r[i] = x[0] * std::pow(x[1] + xv, -1.0 / x[2]) - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
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
      auto residual_static = [&](ConstVectorView<3> x, VectorView<154> r) -> ErrorOrVoid {
        for (Index i = 0; i < 154; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          r[i] = x[0] * std::pow(x[1] + xv, -1.0 / x[2]) - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<3> x, MatrixView<154, 3> J) -> ErrorOrVoid {
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
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(-x[1] * xv);
          r[i] = x[0] * (1.0 - e) - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(-x[1] * xv);
          J[i, 0] = 1.0 - e;
          J[i, 1] = x[0] * xv * e;
        }
        return {};
      };
      auto residual_static = [&](ConstVectorView<2> x, VectorView<6> r) -> ErrorOrVoid {
        for (Index i = 0; i < 6; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(-x[1] * xv);
          r[i] = x[0] * (1.0 - e) - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<2> x, MatrixView<6, 2> J) -> ErrorOrVoid {
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
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(-x[1] * xv);
          r[i] = x[0] * (1.0 - e) - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(-x[1] * xv);
          J[i, 0] = 1.0 - e;
          J[i, 1] = x[0] * xv * e;
        }
        return {};
      };
      auto residual_static = [&](ConstVectorView<2> x, VectorView<14> r) -> ErrorOrVoid {
        for (Index i = 0; i < 14; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(-x[1] * xv);
          r[i] = x[0] * (1.0 - e) - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<2> x, MatrixView<14, 2> J) -> ErrorOrVoid {
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
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(-x[0] * xv);
          const double d = x[1] + x[2] * xv;
          r[i] = e / d - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
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
      auto residual_static = [&](ConstVectorView<3> x, VectorView<214> r) -> ErrorOrVoid {
        for (Index i = 0; i < 214; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(-x[0] * xv);
          const double d = x[1] + x[2] * xv;
          r[i] = e / d - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<3> x, MatrixView<214, 3> J) -> ErrorOrVoid {
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
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(-x[0] * xv);
          const double d = x[1] + x[2] * xv;
          r[i] = e / d - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
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
      auto residual_static = [&](ConstVectorView<3> x, VectorView<54> r) -> ErrorOrVoid {
        for (Index i = 0; i < 54; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(-x[0] * xv);
          const double d = x[1] + x[2] * xv;
          r[i] = e / d - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<3> x, MatrixView<54, 3> J) -> ErrorOrVoid {
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
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          double value = 0.0;
          for (Index j = 0; j < 6; j += 2) {
            value += x[j] * std::exp(-x[j + 1] * xv);
          }
          r[i] = value - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
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
      auto residual_static = [&](ConstVectorView<6> x, VectorView<24> r) -> ErrorOrVoid {
        for (Index i = 0; i < 24; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          double value = 0.0;
          for (Index j = 0; j < 6; j += 2) {
            value += x[j] * std::exp(-x[j + 1] * xv);
          }
          r[i] = value - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<6> x, MatrixView<24, 6> J) -> ErrorOrVoid {
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
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double d1 = xv - x[3];
          const double d2 = xv - x[6];
          r[i] = x[0] * std::exp(-x[1] * xv) +
                 x[2] * std::exp(-(d1 * d1) / (x[4] * x[4])) +
                 x[5] * std::exp(-(d2 * d2) / (x[7] * x[7])) - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
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
      auto residual_static = [&](ConstVectorView<8> x, VectorView<250> r) -> ErrorOrVoid {
        for (Index i = 0; i < 250; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double d1 = xv - x[3];
          const double d2 = xv - x[6];
          r[i] = x[0] * std::exp(-x[1] * xv) +
                 x[2] * std::exp(-(d1 * d1) / (x[4] * x[4])) +
                 x[5] * std::exp(-(d2 * d2) / (x[7] * x[7])) - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<8> x, MatrixView<250, 8> J) -> ErrorOrVoid {
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
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          r[i] = x[0] * std::pow(row[0], x[1]) - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double x_pow = std::pow(xv, x[1]);
          J[i, 0] = x_pow;
          J[i, 1] = x[0] * x_pow * std::log(xv);
        }
        return {};
      };
      auto residual_static = [&](ConstVectorView<2> x, VectorView<6> r) -> ErrorOrVoid {
        for (Index i = 0; i < 6; ++i) {
          const auto &row = corpus.data[i];
          r[i] = x[0] * std::pow(row[0], x[1]) - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<2> x, MatrixView<6, 2> J) -> ErrorOrVoid {
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
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double t = 1.0 + x[1] * xv / 2.0;
          r[i] = x[0] * (1.0 - std::pow(t, -2.0)) - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double t = 1.0 + x[1] * xv / 2.0;
          J[i, 0] = 1.0 - std::pow(t, -2.0);
          J[i, 1] = x[0] * xv * std::pow(t, -3.0);
        }
        return {};
      };
      auto residual_static = [&](ConstVectorView<2> x, VectorView<14> r) -> ErrorOrVoid {
        for (Index i = 0; i < 14; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double t = 1.0 + x[1] * xv / 2.0;
          r[i] = x[0] * (1.0 - std::pow(t, -2.0)) - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<2> x, MatrixView<14, 2> J) -> ErrorOrVoid {
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
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double x2 = xv * xv;
          const double n = x[0] + x[1] * xv + x[2] * x2;
          const double d = 1.0 + x[3] * xv + x[4] * x2;
          r[i] = n / d - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
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
      auto residual_static = [&](ConstVectorView<5> x, VectorView<151> r) -> ErrorOrVoid {
        for (Index i = 0; i < 151; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double x2 = xv * xv;
          const double n = x[0] + x[1] * xv + x[2] * x2;
          const double d = 1.0 + x[3] * xv + x[4] * x2;
          r[i] = n / d - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<5> x, MatrixView<151, 5> J) -> ErrorOrVoid {
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
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double x2 = xv * xv;
          const double x3 = x2 * xv;
          const double n = x[0] + x[1] * xv + x[2] * x2 + x[3] * x3;
          const double d = 1.0 + x[4] * xv + x[5] * x2 + x[6] * x3;
          r[i] = n / d - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
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
      auto residual_static = [&](ConstVectorView<7> x, VectorView<236> r) -> ErrorOrVoid {
        for (Index i = 0; i < 236; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double x2 = xv * xv;
          const double x3 = x2 * xv;
          const double n = x[0] + x[1] * xv + x[2] * x2 + x[3] * x3;
          const double d = 1.0 + x[4] * xv + x[5] * x2 + x[6] * x3;
          r[i] = n / d - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<7> x, MatrixView<236, 7> J) -> ErrorOrVoid {
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
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double x2 = xv * xv;
          const double x3 = x2 * xv;
          const double n = x[0] + x[1] * xv + x[2] * x2 + x[3] * x3;
          const double d = 1.0 + x[4] * xv + x[5] * x2 + x[6] * x3;
          r[i] = n / d - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
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
      auto residual_static = [&](ConstVectorView<7> x, VectorView<37> r) -> ErrorOrVoid {
        for (Index i = 0; i < 37; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double x2 = xv * xv;
          const double x3 = x2 * xv;
          const double n = x[0] + x[1] * xv + x[2] * x2 + x[3] * x3;
          const double d = 1.0 + x[4] * xv + x[5] * x2 + x[6] * x3;
          r[i] = n / d - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<7> x, MatrixView<37, 7> J) -> ErrorOrVoid {
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
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double x1 = row[0];
          const double x2 = row[1];
          r[i] = x[0] - x[1] * x1 * std::exp(-x[2] * x2) - std::log(row.back());
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
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
      auto residual_static = [&](ConstVectorView<3> x, VectorView<128> r) -> ErrorOrVoid {
        for (Index i = 0; i < 128; ++i) {
          const auto &row = corpus.data[i];
          const double x1 = row[0];
          const double x2 = row[1];
          r[i] = x[0] - x[1] * x1 * std::exp(-x[2] * x2) - std::log(row.back());
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<3> x, MatrixView<128, 3> J) -> ErrorOrVoid {
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
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          r[i] = x[0] + x[1] * std::exp(-x[3] * xv) +
                 x[2] * std::exp(-x[4] * xv) - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
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
      auto residual_static = [&](ConstVectorView<5> x, VectorView<33> r) -> ErrorOrVoid {
        for (Index i = 0; i < 33; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          r[i] = x[0] + x[1] * std::exp(-x[3] * xv) +
                 x[2] * std::exp(-x[4] * xv) - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<5> x, MatrixView<33, 5> J) -> ErrorOrVoid {
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
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double t = 1.0 + 2.0 * x[1] * xv;
          r[i] = x[0] * (1.0 - 1.0 / std::sqrt(t)) - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double t = 1.0 + 2.0 * x[1] * xv;
          J[i, 0] = 1.0 - 1.0 / std::sqrt(t);
          J[i, 1] = x[0] * xv * std::pow(t, -1.5);
        }
        return {};
      };
      auto residual_static = [&](ConstVectorView<2> x, VectorView<14> r) -> ErrorOrVoid {
        for (Index i = 0; i < 14; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double t = 1.0 + 2.0 * x[1] * xv;
          r[i] = x[0] * (1.0 - 1.0 / std::sqrt(t)) - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<2> x, MatrixView<14, 2> J) -> ErrorOrVoid {
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
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double d = 1.0 + x[1] * xv;
          r[i] = (x[0] * x[1] * xv) / d - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
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
      auto residual_static = [&](ConstVectorView<2> x, VectorView<14> r) -> ErrorOrVoid {
        for (Index i = 0; i < 14; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double d = 1.0 + x[1] * xv;
          r[i] = (x[0] * x[1] * xv) / d - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<2> x, MatrixView<14, 2> J) -> ErrorOrVoid {
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
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          r[i] = x[0] - x[1] * xv -
                 std::atan(x[2] / (xv - x[3])) / std::numbers::pi - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
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
      auto residual_static = [&](ConstVectorView<4> x, VectorView<25> r) -> ErrorOrVoid {
        for (Index i = 0; i < 25; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          r[i] = x[0] - x[1] * xv -
                 std::atan(x[2] / (xv - x[3])) / std::numbers::pi - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<4> x, MatrixView<25, 4> J) -> ErrorOrVoid {
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
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double annual = 2.0 * std::numbers::pi * xv / 12.0;
          const double p4 = 2.0 * std::numbers::pi * xv / x[3];
          const double p7 = 2.0 * std::numbers::pi * xv / x[6];
          const double value = x[0] + x[1] * std::cos(annual) + x[2] * std::sin(annual) +
                               x[4] * std::cos(p4) + x[5] * std::sin(p4) +
                               x[7] * std::cos(p7) + x[8] * std::sin(p7);
          r[i] = value - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
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
      auto residual_static = [&](ConstVectorView<9> x, VectorView<168> r) -> ErrorOrVoid {
        for (Index i = 0; i < 168; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double annual = 2.0 * std::numbers::pi * xv / 12.0;
          const double p4 = 2.0 * std::numbers::pi * xv / x[3];
          const double p7 = 2.0 * std::numbers::pi * xv / x[6];
          const double value = x[0] + x[1] * std::cos(annual) + x[2] * std::sin(annual) +
                               x[4] * std::cos(p4) + x[5] * std::sin(p4) +
                               x[7] * std::cos(p7) + x[8] * std::sin(p7);
          r[i] = value - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<9> x, MatrixView<168, 9> J) -> ErrorOrVoid {
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
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double num = xv * xv + x[1] * xv;
          const double den = xv * xv + x[2] * xv + x[3];
          r[i] = x[0] * num / den - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
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
      auto residual_static = [&](ConstVectorView<4> x, VectorView<11> r) -> ErrorOrVoid {
        for (Index i = 0; i < 11; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double num = xv * xv + x[1] * xv;
          const double den = xv * xv + x[2] * xv + x[3];
          r[i] = x[0] * num / den - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<4> x, MatrixView<11, 4> J) -> ErrorOrVoid {
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
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(x[1] - x[2] * xv);
          r[i] = x[0] / (1.0 + e) - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
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
      auto residual_static = [&](ConstVectorView<3> x, VectorView<9> r) -> ErrorOrVoid {
        for (Index i = 0; i < 9; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(x[1] - x[2] * xv);
          r[i] = x[0] / (1.0 + e) - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<3> x, MatrixView<9, 3> J) -> ErrorOrVoid {
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
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(x[1] / (xv + x[2]));
          r[i] = x[0] * e - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
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
      auto residual_static = [&](ConstVectorView<3> x, VectorView<16> r) -> ErrorOrVoid {
        for (Index i = 0; i < 16; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(x[1] / (xv + x[2]));
          r[i] = x[0] * e - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<3> x, MatrixView<16, 3> J) -> ErrorOrVoid {
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
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double delta = xv - x[2];
          const double e = std::exp(-(delta * delta) / (2.0 * x[1] * x[1]));
          r[i] = (x[0] / x[1]) * e - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double b1 = x[0];
          const double b2 = x[1];
          const double b3 = x[2];
          const double delta = xv - b3;
          const double e = std::exp(-(delta * delta) / (2.0 * b2 * b2));
          J[i, 0] = e / b2;
          J[i, 1] = b1 * e * (delta * delta / (b2 * b2 * b2 * b2) - 1.0 / (b2 * b2));
          J[i, 2] = b1 * e * delta / (b2 * b2 * b2);
        }
        return {};
      };
      auto residual_static = [&](ConstVectorView<3> x, VectorView<35> r) -> ErrorOrVoid {
        for (Index i = 0; i < 35; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double delta = xv - x[2];
          const double e = std::exp(-(delta * delta) / (2.0 * x[1] * x[1]));
          r[i] = (x[0] / x[1]) * e - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<3> x, MatrixView<35, 3> J) -> ErrorOrVoid {
        for (Index i = 0; i < 35; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double b1 = x[0];
          const double b2 = x[1];
          const double b3 = x[2];
          const double delta = xv - b3;
          const double e = std::exp(-(delta * delta) / (2.0 * b2 * b2));
          J[i, 0] = e / b2;
          J[i, 1] = b1 * e * (delta * delta / (b2 * b2 * b2 * b2) - 1.0 / (b2 * b2));
          J[i, 2] = b1 * e * delta / (b2 * b2 * b2);
        }
        return {};
      };
      run_pair.template operator()<35, 3>(residual_dynamic, jacobian_dynamic,
                                          residual_static, jacobian_static);
    } else if (corpus.model_id == "rat43") {
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(x[1] - x[2] * xv);
          const double t = 1.0 + e;
          const double p = std::pow(t, -1.0 / x[3]);
          r[i] = x[0] * p - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
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
      auto residual_static = [&](ConstVectorView<4> x, VectorView<15> r) -> ErrorOrVoid {
        for (Index i = 0; i < 15; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          const double e = std::exp(x[1] - x[2] * xv);
          const double t = 1.0 + e;
          const double p = std::pow(t, -1.0 / x[3]);
          r[i] = x[0] * p - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<4> x, MatrixView<15, 4> J) -> ErrorOrVoid {
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
    } else if (corpus.model_id == "linear_dense" && corpus.m == 1000 && corpus.n == 4) {
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          double value = 0.0;
          for (Index j = 0; j < 4; ++j) {
            value += x[j] * row[j];
          }
          r[i] = value - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent>,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          for (Index j = 0; j < 4; ++j) {
            J[i, j] = row[j];
          }
        }
        return {};
      };
      auto residual_static = [&](ConstVectorView<4> x, VectorView<1000> r) -> ErrorOrVoid {
        for (Index i = 0; i < 1000; ++i) {
          const auto &row = corpus.data[i];
          double value = 0.0;
          for (Index j = 0; j < 4; ++j) {
            value += x[j] * row[j];
          }
          r[i] = value - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<4>, MatrixView<1000, 4> J) -> ErrorOrVoid {
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
    } else if (corpus.model_id == "linear_dense" && corpus.m == 10000 && corpus.n == 4) {
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          double value = 0.0;
          for (Index j = 0; j < 4; ++j) {
            value += x[j] * row[j];
          }
          r[i] = value - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent>,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          for (Index j = 0; j < 4; ++j) {
            J[i, j] = row[j];
          }
        }
        return {};
      };
      auto residual_static = [&](ConstVectorView<4> x, VectorView<10000> r) -> ErrorOrVoid {
        for (Index i = 0; i < 10000; ++i) {
          const auto &row = corpus.data[i];
          double value = 0.0;
          for (Index j = 0; j < 4; ++j) {
            value += x[j] * row[j];
          }
          r[i] = value - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<4>, MatrixView<10000, 4> J) -> ErrorOrVoid {
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
    } else if (corpus.model_id == "rational_dense" && corpus.m == 32 && corpus.n == 32) {
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        constexpr Index Half = 16;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          double powers[17];
          powers[0] = 1.0;
          for (Index j = 1; j <= Half; ++j) powers[j] = powers[j-1] * xv;
          double num = 0.0;
          double den = 1.0;
          for (Index j = 0; j < Half; ++j) {
            num += x[j] * powers[j];
            den += x[Half + j] * powers[j + 1];
          }
          r[i] = num / den - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
        constexpr Index Half = 16;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          double powers[17];
          powers[0] = 1.0;
          for (Index j = 1; j <= Half; ++j) powers[j] = powers[j-1] * xv;
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
      auto residual_static = [&](ConstVectorView<32> x, VectorView<32> r) -> ErrorOrVoid {
        constexpr Index Half = 16;
        for (Index i = 0; i < 32; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          double powers[17];
          powers[0] = 1.0;
          for (Index j = 1; j <= Half; ++j) powers[j] = powers[j-1] * xv;
          double num = 0.0;
          double den = 1.0;
          for (Index j = 0; j < Half; ++j) {
            num += x[j] * powers[j];
            den += x[Half + j] * powers[j + 1];
          }
          r[i] = num / den - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<32> x, MatrixView<32, 32> J) -> ErrorOrVoid {
        constexpr Index Half = 16;
        for (Index i = 0; i < 32; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          double powers[17];
          powers[0] = 1.0;
          for (Index j = 1; j <= Half; ++j) powers[j] = powers[j-1] * xv;
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
      run_pair.template operator()<32, 32>(residual_dynamic, jacobian_dynamic,
                                           residual_static, jacobian_static);
    } else if (corpus.model_id == "rational_dense" && corpus.m == 64 && corpus.n == 64) {
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        constexpr Index Half = 32;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          double powers[33];
          powers[0] = 1.0;
          for (Index j = 1; j <= Half; ++j) powers[j] = powers[j-1] * xv;
          double num = 0.0;
          double den = 1.0;
          for (Index j = 0; j < Half; ++j) {
            num += x[j] * powers[j];
            den += x[Half + j] * powers[j + 1];
          }
          r[i] = num / den - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
        constexpr Index Half = 32;
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          double powers[33];
          powers[0] = 1.0;
          for (Index j = 1; j <= Half; ++j) powers[j] = powers[j-1] * xv;
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
      auto residual_static = [&](ConstVectorView<64> x, VectorView<64> r) -> ErrorOrVoid {
        constexpr Index Half = 32;
        for (Index i = 0; i < 64; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          double powers[33];
          powers[0] = 1.0;
          for (Index j = 1; j <= Half; ++j) powers[j] = powers[j-1] * xv;
          double num = 0.0;
          double den = 1.0;
          for (Index j = 0; j < Half; ++j) {
            num += x[j] * powers[j];
            den += x[Half + j] * powers[j + 1];
          }
          r[i] = num / den - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<64> x, MatrixView<64, 64> J) -> ErrorOrVoid {
        constexpr Index Half = 32;
        for (Index i = 0; i < 64; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          double powers[33];
          powers[0] = 1.0;
          for (Index j = 1; j <= Half; ++j) powers[j] = powers[j-1] * xv;
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
    } else if (corpus.model_id == "exp_sum" && corpus.m == 512 && corpus.n == 32) {
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          double value = 0.0;
          for (Index j = 0; j < corpus.n; j += 2) {
            value += x[j] * std::exp(-x[j + 1] * xv);
          }
          r[i] = value - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
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
      auto residual_static = [&](ConstVectorView<32> x, VectorView<512> r) -> ErrorOrVoid {
        for (Index i = 0; i < 512; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          double value = 0.0;
          for (Index j = 0; j < 32; j += 2) {
            value += x[j] * std::exp(-x[j + 1] * xv);
          }
          r[i] = value - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<32> x, MatrixView<512, 32> J) -> ErrorOrVoid {
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
    } else if (corpus.model_id == "exp_sum" && corpus.m == 1024 && corpus.n == 64) {
      auto residual_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
        for (Index i = 0; i < corpus.m; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          double value = 0.0;
          for (Index j = 0; j < corpus.n; j += 2) {
            value += x[j] * std::exp(-x[j + 1] * xv);
          }
          r[i] = value - row.back();
        }
        return {};
      };
      auto jacobian_dynamic = [&](ConstVectorView<std::dynamic_extent> x,
                                  MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
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
      auto residual_static = [&](ConstVectorView<64> x, VectorView<1024> r) -> ErrorOrVoid {
        for (Index i = 0; i < 1024; ++i) {
          const auto &row = corpus.data[i];
          const double xv = row[0];
          double value = 0.0;
          for (Index j = 0; j < 64; j += 2) {
            value += x[j] * std::exp(-x[j + 1] * xv);
          }
          r[i] = value - row.back();
        }
        return {};
      };
      auto jacobian_static = [&](ConstVectorView<64> x, MatrixView<1024, 64> J) -> ErrorOrVoid {
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
      throw std::runtime_error("Explicit callback dispatch not yet implemented for " +
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
  std::cout << "passed " << report.name
            << " dynamic_total=" << milliseconds(report.dynamic_timing.total_seconds)
            << "ms static_total=" << milliseconds(report.static_timing.total_seconds)
            << "ms dynamic_residual=" << milliseconds(report.dynamic_timing.residual_seconds)
            << "ms static_residual=" << milliseconds(report.static_timing.residual_seconds)
            << "ms dynamic_analytic=" << milliseconds(report.dynamic_timing.analytic_jacobian_seconds)
            << "ms static_analytic=" << milliseconds(report.static_timing.analytic_jacobian_seconds)
            << "ms dynamic_fd=" << milliseconds(report.dynamic_timing.forward_difference_seconds)
            << "ms static_fd=" << milliseconds(report.static_timing.forward_difference_seconds)
            << "ms dynamic_central=" << milliseconds(report.dynamic_timing.central_difference_seconds)
            << "ms static_central=" << milliseconds(report.static_timing.central_difference_seconds) << "ms";
  if (report.numerical_derivatives_skipped) {
    std::cout << " numerical_derivatives_skipped=true";
  }
  std::cout << '\n';
  print_stats_line("residual", report.residual_stats);
  print_stats_line("analytic", report.analytic_stats);
  if (report.forward_difference_stats.count > 0) {
    print_stats_line("forward_diff", report.forward_difference_stats);
  }
  if (report.central_difference_stats.count > 0) {
    print_stats_line("central_diff", report.central_difference_stats);
  }
}

void print_summary(const SummaryStats &summary) {
  std::cout << std::fixed << std::setprecision(4);
  std::cout << "summary problems=" << summary.problems
            << " numerical_derivative_skips="
            << summary.numerical_derivative_skips
            << " dynamic_total=" << milliseconds(summary.dynamic_timing.total_seconds)
            << "ms static_total=" << milliseconds(summary.static_timing.total_seconds)
            << "ms dynamic_residual=" << milliseconds(summary.dynamic_timing.residual_seconds)
            << "ms static_residual=" << milliseconds(summary.static_timing.residual_seconds)
            << "ms dynamic_analytic=" << milliseconds(summary.dynamic_timing.analytic_jacobian_seconds)
            << "ms static_analytic=" << milliseconds(summary.static_timing.analytic_jacobian_seconds)
            << "ms dynamic_fd=" << milliseconds(summary.dynamic_timing.forward_difference_seconds)
            << "ms static_fd=" << milliseconds(summary.static_timing.forward_difference_seconds)
            << "ms dynamic_central=" << milliseconds(summary.dynamic_timing.central_difference_seconds)
            << "ms static_central=" << milliseconds(summary.static_timing.central_difference_seconds)
            << "ms\n";
  print_stats_line("residual", summary.residual_stats);
  print_stats_line("analytic", summary.analytic_stats);
  if (summary.forward_difference_stats.count > 0) {
    print_stats_line("forward_diff", summary.forward_difference_stats);
  }
  if (summary.central_difference_stats.count > 0) {
    print_stats_line("central_diff", summary.central_difference_stats);
  }
}

void write_problem_csv_row(std::ofstream &file, const ProblemReport &report) {
  file << report.name << ",problem,"
       << (report.numerical_derivatives_skipped ? "true" : "false") << ','
       << std::setprecision(17) << milliseconds(report.dynamic_timing.total_seconds) << ','
       << milliseconds(report.dynamic_timing.residual_seconds) << ','
       << milliseconds(report.dynamic_timing.analytic_jacobian_seconds) << ','
       << milliseconds(report.dynamic_timing.forward_difference_seconds) << ','
       << milliseconds(report.dynamic_timing.central_difference_seconds) << ','
       << milliseconds(report.static_timing.total_seconds) << ','
       << milliseconds(report.static_timing.residual_seconds) << ','
       << milliseconds(report.static_timing.analytic_jacobian_seconds) << ','
       << milliseconds(report.static_timing.forward_difference_seconds) << ','
       << milliseconds(report.static_timing.central_difference_seconds) << ','
       << report.residual_stats.count << ','
       << report.residual_stats.max_abs_error << ','
       << report.residual_stats.max_rel_error << ','
       << report.residual_stats.rms_abs_error() << ','
       << report.residual_stats.rms_rel_error() << ','
       << report.analytic_stats.count << ','
       << report.analytic_stats.max_abs_error << ','
       << report.analytic_stats.max_rel_error << ','
       << report.analytic_stats.rms_abs_error() << ','
       << report.analytic_stats.rms_rel_error() << ','
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
  file << "summary,summary,false,"
       << std::setprecision(17) << milliseconds(summary.dynamic_timing.total_seconds) << ','
       << milliseconds(summary.dynamic_timing.residual_seconds) << ','
       << milliseconds(summary.dynamic_timing.analytic_jacobian_seconds) << ','
       << milliseconds(summary.dynamic_timing.forward_difference_seconds) << ','
       << milliseconds(summary.dynamic_timing.central_difference_seconds) << ','
       << milliseconds(summary.static_timing.total_seconds) << ','
       << milliseconds(summary.static_timing.residual_seconds) << ','
       << milliseconds(summary.static_timing.analytic_jacobian_seconds) << ','
       << milliseconds(summary.static_timing.forward_difference_seconds) << ','
       << milliseconds(summary.static_timing.central_difference_seconds) << ','
       << summary.residual_stats.count << ','
       << summary.residual_stats.max_abs_error << ','
       << summary.residual_stats.max_rel_error << ','
       << summary.residual_stats.rms_abs_error() << ','
       << summary.residual_stats.rms_rel_error() << ','
       << summary.analytic_stats.count << ','
       << summary.analytic_stats.max_abs_error << ','
       << summary.analytic_stats.max_rel_error << ','
       << summary.analytic_stats.rms_abs_error() << ','
       << summary.analytic_stats.rms_rel_error() << ','
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

  file << "name,row_type,numerical_derivatives_skipped,"
       << "dynamic_total_ms,dynamic_residual_ms,dynamic_analytic_jacobian_ms,"
       << "dynamic_forward_difference_ms,dynamic_central_difference_ms,"
       << "static_total_ms,static_residual_ms,static_analytic_jacobian_ms,"
       << "static_forward_difference_ms,static_central_difference_ms,"
       << "residual_count,residual_max_abs,residual_max_rel,residual_rms_abs,residual_rms_rel,"
       << "analytic_count,analytic_max_abs,analytic_max_rel,analytic_rms_abs,analytic_rms_rel,"
       << "forward_difference_count,forward_difference_max_abs,forward_difference_max_rel,forward_difference_rms_abs,forward_difference_rms_rel,"
       << "central_difference_count,central_difference_max_abs,central_difference_max_rel,central_difference_rms_abs,central_difference_rms_rel\n";

  for (const auto &report : reports) {
    write_problem_csv_row(file, report);
  }
  write_summary_csv_row(file, summary);
}

void print_timing_moment_line(const std::string &label, const ScalarMoments &dynamic,
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
  print_timing_moment_line("forward_diff",
                           benchmark.dynamic_timing.forward_difference,
                           benchmark.static_timing.forward_difference);
  print_timing_moment_line("central_diff",
                           benchmark.dynamic_timing.central_difference,
                           benchmark.static_timing.central_difference);
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
  print_timing_moment_line("forward_diff",
                           benchmark.dynamic_timing.forward_difference,
                           benchmark.static_timing.forward_difference);
  print_timing_moment_line("central_diff",
                           benchmark.dynamic_timing.central_difference,
                           benchmark.static_timing.central_difference);
}

SummaryBenchmark benchmark_summary(
    const std::vector<std::filesystem::path> &problem_dirs,
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
      const auto report = run_problem(problem_dirs[i]);
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

} // namespace

int main(int argc, char **argv) {
  try {
    std::filesystem::path corpus_dir;
    std::filesystem::path csv_path;
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

    std::vector<std::filesystem::path> problem_dirs;
    for (const auto &entry : std::filesystem::directory_iterator(corpus_dir)) {
      if (entry.is_directory()) {
        problem_dirs.push_back(entry.path());
      }
    }
    std::ranges::sort(problem_dirs);

    SummaryStats summary;
    std::vector<ProblemReport> reports;
    for (const auto &path : problem_dirs) {
      const auto report = run_problem(path);
      print_problem_report(report);
      merge_summary(summary, report);
      reports.push_back(report);
    }
    print_summary(summary);
    write_csv_report(csv_path, reports, summary);

    std::vector<ProblemBenchmark> problem_benchmarks;
    const auto summary_benchmark =
        benchmark_summary(problem_dirs, benchmark_iterations, problem_benchmarks);
    for (const auto &benchmark : problem_benchmarks) {
      print_problem_benchmark(benchmark);
    }
    print_summary_benchmark(summary_benchmark, benchmark_iterations);
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }

  std::cout << "all corpus checks passed\n";
  return 0;
}
