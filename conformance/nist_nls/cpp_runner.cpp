#include "../../cpp/lm.h"

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

using ModelResidualFunction = double (*)(ConstVectorView beta,
                                         const std::vector<double> &row);
using ModelJacobianFunction = void (*)(ConstVectorView beta,
                                       const std::vector<double> &row,
                                       double *out);

struct ModelFunctions {
  ModelResidualFunction residual;
  ModelJacobianFunction jacobian;
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

struct ProblemReport {
  std::string name;
  bool numerical_derivatives_skipped = false;
  TimingStats timing;
  ComparisonStats residual_stats;
  ComparisonStats analytic_stats;
  ComparisonStats forward_difference_stats;
  ComparisonStats central_difference_stats;
};

struct SummaryStats {
  std::uint64_t problems = 0;
  std::uint64_t numerical_derivative_skips = 0;
  TimingStats timing;
  ComparisonStats residual_stats;
  ComparisonStats analytic_stats;
  ComparisonStats forward_difference_stats;
  ComparisonStats central_difference_stats;
};

using Clock = std::chrono::steady_clock;

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
  summary.timing.residual_seconds += report.timing.residual_seconds;
  summary.timing.analytic_jacobian_seconds +=
      report.timing.analytic_jacobian_seconds;
  summary.timing.forward_difference_seconds +=
      report.timing.forward_difference_seconds;
  summary.timing.central_difference_seconds +=
      report.timing.central_difference_seconds;
  summary.timing.total_seconds += report.timing.total_seconds;
  merge_stats(summary.residual_stats, report.residual_stats);
  merge_stats(summary.analytic_stats, report.analytic_stats);
  merge_stats(summary.forward_difference_stats, report.forward_difference_stats);
  merge_stats(summary.central_difference_stats, report.central_difference_stats);
}

double x_of(const std::vector<double> &row, Index index = 0) { return row[index]; }
double y_of(const std::vector<double> &row) { return row.back(); }

double monomolecular_residual(ConstVectorView beta, const std::vector<double> &row) {
  const double x = x_of(row);
  const double e = std::exp(-beta[1] * x);
  return beta[0] * (1.0 - e) - y_of(row);
}

void monomolecular_jacobian(ConstVectorView beta, const std::vector<double> &row,
                            double *out) {
  const double x = x_of(row);
  const double e = std::exp(-beta[1] * x);
  out[0] = 1.0 - e;
  out[1] = beta[0] * x * e;
}

double chwirut_residual(ConstVectorView beta, const std::vector<double> &row) {
  const double x = x_of(row);
  const double e = std::exp(-beta[0] * x);
  const double d = beta[1] + beta[2] * x;
  return e / d - y_of(row);
}

void chwirut_jacobian(ConstVectorView beta, const std::vector<double> &row,
                      double *out) {
  const double x = x_of(row);
  const double e = std::exp(-beta[0] * x);
  const double d = beta[1] + beta[2] * x;
  const double d2 = d * d;
  out[0] = -x * e / d;
  out[1] = -e / d2;
  out[2] = -x * e / d2;
}

double triple_exponential_residual(ConstVectorView beta, const std::vector<double> &row) {
  const double x = x_of(row);
  double value = 0.0;
  for (Index i = 0; i < 6; i += 2) {
    value += beta[i] * std::exp(-beta[i + 1] * x);
  }
  return value - y_of(row);
}

void triple_exponential_jacobian(ConstVectorView beta, const std::vector<double> &row,
                                 double *out) {
  const double x = x_of(row);
  for (Index i = 0; i < 6; i += 2) {
    const double e = std::exp(-beta[i + 1] * x);
    out[i] = e;
    out[i + 1] = -beta[i] * x * e;
  }
}

double gauss_mixture_residual(ConstVectorView beta, const std::vector<double> &row) {
  const double x = x_of(row);
  const double d1 = x - beta[3];
  const double d2 = x - beta[6];
  return beta[0] * std::exp(-beta[1] * x) +
             beta[2] * std::exp(-(d1 * d1) / (beta[4] * beta[4])) +
             beta[5] * std::exp(-(d2 * d2) / (beta[7] * beta[7])) -
         y_of(row);
}

void gauss_mixture_jacobian(ConstVectorView beta, const std::vector<double> &row,
                            double *out) {
  const double x = x_of(row);
  const double e1 = std::exp(-beta[1] * x);
  const double d1 = x - beta[3];
  const double d2 = x - beta[6];
  const double s1 = beta[4] * beta[4];
  const double s2 = beta[7] * beta[7];
  const double e2 = std::exp(-(d1 * d1) / s1);
  const double e3 = std::exp(-(d2 * d2) / s2);
  out[0] = e1;
  out[1] = -beta[0] * x * e1;
  out[2] = e2;
  out[3] = beta[2] * e2 * 2.0 * d1 / s1;
  out[4] = beta[2] * e2 * 2.0 * d1 * d1 / (beta[4] * beta[4] * beta[4]);
  out[5] = e3;
  out[6] = beta[5] * e3 * 2.0 * d2 / s2;
  out[7] = beta[5] * e3 * 2.0 * d2 * d2 / (beta[7] * beta[7] * beta[7]);
}

double danwood_residual(ConstVectorView beta, const std::vector<double> &row) {
  const double x = x_of(row);
  return beta[0] * std::pow(x, beta[1]) - y_of(row);
}

void danwood_jacobian(ConstVectorView beta, const std::vector<double> &row,
                      double *out) {
  const double x = x_of(row);
  const double x_pow = std::pow(x, beta[1]);
  out[0] = x_pow;
  out[1] = beta[0] * x_pow * std::log(x);
}

double misra1b_residual(ConstVectorView beta, const std::vector<double> &row) {
  const double x = x_of(row);
  const double t = 1.0 + beta[1] * x / 2.0;
  return beta[0] * (1.0 - std::pow(t, -2.0)) - y_of(row);
}

void misra1b_jacobian(ConstVectorView beta, const std::vector<double> &row,
                      double *out) {
  const double x = x_of(row);
  const double t = 1.0 + beta[1] * x / 2.0;
  out[0] = 1.0 - std::pow(t, -2.0);
  out[1] = beta[0] * x * std::pow(t, -3.0);
}

double rational_quadratic_residual(ConstVectorView beta,
                                   const std::vector<double> &row) {
  const double x = x_of(row);
  const double x2 = x * x;
  const double n = beta[0] + beta[1] * x + beta[2] * x2;
  const double d = 1.0 + beta[3] * x + beta[4] * x2;
  return n / d - y_of(row);
}

void rational_quadratic_jacobian(ConstVectorView beta,
                                 const std::vector<double> &row, double *out) {
  const double x = x_of(row);
  const double x2 = x * x;
  const double n = beta[0] + beta[1] * x + beta[2] * x2;
  const double d = 1.0 + beta[3] * x + beta[4] * x2;
  const double d2 = d * d;
  out[0] = 1.0 / d;
  out[1] = x / d;
  out[2] = x2 / d;
  out[3] = -n * x / d2;
  out[4] = -n * x2 / d2;
}

double rational_cubic_residual(ConstVectorView beta, const std::vector<double> &row) {
  const double x = x_of(row);
  const double x2 = x * x;
  const double x3 = x2 * x;
  const double n = beta[0] + beta[1] * x + beta[2] * x2 + beta[3] * x3;
  const double d = 1.0 + beta[4] * x + beta[5] * x2 + beta[6] * x3;
  return n / d - y_of(row);
}

void rational_cubic_jacobian(ConstVectorView beta, const std::vector<double> &row,
                             double *out) {
  const double x = x_of(row);
  const double x2 = x * x;
  const double x3 = x2 * x;
  const double n = beta[0] + beta[1] * x + beta[2] * x2 + beta[3] * x3;
  const double d = 1.0 + beta[4] * x + beta[5] * x2 + beta[6] * x3;
  const double d2 = d * d;
  out[0] = 1.0 / d;
  out[1] = x / d;
  out[2] = x2 / d;
  out[3] = x3 / d;
  out[4] = -n * x / d2;
  out[5] = -n * x2 / d2;
  out[6] = -n * x3 / d2;
}

double nelson_log_residual(ConstVectorView beta, const std::vector<double> &row) {
  const double x1 = x_of(row, 0);
  const double x2 = x_of(row, 1);
  return beta[0] - beta[1] * x1 * std::exp(-beta[2] * x2) - std::log(y_of(row));
}

void nelson_log_jacobian(ConstVectorView beta, const std::vector<double> &row,
                         double *out) {
  const double x1 = x_of(row, 0);
  const double x2 = x_of(row, 1);
  const double e = std::exp(-beta[2] * x2);
  out[0] = 1.0;
  out[1] = -x1 * e;
  out[2] = beta[1] * x1 * x2 * e;
}

double mgh17_residual(ConstVectorView beta, const std::vector<double> &row) {
  const double x = x_of(row);
  return beta[0] + beta[1] * std::exp(-beta[3] * x) +
             beta[2] * std::exp(-beta[4] * x) -
         y_of(row);
}

void mgh17_jacobian(ConstVectorView beta, const std::vector<double> &row,
                    double *out) {
  const double x = x_of(row);
  const double e1 = std::exp(-beta[3] * x);
  const double e2 = std::exp(-beta[4] * x);
  out[0] = 1.0;
  out[1] = e1;
  out[2] = e2;
  out[3] = -beta[1] * x * e1;
  out[4] = -beta[2] * x * e2;
}

double misra1c_residual(ConstVectorView beta, const std::vector<double> &row) {
  const double x = x_of(row);
  const double t = 1.0 + 2.0 * beta[1] * x;
  return beta[0] * (1.0 - 1.0 / std::sqrt(t)) - y_of(row);
}

void misra1c_jacobian(ConstVectorView beta, const std::vector<double> &row,
                      double *out) {
  const double x = x_of(row);
  const double t = 1.0 + 2.0 * beta[1] * x;
  out[0] = 1.0 - 1.0 / std::sqrt(t);
  out[1] = beta[0] * x * std::pow(t, -1.5);
}

double misra1d_residual(ConstVectorView beta, const std::vector<double> &row) {
  const double x = x_of(row);
  const double d = 1.0 + beta[1] * x;
  return (beta[0] * beta[1] * x) / d - y_of(row);
}

void misra1d_jacobian(ConstVectorView beta, const std::vector<double> &row,
                      double *out) {
  const double x = x_of(row);
  const double d = 1.0 + beta[1] * x;
  const double d2 = d * d;
  out[0] = beta[1] * x / d;
  out[1] = beta[0] * x / d2;
}

double roszman1_residual(ConstVectorView beta, const std::vector<double> &row) {
  const double x = x_of(row);
  return beta[0] - beta[1] * x - std::atan(beta[2] / (x - beta[3])) /
             std::numbers::pi -
         y_of(row);
}

void roszman1_jacobian(ConstVectorView beta, const std::vector<double> &row,
                       double *out) {
  const double x = x_of(row);
  const double u = beta[2] / (x - beta[3]);
  const double common = -1.0 / (std::numbers::pi * (1.0 + u * u));
  out[0] = 1.0;
  out[1] = -x;
  out[2] = common / (x - beta[3]);
  out[3] = common * beta[2] / ((x - beta[3]) * (x - beta[3]));
}

double enso_residual(ConstVectorView beta, const std::vector<double> &row) {
  const double x = x_of(row);
  const double annual = 2.0 * std::numbers::pi * x / 12.0;
  const double p4 = 2.0 * std::numbers::pi * x / beta[3];
  const double p7 = 2.0 * std::numbers::pi * x / beta[6];
  const double value = beta[0] + beta[1] * std::cos(annual) +
                       beta[2] * std::sin(annual) + beta[4] * std::cos(p4) +
                       beta[5] * std::sin(p4) + beta[7] * std::cos(p7) +
                       beta[8] * std::sin(p7);
  return value - y_of(row);
}

void enso_jacobian(ConstVectorView beta, const std::vector<double> &row,
                   double *out) {
  const double x = x_of(row);
  const double annual = 2.0 * std::numbers::pi * x / 12.0;
  const double p4 = 2.0 * std::numbers::pi * x / beta[3];
  const double p7 = 2.0 * std::numbers::pi * x / beta[6];
  const double d4 = 2.0 * std::numbers::pi * x / (beta[3] * beta[3]);
  const double d7 = 2.0 * std::numbers::pi * x / (beta[6] * beta[6]);
  out[0] = 1.0;
  out[1] = std::cos(annual);
  out[2] = std::sin(annual);
  out[3] = d4 * (beta[4] * std::sin(p4) - beta[5] * std::cos(p4));
  out[4] = std::cos(p4);
  out[5] = std::sin(p4);
  out[6] = d7 * (beta[7] * std::sin(p7) - beta[8] * std::cos(p7));
  out[7] = std::cos(p7);
  out[8] = std::sin(p7);
}

double mgh09_residual(ConstVectorView beta, const std::vector<double> &row) {
  const double x = x_of(row);
  const double n = x * x + beta[1] * x;
  const double d = x * x + beta[2] * x + beta[3];
  return beta[0] * n / d - y_of(row);
}

void mgh09_jacobian(ConstVectorView beta, const std::vector<double> &row,
                    double *out) {
  const double x = x_of(row);
  const double n = x * x + beta[1] * x;
  const double d = x * x + beta[2] * x + beta[3];
  const double d2 = d * d;
  out[0] = n / d;
  out[1] = beta[0] * x / d;
  out[2] = -beta[0] * n * x / d2;
  out[3] = -beta[0] * n / d2;
}

double rat42_residual(ConstVectorView beta, const std::vector<double> &row) {
  const double x = x_of(row);
  const double e = std::exp(beta[1] - beta[2] * x);
  return beta[0] / (1.0 + e) - y_of(row);
}

void rat42_jacobian(ConstVectorView beta, const std::vector<double> &row,
                    double *out) {
  const double x = x_of(row);
  const double e = std::exp(beta[1] - beta[2] * x);
  const double d = 1.0 + e;
  const double d2 = d * d;
  out[0] = 1.0 / d;
  out[1] = -beta[0] * e / d2;
  out[2] = beta[0] * x * e / d2;
}

double mgh10_residual(ConstVectorView beta, const std::vector<double> &row) {
  const double x = x_of(row);
  const double e = std::exp(beta[1] / (x + beta[2]));
  return beta[0] * e - y_of(row);
}

void mgh10_jacobian(ConstVectorView beta, const std::vector<double> &row,
                    double *out) {
  const double x = x_of(row);
  const double q = x + beta[2];
  const double e = std::exp(beta[1] / q);
  out[0] = e;
  out[1] = beta[0] * e / q;
  out[2] = -beta[0] * e * beta[1] / (q * q);
}

double eckerle4_residual(ConstVectorView beta, const std::vector<double> &row) {
  const double x = x_of(row);
  const double delta = x - beta[2];
  const double e = std::exp(-(delta * delta) / (2.0 * beta[1] * beta[1]));
  return (beta[0] / beta[1]) * e - y_of(row);
}

void eckerle4_jacobian(ConstVectorView beta, const std::vector<double> &row,
                       double *out) {
  const double x = x_of(row);
  const double b1 = beta[0];
  const double b2 = beta[1];
  const double b3 = beta[2];
  const double delta = x - b3;
  const double e = std::exp(-(delta * delta) / (2.0 * b2 * b2));
  out[0] = e / b2;
  out[1] = b1 * e * (delta * delta / (b2 * b2 * b2 * b2) - 1.0 / (b2 * b2));
  out[2] = b1 * e * delta / (b2 * b2 * b2);
}

double rat43_residual(ConstVectorView beta, const std::vector<double> &row) {
  const double x = x_of(row);
  const double e = std::exp(beta[1] - beta[2] * x);
  const double t = 1.0 + e;
  const double p = std::pow(t, -1.0 / beta[3]);
  return beta[0] * p - y_of(row);
}

void rat43_jacobian(ConstVectorView beta, const std::vector<double> &row,
                    double *out) {
  const double x = x_of(row);
  const double b1 = beta[0];
  const double b2 = beta[1];
  const double b3 = beta[2];
  const double b4 = beta[3];
  const double e = std::exp(b2 - b3 * x);
  const double t = 1.0 + e;
  const double p = std::pow(t, -1.0 / b4);
  const double yhat = b1 * p;
  out[0] = p;
  out[1] = -yhat * e / (b4 * t);
  out[2] = yhat * x * e / (b4 * t);
  out[3] = yhat * std::log(t) / (b4 * b4);
}

double bennett5_residual(ConstVectorView beta, const std::vector<double> &row) {
  const double x = x_of(row);
  return beta[0] * std::pow(beta[1] + x, -1.0 / beta[2]) - y_of(row);
}

void bennett5_jacobian(ConstVectorView beta, const std::vector<double> &row,
                       double *out) {
  const double x = x_of(row);
  const double t = beta[1] + x;
  const double p = std::pow(t, -1.0 / beta[2]);
  out[0] = p;
  out[1] = -beta[0] * p / (beta[2] * t);
  out[2] = beta[0] * p * std::log(t) / (beta[2] * beta[2]);
}

ModelFunctions get_model_functions(const std::string &model_id) {
  if (model_id == "monomolecular") {
    return {monomolecular_residual, monomolecular_jacobian};
  }
  if (model_id == "chwirut") {
    return {chwirut_residual, chwirut_jacobian};
  }
  if (model_id == "triple_exponential") {
    return {triple_exponential_residual, triple_exponential_jacobian};
  }
  if (model_id == "gauss_mixture") {
    return {gauss_mixture_residual, gauss_mixture_jacobian};
  }
  if (model_id == "danwood") {
    return {danwood_residual, danwood_jacobian};
  }
  if (model_id == "misra1b") {
    return {misra1b_residual, misra1b_jacobian};
  }
  if (model_id == "rational_quadratic") {
    return {rational_quadratic_residual, rational_quadratic_jacobian};
  }
  if (model_id == "rational_cubic") {
    return {rational_cubic_residual, rational_cubic_jacobian};
  }
  if (model_id == "nelson_log") {
    return {nelson_log_residual, nelson_log_jacobian};
  }
  if (model_id == "mgh17") {
    return {mgh17_residual, mgh17_jacobian};
  }
  if (model_id == "misra1c") {
    return {misra1c_residual, misra1c_jacobian};
  }
  if (model_id == "misra1d") {
    return {misra1d_residual, misra1d_jacobian};
  }
  if (model_id == "roszman1") {
    return {roszman1_residual, roszman1_jacobian};
  }
  if (model_id == "enso") {
    return {enso_residual, enso_jacobian};
  }
  if (model_id == "mgh09") {
    return {mgh09_residual, mgh09_jacobian};
  }
  if (model_id == "rat42") {
    return {rat42_residual, rat42_jacobian};
  }
  if (model_id == "mgh10") {
    return {mgh10_residual, mgh10_jacobian};
  }
  if (model_id == "eckerle4") {
    return {eckerle4_residual, eckerle4_jacobian};
  }
  if (model_id == "rat43") {
    return {rat43_residual, rat43_jacobian};
  }
  if (model_id == "bennett5") {
    return {bennett5_residual, bennett5_jacobian};
  }
  throw std::runtime_error("Unknown model_id: " + model_id);
}

ProblemReport run_problem(const std::filesystem::path &problem_dir) {
  const auto problem_start = Clock::now();
  const auto corpus = load_problem(problem_dir);
  const auto functions = get_model_functions(corpus.model_id);
  ProblemReport report;
  report.name = corpus.name;
  report.numerical_derivatives_skipped = !corpus.numerical_derivatives_recommended;

  std::vector<double> beta_storage(corpus.n, 0.0);

  Problem problem;
  problem.num_residuals = corpus.m;
  problem.num_parameters = corpus.n;
  problem.residual = [&](ConstVectorView x, VectorView r) {
    for (Index i = 0; i < corpus.m; ++i) {
      r[i] = functions.residual(x, corpus.data[i]);
    }
    return true;
  };
  problem.jacobian = [&](ConstVectorView x, MatrixView J) {
    std::vector<double> row(corpus.n, 0.0);
    for (Index i = 0; i < corpus.m; ++i) {
      functions.jacobian(x, corpus.data[i], row.data());
      for (Index j = 0; j < corpus.n; ++j) {
        J[i, j] = row[j];
      }
    }
    return true;
  };

  for (const auto &[label, params] : corpus.params) {
    beta_storage = params;
    const auto expected_residuals =
        read_numeric_csv(problem_dir / ("residuals_" + label + ".csv"));
    const auto expected_jacobian =
        read_numeric_csv(problem_dir / ("jacobian_" + label + ".csv"));

    {
      Options options;
      options.jacobian_mode = JacobianMode::User;

      Result result;
      LMWorkspace workspace(corpus.m, corpus.n);
      LMSolveContext context{&problem, &options, &result, &workspace,
                             ConstVectorView(beta_storage)};

      const auto residual_start = Clock::now();
      if (!evaluate_residual(context, corpus.name + " residual " + label)) {
        throw std::runtime_error(result.message);
      }
      report.timing.residual_seconds +=
          elapsed_seconds(residual_start, Clock::now());
      for (Index i = 0; i < corpus.m; ++i) {
        report.residual_stats.add(workspace.r[i], expected_residuals[i][0]);
        expect_close(workspace.r[i], expected_residuals[i][0], 1e-12, 1e-10,
                     corpus.name + " residual " + label + " row " +
                         std::to_string(i));
      }

      const auto analytic_start = Clock::now();
      if (!evaluate_jacobian(context,
                             corpus.name + " analytic jacobian " + label)) {
        throw std::runtime_error(result.message);
      }
      report.timing.analytic_jacobian_seconds +=
          elapsed_seconds(analytic_start, Clock::now());
      for (Index j = 0; j < corpus.n; ++j) {
        for (Index i = 0; i < corpus.m; ++i) {
          report.analytic_stats.add(workspace.J(i, j), expected_jacobian[i][j]);
          expect_close(workspace.J(i, j), expected_jacobian[i][j], 1e-12,
                       1e-10, corpus.name + " analytic jacobian " + label +
                                  " row " + std::to_string(i) + " col " +
                                  std::to_string(j));
        }
      }
    }

    if (corpus.numerical_derivatives_recommended && label == "certified") {
      {
        Options options;
        options.jacobian_mode = JacobianMode::ForwardDifference;

        Result result;
        LMWorkspace workspace(corpus.m, corpus.n);
        LMSolveContext context{&problem, &options, &result, &workspace,
                               ConstVectorView(beta_storage)};

        const auto residual_start = Clock::now();
        if (!evaluate_residual(context, corpus.name + " fd residual " + label)) {
          throw std::runtime_error(result.message);
        }
        report.timing.residual_seconds +=
            elapsed_seconds(residual_start, Clock::now());
        const auto fd_start = Clock::now();
        if (!evaluate_jacobian(context, corpus.name + " fd jacobian " + label)) {
          throw std::runtime_error(result.message);
        }
        report.timing.forward_difference_seconds +=
            elapsed_seconds(fd_start, Clock::now());

        for (Index j = 0; j < corpus.n; ++j) {
          for (Index i = 0; i < corpus.m; ++i) {
            report.forward_difference_stats.add(workspace.J(i, j),
                                                expected_jacobian[i][j]);
            expect_close(workspace.J(i, j), expected_jacobian[i][j], 1e-7, 2e-3,
                         corpus.name + " fd jacobian " + label + " row " +
                             std::to_string(i) + " col " + std::to_string(j));
          }
        }
      }

      {
        Options options;
        options.jacobian_mode = JacobianMode::CentralDifference;

        Result result;
        LMWorkspace workspace(corpus.m, corpus.n);
        LMSolveContext context{&problem, &options, &result, &workspace,
                               ConstVectorView(beta_storage)};

        const auto central_start = Clock::now();
        if (!evaluate_jacobian(context,
                               corpus.name + " central jacobian " + label)) {
          throw std::runtime_error(result.message);
        }
        report.timing.central_difference_seconds +=
            elapsed_seconds(central_start, Clock::now());

        for (Index j = 0; j < corpus.n; ++j) {
          for (Index i = 0; i < corpus.m; ++i) {
            report.central_difference_stats.add(workspace.J(i, j),
                                                expected_jacobian[i][j]);
            expect_close(workspace.J(i, j), expected_jacobian[i][j], 1e-8, 5e-4,
                         corpus.name + " central jacobian " + label +
                             " row " + std::to_string(i) + " col " +
                             std::to_string(j));
          }
        }
      }
    }
  }

  report.timing.total_seconds = elapsed_seconds(problem_start, Clock::now());
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
  std::cout << "passed " << report.name << " total=" << report.timing.total_seconds
            << "s residual=" << report.timing.residual_seconds
            << "s analytic=" << report.timing.analytic_jacobian_seconds
            << "s fd=" << report.timing.forward_difference_seconds
            << "s central=" << report.timing.central_difference_seconds << "s";
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
            << " total=" << summary.timing.total_seconds
            << "s residual=" << summary.timing.residual_seconds
            << "s analytic=" << summary.timing.analytic_jacobian_seconds
            << "s fd=" << summary.timing.forward_difference_seconds
            << "s central=" << summary.timing.central_difference_seconds
            << "s\n";
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
       << std::setprecision(17) << report.timing.total_seconds << ','
       << report.timing.residual_seconds << ','
       << report.timing.analytic_jacobian_seconds << ','
       << report.timing.forward_difference_seconds << ','
       << report.timing.central_difference_seconds << ','
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
       << std::setprecision(17) << summary.timing.total_seconds << ','
       << summary.timing.residual_seconds << ','
       << summary.timing.analytic_jacobian_seconds << ','
       << summary.timing.forward_difference_seconds << ','
       << summary.timing.central_difference_seconds << ','
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
       << "total_seconds,residual_seconds,analytic_jacobian_seconds,"
       << "forward_difference_seconds,central_difference_seconds,"
       << "residual_count,residual_max_abs,residual_max_rel,residual_rms_abs,residual_rms_rel,"
       << "analytic_count,analytic_max_abs,analytic_max_rel,analytic_rms_abs,analytic_rms_rel,"
       << "forward_difference_count,forward_difference_max_abs,forward_difference_max_rel,forward_difference_rms_abs,forward_difference_rms_rel,"
       << "central_difference_count,central_difference_max_abs,central_difference_max_rel,central_difference_rms_abs,central_difference_rms_rel\n";

  for (const auto &report : reports) {
    write_problem_csv_row(file, report);
  }
  write_summary_csv_row(file, summary);
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
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }

  std::cout << "all corpus checks passed\n";
  return 0;
}
