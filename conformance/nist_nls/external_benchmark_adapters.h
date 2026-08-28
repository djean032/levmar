#pragma once

#ifdef LEVMAR_HAVE_CMINPACK
template <class ResidualFn, class JacobianFn> struct MinpackCallbackContext {
  const CorpusProblem &corpus;
  ResidualFn &residual;
  JacobianFn &jacobian;
};

template <class ResidualFn, class JacobianFn>
int minpack_residual_callback(void *opaque, int m, int n, const double *x,
                              double *fvec, int) {
  auto &context =
      *static_cast<MinpackCallbackContext<ResidualFn, JacobianFn> *>(opaque);
  if (m != static_cast<int>(context.corpus.m) ||
      n != static_cast<int>(context.corpus.n)) {
    return -1;
  }
  const auto result = context.residual(
      ConstVectorView<std::dynamic_extent>(x, static_cast<Index>(n)),
      VectorView<std::dynamic_extent>(fvec, static_cast<Index>(m)));
  return result ? 0 : -1;
}

template <class ResidualFn, class JacobianFn>
int minpack_analytic_callback(void *opaque, int m, int n, const double *x,
                              double *fvec, double *fjac, int ldfjac,
                              int iflag) {
  auto &context =
      *static_cast<MinpackCallbackContext<ResidualFn, JacobianFn> *>(opaque);
  if (m != static_cast<int>(context.corpus.m) ||
      n != static_cast<int>(context.corpus.n)) {
    return -1;
  }
  if (iflag == 1) {
    return minpack_residual_callback<ResidualFn, JacobianFn>(opaque, m, n, x,
                                                             fvec, iflag);
  }
  if (iflag != 2 || ldfjac < m) {
    return -1;
  }
  const auto result = context.jacobian(
      ConstVectorView<std::dynamic_extent>(x, static_cast<Index>(n)),
      MatrixView<std::dynamic_extent, std::dynamic_extent>(
          fjac, static_cast<Index>(m), static_cast<Index>(n)));
  return result ? 0 : -1;
}

template <class ResidualFn, class JacobianFn>
ExternalSolverResult
run_minpack_solver(const std::string &name, const CorpusProblem &corpus,
                   ResidualFn &residual, JacobianFn &jacobian,
                   const std::vector<double> &start,
                   const std::vector<double> &certified, bool analytic) {
  const auto start_time = Clock::now();
  MinpackCallbackContext context{corpus, residual, jacobian};
  std::vector<double> x = start;
  std::vector<double> fvec(corpus.m);
  std::vector<double> fjac(corpus.m * corpus.n);
  std::vector<double> diag(corpus.n);
  std::vector<int> ipvt(corpus.n);
  std::vector<double> qtf(corpus.n);
  std::vector<double> wa1(corpus.n);
  std::vector<double> wa2(corpus.n);
  std::vector<double> wa3(corpus.n);
  std::vector<double> wa4(corpus.m);
  int nfev = 0;
  int njev = 0;
  constexpr double tolerance = std::numeric_limits<double>::epsilon();
  const int status =
      analytic
          ? lmder(minpack_analytic_callback<ResidualFn, JacobianFn>, &context,
                  corpus.m, corpus.n, x.data(), fvec.data(), fjac.data(),
                  corpus.m, tolerance, tolerance, tolerance, 100'000,
                  diag.data(), 1, 100.0, 0, &nfev, &njev, ipvt.data(),
                  qtf.data(), wa1.data(), wa2.data(), wa3.data(), wa4.data())
          : lmdif(minpack_residual_callback<ResidualFn, JacobianFn>, &context,
                  corpus.m, corpus.n, x.data(), fvec.data(), tolerance,
                  tolerance, tolerance, 100'000, 0.0, diag.data(), 1, 100.0, 0,
                  &nfev, fjac.data(), corpus.m, ipvt.data(), qtf.data(),
                  wa1.data(), wa2.data(), wa3.data(), wa4.data());
  ExternalSolverResult result;
  result.solver = name;
  result.seconds = elapsed_seconds(start_time, Clock::now());
  result.lre = log_relative_error(x, certified);
  result.function_evaluations = nfev;
  result.jacobian_evaluations = njev;
  result.has_jacobian_evaluations = analytic;
  result.iterations = analytic ? njev : nfev / (corpus.n + 1);
  result.usable = status >= 1 && status <= 4 && result.lre >= 4.0;
  return result;
}
#endif

#ifdef LEVMAR_HAVE_CERES
template <class ResidualFn, class JacobianFn>
class CeresAnalyticCost final : public ceres::DynamicCostFunction {
public:
  CeresAnalyticCost(const CorpusProblem &corpus, ResidualFn residual,
                    JacobianFn jacobian)
      : corpus_(corpus), residual_(std::move(residual)),
        jacobian_(std::move(jacobian)),
        jacobian_column_major_(corpus.m * corpus.n) {
    AddParameterBlock(corpus.n);
    SetNumResiduals(corpus.m);
  }

  bool Evaluate(double const *const *parameters, double *residuals,
                double **jacobians) const override {
    const auto x =
        ConstVectorView<std::dynamic_extent>(parameters[0], corpus_.n);
    if (const auto evaluated =
            residual_(x, VectorView<std::dynamic_extent>(residuals, corpus_.m));
        !evaluated) {
      return false;
    }
    if (jacobians == nullptr || jacobians[0] == nullptr) {
      return true;
    }
    if (const auto evaluated = jacobian_(
            x, MatrixView<std::dynamic_extent, std::dynamic_extent>(
                   jacobian_column_major_.data(), corpus_.m, corpus_.n));
        !evaluated) {
      return false;
    }
    for (Index row = 0; row < corpus_.m; ++row) {
      for (Index col = 0; col < corpus_.n; ++col) {
        jacobians[0][row * corpus_.n + col] =
            jacobian_column_major_[row + col * corpus_.m];
      }
    }
    return true;
  }

private:
  const CorpusProblem &corpus_;
  ResidualFn residual_;
  JacobianFn jacobian_;
  mutable std::vector<double> jacobian_column_major_;
};

template <Index M, Index N, class ResidualFn> struct CeresAutoDiffCost {
  ResidualFn residual;

  template <class Scalar> bool operator()(const Scalar *x, Scalar *r) const {
    const auto evaluated =
        residual(ConstVectorView<N, Scalar>(x, N), VectorView<M, Scalar>(r, M));
    return static_cast<bool>(evaluated);
  }
};

inline ceres::Solver::Options ceres_options() {
  ceres::Solver::Options options;
  options.minimizer_type = ceres::TRUST_REGION;
  options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
  options.linear_solver_type = ceres::DENSE_QR;
  options.dense_linear_algebra_library_type = ceres::EIGEN;
  options.num_threads = 1;
  options.use_nonmonotonic_steps = false;
  options.jacobi_scaling = true;
  options.logging_type = ceres::SILENT;
  // Match Ceres' upstream NIST driver, except for its explicit DENSE_QR mode.
  options.initial_trust_region_radius = 1e4;
  options.function_tolerance = std::numeric_limits<double>::epsilon();
  options.parameter_tolerance = std::numeric_limits<double>::epsilon();
  options.gradient_tolerance = std::numeric_limits<double>::epsilon();
  options.max_num_iterations = 10'000;
  return options;
}

template <class ResidualFn, class JacobianFn>
ExternalSolverResult
run_ceres_analytic_solver(const CorpusProblem &corpus, ResidualFn residual,
                          JacobianFn jacobian, const std::vector<double> &start,
                          const std::vector<double> &certified) {
  const auto start_time = Clock::now();
  std::vector<double> x = start;
  ceres::Problem problem;
  problem.AddResidualBlock(
      new CeresAnalyticCost<ResidualFn, JacobianFn>(corpus, std::move(residual),
                                                    std::move(jacobian)),
      nullptr, x.data());
  ceres::Solver::Summary summary;
  ceres::Solve(ceres_options(), &problem, &summary);
  ExternalSolverResult result;
  result.solver = "ceres_analytic";
  result.seconds = elapsed_seconds(start_time, Clock::now());
  result.lre = log_relative_error(x, certified);
  result.iterations = summary.iterations.size();
  result.function_evaluations = summary.num_residual_evaluations;
  result.jacobian_evaluations = summary.num_jacobian_evaluations;
  result.linear_solves =
      summary.num_successful_steps + summary.num_unsuccessful_steps;
  result.accepted_steps = summary.num_successful_steps;
  result.rejected_steps = summary.num_unsuccessful_steps;
  result.has_jacobian_evaluations = true;
  result.has_linear_solves = true;
  result.has_accepted_steps = true;
  result.has_rejected_steps = true;
  result.usable = summary.IsSolutionUsable() && result.lre >= 4.0;
  return result;
}

template <Index M, Index N, class ResidualFn>
ExternalSolverResult
run_ceres_autodiff_solver(ResidualFn residual, const std::vector<double> &start,
                          const std::vector<double> &certified) {
  const auto start_time = Clock::now();
  std::vector<double> x = start;
  ceres::Problem problem;
  using Cost = CeresAutoDiffCost<M, N, ResidualFn>;
  problem.AddResidualBlock(new ceres::AutoDiffCostFunction<Cost, M, N>(
                               new Cost{std::move(residual)}),
                           nullptr, x.data());
  ceres::Solver::Summary summary;
  ceres::Solve(ceres_options(), &problem, &summary);
  ExternalSolverResult result;
  result.solver = "ceres_autodiff";
  result.seconds = elapsed_seconds(start_time, Clock::now());
  result.lre = log_relative_error(x, certified);
  result.iterations = summary.iterations.size();
  result.function_evaluations = summary.num_residual_evaluations;
  result.jacobian_evaluations = summary.num_jacobian_evaluations;
  result.linear_solves =
      summary.num_successful_steps + summary.num_unsuccessful_steps;
  result.accepted_steps = summary.num_successful_steps;
  result.rejected_steps = summary.num_unsuccessful_steps;
  result.has_jacobian_evaluations = true;
  result.has_linear_solves = true;
  result.has_accepted_steps = true;
  result.has_rejected_steps = true;
  result.usable = summary.IsSolutionUsable() && result.lre >= 4.0;
  return result;
}
#endif
