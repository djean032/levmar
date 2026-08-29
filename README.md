# levmar

Small dense nonlinear least-squares experiments with dynamic and static callback paths.

Recommended benchmark/accuracy flags:
- C++: `-O3 -march=native -fno-fast-math -ffp-contract=off`

Clang and Intel are the recommended benchmark compilers. GCC is a valid
reference compiler, but is slower in the current DAG benchmark results. Use
the same portable flags with Intel.

## Build

Configure and build locally with CMake:

```sh
cmake -S . -B build
cmake --build build
```

This also exports `build/compile_commands.json` for editor and LSP tooling.

The conformance runner target is `levmar_nist_runner` when
`LEVMAR_BUILD_CONFORMANCE_RUNNER=ON`.

## Benchmarking

Build and run the NIST conformance benchmark with the portable flags above:

```sh
python3 scripts/benchmark.py
python3 scripts/benchmark.py --iterations 100 --require-external
python3 scripts/benchmark.py --compiler icpx --env /path/to/compiler-env.sh
```

The script configures a Release `build-benchmark/`, builds
`levmar_nist_runner`, verifies the corpus, and writes `nist-validation.csv`,
`nist-solver-benchmark.csv`, and `nist-report.md` to `benchmark-results/` by
default. The solver table compares levmar analytic, graph autodiff, and static
dual autodiff paths with Ceres `DENSE_QR` analytic/autodiff and MINPACK
`lmder`/`lmdif` when those optional dependencies are installed.
It reports time, function and Jacobian evaluations, linear solves, and
accepted/rejected steps; unavailable MINPACK counters are `nan`.
Use `--build-dir` and `--output-dir` to select different locations. `--env`
sources a shell setup file before CMake invokes the requested compiler.

For per-start solver work across levmar and optional external solvers:

```sh
python3 scripts/benchmark.py \
  --solver-work \
  --iterations 1 \
  --build-dir build-external-ninja \
  --output-dir /tmp/levmar-work
```

This writes `nist-solver-work.csv`, including termination, cost, evaluation,
linear-solve, accepted-step, and rejected-step counters. For targeted internal
LM diagnostics:

```sh
python3 scripts/benchmark.py \
  --controller-trace \
  --iterations 1 \
  --build-dir build-external-ninja \
  --output-dir /tmp/levmar-controller-trace
```

The resulting `nist-controller-trace.csv` records selected NIST trajectories,
including trial costs, gain ratio, lambda, parameter and damping scales, step
vectors, parameter vectors, gradients, and Jacobian column norms.

The solver-work workflow can target one independent runner while iterating on
one backend:

```sh
python3 scripts/benchmark.py --solver-work --runner levmar
python3 scripts/benchmark.py --solver-work --runner ceres
python3 scripts/benchmark.py --solver-work --runner minpack
```

`--runner all` is the default and builds all three runners before merging their
rows. The Ceres and MINPACK runners compile and execute only their respective
adapter paths; the levmar runner contains the conformance and controller-trace
paths without external solver dependencies.

The runner exposes separate modes for direct use:

```sh
levmar_nist_runner conformance/nist_nls/corpus /tmp/validation.csv 1 --validate
levmar_nist_runner conformance/nist_nls/corpus /tmp/validation.csv 40 \
  /tmp/solvers.csv --benchmark-solvers
```

External solver benchmarks are optional. Enable them with
`-DLEVMAR_BUILD_EXTERNAL_BENCHMARKS=ON`; CMake discovers `Ceres::ceres` and
`cminpack::cminpack` independently.

For manual CMake benchmark builds, pass the flags through the runner-only
`LEVMAR_BENCHMARK_COMPILE_OPTIONS` cache variable:

```sh
cmake -S . -B build-benchmark -DCMAKE_BUILD_TYPE=Release \
  '-DLEVMAR_BENCHMARK_COMPILE_OPTIONS=-O3 -march=native -fno-fast-math -ffp-contract=off'
cmake --build build-benchmark --target levmar_nist_runner
```

## Current Status

The library currently provides residual/Jacobian callback infrastructure,
static and dynamic storage paths, user-supplied and finite-difference
Jacobians, and internal DAG-based forward-mode autodiff.

Autodiff supports unary negation, `exp`, `log`, `log1p`, `expm1`, `sqrt`,
`sin`, `cos`, and `tan`; binary addition, subtraction, multiplication,
division, power, and `atan2`. Residuals selected for `JacobianMode::AutoDiff`
must be scalar-generic so they operate on internal autodiff views as well as
standard double-based views. `levmar/lm.h` is the public umbrella header; the
implementation headers under `levmar/internal/` are not public API.

The internal policy-templated Levenberg-Marquardt solve path is implemented and
is exercised by the NIST runner. Its current experiment uses fixed first-Jacobian
coordinate scales, damping scales floored by those coordinate scales, and an
upper damping-diagonal cap. The public solver API remains experimental; see
`plans.md` for the active trust-region controller work.

## Install

Install to a user-local prefix without touching system directories:

```sh
cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$HOME/.local"
cmake --build build
cmake --install build
```

This installs the public header as `levmar/lm.h` under the chosen prefix and
exports a CMake package so downstream projects can use:

```cmake
find_package(levmar CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE levmar::levmar)
```

## API Shape

The public callback shapes are:

```cpp
template <class Residual, levmar::Index M, levmar::Index N>
concept ResidualCallable =
    requires(Residual residual, levmar::ConstVectorView<N> x,
             levmar::VectorView<M> r) {
      { residual(x, r) } -> std::same_as<levmar::ErrorOrVoid>;
    };

template <class Jacobian, levmar::Index M, levmar::Index N>
concept JacobianCallable =
    requires(Jacobian jacobian, levmar::ConstVectorView<N> x,
             levmar::MatrixView<M, N> J) {
      { jacobian(x, J) } -> std::same_as<levmar::ErrorOrVoid>;
    };
```

That means user callbacks fill solver-owned residual and Jacobian storage directly.

## Dynamic Vs Static

Benchmark takeaway:

1. For small problems, the dynamic and static paths are close enough that you usually do not need to think about it.
2. For larger fixed-size problems, the static path becomes worthwhile, especially for finite-difference Jacobians.
3. A good default is: write the clearest model first, then move to static callbacks only if the problem is fixed-size and performance-sensitive.

## Dynamic Example

This is the most straightforward style. Dimensions are provided at runtime.
The direct evaluation objects below are temporary implementation APIs; the
future normal user path is `levmar::solve(...)`.

```cpp
#include <levmar/lm.h>

#include <cmath>
#include <string>
#include <vector>

int main() {
  const std::vector<double> x_data{0.25, 0.50, 0.75, 1.00};
  const std::vector<double> y_data{0.28, 0.39, 0.46, 0.51};

  auto residual = [&](levmar::ConstVectorView<std::dynamic_extent> x,
                      levmar::VectorView<std::dynamic_extent> r)
      -> levmar::ErrorOrVoid {
    for (levmar::Index i = 0; i < x_data.size(); ++i) {
      r[i] = x[0] * (1.0 - std::exp(-x[1] * x_data[i])) - y_data[i];
    }
    return {};
  };

  auto jacobian = [&](levmar::ConstVectorView<std::dynamic_extent> x,
                      levmar::MatrixView<std::dynamic_extent,
                                         std::dynamic_extent> J)
      -> levmar::ErrorOrVoid {
    for (levmar::Index i = 0; i < x_data.size(); ++i) {
      const double xv = x_data[i];
      const double e = std::exp(-x[1] * xv);
      J[i, 0] = 1.0 - e;
      J[i, 1] = x[0] * xv * e;
    }
    return {};
  };

  auto problem = levmar::make_dynamic_problem(x_data.size(), 2, residual, jacobian);

  levmar::Options options;
  levmar::Result result;
  levmar::detail::LMWorkspace<std::dynamic_extent, std::dynamic_extent> work;

  const std::vector<double> beta0{0.9, 1.5};
  levmar::detail::LMSolveContext<std::dynamic_extent,
                                 std::dynamic_extent,
                                 decltype(residual),
                                 decltype(jacobian)> context(
      problem, options, result, work, beta0);

  if (auto validation = levmar::detail::validate_context(context); !validation) {
    return 1;
  }

  std::ranges::copy(context.x, work.x_current.view().begin());

  if (auto residual_result = levmar::detail::evaluate_residual(context);
      !residual_result) {
    return 1;
  }

  if (auto jacobian_result =
          levmar::detail::evaluate_jacobian<levmar::UserJacobian>(context);
      !jacobian_result) {
    return 1;
  }

  return 0;
}
```

## Static Example

If the problem shape is fixed and known at compile time, use concrete extents directly.

```cpp
#include <levmar/lm.h>

#include <array>
#include <cmath>
#include <string>

int main() {
  constexpr std::array<double, 6> x_data{0.25, 0.50, 0.75, 1.00, 1.25, 1.50};
  constexpr std::array<double, 6> y_data{0.28, 0.39, 0.46, 0.51, 0.54, 0.56};

  auto residual = [&](levmar::ConstVectorView<2> x, levmar::VectorView<6> r)
      -> levmar::ErrorOrVoid {
    for (levmar::Index i = 0; i < 6; ++i) {
      r[i] = x[0] * (1.0 - std::exp(-x[1] * x_data[i])) - y_data[i];
    }
    return {};
  };

  auto jacobian = [&](levmar::ConstVectorView<2> x,
                      levmar::MatrixView<6, 2> J) -> levmar::ErrorOrVoid {
    for (levmar::Index i = 0; i < 6; ++i) {
      const double xv = x_data[i];
      const double e = std::exp(-x[1] * xv);
      J[i, 0] = 1.0 - e;
      J[i, 1] = x[0] * xv * e;
    }
    return {};
  };

  auto problem = levmar::make_problem<6, 2>(residual, jacobian);

  levmar::Options options;
  levmar::Result result;
  levmar::detail::LMWorkspace<6, 2> work;

  const std::array<double, 2> beta0{0.9, 1.5};
  levmar::detail::LMSolveContext<6, 2, decltype(residual), decltype(jacobian)>
      context(problem, options, result, work, beta0);

  if (auto validation = levmar::detail::validate_context(context); !validation) {
    return 1;
  }

  std::ranges::copy(context.x, work.x_current.view().begin());

  if (auto residual_result = levmar::detail::evaluate_residual(context);
      !residual_result) {
    return 1;
  }

  if (auto jacobian_result =
          levmar::detail::evaluate_jacobian<levmar::UserJacobian>(context);
      !jacobian_result) {
    return 1;
  }

  return 0;
}
```

## Notes

1. `levmar::ConstVectorView<N>` and `levmar::VectorView<M>` are thin aliases
   over `std::span`.
2. `levmar::MatrixView<M, N>` is a thin alias over `std::mdspan` using
   `std::layout_left`.
3. `levmar::detail` is unsupported implementation API. The examples use it
   only because a top-level `levmar::solve(...)` entry point is not implemented
   yet.
