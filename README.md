# levmar

Small dense nonlinear least-squares experiments with dynamic and static callback paths.

Recommended benchmark/accuracy flags:
- C++: `-O3 -march=native -ffp-contract=off`

## Build

Configure and build locally with CMake:

```sh
cmake -S . -B build
cmake --build build
```

The conformance runner target is `levmar_nist_runner` when
`LEVMAR_BUILD_CONFORMANCE_RUNNER=ON`.

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
template <class Residual, Index M, Index N>
concept ResidualCallable =
    requires(Residual residual, ConstVectorView<N> x, VectorView<M> r) {
      { residual(x, r) } -> std::same_as<ErrorOrVoid>;
    };

template <class Jacobian, Index M, Index N>
concept JacobianCallable =
    requires(Jacobian jacobian, ConstVectorView<N> x, MatrixView<M, N> J) {
      { jacobian(x, J) } -> std::same_as<ErrorOrVoid>;
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

```cpp
#include <levmar/lm.h>

#include <cmath>
#include <string>
#include <vector>

int main() {
  const std::vector<double> x_data{0.25, 0.50, 0.75, 1.00};
  const std::vector<double> y_data{0.28, 0.39, 0.46, 0.51};

  auto residual = [&](ConstVectorView<std::dynamic_extent> x,
                      VectorView<std::dynamic_extent> r) -> ErrorOrVoid {
    for (Index i = 0; i < x_data.size(); ++i) {
      r[i] = x[0] * (1.0 - std::exp(-x[1] * x_data[i])) - y_data[i];
    }
    return {};
  };

  auto jacobian = [&](ConstVectorView<std::dynamic_extent> x,
                      MatrixView<std::dynamic_extent, std::dynamic_extent> J) -> ErrorOrVoid {
    for (Index i = 0; i < x_data.size(); ++i) {
      const double xv = x_data[i];
      const double e = std::exp(-x[1] * xv);
      J[i, 0] = 1.0 - e;
      J[i, 1] = x[0] * xv * e;
    }
    return {};
  };

  auto problem = make_dynamic_problem(x_data.size(), 2, residual, jacobian);

  Options options;
  Result result;
  LMWorkspace<std::dynamic_extent, std::dynamic_extent> work;

  const std::vector<double> beta0{0.9, 1.5};
  LMSolveContext<std::dynamic_extent,
                 std::dynamic_extent,
                 decltype(residual),
                 decltype(jacobian)> context(problem, options, result, work, beta0);

  if (auto validation = validate_context(context); !validation) {
    return 1;
  }

  std::ranges::copy(context.x, work.x_current.view().begin());

  if (auto residual_result = evaluate_residual(context); !residual_result) {
    return 1;
  }

  if (auto jacobian_result = evaluate_jacobian(context); !jacobian_result) {
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

  auto residual = [&](ConstVectorView<2> x, VectorView<6> r) -> ErrorOrVoid {
    for (Index i = 0; i < 6; ++i) {
      r[i] = x[0] * (1.0 - std::exp(-x[1] * x_data[i])) - y_data[i];
    }
    return {};
  };

  auto jacobian = [&](ConstVectorView<2> x, MatrixView<6, 2> J) -> ErrorOrVoid {
    for (Index i = 0; i < 6; ++i) {
      const double xv = x_data[i];
      const double e = std::exp(-x[1] * xv);
      J[i, 0] = 1.0 - e;
      J[i, 1] = x[0] * xv * e;
    }
    return {};
  };

  auto problem = make_problem<6, 2>(residual, jacobian);

  Options options;
  Result result;
  LMWorkspace<6, 2> work;

  const std::array<double, 2> beta0{0.9, 1.5};
  LMSolveContext<6, 2, decltype(residual), decltype(jacobian)> context(
      problem, options, result, work, beta0);

  if (auto validation = validate_context(context); !validation) {
    return 1;
  }

  std::ranges::copy(context.x, work.x_current.view().begin());

  if (auto residual_result = evaluate_residual(context); !residual_result) {
    return 1;
  }

  if (auto jacobian_result = evaluate_jacobian(context); !jacobian_result) {
    return 1;
  }

  return 0;
}
```

## Notes

1. `ConstVectorView<N>` and `VectorView<M>` are thin aliases over `std::span`.
2. `MatrixView<M, N>` is a thin alias over `std::mdspan` using `std::layout_left`.
3. `x` is immutable input in `LMSolveContext`; the workspace owns `x_current` and `x_trial`.
4. The examples above exercise residual and Jacobian evaluation directly. A top-level `solve(...)` entry point is still being wired through the refactor.
