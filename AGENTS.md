# Repository Guidelines

## Project Structure & Module Organization

`src/lm.h` contains the header-only C++23 nonlinear least-squares implementation,
including static- and dynamic-extent APIs. `conformance/nist_nls/cpp_runner.cpp`
is the primary executable test and benchmark harness. Its checked-in
`corpus/` directories hold NIST inputs and expected residual/Jacobian CSVs;
`generator/` contains the Python scripts that recreate them.
`conformance/benchmark_corpus/` stores larger synthetic performance cases.
`README.md` documents the public API, while `plans.md` records design work that
is not necessarily implemented.

## Build, Test, and Development Commands

There is currently no CMake or Make target. Build directly from the repository
root:

```sh
clang++ -std=c++23 -O3 -march=native -ffp-contract=off \
  conformance/nist_nls/cpp_runner.cpp -o /tmp/levmar-cpp-runner
/tmp/levmar-cpp-runner conformance/nist_nls/corpus /tmp/results.csv 1
```

The runner compares dynamic and static residuals and Jacobians with the corpus;
the final argument is the benchmark iteration count. Use a larger count for
timing work. Regenerate NIST fixtures with:

```sh
python3 conformance/nist_nls/generator/generate.py
```

This generator downloads upstream NIST data and rewrites checked-in corpus
files, so review the resulting diff carefully.

## Coding Style & Naming Conventions

Use C++23 and run `clang-format -i` on changed C++ files. The repository
configuration uses two-space indentation, attached braces, and an 80-column
limit. Follow existing names: `PascalCase` for types and concepts, `snake_case`
for functions and local variables, and descriptive template extents such as
`M`, `N`, `Rows`, and `Cols`. Keep the public API header self-contained and
avoid allocations in fixed-size paths.

## Testing Guidelines

No separate unit-test framework is configured. Add numerical coverage to
`cpp_runner.cpp` and stable fixtures under the appropriate corpus directory.
Test static, dynamic, and mixed extents when changing shared templates. Preserve
the residual convention `model_value - observed_value`, column-major Jacobian
layout, and existing tolerance checks. Do not commit generated executables or
ad hoc output CSVs.

## Commit & Pull Request Guidelines

Recent history uses short, imperative, lowercase summaries, for example
`started templating refactor`. Keep each commit focused and mention regenerated
corpus data explicitly. Pull requests should describe API or numerical behavior
changes, list the exact validation command, and include benchmark comparisons
for performance-sensitive work. Link relevant issues and call out any changed
tolerances or generated files.
