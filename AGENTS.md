# Repository Guidelines

## Project Structure & Module Organization

`levmar/lm.h` contains the header-only C++23 nonlinear least-squares implementation,
including static- and dynamic-extent APIs. `conformance/nist_nls/cpp_runner.cpp`
is the primary executable test and benchmark harness;
`external_benchmark_adapters.h` contains optional Ceres and CMinpack adapters.
Its checked-in `corpus/` directories hold NIST inputs and expected
residual/Jacobian CSVs; `generator/` contains the Python scripts that recreate
them.
`conformance/benchmark_corpus/` stores larger synthetic performance cases.
`README.md` documents the public API, while `plans.md` records design work that
is not necessarily implemented.

## Build, Test, and Development Commands

Use the benchmark wrapper for the complete NIST build and comparison workflow:

```sh
python3 scripts/benchmark.py \
  --solver-work \
  --iterations 1 \
  --build-dir build-external-ninja \
  --output-dir /tmp/levmar-work
```

The wrapper configures Ninja, builds the levmar/Ceres/CMinpack benchmark runners
when available, and writes `nist-solver-work.csv`. Use a larger iteration count
for timing work. Generate targeted LM controller diagnostics with:

```sh
python3 scripts/benchmark.py \
  --controller-trace \
  --iterations 1 \
  --build-dir build-external-ninja \
  --output-dir /tmp/levmar-controller-trace
```

This writes `nist-controller-trace.csv` for selected problematic NIST starts.
Regenerate NIST fixtures with:

```sh
python3 conformance/nist_nls/generator/generate.py
```

This generator downloads upstream NIST data and rewrites checked-in corpus
files, so review the resulting diff carefully.

The three runner targets are independent: `levmar_nist_runner`,
`levmar_nist_ceres_runner`, and `levmar_nist_minpack_runner`. During backend
iteration, use `--solver-work --runner levmar`, `ceres`, or `minpack` to avoid
building the aggregate target.

## Coding Style & Naming Conventions

Use C++23 and run `clang-format -i` on changed C++ files. The repository
configuration uses two-space indentation, attached braces, and an 80-column
limit. Follow existing names: `PascalCase` for types and concepts, `snake_case`
for functions and local variables, and descriptive template extents such as
`M`, `N`, `Rows`, and `Cols`. Keep the public API header self-contained and
avoid allocations in fixed-size paths.

## Modification Authorization

OpenCode may modify only tests, benchmarks, conformance harnesses, and their
supporting documentation. Do not modify library implementation or public API
files under `levmar/` unless the user explicitly authorizes that specific
library change.

## Testing Guidelines

No separate unit-test framework is configured. Add numerical coverage to
`cpp_runner.cpp` and stable fixtures under the appropriate corpus directory.
Test static, dynamic, and mixed extents when changing shared templates. Preserve
the residual convention `model_value - observed_value`, column-major Jacobian
layout, and existing tolerance checks. For LM-controller changes, run both the
targeted controller trace and `--solver-work`; compare MGH10/start1, MGH17/start1,
Gauss1/start2, and Rat43/start1 with the baseline. Do not commit generated
executables or ad hoc output CSVs.

## Commit & Pull Request Guidelines

Recent history uses short, imperative, lowercase summaries, for example
`started templating refactor`. Keep each commit focused and mention regenerated
corpus data explicitly. Pull requests should describe API or numerical behavior
changes, list the exact validation command, and include benchmark comparisons
for performance-sensitive work. Link relevant issues and call out any changed
tolerances or generated files.
