# Repository Guidelines

## Project Structure & Module Organization

`levmar/lm.h` is the header-only C++23 nonlinear least-squares implementation.
`conformance/nist_nls/cpp_runner.cpp` is the NIST test and benchmark harness;
its corpus directories contain inputs and expected residual/Jacobian CSVs, and
`generator/` recreates them. `external_benchmark_adapters.h` provides optional
Ceres and CMinpack adapters. `conformance/benchmark_corpus/` contains synthetic
performance cases. `README.md` documents the public API; `plans.md` records
current design work.

## Build, Test, and Development Commands

Use the benchmark wrapper for the complete NIST build and comparison workflow:

```sh
python3 scripts/benchmark.py \
  --solver-work \
  --iterations 1 \
  --build-dir build-external-ninja \
  --output-dir /tmp/levmar-work
```

The wrapper configures Ninja, builds available levmar/Ceres/CMinpack runners,
and writes `nist-solver-work.csv`. Use a larger iteration count for timing work.
Generate targeted controller diagnostics with:

```sh
python3 scripts/benchmark.py \
  --controller-trace \
  --iterations 1 \
  --build-dir build-external-ninja \
  --output-dir /tmp/levmar-controller-trace
```

This writes `nist-controller-trace.csv`. Regenerate NIST fixtures with:

```sh
python3 conformance/nist_nls/generator/generate.py
```

This generator downloads upstream NIST data and rewrites checked-in corpus
files, so review the resulting diff carefully.

The runner targets are independent: `levmar_nist_runner`,
`levmar_nist_ceres_runner`, and `levmar_nist_minpack_runner`. During backend
iteration, use `--solver-work --runner levmar`, `ceres`, or `minpack` to avoid
the aggregate target.

## Coding Style & Naming Conventions

Use C++23 and run `clang-format -i` on changed C++ files. Use two-space
indentation, attached braces, and an 80-column limit. Use `PascalCase` for
types/concepts and `snake_case` for functions/locals. Keep public headers
self-contained and avoid allocations in fixed-size paths.

## Modification Authorization

OpenCode may modify tests, benchmarks, conformance harnesses, and supporting
documentation. Do not modify implementation or public API files under `levmar/`
without explicit user authorization for that library change.

## Testing Guidelines

No separate unit-test framework is configured. Add direct numerical coverage to
`tests/autodiff_graph_tests.cpp`; add NIST coverage or stable corpus fixtures
when appropriate. Test static, dynamic, and mixed extents for shared templates.
Preserve `model_value - observed_value`, column-major Jacobians, and tolerance
checks.

For LM controller or dense backend changes, run the controller trace and
`--solver-work`; compare MGH10/start1, MGH17/start1, Gauss1/start2, Rat43/start1,
and tall synthetic cases with the baseline. Report factorization count and
lambda solves separately. Inspect compiler vectorization diagnostics before
adding intrinsics, alignment assumptions, or nonportable aliasing annotations.
Do not commit generated executables or ad hoc CSVs.

## Commit & Pull Request Guidelines

Use short, imperative, lowercase commit summaries. Keep commits focused and
mention regenerated corpus data explicitly. Pull requests should state numerical
or API changes, exact validation commands, and benchmark comparisons for
performance-sensitive work.
