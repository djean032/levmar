# Plans

## Current State

`levmar` is a header-only C++23 nonlinear least-squares solver with static,
dynamic, and mixed-extent APIs. It has a policy-specialized LM controller,
cached autodiff graph evaluation, fixed-width direct dual autodiff, and two
dense QR paths:

1. `DampedQr` is the default streamed-Givens backend.
2. `PivotedHouseholderQr` is an opt-in cached CPQR backend. It factorizes a
   scaled Jacobian once per accepted model, reuses the factor for positive
   lambda solves, reports factorization counts, and provides a rank-aware
   Gauss-Newton basic solution.

The pivoted path uses GN-first selection, Moré-style `parl`/`paru` bounds,
accepted-lambda warm starts, safeguarded refinement, and controller trace
counters. It distinguishes a rejected-radius exhaustion
(`TrustRegionTooSmall`) from lambda-selection exhaustion (`DampingLimit`).
Accepted convergence uses gradient, scale-relative step, and actual plus
predicted cost-reduction checks.

## Invariants

All work must preserve:

1. column-major Jacobian storage and `model_value - observed_value` residuals
2. allocation-free fixed-size paths
3. benchmark flags `-fno-fast-math` and `-ffp-contract=off`
4. cached factor validity across rejected trials, invalidating only after an
   accepted model/Jacobian/scaling refresh
5. factorization count separate from total linear solves
6. rank handling separate from trust-radius heuristics and lambda floors
7. no production external linear-algebra dependency

## 1. Solver Structure Refactor

Split the solver into cohesive header-only units without changing public API,
numerical behavior, allocations, or test sources. `solve()` must become the
outer lifecycle/orchestration function.

Target layout:

```text
levmar/internal/
  solver.h
  solver_initialization.h
  solver_model.h
  solver_scaling.h
  solver_linear.h
  solver_step_controller.h
  linear/
    givens_qr.h
    damped_qr.h
    pivoted_householder_qr.h
```

Each checkpoint moves one cohesive production unit, updates only production
includes/call sites, builds, and runs the existing tests before the next move.
Do not reorganize test sources until the refactor is complete.

1. Capture test, solver-work, and Householder controller-trace baselines.
2. Extract the shared Givens QR row-update primitive.
3. Extract the `DampedQr` backend.
4. Extract the pivoted Householder backend, including factorization, rank-aware
   GN, and cached damped solves.
5. Add the linear facade and move backend preparation/dispatch there.
6. Extract scaling helpers.
7. Extract initialization and model lifecycle helpers.
8. Extract the unchanged step controller as one cohesive unit.
9. Extract lifecycle/convergence helpers until `solve()` owns only sequencing,
   iteration accounting, and finalization.
10. Remove unused includes and format the changed headers.

Do not split `try_lm_step()` into free functions that require many coupled
references or callback parameters. A later controller-internal state object is
appropriate only after the mechanical split has stable baselines.

## 2. CPQR Rank And Conditioning Diagnostics

Keep the current rank-aware CPQR Gauss-Newton basic solution: zero trailing
transformed RHS entries beyond numerical rank and backsolve the leading rank
block. Positive-lambda solves remain untruncated.

Add a diagnostic separate from strict numerical rank:

```text
last_pivot_ratio = |R[n-1,n-1]| / |R[0,0]|
min_active_pivot_ratio = min_{k < numerical_rank} |R[k,k]| / |R[0,0]|
```

Trace numerical rank, rank threshold, pivot ratios, permutation, and weak
coordinate information. Check that numerical rank is a contiguous leading
block before relying on the rank-truncated Gauss-Newton solve.

Initially, weak conditioning is diagnostic only. Sweep candidate thresholds
such as `1e-6`, `1e-8`, and `1e-10` on real cases before defining a policy.

Add:

1. exact rank-deficient synthetic cases
2. near-rank-deficient epsilon sweeps
3. CPQR reconstruction and reflector-orthogonality checks
4. MGH17/start1 as the primary difficult rank regression

## 3. SVD As A Reference

Add SVD first as an independent validation/reference path, not an automatic
production backend.

For MGH17 starts, Lanczos cases, and other low-LRE cases, compare CPQR with
SVD and record:

```text
sigma_min / sigma_max
weakest right singular vector
final cost, parameter LRE, termination, and solve work
```

This distinguishes conditioning-limited parameter accuracy from controller
failures and identifies whether a minimum-norm rank-deficient solve is useful.
Only after this data exists should the project consider an SVD backend or a
CPQR-to-SVD fallback. Do not choose a fallback threshold in advance.

## 4. Compile-Time Trace Policy

After rank diagnostics are stable, introduce an internal compile-time trace
policy.

1. Normal solver and solver-work builds use `NoTrace`.
2. Controller-trace builds use `CollectTrace`.
3. In `NoTrace` builds, trace vectors, correlation diagnostics, trace counters,
   trace record construction, and CSV-facing state must be compiled out, not
   merely bypassed by a runtime null check.
4. Keep public solver API and fixed-size allocation guarantees unchanged.
5. Verify no-trace code generation and benchmark trace-off versus trace-on
   separately; trace-on performance is diagnostic, not production timing.

## 5. Measured Performance Work

Profile before optimizing. Use solver-work, controller traces, and compiler
vectorization diagnostics to identify the actual dynamic-size bottleneck.

Potential follow-up work, only when measurements justify it:

1. specialize damping-row QR updates instead of the generic Givens row path
2. add CPQR column-norm downdates with periodic exact recomputation safeguards
3. optimize dynamic-size workspace/layout and allocation behavior
4. benchmark dynamic versus static and mixed extents separately

Do not add intrinsics, aliasing assumptions, custom allocators, or BLAS/LAPACK
production dependencies without profile evidence.

## 6. Autodiff Evaluation Roadmap

Keep forward dual work separate from matrix-expression work.

1. Benchmark eager forward duals at derivative widths 2, 4, 8, 16, 32, 64,
   and 128.
2. Benchmark fused eager operations, especially dual multiply-add.
3. Prototype chunked fixed-width forward AD for medium and large dynamic
   parameter counts if the baseline supports it.
4. Consider expression templates only if eager and fused measurements show a
   material gap.
5. Any expression-template design must avoid recomputing primal subexpressions
   for each derivative lane; use cached primal values or a two-phase evaluator.

## Validation And Benchmarking

For every solver-structure checkpoint, build the affected runners and run the
existing test executable before proceeding:

```sh
cmake --build build-external-ninja --target levmar_autodiff_graph_tests
ctest --test-dir build-external-ninja --output-on-failure
cmake --build build-external-ninja --target levmar_nist_runner \
  levmar_nist_householder_runner
```

At baseline, after the linear facade, after the step-controller extraction, and
at completion, run:

```sh
python3 scripts/benchmark.py --controller-trace --runner householder \
  --iterations 1 --build-dir build-external-ninja \
  --output-dir /tmp/levmar-controller-trace
python3 scripts/benchmark.py --solver-work --runner householder \
  --iterations 1 --build-dir build-external-ninja \
  --output-dir /tmp/levmar-householder-work
```

Compare non-timing solver-work and trace fields against the baseline,
particularly MGH10/start1, MGH17/start1, Gauss1/start2, Rat43/start1, and tall
synthetic/rank-deficient corpora. Track solved/pass count, LRE, final cost,
termination, gradient, iterations, accepted/rejected steps, factorization
count, total linear solves, and Moré iterations. Use multiple timing iterations
only for performance decisions.

Do not commit generated executables or ad hoc CSV output.

## Deferred Work

The following work remains deferred until the structural refactor, rank
diagnostics, and benchmark baselines are stable:

1. production SVD backend and automatic CPQR-to-SVD fallback
2. BLAS/LAPACK-backed dense production kernels
3. geodesic acceleration
4. analytic and automatic directional second derivatives (`Dual2`)
5. centered finite-difference directional-second/JVP fallbacks
6. matrix-free JVP/HVP and Newton-Krylov interfaces
7. ODE directional sensitivity and second-order adjoint methods
