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

The pivoted path uses Moré-style lambda selection with `parl`/`paru` bounds,
warm-start state, correction iterations, and controller trace counters. The
remaining work is controller-state correctness, convergence accuracy, rank
diagnosis, and measured performance work.

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

## 1. LM Controller State And Safeguards

### Accepted Lambda Warm Start

`selected_lambda` must represent the previous accepted outer iteration only.

1. Recompute `parl` and `paru` from the current factorization and trust radius
   on every outer iteration.
2. Clamp the previous accepted lambda into that fresh bracket for the Moré
   initial candidate.
3. Keep a trial lambda local while evaluating its residual and gain ratio.
4. Commit it as the warm-start value only after acceptance.
5. Never let a rejected trial update the next iteration's warm start.
6. Keep rejected-trial lambda values in trace records so controller behavior
   remains diagnosable.

Provide a benchmark-only cold-start mode in the conformance runner. It must
not become a permanent public `LMOptions` switch. Compare warm and cold starts
by Moré iterations, total linear solves, accepted/rejected steps,
factorizations, and elapsed time.

### Bracket-Preserving Safeguards

The initial solve at `min_lambda` remains only a near-Gauss-Newton radius test.
It must not become the restart point for a failed Moré update.

For cached pivoted QR:

1. Maintain the current positive `[parl, paru]` bracket and the most recent
   evaluated lambda throughout the inner solve.
2. Retain the best feasible endpoint, defined by a scaled step inside the
   trust radius and closest to its boundary.
3. Use the Moré correction as the primary proposal.
4. If the correction is non-finite or outside the bracket, evaluate the
   geometric midpoint of the existing bracket.
5. If the Moré iteration cap is reached, continue bounded safeguarded
   interpolation inside that same bracket. Do not restart bisection from
   `min_lambda` or the previous accepted lambda.
6. Bound total inner work, initially at 20 linear solves after the initial
   radius test. On numeric bracket stagnation, return the best feasible
   endpoint already evaluated.
7. Keep generic `DampedQr` bisection unchanged. This replacement is specific
   to cached pivoted QR.

The target trust-region contract is

```text
0.9 * delta <= ||D p|| <= delta
```

If the standard symmetric Moré tolerance is retained temporarily, record and
benchmark oversized selections explicitly. Do not silently accept a different
radius contract.

Trace `lmpar_iterations`, safeguarded-refinement count, fallback use, total
linear solves, and the final bracket. A fallback means safeguarded work inside
the Moré bracket, never an old-search restart.

### Radius Updates

Keep outer trust-radius updates separate from lambda selection. Rejection must
have one documented shrink rule. The current `rho <= 0.25` update and the
separate rejected-step shrink rule compete; choose one policy, test it at the
minimum radius, and report the resulting termination reason.

Reconcile `initial_trust_region_radius` with initialization behavior. It is a
validated option, while the current solver initializes from
`initial_trust_region_factor * ||D x||`. Either honor an explicit initial
radius when configured or remove/rename the unused option; do not leave two
apparently authoritative initialization controls.

## 2. Convergence Accuracy And LRE

### Relative Parameter Step

Replace raw absolute Euclidean `SmallStep` termination with an
original-coordinate componentwise metric:

```text
relative_step = max_j |p_j| / max(1, |x_j|)
```

The denominator is a parameter-accuracy convention, not Jacobian-column
scaling. Do not use `work.scale` as a substitute for parameter units.

Store the metric in results and benchmark output. Add static, dynamic, and
mixed-extent coverage with parameters below one and far above one.

### Gradient-Gated Termination

A tiny step or a tiny cost change alone is not successful convergence.

1. `relative_step <= step_tolerance` and a small gradient may terminate as
   `SmallStep`.
2. A tiny step with a non-small gradient terminates as `Stalled`.
3. A small absolute or relative cost reduction requires a small gradient
   before terminating as `SmallCostReduction`.
4. Apply the cost-and-gradient rule both after accepted steps and when the
   predicted reduction is non-positive.
5. Define and test the gradient multiplier explicitly instead of relying on
   an undocumented constant.

`Stalled` must be distinct from numerical failure, damping limit, and maximum
iteration/function-evaluation limits.

### Near-Convergence Lambda Accuracy

After stopping semantics and rank diagnostics have baseline data, benchmark a
tighter Moré target near convergence:

1. normal iterations: 10% radius tolerance
2. near convergence: 1-2% radius tolerance

Define near convergence from the relative step, final cost change, gradient,
or radius; record which condition activated the tighter target. Do not apply
the tighter tolerance globally before measuring extra inner linear solves.

### LRE Diagnosis

For low-LRE NIST cases, record:

1. final residual cost
2. final gradient infinity norm
3. relative parameter-step metric
4. termination reason
5. per-parameter relative error and aggregate LRE

Compare residual accuracy and parameter accuracy. A low parameter LRE with a
small final cost can be a flat-valley/identifiability result rather than a
controller failure.

## 3. CPQR Rank And Conditioning Diagnostics

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

## 4. SVD As A Reference

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

## 5. Compile-Time Trace Policy

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

## 6. Measured Performance Work

Profile before optimizing. Use solver-work, controller traces, and compiler
vectorization diagnostics to identify the actual dynamic-size bottleneck.

Potential follow-up work, only when measurements justify it:

1. specialize damping-row QR updates instead of the generic Givens row path
2. add CPQR column-norm downdates with periodic exact recomputation safeguards
3. optimize dynamic-size workspace/layout and allocation behavior
4. benchmark dynamic versus static and mixed extents separately

Do not add intrinsics, aliasing assumptions, custom allocators, or BLAS/LAPACK
production dependencies without profile evidence.

## 7. Autodiff Evaluation Roadmap

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

Every controller, stopping, or rank change must run direct static, dynamic,
and mixed-extent coverage, then:

```sh
cmake --build build-external-ninja --target levmar_autodiff_graph_tests
ctest --test-dir build-external-ninja --output-on-failure
python3 scripts/benchmark.py --controller-trace --runner householder \
  --iterations 1 --build-dir build-external-ninja \
  --output-dir /tmp/levmar-controller-trace
python3 scripts/benchmark.py --solver-work --runner householder \
  --iterations 1 --build-dir build-external-ninja \
  --output-dir /tmp/levmar-householder-work
```

Compare MGH10/start1, MGH17/start1, Gauss1/start2, Rat43/start1, and tall
synthetic/rank-deficient corpora. Track solved/pass count, mean and median LRE,
final cost, termination, final gradient, relative step, iterations, accepted
and rejected steps, factorization count, total linear solves, Moré iterations,
and runtime. Use multiple timing iterations for performance decisions.

Keep the old lambda controller only as a temporary benchmark/reference policy.
Do not retain it as permanent production compatibility code. Do not commit
generated executables or ad hoc CSV output.

## Deferred Work

The following work remains deferred until controller state, convergence
semantics, rank diagnostics, and benchmark baselines are stable:

1. production SVD backend and automatic CPQR-to-SVD fallback
2. BLAS/LAPACK-backed dense production kernels
3. geodesic acceleration
4. analytic and automatic directional second derivatives (`Dual2`)
5. centered finite-difference directional-second/JVP fallbacks
6. matrix-free JVP/HVP and Newton-Krylov interfaces
7. ODE directional sensitivity and second-order adjoint methods
