# Plans

## Current Direction

`levmar` is a header-only C++23 nonlinear least-squares solver with static,
dynamic, and mixed-extent APIs. The current implementation has a
policy-specialized Levenberg-Marquardt controller, cached autodiff graph
evaluation, fixed-width direct dual autodiff for small static parameter counts,
and a streamed-Givens damped-QR backend.

The next solver objective is a dependency-free dense linear-algebra backend
that closes the gap to Ceres and CMinpack without weakening rank-deficient
behavior or changing the current trust-region controller.

## Baseline And Constraints

Use the current solver-work report at `/tmp/levmar-work/nist-solver-work.csv`
as the active controller baseline. MGH10/start1 is the primary dense-work
outlier: the current controller is close in nonlinear iterations but repeats a
full QR factorization for every lambda trial.

The implementation must preserve:

1. column-major Jacobian storage and the residual convention
   `model_value - observed_value`
2. fixed-size allocation-free paths
3. `-fno-fast-math` and `-ffp-contract=off` numerical behavior in benchmark
   runners
4. existing trust-radius initialization, acceptance, and gain-ratio behavior
5. no external linear-algebra dependency

## Cached Pivoted QR Backend

Add `PivotedHouseholderQr` as an internal linear-algebra policy for A/B
benchmarking against `DampedQr`. Make it the default only after correctness and
performance validation.

### Stage 1: Policy And Workspace

Generalize static validation and the runtime `m >= n` check to accept both QR
policies. Keep `DampedQr` as the default policy until the new backend passes
the full comparison suite.

`PivotedHouseholderQrWorkspace<M, N>` owns:

1. an `M x N` packed column-major factor matrix containing `R` and reflector
   tails
2. `N` Householder coefficients
3. `N` pivot indices
4. contiguous `N x N` `R_base` and `R_work` matrices
5. transformed RHS, triangular-solve, and Givens-row scratch vectors
6. the pivoted effective damping diagonal
7. factor-valid state, numerical rank, and rank threshold

`R_base` is intentionally separate from packed storage: lambda trials copy a
small contiguous triangle instead of striding through an `M`-leading-dimension
matrix. The factor is valid after one model evaluation and remains valid across
rejected trials. Invalidate it only after a model, Jacobian, or scaling refresh
following an accepted step.

Add `factorization_count` to `Result` and solver-work output. Keep it separate
from `linear_solves`: one cached QR factorization can serve many lambda solves.

### Stage 2: Factor And Solve

For each evaluated, scaled model, factor once:

```text
J_s P = Q R
```

where `J_s` is the scaled Jacobian, `P` is the column permutation, and `R` is
upper triangular.

At each lambda, solve the augmented system without refactoring the tall
Jacobian:

```text
[ R_base                 ] y = [ Q^T(-r) ]
[ sqrt(lambda) D_permuted]     [     0     ]
```

Insert damping rows by Givens rotations, backsolve for pivoted scaled `y`, then
unpermute and convert to the public step coordinates. This makes lambda trials
`O(n^2)` rather than `O(m n^2)`.

Use initial exact trailing column norms for pivot selection. Record numerical
rank from `abs(R(j,j))` with:

```text
rank_threshold = rank_tolerance_multiplier * epsilon * max(m, n) * abs(R(0,0))
```

The multiplier should be a validated runtime option, defaulting to `32.0`.
Rank is diagnostic only: do not truncate a positive-lambda damped step.

Factorization steps:

1. Copy `J / scale` to packed workspace one column at a time.
2. At factor step `k`, recompute each active trailing column norm exactly and
   select the largest pivot.
3. Swap complete packed columns and permutation entries.
4. Form a Householder reflector in the pivot column and store its tail and
   coefficient in place.
5. Apply the reflector to each remaining contiguous target column and to
   `-r`.
6. Copy the upper triangle to `R_base`, cache the first `n` entries of
   `Q^T(-r)`, and permute the effective damping diagonal.

For a lambda trial, copy `R_base` to `R_work`, insert the permuted damping rows
with Givens rotations, backsolve in pivoted scaled coordinates, then unpermute
and unscale into `work.step`. Stage 1 retains the current bisection controller
unchanged, so factorization correctness and controller behavior can be compared
independently.

## Lambda Selection

After Stage 2 matches the bisection baseline, replace repeated pure bisection
with an `lmpar`-style safeguarded update.

1. Solve at the minimum damping to test whether the Gauss-Newton-like step is
   inside the trust region.
2. If not, bracket lambda using the previous accepted lambda as the initial
   upper candidate when possible.
3. For a trial factor `R_lambda`, obtain the norm derivative from:

   ```text
   (R^T R + lambda D^T D) y' = -D^T D y
   ```

   using two additional triangular solves with `R_lambda`.
4. Propose a Newton correction toward `0.9 * trust_radius`.
5. Keep lower and upper lambda bounds. Reject non-finite, non-monotone, or
   out-of-bracket Newton proposals and use geometric-mean bisection instead.
6. Accept a step in the existing radius band; do not seek an exact boundary.

This preserves bisection as the robustness fallback, while normally requiring
only a few small triangular solves per selected step. It must not add model or
residual evaluations during the lambda search.

## Memory And SIMD Strategy

`MatrixStorage` is column-major with `storage[i + j * leading_dim()]`, which is
the intended QR layout. Keep all tall hot loops unit-stride:

1. copy and scale one Jacobian column at a time
2. compute exact trailing pivot norms by streaming column tails
3. swap whole contiguous pivot columns
4. apply each Householder reflector with contiguous dot-product and axpy passes
   over one target column at a time
5. apply reflectors to the RHS with the same contiguous pattern

`R` and lambda scratch are small `n x n` matrices expected to remain in L1;
their column-major row accesses are not the hot bottleneck. Do not add BLAS,
intrinsics, unportable `restrict`, custom allocators, or alignment assumptions
without profile and compiler evidence. First inspect Clang loop-vectorization
diagnostics for the hot kernels using benchmark-only `-Rpass=loop-vectorize`
and `-Rpass-missed=loop-vectorize` options. If strict reductions block
vectorization, prefer a small number of independent scalar accumulators before
changing floating-point flags.

## Future Backend Boundary

The trust-region controller must be independent of the factorization method.
Linear-algebra policies should provide conceptually equivalent operations:

```text
prepare(scaled_jacobian, residual, effective_damping)
solve(lambda) -> scaled step
scaled_step_norm()
rank_diagnostic()
```

QR additionally owns permutation and reflector details internally. This keeps a
future SVD policy possible:

```text
J_s = U Sigma V^T
```

With identity damping in solved coordinates, SVD gives an elementwise damped
solve. With arbitrary diagonal damping, it instead solves the small dense
system involving `Sigma^2 + lambda V^T D^2 V`. Therefore the shared backend
boundary must accept the effective damping diagonal rather than assuming a
QR-specific permutation.

Dogleg, SVD, sparse iterative, and normal-equation backends remain separate
future policies. Do not substitute normal equations for the rank-robust QR
path.

## Validation And Benchmarks

Add direct linear-algebra tests for static, dynamic, and mixed extents:

1. QR reconstruction and reflector orthogonality
2. pivot ordering and rank diagnostics
3. damped solve agreement with the current backend on full-rank problems
4. rank-deficient and nearly rank-deficient damped solves
5. permuted damping and unpermutation correctness
6. safeguarded lambda updates and bisection fallback

Run:

```sh
ctest --test-dir build-external-ninja --output-on-failure
python3 scripts/benchmark.py --controller-trace --iterations 1 \
  --build-dir build-external-ninja --output-dir /tmp/levmar-controller-trace
python3 scripts/benchmark.py --solver-work --iterations 1 \
  --build-dir build-external-ninja --output-dir /tmp/levmar-work
```

Compare MGH10/start1, MGH17/start1, Gauss1/start2, Rat43/start1, and the tall
synthetic corpora. Report elapsed time, nonlinear iterations, lambda solves,
factorization count, rank diagnostics, and vectorization findings. Do not
commit generated executables or ad hoc CSV output.

## Deferred Work

1. persistent autodiff graph reuse across independent solve contexts
2. dynamic chunked direct dual evaluation
3. blocked-forward and reverse-mode autodiff backends
4. robust losses and geodesic acceleration
5. SVD, Dogleg, sparse iterative, and normal-equation policies
