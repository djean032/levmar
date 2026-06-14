# Plans

## Chem Intensity Propagation

For the Z-scan chemistry model, the intensity propagation step does not need a second ODE solve.

Current intensity equation:

```text
dI/dz = -(k1*N2 + k2*N3 + k3*N5) * I
```

Within each spatial slice, the populations are treated as fixed after the chemistry solve, so the coefficient is constant over that slice:

```text
alpha(t) = k1*N2(t) + k2*N3(t) + k3*N5(t)
```

This gives the exact per-slice update:

```text
I_out(t) = I_in(t) * exp(-alpha(t) * dz)
```

Planned improvement:
1. Remove the separate scalar intensity ODE solve.
2. Replace `solve_intensity(...)` with the closed-form exponential update.
3. Keep the chemistry system as the only true ODE integration.
4. Avoid restarting the chemistry solver at every tiny time step in future refactors; prefer one initialized solve per sample/slice marched across the full time grid.

Expected benefits:
1. Less solver overhead
2. Simpler implementation
3. Cleaner sensitivity/Jacobian path later
4. Better fit performance

## Normal Equations Pseudocode

```text
# Inputs:
# J: m x n
# r: length m
# Outputs:
# A: n x n   = J^T J
# g: length n = J^T r

# zero A and g
for j = 0 .. n-1
  g[j] = 0
  for k = 0 .. n-1
    A[j,k] = 0

# build g = J^T r
for j = 0 .. n-1
  sum = 0
  for i = 0 .. m-1
    sum += J[i,j] * r[i]
  g[j] = sum

# build A = J^T J
for j = 0 .. n-1
  for k = j .. n-1
    sum = 0
    for i = 0 .. m-1
      sum += J[i,j] * J[i,k]
    A[j,k] = sum
    A[k,j] = sum
```

For LM damping:

```text
for j = 0 .. n-1
  A[j,j] += lambda * D[j]
```

Then Cholesky:

```text
# factor A = L L^T in-place into lower triangle

for j = 0 .. n-1
  sum = A[j,j]
  for k = 0 .. j-1
    sum -= A[j,k] * A[j,k]

  if sum <= 0
    fail

  A[j,j] = sqrt(sum)

  for i = j+1 .. n-1
    sum = A[i,j]
    for k = 0 .. j-1
      sum -= A[i,k] * A[j,k]
    A[i,j] = sum / A[j,j]
```

Forward solve `L y = -g`:

```text
for i = 0 .. n-1
  sum = -g[i]
  for k = 0 .. i-1
    sum -= A[i,k] * y[k]
  y[i] = sum / A[i,i]
```

Backward solve `L^T p = y`:

```text
for i = n-1 down to 0
  sum = y[i]
  for k = i+1 .. n-1
    sum -= A[k,i] * p[k]
  p[i] = sum / A[i,i]
```

Update:

```text
for j = 0 .. n-1
  x_new[j] = x[j] + p[j]
```

## LM Solver Backend Plan

Use a custom small-matrix implementation for the `NormalEquationsCholesky` path.

Rationale:
1. Small LM problems often have modest parameter counts, so hand-written `J^T r`, `J^T J`, damping, Cholesky, and triangular solves are simple and can be competitive or faster than BLAS/LAPACK at small sizes.
2. Residual and Jacobian evaluation may dominate total runtime, so BLAS/LAPACK overhead is not guaranteed to help for the normal-equations path.

Use LAPACK-backed implementations for `QR` and `SVD`.

Rationale:
1. QR is more code and easier to get subtly wrong than Cholesky.
2. SVD is not a good hand-written target for this project.
3. QR/SVD are selected mainly for numerical robustness and maintainability, not just raw speed.

Build/configuration plan:
1. Add a build option such as `LEVMAR_USE_LAPACK`.
2. When enabled and LAPACK is found, enable LAPACK-backed `QR` and `SVD`.
3. Keep `NormalEquationsCholesky` available in all builds via the internal implementation.
4. Keep the public `LinearSolver` enum stable regardless of build configuration.
5. In non-LAPACK builds, selecting `QR` or `SVD` should fail cleanly with `Status::InvalidProblem` and a clear message that the solver requires a LAPACK-enabled build.

Initial policy:
1. `NormalEquationsCholesky`: internal implementation.
2. `QR`: LAPACK only.
3. `SVD`: LAPACK only.
4. Only revisit a size-based dispatch policy after benchmarking shows the linear algebra path is a real bottleneck.

## QR Solver Pseudocode

Solve the damped LM step through the augmented least-squares system:

```text
[ J              ] p ~= [ -r ]
[ sqrt(lambda D) ]      [  0 ]
```

where `sqrt(lambda D)` is the `n x n` diagonal matrix with entries
`sqrt(lambda * D[j])`.

Assume column-major storage throughout, matching BLAS/LAPACK conventions.
For an `(m+n) x n` augmented matrix, element `(i, j)` is stored at
`A_aug[i + j * lda]` with `lda = m + n`.

```text
# Inputs:
# J: m x n
# r: length m
# D: length n
# lambda: scalar
# Output:
# p: length n

rows_aug = m + n
cols_aug = n
lda = rows_aug

# build augmented matrix A_aug of size (m+n) x n
for i = 0 .. m-1
  for j = 0 .. n-1
    A_aug[i + j * lda] = J[i,j]

for i = 0 .. n-1
  for j = 0 .. n-1
    A_aug[(m + i) + j * lda] = 0

for j = 0 .. n-1
  A_aug[(m + j) + j * lda] = sqrt(lambda * D[j])

# build augmented rhs b_aug of length (m+n)
for i = 0 .. m-1
  b_aug[i] = -r[i]

for i = 0 .. n-1
  b_aug[m + i] = 0

# factor A_aug = Q R in-place
qr_factor(A_aug, tau)

# apply Q^T to rhs
apply_qt(A_aug, tau, b_aug)

# solve R p = first n entries of b_aug, where R is stored in the
# upper triangle of A_aug in column-major layout
for i = n-1 down to 0
  sum = b_aug[i]
  for k = i+1 .. n-1
    sum -= A_aug[i + k * lda] * p[k]
  p[i] = sum / A_aug[i + i * lda]
```

Relevant LAPACK calls:
1. `dgeqrf` or `LAPACKE_dgeqrf` for QR factorization.
2. `dormqr` or `LAPACKE_dormqr` for applying `Q^T` to the right-hand side.
3. `dtrtrs` or `LAPACKE_dtrtrs` for the upper-triangular solve.

LAPACK-style dimensions:
1. `A_aug` is `rows_aug x n` with `lda = rows_aug`.
2. `b_aug` can be treated as a `rows_aug x 1` dense column with leading dimension `ldb = rows_aug`.
3. `R` is the upper triangle of the first `n` rows of `A_aug` after `dgeqrf`.

Relevant BLAS calls:
1. None are required for the basic QR path.
2. `dcopy` may be useful for vector copies.

## SVD Solver Pseudocode

Solve the same augmented least-squares system through the SVD of the
augmented matrix:

```text
A_aug = U Sigma V^T
```

Then solve:

```text
min ||A_aug p - b_aug||^2
```

with `b_aug = [-r; 0]`.

Assume column-major storage throughout, matching BLAS/LAPACK conventions.
For an `(m+n) x n` augmented matrix, element `(i, j)` is stored at
`A_aug[i + j * lda]` with `lda = m + n`.

```text
# Inputs:
# J: m x n
# r: length m
# D: length n
# lambda: scalar
# Output:
# p: length n

rows_aug = m + n
cols_aug = n
lda = rows_aug

# build augmented matrix A_aug of size (m+n) x n
for i = 0 .. m-1
  for j = 0 .. n-1
    A_aug[i + j * lda] = J[i,j]

for i = 0 .. n-1
  for j = 0 .. n-1
    A_aug[(m + i) + j * lda] = 0

for j = 0 .. n-1
  A_aug[(m + j) + j * lda] = sqrt(lambda * D[j])

# build augmented rhs b_aug of length (m+n)
for i = 0 .. m-1
  b_aug[i] = -r[i]

for i = 0 .. n-1
  b_aug[m + i] = 0

# compute A_aug = U Sigma V^T
svd(A_aug, U, sigma, VT)

# tmp = U^T b_aug
for i = 0 .. n-1
  sum = 0
  for k = 0 .. rows_aug-1
    sum += U[k,i] * b_aug[k]
  tmp[i] = sum

# divide by singular values
for i = 0 .. n-1
  if sigma[i] is small
    tmp[i] = 0
  else
    tmp[i] = tmp[i] / sigma[i]

# p = V tmp
for i = 0 .. n-1
  sum = 0
  for k = 0 .. n-1
    sum += VT[k,i] * tmp[k]
  p[i] = sum
```

Relevant LAPACK calls:
1. `dgesdd` or `LAPACKE_dgesdd` for SVD.
2. `dgesvd` or `LAPACKE_dgesvd` as an alternative SVD routine.

LAPACK-style dimensions:
1. `A_aug` is `rows_aug x n` with `lda = rows_aug`.
2. For economy-size SVD, `U` is `rows_aug x n` and `VT` is `n x n`.
3. `U` uses `ldu = rows_aug` and `VT` uses `ldvt = n`.

Relevant BLAS calls:
1. `dgemv` for `tmp = U^T b_aug`.
2. `dgemv` for `p = V tmp`.

Notes:
1. `dgesdd` is likely the better default SVD routine.
2. Use a singular value tolerance so very small `sigma[i]` values are treated as zero.
3. The augmented-system formulation handles general diagonal damping `D` cleanly.

## Single LMWorkspace Design

Use one overall `LMWorkspace` with:
1. Shared LM state used by all solver paths.
2. Shared augmented-system buffers used by both `QR` and `SVD`.
3. Solver-specific scratch buffers for the selected linear solver.

Shared LM state:

```text
m, n
x_current       length n
x_trial         length n
r               length m
r_trial         length m
r_trial_minus   length m
J               m x n
step            length n
scale           length n
weights         length m
```

Normal-equations / Cholesky buffers:

```text
JTJ             n x n
g               length n
```

Shared augmented-system buffers for QR/SVD:

```text
aug_matrix      (m+n) x n
aug_rhs         length (m+n)
```

QR-specific scratch:

```text
qr_tau          length n
```

SVD-specific scratch:

```text
svd_sigma       length n
svd_U           (m+n) x n     # economy-size U
svd_VT          n x n
svd_tmp         length n
```

Notes:
1. `aug_matrix` and `aug_rhs` should be shared between `QR` and `SVD` rather than duplicated.
2. LAPACK routines overwrite their input matrices, so `aug_matrix` is solver scratch, not a persistent representation of `J`.
3. Since solver choice is fixed for a solve, allocate solver-specific scratch for the selected solver during workspace setup.
4. If needed, store prevalidated `lapack_int` dimensions in the workspace as well so LAPACKE call sites do not repeat casts.

## Solve API And Control Flow

Use `LMSolveContext<ProblemT>` as a lightweight non-owning bundle of
already-initialized solver state.

Recommended sequence:
1. Initialize `Problem`, `Options`, `Result`, `LMWorkspace`, and immutable parameter storage.
2. Resize `LMWorkspace` before solve.
3. Bundle them into `LMSolveContext<ProblemT>`.
4. Call `solve(context)`.
5. Report from `context.result` and final parameter storage in workspace.

Recommended API shape:

```text
template <class ProblemT>
bool solve(LMSolveContext<ProblemT>& context)
```

Responsibilities:
1. Setup stays outside the solver.
2. `solve(context)` owns validation, iteration control flow, and solver dispatch.
3. Reporting stays outside the solver.

Recommended `LMSolveContext` role:
1. Hold references to `Problem`, `Options`, `Result`, `LMWorkspace`, and immutable input `x`.
2. Remain non-owning.
3. Provide a core constructor taking `ConstVectorView<...>`.
4. Provide constrained convenience constructors for `std::array` and `std::vector`.

Avoid making `LMSolveContext` a heavy initialization object. It should not allocate workspace, construct result objects, or implicitly run validation beyond simple bundling.

Validation policy:
1. Constructors do not throw and do not validate.
2. `validate_problem<ProblemT>(...)` handles problem/options checks.
3. `validate_context<ProblemT>(...)` handles context-specific checks.
4. `validate_context(...)` should always verify:
5. `problem.num_residuals > 0`
6. `problem.num_parameters > 0`
7. `context.x.size() == problem.num_parameters`
8. If `JacobianMode::User`, `problem.has_user_jacobian()`
9. `work.m == problem.num_residuals`
10. `work.n == problem.num_parameters`

Internal solver dispatch should remain centralized inside the solver implementation:

```text
switch (context.options.linear_solver) {
  case NormalEquationsCholesky:
    compute_step_normal_equations_cholesky(context)
  case QR:
    compute_step_qr(context)
  case SVD:
    compute_step_svd(context)
}
```

Recommended internal split:
1. `solve(context)` runs the overall LM algorithm.
2. `compute_step_normal_equations_cholesky(context)` computes the LM step for the custom normal-equations path.
3. `compute_step_qr(context)` computes the LM step for the LAPACK QR path.
4. `compute_step_svd(context)` computes the LM step for the LAPACK SVD path.

## Problem API

Use a single templated `Problem` type:

```text
template <Index M, Index N, class Residual, class Jacobian = NoJacobian>
struct Problem
```

Design:
1. Store residual and jacobian as concrete callable types.
2. Expose `residual_extent` and `parameter_extent` as compile-time metadata.
3. Store `num_residuals` and `num_parameters` for runtime validation.
4. Use `NoJacobian` as the marker for residual-only problems.

Construction policy:
1. Static/static problems use `make_problem<M, N>(...)`.
2. Dynamic/dynamic problems use `make_dynamic_problem(m, n, ...)`.
3. Dynamic residual-count problems use `make_problem_dynamic_residuals<N>(m, ...)`.
4. Dynamic parameter-count problems use `make_problem_dynamic_parameters<M>(n, ...)`.

Rationale:
1. Factories make the public API unambiguous.
2. Direct callable storage avoids `std::function` type erasure.
3. Mixed dynamic/static problem shapes remain readable at the call site.

## Current Refactor Checklist

1. Finish `LMSolveContext<ProblemT>`.
2. Template `validate_problem<ProblemT>(...)`.
3. Add `validate_context<ProblemT>(...)`.
4. Template `evaluate_residual_at<ProblemT>(...)`.
5. Template `evaluate_residual<ProblemT>(...)`.
6. Template finite-difference Jacobian helpers.
7. Template `evaluate_jacobian<ProblemT>(...)`.
8. Update helper call sites to use `VectorStorage` and `MatrixStorage` directly.
9. Initialize `work.x_current` from `context.x` at solve start.
10. Use `work.x_trial` for trial steps and `work.x_current` for accepted iterates.
11. Add templated `solve(LMSolveContext<ProblemT>&)`.
12. Remove old non-templated `Problem`, `LMSolveContext`, `ResidualFunction`, and `JacobianFunction` leftovers.
13. Restore the normal-equations path first.
14. Revisit QR/SVD and LAPACK integration after the templated core compiles cleanly.
