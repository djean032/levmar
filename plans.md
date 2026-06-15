# Plans

## Solving Architecture

Current direction:
1. Stabilize the core solve loop first.
2. Stabilize `GaussNewton` and `LevenbergMarquardt` with one working
   linear backend first.
3. Defer user-extensible solver backends and larger abstraction refactors
   until the algorithms and data flow feel stable.

The immediate implementation target is:
1. `Strategy::GaussNewton`
2. `Strategy::LevenbergMarquardt`
3. `LinearSolver::NormalEquationsCholesky`

For now:
1. `LinearSolver::QR` and `LinearSolver::SVD` should fail cleanly with
   `Status::InvalidProblem` and a clear message.
2. `Strategy::TrustRegionLM` and `Strategy::DogLeg` should also fail
   cleanly with `Status::InvalidProblem` and a clear message.

## Core Solver Structure

Recommended top-level split:

```text
solve(context)
  validate_context(...)
  initialize result
  copy context.x -> work.x_current
  evaluate initial residual
  dispatch on strategy
```

Strategy dispatch should stay runtime-based for now:

```text
switch (context.options.strategy) {
  case GaussNewton:
    solve_gauss_newton(context)
  case LevenbergMarquardt:
    solve_levenberg_marquardt(context)
  case TrustRegionLM:
    fail unsupported
  case DogLeg:
    fail unsupported
}
```

This keeps control flow explicit while the algorithm is still moving.

## Shared Iteration Pieces

Both `GaussNewton` and `LevenbergMarquardt` share these steps:
1. Evaluate the Jacobian at `work.x_current`.
2. Form `g = J^T r`.
3. Form `JTJ = J^T J`.
4. Check gradient-based convergence.
5. Compute a step.
6. Build `work.x_trial = work.x_current + work.step`.
7. Evaluate trial residuals and trial cost.
8. Either accept the trial or reject it.

The main difference is:
1. `GaussNewton` computes an undamped step.
2. `LevenbergMarquardt` computes a damped step and updates `lambda` based
   on acceptance or rejection.
3. `GaussNewton` takes one full trial step per outer iteration.
4. `LevenbergMarquardt` may retry multiple damped trial steps against the
   same accepted iterate before consuming the next outer iteration.

## Step Computation

Use one shared step entry point with a damping parameter:

```text
compute_step(context, damping_lambda)
```

Then dispatch on the selected linear solver inside that helper:

```text
switch (context.options.linear_solver) {
  case NormalEquationsCholesky:
    compute_step_normal_equations_cholesky(context, damping_lambda)
  case QR:
    fail unsupported
  case SVD:
    fail unsupported
}
```

For the first implementation:
1. `GaussNewton` passes `damping_lambda = 0.0`.
2. `LevenbergMarquardt` passes its current `lambda`.

This keeps the linear solve backend shared between strategies while still
leaving strategy-specific acceptance logic separate.

For LM retries within one outer iteration:
1. Keep the accepted iterate fixed in `work.x_current`.
2. Keep the accepted residual and Jacobian fixed for that iterate.
3. Retry step computation with larger `lambda` after rejection.
4. Do not reevaluate the Jacobian until a step is accepted or the solve
   terminates.

## Normal Equations / Cholesky Path

Shared normal-equations build:

```text
g   = J^T r
JTJ = J^T J
```

For LM damping:

```text
JTJ(j,j) += lambda * D[j]
```

Initial damping policy can stay simple:
1. Use diagonal damping only.
2. A reasonable starting point is `D[j] = max(JTJ(j,j), 1.0)`.
3. Revisit scaling policy only after the base algorithm is stable.

The backend responsibilities are:
1. Build the damped system.
2. Factor it with in-place Cholesky.
3. Solve the triangular systems.
4. Write the step into `work.step`.

Storage policy for the first implementation:
1. Rebuild `g` and `JTJ` from `J` for each LM retry.
2. Do not try to preserve a pristine undamped copy yet.
3. This is simpler and avoids extra scratch-state design during the initial
   refactor.

Pseudocode:

```text
form g = J^T r
form JTJ = J^T J
apply damping to JTJ diagonal if lambda > 0
factor JTJ = L L^T
solve L y = -g
solve L^T step = y
```

## Strategy Behavior

### Gauss-Newton

Step equation:

```text
(J^T J) step = -J^T r
```

Policy:
1. No damping.
2. Evaluate the trial point.
3. Accept the step if cost decreases.
4. If cost does not decrease, terminate cleanly rather than adding an LM-style
   retry path.
5. Map this stagnation case to `Status::SmallCostReduction`.

### Levenberg-Marquardt

Step equation:

```text
(J^T J + lambda D) step = -J^T r
```

Policy:
1. Evaluate the trial point.
2. If cost decreases, accept the step and decrease `lambda`.
3. If cost does not decrease, reject the step and increase `lambda`.
4. Retry with the larger `lambda` inside the same outer iteration.
5. Keep the first implementation simple before refining the lambda update rule.

## Workspace And Context

`LMSolveContext` should remain a lightweight bundle of references.

Current policy:
1. It owns no storage.
2. It stores immutable input `x` as a view.
3. It auto-sizes `LMWorkspace` when either extent is dynamic.
4. It does not run validation itself.

`solve(context)` is responsible for:
1. Validation.
2. Copying `context.x` into `work.x_current` at solve start.
3. Driving all iteration state transitions.

For the first implementation, existing option surface area is intentionally
limited:
1. `LossKind::Squared` only.
2. `ScalingMode::None` only.
3. `weights` remain unused until robust loss/scaling work is added.
4. Unsupported modes should fail cleanly with `Status::InvalidProblem` and a
   clear message rather than being silently ignored.

Use:
1. `work.x_current` for the accepted iterate.
2. `work.x_trial` for the proposed step.
3. `work.r` for residuals at `x_current`.
4. `work.r_trial` for residuals at `x_trial`.

## Result And Termination Policy

Normal termination statuses should include:
1. `Status::SmallGradient`
2. `Status::SmallStep`
3. `Status::SmallCostReduction`
4. `Status::MaxIterations`

Hard-failure statuses remain:
1. `Status::InvalidProblem`
2. `Status::UserFunctionError`
3. `Status::NumericalFailure`

Recommended `solve(...)` return convention:
1. Return `true` for normal solver termination, including `MaxIterations`.
2. Return `false` only for hard failures.

Iteration accounting policy:
1. Count one outer iteration per accepted iterate attempt.
2. LM rejection retries do not advance the Jacobian to a new iterate.
3. LM rejection retries still consume residual evaluations.
4. `max_iterations` limits outer iterations.
5. `max_function_evaluations` limits total residual evaluations, including LM
   retries.

## Runtime Dispatch First

For now, keep `Strategy` and `LinearSolver` runtime-dispatched.

Rationale:
1. They are already runtime options.
2. Switch overhead is negligible relative to residual/Jacobian evaluation and
   factorization.
3. Runtime dispatch keeps the control flow easy to inspect while the
   implementation is still changing.

Do not refactor to user-pluggable solver backend templates yet.

That refactor can happen later, after:
1. `GaussNewton` works.
2. `LevenbergMarquardt` works.
3. The normal-equations backend contract feels stable.

## Near-Term Checklist

1. Add `solve(context)`.
2. Reset and populate `Result` inside `solve(context)`.
3. Copy `context.x` into `work.x_current` at solve start.
4. Add residual-cost evaluation helper.
5. Add normal-equations formation helper.
6. Add Cholesky factor and triangular solve helpers.
7. Add `compute_step_normal_equations_cholesky(context, damping_lambda)`.
8. Add `compute_step(context, damping_lambda)` with runtime linear-solver
   dispatch.
9. Add `solve_gauss_newton(context)`.
10. Add `solve_levenberg_marquardt(context)`.
11. Return clear unsupported messages for `QR`, `SVD`, `TrustRegionLM`, and
    `DogLeg`.
12. Validate the new path with the existing conformance runner.

## Later Refactors

Only after the above is stable:
1. Revisit QR and SVD integration.
2. Revisit workspace scratch for augmented-system solvers.
3. Revisit trust-region and dogleg strategies.
4. Revisit whether users should be able to inject custom linear solver
   backends from their own code.
5. If that extensibility is still desirable, make the enum-based built-in
   solver selection a convenience wrapper over a more general backend
   interface.
