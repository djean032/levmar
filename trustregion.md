# Scaled Trust-Region Controller

## Goal

Replace persistent scalar-lambda updates with an explicit scaled trust-region
radius. This targets the long sequence of locally accepted, low-progress steps
observed in MGH10/start1 while retaining the current scaling experiment.

## Existing Scaling

Keep the current split between fixed coordinate scaling and damping scaling:

```text
parameter_scale[j] = first Jacobian column norm
damping_scale[j] = max(parameter_scale[j], current Jacobian column norm)
damping_diagonal[j] = min(damping_scale[j]^2, 1e32)
```

The upper cap prevents a pathological Jacobian column from permanently
producing unbounded damping. Do not apply an absolute lower damping-diagonal
cap: it materially regressed small-scale MGH17 coordinates. The fixed
`parameter_scale` already provides the parameter-relative floor.

## Controller State

Store the persistent trust-region state in `SolverContext`:

```cpp
double trust_radius = initial_trust_region_radius;
double selected_lambda = initial_lambda;
```

`selected_lambda` is a warm start for the next damping-parameter bracket and
remains available through `Result::lambda` for diagnostics. The current
`damping_multiplier` is not part of this controller.

Use the fixed coordinate scale for the trust-region norm:

```text
step_norm_scaled = ||parameter_scale * step||_2
```

## Initial Settings

Use the following values for the first experiment:

```text
initial_trust_region_radius = 1e4
min_trust_region_radius = 1e-12
max_trust_region_radius = 1e16
```

`1e4` matches the Ceres adapter's initial trust-region radius. Do not use
MINPACK's `factor * ||D * x||` initialization in this experiment: MGH10's
scaled starting parameters make that radius effectively unconstrained.

These should become `LMOptions` fields rather than hidden constants because
they are solver numerical controls.

## Step Selection

Continue solving the augmented QR system in fixed scaled coordinates:

```text
J_scaled = J / parameter_scale
damping_row[j] = sqrt(lambda) * sqrt(damping_diagonal[j]) /
                 parameter_scale[j]
```

For each outer iteration:

1. Solve at the configured minimum lambda.
2. If `step_norm_scaled <= trust_radius`, select that interior,
   Gauss-Newton-like step.
3. Otherwise, bracket a lambda whose step lies inside the radius. Start from
   the previous selected lambda and double until the scaled step norm is no
   larger than the radius.
4. Bisect in log-lambda space:

   ```cpp
   lambda_mid = std::exp(0.5 * (std::log(lambda_low) +
                                 std::log(lambda_high)));
   ```

5. Select the inside-radius endpoint once:

   ```text
   0.9 * trust_radius <= step_norm_scaled <= trust_radius
   ```

Each bracketing or bisection solve increments `Result::linear_solves`. Evaluate
the residual only once, for the final selected trial step.

If the configured maximum lambda cannot produce a step inside the requested
radius, terminate with the existing `DampingLimit` status.

## Acceptance And Radius Update

Keep the existing predicted-reduction calculation and initial acceptance
threshold to isolate radius selection:

```text
accept when predicted_reduction > 0 and rho > 1e-3
```

Update the radius after the trial:

```text
rho <= 1e-3:
    reject
    trust_radius = max(min_trust_region_radius, 0.25 * step_norm_scaled)

rho > 0.75 and step_norm_scaled >= 0.9 * trust_radius:
    accept
    trust_radius = min(max_trust_region_radius, 2.0 * trust_radius)

otherwise:
    accept or reject from rho
    retain trust_radius
```

Treat non-finite trial cost and non-positive predicted reduction as rejected
trials and shrink the radius by the same rule. If a rejection would reduce the
radius below its configured minimum, terminate with `DampingLimit`.

## Trace Requirements

Add these fields to every controller-trace trial:

```text
trust_radius_before
trust_radius_after
selected_lambda
inner_linear_solves
radius_bound_active
```

The main MGH10 diagnostic is whether the initial scaled step, previously about
`3.9e7`, is constrained by the initial `1e4` radius and whether later radius
updates replace lambda collapse as the controller behavior.

## Validation

1. Build `levmar_nist_runner`.
2. Run `--controller-trace` and compare MGH10/start1's first 20 trials against
   the current scaling baseline.
3. Run `--solver-work`.
4. Compare MGH10/start1, MGH17/start1, Gauss1/start2, and Rat43/start1 against
   the current upper-only damping-cap baseline.
5. Verify static/dynamic and analytic/autodiff consistency, and do not commit
   generated benchmark CSVs.
