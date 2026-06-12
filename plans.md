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
