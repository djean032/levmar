# Plans

## Immediate Focus: Autodiff Infrastructure

Current direction:
1. Implement the internal autodiff building blocks before wiring autodiff into
   solver evaluation.
2. Keep the autodiff surface internal to `levmar/lm.h` for now.
3. Keep persistent autodiff storage in `LMWorkspace` only for dynamic
   autodiff paths.
4. Use two internal autodiff implementations:
   static inline-owning duals and dynamic arena-backed duals.
5. Avoid introducing a program-wide arena or allocator policy.

The immediate implementation target is:
1. `Dual<N>`
2. autodiff workspace storage in `LMWorkspace<M, N>`
3. internal operator overloads and math helpers for `Dual`

Conflicting older plans about prioritizing the solve loop first are superseded
by this document.

## Dual Type

Implement an internal dual number type templated on derivative extent `N`, with
`double` primal values.

Static and dynamic extents use different storage models:

```text
Dual<N>
  value: double
  grad: owned inline storage

Dual<dynamic>
  value: double
  grad:  VectorView<dynamic>
  arena: ADArena*
```

Design rules:
1. `Dual` is internal-only and is not part of the public API yet.
2. `Dual` is templated on derivative extent `N` only.
3. `Dual` keeps the primal type fixed as `double` for now.
4. `Dual<N>` for fixed `N` owns its gradient storage inline.
5. `Dual<dynamic>` remains non-owning and uses workspace- or arena-backed
   gradient memory.
6. `Dual<dynamic>` carries an `ADArena*` so internal operator overloads can
   allocate temporary gradient vectors naturally.

Initial scope:
1. fixed-extent `Dual<N>` uses inline gradient storage, likely `std::array`
2. `Dual<dynamic>` uses `VectorView<dynamic>` and `ADArena*`
3. default construction
4. no hidden dynamic allocation in constructors
5. add a dedicated `Dual<std::dynamic_extent>` specialization

## Arena Design

Use a small internal bump allocator over workspace-owned `double` scratch:

```text
ADArena
  buffer: span<double>
  used: Index
  overflowed: bool
```

Arena policy:
1. `ADArena` is non-owning.
2. `LMWorkspace` owns the backing storage.
3. `ADArena` is only used by `Dual<dynamic>` and dynamic autodiff paths.
4. `ADArena` is reset once per autodiff residual evaluation.
5. Arena scratch is always dynamic and may grow.
6. Individual temporary allocations are never freed.
7. Overflow sets a sticky `overflowed` flag.
8. Later autodiff evaluation should grow scratch and retry on overflow.
9. Operator code may continue after overflow, but later autodiff evaluation must
   detect the sticky failure and return a proper solver error.

Required helpers:
1. `reset()`
2. `allocate_grad(Index n, VectorView<dynamic>& out) -> bool`
3. convenience wrapper returning `VectorView<dynamic>` if needed

## Workspace Storage

Split autodiff storage into persistent state and temporary scratch.

Persistent workspace state:
1. parameter gradient backing storage for dynamic paths
2. residual gradient backing storage for dynamic paths
3. dynamic wrapper objects only when their count is dynamic

Temporary workspace state:
1. one `double` scratch buffer used only by `ADArena`

Recommended shape inside `LMWorkspace<M, N>`:

```text
ad_x_grads  : MatrixStorage<N, N>
ad_r_grads  : MatrixStorage<N, M>
ad_scratch  : VectorStorage<dynamic>
ad_arena    : ADArena
ad_x_dynamic: VectorStorage<dynamic, Dual<dynamic>>
ad_r_dynamic: VectorStorage<dynamic, Dual<N>>
```

Binding rules:
1. each parameter dual gradient is bound to a contiguous column in `ad_x_grads`
2. each residual dual gradient is bound to a contiguous column in `ad_r_grads`
3. `ad_r_grads` stores autodiff output gradients as `J^T`
4. residual output gradients must not come from the arena
5. only dynamic intermediate temporaries allocate from the arena

Static-path rules:
1. static input duals own their gradients inline and are seeded directly as
   identity
2. static output duals own their gradients inline
3. static autodiff fills `J` directly from local output dual gradients
4. static autodiff does not require persistent `ad_x_grads` or `ad_r_grads`

Initialization policy:
1. bind persistent dual views after workspace sizing
2. rebind dynamic views inside `LMWorkspace::resize(...)`
3. bind `ad_arena.buffer` to `ad_scratch.view()` after sizing
4. static wrapper duals are created as local values inside autodiff evaluation,
   not persisted in workspace

## Static Vs Dynamic Policy

Use static inline duals for fixed `N`, but keep arena scratch dynamic.

Policy:
1. if `N != std::dynamic_extent`, use inline-owning `Dual<N>` for operator
   temporaries
2. if `N == std::dynamic_extent`, use arena-backed `Dual<dynamic>` for operator
   temporaries
3. keep `ad_scratch` dynamic for dynamic autodiff paths

Rationale:
1. static small-`N` problems benefit from jet-like inline dual storage
2. dynamic problems need workspace-backed gradients and growable scratch
3. static autodiff can avoid persistent gradient backing by using local owning
   input and output duals
4. temporary demand depends on residual expression complexity and the full
   callback shape, not just `N`
5. fixed arena scratch is brittle for callbacks that compute all `m` residuals
   in one call

Dynamic scratch heuristic:
1. size scratch at runtime from `m` and `n`
2. initial heuristic: `max(16 * m * n, 16 * n * n)` doubles
3. keep overflow detection even with runtime sizing
4. later autodiff evaluation should grow and retry on overflow

## Operator Overloads

Operator overloads are internal implementation helpers only.

Initial arithmetic scope:
1. unary `+`
2. unary `-`
3. binary `+`
4. binary `-`
5. binary `*`
6. binary `/`
7. mixed `Dual` and `double` overloads for the above

Initial math function scope:
1. `exp`, `log`, `sqrt`
2. `sin`, `cos`, `tan`
3. `abs`
4. `pow`

Operator policy:
1. static `Dual<N>` operators return local owning results with no allocation
2. dynamic `Dual<dynamic>` operators allocate result gradients from `ADArena`
3. the dynamic result carries the same arena pointer forward
4. operator overloads assume internal use and do not need public-library style
   generality
5. no thread-local or global arena state
6. arithmetic formulas should be shared between static and dynamic overload
   families via view-based helper kernels

Gradient rules:
1. `+` and `-` are elementwise on gradients
2. `*` uses the product rule
3. `/` uses the quotient rule
4. unary math functions apply the scalar derivative multiplier to the full
   gradient vector

## Helper Utilities

Add small internal helpers to keep operator code direct:
1. zero a gradient view
2. copy one gradient view into another
3. `scal`-style gradient scaling
4. `axpy`-style gradient accumulation
5. linear combination helper kernels for binary ops
6. allocate a dynamic result gradient and build a dynamic `Dual` result object
7. gradient access helpers like `grad_view(...)` so static and dynamic duals can
   share kernels

Keep these helpers local to `levmar/lm.h` and avoid introducing a separate
autodiff subsystem yet.

## Transition Plan

Move from the current single non-owning dual model to a split static/dynamic
implementation.

Target end state:
1. `Dual<N>` for fixed `N` owns inline gradients
2. `Dual<dynamic>` remains non-owning and arena-backed
3. math formulas are shared through view-based helper kernels
4. static autodiff writes `J` directly from local output duals
5. dynamic autodiff keeps persistent workspace-backed gradient buffers

Migration steps:
1. Keep the current dynamic path contract stable:
    `Dual<dynamic>`, `ADArena`, `ad_x_grads`, `ad_r_grads`, and dynamic wrapper
    binding remain the basis for the dynamic path.
2. Redefine `Dual` by specialization:
   `Dual<N>` becomes owning for static `N`, and `Dual<dynamic>` stays
   non-owning.
3. Add `grad_view(...)` helpers for static and dynamic duals.
4. Rewrite gradient helper utilities to operate only on views.
5. Split temporary-result construction helpers:
   static local result construction and dynamic arena-backed result
   construction.
6. Implement operators in two families:
   static-owning overloads and dynamic-arena-backed overloads.
7. Remove static temporary scratch assumptions from workspace.
8. Remove static-path dependence on persistent `ad_x_grads` and `ad_r_grads`.
9. Keep workspace scratch dynamic-only and later support grow-and-retry.
10. Later, add autodiff evaluation in two branches:
    static local inline dual arrays and dynamic workspace-backed wrapper arrays.

## Implementation Checklist

### Dual Refactor

1. Add a dedicated `Dual<std::dynamic_extent>` specialization.
2. Change fixed-extent `Dual<N>` to own inline gradients, likely via
   `std::array<double, N>`.
3. Add `grad_view(...)` helpers for mutable and const access.
4. Ensure static duals can be seeded directly without workspace backing.

### Helper And Kernel Layer

1. Add `zero_grad`, `copy_grad`, `scal_grad`, `axpy_grad`, and
   `linear_combine_grad` helpers on views.
2. Keep helper signatures BLAS-friendly so the implementation can be swapped
   later.
3. Add dynamic temp helpers such as `allocate_temp_grad(...)` and
   `make_temp_dual_like(...)`.
4. Add shared derivative kernels used by both static and dynamic overloads.

### Evaluation Split

1. Static autodiff evaluation builds local input and output dual arrays.
2. Static autodiff seeds input gradients directly as identity.
3. Static autodiff writes residual values to `r` and gradients directly to `J`.
4. Dynamic autodiff continues to use `ad_x_grads`, `ad_r_grads`, and dynamic
   wrapper binding.
5. Dynamic autodiff later adds grow-and-retry on arena overflow.

### Operator Migration Order

1. Port `operator*` first as the model implementation.
2. Add unary `+` and unary `-`.
3. Add binary `+` and `-`.
4. Add binary `/`.
5. Add mixed `Dual` / `double` overloads.
6. Add unary math functions.
7. Add `pow` overloads last.

## Near-Term Checklist

1. Add `ADArena` and its allocation helpers.
2. Add internal `Dual<N>` and `Dual<dynamic>` split storage models.
3. Keep `LMWorkspace<M, N>` persistent autodiff storage only for dynamic paths.
4. Keep autodiff scratch dynamic and arena-backed.
5. Add workspace binding helpers for dynamic autodiff views and arena buffer
   setup.
6. Add gradient helper kernels.
7. Add the initial arithmetic operator overload set.
8. Add the basic math function set.
9. Add overflow-aware internal assertions or checks where useful during early
   bring-up.

## Deferred Until After This Work

Only after the above is stable:
1. add `JacobianMode::AutoDiff`
2. add autodiff evaluation helpers that fill both residuals and Jacobians
3. wire autodiff into `evaluate_jacobian(...)` or a fused evaluation path
4. resume the broader solve-loop implementation work
