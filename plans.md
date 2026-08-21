# Plans

## Benchmark Baseline

The DAG is now recorded once per context and reused through the autodiff graph
cache. Jacobian extraction uses SIMD-sized tangent blocks. Benchmark reporting
is available through `scripts/benchmark.py`, using Clang or Intel with
`-O3 -march=native -fno-fast-math -ffp-contract=off` as the baseline; GCC
remains a valid reference compiler but is slower in current DAG results.

## Current Focus: Finish the AD Foundation and Build the Solver

Current direction:
1. Keep the DAG autodiff engine internal to `levmar/lm.h`.
2. Keep the public least-squares API `double`-based.
3. Keep generic `Scalar` plumbing only in storage/view utilities.
4. Use the unified DAG-based autodiff engine for all internal AD backends.
5. Support both static and dynamic parameter extents from the start.
6. Build graphs per evaluation until benchmark data justifies persistent graph
   storage.
7. Use scalar forward mode as the initial backend; add blocked forward and
   reverse mode only after solver workloads are measurable.
8. Avoid introducing a program-wide allocator or runtime framework.

The immediate implementation target is:
1. complete the AD primitive set needed by representative residual models
2. implement a robust, minimal Levenberg-Marquardt solve loop
3. validate solver convergence and AD Jacobians against the existing corpus
4. benchmark before optimizing graph construction or derivative passes

## Current Status

Completed:
1. removed the old eager-dual / arena-backed autodiff scaffolding from
   `levmar/lm.h`
2. added `NodeId`, `kInvalidNode`, and the initial `NodeKind` set
3. added `Node`, `NodeKey`, and `NodeKeyHash`
4. implemented `ADGraph`
5. implemented `intern_node(...)`
6. implemented `make_constant`, `make_variable`, `make_unary`, and
   `make_binary`
7. enabled commutative normalization for `Add` and `Mul`
8. added `ADExprRef`
9. added literal lifting helpers:
   `constant(graph, value)` and `variable(graph, parameter_index)`
10. added the initial operator/function set:
   unary `-`, binary `+`, binary `-`, binary `*`, and `exp(...)`
11. added `AdEvalContext` with `values` and `tangents`
12. implemented the primal forward pass
13. implemented the scalar tangent forward pass
14. implemented `evaluate_roots_forward(...)`
15. added internal residual autodiff bridge types and helpers:
    `AdParamView`, `AdResidualSlot`, `AdResidualSink`, and
    `invoke_residual_autodiff(...)`
16. added `work.ad_eval` to `LMWorkspace`
17. implemented `evaluate_autodiff_residual_and_jacobian(...)`
18. wired `JacobianMode::AutoDiff` into `evaluate_jacobian(...)`
19. added a dedicated zero-dependency graph-construction and forward-mode unit
    test executable
20. generalized residual and Jacobian callable concepts through
    `ResidualCallableOn` and `JacobianCallableOn`
21. validated:
   constant interning, variable interning, unary/binary interning,
   commutative normalization, topological node creation, `value_slot == node_id`,
   literal lifting, operator-built graph reuse, single-root value evaluation,
   multi-root shared-subexpression value evaluation, scalar tangent
   derivatives for single-root and multi-root graphs, and full residual/Jacobian
   extraction through `evaluate_roots_forward(...)`

Next:
1. Harden the AD bridge: report invalid residual-root assignment and invalid
   graph inputs as errors in release builds, and give incompatible residuals a
   clear AutoDiff-specific diagnostic.
2. Add `Div`, `Log`, `Sqrt`, `Sin`, `Cos`, and `Tan`, with primal, forward, and
   test coverage added together. Define the domain behavior for `Pow` before
   adding it.
3. Add AutoDiff conformance and timing coverage for AD-capable models across
   static, dynamic, and mixed extents.
4. Implement a minimal squared-loss Levenberg-Marquardt solve path: cost and
   gradient construction, scaling, a damped QR step, trial-step acceptance,
   damping updates, termination checks, and evaluation budgets.
5. Add NIST start-point convergence tests plus targeted failure tests for
   rank-deficiency, rejected steps, nonfinite values, and callback errors.
6. Measure graph-build and derivative-pass costs in solve-like workloads before
   adding persistent graphs, blocked forward mode, or reverse mode.

Conflicting earlier plans about eager dual storage, arena-backed temp
gradients, or reverse-mode-first execution are superseded by this document.

## Public Solver Boundary

Keep unchanged:
1. `Problem<M, N, Residual, Jacobian>`
2. `make_problem(...)` and dynamic variants
3. `ResidualCallable`
4. `JacobianCallable`
5. `LMSolveContext`
6. `Options`
7. `Result`
8. plain residual and Jacobian callback shapes using `double`-based views

Policy:
1. Public solver numerics remain `double`-based.
2. The autodiff engine is an internal implementation detail.
3. The storage/view layer remains generic where it already is:
   `VectorView`, `MatrixView`, `VectorStorage`, `MatrixStorage`.

## Core AD Model

Use a graph-based autodiff IR rather than eager dual propagation.

High-level model:
1. Build one shared DAG of scalar operations.
2. Keep a list of root nodes, one per residual/observation.
3. Run one forward pass over the whole graph to compute scalar values.
4. First backend:
   run one tangent forward pass per parameter and read all roots to fill one
   Jacobian column.
5. Second backend:
   run one tangent forward pass per parameter block of width `K`.
6. Third backend:
   run one reverse pass per root.

The DAG is the shared IR for all backends.

## Backend Roadmap

### Stage 1: Scalar Forward Mode

Execution model:
1. build one shared graph containing shared subexpressions
2. build `std::vector<NodeId> roots`, one root per residual output
3. run one primal forward pass over the whole graph
4. write residual values from the root node values
5. for each parameter column `j`:
   1. zero node tangents
   2. seed the tangent for parameter `j`
   3. run a tangent forward pass
   4. read all roots and write `J(i, j)` directly

This is the first implementation target.

### Stage 2: Blocked Forward Mode

Execution model:
1. keep the same graph and primal forward pass
2. replace scalar tangents with `K` tangent lanes per node
3. run one tangent pass per parameter block of width `K`
4. read all roots and write `K` Jacobian columns per pass

This is the first performance optimization after scalar forward mode is
correct.

### Stage 3: Reverse Mode

Execution model:
1. keep the same graph and primal forward pass
2. zero node adjoints
3. seed one root at a time
4. run one reverse pass per root
5. extract one Jacobian row per reverse pass

This is a later backend added on top of the same DAG infrastructure.

## Graph Identity

Define:
1. `using NodeId = Index`
2. `kInvalidNode = std::numeric_limits<Index>::max()`

`NodeId` is the index into `ADGraph::nodes`.

## Node Kinds

Initial required node set:
1. `Constant`
2. `Variable`
3. `Neg`
4. `Add`
5. `Sub`
6. `Mul`
7. `Exp`

Later expansion:
1. `Div`
2. `Log`
3. `Sqrt`
4. `Sin`
5. `Cos`
6. `Tan`
7. `Abs`
8. `Pow`

## Node Layout

Use one compact universal node structure.

Recommended fields:
1. `NodeKind kind`
2. `NodeId a`
3. `NodeId b`
4. `Index parameter_index`
5. `std::uint64_t literal_bits`
6. `Index value_slot`

Field meaning:
1. `a` and `b` are child node ids.
2. `parameter_index` is used only for `Variable`.
3. `literal_bits` is used only for `Constant`.
4. `value_slot` indexes the scalar-value cache for this node.

First implementation rule:
1. `value_slot == node_id`

That keeps evaluation buffers simple and avoids a separate slot allocator.

## Interning Keys

Use a structural `NodeKey` for interning, separate from `Node`.

Recommended fields:
1. `NodeKind kind`
2. `NodeId a`
3. `NodeId b`
4. `Index parameter_index`
5. `std::uint64_t literal_bits`

Equality:
1. field equality only

Hash:
1. hash all fields

Interning semantics:
1. constants are interned by exact `double` bit pattern
2. variables are interned by `parameter_index` only
3. unary nodes are interned by `(kind, child)`
4. binary nodes are interned by `(kind, left, right)`

Current policy:
1. normalize child order for `Add` and `Mul`

## Constant Semantics

Constants should be keyed by exact raw `double` bits, not `operator==`.

Implications:
1. `+0.0` and `-0.0` are distinct
2. NaN payloads remain distinct
3. interning is exact and predictable
4. no floating-point canonicalization is attempted initially

## Variable Semantics

Variable nodes represent active fit parameters only.

Policy:
1. variables are interned by `parameter_index`
2. no multiple variable namespaces
3. every occurrence of the same parameter may resolve to the same interned node

This matches the differentiation target exactly.

## Graph Builder

`ADGraph` owns:
1. `std::vector<Node> nodes`
2. `std::unordered_map<NodeKey, NodeId, NodeKeyHash> interned`

Provide:
1. `make_constant(double v)`
2. `make_variable(Index parameter_index)`
3. `make_unary(NodeKind kind, NodeId child)`
4. `make_binary(NodeKind kind, NodeId left, NodeId right)`

Interning helper:
1. `intern_node(const NodeKey&, Node) -> NodeId`

`intern_node(...)` responsibilities:
1. return an existing `NodeId` if the key already exists
2. otherwise append a new node
3. assign `value_slot = node_id`
4. insert into the interning map

Creation-order rule:
1. child nodes must exist before parent nodes are created

Consequence:
1. node creation order is already topological
2. no separate topo sort is required in the first implementation

## Expression Wrapper

Use a lightweight internal wrapper for graph-built expressions.

Recommended shape:
1. `ADGraph* graph`
2. `NodeId id`

Policy:
1. no heap ownership
2. cheap to copy
3. all operator syntax is built on this wrapper
4. binary operators require both operands to belong to the same graph
5. literals are lifted into the graph of the non-literal operand

This is an internal ergonomic layer only.

## Operator Surface

Use:
1. explicit overloads for `ADExprRef op ADExprRef`
2. templated scalar-lifting overloads for `ADExprRef op T` and `T op ADExprRef`
3. constrain `T` to values convertible to `double`

Initial operator/function set:
1. unary `-`
2. binary `+`
3. binary `-`
4. binary `*`
5. `exp(...)`

Later add:
1. `/`
2. `log`
3. `sqrt`
4. `sin`
5. `cos`
6. `tan`
7. `abs`
8. `pow`

Literal lifting helpers:
1. `constant(graph, value)`
2. `variable(graph, parameter_index)`

## Evaluation Context

Use an external evaluation context owned separately from the graph.

Recommended shape in the first backend:
1. `std::vector<double> values`
2. `std::vector<double> tangents`

Responsibilities:
1. ensure capacity to at least `graph.nodes.size()`
2. hold scalar values for every node during primal forward pass
3. hold one tangent value for every node during scalar forward-mode passes
4. zero tangents per parameter-direction evaluation

Because `value_slot == node_id` initially:
1. no separate scalar-slot allocator is needed
2. no retry logic is needed for scalar cache sizing
3. `values.size()` and `tangents.size()` simply track `nodes.size()`

Later, extend the same context with:
1. blocked tangent storage for `K` directions
2. adjoint storage for reverse mode

## Primal Forward Pass

Primal forward pass input:
1. graph
2. current parameter vector `x`
3. evaluation context

Pass order:
1. iterate nodes in creation order

Initial formulas:
1. `Constant`:
   decode `literal_bits` to `double`
2. `Variable`:
   `values[id] = x[parameter_index]`
3. `Neg`:
   `values[id] = -values[a]`
4. `Add`:
   `values[id] = values[a] + values[b]`
5. `Sub`:
   `values[id] = values[a] - values[b]`
6. `Mul`:
   `values[id] = values[a] * values[b]`
7. `Exp`:
   `values[id] = exp(values[a])`

Policy:
1. every node scalar value is computed exactly once per primal forward pass

## Scalar Tangent Forward Pass

Tangent forward pass input:
1. graph
2. current seeded parameter column index `j`
3. evaluation context containing already-computed primal values

Pass flow:
1. zero all node tangents
2. iterate nodes in creation order

Initial formulas:
1. `Constant`:
   `tangents[id] = 0`
2. `Variable`:
   `tangents[id] = 1` if `parameter_index == j`, else `0`
3. `Neg`:
   `tangents[id] = -tangents[a]`
4. `Add`:
   `tangents[id] = tangents[a] + tangents[b]`
5. `Sub`:
   `tangents[id] = tangents[a] - tangents[b]`
6. `Mul`:
   `tangents[id] = tangents[a] * values[b] + values[a] * tangents[b]`
7. `Exp`:
   `tangents[id] = values[id] * tangents[a]`

Policy:
1. one tangent pass computes one Jacobian column for all roots
2. primal values are shared across all tangent passes

## Multi-Root Forward Execution Model

Support multiple residual observations from the start.

Execution model for Stage 1:
1. build one graph containing shared subexpressions
2. build `std::vector<NodeId> roots`, one root per residual output
3. run one primal forward pass over the whole graph
4. write residual values directly from `values[root_i]`
5. for each parameter column `j`:
   1. run one tangent forward pass
   2. for each root `i`, write `J(i, j) = tangents[root_i]`

This is the first-class execution model for the first implementation.

## Jacobian Output In Stage 1

For scalar forward mode:
1. write residual values directly to `r`
2. write Jacobian entries directly to `J(i, j)` by columns

Reason:
1. one tangent pass naturally gives one Jacobian column
2. this matches `J`'s column-major layout well
3. no row staging or transpose-copy is needed in the first backend

## Blocked Forward Mode

After scalar forward mode is correct, widen tangent storage to `K` lanes per
node.

Execution model for Stage 2:
1. keep the same graph
2. keep the same primal forward pass
3. replace scalar `tangents[node]` with blocked `tangents[node][lane]`
4. process `K` parameters at once
5. write `K` Jacobian columns per tangent block pass

Policy:
1. this is the first performance optimization after the scalar backend works
2. graph building and interning remain unchanged

## Reverse Mode Later

After both forward backends are stable, add reverse mode as a second execution
backend.

Execution model for Stage 3:
1. keep the same graph and primal forward pass
2. add `adjoints[node]` to the evaluation context
3. seed one root at a time
4. run one reverse pass per root
5. extract one Jacobian row per reverse pass

Policy:
1. reverse mode is a later backend, not part of the first implementation
2. if row-oriented writes are used there, row staging and transpose-copy become
   relevant at that stage

## Workspace Storage

Stage 1 needs no dedicated Jacobian row staging for the forward backend.

Policy:
1. scalar forward mode writes `J` directly by columns
2. `LMWorkspace` keeps existing solver state unchanged for now
3. if reverse mode later benefits from row staging, add that storage at the
   reverse-mode stage rather than forcing it into the first forward-mode phase

Dynamic autodiff-specific storage from the old eager-dual design should be
de-emphasized or removed as the DAG path lands.

## Execution API Split

Keep the execution pieces separate first.

Recommended functions for Stage 1:
1. `forward_pass(...)`
2. `forward_tangent_pass(...)`
3. `evaluate_roots_forward(...)`

Responsibilities:
1. `forward_pass(...)` computes all node scalar values once
2. `forward_tangent_pass(...)` computes one tangent column for all nodes
3. `evaluate_roots_forward(...)` orchestrates:
   1. one primal forward pass
   2. residual value writes
   3. one tangent pass per parameter column
   4. direct Jacobian column writes

Later add:
1. `forward_tangent_block_pass(...)`
2. `evaluate_roots_forward_blocked(...)`
3. `reverse_from_root(...)`
4. `evaluate_roots_reverse(...)`

Current status:
1. `forward_pass(...)` is implemented
2. `forward_tangent_pass(...)` is implemented
3. `evaluate_roots_forward(...)` is implemented
4. `evaluate_autodiff_residual_and_jacobian(...)` is implemented
5. `evaluate_jacobian(...)` dispatches `JacobianMode::AutoDiff` through the
   fused helper

## Static And Dynamic Support

Support both static and dynamic parameter extents from the first
implementation.

Unified policy:
1. same graph representation
2. same node kinds
3. same primal forward formulas
4. same tangent forward formulas
5. same multi-root execution model

Only storage differs:
1. static paths use fixed-size views/buffers where appropriate
2. dynamic paths use runtime-sized views/buffers

No split eager-dual architecture should be pursued in parallel with the DAG
path.

## Rebuild-Per-Eval First

First implementation:
1. rebuild the DAG per evaluation

Reason:
1. simpler correctness story
2. no stale graph structure issues
3. enough to prove the model

Later optimization:
1. persistent graph storage
2. graph reuse across LM iterations
3. eventual amortization of interning/build cost

## Persistent Graph Storage Later

Long-term direction:
1. separate graph shape from evaluation values
2. keep topology and interning persistent
3. rebind current parameter values per evaluation
4. reuse roots and node structure across repeated solves

This is a later optimization, not a prerequisite for correctness.

## Initial Implementation Subset

Implement first:
1. node kinds:
   `Constant`, `Variable`, `Neg`, `Add`, `Sub`, `Mul`, `Exp`
2. graph builder
3. interning
4. expression wrapper
5. scalar-lifting overloads
6. primal forward pass
7. scalar tangent forward pass
8. multi-root forward evaluation
9. direct Jacobian column fill into `J`

This is enough to represent common expressions such as:
1. `x0 * (1.0 - exp(-x1 * t)) - y`

## Product Rule And Shared Work

The DAG design should be treated as the primary answer to shared-subexpression
reuse.

Within one evaluation:
1. shared nodes are evaluated once in the primal forward pass
2. every tangent pass reuses those cached scalar values
3. every root reuses the same primal values
4. no temp gradient vectors are materialized per operator

This is the main performance goal of the new autodiff design.

## Later Extensions

After primitive coverage and a minimal solver are stable:
1. add persistent graph storage and reuse it across solve iterations when the
   residual structure is fixed
2. add blocked forward mode with width `K`, guided by benchmarks
3. add reverse mode on the same graph for wide problems where `N > M`
4. select the AD backend according to problem shape and measured cost
5. add robust losses, additional linear solvers, and solver strategies beyond
   Levenberg-Marquardt

## Near-Term Checklist

1. Done: define `NodeId`, `kInvalidNode`, and `NodeKind`
2. Done: define `Node`
3. Done: define `NodeKey`
4. Done: define `NodeKeyHash`
5. Done: implement `ADGraph`
6. Done: implement `intern_node(...)`
7. Done: implement `make_constant`, `make_variable`, `make_unary`,
   `make_binary`
8. Done: implement `ADExprRef`
9. Done: implement literal lifting
10. Done: implement initial operator/function set
11. Done: implement `ADEvalContext` with `values` and `tangents`
12. Done: implement primal forward pass
13. Done: add value-only evaluation tests
14. Done: implement scalar tangent forward pass
15. Done: implement multi-root forward evaluation
16. Done: validate static and dynamic multi-root cases through full residual
    and Jacobian extraction in the forward backend
17. Done: add tests for the fused autodiff residual/Jacobian path and
    `AutoDiff` dispatch through solver-facing helpers
18. Done: Harden the fused autodiff integration path
19. Done: Expand math coverage and add AD conformance coverage
20. Implement the minimal Levenberg-Marquardt solve loop
21. Add solver convergence and failure-mode tests
22. Benchmark persistent graphs, blocked forward mode, and reverse mode before
    implementing them

## Deferred Until After This Work

Only after the minimal solver and its validation are stable:
1. persistent graph storage
2. blocked-forward and reverse-mode backends
3. robust losses, SVD, DogLeg, and TrustRegionLM
