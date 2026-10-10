# ADR: `optim` — Optimizers, Update Rules, Gradient Clipping, and Learning-Rate Schedules

Status: Accepted
Scope: `crates/optim` — public contracts, update equations, invariants, error semantics, state persistence compatibility, and dependency/naming choices
Roadmap step: P9
Date: 2026-10-10

## Context

Roadmap stage P9 introduces the `optim` crate. Per `docs/crate-boundaries.md`
section 5.12, `optim` owns:

* Optimizer contracts.
* AdamW and SGD implementations.
* Gradient clipping.
* Learning-rate schedules.
* Optimizer-specific mutable state.

It MUST NOT own model architecture, dataset loading, checkpoint file encoding,
or training-loop orchestration. Its workspace dependencies are `tensor` and
`lib` only. It deliberately does **not** depend on `autograd`, `nn`, or
`checkpoint` in its production closure:

* It does not depend on `autograd` because the plan requires it to be decoupled
  from any specific autograd graph. An update rule operates on owned parameter
  tensors and their gradient tensors, taken as borrowed slices; a gradient is
  any `tensor<f32>`, regardless of how it was produced.
* It does not depend on `nn` because `nn` currently exposes forward-only
  parameters, and a smaller `tensor`-based abstraction is sufficient for the
  optimizer state and update surface.
* It does not depend on `checkpoint` because `optim` owns neither checkpoint
  encoding nor the training state. It exposes named state tensors that a higher
  layer maps onto the checkpoint tensor contract.

Role and responsibility (`rule.md` IDN3): this crate is the optimization-rule
owner. It defines boundaries and invariants before implementation, it verifies
each update against hand-computed references and boundary cases, and its one
performance claim is a measured baseline rather than an optimization.

Trigger domains (`rule.md` M2): domain 3 (architecture and contract — a new
crate, trait, and public symbols), domain 4 (numerics and performance — floating
point update rules plus a benchmark), and domain 1 (failures, inputs, and
resources — operand validation and untrusted restored state). The crate follows
the repository no-separator naming convention (`rule.md` NM1), the workspace
no-`unsafe` and no-`panic`/`unwrap`/`expect` rules (`AGENTS.md` sections 4 and
6), and owns its own `thiserror` error enum.

## Decision

The crate exposes eleven public items rooted at the crate root and re-exported
from collision-free module files:

| Public item | Module file | Responsibility |
| --- | --- | --- |
| `optimizer` (trait) | `optimizers.rs` | Update-rule contract and shared validation |
| `adamw` | `adamws.rs` | Bias-corrected AdamW with decoupled weight decay |
| `adamwconfig` | `adamws.rs` | Validated AdamW hyperparameters |
| `sgd` | `sgds.rs` | Momentum SGD with coupled weight decay |
| `sgdconfig` | `sgds.rs` | Validated SGD hyperparameters |
| `clipconfig` | `gradientclip.rs` | Validated clipping method |
| `clipgradients` | `gradientclip.rs` | In-place global-norm / value clipping |
| `lrschedule` | `scheduler.rs` | Pure step-to-rate schedule |
| `optimizerstate` | `state.rs` | Serializable optimizer snapshot |
| `statetensor` | `state.rs` | One named `f32` state tensor |
| `opterror` | `error.rs` | Crate-owned error enum |

### DS1 — Boundaries and interfaces

* `optim` depends on `tensor` (owned tensors and shapes) and `lib`
  (foundational types) only, plus `thiserror` for the error enum. No workflow
  crate is imported.
* The `optimizer` trait has seven methods (`getname`, `getlr`, `setlr`,
  `getstep`, `update`, `getstate`, `restore`). `update` takes the parameters and
  gradients as two parallel `&mut [tensor<f32>]` slices in a caller-defined and
  stable order; the optimizer keys its per-parameter state by that position.
* Public interface width: 11 crate-root items plus the seven trait methods, for
  18 counted symbols, under the `A4` bound of 20. *依据：单模块公共契约应能在一轮
  评审内完整复核（`rule.md` A4）。*

### DS2 — Invariants and error paths

Algorithm-level invariants:

* An update reads each parameter/gradient pair elementwise; a gradient's shape
  must equal its parameter's shape, and the two slices must have equal length.
* Moment buffers are lazily initialized to zero on the first non-empty update,
  then fixed in count and shape; a later update with a different parameter set
  is rejected.
* A step counter is incremented exactly once per successful non-empty update,
  so bias correction and state persistence observe the same count.
* `lrschedule::getlr` is total and never returns a negative rate; schedules with
  a `min` bound are monotone non-increasing after any warmup and never fall
  below `min`.

Implementation-level invariants:

* `update` validates lengths, shapes, and existing state before mutating any
  tensor, so a rejected call leaves parameters, gradients, and state unchanged.
* The element loop zips equal-length slices; the length equality follows from
  the shape check because a tensor's buffer length equals its shape's element
  count.
* `statetensor` construction and `frombytes` validate the element count against
  the shape, so `tobytes`/`frombytes` round-trip exactly and `optimizerstate`
  cannot carry a malformed tensor.

Error branches and return semantics (`opterror`):

* `tensor` — a transparent `tensorerror` propagated from shape, element-count,
  or tensor construction.
* `mismatchedcount` — parameter and gradient slice lengths differ.
* `shapemismatch` — a gradient's shape differs from its parameter's shape.
* `invalidstate` — initialized state or a restored snapshot is inconsistent
  (wrong tensor count, unknown or duplicate slot name, out-of-range index, or a
  missing slot).
* `invalidhyperparameter` — a learning rate, decay, epsilon, momentum, bound,
  period, or warmup value is outside its supported domain.

Resource rollback: all construction is value-based; on error no partially built
value is returned and owned buffers are dropped (scope release, `OW1`/`ER4`).
`update` performs its validation before the first mutation, so a rejected call
is a no-op. Gradients are clipped in place only after validation succeeds.

### DS3 — Complexity pre-control

Each function stays under the `rule.md` DS3 bounds of cyclomatic complexity ≤ 10
and ≤ 80 effective lines. Config validation is factored into small helpers
(`checklr`, `checknonnegative`, `checkunitinterval`, `fraction`), and the
update loops use iterator zips rather than nested indexing with branches.
*依据：超限函数无法在一轮评审内完成分支验证（`rule.md` DS3）。*

### DS4 — Concurrency design

`optim` introduces no shared mutable state and no synchronization primitive. An
optimizer owns its buffers outright and is mutated through `&mut self`, so two
threads cannot share one optimizer without an external lock and the type system
expresses that constraint (`CC2`). Tensors are borrowed mutably for the duration
of an update and returned to the caller afterward; no reference outlives the
call. No lock ordering is required because no lock exists.

## Update equations

An optimizer's state is keyed by parameter position. The completed-step count
`t` starts at 1 for the first update.

### AdamW (`adamw`)

```text
m = beta1 * m_prev + (1 - beta1) * g
v = beta2 * v_prev + (1 - beta2) * g * g
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
p = p - lr * ( m_hat / (sqrt(v_hat) + eps) + weightdecay * p )
```

`m` and `v` start at zero, so the first update uses `m = (1 - beta1) * g` and
`v = (1 - beta2) * g * g`. Bias correction divides by a strictly positive factor
because `beta1, beta2 < 1` and `t >= 1`. Weight decay is decoupled: it is applied
to the parameter, not added to the gradient, matching Loshchilov & Hutter's
AdamW.

### SGD (`sgd`)

```text
g_tilde = g + weightdecay * p
b = momentum * b_prev + g_tilde
p = p - lr * b
```

The velocity buffer `b` starts at zero. Weight decay is coupled (added to the
gradient before momentum accumulation), matching the classical L2 penalty and
PyTorch's SGD. With `momentum = 0`, `b = g_tilde` and the rule reduces to plain
SGD. Nesterov momentum and dampening are deliberately unsupported in this
stage.

## Gradient clipping

`clipconfig` is built through validated constructors, so `clipgradients` is
infallible:

* `clipconfig::none` leaves gradients unchanged.
* `clipconfig::byvalue(min, max)` clamps each element independently into the
  inclusive range; the constructor rejects non-finite bounds and `min > max`.
* `clipconfig::bynorm(maxnorm)` treats every element of every gradient as one
  vector and, when its L2 norm exceeds `maxnorm`, scales every element by
  `maxnorm / norm`. The norm is accumulated in `f64` for determinism and
  headroom. A zero norm, or a norm within `maxnorm`, leaves the gradients
  unchanged, so `maxnorm = 0` zeroes a non-zero gradient exactly with no
  division by zero; the constructor rejects a negative or non-finite `maxnorm`.

An optimizer may attach a `clipconfig` through its config's `withclip`; when
set, `update` clips the gradients in place before applying the update.

## Learning-rate schedules

`lrschedule` is a pure function from a zero-based step index to a rate; it holds
no state and is applied by the caller through `optimizer::setlr`. Let `s` be
the step index:

```text
constant:          lr
steplr:            max(min, initial * gamma^(s / steplen))
exponential:       max(min, initial * gamma^s)
cosineannealing:   min + (initial - min) * (1 + cos(pi * min(s, period) / period)) / 2
warmupcosine:      initial + (peak - initial) * s / warmup                  if s < warmup
                   min + (peak - min) * (1 + cos(pi * min(s - warmup, period) / period)) / 2   otherwise
```

Constructors validate every domain: non-negative rates, `gamma` in `[0, 1]`,
non-zero `steplen`/`period`, and `warmup <= period`.

## State and checkpoint compatibility

`optimizerstate` carries the completed step count, the current learning rate,
the parameter count, and the moment tensors. It is deliberately separate from
model architecture and from checkpoint encoding. Each moment is exposed as a
named `statetensor`:

* AdamW: `m.{i}` and `v.{i}` for parameter index `i`.
* SGD: `momentum.{i}`.

`statetensor::tobytes`/`frombytes` encode exactly the little-endian `f32`
bytes a checkpoint tensor needs, so a higher layer persists the snapshot by
adding each tensor to a checkpoint and restores it by reading the bytes back and
rebuilding `optimizerstate`. `restore` revalidates the snapshot (count, names,
element counts, learning rate) before adopting it, treating restored data as
untrusted (`IN1`). The training state that owns `step`/`epoch` remains the
checkpoint crate's `trainingstate`; `optim` and the caller keep the two in sync.
The `checkpoint` crate is therefore a development-only dependency used to prove
this contract in an integration test; it is not part of the production closure.

## Naming and module layout

The repository no-separator convention makes a module and a type re-exported
from it collide in one name namespace (`E0255`; see
`docs/decision-records/workspacelayout.md`). The plan's file list is therefore
translated (`rule.md` A8) so no module collides with its exported type:
`optimizer.rs` → `optimizers.rs` (trait `optimizer`), `adamw.rs` → `adamws.rs`
(struct `adamw`), and `sgd.rs` → `sgds.rs` (struct `sgd`). `gradientclip.rs`,
`scheduler.rs`, `state.rs`, `error.rs`, and `lib.rs` keep their names because
they export no same-named item.

## Alternatives considered

* **Depend on `autograd` and take `variable`/graph handles.** Rejected: the plan
  requires decoupling `optim` from any specific autograd graph, and `autograd`
  currently exposes gradients only through its own tape. Borrowing plain
  `tensor<f32>` slices keeps the update rules usable by any gradient producer.
* **Depend on `nn` and update `parameter` values.** Rejected:
  `nn::parameter` owns an immutable tensor with no mutable update path, and
  depending on `nn` from `optim` would pull a larger closure for no benefit; a
  `tensor`-based slice interface is smaller and sufficient.
* **Depend on `checkpoint` and serialize the whole optimizer state directly.**
  Rejected: `optim` would then own checkpoint encoding and invert the
  `train`-coordinates-persistence boundary (`docs/crate-boundaries.md` section
  7.4). Named `statetensor` values keep persistence at the caller.
* **A `dyn` optimizer trait object selected at runtime.** Rejected: no runtime
  selection requirement exists; concrete generic dispatch over the `optimizer`
  trait is sufficient (`A3`).
* **Embed the learning-rate schedule inside the optimizer.** Rejected: it would
  couple schedule evolution to optimizer state persistence and make both harder
  to test; a pure step-to-rate function applied via `setlr` is smaller and
  independently verifiable.
* **Attach gradient clipping as a mandatory step of `update` with an immutable
  gradient slice.** Rejected: it would force a per-step copy of the gradients.
  A validated optional `clipconfig` clips the caller's buffers in place, and a
  standalone `clipgradients` remains available for explicit control.
* **Nesterov momentum and dampening in SGD.** Rejected for this stage: they add
  branches and equations without a consumer yet; the documented rule set is the
  minimal representative pair (AdamW decoupled, SGD coupled).

## Consequences

* `optim` is a dependency leaf above `tensor` and `lib`. Adding it does not
  change any existing manifest because root membership is glob-based
  (`members = ["crates/*"]`).
* Update rules are testable without a graph, and state can be persisted through
  the existing checkpoint tensor contract without a production dependency edge.
* The benchmark was initially crate-local (`crates/optim/benches/optim.rs`)
  because the P9 instruction scoped new-operator benchmarks to the owning crate,
  with criterion declared as a development dependency and a `[[bench]]` entry in
  `crates/optim/Cargo.toml`. The later consolidation pass moved it to the
  repository-root `benches/optim.rs` (see `testconsolidation.md`), so
  `crates/optim` now keeps no `[[bench]]` target and no criterion dependency.
* The crate is acyclic; its only development-only edge is to `checkpoint` in the
  integration tests.

## Validation

Executed on 2026-10-10 against `crates/optim` in the real workspace (macOS
27.0.1, `aarch64-apple-darwin`, rustc 1.99.0, edition 2024):

```sh
cargo fmt --all                                     # exit 0
cargo check -p optim --all-targets                  # exit 0
cargo clippy -p optim --all-targets -- -D warnings  # exit 0, no warnings
cargo test -p optim                                 # 48 + 3 + 0 tests, 0 failed
cargo bench -p optim --no-run                       # exit 0
RUSTDOCFLAGS='-D warnings' cargo doc -p optim --no-deps   # exit 0
```

`cargo test -p optim` runs 48 inline unit tests, all passing. The 3 integration
tests moved to the repository-root `tests/optim.rs` in the later consolidation
pass (see `testconsolidation.md`).

Coverage:

* `adamw` — first-update hand computation; constant-gradient step size equal to
  the learning rate; decoupled weight decay under a zero gradient; in-place
  clipping; `setlr`; operand length/shape rejection without mutation; changed
  parameter set rejection; state round-trip; snapshot-count and slot-name
  rejection; invalid hyperparameters.
* `sgd` — plain descent; momentum accumulation (`-0.1, -0.29, -0.561`);
  coupled weight decay; value clipping; state round-trip; count rejection;
  invalid hyperparameters.
* `gradientclip` — none; by-value clamp; global by-norm scaling across multiple
  gradients; within-bound no-op; zero norm and zero limit without division by
  zero; invalid configurations.
* `lrschedule` — constant; step decay and its `min`; exponential; cosine
  boundaries and post-period clamp; warmup rise and cosine decay; invalid
  domains.
* `state` — construction and length rejection; byte round-trip including
  `f32::MAX`; short-buffer rejection; scalar and tensor accessors.
* `opterror` — `tensorerror` conversion; helper formatting; variant
  distinguishability.
* `tests/optim.rs` — checkpoint round-trip of AdamW and SGD state through
  `checkpoint::encode`/`decode`, continued-update equality, and a schedule
  driving `setlr`.

### Benchmark

Reproducible baseline (no improvement is claimed). Command:

```sh
cargo bench -p optim --bench optim -- --warm-up-time 1 --measurement-time 2 --sample-size 30
```

Workload: one `[1024, 1024]` `f32` parameter block (1,048,576 elements) and a
same-shaped gradient; one `update` per timed iteration with fresh buffers built
in criterion's setup phase. Release profile: `opt-level = 3` (default),
`lto = "thin"`, `codegen-units = 1`, `debug = 1`. Environment: macOS 27.0.1,
`aarch64-apple-darwin`, rustc 1.99.0 (the exact CPU brand string was not
readable under the command sandbox).

| Benchmark | Median | 95% interval | Throughput |
| --- | --- | --- | --- |
| `optim/adamw/step` | 762 µs | 753–775 µs | ~1.38 Gelem/s |
| `optim/sgd/step` | 160 µs | 159–162 µs | ~6.54 Gelem/s |
| `optim/clipgradients/bynorm` | 576 µs | 574–578 µs | ~1.82 Gelem/s |

AdamW performs roughly five times the per-element work of SGD (two moment
updates, bias correction, and a square root), which matches the observed ratio.
The clipping benchmark's by-norm pass reads every element twice (accumulate,
then scale); it is measured as a correctness-bearing baseline only.

## Revisit conditions

* When `autograd` exposes a stable public gradient accessor, revisit whether a
  thin adapter from graph variables to `tensor` slices belongs in `train`
  (preferred) rather than adding an `optim` → `autograd` edge.
* When a consumer needs Nesterov momentum, dampening, or a second adaptive
  method, extend the rule set behind the existing trait rather than widening
  `update`.
* When the training layer persists optimizer state, confirm the `m.{i}` /
  `v.{i}` / `momentum.{i}` naming and `optimizerstate` layout against the
  checkpoint tensor contract; a rename is a compatibility change.
* When a distributed or mixed-precision optimizer is required, revisit the
  `f32`-only and single-process assumptions.
