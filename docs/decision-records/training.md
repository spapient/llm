# ADR: `train` — Training Orchestration, Loss, Gradient Accumulation, Step, Evaluation, and Resume

Status: Accepted
Scope: `crates/train` — public contracts, the training-step contract, loss/gradient validation, failure and cancellation semantics, resume against persisted state, the backward-path limitation, and dependency/naming choices
Roadmap step: P10
Date: 2026-10-10

## Context

Roadmap stage P10 introduces the `train` crate. Per `docs/crate-boundaries.md`
section 5.14, `train` owns:

* Training-loop orchestration.
* Training configuration.
* Loss calculation and workflow integration.
* Gradient accumulation scheduling.
* Evaluation workflows.
* Resume orchestration.

It MUST NOT own CPU kernels, tensor storage, optimizer internals, or external
checkpoint encoding. Per section 7.4, `train` coordinates save and restore while
`checkpoint` owns persistence semantics; the checkpoint crate never depends on
the training loop to interpret its format.

Role and responsibility (`rule.md` IDN3): this crate is the training-workflow
owner. It defines boundaries and invariants before implementation, it verifies
each step against measured loss decrease and continued-training equivalence, and
its only performance claims are the measured loss values it asserts.

Trigger domains (`rule.md` M2): domain 1 (failures, inputs, and resources —
untrusted restored state and gradient validation), domain 2 (concurrency and
shared state — cancellation and request-scoped training state), domain 3
(architecture and contract — a new crate and its public symbols), and domain 5
(delivery and governance — the resume contract and the disclosed limitation).

### Reality constraint that shapes the design

`nn` and `models` are **forward-only** today: they build and run a forward pass
but expose no autograd backward path, and `autograd` provides vector-Jacobian
products only for `matmul`, `add`, `mul`, `rmsnorm`, `softmax`, `relu`, `silu`,
and `gelutanh`. `optim` is deliberately graph-independent and updates owned
`tensor<f32>` parameters from borrowed gradient slices. A full end-to-end "train
a `models` Llama by backprop" loop is therefore not achievable at this stage
without inventing an autograd path through `nn`, which this stage must not do.
The crate is designed around the real, available contract instead.

## Decision

The crate exposes thirteen public items rooted at the crate root and re-exported
from collision-free module files:

| Public item | Module file | Responsibility |
| --- | --- | --- |
| `trainconfig` | `trainconfigs.rs` | Validated run configuration |
| `trainer` | `trainers.rs` | Loss/step loop over parameters and gradients |
| `stepresult` | `step.rs` | One completed step's loss, step, and rate |
| `crossentropy` | `loss.rs` | Stable cross-entropy forward and gradient |
| `lossresult` | `loss.rs` | Scalar loss plus logit gradient |
| `gradientaccumulator` | `gradientaccumulation.rs` | Window-averaged gradient buffer |
| `evalsample` | `evaluation.rs` | Logits paired with target classes |
| `evalresult` | `evaluation.rs` | Row-weighted loss and perplexity |
| `evaluate` | `evaluation.rs` | Evaluation aggregation |
| `saveresume` | `resume.rs` | Write parameters, optimizer state, progress |
| `loadresume` | `resume.rs` | Validate and restore them atomically |
| `resumestate` | `resume.rs` | Restored step and epoch |
| `trainerror` | `error.rs` | Crate-owned error enum |

### DS1 — Boundaries and interfaces

* `train` depends in production on `lib`, `tensor`, `optim`, `checkpoint`, and
  `context`, plus `thiserror` for the error enum. `autograd` and `cpu` are
  **development-only** dependencies used to prove the loss-decrease contract in
  integration tests; they are not part of the production closure.
* It does **not** depend on `nn` or `models`. Both are forward-only, so nothing
  in this crate could call them for training; adding the edge would enlarge the
  dependency closure for no consumer (`AGENTS.md` section 3.2). The limitation
  this represents is declared below rather than papered over.
* It does **not** depend on `data`. Evaluation is deliberately decoupled from
  the dataset representation: because `train` owns no forward pass, the caller
  builds `evalsample` values (logits plus targets) from whatever batch or
  dataset representation it has. This keeps `train` from acquiring a dependency
  it does not use.
* The loss function is passed per step as `FnMut(&[tensor<f32>]) ->
  Result<(f32, Vec<tensor<f32>>), trainerror>`, so a hand-written gradient, a
  `crossentropy` result, and an `autograd`-computed gradient all drive the same
  loop with static dispatch (`A3`).
* Public interface width: 13 crate-root items, under the `A4` bound of 20.
  *依据：单模块公共契约应能在一轮评审内完整复核（`rule.md` A4）。*

### DS2 — Invariants and error paths

Algorithm-level invariants:

* One optimizer update is applied per `accumulationsteps` micro-batches. The
  window is averaged by `1 / count`, so a window of ragged micro-batches is
  balanced.
* The learning rate applied to a step is `schedule.getlr(stepindex)` for
  `stepindex` equal to the number of completed steps, or the optimizer's current
  rate when no schedule is configured.
* `crossentropy` returns the exact analytic gradient of its own loss
  (softmax minus one-hot, mean over rows); the finite-difference test in
  `loss.rs` pins this to within `1e-3`.
* `loadresume` restores the optimizer snapshot only if its own step count equals
  the persisted training step and the optimizer name matches the trainer's.

Implementation-level invariants:

* State advances only on a completed `optimizer::update`. On any loss, gradient,
  cancellation, schedule, or optimizer error the buffered window is zeroed and
  neither the local step counter nor the optimizer's counter moves.
* `optimizer::update` validates lengths, shapes, and existing state before
  mutating, so a rejected update leaves parameters and state unchanged; `train`
  relies on that contract and adds no partial mutation of its own.
* `gradientaccumulator::accumulate` validates count and shapes before adding any
  element, so a rejected micro-batch leaves the buffered sum unchanged.
* `loadresume` is atomic with respect to the trainer: the optimizer state is
  restored (and may fail) before parameters or counters are replaced, so a
  rejected load is a no-op on the trainer (`OW3`).

Error branches and return semantics (`trainerror`):

* `tensor` — a transparent `tensorerror` from shape, element-count, or tensor
  construction.
* `optim` — a transparent `opterror` from an update or state operation.
* `checkpoint` — a transparent `checkpointerror` from load, save, or a
  compatibility check.
* `cancelled` — a cancellation token was observed.
* `invalidaccumulation` — a zero-length accumulation window.
* `emptyparameters` — a trainer was constructed with no parameters.
* `steplimitreached` — the configured step budget is exhausted.
* `gradientcountmismatch` / `gradientshapemismatch` — the loss function returned
  the wrong number of gradients, or a gradient whose shape differs from its
  parameter.
* `nonfiniteloss` — a loss value that is not finite.
* `accumulatorfull` — a gradient was added to a full window.
* `invalidstate` — a local inconsistency, including an empty `take`, a resume
  with inconsistent metadata, or a shape mismatch on restore.
* `lossnotmatrix` / `lossemptymatrix` / `targetcountmismatch` /
  `targetoutofrange` / `lossnonfinite` — cross-entropy contract violations.
* `unsupportedtensordtype` — a checkpoint tensor that must be `f32` is not.

Resource rollback: buffers are value-based; dropping a partially built value
releases its storage (`OW1`/`ER4`). `take` moves the averaged gradients out and
re-zeroes the buffer in the same call, so no window spans a failure.

### DS3 — Complexity pre-control

Each function stays under the `rule.md` DS3 bounds of cyclomatic complexity ≤ 10
and ≤ 80 effective lines. The step loop's fallible work is expressed with
`match`/early-return and a single `resetwindow` helper; configuration and
loss validation are factored into small helpers. *依据：超限函数无法在一轮评审内
完成分支验证（`rule.md` DS3）。*

### DS4 — Concurrency design

`train` introduces no shared mutable state and no synchronization primitive of
its own. A `trainer` owns its parameters, optimizer, accumulator, and counters
and is mutated through `&mut self`, so the type system prevents two threads from
sharing one trainer without an external lock (`CC2`). Cancellation is the only
shared signal; it uses `context::cancellation`, a cheaply cloneable handle over
an `AtomicBool` with `Release`/`Acquire` ordering, which is an explicit
synchronization primitive rather than global state (`A6`/`CC3`). Cancellation is
checked before each micro-batch and again before the update.

## The training-step contract

`trainer::step(loss)` runs exactly `accumulationsteps` micro-batches and applies
one update:

1. For each micro-batch: check the step budget and cancellation; call the loss
   function with the current parameters; reject a non-finite loss; validate and
   add the gradients to the window.
2. On the micro-batch that fills the window: check cancellation again, take the
   averaged gradients, resolve and apply the scheduled rate, call
   `optimizer::update` once, and increment the step counter.
3. Return a `stepresult` carrying the mean of the window's micro-batch losses,
   the new step count, the applied rate, and the window size.

`trainer::accumulatemicrobatch` exposes the same logic one micro-batch at a
time, returning `None` while filling and `Some(stepresult)` on the closing
micro-batch. `trainer::step` requires an empty window on entry so a partial
window is never silently discarded; `trainer::step` is implemented as a bounded
loop over `accumulatemicrobatch`.

## Loss and gradient validation

`crossentropy(logits, targets)` requires a rank-two `[rows, classes]` matrix with
at least one row and one class, and exactly one in-range target per row. It
computes `logsumexp(row) - logit[target]` with max subtraction, so large logits
shift toward zero instead of overflowing `f32`, and returns the mean loss plus
the analytic gradient `(softmax - onehot) / rows`. It rejects a non-finite loss
or gradient and never mutates its inputs. The gradient is checked against a
central finite difference in the module tests.

## No state advance on failure or cancellation

State advances only through a completed `optimizer::update`. On cancellation or
any error the buffered window is discarded through `resetwindow` and the step
counter is untouched; the optimizer's validation-before-mutation contract keeps
its own state unchanged. Integration tests assert that after a between-steps
cancellation and after a mid-window cancellation the parameters, the local step
counter, and the optimizer step counter are all unchanged and the window is
empty.

## Resume against persisted state

`saveresume` writes a `checkpoint` container holding `p.{i}` parameter tensors,
`trainingstate { step, epoch }`, and the optimizer snapshot's tensors under their
own names (`m.{i}`, `v.{i}`, `momentum.{i}`), with the optimizer name, step, rate,
parameter count, and tensor-name list recorded as manifest metadata.
`loadresume` checks the container version, verifies the parameter specs with
`checkcompat`, decodes the parameters, reconstructs `optimizerstate` from the
metadata and named tensors, and restores it before replacing parameters and
counters. A checkpoint written for a different optimizer, or whose optimizer step
disagrees with the training step, is rejected rather than applied silently. A
shape mismatch and a missing file are both rejected without mutating the trainer.

## Declared limitation: no backward path through `nn`/`models`

Training through the full `nn`/`models` Llama stack is **not wired**, because
`nn` and `models` have no backward path and `autograd` lacks the operators to
express a differentiable Llama forward (and, in particular, a cross-entropy
backward). This crate does not fabricate one. What remains is a backward path
through the `nn`/`models` layers — autograd operators and VJPs for the
forward-only layers (normalization, attention, rotary, embedding, and model
assembly) so a forward pass produces a reversible graph — after which the same
`trainer` loop here would consume per-parameter gradients unchanged. This is a
declared limitation (`rule.md` M4/VF5), stated in the crate docs as well.

The loss-decrease evidence is therefore taken where it is real: on an
`autograd` graph built only from operators that currently have VJPs
(`matmul`+`add`+`mul` least squares with the known optimum `w = 2`), and on the
crate's own cross-entropy loss and analytic gradient.

## Naming and module layout

The repository no-separator convention makes a module and a type re-exported from
it collide in one name namespace (`E0255`). The plan's file list is translated
(`rule.md` A8) so no module collides with its exported type: `trainer.rs` →
`trainers.rs` (type `trainer`) and `trainconfig.rs` → `trainconfigs.rs` (type
`trainconfig`). `step.rs` (exports `stepresult`), `loss.rs`, `evaluation.rs`,
`resume.rs`, `gradientaccumulation.rs`, `error.rs`, and `lib.rs` keep their
names because they export no same-named item.

## Alternatives considered

* **Invent a minimal autograd path through one `nn` layer to demonstrate
  end-to-end training.** Rejected: the instruction forbids fabricating an
  unrequested backward path, and a demo built on a private ad-hoc backward would
  misrepresent the stack's readiness.
* **Depend on `nn`/`models` and reference them in docs but not in code.**
  Rejected: an unused dependency edge enlarges the closure for no consumer
  (`AGENTS.md` section 3.2); the limitation is documented instead.
* **Make the loss a boxed `dyn FnMut` field on the trainer.** Rejected: no
  runtime extensibility requirement exists; a per-call generic method keeps
  static dispatch and avoids a heap allocation (`A3`).
* **Expose a separate `flush` that applies a partial window.** Rejected: a
  partial window is exactly the state that must not silently advance training;
  `step` requires an empty window and `accumulatemicrobatch` exposes the
  primitive instead.
* **Store the `f32` scalars of the optimizer snapshot in tensors.** Rejected:
  the checkpoint container already carries arbitrary string metadata and a
  typed `trainingstate`; encoding step/rate/count as metadata avoids inventing
  fake scalar tensors.
* **Depend on `data` to consume a loader during evaluation.** Rejected: with no
  forward pass in `train`, the caller must produce logits anyway, so `data` would
  be an unused edge; the decoupled `evalsample` contract is sufficient.

## Consequences

* `train` is acyclic above `lib`, `tensor`, `optim`, `checkpoint`, and `context`.
  Adding it changes no existing manifest because root membership is glob-based
  (`members = ["crates/*"]`).
* The training step is testable without a graph, and the same loop consumes
  `autograd` gradients from the dev-dependency graph used in tests.
* The crate does not depend on the forward-only layers, so it does not encode a
  false claim of end-to-end training.
* The divergence from the plan's root `tests/training_resume.rs` was intentional
  at this stage (the P10 instruction scoped tests crate-local under
  `crates/train/tests/*.rs`); the later consolidation pass moved them to the
  repository-root `tests/` as `trainingresume.rs`, `lossdecrease.rs`, and
  `cancellation.rs` (see `testconsolidation.md`).

## Validation

Executed on 2026-10-10 against `crates/train` in the real workspace (macOS
27.0.1, `aarch64-apple-darwin`, rustc 1.99.0, edition 2024):

```sh
cargo fmt --all                                          # exit 0
cargo check -p train --all-targets                       # exit 0
cargo clippy -p train --all-targets -- -D warnings       # exit 0, no warnings
cargo test -p train                                      # 48 tests, 0 failed
RUSTDOCFLAGS='-D warnings' cargo doc -p train --no-deps  # exit 0
```

`cargo test -p train` runs 40 inline unit tests and 1 doc-test, all passing. The
7 integration tests moved to the repository root (`tests/cancellation.rs` 2,
`tests/lossdecrease.rs` 2, `tests/trainingresume.rs` 3; see
`testconsolidation.md`).

### Measured loss decrease (E1)

Both decreases are asserted against measured values with fixed RNG seeds; the
numbers below are printed by the tests.

* `autograd_least_squares_decreases_the_loss`: SGD (`lr = 0.01`, no momentum,
  seed 2024) over `matmul`+`add`+`mul` on `y = 2x`, 300 steps.
  Measured loss `56.0 → 2.98e-13` and final weight `1.9999999` against the
  analytical optimum `2`.
* `cross_entropy_descent_decreases_the_loss`: SGD (`lr = 0.3`, seed 7) on the
  crate's cross-entropy and analytic gradient, two separable rows over three
  classes, 500 steps. Measured loss `1.0986123` (`ln 3`, the uniform-logit
  value) `→ 0.009226322`.

### Behaviors covered

* `loss` — uniform logits give `ln(classes)`; confident-correct loss is lower;
  gradient matches a central finite difference; huge logits do not overflow;
  non-matrix, empty, wrong-target-count, and out-of-range targets are rejected.
* `gradientaccumulation` — arithmetic mean over a window; zero capacity, empty
  parameters, overflow, wrong count, wrong shape, and empty `take` are rejected;
  reset re-zeroes the buffer.
* `trainer` — a quadratic loss decreases over 20 steps; accumulation of 3 gives
  the hand-computed `w = 0.8` after one update; empty parameters rejected; `step`
  requires an empty window; cancellation does not advance state; non-finite loss
  does not advance state; the step budget is enforced; a wrong gradient count
  resets the window.
* `crossentropy`/`evaluate` — row-weighted aggregation and perplexity; empty
  samples rejected.
* `step`/`resume` unit tests — schedule application, step-result fields, `f32`
  byte round-trip, and bad-byte-length rejection.
* `tests/loss_decrease.rs` — the two measured decreases above.
* `tests/training_resume.rs` — parameter and optimizer-state round-trip,
  continued-training equivalence, shape-mismatch rejection without mutation,
  and missing-file reporting.
* `tests/cancellation.rs` — between-steps and mid-window cancellation both leave
  parameters, step counters, and the window unchanged.
* `error` — transparent `tensorerror`/`opterror` conversion, helper formatting,
  and cancellation distinguishability.

## Revisit conditions

* When `nn`/`models` gain a backward path (or `autograd` gains the missing
  operators), wire the real forward pass into the per-step loss closure and
  replace the least-squares demonstration with a `models` end-to-end
  loss-decrease test.
* When a training loop obtains gradients directly from an `autograd` graph, add
  the thin graph-to-`tensor` adapter in `train` (not in `optim`).
* When data-parallel or multi-device training is required, revisit the
  single-owner trainer and add explicit synchronization for gradient reduction.
* When mixed precision is required, revisit the `f32`-only loss, accumulator,
  and checkpoint encoding.
