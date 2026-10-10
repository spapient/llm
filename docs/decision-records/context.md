# ADR: `context` — Execution Configuration, Cancellation, Memory Budgets, Worker Coordination, and Metrics

Status: Accepted
Scope: `crates/context` — public contracts, invariants, concurrency semantics, error semantics, and dependency/naming choices
Roadmap step: P6
Date: 2026-10-10

## Context

Roadmap stage P6 introduces the `context` crate. Per
`docs/crate-boundaries.md` section 5.13, `context` owns:

* Execution configuration.
* Cancellation contracts.
* Memory-budget abstractions.
* Worker coordination primitives where required.
* Metrics interfaces.

It MUST NOT own model architecture, domain-specific sampling algorithms, a
universal container for unrelated application state, or hidden global mutable
state. Its only workspace dependency is `lib` (for `requestid`); synchronization
comes from the Rust standard library only. It deliberately does **not** depend
on `infer`, `train`, `models`, or any other workflow crate, so it can be consumed
by both inference and training without a reverse edge.

The crate follows the repository no-separator naming convention (`rule.md`
NM1), the workspace no-`unsafe` and no-`panic`/`unwrap`/`expect` rules
(`AGENTS.md` sections 4 and 6), and owns its own `thiserror` error enum.

## Decision

The crate exposes eleven public items rooted at the crate root and re-exported
from collision-free module files:

| Public item | Module file | Responsibility |
| --- | --- | --- |
| `executionconfig` | `execution.rs` | Validated immutable bounds |
| `executioncontext` | `execution.rs` | Per-execution control owner |
| `cancellation` | `cancel.rs` | Cloneable `AtomicBool` handle + checkpoint |
| `memorybudget` | `budget.rs` | Capacity + atomic byte accounting |
| `memoryreservation` | `budget.rs` | RAII byte guard released on drop |
| `workergroup` | `worker.rs` | Bounded scoped-thread job runner |
| `workererror` | `worker.rs` | Generic coordination error |
| `metrics` | `metric.rs` | Narrow counter/gauge sink trait |
| `collectingmetrics` | `metric.rs` | In-memory sink for tests |
| `noopmetrics` | `metric.rs` | Discarding default sink |
| `contexterror` | `error.rs` | Crate-owned error enum |

### DS1 — Boundaries and interfaces

* `context` depends on `lib` only (`requestid`). It imports no higher-level
  crate and uses only `core`/`std` for atomics, threads, and mutexes.
* `executionconfig` is `Copy` and immutable; `executioncontext` owns the
  request-scoped controls and is created per execution. Neither is global and
  neither is stored in model weights, satisfying `OW2` and
  `docs/crate-boundaries.md` section 5.13.
* The `metrics` trait has two methods with `&'static str` names, keeping
  metric cardinality bounded and imposing no backend, registry, or crate
  coupling on foundational crates.
* Public interface width: 11 items at the crate root, under the `A4` bound of
  20. *依据：单模块公共契约应能在一轮评审内完整复核（`rule.md` A4）。*

### DS2 — Invariants and error paths

Algorithm-level invariants:

* `cancellation` starts not-cancelled; `cancel` is idempotent and, once set,
  every clone observes cancelled.
* `memorybudget`: `reserved ≤ capacity` at all times; a successful
  `tryreserve` increases `reserved` by exactly the requested bytes and the
  returned guard's drop decreases it by exactly those bytes.
* `executioncontext`: `steps ≤ maxsteps`; a successful `checkpoint` increments
  `steps` by one; cancellation is reported before the step budget is consulted.
* `workergroup::run` returns exactly `count` outputs, ordered by job index, when
  no failure occurs; on failure it returns the single first observed failure and
  no partial results.

Implementation-level invariants:

* `memorybudget` accounting uses `AtomicUsize::try_update` (the Rust 1.99
  replacement for the deprecated `fetch_update`) with
  `checked_add` in the update closure, so an overflowing request is rejected
  (`budgetoverflow`) rather than wrapping; a request that would exceed capacity
  is rejected (`budgetexceeded`) and reserves nothing.
* `executioncontext::checkpoint` uses the same compare-and-update pattern on its
  step counter, so concurrent checkpoints cannot exceed the step budget.
* `workergroup` distributes job indices with `AtomicUsize::fetch_add` and uses a
  `stop` flag plus a first-writer-wins `Mutex<Option<workererror<E>>>`; workers
  hold their local results separately and merge only after all joins, so no lock
  is held across job execution.

Error branches and return semantics (`contexterror`, 5 variants):

* `cancelled` — a checkpoint observed a cancelled execution.
* `stepbudgetexceeded { used, max }` — the step budget is exhausted.
* `budgetexceeded { requested, reserved, capacity }` — a reservation exceeds
  capacity.
* `budgetoverflow` — byte accounting would overflow `usize`.
* `invalidconfig { detail }` — zero `maxworkers` in `executionconfig::new` or
  `workergroup::new`; an execution that can run no work is a wiring error rather
  than a valid resource limit.

Coordination failures are modelled by `workererror<E>` with `context { source }`
(a `contexterror`), `job { index, source }` (the job's own error `E`), and
`workerpanicked`. Distinguishing these three lets a caller react differently to
cancellation, a caller-supplied job failure, and a worker panic.

Resource rollback: reservations are RAII guards and are released in `Drop` on
every termination path (normal return, `?` early return, cancelled loop,
thread unwind). This is the sole release mechanism; there is no explicit release
call (`OW1`, `ER4`, `NM3`). A failed `workergroup::run` returns no partial
outputs, so an aborted run cannot leak results into a caller's state (`OW3`).

### DS3 — Complexity pre-control

Each function stays under the `rule.md` DS3 bounds of cyclomatic complexity ≤ 10
and ≤ 80 effective lines. `workergroup::run` is the largest body and is kept to a
single scoped-thread loop plus a join loop with no nested control flow beyond the
job loop. *依据：超限函数无法在一轮评审内完成分支验证（`rule.md` DS3）。*

### DS4 — Concurrency design

Shared state and its synchronization, one primitive per resource:

| State | Primitive | Ordering |
| --- | --- | --- |
| `cancellation` flag | `Arc<AtomicBool>` | store `Release`, load `Acquire` |
| `memorybudget::reserved` | `AtomicUsize` | `try_update(AcqRel, Acquire)`, drop `fetch_sub(AcqRel)` |
| `executioncontext::steps` | `AtomicUsize` | `try_update(AcqRel, Acquire)` |
| `workergroup` job cursor | `AtomicUsize` | `fetch_add(AcqRel)` |
| `workergroup` stop flag | `AtomicBool` | store `Release`, load `Acquire` |
| `workergroup` first failure | `Mutex<Option<..>>` | scope-local, never held across a job |
| `collectingmetrics` maps | two `Mutex<BTreeMap>` | per-metric, poison-tolerant |

Cancellation visibility is established by the atomic itself: the `Release` store
and `Acquire` load form the required happens-before pair (`CC3`). The mutexes are
scope-local and are never nested, so no lock ordering exists and no deadlock
cycle is possible (`CC1`, `DS4`). `collectingmetrics` recovers from a poisoned
lock with `PoisonError::into_inner` rather than panicking, so a panicking writer
cannot wedge later readers. There is no global lock, no static mutable state, and
no cross-execution shared counter (`A6`).

`workergroup::run` is a blocking, cancellable call: it blocks until every worker
joins, checks `exec::checkpoint` before each job so cancellation and the step
budget propagate (`CC5`), and performs no observable side effect after a
cancellation is observed. Jobs return their own error value; the group does not
translate `E`.

### DS5 — Alternatives considered

* **Make `context` a universal `Vec<Box<dyn Any>>` request container.**
  Rejected: `docs/crate-boundaries.md` section 5.13 explicitly forbids a
  universal container; explicit typed fields keep ownership and lifecycle clear.
* **Use an `AtomicBool` plus a global registry of cancellation handles.**
  Rejected: a global registry is global mutable state (`A6`); the handle is
  cloned by value instead, so lifetime follows ownership.
* **Use a condition variable / `park`/`unpark` for cancellation.** Rejected:
  cancellation is a one-way latch, so a release/acquire `AtomicBool` is
  sufficient and simpler; polling at explicit checkpoints keeps cancellation
  deterministic and testable (`CC5`).
* **Track memory with a `Mutex<usize>` counter.** Rejected: a compare-and-update
  atomic gives the same accounting without a lock and without a lock-ordering
  obligation, and the RAII guard makes release infallible.
* **Have `memoryreservation` own bytes and return them via an explicit
  `release()`.** Rejected: `Drop`-based release is required by `OW1`/`NM3` and
  is the only path that is also correct on early return and unwind.
* **Make `metrics` take `&str` labels.** Rejected: unbounded label cardinality
  is an observability hazard; `&'static str` bounds it by construction.
* **Depend on a `tokio`/`rayon` executor for `workergroup`.** Rejected: the
  primitive is a bounded blocking fan-out, so `std::thread::scope` is enough and
  avoids an async runtime dependency on a foundational crate.
* **Make `workergroup` return partial results on failure.** Rejected: a caller
  must not be able to mistake an aborted run's outputs for a complete one
  (`OW3`); failure returns `Err` only.

## Consequences

* `context` is a dependency leaf above `lib`; adding it does not change any
  existing crate's manifest because root membership is glob-based
  (`members = ["crates/*"]`).
* Cancellation, step budget, and memory budget are explicit, testable, and
  distinguishable; a caller can tell "cancelled" from "out of memory" from
  "step budget exhausted".
* No global mutable state was introduced; every shared field is an atomic or a
  scope-local mutex with documented ordering.
* `workererror` is generic over the job error, so higher layers keep their own
  error types without translation.

## Validation

Commands are scoped to `context` because sibling crates are being built
concurrently. Executed on 2026-10-10 from `/Users/fei/Doubao/llm`:

```sh
cargo fmt --all
cargo check -p context --all-targets
cargo clippy -p context --all-targets -- -D warnings
cargo test -p context
RUSTDOCFLAGS='-D warnings' cargo doc -p context --no-deps
```

Results:

* `cargo clippy -p context --all-targets -- -D warnings` — pass.
* `cargo test -p context` — 28 unit tests + 5 integration tests, 0 failed.
* `RUSTDOCFLAGS='-D warnings' cargo doc -p context --no-deps` — pass.

Coverage:

* `cancel` — initial state, propagation to clones, idempotence, cross-thread
  propagation.
* `budget` — reserve/report, request past capacity, request equal to capacity,
  counter overflow, release on error path, concurrent reservations never
  exceeding capacity (four threads racing for two slots).
* `metric` — counter accumulation, gauge overwrite, absent metrics, noop sink.
* `execution` — zero-worker rejection, configuration exposure, checkpoint
  advance and step-budget enforcement, cancellation precedence, budget sizing,
  metrics forwarding, optional metrics.
* `worker` — zero-worker rejection, zero jobs, index-ordered results, job-error
  propagation (single worker), step-budget exhaustion, cancellation before any
  job.
* `error` — structured context in `Display`, distinct causes.
* `tests/context.rs` — end-to-end composition of cancellation, step budget,
  memory budget, worker coordination, and metrics through the public API.

## Revisit conditions

* When more than one lock must be held across a shared structure, define a
  global lock order and record it here (`CC1`/`DS4`).
* When cancellation must interrupt a blocking leaf call (not just between
  jobs), add a cancellable wait primitive rather than a global registry.
* When a production metrics backend is required, add it as a new `metrics`
  implementation, keeping the trait free of backend coupling.
* When worker coordination needs async execution or work stealing with
  priorities, revisit the `std::thread::scope` design and justify the added
  dependency.
