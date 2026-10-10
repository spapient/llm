# ADR: `data` — Dataset, Packing, Batching, Sampling, and Loader

Status: Accepted
Scope: `crates/data` — public contracts, invariants, resource limits, error semantics, and dependency/naming choices
Date: 2026-10-10

## Context

Roadmap stage P8 introduces the `data` crate. Per `docs/crate-boundaries.md`
section 5.9, `data` owns:

* Dataset interfaces.
* Records and text-dataset representations.
* Packing and batching.
* Sampling.
* Data-loader lifecycle and error propagation.

It MUST NOT own model forward execution, optimizer updates, inference request
state, or checkpoint file formats. Its only workspace dependency is `lib`
(`tokenid`, `seed`, `limits`). It deliberately does **not** depend on
`tokenizer`: converting raw text to token identifiers is a composition owned by a
higher layer, so this crate accepts already-tokenized documents. This keeps the
dependency direction one-way and avoids a low-level reverse dependency.

The crate follows the repository no-separator naming convention (`rule.md`
NM1), the workspace no-`unsafe` and no-`panic`/`unwrap`/`expect` rules
(`AGENTS.md` sections 4 and 6), and owns its own `thiserror` error enum.

## Decision

The crate exposes nine public items rooted at the crate root and re-exported
from collision-free module files:

| Public item | Module file | Responsibility |
| --- | --- | --- |
| `record` | `records.rs` | Owns an ordered `Vec<tokenid>` |
| `dataset` (trait) | `datasets.rs` | `getlen`, `isempty`, `getrecord` |
| `textdataset` | `textdatasets.rs` | In-memory concrete `dataset` |
| `packing` | `packings.rs` | Fixed-length sequence packing |
| `batch` | `batches.rs` | Rectangular batch of tokens |
| `sampler` | `samplers.rs` | Reproducible index sampler |
| `loader` | `loaders.rs` | Lifecycle-explicit epoch iterator |
| `loaderconfig` | `loaders.rs` | Plain-data loader configuration |
| `dataerror` | `error.rs` | Crate-owned error enum |

### DS1 — Boundaries and interfaces

* `data` depends on `lib` only. No `tokenizer`, `tensor`, `api`, `cpu`, or any
  workflow crate is imported.
* `dataset` is a narrow capability trait returning borrowed records; it holds no
  iteration state, so a dataset can be shared immutably while a `loader` keeps
  request-scoped cursors privately (`docs/crate-boundaries.md` section 5.9;
  `AGENTS.md` sections 3.3 and 5).
* Public interface width: 9 items at the crate root, well under the `A4` bound
  of 20. *依据：单模块公共契约应能在一轮评审内完整复核（`rule.md` A4）。*

### DS2 — Invariants and error paths

Algorithm-level invariants:

* A `batch` holds exactly `batchsize * seqlength` tokens; every row has the same
  length. A `batch` always has at least one row.
* Every packed sequence produced by `packing` has length `seqlength`, except a
  final partial sequence which has length in `1..seqlength`; empty input yields
  no sequence.
* A `sampler` shuffle is a permutation of `0..len`.
* After construction, a `loader`'s record order contains exactly the indices
  `0..data.getlen()`, and `position`/`offset` always address a valid location or
  the end of that order.

Implementation-level invariants:

* `batch::gettokens().len() == batchsize * seqlength` for every constructed
  `batch`; `getrow` uses checked multiply/add and slice `get` rather than an
  unchecked index.
* `packing::pack` allocates at most the validated total token count.
* `loader::fillsequence` only advances `position`/`offset` after fully consuming
  a record; `remaining` is obtained with `get(offset..)` so a violated invariant
  becomes a typed error rather than a panic.

Error branches and return semantics (`dataerror`):

* `indexoutofrange` — `dataset::getrecord`, `batch::getrow` with `index >= len`.
* `emptydataset` — `sampler::sample` with positive `count` on an empty range.
* `zeroseqlength` / `zerobatchsize` — zero dimensions in `packing`, `batch`,
  `loader`.
* `emptybatch` — `batch::fromrecords` with zero records.
* `recordeceedseqlength` — a record longer than the batch `seqlength`.
* `lengthmismatch` — a token buffer whose length disagrees with the expected
  element count.
* `elementcountoverflow` — a checked multiply/add overflows `usize`.
* `elementcountlimitexceeded` / `sequencelenlimitexceeded` /
  `batchlimitexceeded` — a requested value strictly exceeds the corresponding
  `lib::limits` bound.

Resource rollback: all construction is value-based. On error, no partially built
value is returned and all owned buffers are dropped (scope-based release, `OW1`
and `ER4`). `batch::fromrecords` validates dimensions and the element budget
before reserving, and reserves fallibly with `Vec::try_reserve_exact`, so a
rejected input allocates nothing.

### DS3 — Complexity pre-control

Each function stays under the `rule.md` DS3 bounds of cyclomatic complexity ≤ 10
and ≤ 80 effective lines. The loader is decomposed into `getnext`,
`fillsequence`, and `finish` so each stays small. *依据：超限函数无法在一轮评审
内完成分支验证（`rule.md` DS3）。*

### DS4 — Concurrency design

`data` introduces no shared mutable state and no synchronization primitive. A
`loader` borrows its dataset immutably and owns all cursor, sampler, and buffer
state privately, so two loaders (or a loader and any reader of the dataset)
cannot race. This satisfies the request-isolation rule (`OW2`) with types rather
than a runtime convention (`CC2`). No lock ordering is required because no lock
exists.

## Sampling reproducibility

`sampler` owns a SplitMix64 generator seeded from `lib::seed`. The algorithm is
fixed:

```text
state = seed
nextu64():
    state += 0x9E3779B97F4A7C15   (wrapping)
    z = state
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9   (wrapping)
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB   (wrapping)
    return z ^ (z >> 31)
```

`shuffle(len)` performs an in-place Fisher–Yates pass from index `len-1` down to
`1`, swapping each index with `bound(i + 1)`. `sample(len, count)` draws `count`
indices with replacement via `bound(len)`. `bound(upper)` uses rejection
sampling — it rejects a draw below `(u64::MAX - upper + 1) % upper` and otherwise
returns `draw % upper` — so the index distribution is unbiased. Because every
step is deterministic and platform-independent, two samplers from the same seed
produce identical sequences, and successive mutating calls advance the stream.

The loader uses a `sampler` for its per-epoch record order. Construction starts
epoch 1; each `reset` advances the stream and rebuilds a fresh permutation.
Consequently the *sequence of epochs* is reproducible from one seed, while
distinct epochs differ.

## Batch shape and boundary behavior

* `batch::new(tokens, batchsize, seqlength, lim)` requires
  `tokens.len() == batchsize * seqlength` exactly; any other length is
  `lengthmismatch`. Dimensions are validated first, so a zero dimension or an
  exceeded batch/sequence budget is reported before the length is examined.
* `batch::fromrecords(records, seqlength, pad, lim)` is the padding constructor.
  Each record is copied and padded to `seqlength` with the caller-supplied `pad`.
  A record longer than `seqlength` is rejected with `recordeceedseqlength`
  rather than silently truncated, so a misconfigured `seqlength` fails loudly.
  An empty record set is rejected with `emptybatch`.
* Rows are laid out row-major; `getrow(r)` returns
  `[r * seqlength, (r + 1) * seqlength)` and rejects `r >= batchsize`.
* The padding token is supplied by the caller; the crate never invents a default
  token (`AGENTS.md` section 5; `IN4`).

Packing boundary behavior (`packing::pack`):

* Empty input → empty sequence list.
* Token count an exact multiple of `seqlength` → only full sequences.
* Otherwise the final sequence is shorter and left unpadded; callers that need a
  rectangle pad it through `batch`.
* A record longer than `seqlength` is split across several sequences; tokens
  from different records may share a sequence.

## Loader failure, termination, and lifecycle

`loader::getnext` returns:

* `Ok(Some(batch))` while the epoch yields batches.
* `Ok(None)` exactly at epoch exhaustion, and on every subsequent call until
  `reset`.
* `Err(dataerror)` on a record-access or batch-construction failure.

When the epoch is exhausted and a trailing partial batch exists, `droplast`
selects the behavior: `true` discards it (returning `Ok(None)`), `false` pads the
remaining rows with the pad token to a full rectangle and returns it. Either way
the loader becomes terminal and keeps returning `Ok(None)`.

A returned error also marks the loader terminal, so a failed loader is never
silently reused; the caller must `reset` or reconstruct it. This makes the
failure path reset request-scoped state (`OW3`). All loader buffers are owned and
released when the loader is dropped; there is no manual release verb (`NM3`).

## Resource limits

All externally influenced sizes are validated against an explicit
`lib::limits` budget:

| Operation | Bound enforced |
| --- | --- |
| `textdataset::new` | record count and total token count ≤ `maxelements` |
| `packing::pack` | total token count ≤ `maxelements` |
| `batch::new` / `fromrecords` | `batchsize` ≤ `maxbatch`; `seqlength` ≤ `maxsequencelen`; `batchsize * seqlength` ≤ `maxelements` |
| `sampler::shuffle` / `sample` | `len` / `count` ≤ `maxelements` |
| `loader::new` / `reset` | the above for its config and record order |

Overflow in any element-count product is detected with checked arithmetic and
reported as `elementcountoverflow`. The loader's per-batch buffer capacity is
`batchsize * seqlength`, already bounded by `maxelements`. Limits are always
supplied by the caller; `data` invents no default bound.

## Naming and module layout

The repository no-separator convention makes a module and a type re-exported
from it collide in one name namespace (`E0255`; see
`docs/decision-records/workspacelayout.md`). The module files are therefore
plural containers (`records.rs` → `record`, `datasets.rs` → `dataset`,
`textdatasets.rs` → `textdataset`, `packings.rs` → `packing`, `batches.rs` →
`batch`, `samplers.rs` → `sampler`, `loaders.rs` → `loader`), matching the
existing `lib` and `tensor` crates. This is the repository's `rule.md` A8
translation of the roadmap's filename list onto Rust.

## Alternatives considered

* **Depend on `tokenizer` and tokenize text inside `data`.** Rejected: it would
  make a lower-level data crate depend on a higher-level text crate, creating a
  reverse edge and coupling. The boundary requires tokenizer composition to live
  above `data`. `docs/crate-boundaries.md` section 5.9 and the P8 plan both
  assign tokenizer composition to a higher layer.
* **Make `textdataset` a lazy tokenizing dataset.** Rejected: the `dataset`
  contract returns borrowed records, so a lazily tokenized implementation cannot
  return a reference to a freshly built record. Eager materialization plus a
  caller-supplied tokenizer upstream keeps the trait object-safe and simple.
* **Pack with an explicit EOS/BOS separator between records.** Rejected for this
  stage: it invents token semantics that belong to the tokenizer/model
  configuration. Packing is defined purely as concatenation; separators can be
  inserted by the caller before packing.
* **Make the loader own an owned `Vec<record>` copy of the dataset.** Rejected:
  it would duplicate potentially large data and break the immutable-share model.
  The loader borrows the dataset instead.
* **Use a runtime-selected sampler via `dyn`.** Rejected: no runtime selection
  requirement exists (`A3`); a single concrete sampler keeps dispatch static.
* **Introduce a shared `data`-independent PRNG crate.** Rejected: the required
  algorithm is a few lines, and extracting a crate would enlarge the dependency
  closure for no independent responsibility (`A5`).

## Consequences

* `data` is a dependency leaf above `lib`; adding it does not change any existing
  crate's manifest because root membership is glob-based (`members = ["crates/*"]`).
* Sampling and loader epochs are reproducible from a seed, which makes training
  and evaluation runs reproducible.
* The crate is acyclic and depends only on `lib`.

## Validation

Executed on 2026-10-10 against `crates/data` in the real workspace (macOS,
edition 2024, rust-version 1.99):

```sh
cargo fmt --all -- --check                                  # exit 0
cargo check -p data --all-targets                           # exit 0
cargo clippy -p data --all-targets -- -D warnings           # exit 0, no warnings
cargo test -p data                                          # 49 + 3 + 0 tests, 0 failed
RUSTDOCFLAGS='-D warnings' cargo doc -p data --no-deps      # exit 0
```

`cargo test -p data` runs 49 inline unit tests, all passing. The 3 end-to-end
integration tests moved to the repository-root `tests/pipeline.rs` in the later
consolidation pass (see `testconsolidation.md`).

While the sibling `tokenizer`/`format`/`checkpoint`/`context` crates were being
created concurrently, the workspace root manifest was transiently unparseable
(glob membership `crates/*`). The same commands were therefore also run in an
isolated workspace containing only `lib` and `data`, with the workspace lint and
package configuration mirrored; the results were identical (clippy exit 0, 49 + 3
tests passing, doc exit 0). The isolated run is a supplement, not a substitute;
the final numbers above are from the real workspace.

Coverage:

* `record` — ordering, empty record.
* `dataset` / `textdataset` — indexed access, out-of-range, empty, record-count
  and token-count limits at the boundary and just above it.
* `packing` — empty input, exact multiple, partial tail, cross-record
  concatenation, long-record splitting, empty records, token budget.
* `batch` — rectangular construction, length mismatch, zero dimensions, padding,
  overlong record, empty input, row access, batch/sequence/element budgets.
* `sampler` — permutation property, same-seed reproducibility, stream advance,
  short inputs, range and count of draws, empty dataset, element budget.
* `loader` — padding across batches, `droplast`, cross-record packing, empty
  dataset, identity and shuffled orders, epoch reproducibility, epoch reset,
  terminal `None`, config validation, generic `dataset` bound.
* `dataerror` — distinguishing values and display.
* `tests/pipeline.rs` — end-to-end composition of dataset, packing, batch,
  sampler, and loader through the public API.

## Revisit conditions

* When a higher layer needs a lazy or streamed dataset that cannot be
  materialized in memory, revisit the borrowed-record `dataset` contract.
* When record-level metadata (document ids, labels, attention masks) is needed,
  extend `record` rather than overloading the token buffer.
* When multi-worker loading is required, revisit loader ownership; the current
  design is single-owner and requires no synchronization.
