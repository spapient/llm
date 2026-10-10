# ADR: Persistence Layer (`format` + `checkpoint`)

Status: Accepted
Scope: External artifact parsing (`format`) and checkpoint persistence (`checkpoint`)
Date: 2026-10-10
Roadmap step: P5

## Context

The runtime needs to read external model artifacts and persist model and
training state. Two responsibilities are involved and they are not the same:

* **Parsing untrusted external bytes** into validated metadata — an
  `config.json` and a safetensors container. This is a pure, read-only
  transformation of untrusted input and owns no state.
* **Persisting and reloading** our own checkpoint artifact, including a
  manifest, model tensors, and training state, with atomic replacement and a
  compatibility check against an expected model.

`rule.md` IN1/IN2/IN3 requires that all external bytes be treated as untrusted:
every length, offset, dimension, and element count must be validated with
checked arithmetic before any allocation or access derived from it, and
truncated or malformed input must produce a typed error, never a panic. `A2`
requires an acyclic dependency graph; `A5` keeps these crates out of the
kernel. `OW1`/`ER4` require that resource acquisition and release be
scope-bound and that failure release resources in reverse order.

`docs/crate-boundaries.md` sections 5.10, 5.11, and 7.4 assign the two
responsibilities to separate crates, with `checkpoint` permitted to depend on
`format` but not the reverse.

## Decision

Create two crates, `format` and `checkpoint`, with a one-way dependency
`checkpoint -> format`. `format` depends only on `lib` (resource limits) and
`tensor` (dtype/shape taxonomy); `checkpoint` adds `format`. No reverse edge and
no cycle is introduced.

```text
format     -> lib, tensor
checkpoint -> lib?, tensor, format        (checkpoint does not import lib directly)
```

### DS1 — Boundaries, responsibilities, and public interfaces

**`format` owns** parsing and structural validation of external artifacts. It
MUST NOT execute models, manage request state, allocate device buffers, or
depend on any application crate.

| Module (file) | Exported interface | Width |
| --- | --- | --- |
| `configs.rs` | `modelconfig` (+ `parse`, 14 getters); Llama-family `config.json` | 1 type |
| `metadata.rs` | `tensormetadata`, `dtypetoken`, `tokendtype` | 3 |
| `safetensor.rs` | `safetensorfile`, `HEADERLENGTHSIZE` | 2 |
| `validation.rs` | `checkoverlaps` | 1 |
| `error.rs` | `formaterror` (16 variants) | 1 type |

**`checkpoint` owns** the checkpoint artifact: its `manifest`, the
`checkpoint` container with load/save, best-effort atomic persistence, and
compatibility validation. It MUST NOT own the training loop, optimizer update
rules, or the inference scheduler.

| Module (file) | Exported interface | Width |
| --- | --- | --- |
| `manifests.rs` | `manifest`, `manifestentry`, `checkpoint` | 3 |
| `training.rs` | `trainingstate` | 1 |
| `atomicwrite.rs` | `writeatomic` | 1 |
| `compatibility.rs` | `tensorspec`, `checkcompat`, `checkversion` | 3 |
| `save.rs` / `load.rs` | `savecheckpoint` / `loadcheckpoint` | 1 each |
| `error.rs` | `checkpointerror` (15 variants) + `isrecoverable`/`isincompatibility` | 1 type |
| `lib.rs` | `CHECKPOINTVERSION` | 1 const |

Interface width is the count of distinct exported top-level symbols per
module; inherent `pub fn` accessors are part of their owning type. Every module
is well under the `A4` upper bound of 20, so each module's contract fits in one
review pass.

### Module file names deliberately differ from the roadmap

The roadmap names modules after the public types they export
(`modelconfig.rs`, `tensormetadata.rs`, `safetensors.rs`, `manifest.rs`,
`trainingstate.rs`). Adopting those names literally is impossible under the
repository's no-separator convention: a module and the identically named
lowercased type it re-exports share one namespace, so `mod manifest` plus
`pub use manifest::manifest;` fails with `E0255`. This is the same
translation recorded in `workspacelayout.md` under `rule.md` A8. The crates use
collision-free file names (`configs`, `metadata`, `safetensor`, `manifests`,
`training`) and re-export the intended public type names from `lib.rs`.

### Artifact layout

A safetensors container is an 8-byte little-endian header length, a JSON header
of per-tensor `{dtype, shape, data_offsets}` entries, and the contiguous data
section. A checkpoint (our own format, version `CHECKPOINTVERSION = 1`) is the
8-byte magic `LLMCKPT1`, a little-endian `u64` manifest length, the UTF-8 JSON
manifest, and the data section. Both layouts are self-describing.

### DS2 — Invariants and error paths

**Algorithm-level invariants**

* Every tensor's byte range `[begin, end)` is contained in the data section
  (`end <= datalength`) and `end - begin` equals `element_count * dtype_width`.
* No two tensor byte ranges overlap; each name is unique.
* `element_count = product(shape)` is computed with checked multiplication and
  must fit in `usize` and within the configured element budget.
* The safetensors header is valid UTF-8 JSON and every declared dtype is in the
  supported storage taxonomy; unsupported dtypes are rejected explicitly.

**Implementation-level invariants**

* All length/offset/dimension arithmetic uses `checked_*` operations and the
  outcome is re-validated before slicing; no `as` truncation is used to widen a
  validated value into an index.
* `checkpointerror` keeps three categories distinct: recoverable I/O, structural
  incompatibility, and malformed artifact. `isrecoverable()` is true only for
  `io`; `isincompatibility()` is true only for `unsupportedversion`,
  `missingtensor`, `dtypemismatch`, `shapemismatch`.
* A failed save never publishes a partial artifact: writes go to a
  same-directory staging file and are renamed onto the target only after the
  full content is flushed.

**Error branches and return semantics**

* `formaterror` variants: `truncated`, `headertoolarge`, `invalidjson`,
  `invalidentry`, `unsupporteddtype`, `emptyshape`, `shapeoverflow`,
  `elementlimitexceeded`, `offsetsoutofrange`, `bytelengthmismatch`,
  `overlappingtensors`, `invalidmetadata`, `missingfield`, `invalidvalue`,
  `unsupportedmodeltype`, `unsupportedtorchdtype`.
* `checkpointerror` variants: `io`, `badmagic`, `unsupportedversion`,
  `truncated`, `invalidmanifestjson`, `invalidmanifest`, `shapeoverflow`,
  `lengthmismatch`, `dtypeunsupported`, `duplicatetensor`, `offsetoutofrange`,
  `overlappingtensors`, `missingtensor`, `dtypemismatch`, `shapemismatch`,
  `format(#[from] formaterror)`. `format` errors are propagated with their
  context preserved rather than flattened.

**Resource rollback rule**

All buffers are owned `Vec`/`String` values released by scope; the staging file
is removed on any failure path before returning. No manual release call is
exposed, matching `OW1`/`NM3`.

### DS5 — Alternatives considered

* **One combined persistence crate.** Rejected: `docs/crate-boundaries.md`
  5.10/5.11/7.4 separate external-format parsing from checkpoint ownership, and
  a single crate would give the parsing code a dependency edge onto the
  training-state and atomic-write concerns it does not need (`A1`, `A5`).
* **Adopt the roadmap's module filenames literally.** Rejected: infeasible
  under the no-separator convention (`E0255`), as above.
* **Keep crate-local integration tests while the plan points at a root
  `tests/checkpoint_roundtrip.rs`.** The task's workspace mechanics require
  tests to live inside the owning crate under `crates/<name>/tests/*.rs`
  during this concurrent multi-crate build; root `tests/` is owned by the
  workspace-layout work and is out of scope for this change. Superseded: the
  later consolidation pass moved the suite to the root `tests/roundtrip.rs` (see
  `testconsolidation.md`).
* **Write checkpoints by truncating the target in place.** Rejected: an
  interrupted write leaves a truncated file that could be mistaken for a valid
  checkpoint; the staging-file + rename approach guarantees the target is
  either the old artifact or the complete new one.
* **Classify every failure as a single `Error` enum without accessors.**
  Rejected: callers must distinguish a retryable I/O failure from a genuine
  model incompatibility without matching on variant names; the
  `isrecoverable`/`isincompatibility` accessors encode that contract.
* **Use `unsafe` or unvalidated slicing for the hot parse path.** Rejected:
  the workspace forbids `unsafe` (`OW5`), and untrusted-input correctness
  (`IN1`/`IN2`) outweighs any micro-optimization.

## Consequences

* `format` and `checkpoint` are dependency-safe leaves above `lib`/`tensor`;
  they add no reverse edge and no cycle.
* Both crates are `#![forbid(unsafe_code)]`; all runtime paths use `Result` and
  no `panic`/`unwrap`/`expect`/`todo`/`unimplemented` appears in production
  code (enforced by the workspace `clippy` lints, re-verified here).
* A checkpoint written by `writeatomic`/`savecheckpoint` is never observed in a
  partially written state, and a failed save leaves the previous checkpoint
  intact.
* The reported limitations: checkpoints are single-file, in-memory, and
  unencrypted; `f16`/`bf16` are accepted as storage dtypes but no backend
  execution is claimed; compatibility is checked against an explicit
  `tensorspec` list rather than by inspecting a model object.

## Validation

Executed on 2026-10-10 from `/Users/fei/Doubao/llm`:

* `cargo fmt --all` — clean (exit 0).
* `cargo check -p format --all-targets` — pass.
* `cargo clippy -p format --all-targets -- -D warnings` — pass.
* `cargo test -p format` — 43 lib tests + 5 integration tests pass, 0 failed.
  The integration suite `tests/artifact.rs` covers valid parse, truncated
  header, unsupported dtype, overlapping ranges, and malformed `config.json`.
* `cargo check -p checkpoint --all-targets` — pass.
* `cargo clippy -p checkpoint --all-targets -- -D warnings` — pass.
* `cargo test -p checkpoint` — 27 lib tests + 7 integration tests pass,
  0 failed. `tests/roundtrip.rs` covers the save/load round-trip, truncated
  input, corrupt magic, a missing file reported as recoverable I/O,
  compatibility match/mismatch, atomic replacement without partial artifacts,
  and that a failed save leaves the previous checkpoint intact.
* `RUSTDOCFLAGS='-D warnings' cargo doc -p format --no-deps` — pass.
* `RUSTDOCFLAGS='-D warnings' cargo doc -p checkpoint --no-deps` — pass.

## Revisit Conditions

Reconsider when checkpoint artifacts gain multi-file sharding, memory-mapped
loading, or a binary manifest; when a second external format (for example GGUF)
is added and the shared validation should move into a lower crate; or when
checkpoints must be verified against a live model object rather than an explicit
`tensorspec` list.
