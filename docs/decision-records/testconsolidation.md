# Decision Record — Integration-Test and Benchmark Consolidation

Status: Accepted
Date: 2026-10-10
Scope: Repository-root `tests/`, `benches/`, the root `llm` package manifest, and
the per-crate manifests whose tests or benchmarks moved.

---

## Context

`m.md` / `AGENTS.md` and [testing.md](../testing.md) section 6 place cross-crate
integration tests in a repository-root `tests/` and benchmarks in a
repository-root `benches/` (shared workloads in `crates/bench`), while crate
inline unit tests stay in their modules.

`docs/decision-records/workspacelayout.md` established the root `llm` package +
workspace mechanism and moved the Layer 0/1 suites (`apicontract`,
`cpucontract`, `tensorcontract`, `crosscrate`) and the `autograd`, `nn`, and
`models` suites to the root `tests/`, and their benchmarks to the root
`benches/`.

The remaining crates (P4–P12: `tokenizer`, `format`, `checkpoint`, `context`,
`infer`, `data`, `optim`, `train`, `serve`, `cli`) were each built by a
concurrent worker that, to avoid editing the shared root manifest mid-build,
placed its integration tests and the `optim` benchmark crate-locally under
`crates/<name>/tests/` and `crates/optim/benches/`. Those per-crate ADRs record
that placement as a deliberate, stage-scoped deviation and note that a later
consolidation pass owns the root directories.

This record is that pass.

## Decision

1. **Move every crate-local integration suite to the repository-root `tests/`.**
   `crates/{tokenizer,format,checkpoint,context,infer,data,optim,train,serve,cli}/tests/`
   no longer exist. The moved files are `tokens`, `artifact`, `roundtrip`,
   `context`, `generation`, `isolation`, `pipeline`, `optim`, `cancellation`,
   `lossdecrease`, `trainingresume`, `service`, `server`, and `cli`. Files whose
   names contained `_` were renamed to the no-separator form (`loss_decrease.rs`
   → `lossdecrease.rs`, `training_resume.rs` → `trainingresume.rs`,
   `model_forward.rs` → `modelforward.rs`).

2. **Move the `optim` benchmark to the repository-root `benches/`.** The crate
   keeps no `[[bench]]` target and no `criterion`/`checkpoint` development
   dependencies; the `checkpoint` round-trip coverage now lives in the root
   `optim` suite, which the test-only root package can reach directly.

3. **Merge the two fixture modules into one root `tests/common/mod.rs`.** The
   `infer` and `serve` suites both needed the same tiny Llama configuration,
   deterministic weights, a bounded CPU provider, and an independent greedy
   `argmaxlast`; the `serve` suite additionally needed transport helpers. A
   single shared fixture module replaces both `crates/infer/tests/common/` and
   `crates/serve/tests/common/`, avoiding duplicated fixtures and the collision
   of two same-named `common` modules under one root `tests/`.

4. **Wire the root manifest.** The root `llm` package declares dev-dependencies
   on `tokenizer`, `format`, `checkpoint`, `context`, `infer`, `data`, `optim`,
   `train`, `serve`, and `cli` (all test-only) and adds the `optim` `[[bench]]`
   target. The workspace still uses glob membership `members = ["crates/*"]`, so
   no member list changed.

5. **Drop the now-unused crate development dependencies.** `train`'s
   `autograd`/`cpu`, `serve`'s and `infer`'s `cpu`, and `cli`'s `tensor`
   development dependencies were used only by the moved suites; they were
   removed. No production dependency edge changed, so the crate dependency graph
   and its direction are unchanged.

## Superseded deviations

The following stage-scoped notes in earlier ADRs are superseded by this record;
the underlying decisions they describe (thin adapters, no backward path, no
performance claims) still stand:

* `cli.md` — "Integration tests are crate-local under `crates/cli/tests/`."
* `inference.md` — "Integration tests are crate-local."
* `serve.md` — "Integration tests are crate-local."
* `persistence.md` — "Keep crate-local integration tests...".
* `training.md` — "the P10 instruction scopes tests crate-local..."
* `optim.md` — "The benchmark is crate-local...".
* `data.md`, `optim.md`, `training.md` — the "`cargo test -p <crate>` runs ...
  integration tests" lines now describe suites that live at the root.

## Rejected alternatives

* **Leave the suites crate-local and amend the layout rule.** Rejected: it
  contradicts `testing.md` section 6 and `AGENTS.md`, and would leave the
  workspace inconsistent between the Layer 0/1 + `autograd`/`nn`/`models` suites
  (root) and the rest (crate-local).
* **Keep two `tests/common/mod.rs` fixtures under namespaced submodules.**
  Rejected in favor of one merged module: the `serve` suite references the
  fixtures by bare `common::…` paths in many places, so one shared module keeps
  the moved files unchanged and avoids duplicating the model fixtures.
* **Move suites incrementally per crate.** Rejected: the root manifest must list
  the dev-dependency for any suite it compiles, so a single coordinated pass
  avoids intermediate states where the workspace does not build.

## Consequences

* All cross-crate integration suites and all benchmarks now live where
  `testing.md` section 6 and `m.md` require, and every artifact name follows the
  no-separator convention.
* The test-only root package now depends on the application crates (`cli`,
  `serve`); this is a dev-dependency only and does not create a production edge,
  so `cli`/`serve` remain non-dependencies of production compute crates.
* Crate-local coverage is now inline unit tests only, which keeps each crate's
  own build fast and its public contract tested from outside at the root.

## Validation

Executed on 2026-10-10 against the whole workspace (macOS 27.0.1,
`aarch64-apple-darwin`, rustc 1.99.0, edition 2024):

```sh
cargo fmt --all -- --check                                   # exit 0
cargo check --workspace --all-targets                        # exit 0
cargo clippy --workspace --all-targets -- -D warnings        # exit 0, no warnings
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps   # exit 0
cargo test --workspace                                       # 746 passed, 0 failed, 1 ignored
cargo bench --workspace --no-run                             # exit 0 (7 benchmark targets)
```

`scripts/check.sh` and `scripts/test.sh` complete successfully, and
`cargo bench --workspace --no-run` builds `cpuops`, `kernels`, `metadata`,
`autograd`, `nn`, `models`, and `optim`. No performance claim is made; the
benchmark targets establish measurability only (PF1/VF5).
