# ADR: Workspace Layout Alignment with `m.md`

Status: Accepted
Scope: Repository directory layout — the root package, `tests/`, `benches/`, `crates/bench`, and `crates/testkit`
Date: 2026-10-10

## Context

`m.md` states the target repository layout. The actual layout differed in the
places the user called out:

* `m.md` places cross-crate integration tests in a repository-root `tests/`.
* `m.md` places benchmarks in a repository-root `benches/`.
* `m.md` places documentation in `docs/`; the actual directory was `doc/`.
* `m.md` declares `crates/bench/` and `crates/testkit/`; neither existed.

An earlier revision of this ADR concluded that root `tests/` and `benches/`
**cannot** be Cargo targets, because the root `Cargo.toml` is a virtual
workspace manifest with no `[package]`, and Cargo resolves `tests/` and
`benches/` relative to a package root. That conclusion was wrong.

## Corrected Finding

A manifest may be **both** a package and a workspace root. A package with no
`src/` — only `tests/` and/or `benches/` — is a valid Cargo package: `cargo
build` succeeds with nothing to build, `cargo test` runs the root integration
tests, and `cargo bench` builds the root benchmarks. Verified with a throwaway
workspace (package + `[workspace] members`, no library or binary target).
Therefore the repository root can satisfy `m.md` literally.

## Decision

Make the repository root a package and place all integration tests and
benchmarks there, matching `m.md`.

1. **The root `Cargo.toml` is a package *and* a workspace root.** It defines a
   package `llm` with no library or binary target and no runtime dependencies.
   Its only dependencies are dev-dependencies used by the root `tests/` and
   `benches/` targets. This is the minimal mechanism that makes root `tests/`
   and `benches/` real Cargo targets.
2. **All integration tests live in the repository-root `tests/`**:
   `apicontract.rs`, `cpucontract.rs`, `tensorcontract.rs`, and `crosscrate.rs`.
   These were moved out of `crates/api/tests/`, `crates/cpu/tests/`,
   `crates/tensor/tests/`, and `crates/testkit/tests/`. Inline unit tests stay
   in their modules (`#[cfg(test)]`), as before.
3. **All criterion benchmarks live in the repository-root `benches/`**:
   `cpuops.rs` (executor-level), `kernels.rs` (raw kernels), and `metadata.rs`
   (tensor metadata). The `criterion` dependency is declared once, as a root
   dev-dependency; it is configured, not removed.
4. **`crates/bench` owns the shared benchmark workloads.** The root benchmark
   harnesses consume `bench`'s deterministic workload constructors, so the
   measurement harness stays thin and the workloads stay reusable. `bench`
   is a library (`src/lib.rs`) rather than the `src/main.rs` shown in `m.md`:
   a Cargo binary cannot be imported by the root benchmark targets, so a
   library is the mechanism that lets `bench` own the workloads while the
   harnesses stay at the root.
5. **`crates/testkit` holds only shared test infrastructure** (`src/lib.rs` plus
   `fixtures`, `assertions`, `reference`, `tinymodel`), matching `m.md`. It is
   a dependency leaf and MUST NOT be a production dependency.
6. **Documentation** moved from `doc/` to `docs/`, matching `m.md`.
7. **Root files required by `m.md`** were added: `README.md`, `LICENSE`
   (with `LICENSE-APACHE` and `LICENSE-MIT`), and `clippy.toml`.

### Module file names deliberately differ from `m.md`

`m.md` lists crate module files such as `tensor/src/shape.rs`,
`tensor/src/tensor.rs`, `tensor/src/view.rs`, `tensor/src/layout.rs`,
`tensor/src/dtype.rs`, `api/src/backend.rs`, `api/src/device.rs`,
`lib/src/limits.rs`, `lib/src/precision.rs`, and `lib/src/seed.rs`. These
filenames **cannot** be adopted under the repository's no-separator identifier
convention: a module and the identically named lowercased type it re-exports
share one namespace, so `mod shape` plus `pub use shape::shape;` fails with
`E0255` (verified). The existing modules therefore use collision-free names
(`shapes`, `tensors`, `views`, `layouts`, `dtypes`, `backends`, `devices`,
`limit`, `prec`, `seeds`), and each crate's `lib.rs` documents this. This is
the repository's translation of `m.md`'s filenames onto a Rust mechanism
(`rule.md` A8); adopting `m.md`'s names literally would require dropping the
crate-root re-exports and changing every public path.

The dependency graph remains acyclic; both helper crates are dependency leaves:

```text
lib
tensor -> lib
api -> lib, tensor
cpu -> lib, tensor, api
bench -> lib, tensor, cpu
testkit -> tensor            (root tests also use lib, api, cpu, testkit, bench)
llm (root package) -> (dev-dependencies only)
```

## Consequences

* `cargo test --workspace` and `cargo bench --workspace` build and run the
  cross-crate suite and every benchmark from the repository root, matching
  `m.md`.
* The root package has no library or binary target, so `cargo build
  --workspace` compiles the member crates and skips the root package. The root
  package is never published (`publish = false`).
* `crates/*` no longer contain `tests/` or `benches/` directories.
* `crates/cpu` and `crates/tensor` no longer declare `criterion`; it is
  declared once at the root. Their `proptest` dev-dependency remains because
  their inline unit tests use it.
* The reference implementations in `testkit::reference` deliberately duplicate
  the scalar math of the CPU kernels so they can detect errors in them; this is
  the one place where duplication is the point.

## Alternatives Considered

* **Keep the virtual workspace and map root `tests/`/`benches/` onto
  `crates/testkit/tests` and `crates/bench/benches/`.** Rejected: it does not
  match `m.md`, which the user requires; the root package makes the literal
  layout work.
* **Turn the root into a package with a facade `src/lib.rs`.** Rejected: `m.md`
  shows no root `src/`, and a facade library would add an artificial crate with
  no production responsibility, contradicting the minimal-kernel preference in
  `AGENTS.md` section 3.3. A package with only test/bench targets needs no
  `src/`.
* **Adopt `m.md`'s module filenames literally.** Rejected: infeasible under the
  no-separator convention (`E0255`), as documented above.
* **Scatter crate-local `benches/` and `criterion` dev-dependencies across
  `crates/cpu` and `crates/tensor`.** Rejected: `m.md` centralizes benchmarks at
  the root and assigns `bench` the benchmark workloads, so a single root
  `criterion` dependency is the correct configuration.

## Validation

Executed on 2026-10-10:

* `cargo fmt --all -- --check` — clean.
* `cargo check --workspace --all-targets` — pass.
* `cargo clippy --workspace --all-targets -- -D warnings` — pass.
* `cargo test --workspace` — 131 tests pass, 0 failed. This includes the root
  integration suites `tests/apicontract.rs` (3), `tests/cpucontract.rs` (7),
  `tests/tensorcontract.rs` (9), and `tests/crosscrate.rs` (6), plus the crate
  unit tests and `testkit`/`bench` unit tests.
* `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` — pass.
* `cargo bench --workspace --no-run` — pass; `benches/cpuops.rs`,
  `benches/kernels.rs`, and `benches/metadata.rs` are built.

## Update (2026-10-10)

The mechanism defined here was extended to every remaining crate. The root
`tests/` now also holds the `tokens`, `artifact`, `roundtrip`, `context`,
`generation`, `isolation`, `pipeline`, `optim`, `cancellation`, `lossdecrease`,
`trainingresume`, `service`, `server`, `cli`, `autograd`, `nnlayers`, and
`modelforward` suites, and the root `benches/` also holds `autograd`, `nn`,
`models`, and `optim`. `cargo test --workspace` reports 746 passing, 0 failed,
1 ignored. See `testconsolidation.md` for the pass that moved the P4–P12 suites.

## Revisit Conditions

Reconsider when the root package needs a real library target (for example a
facade re-exporting the kernel crates), at which point `src/lib.rs` may be
added; or when end-to-end application tests spanning the application crates
(`cli`, `serve`, `infer`) are introduced and may warrant a dedicated test crate.
