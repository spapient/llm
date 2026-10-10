# Decision Record — Black-Box, Contract-First Testing

Status: Accepted
Date: 2026-10-10
Scope: Test placement, test visibility, dead-code policy, and production
diagnostic instrumentation across every workspace crate.

---

## Context

The workspace currently places 599 unit tests in 121 `#[cfg(test)]` modules
inside `crates/*/src`, and only 146 tests at the repository-root `tests/`. The
root suites already exercise crates through their public interfaces; the inline
modules are the exception.

A contract-first mandate for long-lived infrastructure requires the opposite
default: fix the public contract first, test only through it, and treat the
public interface as the single testable surface. Two concrete costs motivate it:

1. **Test maintenance follows the interface.** When a private helper is renamed,
   every inline test that reaches into it must change. A test bound to the public
   contract survives internal refactors unchanged.
2. **Deep-white-box tests hide interface gaps.** A test that constructs illegal
   internal state via a private path proves nothing about whether a caller can
   reach that state. If a boundary branch is not reachable through the public
   interface, the interface cannot defend against it.

This record makes black-box testing the normative default, replaces the
inline-unit-test rule, and defines the diagnostic seam that keeps black-box
failures observable.

## Decision

1. **Production source contains no tests.** `crates/*/src` MUST NOT contain
   `#[cfg(test)]` modules, `#[test]` functions, or `#[cfg(test)]`-gated items.
   All unit, integration, regression, and benchmark tests live under the
   repository-root `tests/` (benchmarks under `benches/`, per the Cargo
   convention that `cargo bench` depends on).

2. **Tests use the public surface only.** A test MUST reach the crate through
   its `pub` items. It MUST NOT call private functions, read private fields, or
   name private types. `testkit` is the only permitted shared test dependency and
   remains a non-production crate.

3. **The public interface is not widened for tests.** Exposing an internal type
   or adding a `pub` item whose only consumer is a test is forbidden
   (`public-api.md` section 4.1). A crate that needs a test-only hook designs a
   legitimate public entry point with its own contract, or does not test that
   path.

4. **Unreachable private logic is dead code.** If internal logic cannot be
   reached from the public interface, it is deleted rather than tested. Logic
   that *is* reachable but not *directly constructible* is not dead: its
   behavior is asserted through the public entry point that reaches it.

5. **Boundary paths are designed in, not tested around.** When a contract has an
   extreme or exceptional branch, the interface MUST expose a parameter or entry
   point that lets a caller reach it, so the branch is triggerable through the
   public interface. Deferred alternatives are recorded in section 11.

6. **Tests verify the pre-declared contract; coverage is not a goal.** A test
   MUST assert a documented input/output or error contract. Tests written only to
   raise a coverage number, or that assert implementation details rather than
   contract, are defects.

7. **Production carries a toggleable diagnostic seam.** Production code MAY emit
   lightweight `tracing` events and spans at decision points. This is a
   production diagnostic capability, not test code. It is disabled by default and
   enabled at runtime without recompiling (section "Diagnostic seam").

## Migration strategy

The migration is staged, and the gate `scripts/checksrctests.sh` enforces a
ratchet:

* Every file that still contains inline tests is listed in
  `scripts/srctestbaseline.txt`.
* A file with inline tests that is **not** in the baseline fails the gate (no new
  inline tests).
* A baseline entry with **no** remaining inline tests fails the gate (a migrated
  file must be removed from the baseline).

The ratchet only tightens. When the baseline is empty, the prohibition is
absolute. The pilot crate `lib` is migrated in this change; the remaining crates
follow one crate per commit.

## Diagnostic seam

`tracing` and `tracing-subscriber` are already declared in the workspace
dependencies. The seam is:

* Production decision points emit `tracing::trace!`/`debug!` events; request- or
  operation-scoped work uses `#[tracing::instrument]` spans. Genuine anomalies
  use `warn`/`error`.
* Hot paths emit at `trace`/`debug` only, so the default filter discards them
  with minimal cost (`PF2`).
* The toggle is the subscriber: with no subscriber installed the events are
  dropped; an application (`cli`, `serve`) enables them by installing a
  `tracing_subscriber` layer with an `EnvFilter` (`RUST_LOG`). No recompilation
  and no test-only code are involved.
* The seam makes a black-box failure localizable: the test asserts the public
  contract, and the trace reports the internal branch that produced the result.

## Superseded rules

This record supersedes the inline-unit-test rule in the following places, which
MUST be updated in the same change:

* `docs/testing.md` section 5 (unit-test location) and section 6.1.
* `docs/public-api.md` section 4.1, final sentence.
* `AGENTS.md` bullet on test organization.
* `docs/decision-records/testconsolidation.md`, consequence "Crate-local coverage
  is now inline unit tests only."

The rest of `testconsolidation.md` (integration tests and benchmarks at the
repository root) still stands and is reinforced here.

## Rejected alternatives

* **Keep inline tests for private helpers.** Rejected: it preserves the coupling
  the mandate removes. A private helper's behavior is asserted through the public
  entry point that reaches it, or the helper is dead.
* **Expose helpers as `pub` to relocate their tests.** Rejected by decision 3: it
  widens the contract for test convenience and creates a compatibility
  commitment with no real consumer.
* **A `#[cfg(test)]` "seam" accessor gated per test.** Rejected: it reintroduces
  test-only code in production sources and depends on `cfg(test)`, which is the
  thing being removed.
* **Runtime tracing replaced by compile-time `feature = "diagnostics"`.**
  Rejected in favor of a subscriber toggle: a feature flag requires a rebuild to
  enable, while the subscriber can be switched on in a running deployment.

## Consequences

* The workspace has one test surface: the public contract. Internal refactors no
  longer break tests.
* Relocating 599 tests requires a genuine public entry point for each asserted
  behavior. Where none exists, either the behavior is already covered indirectly
  through another public path, the interface gains a legitimate parameterized
  entry (decision 5), or the logic is deleted as dead (decision 4).
* `testkit` grows as the shared black-box fixture and reference provider.
* Production code gains a small, disabled-by-default diagnostic dependency in the
  crates that need observability.

## Validation

Executed on 2026-10-10 against the whole workspace (macOS 27.0.1,
`aarch64-apple-darwin`, rustc 1.99.0, edition 2024):

```sh
bash scripts/checksrctests.sh      # exit 0; 0 files pending migration
bash scripts/check.sh              # exit 0 (fmt, clippy, naming, ratchet, thresholds, doc)
cargo test --workspace --locked    # all suites pass, 0 failed
cargo test -p llm --test diagnostics  # 2 passed (toggleable diagnostic seam)
```

All nineteen crates are migrated to zero `#[cfg(test)]` in
`crates/<crate>/src` — `lib`, `api`, `context`, `data`, `cpu`, `testkit`,
`tensor`, `checkpoint`, `bench`, `autograd`, `optim`, `nn`, `models`, `format`,
`infer`, `cli`, `serve`, `train`, `tokenizer`. `scripts/srctestbaseline.txt` is
empty, so the ratchet prohibition is now absolute. The move is driven by
`scripts/migrateinline.py` (with the `scripts/_rustscan.py` lexical brace
scanner) and `scripts/finalizecrate.py`, which strip the source only after the
relocated suite compiles.

### Diagnostic seam

The seam (decision 7) is wired in `infer`, `train`, `checkpoint`, `serve`, and
`cli`: workflow decision points emit `debug!`/`trace!`, and the applications
(`cli::runcli`, `httpserver::serve`/`serveone`) install a `tracing_subscriber`
`EnvFilter` read from `RUST_LOG` (defaulting to `warn`). `tests/diagnostics.rs`
proves the toggle without a rebuild: the same public `loadcheckpoint` call emits
the `"checkpoint loaded"` event under a `DEBUG` subscriber and suppresses it
under a `WARN` subscriber.
