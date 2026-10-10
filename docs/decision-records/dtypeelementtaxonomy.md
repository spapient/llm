# ADR: Storage `dtype` vs. Compute `precision`, and `element` Coverage

Status: Accepted
Scope: Layer 0 `tensor` (`dtype`, `element`) and `lib` (`precision`)
Date: 2026-10-10

## Context

Layer 0 froze three overlapping-but-distinct type sets:

* `lib::precision` — `{f32, f16, bf16}`.
* `tensor::dtype` — `{f32, f16, bf16, i8, u8, i32, i64}`.
* `tensor::element` — implemented for `f32`, `i8`, `u8`, `i32`, `i64`.

`element::DTYPE` returns a `dtype`, so the two sets must agree. `f64` was absent
from all three. The previous justification in `element`'s documentation was
circular: it stated that `element` was not implemented for `f64` because there
was no `dtype::f64` variant, while `dtype` had no documented criterion for
omitting `f64` at all.

The exclusion was inconsistent with the existing taxonomy. `dtype` already
contains `i8`, `u8`, `i32`, and `i64`, none of which any backend supports and
none of which has a `precision` counterpart. `i64` is a native 8-byte primitive
accepted as a tensor storage type; `f64` is the same kind of primitive and was
excluded only by the circular argument above.

## Decision

1. **`dtype` is the complete storage taxonomy; `precision` is the compute
   taxonomy.** They are distinct axes. `From<precision> for dtype` is the
   injection that embeds compute formats into storage formats; it is not
   surjective. Integer storage and `f64` are storage-only.

2. **`dtype::f64` is added as a storage-only type.** It has no `precision`
   counterpart and no backend declares `f64` execution. It exists so that
   tensors can hold double-precision reference data, golden values, and
   high-precision accumulation buffers.

3. **`element` is implemented for every primitive with a native Rust
   representation**: `f32`, `f64`, `i8`, `u8`, `i32`, `i64`. `f16` and `bf16`
   remain excluded because Rust has no native half-precision type; supporting
   them requires an owned wrapper type in a later layer. This is the *real*
   exclusion criterion and replaces the circular one.

4. **The width invariant is enforced mechanically.** Each `element`
   implementation MUST satisfy `DTYPE.sizeinbytes() == size_of::<Self>()`,
   asserted in `element`'s unit tests across all six types. A future `dtype`
   edit that changes a width without updating `element` fails the test.

Adding `dtype::f64` does not widen backend support: `cpu` still declares exactly
`[dtype::f32]`. The dependency graph is unchanged (`tensor -> lib`); no edge,
`unsafe`, or dynamic dispatch is introduced.

## Consequences

- `element` is now a total map from native primitives to storage types, and
  `dtype` is documented by its actual invariant rather than by an omission.
- `f64` tensors are creatable and inspectable through `tensor`, `tensorview`,
  and `shape`/`layout` with no code change, because those types are generic over
  `T: element`. They are not executable by any backend until one declares them.
- Two statements that described the old state are corrected:
  `docs/decision-records/layer1cpu.md` §Context now lists the six implemented
  types, and `lib::precision`'s documentation now names `f64` as storage-only.

## Alternatives Considered

- **Keep `f64` out and document it as unsupported.** Rejected: it leaves the
  circular justification in place and is inconsistent with `i64`, which is an
  equally unsupported native primitive that `dtype` already carries.
- **Add `precision::f64` as well.** Rejected: `precision` names compute formats,
  and no LLM computation is performed at `f64`. Adding it would imply a compute
  capability that does not exist and would make `From<precision>`'s image
  include a format no backend could accept.
- **Remove `const DTYPE` from `element` and carry the dtype separately at the
  call site.** Rejected: it would let a tensor's storage type disagree with its
  element type, remove the compile-time binding between `T` and its `dtype`, and
  widen `tensor`'s constructor surface for no benefit.
- **Implement `element` for a half-precision wrapper now.** Rejected: it is a
  new type with its own rounding, arithmetic, and conversion contract; it is out
  of scope for this change and belongs to a later additive stage.

## Validation

Executed on 2026-10-10:

- `cargo fmt --all` — clean.
- `cargo check --workspace --all-targets` — pass.
- `cargo clippy --workspace --all-targets -- -D warnings` — pass.
- `cargo test --workspace` — 106 tests pass, 0 failed. `tensor` unit tests
  increased from 28 to 31 (added the width invariant, the width-preservation
  check for `From<precision>`, and the `f64`-is-not-a-precision check); the
  `tensor` contract suite increased from 8 to 9 (added an `f64` round-trip
  through a view).
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` — pass.

## Revisit Conditions

Reconsider when: a backend declares `f64` execution (then `precision::f64` may
become meaningful); a native or wrapper half-precision type is added (then
`element` coverage and the width invariant extend to it); or a validated model
requires a storage type not represented in `dtype`.
