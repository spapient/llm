# `llm-dtype`

Element-type floor for the `llm` runtime: one closed set per precision axis, and every identity
fact lives in exactly one table.

| Axis | Type | Answers |
| --- | --- | --- |
| Scalar | `DType` | how is **one element** stored? (`Bool`, `U1`–`U64`, `I1`–`I64`, `F4`, FP8, `F16`, `BF16`, `F32`, `F64`, plus exponent/mantissa/bias) |
| GGML block | `QType` | how is **a block** packed? (25 formats: legacy Q, K-quant, IQ, TQ) |
| Micro-scaled | `MxType` | low-precision element + shared block exponent (OCP MX v1.1 and NVFP4) |
| MX element | `Element` | which E2M1/E2M3/E3M2/E4M3/E5M2/INT8 element sits inside an `MxType`, and is it float or integer? |

```rust
use llm_dtype::{DType, QType};

assert_eq!(DType::F32.code(), 20);
assert_eq!(DType::from_code(20), Some(DType::F32));
assert_eq!(QType::Q4_K.bytes_for(256), 144);
assert_eq!("q4_k".parse::<QType>(), Ok(QType::Q4_K));
```

## Guarantees

- **One table per closed set.** Each row table (`SPECS`, `ELEMENT_SPECS`) is length-anchored by
  `T::ALL.len()`, so a missing row is a compile error, and `repr(u8)` makes "discriminant = row
  index" a type guarantee rather than a test observation.
- **Zero cost.** No dependencies by default, unconditional `no_std`, `forbid(unsafe_code)`. Every
  accessor that can be is a `const fn`; geometry has exactly one implementation (the inherent
  `const fn`), which the `Format`/`BlockFormat` traits forward to generic code. Name parsing is the
  one exception: it delegates to the trait default because stable Rust has no const traits.
- **Float bit layout is identity, not derived.** `DTypeSpec`/`ElementSpec` carry a `FloatLayout`
  column (`exp_bits`, `mantissa_bits`, `exponent_bias`) next to the width, so `F8E4M3 = 1 + 4 + 3`,
  `BF16 = (8, 7, 127)` and the `FNUZ` variants (bias one higher than their IEEE twin) live in the
  same row table a dequantizer already reads — no hard-coded bit fields downstream. The sign bit is
  implicit (always 1), so the full width is the single derivation `FloatLayout::bits()`;
  `FloatLayout::bits() == bits` for every float, and an MX element's layout equals that of its
  scalar `DType` counterpart. `Element::is_float()`/`is_integer()` read the same layout column, so
  the float/integer split is a table fact too — no separate `Class` column to drift.
- **Identity is locked.** Every row is pinned by a literal table in that module's tests: `DType`
  (wire code, name, variant, bits, class, float layout), the MX element set (name, variant, bits,
  scalar counterpart, float layout) and the block layout of `QType`/`MxType` (block length, block
  bytes, scale, family). Every public accessor is asserted against its own column on top of that,
  so a getter reading the neighbouring field cannot hide behind a locked table. `Display`,
  `FromStr` and the optional `serde` representation all read the same name.
- **Facts, not algorithms.** This crate says *what a format is and how many bytes it takes*; it
  never says *how to decode it*. Dequantization stays in `llm-weights` (ADR 0018).

## Optional features

Both are **off by default**, so the default dependency set stays empty:

- `serde` — `Serialize`/`Deserialize` for the four closed sets, wire name = canonical name.
- `bytemuck` — `NoUninit` + `CheckedBitPattern`. `Pod` is deliberately not offered: a `repr(u8)`
  closed set has only a handful of valid byte values, so reads must be checked.

## Status

API-stable enough to be the single source inside this workspace, but not published to crates.io
yet (`publish = false`): publishing implies an MSRV and semver commitment this repository has not
made. `MxType` also has no decoder consumer yet — see `crates/weights` and `docs/adr/0018`.
