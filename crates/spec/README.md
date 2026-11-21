# `llm-spec`

Operator contract: everything that answers *what an operator is*, separated from *how it is
computed*. Four crates form one partial order — `op-spec` → `op-kernel` → `op-neural` → `op`, with
the arrow pointing at the user, so no layer ever reaches back into a later one.

| Layer | Crate | Owns |
| --- | --- | --- |
| Contract | `llm-spec` (this crate) | signature ABI, layout/schedule closed sets, template vocabulary, cost expressions, semantic model, deterministic samples |
| Dense kernels | `llm-gemm` | gemv/gemm/matmul, quantization, elementwise, unary, reduce, norm, activation |
| Structured kernels | `llm-conv` | convolution, deconvolution, pooling/norm families, attention, scan |
| Registry | `llm-op` | `ops/*.toml` declarations, `REGISTRY`, `builtin()`, `model(name)` |

```rust
use llm_spec::{einsum, model, tpl};

// Names are compared byte-wise because `str` equality is not const on this toolchain.
assert!(llm_spec::same_bytes("gemv_i4", "gemv_i4"));
// One closed set for scalar types; the kernels project from it rather than redeclaring it.
assert_eq!(model::DType::parse_name("f32").map(model::DType::name), Some("f32"));
```

## Guarantees

- **One source per fact.** The entry signature (`sig`) is the only ABI fact; `flops`/`bytes` come
  from one declared expression per operator, and `model::Cost` re-exports the result instead of
  recomputing it. Layouts and schedules are closed sets here, consumed by the registry and `cutile`.
- **No reverse edges.** Nothing in this crate mentions a kernel module: the whole crate is testable
  with `cargo test -p llm-spec` and cannot regress because a device kernel changed.
- **No `unsafe`.** `deny(unsafe_code)` is unconditional; every operation here is pure parsing,
  arithmetic on `expr` trees, or table lookup.
- **Vocabulary is declared once.** `Bits`, `Prim`, `DType`, `Sample` and the schedule tokens all
  implement `prof::atom::Closed` (or read its tables), so `ALL`/`as_str`/`parse` cannot drift from
  the row tables.

## Status

`publish = false`, like the rest of the workspace: the API is stable inside this workspace but no
MSRV or semver commitment exists yet.
