# llm

A minimal, composable LLM runtime written in Rust, organized as a static-crate
kernel with narrow traits. See [`AGENTS.md`](AGENTS.md) for the engineering
constitution and [`m.md`](m.md) for the target repository layout.

## Status

The workspace implements the foundational layers only. Crates listed in `m.md`
that are not listed below are not yet implemented.

| Crate | Role | Status |
| --- | --- | --- |
| `lib` | Foundational shared types (identifiers, precision, seeds, limits) | Implemented |
| `tensor` | Tensor metadata, storage, shapes, layouts, and views | Implemented |
| `api` | Backend and kernel execution contracts | Implemented |
| `cpu` | CPU execution backend and unary f32 kernels | Implemented (unary only) |
| `testkit` | Test fixtures, assertions, and reference implementations | Implemented (non-production) |
| `bench` | Benchmark workloads and measurement orchestration | Implemented (non-production) |

The binary, weighted, and model-level operators (`matmul`, `attention`,
`embedding`, weighted normalization), `f16`/`bf16`, training, inference, and
the application crates are not implemented.

## Layout

```text
crates/     workspace crates (lib, tensor, api, cpu, testkit, bench)
tests/      cross-crate integration tests
benches/    criterion benchmark harnesses
docs/       architecture, standards, and conformance records
scripts/    check, test, and benchmark entry points
```

The repository root is a Cargo package without a library or binary target; it
exists so that the root `tests/` and `benches/` directories are real Cargo
targets. See [`docs/architecture.md`](docs/architecture.md).

## Requirements

- Rust 1.99.0 (see `rust-toolchain.toml`)

## Commands

```sh
bash scripts/check.sh   # formatting, check, Clippy (-D warnings), rustdoc (-D warnings)
bash scripts/test.sh    # workspace unit, integration, and doc tests
bash scripts/bench.sh   # build and run the criterion benchmarks
```

## License

Dual-licensed under Apache-2.0 OR MIT. See [`LICENSE`](LICENSE),
[`LICENSE-APACHE`](LICENSE-APACHE), and [`LICENSE-MIT`](LICENSE-MIT).
