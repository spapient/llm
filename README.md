# llm

A minimal, composable LLM runtime written in Rust, organized as a static-crate
kernel with narrow traits. See [`AGENTS.md`](AGENTS.md) for the engineering
constitution and [`m.md`](m.md) for the target repository layout.

## Status

The workspace implements the foundational, neural-network, model, workflow, and
application layers, plus two native GPU backends.

| Crate | Role | Status |
| --- | --- | --- |
| `lib` | Foundational shared types (identifiers, precision, seeds, limits) | Implemented |
| `tensor` | Tensor metadata, storage, shapes, layouts, and views | Implemented |
| `api` | Backend and kernel execution contracts | Implemented |
| `cpu` | CPU execution backend and numerical kernels | Implemented |
| `metal` | Apple Metal GPU backend (full operator set) — macOS only | Implemented; run on macOS |
| `cuda` | NVIDIA CUDA GPU backend (full operator set) — Linux only | Implemented; see below |
| `autograd` | Define-by-run reverse-mode differentiation | Implemented |
| `nn` | Reusable layers and parameter containers | Implemented |
| `models` | Llama-family causal language model | Implemented |
| `tokenizer` | Vocabulary, byte-pair encoding, special tokens, chat templates | Implemented |
| `data` | Datasets, records, packing, batching, sampling, loaders | Implemented |
| `optim` | Optimizers, clipping, and learning-rate schedules | Implemented |
| `format` | External model format parsing and validation | Implemented |
| `checkpoint` | Model and training-state persistence | Implemented |
| `context` | Execution controls, cancellation, budgets, workers, metrics | Implemented |
| `infer` | Generation, request lifecycle, sampling, stop conditions, KV cache | Implemented |
| `train` | Training orchestration, evaluation, and resume | Implemented |
| `cli` | Command-line application and exit-code mapping | Implemented |
| `serve` | std-only HTTP application with admission control | Implemented |
| `testkit` | Test fixtures, assertions, and reference implementations | Implemented (non-production) |
| `bench` | Benchmark workloads and measurement orchestration | Implemented (non-production) |

Known gaps: only `f32` dtypes are implemented. Both GPU backends implement the
whole `compute` contract — `matmul`, `add`, `mul`, `silu`, `rmsnorm`, `softmax`,
`rope`, `embedding`, and a single fused `attention` kernel. The Metal backend
runs these through in-repo MSL compiled at run time by `newLibraryWithSource`;
CUDA runs `matmul` through cuBLAS, `softmax` through cuDNN, and the rest through
in-repo CUDA C++ kernels compiled at run time by NVRTC. The Metal operator set
has been run and checked against a scalar reference on macOS. The CUDA kernel's
runtime numerical check requires an NVIDIA GPU and is therefore reported as
**not executed** on hosts without one — only its compile gate and its
device-free validation cases are currently exercised. No optimization has been
attempted, so no performance claim is made.

## Layout

```text
crates/     workspace crates (21 crates: layers 0-4 plus the two GPU backends)
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

The GPU backends are platform-gated, and neither needs its platform's GPU
toolchain to compile:

- `metal` builds only on macOS. The MSL kernels are compiled at run time through
  `newLibraryWithSource`, so no Xcode or Metal toolchain is involved.
- `cuda` builds only on Linux and resolves the driver, NVRTC, cuBLAS, and cuDNN
  at run time (`cudarc` with `dynamic-loading`), so `cargo check -p cuda` needs
  no CUDA toolkit. Running it needs an NVIDIA driver and GPU.

Both crates are the only carriers of the workspace's restricted `unsafe`
exemption; the exemption, its file whitelist, and its mechanical gate are
documented in [`docs/design.md`](docs/design.md) and
[`docs/engineering.md`](docs/engineering.md).

## Commands

```sh
bash scripts/check.sh   # formatting, check, Clippy (-D warnings), rustdoc (-D warnings)
bash scripts/test.sh    # workspace unit, integration, and doc tests
bash scripts/bench.sh   # build and run the criterion benchmarks
```

## License

Dual-licensed under Apache-2.0 OR MIT. See [`LICENSE`](LICENSE),
[`LICENSE-APACHE`](LICENSE-APACHE), and [`LICENSE-MIT`](LICENSE-MIT).
