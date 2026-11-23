# `llm-nn`

Layer definitions and the reference arithmetic behind them: every layer is a **declaration**
(`ContractSpec`, bounds, contract) with no device kernel, and every mechanism is a pure
computation with a testable invariant.

## Layers

The public surface below mirrors `src/lib.rs` exports; kernel entry points are under
`nn::kernel::{gemv, gemm, matmul, ...}`.

| Group | Types |
| --- | --- |
| Core | `Linear`, `MatMul`, `LmHead`, `Attention`, `AttentionBlock`, `DecodeAttention`, `LatentAttention`, `Embedding`, `Stack`, `Floor`, `Recurrent` |
| Pointwise / activation | `RmsNorm`, `Rope`, `Softmax`, `Gelu`, `Silu`, `Relu`, `Sigmoid`, `Tanh`, `HardSigmoid`, `LeakyRelu`, `Exp`, `Log`, `Scale`, `Add`, `Mul`, `Sub`, `Div`, `Max`, `Min`, `Scan` |
| Spatial / conv | `Conv1d`, `Conv2d`, `ConvTranspose1d`, `ConvTranspose2d`, `AvgPool2d`, `AdaptiveAvgPool2d`, `MaxPool1d`, `MaxPool2d`, `LayerNorm`, `BatchNorm2d`, `GroupNorm`, `InstanceNorm`, `Interpolate` |
| Conditional memory (`engram`) | `Bypass`, `MemoryBlock`, `Table`, `allocate`, `next_prime`, `phi`, `slot_of`, `slots` |

## Responsibilities

- **Owns**: `Contract`, `ContractSpec`, `Bounds`, the layers above, reference arithmetic and the
  operator registry, `WeightFormat`, `Stack`.
- **Does not own**: operator contract vocabulary, archives and templates (`prof`); kernel
  mechanism (`cutile`); weight bytes and parsing (`weights`).
- **Depends on** `weights`, `tensor`, `prof`, `annotation`, `diagnostic`; **depended on by** `train`.

A layer definition produces one `ContractSpec`; its operator name must already exist in
`prof::catalogue::all()`. A wrong name fails here at construction, not at emit time.

## Conditional memory (`engram`)

The only mechanism family kept in `nn`: hash N-grams, `O(1)` slot lookup, `ρ` sparse allocation
(`Table`, `Bypass`, `slot_of`). It stays in `nn` because `MemoryBlock` has a `Contract`
implementation; its typed accessor cannot make `nn` depend back on `algorithm`.

## Architecture variants

The other reference semantics for architecture variants — CSA / HCA attention, three-channel
gating, DeepSeek MLA, byte-latent, KDA / MoE, mHC, Mamba, Spiking — live in the `algorithm`
crate, not here. Weight encoding is `weights`; the device kernel mechanism is `cutile`.

## Guarantees

- **States cannot cross layers.** `engram::MemoryBlock` is constructible only through its own
  layer, so a state built for one output width is rejected at the entry instead of producing a
  plausible wrong number.
- **Rejections come before the first write.** Geometry, finiteness, spectrum and step size are
  checked at the entry; a failing call leaves no half-advanced state.
- **No second implementation of a formula.** Dot products, softmax and weighted sums reuse the
  single helper in the kernel family; a module that needs one reuses it rather than restating it.
