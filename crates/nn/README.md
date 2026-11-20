# `llm-nn`

Layer definitions and the mechanisms behind them: every layer is a **declaration** (spec, bounds,
contract) with no arithmetic, and every mechanism is a pure computation with a testable invariant.

| Layer group | Declarations |
| --- | --- |
| core | `Embedding`, `LmHead`, `MatMul`, `Attention`, `AttentionBlock`, `Stack`, `Floor` |
| pointwise | `RmsNorm`, `Rope`, `Silu`, `Gelu`, `Softmax`, `ResidualAdd`, `Scale`, `Add`, `Mul`, `ReduceSum`, `ReduceMax` |
| conditional memory | `Bypass`, `MemoryBlock`, `Table` |

Arithmetic lives in `llm-op`; this crate names the shapes, the bounds and the contracts, and by
`#[spec]` keeps "declared cost" and "executed op" the same fact.

## Mechanisms

The eleven architecture benchmarks in `docs/lld.md` §5 are implemented here as testable modules.
Each one is a computation plus the property that makes it falsifiable:

| Mechanism | Module | Invariant that is actually asserted |
| --- | --- | --- |
| CSA / HCA long attention | `csa` (`BlockTable`) | `m = win = 1` reproduces dense causal attention bitwise |
| Engram conditional memory | `engram` (`Table`, `slot_of`, `Bypass`) | a short history adds nothing instead of padding with zeros |
| three-channel gating | `mix`, `policy` (`Channel`, `Gating`, `MissRate`) | coefficients form a partition of unity; synthesis has one entry point |
| DeepSeek MLA | `mla` (`Mla`, `Latent`, `Path`) | absorbed and up-reference paths agree; cache holds only `(c^KV, k^R)` |
| byte-latent (BLT) | `patch`, `codec`, `latent`, `emit` | prefix stability, incremental split equals whole-input split |
| Kimi K3 | `kda`, `attnres`, `moe`, `train::Muon` | writing the same key twice returns the later value; output stays in the convex hull |
| double-stochastic mHC | `mhc` | `‖B ⊗ I‖₁ = 1`; non-negativity is an entry condition, not a comment |
| Mamba-3 | `mamba` | trapezoid and zero-order discretizations differ where they should, and agree where they must |
| SpikingBrain 2.0 | `spike` (`Spiking`, `Membrane`) | the SSE path and the dense path agree bitwise on the same integer arithmetic |

## Guarantees

- **States cannot cross layers.** `kda::Memory`, `mamba::Carry`, `spike::Membrane` are only
  constructible through their own layer (`memory`, `membrane`), so a state built for one output
  width is rejected at the entry instead of producing a plausible wrong number.
- **Rejections come before the first write.** Geometry, finiteness, spectrum and step size are
  checked at the entry; a failing call leaves no half-advanced state.
- **The metrics a mechanism produces are quantities, not verdicts.** `mix::Events::sparsity` and
  `Audit` measure E14/E15; whether a number PASSes is `tool::card`'s job, on a frozen sample set
  with a fixed encoder — never inferred here.
- **No second implementation of a formula.** Dot products, softmax and weighted sums are `csa`'s
  `pub(crate)` helpers; slot mapping is `engram::slot_of`; threshold firing is `mix::Lif`. A module
  that needs one of these reuses it rather than restating it.
