# `llm-engine`

Serving state machine and device floor: sequences, their KV pages, the admission scheduler, the
submission contract — and the one place that asks "can this machine run this row at all".

| Module | Answers |
| --- | --- |
| `exec` | how is a program submitted and executed? (`Executor` is the runtime seam, ADR 0051) |
| `batch` | how many rows fit **this** round? (`Batch::round`, `Admission`, `OverSeqs` / `OverTokens`) |
| `kv` | where do a sequence's tokens live? (`Page` / `Table` / `Cache` / `Pager`, refcounted) |
| `radix` | which prefix is already computed? (radix block tree over page tables) |
| `ifb` | which three constraints are being decided? (`Budget` / `Candidate` / `Phase` / `pack`) |
| `dialect` / `cuda` / `msl` | what source runs this family? (`Dialect` lowering primitives; CUDA + Metal vendor drivers) |
| `metal::host` | how does a Metal kernel reach the GPU? (`xcrun metal` → `metallib` → host driver → readback) |
| `metal_pipeline` | which stages and host folds does a chunked family need? (`reduce` / `rmsnorm` / `softmax` / `scan`) |
| `tiled` | how is a launch shaped? (`Flash { seq, dim, threads, block }`, `lanes` / `tail`) |
| `backends` | which compiler and device exist here? (`nvcc` / `xcrun metal` / MSVC probe) |
| `metal` | how is the process-internal Metal executor driven? (the only `unsafe` island) |
| `norm` / `scan` / `wmma` | which kernel family is this, and what launch does it need? |
| `platform` | is this row runnable on this machine? (`Row` / `Standing` / `Why`, ADR 0052) |

## Guarantees

- **Admission separates retryable from terminal.** `OverSeqs` means the round is full and the same
  request succeeds later; `OverTokens` means the generating row alone exceeds the budget and retrying
  changes nothing. Collapsing them into one "rejected" loses the only actionable bit.
- **Pages are bookkept, not copied.** Data lives in `Cache`, page numbers in `Table`; sharing a
  prefix means two tables referencing the same pages, and `finish` returns them. `in_use` is the
  single count of "how many pages are spoken for".
- **The platform verdict never fails the audit.** `audit` emits all six rows every time — a missing
  row would make "not covered" and "never asked" the same output. `require` reports `E-PLAT` for the
  one row you asked about. It decides **eligibility**, not correctness: a passing `require` says
  nothing about whether a kernel computes the right number (`cutile::verify` and `tool::card` do).
- **`Why` is a three-value closed set.** `NoDevice` and `NoToolchain` have different fixes (another
  machine vs another install), so they are never merged into free-text "unsupported".
- **The vendor boundary stays behind one FFI island.** `engine` owns the probe, the source generation
  and the launch shape; the `unsafe` calls that actually reach a device live in `metal` and nowhere
  else (ADR 0035).
- **A chunked family is one program, not one kernel.** Partial results land in the output buffer per
  block, the host folds them (f64 sums, `a^len` by squaring) and the next stage re-launches on the
  same grid — so a family cannot claim "parallel" from its device segments alone.

## Where the numbers come from

The launch geometry (`tiled`) and the source text (`dialect` / `cuda` / `msl`) are mechanism; the measurements that
judge them (`E10` predictability, `E12` page access) live in `#[ignore]` probes on the modules they
measure, and their verdicts belong to `tool::card`. A mechanism probe reports a quantity, never a
PASS.
