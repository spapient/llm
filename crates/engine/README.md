# `llm-engine`

Serving state machine: sequences, their KV pages, the admission scheduler, and the submission
contract. One file per concept, no subdirectories.

| Module | Answers |
| --- | --- |
| `scheduler` | which sequence takes the next step, and can it get the page? (`Scheduler`, private `Plan`) |
| `sequence` | how far has one request got? (`Sequence` / `State` / `Progress`) |
| `batch` | how many rows fit **this** round, and what settles when? (`Batch::round`, `Admission`) |
| `ifb` | which three constraints are being decided? (`Budget` / `Candidate` / `Phase` / `pack`) |
| `kv` | where do a sequence's tokens live? (`Page` / `Table` / `Slot` / `Cache`, refcounted) |
| `pager` | which page is free, and who holds it? (page pool behind `kv`) |
| `radix` | which prefix is already computed? (radix block tree over page tables) |
| `queue` | how is a program submitted and executed? (`Executor` seam, ADR 0051; `Buf` / `View`) |
| `exec` | what does a submission *mean*? (`Stream` / `Graph` / `Node` / `Launch` / `Operand`) |
| `cpu` | how does a graph run in-process? (`LocalExecutor`, `Program`, `Bindings`) |

Device sources, vendor chains and platform eligibility live in `llm-device`; `engine` only defines
the seam (`exec::Executor`) that it fills.

## Guarantees

- **Admission separates retryable from terminal.** `OverSeqs` means the round is full and the same
  request succeeds later; `OverTokens` means the generating row alone exceeds the budget and retrying
  changes nothing. Collapsing them into one "rejected" loses the only actionable bit.
- **Pages are bookkept, not copied.** Data lives in `Cache`, page numbers in `Table`; sharing a
  prefix means two tables referencing the same pages, and `finish` returns them. `in_use` is the
  single count of "how many pages are spoken for".
- **Prefix registration is incremental, not a rescan.** Block `b` settles once rows `(b+1)·width`
  are written. Rescanning from 0 per write would re-walk the tree to hear "already there" — a
  prefill of length P paying `O(P²/width)` walks. `Batch::settle` asks the tree how deep this token
  path already goes and registers only past that, so the cost tracks blocks that are genuinely new —
  including when a caller writes rows out of order and skipped blocks still need publishing.
- **Nothing owns the epoch but the page count.** `in_use` is the single count of "how many pages are
  spoken for"; there is no second ledger to drift from it.

## Where the numbers come from

Launch geometry and vendor source text are `llm-device`'s; the measurement that judges the page
account here is `E12` page access — an `#[ignore]` probe in `tests/kv.rs` that reports random-access
recall and slot-lookup cost. A mechanism probe reports a quantity, never a PASS; the verdict belongs
to `tool::card`.
