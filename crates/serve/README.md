# `llm-serve`

The request-to-scheduler bridge: what a client may ask for, how a plan is assembled into a graph,
and where a decode step actually computes.

| Module | Answers |
| --- | --- |
| `protocol`, `frame`, `http` | what is on the wire? (request/response shape, framing, `Httpd`) |
| `request`, `router` | which route handles this, and with what limits? |
| `plan`, `params` | which layers does this model have? (`LayerKind` closed set, `MemorySpec`) |
| `assembly`, `model_host` | how does the plan become a graph, and who feeds batch tables each step? |
| `completion`, `generate` | how does one completion advance? (prefill chunk, decode step) |
| `kernels` | which execution path does this geometry take? (`gemv_quant` row split, `attention_f32`) |
| `speculate` | how many drafted tokens does the target accept? (`Verdict`, `Drafter`) |
| `block` | which token does the head pick? (shared `best` argmax) |
| `vocab` | how are ids turned into text? (`tokenizer_from_segment`) |

## Guarantees

- **One argmax.** Draft acceptance, sampling and greedy decoding all read `generate::best`; a second
  argmax would let two layers disagree about the same distribution.
- **The graph is fixed at assembly time.** `LayerKind` is a closed set (projection / unary / norm /
  embedding / binary / rope / attention block / memory); per-step variation goes through batch
  tables that `model_host` overwrites, never through graph edits.
- **The memory layer's gate is a multiply, not a route.** `γ = 0` adds a zero row and leaves the
  main chain bitwise unchanged — the same result as "this row has no memory", not a second code path.
- **Budget and pages are one decision.** `B_max` exhaustion returns the same 503 as "no pages left";
  chunked prefill gets its block tail from one place.
- **Quantized weights are unpacked once per call.** `kernels::gemv_quant` splits by row count so a
  batch reuses a single decode of the packed stream.
- **Token-only prediction requires an embedding first layer.** A model whose first layer consumes
  hidden vectors is rejected by `CpuModel::logits`; this API has token IDs but no external vector
  source, so it never fabricates a zero-vector input.

## Status boundary

What runs end-to-end is the dense Transformer main axis (embedding → N attention blocks → head), the
conditional-memory layer, and multi-sequence batched decode. Speculative **acceptance** is decided
here, but the draft model (EAGLE-3 / MTP) is not part of this crate; the `ifb` budget decides how many
rows fit a round, while the continuous-batching scheduler still admits in arrival order. Benchmarks
E1–E15 have no measured cards on this path yet — mechanism probes report quantities, verdicts come
from `tool::card`.
