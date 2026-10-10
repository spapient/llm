# ADR: `tokenizer` — Vocabulary, BPE, Encode/Decode, Special Tokens, and Chat Templates

Status: Accepted
Scope: `crates/tokenizer` — public contracts, invariants, error semantics, and dependency/naming choices
Roadmap step: P4
Date: 2026-10-10

## Context

Roadmap stage P4 introduces the `tokenizer` crate. Per
`docs/crate-boundaries.md` section 5.8, `tokenizer` owns:

* Vocabulary representation.
* Encoding and decoding.
* Byte-pair encoding where applicable.
* Special-token handling.
* Chat-template processing.

It MUST NOT own tensor execution, model weights, KV cache, or request
scheduling. Its only workspace dependency is `lib` (for `tokenid`); it uses
`serde_json` for parsing the HuggingFace `tokenizer.json` document. It
deliberately does **not** depend on `format`, `data`, `models`, or any workflow
crate. Text-to-token conversion is a composition consumed by higher layers
(for example `infer`), so no higher crate is imported here.

The crate follows the repository no-separator naming convention (`rule.md`
NM1), the workspace no-`unsafe` and no-`panic`/`unwrap`/`expect` rules
(`AGENTS.md` sections 4 and 6), and owns its own `thiserror` error enum.

## Decision

The crate exposes nine public items rooted at the crate root and re-exported
from collision-free module files:

| Public item | Module file | Responsibility |
| --- | --- | --- |
| `vocabulary` | `vocab.rs` | Immutable bidirectional token/id map |
| `bytepair` | `merges.rs` | Ordered BPE merge table with rank precedence |
| `specialtokens` | `special.rs` | Immutable set of structural markers |
| `chatrole` | `chat.rs` | Conversational role enumeration |
| `chatmessage` | `chat.rs` | One role/content message |
| `chattemplateentry` | `chat.rs` | Per-role prefix/suffix pair |
| `chattemplate` | `chat.rs` | Explicit non-Jinja chat template |
| `tokenizer` | `engine.rs` | Loads the JSON subset and orchestrates conversion |
| `tokenizererror` | `error.rs` | Crate-owned error enum |

### DS1 — Boundaries and interfaces

* `tokenizer` depends on `lib` only (`tokenid`); external parsing is
  `serde_json`, declared as a workspace dependency in the crate manifest. No
  `tensor`, `api`, `cpu`, `format`, `data`, or workflow crate is imported.
* `tokenizer` is a value type with no interior mutability except for the
  install-once chat template (`setchattemplate`); it holds no request state and
  no cache, so it can be shared immutably across requests (`OW2`,
  `docs/crate-boundaries.md` section 5.8).
* Public interface width: 9 items at the crate root, well under the `A4` bound
  of 20. *依据：单模块公共契约应能在一轮评审内完整复核（`rule.md` A4）。*

### DS2 — Invariants and error paths

Algorithm-level invariants:

* `vocabulary` is a bijection: for every entry, `getcontent(getid(content))`
  returns `content` and `getid(getcontent(id))` returns `id`. Repeating the
  exact same `(content, id)` pair is accepted; any other conflict is rejected.
* `bytepair` ranks are the array indices of first occurrence; a duplicate merge
  string keeps its first rank. `apply` is deterministic: at each step it merges
  the adjacent pair with the lowest rank and resolves ties toward the leftmost
  pair, until no rule applies.
* `specialtokens` resolves by content and by id over the same entry set.
* `decodetokens` consumes byte-fallback tokens into a pending byte buffer and
  emits a single UTF-8 string; the concatenation is byte-exact when the ids
  form valid UTF-8.

Implementation-level invariants:

* `vocabulary` stores the id to token direction in a `BTreeMap<u32, String>`
  rather than a dense `Vec`, so an out-of-range id from untrusted input cannot
  trigger a large allocation; ids are validated against inserted entries, never
  against their numeric magnitude (`IN4`).
* `encodespecial` sorts candidate literals longest-first with a content
  tie-break, so overlapping markers resolve deterministically.
* `parsebytetoken` only accepts exactly `<0x` + two hex digits + `>`; anything
  else is treated as an ordinary token.

Error branches and return semantics (`tokenizererror`, 15 variants):

* `invalidjson { line, column }` — the document is not syntactically valid JSON.
* `rootnotobject` — the JSON root is not an object.
* `missingfield { field }` — a required dotted path is absent.
* `unexpectedtype { field, expected }` — a present field has the wrong JSON type.
* `unsupportedmodel { modeltype }` — `model.type` is not `"BPE"`.
* `idoutofrange { id }` — a token id does not fit in `u32`.
* `duplicateid { id }` / `duplicatecontent { token }` — an inconsistent
  vocabulary.
* `malformedmerge { index }` — a merge entry is not a `"left right"` pair with
  two non-empty sides.
* `malformedspecialtoken { index }` — an `added_tokens` entry is missing `id`
  or `content` or has a mistyped `special`.
* `unknowntoken { token }` — a symbol has no vocabulary entry, no complete byte
  fallback, and no configured unknown token.
* `unknowntokenid { id }` — a decode input id is absent from the vocabulary.
* `invalidutf8` — byte tokens do not reconstruct valid UTF-8.
* `nochattemplate` — `applychattemplate` was called with no template installed.
* `unsupportedrole { role }` — a message role has no template entry.

Resource rollback: all construction is value-based. `fromjson` builds local maps
and only returns `Ok` after every stage validates, so a rejected document
allocates no `tokenizer`. Decoding accumulates into a `String` and a pending byte
`Vec` that are dropped on error (scope-based release, `OW1`, `ER4`).

### DS3 — Complexity pre-control

Each function stays under the `rule.md` DS3 bounds of cyclomatic complexity ≤ 10
and ≤ 80 effective lines. `fromjson` is decomposed into `parsevocab`,
`parsemerges`, and `parseaddedtokens`; encoding is decomposed into
`encodeplain`, `encodespecial`, `flushsegment`, and `resolvetoken`; decoding is
decomposed into `decodetokens`, `parsebytetoken`, and `flushbytes`. *依据：超限
函数无法在一轮评审内完成分支验证（`rule.md` DS3）。*

### DS4 — Concurrency design

`tokenizer` introduces no shared mutable state and no synchronization primitive.
Every component is an immutable value; the only mutation is installing a chat
template on an owned `tokenizer` before sharing it. Consequently a `tokenizer`
can be shared by reference across threads without locking (`OW4`), and no lock
ordering exists to define. The crate holds no request-scoped state, so
cancellation and request isolation are not its concern (`OW2`).

## HuggingFace `tokenizer.json` subset

`fromjson` reads only the following fields; unknown fields are ignored:

| Path | Type | Required | Handling |
| --- | --- | --- | --- |
| `model.type` | string | yes | must equal `"BPE"` |
| `model.vocab` | object | yes | token to integer id; ids must fit `u32` |
| `model.merges` | array of strings | no | defaults to empty |
| `model.byte_fallback` | boolean | no | defaults to `false` |
| `model.unk_token` | string | no | must resolve in the vocabulary |
| `added_tokens` | array of objects | no | `{id, content, special?}` |

`added_tokens` entries are merged into the vocabulary entries before the
bijection check, so a duplicate id/content across `model.vocab` and
`added_tokens` is rejected. Only entries with `special == true` join
`specialtokens`; a missing `special` defaults to `false`.

**Deliberately not implemented** (documented limitations, not silent gaps):

* Pre-tokenization / splitting rules (the input is treated as Unicode scalars).
* The GPT-2 byte-to-unicode alphabet; byte fallback uses the literal `<0xNN>`
  token form.
* Jinja chat-template evaluation; only the explicit prefix/suffix form is
  supported.
* `normalizer`, `pre_tokenizer`, `post_processor`, and `decoder` sections.

## Encoding, byte fallback, and decoding

`encode` / `encodeplain` treat the text as a sequence of Unicode scalars, apply
the BPE merge table, then resolve each resulting symbol in this order:

1. Exact vocabulary match.
2. If `byte_fallback` is enabled and every UTF-8 byte of the symbol has a
   `<0xNN>` vocabulary entry, emit those byte tokens.
3. The configured `unk_token`, if any.
4. Otherwise `unknowntoken`.

`encodespecial` scans left to right; at each position the longest special literal
that matches is emitted directly, and the intervening text is encoded with
`encodeplain`. `encode` never interprets special literals.

`decode` validates every id against the vocabulary (`unknowntokenid` on
failure, never a silent skip). A `<0xNN>` token appends a byte to a pending
buffer; any other token flushes the buffer as UTF-8 and is appended verbatim. A
flush of bytes that are not valid UTF-8 yields `invalidutf8`. This satisfies the
`IN1` requirement that token ids are validated before use.

## Chat template

`chattemplate` holds an optional BOS token, an optional EOS token, and a list of
per-role `chattemplateentry { role, prefix, suffix }`. `render` concatenates
`prefix + content + suffix` for each message and returns `unsupportedrole` when a
message role has no entry. `applychattemplate` emits the template BOS (if any)
and then encodes the rendered string with special awareness; the EOS token is
exposed through `geteos` for the caller to append, because whether and when to
emit EOS is a generation concern owned by `infer`, not by the tokenizer.

## Resource limits

* Vocabulary ids are validated against inserted entries, and the id to token map
  is sparse (`BTreeMap`), so a hostile id cannot cause an oversized allocation
  (`IN4`).
* `fromjson` performs no additional byte-size cap beyond the caller-supplied
  `&str`; bounding the document size is the loader's responsibility (a later
  `format`/`infer` concern), which is recorded here as a limitation.
* Byte-fallback expansion is bounded by the UTF-8 length of the input symbol.
* No benchmark is provided: byte-pair encoding is not a computational operator
  under `AGENTS.md` section 7, and no performance improvement is claimed.

## Naming and module layout

The repository no-separator convention makes a module and a type re-exported
from it collide in one name namespace (`E0255`; see
`docs/decision-records/workspacelayout.md`). The module files are therefore
non-clashing containers: `vocab.rs` to `vocabulary`, `merges.rs` to `bytepair`,
`special.rs` to `specialtokens`, `engine.rs` to `tokenizer`, and `chat.rs` to the
chat types. This is the repository's `rule.md` A8 translation of the roadmap's
filename list (`lib, tokenizer, vocabulary, encode, decode, bytepair,
specialtokens, chattemplate`) onto Rust.

## DS5 — Alternatives considered

* **Depend on a general JSON-schema/deserialization derive.** Rejected: the
  accepted subset is small and the manual parse keeps every malformed-input path
  a distinct typed error. `serde_json::Value` traversal avoids adding `serde`
  derive machinery for one document shape.
* **Store ids in a dense `Vec<String>` indexed by id.** Rejected: a hostile id
  such as `4294967295` would force a multi-gigabyte allocation (`IN4`). The
  `BTreeMap` is sparse and validates against real entries.
* **Implement the GPT-2 byte-to-unicode alphabet.** Rejected for this stage:
  the required round-trip behavior is expressible with literal `<0xNN>` byte
  tokens, and the alphabet adds a large static table without a current consumer.
* **Evaluate HuggingFace Jinja templates.** Rejected: it would require a
  template engine dependency and untrusted-code evaluation, well beyond the
  narrow, testable prefix/suffix composition the runtime needs.
* **Have `applychattemplate` also append EOS.** Rejected: deciding when a
  sequence terminates belongs to generation (`infer`); the tokenizer exposes EOS
  rather than silently emitting it.
* **Put tokenization in `data` or `infer`.** Rejected:
  `docs/crate-boundaries.md` section 5.8 assigns text to token conversion to
  `tokenizer`, and `data` deliberately avoids a reverse dependency on it.

## Consequences

* `tokenizer` is a dependency leaf above `lib`; adding it does not change any
  existing crate's manifest because root membership is glob-based
  (`members = ["crates/*"]`).
* Encode/decode round-trips are deterministic and byte-exact for inputs the
  vocabulary can represent.
* Malformed and unsupported documents fail with distinct typed errors rather
  than panics or opaque messages.

## Validation

Commands are scoped to `tokenizer` because sibling crates are being built
concurrently. Executed on 2026-10-10 from `/Users/fei/Doubao/llm`:

```sh
cargo fmt --all
cargo check -p tokenizer --all-targets
cargo clippy -p tokenizer --all-targets -- -D warnings
cargo test -p tokenizer
RUSTDOCFLAGS='-D warnings' cargo doc -p tokenizer --no-deps
```

Results:

* `cargo clippy -p tokenizer --all-targets -- -D warnings` — pass.
* `cargo test -p tokenizer` — 47 unit tests + 8 integration tests, 0 failed.
* `RUSTDOCFLAGS='-D warnings' cargo doc -p tokenizer --no-deps` — pass.

Coverage:

* `vocabulary` — both directions, count/bounds, duplicate id, duplicate content,
  repeated identical pair, empty.
* `bytepair` — parse/rank, missing separator, empty side, lowest-rank-first,
  repeated merging to one symbol, unmergeable sequences, leftmost tie-break.
* `specialtokens` — both directions, contains/count, content-order iteration,
  empty.
* `encode` — empty input, merge-before-lookup, byte fallback, unknown without
  fallback, unknown with `unk`, special splitting, plain ignores specials.
* `decode` — plain tokens, UTF-8 reconstruction, empty, unknown id, incomplete
  UTF-8, byte-token shape parser.
* `chat` — role names, rendered wrapping, uncovered role, empty conversation.
* `engine` — loading/exposure, invalid JSON, non-object root, missing type,
  non-BPE model, non-integer id, out-of-range id, unknown `unk_token`,
  round-trip, special-aware encoding, missing template.
* `tests/tokens.rs` — ASCII and UTF-8 round-trips, `unk` fallback, structural
  specials, BOS-prefixed chat rendering, absent template, out-of-vocabulary
  decode, and a cluster of malformed documents that must not panic.

## Revisit conditions

* When a model requires pre-tokenization (for example GPT-2 regex splitting) or
  the byte-to-unicode alphabet, extend `encodeplain` and add reference tests
  against the upstream tokenizer.
* When a caller needs JSON `config.json`/`tokenizer.json` size caps enforced in
  this crate, add an explicit `lib::limits` bound at the `fromjson` boundary.
* When the runtime adds a second chat format, generalize `chattemplate` rather
  than evaluating Jinja.
