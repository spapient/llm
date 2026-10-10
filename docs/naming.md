# Naming Standard

Status: Proposed
Scope: Identifier naming for functions, methods, constants, macros, and type aliases across all workspace crates
Audience: Engineers, reviewers, maintainers, and AI coding agents

---

## 1. Purpose

This document is the authoritative source for identifier naming in the `llm`
workspace. It is the Rust translation (`rule.md` A8) of the naming requirements
in `rule.md` §7 (`NM1`–`NM5`), and it defines the single mechanism that turns
those requirements into a mechanical gate: `scripts/checknaming.sh`.

Naming is a contract. A reader must be able to infer what a function does from
its verb and what it returns from its noun, without reading the body. Synonym
drift is the failure mode this standard exists to prevent: two functions that do
the same thing under two different verbs force every reader to re-derive intent.

`rule.md` remains the normative source for the strength and verification axis of
each requirement. This document does not restate those axes; it defines the
concrete verb table, the forbidden synonyms, and the mechanical checks.

## 2. Normative Language

The terms MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY express engineering
requirements, matching `docs/public-api.md` and `docs/testing.md`.

A MUST-level requirement may be waived only through an explicit, reviewable
decision that records the requirement ID, the reason, the fallback, and the
scope (`rule.md` M4).

## 3. Canonical verb table

`rule.md` `NM3` fixes a global verb set so the same operation always reads the
same way. The table below is that set, translated (`A8`) to the Rust mechanism
that expresses each verb. Resource release is not a verb: it is owned by scope
and `Drop` (`OW1`, `NM3`), so no release verb exists.

| Verb | Meaning | Rust mechanism |
| --- | --- | --- |
| `get` | Read a value without transferring ownership | `fn get<field>(&self) -> T` |
| `set` | Overwrite a stored value | `fn set<field>(&mut self, ...)` |
| `new` | Primary constructor of the type (`init` equivalent) | `fn new(...) -> Self` |
| `from` | Construct from an explicit source | `fn from<source>(source) -> Self` |
| `parse` | Validate and decode untrusted input | `fn parse(bytes) -> Result<Self, E>` |
| `load` | Restore persisted state | `fn load<thing>(...) -> Result<T, E>` |
| `store` | Persist state | `fn store<thing>(...) -> Result<(), E>` |
| `read` | Read from an external source without decoding | `fn read<thing>(...)` |
| `write` | Write to an external sink | `fn write<thing>(...)` |
| `open` | Acquire a handle | `fn open(...) -> Result<Handle, E>` |
| `close` | Release a handle explicitly | `fn close(...)` |

`new` names only a constructor that returns `Self`. A fallible factory that
returns a *different* type MUST NOT be named `new`: Clippy's `new_ret_no_self`
rejects it, and the name would mislead a reader about what is constructed. Name
the produced value instead with the `new<object>` form, as in
`architecture::newmodel`, or use `from<source>` when the source is the salient
part of the name. This is the Rust translation (`A8`) of "a constructor is named
for what it constructs".

Beyond the fixed table, the workspace uses additional verbs that are not
synonyms of a table verb and therefore do not violate `NM3`: `check` (validate
a precondition and return `Result`), `apply` (drive a stored operation),
`dispatch` (select an operation), `run`, `reset`, `clear`, `add`, `insert`,
`append`, `bind`, `collect`, `cancel`, `try` (fallible `try_reserve`/`try_admit`
style), `encode`/`decode`, `format`/`render`, `record`, `sample`, `step`,
`restore`/`save`, `generate`, `prefill`, and the operator names that mirror the
math (`matmul`, `softmax`, `relu`, `silu`, `gelutanh`, `rmsnorm`, `rope`,
`embedding`, `attention`). These are registered here so the set is closed and
reviewable; adding a new non-table verb requires a line in this section.

## 4. Forbidden synonyms

`NM3` forbids mixing synonyms. The following verbs MUST NOT appear as the
leading token of a function or method name, because each duplicates a canonical
verb. The gate in section 9 enforces this list mechanically.

| Forbidden | Canonical | Why |
| --- | --- | --- |
| `build`, `create`, `make`, `construct`, `instantiate`, `assemble` | `new` / `from` | A constructor is `new`; construction from a source is `from<source>` |
| `fetch`, `retrieve`, `obtain`, `acquire`, `query` | `get` | Reading a value is `get` |
| `assign`, `put` | `set` | Overwriting a value is `set` |
| `dispose`, `free`, `release`, `dealloc` | scope / `Drop` | Release is not a verb (`OW1`, `NM3`) |

`parse` is canonical and is NOT a synonym of `read`: `read` moves bytes from a
source, `parse` validates and decodes them into a typed value. A function that
does both MUST be named `parse<thing>` and MUST name its source in its
documentation.

### 4.1 Why a synonym deny-list, not a full allow-list

The workspace has 518 distinct production function names. The clear majority are
either trait-implementation methods imposed by the standard library (`fmt`,
`deref`, `drop`, `eq`, `default`, `next`, `iter`, `from`) or operator/domain
nouns that mirror the mathematics (`matmul`, `softmax`, `rmsnorm`, `rope`). A
gate that required every name to start with a verb from the table would either
reject those names or require an exemption file listing several hundred entries,
which would reduce the check to a rubber stamp and defeat its purpose.

The deny-list instead encodes the actual `NM3` requirement — do not mix
synonyms — as a small, closed, reviewable set. A name that starts with a
forbidden synonym is a real defect; a name that starts with any other token is
left to human review for `NM1` self-description. This keeps the mechanical gate
sharp and low-noise. The rejected alternative is recorded in section 11.

## 5. Abbreviations (NM1)

`NM1` requires self-describing names and forbids abbreviations with no shared
convention. A short name is acceptable only when it is in the whitelist below or
is scoped to a local temporary (`NM4`).

Approved abbreviations: `id`, `eps`, `lr`, `dim`, `seq`, `max`, `min`, `fan`,
`cos`, `sin`, `std`, `lo`, `hi`, `n` (element count).

Rejected: any other contraction of a domain word. Write `tokenizer` and
`vocabulary`, not `tok` and `vocab`, when the identifier is broader than one
expression. `rule.md` `NM4` permits single-letter temporaries (`i`, `j`, `k`,
`t`, `tmp`) inside a function body.

## 6. Name gradient (NM4)

The broader the scope, the more complete the name. A public cross-crate symbol
spells its object out in full (`getkeyvalueheads`); a module-private helper may
be terser (`reshape`); a loop temporary is minimal (`i`). This is a SHOULD, not a
defect, and is not mechanically enforced.

## 7. Constants, macros, and type aliases (NM5)

Constants and macros MUST use uppercase compact spelling (`NM5`): `[A-Z][A-Z0-9_]*`.

Type aliases MUST NOT hide ownership or mutability. `type buffer = Vec<f32>` is
acceptable; a name that reads like an owned value but aliases a reference, or a
name that hides `mut`, is not. Type, trait, enum, and variant identifiers are
lowercase-compact by the workspace no-separator convention declared in the root
`Cargo.toml` (`rule.md` M4, `A8`); this document does not change that relaxation.

## 8. Exemption discipline

Any `#[allow(...)]` that silences a naming lint (`non_snake_case`,
`non_camel_case_types`, `non_upper_case_globals`) MUST carry a comment naming the
`rule.md` clause it relies on and the reason. A naming exemption without a clause
reference is a defect. The workspace-level relaxation in the root `Cargo.toml`
already records its `M4` declaration at the point of declaration; source-level
exemptions must do the same.

## 9. Enforcement

`scripts/checknaming.sh` is the mechanical gate. It asserts, over `crates/*/src`
and excluding `#[cfg(test)]` regions:

1. No function or method name starts with a forbidden synonym (section 4).
2. Every `const` name matches `[A-Z][A-Z0-9_]*` (section 7).
3. Every `macro_rules!` name matches `[A-Z][A-Z0-9_]*` (section 7).
4. Every naming-lint `#[allow(...)]` names a `NM` clause (section 8).

The gate is wired into `scripts/check.sh` after Clippy and before rustdoc. A
non-zero exit fails the check. Test regions are excluded because `NM4` permits
terse local names and test helpers are not part of any contract.

## 10. Verdict recording

A change that alters this standard, the deny-list, or the gate MUST update the
`docs/naming.md` row in `docs/conformance.md` with the executed command and its
exit code.

## 11. Alternatives considered

* **Full allow-list of every function name.** Rejected: it would restate 518
  names, most of them trait methods or operator nouns, and would rubber-stamp
  new names instead of reviewing them. The deny-list targets the actual `NM3`
  defect surface.
* **Enforce the verb table through Clippy alone.** Rejected: Clippy has no
  synonym knowledge; it cannot know that `build` duplicates `new`. The rule is
  project-specific and belongs in the project's own gate.
* **Rename the test helpers that use terse names (`ds`, `ckpt`, `caps`).**
  Rejected: `NM4` explicitly permits minimal local names, and test helpers are
  outside every public contract. Renaming them would be churn without a
  verified defect.
