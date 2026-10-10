# Checkpoint Format — Artifact Layout, Metadata, and Integrity

Status: Proposed
Scope: On-disk layout of model and training-state checkpoints, metadata, integrity, and compatibility in the llm/ workspace
Audience: Engineers, reviewers, maintainers, and AI coding agents

---

This document is a placeholder. It is the intended authoritative source for the
checkpoint format: the artifact layout, the metadata schema, integrity
verification, atomic-write and commit semantics, and the compatibility rules
between checkpoints and model versions.

Until this document is completed, persistence behavior is defined only by:

* [error-handling.md](error-handling.md) section 14 (Checkpoint and Persistent-State Failures).
* [crate-boundaries.md](crate-boundaries.md) section 5.14 (`checkpoint`).
* The public API and documentation of the `checkpoint` crate.

The `checkpoint` and `format` crates are now implemented, so persistence
behavior is also defined by their public API and by
[decision-records/persistence.md](decision-records/persistence.md). This body
still needs to be replaced with the normative format specification, the
byte-level layout, and the tested round-trip and recovery guarantees.
