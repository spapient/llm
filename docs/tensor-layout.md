# Tensor Layout — Shapes, Strides, and Views

Status: Proposed
Scope: Tensor metadata, storage, shape validation, layout and stride semantics, and view lifetimes in the llm/ workspace
Audience: Engineers, reviewers, maintainers, and AI coding agents

---

This document is a placeholder. It is referenced by [error-handling.md](error-handling.md) as the authoritative source for tensor layout semantics, but its normative content has not been written yet.

Until this document is completed, the tensor layout contract is defined only by:

* [architecture.md](architecture.md) section 5 (Tensor and Backend Boundary).
* The public API and documentation of the `tensor` crate.

This placeholder MUST NOT be treated as a complete specification. When the layout contract is finalized, replace this body with the normative requirements, invariants, and failure semantics, and update the cross-references in [error-handling.md](error-handling.md) accordingly.
