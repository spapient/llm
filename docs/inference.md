# Inference — Generation, Request Lifecycle, and KV Cache

Status: Proposed
Scope: Autoregressive generation, request lifecycle, KV cache ownership, sampling, and streaming in the llm/ workspace
Audience: Engineers, reviewers, maintainers, and AI coding agents

---

This document is a placeholder. It is the intended authoritative source for the
inference contract: the request lifecycle, prefill and decode phases, KV cache
ownership and reuse, sampling and stop conditions, and streaming-output error
semantics.

Until this document is completed, inference behavior is defined only by:

* [error-handling.md](error-handling.md) section 16.2 (Inference) and section 12 (Cancellation and Timeout).
* [architecture.md](architecture.md) section 6 (Model and Request Lifecycles).
* [crate-boundaries.md](crate-boundaries.md) section 5.17 (`infer`).
* The public API and documentation of the `infer` crate.

The `infer` crate is now implemented, so inference behavior is also defined by
its public API and by
[decision-records/inference.md](decision-records/inference.md). This body still
needs to be replaced with the normative request-isolation invariants, KV cache
lifecycle, and the tested cancellation and streaming guarantees.
