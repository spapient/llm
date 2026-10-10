# Training — Workflows, Optimizers, and Recovery

Status: Proposed
Scope: Training loops, losses, optimizer integration, gradient accumulation, evaluation, and run recovery in the llm/ workspace
Audience: Engineers, reviewers, maintainers, and AI coding agents

---

This document is a placeholder. It is the intended authoritative source for the
training contract: the training loop, loss computation, optimizer update
semantics, gradient accumulation, evaluation, checkpoint/resume behavior, and the
guarantees provided when a step fails after partially mutating state.

Until this document is completed, training behavior is defined only by:

* [error-handling.md](error-handling.md) section 16.1 (Training) and section 13 (Retry Policy).
* [crate-boundaries.md](crate-boundaries.md) sections 5.15 (`optim`) and 5.16 (`train`).
* The public API and documentation of the `optim` and `train` crates.

The `optim` and `train` crates are now implemented, so training behavior is
also defined by their public API and by
[decision-records/training.md](decision-records/training.md). This body still
needs to be replaced with the normative loop specification, optimizer
contracts, and the tested failure and recovery semantics. End-to-end backward
through the forward-only `nn`/`models` stack is a declared limitation, not an
implemented capability.
