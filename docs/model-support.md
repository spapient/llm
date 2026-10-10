# Model Support — Architectures, Configuration, and Compatibility

Status: Proposed
Scope: Supported model architectures, configuration compatibility, and the model registry in the llm/ workspace
Audience: Engineers, reviewers, maintainers, and AI coding agents

---

This document is a placeholder. It is the intended authoritative source for the
model-support contract: which architectures are supported, how a model
configuration is validated, how weights are mapped onto a model, and what
compatibility guarantees are provided across checkpoints and versions.

Until this document is completed, model behavior is defined only by:

* [architecture.md](architecture.md) section 6 (Model and Request Lifecycles).
* [crate-boundaries.md](crate-boundaries.md) sections 5.10 (`models`) and 5.9 (`nn`).
* The public API and documentation of the `models` crate.

The `models` and `nn` crates are now implemented, so model behavior is also
defined by their public API and by
[decision-records/models.md](decision-records/models.md) and
[decision-records/nn.md](decision-records/nn.md). This body still needs to be
replaced with the normative requirements, invariants, supported-architecture
matrix, and version-compatibility rules. Only the forward path exists today.
