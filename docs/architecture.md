# Architecture — Minimal Composable LLM Runtime

Status: Proposed
Scope: Intended architectural boundaries of the Rust LLM workspace
Audience: Engineers, reviewers, maintainers, and AI coding agents

---

## 1. Purpose

This document defines the intended architectural boundaries of the Rust LLM workspace.

The architecture is based on a minimal kernel, static crate composition, narrow traits, explicit ownership, and one-way dependencies.

It is a target architecture. Existing source code and Cargo manifests must be audited before claiming conformance.

## 2. Design Principles

### 2.1 Minimal Kernel

The foundational layer contains only stable, reusable primitives.

It MUST NOT contain model-specific workflows, request scheduling, application logic, or backend-specific algorithms.

### 2.2 Explicit Composition

Capabilities are composed through narrow interfaces and concrete implementations.

Prefer static dispatch unless runtime polymorphism is required by a documented use case.

### 2.3 One-Way Dependencies

Dependencies flow from higher-level workflows toward lower-level capabilities.

Lower-level crates MUST NOT import higher-level application or workflow crates.

### 2.4 Explicit Ownership

Every mutable resource has a defined owner, lifecycle, and failure behavior.

Request-scoped state MUST remain separate from shared model weights.

### 2.5 Evidence-Based Optimization

Correctness is established through contracts and tests. Performance is established through reproducible measurements.

Neither can substitute for the other.

## 3. Architectural Layers

### Layer 0: Foundational Contracts

Crates: `lib`, `tensor`, `api`.

Responsibilities:

* Shared identifiers, configuration primitives, limits, precision, and seed types.
* Tensor shapes, layouts, storage abstractions, and views.
* Device, capability, and execution contracts.

Constraints:

* No application dependencies.
* No model-specific logic.
* No hidden global execution state.
* No backend-specific implementation details in public contracts unless required by the abstraction.

The precise dependency relationship between `tensor` and `api` MUST be explicitly selected and documented.

### Layer 1: Compute and Differentiation

Crates: `cpu`, `autograd`, `nn`.

Responsibilities:

* CPU operator implementations.
* Automatic differentiation.
* Neural network modules and parameter abstractions.

Constraints:

* Backend implementations depend on contracts, not the reverse.
* Operator semantics must be testable independently.
* Gradient behavior must be validated against mathematical or numerical references.
* Model-specific orchestration does not belong in the compute kernel.

### Layer 2: Models and Workflows

Crates: `models`, `infer`, `train`, `optim`, `context`, `data`, `tokenizer`, `format`, `checkpoint`.

Responsibilities:

* Model architecture composition.
* Inference and training workflows.
* Optimization state and update rules.
* Execution budgets, cancellation, and observability.
* Data preparation, tokenization, format validation, and persistence.

Constraints:

* Each crate owns a distinct domain responsibility.
* Shared concepts do not justify circular dependencies.
* Cross-domain workflows are composed by a higher-level owner.
* Model weights do not contain mutable request state.

The exact edges between these crates must be determined from public API requirements and verified against the actual workspace.

### Layer 3: Application and Tooling

Crates: `cli`, `serve`, `bench`, `testkit`.

Responsibilities:

* Application entry points and external interfaces.
* HTTP integration.
* Benchmark execution.
* Test support and reference implementations.

Constraints:

* Application crates must not become dependencies of the runtime kernel.
* `testkit` must not enter production dependency graphs.
* Benchmarks must exercise real code paths and disclose their workload.

## 4. Dependency Graph Policy

The intended dependency graph is a directed acyclic graph.

Before introducing an edge A → B, verify:

1. B owns the required abstraction or capability.
2. A does not require B to import A.
3. The edge does not introduce a dependency cycle.
4. The edge does not unnecessarily enlarge the foundational dependency closure.
5. The edge can be validated through compilation and architecture checks.

If two crates appear to require each other, do not immediately extract a large shared crate.

First identify the minimal shared contract and determine which layer should own it.

Any abstraction extraction must have a clear purpose and a limited public API.

## 5. Tensor and Backend Boundary

Tensor metadata and storage must have explicit semantics.

The architecture must document:

* Shape and dimension validation.
* Element type and storage representation.
* Layout and stride behavior.
* Ownership and view lifetimes.
* Device association.
* Allocation and resource-limit failures.
* Operator input and output contracts.

Backend traits must define observable behavior rather than exposing internal implementation details.

A backend implementation must not silently change the mathematical contract of an operator.

## 6. Model and Request Lifecycles

Model-level state includes architecture configuration and model parameters.

Request-level state may include:

* Request identity.
* Input and generated token state.
* KV Cache ownership.
* Sampling state.
* Cancellation and stop conditions.
* Per-request resource accounting.

The inference workflow coordinates these resources without storing request-specific mutable state in shared weights.

Request termination must define how resources are released after successful completion, cancellation, or failure.

## 7. Error Boundaries

Each crate defines errors for failures within its responsibility.

Examples include:

* Invalid tensor shape.
* Unsupported operator configuration.
* Backend execution failure.
* Invalid model configuration.
* Malformed external data.
* Checkpoint incompatibility.
* Resource budget exhaustion.
* Request cancellation.

Errors must preserve sufficient context for diagnosis without coupling foundational crates to application-specific error types.

Ordinary runtime failures must propagate through explicit error handling.

## 8. Resource Management

Resource limits are architectural contracts, not optional conveniences.

All externally influenced dimensions, sizes, and offsets require appropriate validation.

The implementation must define ownership and cleanup behavior for:

* Tensor allocations.
* Model-loading buffers.
* KV Cache allocations.
* Concurrent requests.
* Checkpoint writes.
* Training state.

Where resource exhaustion is possible, failure behavior must be explicit and testable.

## 9. Performance Architecture

Optimization proceeds in this order:

1. Establish correctness.
2. Establish a reproducible baseline.
3. Identify a measured bottleneck.
4. Select the smallest relevant change.
5. Re-run correctness tests.
6. Measure the change under comparable conditions.
7. Record results and limitations.

The architecture must not assume that SIMD, parallelism, caching, or additional abstraction automatically improves performance.

## 10. Architectural Decision Records

Significant decisions should record:

* Context and problem.
* Constraints.
* Options considered.
* Decision and rationale.
* Consequences and trade-offs.
* Validation requirements.
* Conditions that would justify revisiting the decision.

Architectural decisions must be based on demonstrated requirements rather than fashion or hypothetical future extensibility.

## 11. Conformance Status

Every architectural rule must be classified as one of:

* Proposed: intended but not yet verified.
* Verified: supported by source inspection and relevant checks.
* Non-conforming: existing implementation violates the rule.
* Blocked: verification requires missing code, tools, or environment.

Do not classify the repository as conforming solely because this document exists.
