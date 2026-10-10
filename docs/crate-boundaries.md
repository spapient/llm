# Crate Boundaries and Dependency Rules

Status: Proposed
Scope: Rust workspace architecture, crate ownership, dependency direction, public API boundaries, and architecture verification for all production, testing, benchmarking, and application crates in llm/
Audience: Engineers, reviewers, maintainers, and AI coding agents

---

## 1. Purpose

This document defines the authoritative ownership and dependency boundaries for the llm/ Rust workspace.

Its purpose is to prevent architectural erosion as the system grows. Every crate must have a clearly defined responsibility, a minimal public interface, and a dependency direction that can be inspected and verified.

The workspace follows five fundamental principles:

1. Minimal kernel: foundational crates contain only the abstractions required by higher layers.
2. Single responsibility: each crate owns a coherent domain and its associated invariants.
3. Directed dependencies: dependency relationships form a directed acyclic graph.
4. Static composition: implementations are assembled through concrete types and narrow traits, preferring static dispatch when appropriate.
5. Verifiable boundaries: architectural rules must be enforceable through source review, automated checks, and tests.

This document describes the intended architecture. It does not establish that the existing implementation conforms to it.

## 2. Normative Language

The terms MUST, MUST NOT, REQUIRED, SHOULD, SHOULD NOT, and MAY express the strength of each requirement.

* MUST / MUST NOT: mandatory architectural constraint.
* SHOULD / SHOULD NOT: default rule that may be overridden only with a documented justification.
* MAY: permitted design choice.

A deviation from a mandatory rule requires an explicit architectural decision and a documented exception approved through the project’s review process. An exception does not silently change the rule for other crates.

## 3. Workspace Inventory

The workspace contains the following planned crates.

| Crate | Primary responsibility |
| --- | --- |
| lib | Shared foundational types and primitives |
| tensor | Tensor metadata, storage, shapes, layouts, and views |
| api | Backend, device, capability, and kernel contracts |
| cpu | CPU execution and numerical kernels |
| autograd | Automatic differentiation and gradient propagation |
| nn | Neural network modules and parameters |
| models | Concrete model architectures |
| tokenizer | Tokenization, vocabulary, decoding, and chat templates |
| data | Dataset abstractions, records, sampling, and batching |
| format | External model formats and structural validation |
| checkpoint | Model and training-state persistence |
| optim | Optimizers, update rules, and learning-rate schedules |
| context | Execution controls, cancellation, resource budgets, and metrics |
| train | Training workflows, losses, evaluation, and recovery orchestration |
| infer | Autoregressive generation, request lifecycle, and KV Cache |
| cli | Command-line application |
| serve | Optional HTTP application |
| testkit | Test-only fixtures and reference implementations |
| bench | Benchmark executables and measurement orchestration |

These names represent domain boundaries, not a requirement that every future capability must remain in its initially assigned file.

A crate MAY be split when a stable, independently testable responsibility justifies the additional boundary. It MUST NOT be split merely to increase the crate count.

---

## 4. Dependency Architecture

### 4.1 Fundamental Rule

A crate may depend on another crate only when the dependency follows ownership of a required capability or contract.

Dependencies MUST NOT be introduced solely because an implementation detail is convenient to access.

The workspace dependency graph MUST remain acyclic.

A higher-level crate may compose lower-level capabilities. A lower-level crate MUST NOT depend on a higher-level workflow merely to call back into it.

### 4.2 Intended Dependency Graph

The following graph describes the intended dependency structure. It is a design target, not a verified graph of existing Cargo.toml files.

```text
lib
├── tensor
│   └── api
│       └── cpu
├── context
├── tokenizer
├── data
└── format
api ───────────────────────────────┐
tensor ────────────────────────────┤
lib ──────────────────────────────┤
                                   ▼
                                autograd
                                   │
                                   ▼
                                   nn
                                   │
                                   ▼
                                 models
                                   │
                         ┌─────────┴─────────┐
                         ▼                   ▼
                        infer               train
optim ──────────────────────────────── train
data ───────────────────────────────── train
checkpoint ─────────────────────────── train
checkpoint ─────────────────────────── infer
tokenizer ──────────────────────────── infer
models ─────────────────────────────── infer
context ────────────────────────────── infer
context ────────────────────────────── train
infer ──────────────────────────────── cli
train ──────────────────────────────── cli
models ─────────────────────────────── cli
```

The graph omits optional edges and some shared lower-level dependencies to keep it readable. Every actual edge MUST be listed and justified in the crate-level dependency inventory.

The diagram is intentionally conservative in its layering, but the exact relationships between autograd, nn, models, infer, and train MUST be validated against the public APIs and implementation requirements before enforcement.

### 4.3 Mandatory Dependency Constraints

The following rules are mandatory:

1. lib MUST NOT depend on any other workspace crate.
2. tensor MUST NOT depend on model, training, inference, or application crates.
3. api MUST NOT depend on concrete backend implementations such as cpu.
4. cpu MUST NOT be required by api to define its contracts.
5. models MUST NOT depend on infer, train, cli, or serve.
6. infer MUST NOT depend on train to perform inference.
7. train MUST NOT depend on infer to perform training.
8. cli and serve MUST NOT be dependencies of production compute crates.
9. testkit MUST NOT enter the normal production dependency closure.
10. bench MUST NOT be required by production runtime crates.
11. No crate may introduce a dependency cycle, including a cycle involving optional features.
12. No crate may bypass an established boundary by importing another crate’s private implementation through an alternative path or duplicated internal code.

Development dependencies are not automatically exempt from architecture review. A development-only edge may be permitted when it is necessary for tests or benchmarks, but it MUST NOT become a production dependency.

### 4.4 When Two Crates Need Each Other

A bidirectional requirement is an architectural warning, not an automatic reason to create a new shared crate.

Before extracting a shared abstraction:

1. Identify the two responsibilities that appear to require each other.
2. Determine which crate owns the domain invariant.
3. Identify the smallest stable contract needed by both sides.
4. Check whether an existing lower-level crate can own that contract.
5. Extract a new crate only if the contract has a coherent independent purpose.
6. Revalidate the dependency graph and public API after the change.

Do not create a universal common, core, or shared crate that accumulates unrelated types and behavior.

---

## 5. Crate Contracts

Each crate MUST define its public responsibilities, dependencies, error behavior, and exclusions. The following contracts are authoritative design requirements for the planned workspace.

### 5.1 lib — Foundational Types

Owns:

* Shared identifiers such as TokenId and RequestId.
* Foundational configuration types and validation primitives.
* Resource-limit types.
* Precision descriptors.
* Reproducible seed representations.
* Minimal cross-crate error primitives where a shared abstraction is demonstrably necessary.

MUST NOT own:

* Tensor computation.
* Model architecture.
* Device execution.
* Request scheduling.
* Serialization workflows.
* Application configuration parsing that belongs to a specific application.

Dependencies: Rust standard library by default; external dependencies require explicit justification.

Acceptance criteria:

* Public types have clear invariants.
* Invalid values are rejected where required.
* No application-specific concepts leak into foundational APIs.
* No workspace dependency is introduced.

### 5.2 tensor — Tensor Representation

Owns:

* Shapes and dimension validation.
* Data types and layout metadata.
* Storage abstractions.
* Tensor ownership and views.
* Shape arithmetic and relevant storage-size calculations.

MUST NOT own:

* Model layers.
* Autograd graph semantics.
* Concrete CPU algorithms.
* Generation state.
* HTTP or CLI behavior.

Dependencies: lib and narrowly justified low-level dependencies.

Acceptance criteria:

* Shape calculations use checked arithmetic where overflow is possible.
* Storage ownership and view lifetimes are explicit.
* Invalid layouts and incompatible shapes produce defined errors.
* Tensor metadata does not silently imply unsupported device or storage capabilities.

### 5.3 api — Execution Contracts

Owns:

* Backend interfaces.
* Device descriptions.
* Capability declarations.
* Kernel invocation contracts.

MUST NOT own:

* CPU kernel implementations.
* Model-specific orchestration.
* Application-level scheduling.
* Request-specific mutable state.

Dependencies: lib and, if required by the public contract, tensor.

The dependency relationship between api and tensor MUST be selected explicitly. If kernel signatures directly accept tensors, api -> tensor is acceptable. If not, the two crates SHOULD remain independently composable.

Acceptance criteria:

* Traits describe stable observable behavior.
* Unsupported operations have defined failure semantics.
* Backend selection does not require a reverse dependency.
* Traits do not combine unrelated responsibilities.

### 5.4 cpu — CPU Backend

Owns:

* CPU kernel implementations.
* Execution dispatch within the CPU backend.
* CPU-specific numerical strategies.
* Validated SIMD implementations.
* CPU-specific capability reporting.

MUST NOT own:

* Public backend contracts.
* Model weight lifecycle.
* Request management.
* Training orchestration.

Dependencies: api, tensor, and lib, subject to the selected contract design.

Acceptance criteria:

* Operator behavior matches the documented contract.
* Numerical correctness is tested against an independent reference where practical.
* No unsafe code is introduced.
* New operators include boundary tests, numerical tests, and at least one benchmark.

### 5.5 autograd — Automatic Differentiation

Owns:

* Computational graph representation.
* Differentiation nodes and variables.
* Gradient storage and accumulation semantics.
* Backward traversal.
* Gradient suppression or no-gradient execution semantics.

MUST NOT own:

* Model-specific architecture.
* Optimizer update policy.
* Training-loop scheduling.
* Concrete application workflows.

Dependencies: tensor, lib, and narrowly defined compute contracts as required.

Acceptance criteria:

* Graph ownership and lifetime behavior are explicit.
* Gradient accumulation and graph reuse semantics are documented.
* Backward operations are tested against analytical or numerical references.
* Invalid graph states fail through defined error paths.

### 5.6 nn — Neural Network Components

Owns:

* Parameter and module abstractions.
* Linear and embedding layers.
* Normalization layers.
* Rotary position transformations.
* Attention and MLP components.
* Module-level initialization contracts.

MUST NOT own:

* Concrete full-model architecture registration.
* Training-loop control.
* HTTP requests.
* Per-request generation state.

Dependencies: tensor, autograd, api, and lib, as justified by the implementation.

Acceptance criteria:

* Parameter ownership and gradient semantics are explicit.
* Layer contracts specify expected shapes and supported data types.
* Initializer behavior is reproducible when supplied with an explicit seed.
* Layer tests cover numerical correctness and invalid configurations.

### 5.7 models — Concrete Architectures

Owns:

* Model architecture configurations.
* Model-specific layer composition.
* Model weight structures and immutable weight access.
* Model registry and supported architecture metadata.
* Causal language model interfaces.

MUST NOT own:

* Request-level KV Cache state.
* Sampling state.
* HTTP or CLI logic.
* Training orchestration.
* Backend-specific kernel implementations.

Dependencies: nn, tensor, api, and lib, plus explicitly justified lower-level capabilities.

Acceptance criteria:

* Architecture configuration is validated.
* Weight shapes and names are checked.
* Shared model parameters are not mutated to represent individual requests.
* Model forward behavior is covered by deterministic tests.

### 5.8 tokenizer — Text and Token Conversion

Owns:

* Vocabulary representation.
* Encoding and decoding.
* Byte-pair encoding where applicable.
* Special-token handling.
* Chat-template processing.

MUST NOT own:

* Tensor execution.
* Model weights.
* KV Cache.
* Request scheduling.

Dependencies: lib and narrowly justified text-processing dependencies.

Acceptance criteria:

* Encoding and decoding behavior is documented.
* Special-token behavior is tested.
* Malformed input and unsupported templates produce defined errors.
* Token IDs are validated against the relevant vocabulary contract.

### 5.9 data — Dataset and Batching

Owns:

* Dataset interfaces.
* Records and text-dataset representations.
* Packing and batching.
* Sampling.
* Data-loader lifecycle and error propagation.

MUST NOT own:

* Model forward execution.
* Optimizer updates.
* Inference request state.
* Checkpoint file formats.

Dependencies: lib and only the data representations required by its public API.

Acceptance criteria:

* Sampling reproducibility is specified.
* Batch shape and boundary behavior are tested.
* Data-loader failure and termination semantics are explicit.
* Resource limits are applied where input size or buffering can grow without bound.

### 5.10 format — External Format Validation

Owns:

* External model configuration parsing.
* Safetensors format handling.
* Tensor metadata validation.
* Structural validation of external model artifacts.

MUST NOT own:

* Model execution.
* Optimizer state transitions.
* Request scheduling.
* Application-specific command handling.

Dependencies: lib, tensor where required, and justified format-parsing dependencies.

Acceptance criteria:

* File and metadata validation is explicit.
* Malformed and truncated input is handled without runtime panics.
* Sizes, offsets, and shape calculations are checked.
* Unsupported formats and features are reported explicitly.

### 5.11 checkpoint — Persistence

Owns:

* Checkpoint load and save workflows.
* Manifest representation.
* Training-state persistence.
* Atomic-write behavior where supported.
* Compatibility validation for persisted state.

MUST NOT own:

* The training loop.
* The optimizer’s mathematical update rules.
* The inference scheduler.
* Application-level HTTP behavior.

Dependencies: lib, tensor and format only when required by the selected persistence design.

Acceptance criteria:

* Incomplete writes do not silently appear as valid checkpoints.
* Compatibility failures are explicit.
* Load and save operations are tested through round-trip and failure scenarios.
* Atomicity guarantees are documented according to the actual filesystem operations used.

### 5.12 optim — Optimization Algorithms

Owns:

* Optimizer contracts.
* AdamW and SGD implementations.
* Gradient clipping.
* Learning-rate schedules.
* Optimizer-specific mutable state.

MUST NOT own:

* Model architecture.
* Dataset loading.
* Checkpoint file encoding.
* Training-loop orchestration.

Dependencies: tensor, nn or a smaller parameter/gradient abstraction, and lib.

Acceptance criteria:

* Update equations are documented.
* Optimizer state is distinct from model architecture.
* Numerical tests cover representative updates and boundary conditions.
* State restoration is compatible with the checkpoint contract.

### 5.13 context — Execution Controls

Owns:

* Execution configuration.
* Cancellation contracts.
* Memory-budget abstractions.
* Worker coordination primitives where required.
* Metrics interfaces.

MUST NOT own:

* Model architecture.
* Domain-specific sampling algorithms.
* A universal container for unrelated application state.
* Hidden global mutable state used by all crates.

Dependencies: lib and minimal concurrency or instrumentation dependencies where justified.

Acceptance criteria:

* Cancellation semantics are documented.
* Budget ownership and concurrent accounting rules are explicit.
* Metrics do not impose unnecessary coupling on foundational crates.
* Resource exhaustion and cancellation are distinguishable where callers need that distinction.

### 5.14 train — Training Workflows

Owns:

* Training-loop orchestration.
* Training configuration.
* Loss calculation and workflow integration.
* Gradient accumulation scheduling.
* Evaluation workflows.
* Resume orchestration.

MUST NOT own:

* Low-level CPU kernels.
* Tensor storage implementation.
* Optimizer mathematical internals.
* External checkpoint encoding.

Dependencies: models, nn, autograd, optim, data, context, checkpoint, and lib as required.

Acceptance criteria:

* Training-step behavior is independently testable.
* Loss and gradient propagation are validated.
* Failure and cancellation do not silently advance training state.
* Resume behavior is tested against persisted state.

### 5.15 infer — Autoregressive Inference

Owns:

* Generation engine orchestration.
* Request parameters and lifecycle.
* Prefill and decode workflows.
* KV Cache allocation and state transitions.
* Sampling and stop conditions.
* Output streaming contracts.

MUST NOT own:

* Shared model weight mutation for request-specific behavior.
* CPU kernel implementations.
* External checkpoint encoding.
* HTTP transport implementation.

Dependencies: models, api, tensor, context, tokenizer where needed, and lib.

Acceptance criteria:

* Request-scoped mutable state has an explicit owner.
* Independent requests cannot accidentally share mutable KV Cache state.
* Cancellation, errors, and completion release resources correctly.
* Generation behavior is tested deterministically where possible.
* Concurrency tests verify request isolation.

### 5.16 cli — Command-Line Application

Owns:

* Argument parsing.
* User-facing command dispatch.
* Human-readable output and exit behavior.
* Application-level configuration selection.

MUST NOT own:

* Tensor algorithms.
* Model architecture internals.
* Duplicate training or inference engines.

Dependencies: Relevant workflow crates and lib.

Acceptance criteria:

* Invalid arguments produce useful diagnostics.
* Commands delegate to library workflows.
* Application errors are mapped to meaningful exit behavior.
* Business logic is not duplicated merely to support a CLI command.

### 5.17 serve — Optional HTTP Application

Owns:

* HTTP routes and request/response mapping.
* Transport-level validation.
* Service integration.
* Request admission and transport cancellation mapping.

MUST NOT own:

* Model kernel implementation.
* Model weight definitions.
* Duplicate generation algorithms.

Dependencies: infer, relevant model-loading capabilities, context, and justified HTTP dependencies.

Acceptance criteria:

* Transport errors are separated from model execution errors.
* Request lifecycle and cancellation mapping are documented.
* Concurrent requests preserve isolation.
* Resource limits are enforced before expensive work where practical.

### 5.18 testkit — Test Infrastructure

Owns:

* Reference implementations.
* Tiny test models.
* Fixtures and deterministic test data.
* Shared assertion helpers.

MUST NOT be a production dependency.

It MAY depend on production crates needed to construct fixtures.

Acceptance criteria:

* Test utilities do not hide correctness failures.
* Reference implementations remain independent enough to detect errors in optimized implementations.
* Fixtures are deterministic when reproducibility is required.
* Test-only dependencies remain isolated from production builds.

### 5.19 bench — Benchmark Executables

Owns:

* Benchmark workload setup.
* Benchmark execution and result reporting.
* Reproducibility metadata.
* Comparisons against recorded baselines where supported.

MUST NOT own:

* Production operator implementations.
* Alternative implementations that silently diverge from production behavior.
* Claims of improvement unsupported by comparable measurements.

Dependencies: The production APIs being measured and justified benchmarking dependencies.

Acceptance criteria:

* Benchmarks exercise the intended production code path.
* Workload dimensions and measurement conditions are recorded.
* Correctness checks prevent benchmarking invalid outputs.
* Performance claims identify the comparison baseline and relevant limitations.

---

## 6. Public API Rules

### 6.1 Default Visibility

Every item begins with the narrowest practical visibility.

* Use private functions and types by default.
* Use pub(crate) for crate-internal collaboration.
* Use public types only when they are part of a required cross-crate contract.
* Re-export selected public APIs deliberately rather than exposing every module.
* Avoid making implementation modules public merely to simplify internal imports.

Public visibility creates a compatibility obligation. It must be justified by an actual caller or a documented extension point.

### 6.2 Trait Design

Traits MUST represent coherent capabilities.

A trait SHOULD define:

* The responsibility it abstracts.
* Required invariants.
* Error behavior.
* Ownership and borrowing expectations.
* Supported inputs and outputs.
* Relevant concurrency guarantees.

Traits SHOULD NOT combine unrelated capabilities simply because one implementation currently supports all of them.

Prefer generic parameters and associated types when they improve static composition and preserve a manageable API.

Do not introduce a trait for every concrete type. Abstraction is justified when it expresses a meaningful contract or enables a required form of composition.

### 6.3 Error Ownership

Each crate owns the errors associated with its responsibilities.

Lower-level crates MUST NOT depend on application-specific error enums.

Errors crossing crate boundaries SHOULD preserve useful source information and context.

A shared error type MAY exist when the semantics are genuinely shared. It MUST NOT become a dumping ground for unrelated failures.

### 6.4 Feature Flags

Feature flags MUST NOT be used to conceal dependency cycles or bypass architecture rules.

Optional dependencies MUST be reviewed under the configurations that enable them.

Where a crate supports multiple features, the project SHOULD test relevant minimal and combined feature configurations.

---

## 7. Cross-Crate Contract Rules

### 7.1 Tensor and API

The project MUST document whether execution contracts accept concrete Tensor values or more general storage and view abstractions.

The decision must account for:

* Shape and layout semantics.
* Ownership and borrowing.
* Device compatibility.
* Output allocation.
* Error reporting.
* Future backend requirements.

Do not introduce an additional tensor abstraction solely to avoid a dependency unless it materially improves separation or enables a required implementation.

### 7.2 Autograd and NN

The project MUST define who owns graph nodes, parameter storage, and gradient buffers.

Layer APIs must not hide unexpected graph mutation or retain resources beyond their documented lifetime.

If nn depends on autograd, that direction must remain explicit. If a lower-level parameter abstraction is needed by both, it should be extracted only when the shared ownership and semantics are clear.

### 7.3 Models and Inference

Model architecture and model weights are separate from request execution.

The model API MUST NOT require mutable request state to be stored inside shared weight objects.

Inference-specific cache state belongs to the request or an explicitly defined cache manager with documented ownership.

Any shared cache manager must enforce request isolation through explicit keys, ownership rules, and synchronization.

### 7.4 Training and Checkpoints

train coordinates save and restore operations; checkpoint owns persistence semantics.

The checkpoint crate MUST NOT depend on the training loop to interpret the low-level persistence format.

Compatibility checks must distinguish structural incompatibility from recoverable I/O failures.

### 7.5 Application Boundaries

cli and serve are adapters around reusable library workflows.

Application-specific input and output formats should be translated at the application boundary rather than propagated throughout the compute stack.

---

## 8. Architecture Change Procedure

Any change to crate ownership or dependency direction MUST follow this procedure.

Step 1: Establish the Existing State

Read the relevant manifests, source code, tests, and public APIs.

Record the current dependency path and the reason the existing boundary is insufficient.

Step 2: State the Problem

Describe the specific coupling, duplication, or ownership ambiguity.

Do not justify an architectural change solely by saying that it is cleaner, more extensible, or more future-proof.

Step 3: Identify the Smallest Valid Change

Consider, in order:

1. Reusing an existing public contract.
2. Narrowing an existing interface.
3. Moving ownership to the correct existing crate.
4. Extracting a minimal shared abstraction.
5. Introducing a new crate only when independent responsibility justifies it.

Step 4: Define Acceptance Criteria

Specify:

* Intended dependency changes.
* Public API changes.
* Required tests.
* Compatibility implications.
* Error behavior.
* Resource-lifecycle implications.

Step 5: Implement and Validate

Make one independently testable change.

Run relevant checks and inspect the resulting dependency graph.

Step 6: Document the Outcome

Record what changed, what was verified, what remains unverified, and any remaining limitations.

Significant architectural decisions SHOULD be recorded under docs/decision-records/.

---

## 9. Automated Architecture Verification

Architecture rules SHOULD be checked in continuous integration instead of relying exclusively on review.

### 9.1 Dependency Inventory

Generate the workspace dependency graph from Cargo metadata.

The checker must distinguish:

* Normal dependencies.
* Build dependencies.
* Development dependencies.
* Optional dependencies and feature-enabled edges.

It must also identify workspace package relationships and report the complete path for every forbidden dependency.

### 9.2 Cycle Detection

The checker MUST reject dependency cycles in the relevant dependency graph.

Cycles MUST be checked under relevant optional-feature configurations, not only the default build.

### 9.3 Forbidden Edges

The initial policy MUST reject at least:

* lib depending on any other workspace crate.
* api depending on cpu.
* models depending on infer, train, cli, or serve.
* infer depending on train.
* train depending on infer.
* Production dependencies on testkit.
* Production compute crates depending on cli or serve.

Any additional rule must be grounded in the final approved dependency graph.

### 9.4 Public API Review

Public API changes SHOULD be reviewed for:

* Unnecessary public visibility.
* Leaked implementation details.
* Unstable or ambiguous ownership.
* Oversized traits.
* Inconsistent error semantics.
* Accidental coupling between layers.

Automated tools can identify some issues, but they do not replace semantic review.

### 9.5 Suggested Baseline Commands

```sh
cargo metadata --workspace --format-version 1
cargo tree --workspace
cargo check --workspace --all-targets
cargo test --workspace
```

These commands provide useful baseline information but do not, by themselves, enforce every architecture rule in this document.

The repository SHOULD add a dedicated architecture-check script or equivalent CI job that evaluates the actual dependency graph against an explicit allowlist and denylist.

The check must fail with an actionable explanation of the offending dependency path.

---

## 10. Required Documentation Per Crate

Each crate SHOULD maintain a concise crate-level README or module documentation covering:

1. Purpose and scope.
2. Responsibilities owned by the crate.
3. Responsibilities explicitly excluded.
4. Public APIs and important invariants.
5. Allowed workspace dependencies.
6. Error and resource-lifecycle semantics.
7. Relevant tests and validation commands.
8. Known limitations.

Crate documentation MUST agree with this document.

If a crate’s responsibility changes, update the documentation and dependency rules in the same independently reviewable stage.

Do not copy this entire document into every crate. Local documentation should reference the authoritative rules and describe only crate-specific details.

---

## 11. Definition of Conformance

A crate is considered conformant only when the relevant conditions have been verified.

Structural conformance

* Responsibilities are clearly defined.
* Actual dependencies respect the approved graph.
* No prohibited dependency cycle exists.
* Public interfaces are appropriately narrow.

Behavioral conformance

* Errors follow the documented contract.
* Resource ownership and lifecycle are explicit.
* Shared and request-scoped state are separated.
* Relevant correctness and isolation tests pass.

Process conformance

* Changes have a documented objective.
* Required validation has actually been executed.
* Results and limitations are reported accurately.
* Architectural exceptions have documented justification.

Conformance status MUST be recorded as one of:

* Proposed: the rule has been specified but not verified.
* Verified: source inspection and relevant checks support the rule.
* Non-conforming: the implementation violates the rule.
* Blocked: verification requires unavailable source code, tools, or environment.

A document’s existence is not evidence of implementation conformance.

---

## 12. Final Rule

The purpose of crate boundaries is not to maximize abstraction or minimize the number of dependencies at any cost.

The purpose is to make responsibility, ownership, and failure behavior explicit while preserving a small, composable execution kernel.

When two designs satisfy the required correctness and safety properties, prefer the one with:

* Fewer unnecessary dependency edges.
* Smaller public interfaces.
* Clearer ownership.
* Less duplicated policy.
* Easier independent testing.
* More direct failure propagation.
* Better evidence for its correctness and performance.

Every architectural boundary must justify the complexity it introduces.

Every exception must be explicit.

Every claimed invariant must be verifiable.