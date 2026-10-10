# Public API Design and Evolution

Status: Proposed
Scope: Rust workspace public interfaces, cross-crate interfaces, and externally consumable APIs across all workspace crates and public-facing modules
Audience: Engineers, reviewers, maintainers, and AI coding agents

---

## 1. Purpose

This document defines the rules for designing, exposing, documenting, testing, and evolving Rust APIs within the llm workspace.

A public API is a long-term engineering commitment. Every additional public type, trait, function, method, field, generic parameter, feature flag, and behavioral guarantee increases the number of constraints that future implementations must preserve.

The project therefore prioritizes:

1. Minimal public surface area.
2. Explicit ownership and error behavior.
3. Composition over inheritance-like abstractions.
4. Static dispatch by default.
5. Clear separation between stable interfaces and implementation details.
6. Compatibility-aware evolution.
7. Testable contracts rather than undocumented assumptions.
8. No unnecessary abstraction or speculative extensibility.

Public APIs MUST be designed intentionally. Making a type or module public merely because another crate currently needs access is not sufficient justification for exposing it as part of the supported external interface.

## 2. Normative Language

The terms MUST, MUST NOT, REQUIRED, SHOULD, SHOULD NOT, and MAY indicate the strength of each requirement.

* MUST / MUST NOT: Mandatory engineering constraint.
* SHOULD / SHOULD NOT: Default rule; exceptions require an explicit technical justification.
* MAY: Optional behavior appropriate to the implementation.

A documented exception MUST identify the reason, affected interface, compatibility implications, and validation method.

## 3. API Categories

Every public interface SHOULD have an identifiable stability category.

### 3.1 Internal implementation interface

An interface used only inside one crate SHOULD remain private or pub(crate).

Examples include:

* Internal helper functions.
* Temporary computation structures.
* Private kernel implementation details.
* Internal caching mechanisms.
* Intermediate representations that are not part of a cross-crate contract.

These interfaces SHOULD be changed freely when implementation requirements change.

### 3.2 Workspace interface

A workspace interface is exposed primarily to other crates within this repository.

Examples include:

* Tensor primitives consumed by CPU kernels.
* Operator contracts consumed by autograd.
* Model construction interfaces consumed by inference.
* Checkpoint abstractions consumed by training and inference.

Workspace interfaces SHOULD be as narrow as possible. A dependency between workspace crates does not automatically justify a stable, externally supported API.

When an interface is required across crate boundaries, prefer exposing the smallest set of types, traits, and methods that satisfies the actual consumer.

### 3.3 External public interface

An external public interface is intentionally supported for consumers outside the workspace.

Before treating an interface as externally stable, maintainers MUST determine:

* Whether the containing crate is intended for external consumption.
* Whether the interface is covered by compatibility guarantees.
* Which release and versioning policy applies.
* How breaking changes will be communicated.
* Whether the interface has sufficient tests and documentation.

Do not claim that an interface is stable, semver-compatible, or production-supported unless the project has explicitly adopted the corresponding commitment.

### 3.4 Experimental interface

Experimental interfaces MAY be exposed when experimentation is necessary, but their status and limitations MUST be clear.

Experimental status MUST NOT be used as a substitute for sound design, error handling, or tests.

Before an experimental interface becomes stable, review its naming, type constraints, ownership semantics, error behavior, compatibility implications, and real consumer requirements.

---

## 4. Visibility and Public Surface Area

### 4.1 Private by default

Rust visibility MUST follow the principle of least exposure.

Prefer, in order:

1. Private items.
2. pub(super) where a parent-module boundary is sufficient.
3. pub(crate) for crate-wide implementation needs.
4. Restricted visibility such as pub(in path) when appropriate.
5. pub only when a real consumer requires access.

Developers MUST NOT expose internal types or add public items solely to make
tests convenient. Every test reaches the crate through its public interface at
the repository root; a test-only hook is a contract defect, not a reason to
widen visibility (see `docs/decision-records/blackboxtesting.md`).

### 4.2 Explicit public exports

Each crate SHOULD make its intended public interface easy to discover.

A crate MAY use a small set of public facade modules or root-level re-exports to present a coherent interface. Re-exports MUST be deliberate because they can create additional compatibility commitments.

Do not re-export every internal module, dependency type, or implementation helper.

For example, a crate MAY expose:

```rust
pub use tensor::Tensor;
pub use tensor::TensorError;
```

only if those types genuinely belong to its intended public contract. This example illustrates an API design pattern; it does not assert that these exports already exist in the workspace.

### 4.3 Avoid accidental public APIs

Before merging a change that introduces pub, review whether the new visibility is necessary.

The following are not sufficient reasons on their own:

* The implementation is easier to write with broad visibility.
* A future consumer might need the item.
* A dependency currently makes the type accessible.
* A test can access the item more conveniently.
* A public interface appears more complete with additional methods.

Prefer the smallest interface that supports demonstrated use cases.

### 4.4 Public fields

Public fields expose representation and restrict future implementation changes.

Public structs SHOULD prefer private fields with explicit constructors and accessors when invariants or future representation changes matter.

Public fields MAY be appropriate for simple value types when:

* The fields are independent data.
* Their invariants are trivial.
* Direct construction and mutation are intended behavior.
* Exposing the representation does not create an unreasonable compatibility burden.

Do not add setters merely for symmetry. Each method should express a real operation or supported use case.

---

## 5. API Design Principles

### 5.1 Design around real capabilities

Interfaces SHOULD represent meaningful capabilities or domain concepts.

Good interfaces make it clear:

* What operation is available.
* Which inputs are valid.
* What state is changed.
* What ownership is transferred or borrowed.
* Which failures are possible.
* Which behavior consumers may rely on.

Avoid APIs whose purpose is unclear without reading their implementation.

### 5.2 Minimize conceptual complexity

A small number of explicit, composable interfaces is preferable to a large framework of generic abstractions.

Do not introduce an abstraction solely to eliminate a few repeated lines of code.

Before creating a new public abstraction, answer:

1. Which real implementations or consumers require it?
2. Which behavior does it standardize?
3. What correctness constraint does it enforce?
4. Can the same requirement be met by a private helper or existing interface?
5. What future compatibility obligations will it introduce?

If the benefits are speculative, keep the abstraction private until a concrete need emerges.

### 5.3 Avoid stringly typed APIs

Do not represent structured choices, operation modes, or domain states as arbitrary strings when a type can express the intended contract.

Prefer enums or dedicated types for finite, meaningful alternatives.

For example:

```rust
pub enum DeviceKind {
    Cpu,
}
```

is preferable to accepting arbitrary strings such as "cpu" when the API only supports a fixed set of device kinds.

Do not create a one-variant enum merely to make an API appear more strongly typed unless the abstraction has a concrete purpose.

Strings remain appropriate for genuinely free-form values, such as file paths, labels, or user-provided text.

### 5.4 Avoid hidden global behavior

Public APIs SHOULD NOT depend on undocumented global mutable state, implicit environment configuration, or invisible request-specific state.

Dependencies and configuration that materially affect behavior SHOULD be explicit in function parameters, construction options, or clearly documented environment interfaces.

A caller should be able to understand the important inputs to an operation from its signature and documentation.

---

## 6. Trait Design

Traits are important composition boundaries in Rust. They are also long-term contracts that can become expensive to change.

### 6.1 Define traits around cohesive capabilities

A trait SHOULD describe one coherent capability.

Avoid large traits that combine unrelated responsibilities, such as:

* Tensor storage and neural-network optimization.
* Model loading and HTTP request handling.
* Checkpoint serialization and training-loop scheduling.
* Numerical computation and logging infrastructure.

Separate responsibilities when they have different consumers, lifecycles, failure modes, or implementation requirements.

However, do not split every method into a separate trait. Excessive fragmentation can make ordinary operations harder to discover and compose.

The correct boundary is the smallest coherent capability that has a real reason to exist independently.

### 6.2 Prefer static dispatch

Static dispatch SHOULD be the default for core computation and performance-sensitive paths.

Generic parameters and concrete types allow the compiler to resolve implementations without requiring dynamic dispatch.

Use trait objects such as dyn Trait only when runtime polymorphism provides a demonstrated benefit, such as:

* A runtime-selected collection of implementations.
* Plugin-like extension points.
* Heterogeneous objects behind one interface.
* A meaningful need to reduce generic type propagation.

Do not introduce dynamic dispatch merely because traits exist.

Conversely, do not prohibit dynamic dispatch when runtime selection is an actual requirement. The choice MUST follow the use case rather than an unconditional preference for either mechanism.

### 6.3 Avoid speculative trait hierarchies

Do not create multiple layers of marker traits, supertraits, associated traits, or blanket implementations without a clear need.

Every trait layer should communicate a meaningful capability, constraint, or contract.

Blanket implementations MUST be reviewed carefully because they can:

* Prevent downstream implementations from being added.
* Introduce method-resolution ambiguity.
* Increase trait-bound complexity.
* Create unexpected interactions with future implementations.

Avoid blanket implementations whose only purpose is to save trivial forwarding code.

### 6.4 Use associated types intentionally

Associated types are appropriate when an implementation naturally determines one type for a capability.

Generic type parameters are more appropriate when callers need to select or vary the type independently.

Choose the model that produces the clearest contract for actual consumers. Do not add associated types or generic parameters solely to maximize theoretical flexibility.

### 6.5 Keep trait methods coherent

Each required trait method MUST have a clear semantic contract.

A trait SHOULD define:

* Preconditions.
* Input and output semantics.
* Error behavior.
* State and mutation guarantees.
* Relevant numerical guarantees.
* Thread-safety requirements, if any.
* Whether repeated calls are expected to be deterministic.

Default methods MAY be provided when the default behavior is correct for every conforming implementation.

Do not provide a default implementation that silently weakens correctness or performance guarantees.

### 6.6 Avoid unnecessary trait bounds

Add Send, Sync, 'static, Clone, or other bounds only when the contract requires them.

Unnecessary bounds can prevent valid implementations from being used and can propagate implementation details into unrelated APIs.

In particular:

* Send and Sync SHOULD reflect real cross-thread requirements.
* 'static MUST NOT be added merely to simplify a local compiler error.
* Clone MUST NOT be required when borrowing, ownership transfer, or a different explicit operation is sufficient.
* Bounds introduced for asynchronous execution MUST be justified by the actual execution model.

### 6.7 Object safety

When dynamic dispatch is a supported requirement, verify that the trait can be used as a trait object and that its method signatures are compatible with that requirement.

Do not distort an otherwise appropriate static-dispatch interface solely to make it object-safe when no dynamic consumer exists.

If both static and dynamic use cases are necessary, consider a deliberately designed separation rather than adding complexity without a consumer.

---

## 7. Function Signatures and Type Design

### 7.1 Make ownership explicit

Function signatures SHOULD communicate ownership and borrowing requirements.

Use:

* &T for read-only borrowing.
* &mut T for exclusive mutable borrowing.
* Owned T when the callee needs ownership, may retain the value, or should consume it.
* Cow<'a, T> or similar abstractions only when avoiding a meaningful allocation or ownership transfer justifies the additional complexity.

Do not clone large tensors, parameter arrays, model data, or other expensive values merely to simplify a signature.

Do not introduce complex lifetime relationships unless they express a real relationship between input and output data.

### 7.2 Return values should express outcomes

Functions SHOULD return the smallest useful result that communicates the operation’s outcome.

Prefer explicit return types over:

* Mutating hidden state to deliver results.
* Returning loosely structured tuples that become difficult to interpret.
* Returning opaque collections when a domain type provides meaningful invariants.
* Using sentinel values to represent failure.

For results with multiple semantically distinct fields, prefer a named struct when that improves clarity and maintainability.

### 7.3 Avoid excessive parameter lists

When several parameters represent one cohesive concept, consider a configuration or options type.

Do not create a builder for every function. A builder introduces more API surface and may complicate validation and error handling.

A builder is appropriate when construction has many optional settings, staged configuration, or meaningful validation requirements.

A simple function with a few clear parameters is preferable when a builder adds no meaningful value.

### 7.4 Encode invariants in types where practical

Types SHOULD make invalid states difficult to represent.

Examples include:

* Distinct types for different coordinate systems or units.
* Enums for finite states.
* Validated configuration types for constrained values.
* Newtypes for identifiers that must not be mixed.
* Explicit state transitions when operation ordering matters.

Type-level modeling MUST remain proportionate to the problem. Do not introduce elaborate type machinery when a straightforward runtime check is clearer and equally reliable.

### 7.5 Avoid leaking implementation details

Public signatures SHOULD NOT expose internal data structures, temporary computation buffers, cache layouts, or third-party types unless those details are deliberately part of the contract.

An internal representation change should not require a public API change unless the underlying semantics genuinely change.

---

## 8. Error Contracts

### 8.1 Use explicit error propagation

Fallible public operations MUST expose their failure behavior through an appropriate Result<T, E> return type. Error semantics, the runtime prohibition on panic!/unwrap()/expect(), and the assertion policy are defined by docs/error-handling.md; this section states only the public-signature obligation.

### 8.2 Design meaningful error types

Public error types SHOULD help callers understand and respond to failure.

An error contract SHOULD distinguish materially different failure categories, such as:

* Invalid arguments or configuration.
* Unsupported operations.
* Shape or dimensional incompatibility.
* Numerical failures.
* I/O failures.
* Serialization or deserialization failures.
* Resource exhaustion, where reliably detectable.
* Internal invariant violations.

Do not create a unique public error type for every helper function without a clear benefit.

Use a shared error type when errors belong to the same coherent abstraction. Use dedicated types when consumers need to distinguish failure categories or when the domain requires a clear boundary.

### 8.3 Preserve useful context

When propagating lower-level errors, preserve enough context to identify the failing operation and relevant domain object.

Avoid converting every failure into a generic string if that destroys information callers need for recovery, diagnosis, or testing.

Error messages SHOULD be understandable to developers and MUST NOT be treated as the only stable machine-readable contract.

### 8.4 Do not expose unstable error implementation details

Consumers SHOULD match on documented error categories rather than parsing arbitrary human-readable messages.

If callers need stable classification, expose an enum variant, error kind, or other structured representation.

Error types are part of the public API and MUST be reviewed for compatibility just like ordinary data types.

---

## 9. Numerical and Computational Contracts

Because this workspace includes numerical and machine-learning computation, APIs MUST document behavior that materially affects correctness.

Where applicable, specify:

* Tensor shape and layout requirements.
* Dtype and supported numerical formats.
* Device or execution-backend requirements.
* Broadcasting and reduction semantics.
* Output shape and dtype.
* Accumulation precision.
* Numerical tolerance expectations.
* Behavior for empty inputs and boundary dimensions.
* Handling of NaN, infinity, overflow, and underflow.
* Determinism guarantees and limitations.
* Whether mutation is permitted.
* Whether an operation allocates new storage or may reuse existing storage.

Do not promise bitwise equality across different backends, hardware, compiler configurations, or execution modes unless the project has explicitly established and tested that guarantee.

A public API MUST NOT imply stronger numerical guarantees than its implementations can support.

Performance claims MUST be supported by reproducible measurements. Documentation MUST NOT assert that an API or implementation is faster without appropriate benchmark evidence.

---

## 10. State, Mutability, and Concurrency

### 10.1 Make state transitions visible

Operations that modify state SHOULD communicate that fact through their signatures and documentation.

Avoid APIs that appear read-only but modify externally observable state.

Document whether repeated calls:

* Produce independent outputs.
* Mutate the receiver.
* Consume the receiver.
* Depend on previous calls.
* Require exclusive access.

### 10.2 Separate model state from request state

Model weights and reusable model structures MUST NOT contain request-specific mutable state.

Per-request inputs, temporary activations, inference caches, decoding state, cancellation state, and similar mutable data MUST have an explicit lifecycle appropriate to the operation.

Reusable model objects SHOULD be safe to share according to their documented concurrency contract. If a particular implementation requires exclusive access, that restriction MUST be explicit in its type or API documentation.

Do not add Mutex, RwLock, or atomic state merely to make a type appear thread-safe. Synchronization MUST follow a real concurrency requirement and a defined ownership model.

### 10.3 Document thread-safety guarantees

Implementing Send or Sync is a meaningful API property.

Do not add explicit trait implementations, unsafe implementations, or synchronization mechanisms to satisfy an assumed requirement without analyzing the underlying data and invariants.

This project prohibits unsafe; all API and implementation designs MUST remain within that constraint.

---

## 11. Configuration and Feature Flags

### 11.1 Prefer explicit configuration

Configuration that materially affects behavior SHOULD be visible at construction time or at the operation boundary.

Avoid hidden defaults that make behavior depend on undocumented environment conditions.

Defaults MUST be documented when they affect correctness, numerical behavior, resource usage, or performance.

### 11.2 Keep configuration cohesive

Configuration types SHOULD group related options that share a lifecycle and purpose.

Avoid a single global configuration object that mixes unrelated concerns from tensor execution, training, checkpoint storage, networking, and command-line presentation.

Separate configuration by responsibility and let higher-level components compose the relevant parts.

### 11.3 Feature flags

Feature flags MAY be used for optional capabilities, platform support, or dependency selection when they provide a real benefit.

They SHOULD NOT be introduced merely to create the appearance of modularity.

For each public feature flag, document:

* The capability it enables.
* Any additional dependencies.
* Relevant platform restrictions.
* Interactions with other features.
* Whether it changes public API availability.
* The supported combinations that are tested.

Avoid feature combinations that create undocumented or untestable public API states.

Feature-gated interfaces MUST be checked under the relevant feature configurations before release.

---

## 12. Dependencies and Public Type Leakage

Public APIs SHOULD minimize unnecessary dependence on external crate types.

Exposing a third-party type in a public signature can couple consumers to that dependency’s API and versioning decisions.

Before exposing such a type, determine whether:

1. The external type is the natural domain representation.
2. Consumers need the external type’s full functionality.
3. A small project-owned type would provide a clearer contract.
4. Wrapping the type would add unnecessary conversion or maintenance costs.

Do not wrap every external type automatically. A wrapper is justified when it establishes domain meaning, validates invariants, isolates dependency evolution, or narrows an overly broad interface.

Dependency upgrades MUST include a review of affected public signatures and behavioral contracts.

---

## 13. Documentation Requirements

### 13.1 Public item documentation

Public types, traits, functions, and methods MUST have documentation appropriate to their complexity and intended audience.

Documentation SHOULD explain:

* What the item does.
* When it should be used.
* Preconditions and constraints.
* Ownership and mutation behavior.
* Errors and failure conditions.
* Numerical or determinism guarantees.
* Relevant examples.

Documentation MUST describe observable behavior, not merely restate an item’s name or signature.

### 13.2 Examples and doctests

Public APIs SHOULD include concise examples for nontrivial use cases.

Examples MUST compile when practical and MUST reflect the intended supported usage pattern.

Prefer examples that demonstrate composition across the relevant interface boundary rather than isolated fragments that hide essential setup.

When an example depends on optional features, its feature requirements MUST be clear.

### 13.3 Avoid unsupported promises

Documentation MUST NOT claim:

* Compatibility that has not been adopted as policy.
* Thread safety that has not been established.
* Determinism that has not been verified.
* Numerical precision that implementations cannot guarantee.
* Performance improvements without benchmark evidence.
* Support for backends or formats that have not been validated.

If behavior is implementation-dependent, experimental, or not yet guaranteed, state that limitation explicitly.

---

## 14. Compatibility and API Evolution

### 14.1 Define compatibility policy explicitly

The project MUST distinguish between internal workspace interfaces and externally supported APIs before promising compatibility.

If a crate is published and follows Semantic Versioning, changes to its public API MUST be classified according to the crate’s adopted versioning policy.

Do not assume that every workspace crate is published or that every public item is externally stable.

### 14.2 Potentially breaking changes

The following changes can break source compatibility or alter behavior:

* Removing or renaming a public item.
* Changing a public function’s parameter or return type.
* Adding a required method to a public trait.
* Changing trait bounds.
* Changing ownership or borrowing requirements.
* Adding or removing auto-trait behavior such as Send or Sync.
* Changing a public field’s visibility or type.
* Changing documented numerical, error, or determinism guarantees.
* Changing defaults in a way that alters observable behavior.
* Changing feature requirements for public items.
* Narrowing the set of valid inputs or supported use cases.

The impact of each change MUST be assessed in the context of the actual compatibility contract.

### 14.3 Additive changes are not always harmless

Adding a public item is usually less disruptive than removing one, but additions can still create compatibility problems.

Examples include:

* Adding a required trait method.
* Adding a new enum variant when downstream exhaustive matching is permitted.
* Introducing blanket implementations that conflict with consumer implementations.
* Adding trait bounds to an existing generic API.
* Adding a method that conflicts with methods commonly defined by consumers.
* Adding a feature requirement that changes which APIs are available.

Compatibility review MUST consider downstream implementations and usage patterns, not only whether existing source files still compile.

### 14.4 Use `#[non_exhaustive]` deliberately

`#[non_exhaustive]` MAY be appropriate when a public struct or enum is intentionally designed to evolve without requiring exhaustive downstream construction or matching.

It SHOULD NOT be added automatically to every public type.

Before using it, consider:

* Whether external consumers need to construct the type directly.
* Whether exhaustive matching is part of the intended API.
* Whether future additions are genuinely likely and meaningful.
* The ergonomic costs imposed on callers.
* Whether the type’s stability category justifies the restriction.

The attribute itself affects downstream usage and MUST be treated as a deliberate API decision.

### 14.5 Deprecation

When an externally supported API needs replacement, deprecation SHOULD provide a migration path when practical.

A deprecation notice SHOULD identify:

* The deprecated item.
* The recommended replacement.
* Any semantic differences.
* The expected migration approach.
* Relevant removal or compatibility policy, if one has been adopted.

Do not deprecate an API without a meaningful reason or replacement strategy when continued support is feasible.

### 14.6 Migration strategy

For significant API changes, provide a migration example or release note when appropriate.

Prefer changes that allow consumers to transition incrementally when doing so does not preserve incorrect behavior or impose disproportionate complexity.

Do not maintain redundant compatibility layers indefinitely without a documented maintenance justification.

---

## 15. API Testing

Public API correctness MUST be validated through tests that exercise its documented contract.

### 15.1 Contract tests

Tests SHOULD cover:

* Valid inputs and expected outputs.
* Invalid inputs and error categories.
* Boundary values.
* Ownership and mutation expectations where observable.
* Numerical semantics and tolerances.
* Determinism guarantees, if applicable.
* Supported configuration and feature combinations.
* Important concurrency guarantees, where applicable.

Tests MUST NOT rely solely on private implementation details when the purpose is to validate a public contract.

### 15.2 External-consumer perspective

For interfaces intended for external use, include tests or examples that exercise the API as a consumer would encounter it.

Such validation SHOULD detect accidental reliance on private visibility, undocumented imports, or implementation-only helper functions.

Where practical, compile documentation examples and test relevant public feature configurations.

### 15.3 Compatibility validation

Before a release that promises API compatibility, validate the intended public surface against the project’s adopted compatibility policy.

Automated API comparison MAY be used where appropriate, but tooling MUST be configured to reflect the actual stability guarantees.

An automated diff is evidence for review, not a substitute for understanding semantic and behavioral changes.

### 15.4 Avoid test-only public expansion

Do not make production interfaces public solely to support tests.

Prefer module-level unit tests, integration tests against supported interfaces, or narrowly scoped test infrastructure.

Testability is important, but it does not automatically justify expanding the public contract.

---

## 16. API Change Workflow

Every meaningful public API change MUST follow a deliberate review process.

Step 1: Identify the consumer

Determine which existing or planned consumer requires the change.

Record whether the consumer is:

* Internal to the crate.
* Another workspace crate.
* An external consumer.
* A test or benchmark only.

If there is no concrete consumer, reconsider whether the API should be exposed now.

Step 2: Review ownership and dependency boundaries

Inspect the owning crate’s responsibilities and its dependency direction.

Confirm that the change does not introduce circular dependencies or make a lower-level crate depend on a higher-level subsystem.

Do not move types between crates simply to make a public signature convenient without considering architectural ownership.

Step 3: Define the contract

Specify:

* Inputs and preconditions.
* Outputs and postconditions.
* Error behavior.
* Ownership and mutation.
* Numerical semantics, where applicable.
* Performance expectations, where applicable.
* Stability category.
* Compatibility implications.

Step 4: Compare alternatives

Consider whether the requirement can be satisfied by:

* A private helper.
* A narrower visibility level.
* An existing trait or type.
* A small extension to an existing interface.
* A new public type or trait.

Choose the least complex option that satisfies the real requirement.

Step 5: Implement the smallest coherent change

Avoid combining unrelated API redesigns with implementation work.

Keep each change focused enough to review, test, and revert independently.

Step 6: Validate the interface

Run relevant unit tests, integration tests, documentation tests, feature checks, and compatibility checks.

New operators or computational APIs MUST also satisfy the project’s operator testing and benchmarking requirements, including boundary tests, numerical tests, and at least one performance benchmark.

Step 7: Review the final diff

Inspect all visibility changes, re-exports, trait bounds, error types, feature declarations, documentation, and generated API surface.

Verify that no internal detail became public accidentally.

Step 8: Record compatibility decisions

For externally supported APIs, record any breaking change, deprecation, migration requirement, or stability-policy decision in the appropriate project documentation or release notes.

Do not claim that compatibility has been verified if the relevant checks were not run.

---

## 17. AI-Assisted API Development

AI-assisted changes MUST follow the same engineering requirements as human-authored changes.

AI tools MUST NOT:

* Make items public merely to resolve a compiler error without reviewing the architectural boundary.
* Invent traits or generic abstractions without a concrete use case.
* Introduce dynamic dispatch without a demonstrated requirement.
* Add unnecessary Send, Sync, 'static, or Clone bounds.
* Replace explicit error propagation with panic!, unwrap(), or expect().
* Introduce unsafe.
* Add hidden mutable global state.
* Change public signatures without evaluating downstream consumers.
* Claim API compatibility, test success, or performance improvement without evidence.
* Treat illustrative examples as verified descriptions of the actual workspace.

Before changing an API, AI-assisted work MUST inspect the relevant crate documentation, dependency relationships, existing implementations, and actual call sites.

After changing an API, it MUST inspect the resulting diff and report which relevant checks were run and which remain outstanding.

When evidence is unavailable, state the limitation instead of assuming the repository conforms to the proposed standard.

---

## 18. Review Checklist

A public API change is ready for review when the following questions have clear answers.

Scope and visibility

* [ ] Is there a concrete consumer for this public interface?
* [ ] Is the selected visibility level the narrowest practical one?
* [ ] Are re-exports intentional?
* [ ] Are implementation details kept private?
* [ ] Is the interface’s stability category clear?

Design

* [ ] Does the API represent a coherent capability?
* [ ] Is the abstraction justified by real requirements?
* [ ] Are function signatures and ownership semantics clear?
* [ ] Are type parameters, trait bounds, and lifetimes necessary?
* [ ] Is static dispatch sufficient, or is dynamic dispatch justified?
* [ ] Are hidden global behavior and unnecessary state avoided?

Correctness

* [ ] Are preconditions and postconditions documented?
* [ ] Are errors explicit and actionable?
* [ ] Are numerical and mutation semantics clear?
* [ ] Are concurrency guarantees accurate?
* [ ] Are boundary cases covered?

Compatibility

* [ ] Have downstream consumers and implementations been considered?
* [ ] Have changes to trait bounds, fields, enum variants, and feature requirements been reviewed?
* [ ] Is deprecation or migration guidance necessary?
* [ ] Are any stability promises supported by actual project policy?

Validation

* [ ] Do relevant tests pass?
* [ ] Are documentation examples accurate and tested where practical?
* [ ] Are relevant feature configurations checked?
* [ ] Are required operator benchmarks present?
* [ ] Has the final diff been reviewed?
* [ ] Are unverified checks and remaining risks reported honestly?

---

## 19. Definition of Done

A public API change is complete only when:

1. The consumer and architectural ownership are understood.
2. The interface exposes no more functionality than necessary.
3. Ownership, errors, mutation, and behavioral contracts are explicit.
4. Relevant numerical and concurrency guarantees are documented.
5. The interface has appropriate documentation and tests.
6. Compatibility implications have been reviewed.
7. Relevant validation has been run and its results reported accurately.
8. No circular dependency or prohibited unsafe code has been introduced.
9. The final diff contains no accidental visibility expansion.
10. Any unresolved limitations are documented.

A public API is not complete merely because the code compiles.

---

## 20. Governing Principle

A public API is a contract, not a convenience mechanism.

Expose only what a real consumer needs. Prefer explicit types, coherent traits, predictable ownership, and meaningful errors. Keep internal implementation details replaceable. Evolve interfaces deliberately, and support every compatibility or performance claim with evidence.

The long-term objective is not to maximize the number of reusable abstractions. It is to make the smallest set of interfaces sufficient for the system to remain correct, composable, maintainable, and adaptable as the implementation evolves.