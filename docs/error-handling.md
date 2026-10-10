# Error Handling and Failure Semantics

Status: Proposed
Scope: All production crates in the llm/ workspace
Audience: Engineers, reviewers, maintainers, and AI coding agents

---

## 1. Purpose

This document defines the error-handling architecture for the llm/ workspace.

The goal is not merely to prevent panics. The goal is to make failure explicit, diagnosable, recoverable when appropriate, and consistent across crate boundaries.

Error handling MUST preserve the distinction between:

* Invalid external input.
* Invalid internal state.
* Unsupported functionality.
* Expected absence of a value.
* I/O and serialization failures.
* Backend execution failures.
* Resource exhaustion.
* Cancellation and timeout.
* Programming defects and violated invariants.

These conditions are not interchangeable. They MUST NOT be collapsed into a generic error, silently converted into success, or handled with arbitrary fallback behavior.

This document complements:

* docs/architecture.md
* docs/crate-boundaries.md
* docs/development-workflow.md
* docs/tensor-layout.md

If a conflict exists, the workspace’s explicit architectural constraints and security requirements take precedence until the documentation is reconciled.

## 2. Normative Language

The terms MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY express engineering requirements.

A MUST-level requirement may be waived only through an explicit, reviewed architectural decision that documents its scope, rationale, risks, and verification strategy.

Code fragments in this document are illustrative. They are not the actual workspace implementation; the real types, variants, and error contracts are owned by the relevant crates.

## 3. Non-Negotiable Rules

1. Production runtime failures MUST be represented and propagated through explicit error-handling paths.
2. Runtime errors MUST NOT be handled by panic!, unwrap(), or expect().
3. An error MUST NOT be silently discarded or converted into a success-shaped value.
4. Expected absence MUST use an appropriate representation, such as Option<T>, rather than an error or fabricated default.
5. Error types MUST express stable semantic categories rather than depend on human-readable strings.
6. Each crate MUST own errors associated with its own abstraction and invariants.
7. Errors MUST be translated only at meaningful abstraction boundaries.
8. Low-level diagnostic information SHOULD be preserved without leaking sensitive data to external callers.
9. Retries MUST be bounded, justified by failure semantics, and compatible with operation idempotency.
10. Cancellation and timeout MUST NOT be treated as ordinary success.
11. Resource limits MUST be checked before operations that could exceed the limits whenever practical.
12. Error paths MUST receive tests appropriate to their risk and complexity.
13. Error handling MUST NOT conceal violated invariants or make invalid states appear valid.
14. New error-handling dependencies MUST be justified and compatible with the workspace’s dependency boundaries.

These rules apply to synchronous and asynchronous code, CPU and accelerator backends, inference, training, serialization, checkpoint operations, and service interfaces.

---

## 4. Error-Handling Model

The workspace SHOULD use a layered error model.

```text
External input / caller
        |
        v
Boundary validation
        |
        v
Domain operation
        |
        v
Crate-local implementation
        |
        v
Backend / filesystem / external dependency
```

Each layer has a different responsibility.

### 4.1 Boundary validation

The boundary validates externally supplied values and translates lower-level failures into errors appropriate for its public contract.

Examples:

* Invalid tensor dimensions.
* Malformed model configuration.
* Unsupported model identifiers.
* Invalid tokenizer input.
* Invalid checkpoint metadata.
* Invalid HTTP request parameters.

Boundary validation MUST reject invalid input before performing expensive or irreversible work whenever practical.

### 4.2 Domain operations

Domain operations express failures in terms of the domain.

Examples:

* A tensor shape is incompatible with an operation.
* A requested model feature is unsupported.
* A checkpoint does not contain the required parameter.
* An optimizer configuration is invalid.
* A requested inference operation exceeds its declared budget.

Domain errors SHOULD remain independent of CLI, HTTP, filesystem presentation, and other application-specific protocols.

### 4.3 Infrastructure and backend operations

Infrastructure layers translate failures from the underlying mechanism into errors meaningful to their callers.

Examples:

* File reads and writes.
* Memory allocation or buffer reservation.
* CPU kernel execution.
* Device initialization.
* Backend synchronization.
* Serialization and deserialization.

Infrastructure errors SHOULD preserve the original source error when doing so is meaningful and safe.

### 4.4 Application boundaries

Application layers decide how an error is presented or acted upon.

Examples:

* A CLI prints a concise diagnostic and exits with a nonzero status.
* An HTTP server maps a validated client-input error to an appropriate client response.
* A training application stops the current run and preserves diagnostic context.
* An inference service returns a request failure without exposing internal paths or implementation details.

Application layers MUST NOT redefine lower-level failure semantics merely to simplify presentation.

---

## 5. Choosing Between Result, Option, and Invariants

### 5.1 Use Result<T, E> for failure

Use Result<T, E> when an operation can fail in a way that the caller may need to distinguish, report, recover from, or test.

```rust
pub fn validate_shape(
    shape: &[usize],
) -> Result<(), ShapeError> {
    // Validate the shape and return a typed error if invalid.
    todo!()
}
```

A real implementation MUST perform the actual validation and MUST NOT leave a production todo!() path.

### 5.2 Use Option<T> for expected absence

Use Option<T> when absence is a legitimate outcome rather than a failure.

Examples:

* An optional configuration field is not provided.
* A lookup does not find a matching entry and absence is part of the API contract.
* A model has no optional bias tensor.
* A caller requests an optional metadata value that is not present.

Do not use Option<T> when it would erase a meaningful failure.

For example, a checkpoint lookup that can fail because the file is corrupt SHOULD NOT convert every error into None.

Bad:

```rust
let metadata = read_checkpoint_metadata(path).ok();
```

This discards the distinction between a missing value and a failed read.

Prefer a contract that explicitly represents both absence and failure, where both are possible:

```rust
fn find_metadata(
    path: &Path,
) -> Result<Option<Metadata>, CheckpointError>;
```

The precise signature SHOULD match the actual storage and lookup semantics.

### 5.3 Do not use fabricated defaults to hide failure

The following patterns are prohibited when they turn a genuine failure into apparent success:

```rust
// Prohibited when the fallback hides a real failure.
let config = load_config(path).unwrap_or_default();
// Prohibited when absence is not a valid substitute for failure.
let tensor = load_tensor(path).ok().flatten();
```

Fallbacks MAY be used only when the fallback is explicitly part of the domain contract and does not conceal an error.

### 5.4 Distinguish invalid input from broken invariants

Invalid external input is normally a recoverable error.

An internal invariant violation may indicate a programming defect or corrupted internal state. It MUST NOT be disguised as ordinary user input failure.

Examples:

| Condition | Expected treatment |
| --- | --- |
| Negative dimension represented by a signed external value | Reject during validation |
| Shape dimensions overflow the supported size calculation | Return a validation or size error |
| Unsupported operation requested by a caller | Return an unsupported-operation error |
| An internal buffer is unexpectedly shorter than a proven invariant requires | Investigate as an invariant violation |
| A file cannot be read | Return an I/O-related error |
| An optional metadata field is absent | Use Option if absence is valid |

Production code MUST NOT use panics as routine control flow for any of these conditions.

---

## 6. Error Ownership and Crate Boundaries

Each crate SHOULD define errors for the operations and invariants it owns.

A crate MUST NOT depend on application-level crates merely to reuse their error types.

The workspace MUST NOT introduce a large shared error enum that imports every subsystem and forces unrelated crates to depend on one another.

### 6.1 Recommended ownership

| Crate or layer | Typical error ownership |
| --- | --- |
| tensor | Shape, layout, indexing, dtype, size validation |
| cpu | CPU kernel execution and backend-specific failures |
| autograd | Graph construction and gradient computation failures |
| nn | Layer construction and module configuration failures |
| models | Model configuration, architecture support, parameter compatibility |
| tokenizer | Tokenization configuration, malformed data, vocabulary and encoding failures |
| data | Dataset access, decoding, and input pipeline failures |
| format | Format parsing, validation, and serialization failures |
| checkpoint | Checkpoint metadata, integrity, compatibility, and I/O failures |
| optim | Optimizer configuration and update failures |
| train | Training orchestration and run-level failures |
| infer | Inference orchestration, generation limits, and request execution |
| api | Stable public contracts and boundary-level validation |
| serve | Request validation, service lifecycle, and protocol translation |
| cli | Argument validation, command execution, and presentation |
| testkit | Test support and fixture construction failures |

This table is an ownership guide, not a mandate to create every listed error type. Types SHOULD be introduced when they represent real, distinguishable failure semantics.

### 6.2 Shared error types

A shared error type MAY exist when multiple crates genuinely share the same semantic contract.

It MUST NOT exist merely to eliminate a few From implementations or reduce the number of lines in a match statement.

Before introducing a shared error type, reviewers MUST evaluate:

* Whether the abstraction is genuinely shared.
* Whether the new dependency direction is permitted.
* Whether downstream crates need the detailed variants.
* Whether the shared type will force unrelated modules to depend on each other.
* Whether the shared type can evolve without destabilizing unrelated public APIs.

A common error vocabulary is useful. A common dependency on every subsystem’s implementation details is not.

### 6.3 Error ownership follows abstraction ownership

A lower-level crate MUST NOT return an error that requires the caller to understand an unrelated application framework.

For example, a tensor operation SHOULD NOT return an HTTP response error.

An HTTP server MAY map a tensor shape error into a request response, but the tensor crate MUST remain independent of that presentation decision.

---

## 7. Designing Error Types

### 7.1 Prefer typed variants

Error types SHOULD make meaningful failure categories explicit.

An illustrative crate-local error type:

```rust
#[derive(Debug)]
pub enum ShapeError {
    EmptyShapeNotAllowed,
    DimensionOverflow,
    ElementCountExceedsLimit {
        requested: usize,
        limit: usize,
    },
    IncompatibleDimensions {
        expected: usize,
        actual: usize,
    },
}
```

This example demonstrates the design principle; actual variants MUST reflect the crate’s real contract.

Typed variants allow callers and tests to distinguish errors without relying on their display messages.

### 7.2 Error variants should represent actionable distinctions

Create separate variants when the distinction affects at least one of the following:

* Caller recovery behavior.
* Retry eligibility.
* API compatibility.
* Security handling.
* Diagnostics or observability.
* Testable contract behavior.

Do not create dozens of nearly identical variants merely to mirror every internal function.

Do not merge semantically different failures into a single Other(String) variant when callers need to distinguish them.

### 7.3 Human-readable messages are not control-flow APIs

Error messages may change for clarity, localization, or additional context.

Callers MUST NOT use substring matching on Display output to decide whether to retry, return a particular protocol status, or select a fallback.

Bad:

```rust
if error.to_string().contains("timeout") {
    retry_operation();
}
```

Prefer an explicit classification:

```rust
match error {
    ServiceError::TimedOut => {
        // Apply the documented timeout policy.
    }
    ServiceError::Cancelled => {
        // Follow cancellation semantics.
    }
    _ => {
        // Handle other failure categories.
    }
}
```

Production code SHOULD use the error type actually owned by the relevant abstraction.

### 7.4 Error codes and stable public contracts

Errors exposed through a public API or wire protocol SHOULD have stable machine-readable codes where consumers need to distinguish them.

The stable contract SHOULD define:

* The code.
* Its meaning.
* Whether the caller may retry.
* Whether the condition indicates invalid input or a server-side failure.
* Any documented recovery action.

Human-readable messages SHOULD NOT be the sole machine-readable contract.

Internal Rust error variants MAY evolve according to the crate’s compatibility policy. A wire-level error code SHOULD remain stable unless a documented versioning policy permits a change.

---

## 8. Error Propagation and Conversion

### 8.1 Use ? for straightforward propagation

Use the ? operator when the current function can propagate the error without changing its semantic meaning.

```rust
fn load_weights(
    path: &Path,
) -> Result<Weights, CheckpointError> {
    let bytes = std::fs::read(path)?;
    decode_weights(&bytes)
}
```

This is valid only if CheckpointError has a deliberate conversion from the I/O error and decode_weights returns a compatible result.

The example does not require a universal error type.

### 8.2 Convert errors at meaningful boundaries

Convert an error when crossing an abstraction boundary changes the meaning or ownership of the failure.

For example:

```text
std::io::Error
    -> checkpoint I/O error
    -> training-run failure
    -> CLI diagnostic and exit status
```

Not every arrow needs a new wrapper. The choice depends on the contract and the diagnostic information that must be preserved.

Conversion MUST NOT silently discard information needed for diagnosis or recovery.

### 8.3 Avoid redundant wrapping

Do not wrap the same error at every function call solely to add repetitive text.

Bad design:

```text
operation failed
  caused by: operation failed
    caused by: operation failed
      caused by: operation failed
```

Add context when it identifies a meaningful operation, resource, stage, or boundary.

Useful context might include:

* Loading a particular checkpoint component.
* Validating a model configuration.
* Writing a specific checkpoint generation.
* Executing a named operation in a training step.

Sensitive values, credentials, private prompt content, and other confidential data MUST NOT be added to error context merely because they are available.

### 8.4 Preserve source errors where appropriate

Errors originating from standard-library I/O or external dependencies SHOULD retain their source when the underlying information is useful and safe to expose internally.

A manual implementation can expose the source through std::error::Error::source().

Error libraries such as thiserror MAY be used if the dependency is permitted and justified by the workspace’s dependency policy. This document does not mandate a particular error library.

### 8.5 Avoid unnecessary Box<dyn Error> in domain APIs

Box<dyn std::error::Error> MAY be appropriate at an application boundary where heterogeneous errors are intentionally aggregated.

It SHOULD NOT replace meaningful crate-local errors in core domain APIs when callers need stable variants or typed recovery behavior.

The type-erasure decision belongs to the abstraction contract, not merely to implementation convenience.

---

## 9. Runtime Panic Policy

### 9.1 No panic-based runtime error handling

Production runtime code MUST NOT use:

* panic!()
* unwrap()
* expect()

to handle operations that can fail due to input, I/O, resource limits, backend behavior, concurrency, cancellation, or other runtime conditions.

This prohibition includes error handling in:

* Tensor operations.
* Model loading.
* Tokenization.
* Checkpoint reading and writing.
* Training loops.
* Inference and generation.
* Service request processing.
* Worker threads and background tasks.

Use explicit error propagation instead.

### 9.2 Panics are not a substitute for validation

Do not rely on a panic to reject malformed external data.

Validate dimensions, offsets, lengths, configuration values, and other externally influenced data before operations that assume validity.

Where integer arithmetic can overflow, use checked arithmetic or another operation with explicitly understood overflow behavior.

Where allocation can fail, use an appropriate fallible allocation API when available and relevant to the contract. Do not assume that every possible allocation or process-level out-of-memory failure can always be recovered from.

### 9.3 Invariants and unreachable paths

The runtime prohibition does not imply that every logically impossible condition should be converted into an arbitrary error.

When an invariant appears to be guaranteed by construction, the implementation SHOULD make that guarantee explicit through types, visibility, ownership, validation, or a well-defined internal contract.

If a condition believed to be impossible can actually be triggered by input, corrupted data, concurrency, or a backend, it is not a valid reason to use unwrap() or expect().

An explicit invariant assertion MAY be considered only when the invariant is genuinely internal, its violation indicates a programming defect, and the assertion is not being used to handle an expected runtime failure. Such cases require review and MUST NOT bypass validation of external data.

The preferred solution is to make invalid states unrepresentable or return a typed error at the boundary.

### 9.4 Tests and examples

Tests MAY use unwrap() or expect() for concise setup and assertion when failure should immediately fail the test.

However:

* Test fixtures SHOULD provide useful failure messages.
* Tests MUST NOT hide errors by ignoring Result values.
* Examples intended for production reuse SHOULD demonstrate correct propagation.
* Test-only panic allowances MUST NOT leak into production code.

### 9.5 Panic recovery is not normal control flow

A panic is not equivalent to a returned error. It may leave state partially updated and may prevent cleanup or invalidate assumptions in surrounding code.

A top-level panic boundary MAY be appropriate for isolating a worker or reporting an unexpected defect, but it MUST NOT be treated as a general replacement for typed error handling.

A service MUST NOT assume that catching a panic makes arbitrary internal state safe to reuse.

---

## 10. External Input and Data Validation

All untrusted or externally supplied data MUST be validated before it can violate internal invariants or trigger uncontrolled resource consumption.

Sources include:

* CLI arguments.
* HTTP request bodies.
* Model configuration files.
* Tokenizer vocabularies and merges.
* Tensor dimensions and offsets.
* Checkpoint metadata.
* Serialized model weights.
* Dataset records.
* Plugin or backend configuration.

### 10.1 Validate before expensive work

Where practical, validate:

* Integer ranges and checked arithmetic.
* Shape compatibility.
* File sizes and declared lengths.
* Supported data types.
* Required fields.
* Enum and identifier values.
* Format versions.
* Model and tokenizer compatibility.
* Maximum sequence lengths.
* Maximum batch sizes.
* Memory and work budgets.

Validation MUST be based on actual limits and contracts. Do not invent arbitrary limits without documenting their purpose and configurability.

### 10.2 Treat serialized data as untrusted

A file claiming to contain a tensor of a particular shape MUST NOT be trusted solely because the header says so.

Validate that dimensions, element counts, byte lengths, offsets, and supported limits agree before reading or allocating based on them.

The implementation MUST use checked arithmetic wherever overflow could invalidate size or offset calculations.

Malformed data MUST produce a meaningful error rather than trigger out-of-bounds access, uncontrolled allocation, or panic-based rejection.

### 10.3 Do not leak internal implementation details

External error responses SHOULD communicate the failure category and any safe recovery action.

They SHOULD NOT expose:

* Stack traces.
* Secrets or credentials.
* Internal filesystem layouts.
* Private prompt or training data.
* Raw request bodies by default.
* Backend implementation details that could expose sensitive operational information.

Detailed diagnostics belong in appropriately controlled internal logs.

---

## 11. Resource Exhaustion

Resource exhaustion is a first-class failure category.

Potential causes include:

* Excessive tensor dimensions.
* Large model files.
* Oversized checkpoint entries.
* Long input sequences.
* Large inference batches.
* Unbounded queues.
* Excessive concurrent requests.
* Memory pressure during training.
* Excessive temporary storage use.

### 11.1 Resource budgets

Operations that can consume substantial resources SHOULD operate under explicit budgets.

Depending on the subsystem, budgets may include:

* Maximum bytes.
* Maximum tensor elements.
* Maximum sequence length.
* Maximum batch size.
* Maximum work units.
* Maximum concurrent operations.
* Maximum temporary storage.
* Maximum execution duration.

Budgets MUST be enforced at an appropriate boundary. A caller-provided limit MUST NOT be allowed to bypass a stricter system-wide safety limit.

### 11.2 Distinguish exhaustion from invalid input

An input may be structurally valid but exceed an operational limit.

For example, a tensor shape may be mathematically valid yet too large for the configured memory budget.

Where callers need different recovery behavior, distinguish:

* Invalid shape.
* Unsupported operation.
* Configured resource limit exceeded.
* Actual allocation or backend resource failure.

### 11.3 Partial allocation and cleanup

If an operation fails after acquiring resources, it MUST leave the system in a documented and safe state.

Prefer RAII and ownership-based cleanup.

Do not manually suppress cleanup failures or return success when only part of an operation completed.

If a multi-step operation can partially modify persistent state, its commit and rollback semantics MUST be documented and tested.

---

## 12. Cancellation and Timeout

Cancellation and timeout are distinct from ordinary operation failure and MUST have explicit semantics in asynchronous and long-running operations.

### 12.1 Cancellation

Cancellation indicates that the operation is no longer required or is no longer permitted to continue.

Examples:

* A client disconnects.
* A caller cancels an inference request.
* A training run is explicitly stopped.
* A worker is asked to shut down.

Cancellation MUST NOT be reported as successful completion if the requested work did not complete.

The implementation SHOULD stop work at safe cancellation points and release owned resources.

If cancellation occurs during an operation that modifies persistent state, the resulting state MUST follow a documented consistency policy.

### 12.2 Timeout

A timeout indicates that an operation exceeded a defined time budget.

The timeout policy SHOULD specify:

* Which phase is timed.
* Whether the timeout is a deadline or a duration limit.
* Whether it includes queueing time.
* Whether retries share the original deadline.
* What happens to in-flight work.
* Whether partial output may be returned.

A timeout MUST NOT automatically trigger an unbounded retry.

### 12.3 Preserve the distinction

Cancellation and timeout MAY be represented by separate variants or by a shared type with explicit reasons, provided callers can distinguish them when necessary.

Do not convert both into a generic internal error if the service or caller needs different behavior.

---

## 13. Retry Policy

Retries are a policy decision, not a default error-handling mechanism.

### 13.1 Retry only when justified

An operation SHOULD be retried only when:

1. The failure is plausibly transient.
2. The operation is safe to repeat, or its effects are protected by an idempotency mechanism.
3. The retry is within a documented budget.
4. The retry respects cancellation and deadlines.
5. The retry does not violate resource or concurrency limits.

Examples of potentially transient failures include temporary network errors or temporary backend unavailability. Whether a particular failure is transient MUST be determined from the actual subsystem contract.

Invalid input, unsupported operations, incompatible model formats, and deterministic validation failures SHOULD NOT be retried automatically.

### 13.2 Bounded retries

Retry policies SHOULD specify:

* Maximum attempts.
* Backoff behavior.
* Maximum elapsed time.
* Eligible failure categories.
* Cancellation behavior.
* Observability requirements.

Backoff and jitter MAY be used for distributed or contended systems when appropriate.

The retry budget MUST fit within the operation’s overall deadline.

### 13.3 Avoid duplicate side effects

Training and checkpoint operations require particular care.

Blindly retrying a partially completed optimizer update or checkpoint write may produce duplicated effects or inconsistent state.

Before retrying, determine whether the operation:

* Is read-only.
* Is idempotent.
* Has a transactional commit boundary.
* Can detect an already completed operation.
* Requires rollback or recovery.

If safe retry semantics cannot be established, return the failure and require explicit recovery rather than guessing.

---

## 14. Checkpoint and Persistent-State Failures

Checkpoint operations MUST define their behavior for failures during reading, writing, validation, and recovery.

### 14.1 Read failures

A checkpoint read MUST distinguish meaningful failure categories where needed, including:

* File not found.
* Permission denied.
* I/O failure.
* Unsupported format version.
* Malformed metadata.
* Inconsistent lengths or offsets.
* Integrity validation failure.
* Incompatible model configuration.
* Missing required parameter data.

These categories MAY be combined when the public contract does not require the distinction, but the diagnostic information needed for safe recovery MUST be preserved.

### 14.2 Write failures

Checkpoint writing SHOULD avoid exposing a partially written checkpoint as a successfully committed artifact.

Where supported by the storage system and the chosen format, use an explicit staging and commit strategy, such as writing to a temporary artifact and publishing it only after successful completion and validation.

The implementation MUST document the guarantees it actually provides. Atomic rename, durability, and crash consistency depend on filesystem and platform behavior and MUST NOT be assumed universally.

### 14.3 Recovery

Recovery logic MUST distinguish a verified valid checkpoint from a partially written or corrupted artifact.

The implementation MUST NOT silently replace corrupt state with an empty model, default weights, or another checkpoint unless that behavior is explicitly requested and clearly reported.

Any recovery action that can discard user data or training progress requires a deliberate policy and appropriate tests.

---

## 15. Numerical and Backend Errors

Numerical computation requires explicit treatment of invalid values and backend failures.

Examples include:

* Unsupported dtype or operation.
* Shape mismatch.
* Non-finite values where the contract prohibits them.
* Invalid reduction dimensions.
* Backend initialization failure.
* Device memory exhaustion.
* Kernel execution failure.
* Unsupported hardware capability.

### 15.1 Do not conflate numerical results with errors

NaN or infinity is not automatically an error in every operation. Their meaning depends on the operation and contract.

A kernel MUST NOT silently replace unexpected non-finite values with zero or another default unless the operation explicitly specifies that behavior.

If an operation requires finite values, validation MUST make that requirement explicit.

### 15.2 Backend-specific diagnostics

The CPU backend and any future accelerator backend MAY use backend-specific internal errors.

The public abstraction SHOULD expose only the distinctions needed by callers, such as unsupported operation, resource exhaustion, or execution failure.

Do not force higher-level model or inference code to depend on a particular backend’s internal error representation.

### 15.3 No fabricated success after partial computation

A failed kernel or incomplete operation MUST NOT return a normal result unless the API explicitly defines a valid partial-result contract.

If partial results are supported, their completeness and validity MUST be represented explicitly.

---

## 16. Error Handling in Training and Inference

### 16.1 Training

Training failures may occur after significant work has completed.

The training layer SHOULD distinguish, where relevant:

* Invalid training configuration.
* Dataset or batch loading failure.
* Forward-pass failure.
* Backward-pass failure.
* Optimizer update failure.
* Checkpoint persistence failure.
* Cancellation.
* Resource exhaustion.

A failed training step MUST NOT be reported as a successful step.

If an error occurs during a state-mutating update, the implementation MUST define whether the update is atomic, recoverable, or potentially partial.

Do not assume that every failure can be rolled back. Document and test the guarantees that are actually implemented.

### 16.2 Inference

Inference failures SHOULD distinguish between request validation, model readiness, unsupported functionality, resource limits, cancellation, timeout, and backend execution failure when these distinctions affect the caller.

An inference request MUST NOT mutate shared model weights merely to record request-level errors or temporary state.

Request-specific context, cancellation state, generation limits, and diagnostics SHOULD remain scoped to the request or execution context.

### 16.3 Streaming output

Streaming interfaces MUST document what happens when an error occurs after output has already been emitted.

A stream MUST NOT silently terminate in a way that implies successful completion when the operation failed.

Where the protocol supports terminal error events, use them consistently. If the transport cannot express a terminal error, the limitation MUST be documented and observable through the relevant operational channel.

---

## 17. CLI and Service Error Mapping

Internal errors and external presentation have different requirements.

### 17.1 CLI

The CLI SHOULD:

* Print a concise, actionable diagnostic.
* Use a nonzero exit status for command failure.
* Preserve detailed source context when useful.
* Avoid dumping stack traces by default.
* Avoid exposing secrets or sensitive input.
* Distinguish usage errors from operational failures where appropriate.

A CLI MUST NOT print an error and then exit successfully unless the command’s documented contract explicitly defines that outcome.

### 17.2 HTTP and service APIs

A service SHOULD map internal errors to protocol-level responses at the service boundary.

The mapping MUST be based on error semantics, not string matching.

The response SHOULD include a stable error code when clients need machine-readable distinctions.

The response MUST NOT expose raw internal error chains, filesystem paths, stack traces, credentials, or private model inputs.

A service MUST NOT claim that an operation succeeded merely because the response itself was successfully constructed.

### 17.3 Keep transport dependencies at the boundary

Core crates MUST NOT depend on HTTP status codes or CLI exit codes.

Protocol-specific mapping belongs in serve, cli, or another explicitly designated application boundary.

---

## 18. Logging, Metrics, and Diagnostics

Errors should be observable without creating excessive noise or exposing confidential data.

### 18.1 Log at the appropriate layer

The layer that has sufficient context and owns the recovery decision SHOULD generally be responsible for recording the failure.

Avoid logging the same propagated error at every function boundary.

Repeated logs make a single failure appear to be many independent incidents and obscure the useful diagnostic.

### 18.2 Preserve causal information

Diagnostics SHOULD include useful operation context, such as:

* Operation or stage name.
* Relevant non-sensitive identifiers.
* Error category.
* Whether the operation was retried.
* Whether recovery succeeded.
* Whether the failure was expected or unexpected.

Avoid logging entire tensors, complete model weights, raw request bodies, private prompts, credentials, or arbitrary user data merely to improve debugging.

### 18.3 Metrics

Where operational metrics exist, error categories SHOULD be suitable for aggregation.

Metric labels MUST remain bounded in cardinality. Arbitrary error messages, full paths, request IDs, and user-controlled strings SHOULD NOT be used as metric labels.

Metrics SHOULD distinguish failures that require different operational responses, such as resource exhaustion versus invalid input.

### 18.4 Do not double-report

A low-level function SHOULD usually return an error rather than both logging it as a final failure and returning it for the caller to log again.

Logging before propagation MAY be appropriate when a lower-level event is independently operationally significant. That decision SHOULD be explicit.

---

## 19. Error Testing Requirements

Error behavior is part of the API contract and MUST be tested.

The amount of testing SHOULD scale with the risk and complexity of the operation.

### 19.1 Required test categories

Where applicable, tests SHOULD cover:

* Valid input.
* Invalid input.
* Boundary values.
* Integer overflow.
* Missing files or resources.
* Malformed serialized data.
* Unsupported versions or operations.
* Resource-limit violations.
* Backend failures.
* Cancellation and timeout.
* Partial failure and cleanup.
* Retry limits.
* Persistent-state consistency.
* Error conversion and public mapping.

Not every category applies to every function. Test selection MUST reflect the actual failure modes.

### 19.2 Assert semantics, not incidental wording

Prefer checking the error variant or stable code:

```rust
assert!(matches!(
    result,
    Err(ShapeError::DimensionOverflow)
));
```

Tests MAY assert display text when the exact text is itself a documented user-facing contract. Otherwise, exact wording assertions SHOULD be avoided because they make harmless diagnostic improvements unnecessarily expensive.

### 19.3 Test resource and cleanup behavior

For operations that allocate memory, create files, mutate state, or spawn tasks, failure tests SHOULD verify that resources are released and state remains consistent with the documented contract.

### 19.4 Do not test only the happy path

A function that returns the correct value for valid input but panics for malformed input does not satisfy the error-handling contract.

Where failure is expected and recoverable, tests SHOULD demonstrate explicit error propagation rather than panic-based behavior.

---

## 20. Public API and Compatibility

Public error types are part of the API surface.

Before changing a public error type, reviewers SHOULD evaluate:

* Whether downstream callers match its variants.
* Whether new variants affect exhaustive matching.
* Whether an error code is externally documented.
* Whether the error source remains accessible.
* Whether conversion behavior changes.
* Whether the change alters retry or recovery semantics.

### 20.1 Avoid leaking implementation details

Public errors SHOULD expose domain-relevant meaning rather than every internal helper or backend implementation detail.

A backend replacement SHOULD NOT force unrelated callers to rewrite their error handling unless the public contract intentionally changes.

### 20.2 Error messages and compatibility

Human-readable messages SHOULD be useful and clear, but they SHOULD NOT be treated as stable identifiers unless explicitly documented.

When external clients require stability, define a dedicated error code or protocol-level error structure.

### 20.3 No misleading error conversions

An error conversion MUST preserve the meaning of the original failure.

Do not convert:

* Invalid input into an internal server error without reason.
* Resource exhaustion into invalid input.
* Cancellation into success.
* Corrupt persistent state into an ordinary missing-value result.
* Unsupported functionality into an unrelated backend failure.

A conversion MAY simplify details, but it MUST preserve the semantic distinctions required by the receiving layer.

---

## 21. Dependency Policy for Error Handling

The workspace MAY adopt an error-handling library when it provides a clear benefit.

A dependency MUST NOT be introduced solely because it is popular or because it removes a small amount of boilerplate.

Reviewers SHOULD consider:

* Whether the library is needed in core crates.
* Whether it introduces unwanted dependency coupling.
* Whether it affects compile time or binary size.
* Whether its public types would leak across crate boundaries.
* Whether standard Rust traits and explicit enums are sufficient.

A library used for ergonomic error definitions does not remove the requirement to design error semantics correctly.

The workspace SHOULD avoid mixing incompatible error conventions across crates without a documented reason.

---

## 22. Anti-Patterns

The following patterns MUST be rejected when they conceal meaningful failures or violate crate boundaries.

### 22.1 Silent error suppression

```rust
let _ = save_checkpoint(path);
```

This is unacceptable when checkpoint persistence is required for successful completion.

Ignoring a result MAY be valid only when the failure is intentionally irrelevant to the contract and that decision is explicit. Required persistence, state updates, and safety checks do not qualify.

### 22.2 Converting every failure to None

```rust
let model = load_model(path).ok();
```

This erases the distinction between absence and failure.

### 22.3 String-based retry decisions

```rust
if error.to_string().contains("temporary") {
    retry();
}
```

This creates an unstable, undocumented control-flow contract.

### 22.4 Broad fallback behavior

```rust
let output = run_model(input).unwrap_or_default();
```

This can make a failed model execution appear successful.

### 22.5 Panics on untrusted data

```rust
let dimension = parsed_dimension.unwrap();
```

This turns malformed or missing external data into a process-level failure.

### 22.6 Generic error types everywhere

```rust
type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
```

A type like this MAY be appropriate at a deliberate application boundary, but SHOULD NOT replace all domain-specific errors across the workspace.

### 22.7 Repeated logging at every layer

Logging every propagation step creates duplicate diagnostics and makes causal analysis harder.

### 22.8 Unbounded retries

Retrying indefinitely can amplify an outage, exhaust resources, or ignore cancellation.

### 22.9 Errors that imply recovery without performing it

An error message MUST NOT claim that state was restored, rolled back, or preserved unless the implementation has actually established that guarantee.

---

## 23. Review Checklist

Every production change involving error handling SHOULD be reviewed against the following checklist.

Semantics

* [ ] Does the error represent a real and distinguishable failure?
* [ ] Is expected absence represented appropriately?
* [ ] Are invalid input and internal invariant violations distinguished?
* [ ] Are resource exhaustion, cancellation, and timeout handled according to the contract?
* [ ] Is partial success explicitly represented when applicable?

Architecture

* [ ] Does the error type belong to the correct crate?
* [ ] Does the change preserve the permitted dependency direction?
* [ ] Is error conversion performed at a meaningful abstraction boundary?
* [ ] Does the public API avoid leaking unnecessary implementation details?

Runtime behavior

* [ ] Are runtime failures propagated without panic!, unwrap(), or expect()?
* [ ] Are errors prevented from becoming fabricated defaults or silent success?
* [ ] Are retries bounded and safe?
* [ ] Are resources and persistent state left in a documented condition after failure?
* [ ] Is cancellation respected?

Diagnostics and security

* [ ] Is sufficient source context preserved?
* [ ] Are sensitive values excluded from external messages and inappropriate logs?
* [ ] Are errors logged at an appropriate layer?
* [ ] Are machine-readable decisions based on typed variants or stable codes rather than message strings?

Verification

* [ ] Are relevant error paths covered by tests?
* [ ] Are boundary conditions and overflow paths tested where applicable?
* [ ] Are cleanup and partial-failure behavior tested when relevant?
* [ ] Have the relevant workspace checks been run?
* [ ] Are unverified behavior and remaining limitations reported honestly?

---

## 24. Adoption and Conformance

This document defines the target standard for error handling. It does not assert that the existing workspace already conforms.

Each crate SHOULD be audited independently.

For each finding, record:

* Crate and affected operation.
* Current failure behavior.
* Required behavior.
* Risk and impact.
* Minimal remediation.
* Tests required.
* Verification status.

Use these conformance states:

* Proposed — the desired behavior is specified but not verified.
* Verified — implementation and relevant tests demonstrate conformance.
* Non-conforming — a known violation exists.
* Blocked — verification or remediation requires an unresolved dependency, design decision, or environmental capability.

Do not mark a crate as verified based only on a successful workspace compilation. Compilation does not prove correct failure semantics.

Adoption SHOULD proceed incrementally, with each change forming a small, independently testable stage.

---

## 25. Final Principle

The purpose of error handling is not to make failures disappear. It is to make them explicit, meaningful, bounded, observable, and safe to respond to.

Never turn an unknown failure into a false success. Never use an error abstraction to conceal an architectural defect. Never claim recovery without evidence.

When in doubt:

1. Preserve the failure.
2. Preserve its meaning.
3. Add only useful context.
4. Propagate it to the layer that owns the recovery decision.
5. Test the failure path.
6. Report exactly what has and has not been verified.