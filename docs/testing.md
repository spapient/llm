# Testing Strategy and Quality Gates

Status: Proposed
Scope: All crates, binaries, integration tests, examples, and benchmarks in the llm/ workspace
Audience: Engineers, reviewers, maintainers, and AI coding agents

---

## 1. Purpose

This document defines the testing strategy for the llm/ workspace.

Testing is not a final step performed after implementation. It is part of the design process and a required source of evidence for correctness, compatibility, numerical behavior, and performance claims.

The objectives are to:

* Verify behavior at explicit contract boundaries.
* Detect regressions before they propagate across crates.
* Validate numerical correctness independently of implementation details.
* Exercise malformed input, boundary conditions, and failure paths.
* Preserve architectural boundaries through automated checks.
* Establish reproducible evidence for performance measurements.
* Make every completed change independently testable.
* Keep tests deterministic, focused, maintainable, and useful during refactoring.

A passing test suite is evidence of tested behavior, not proof that every possible execution is correct. Test claims MUST remain proportional to the coverage and strength of the evidence.

## 2. Normative Language

The terms MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY express engineering requirements.

MUST-level requirements may be waived only through an explicit, reviewed decision that documents the rationale, scope, risk, and alternative verification strategy.

## 3. Non-Negotiable Rules

1. Every production behavior MUST have an appropriate verification strategy.
2. Every bug fix MUST include a regression test when the failure can reasonably be reproduced in a test.
3. Every new operator MUST include boundary tests, numerical correctness tests, and at least one performance benchmark.
4. Tests MUST NOT depend on execution order, wall-clock timing, network availability, or uncontrolled external state unless the test explicitly targets that dependency.
5. Numerical tests MUST use an appropriate reference and documented tolerances.
6. Tests MUST cover failure behavior when failure semantics are part of the contract.
7. Tests MUST NOT be weakened merely to make an incorrect implementation pass.
8. Benchmarks MUST NOT be used as substitutes for correctness tests.
9. Performance improvements MUST NOT be claimed without comparable measurements.
10. Architecture and dependency constraints SHOULD be enforced automatically where practical.
11. A change MUST NOT be considered complete solely because it compiles.
12. Production code and tests MUST follow the workspace’s prohibition on unsafe.
13. Runtime error handling MUST follow docs/error-handling.md.
14. Each implementation stage SHOULD remain small enough to validate independently.
15. All reported test results MUST accurately describe what was actually executed.

---

## 4. Testing Philosophy

The workspace SHOULD follow five principles.

### 4.1 Test contracts, not implementation trivia

Tests SHOULD verify externally observable behavior, documented invariants, and domain contracts.

They SHOULD NOT unnecessarily depend on private helper names, internal call sequences, incidental allocation patterns, or other details that may change during a valid refactor.

Internal tests MAY inspect implementation details when doing so is the most effective way to verify a critical invariant.

### 4.2 Prefer independent evidence

Whenever practical, compare a result against a reference that is independent of the implementation being tested.

Examples include:

* A simple scalar implementation as a reference for an optimized tensor operator.
* Hand-calculated results for small matrices.
* Finite-difference approximations for gradient validation.
* Known serialization fixtures for format compatibility.
* An independent implementation for validating numerical behavior.

Comparing two code paths that share the same underlying defect does not establish independent correctness.

### 4.3 Test at the lowest useful layer

A tensor shape invariant belongs primarily in tensor tests.

A CPU kernel’s numerical behavior belongs primarily in cpu tests.

A model’s parameter mapping belongs primarily in models tests.

Application integration belongs in integration or application-level tests.

Higher-level tests SHOULD verify that components work together, rather than duplicate every low-level test.

### 4.4 Make failures diagnosable

A failing test SHOULD identify:

* The operation under test.
* The relevant input or minimized fixture.
* The expected behavior.
* The actual behavior.
* The numerical tolerance or invariant involved, when applicable.

Tests SHOULD fail with actionable diagnostics rather than unexplained Boolean assertions.

### 4.5 Keep tests proportional to risk

A simple metadata accessor does not need the same testing machinery as a checkpoint writer, optimizer update, or numerically sensitive kernel.

Testing effort MUST reflect failure impact, complexity, external exposure, and difficulty of detecting regressions.

---

## 5. Test Taxonomy

Use the smallest test category that can establish the required contract, then add broader tests where integration risk justifies them.

| Test category | Primary purpose | Typical location |
| --- | --- | --- |
| Unit tests | Verify a small function, type, or invariant | Repository-root `tests/`, public API only |
| Contract tests | Verify public behavior and trait contracts | Repository-root `tests/` and dedicated contract suites |
| Property tests | Verify general properties over generated inputs | Crate tests or dedicated test modules |
| Numerical tests | Verify floating-point results against a reference | tensor, cpu, autograd, nn, optim |
| Regression tests | Prevent a known defect from returning | Near the affected component |
| Integration tests | Verify interactions across public crate boundaries | Repository-root `tests/` (owned by the workspace root package) |
| Format compatibility tests | Verify reading and writing defined data formats | format, checkpoint |
| End-to-end tests | Verify complete user-visible workflows | Application or workspace integration tests |
| Architecture tests | Verify dependency and layering rules | Scripts, CI checks, or dedicated validation tools |
| Benchmarks | Measure performance under controlled conditions | Repository-root `benches/`, with shared workloads in `crates/bench` |
| Stress tests | Exercise sustained load and resource behavior | Dedicated test targets |
| Fault-injection tests | Verify behavior when a dependency or operation fails | Infrastructure and orchestration tests |

These categories are complementary. A benchmark does not establish correctness, and a unit test does not establish correct integration across the workspace.

---

## 6. Workspace Test Organization

The workspace SHOULD organize tests according to ownership and scope.

### 6.1 Unit tests

Unit tests live at the repository root `tests/` and exercise the crate through
its public `pub` interface only. Production source under `crates/*/src` MUST NOT
contain `#[cfg(test)]` modules, `#[test]` functions, or `#[cfg(test)]`-gated
items (see
[decision-records/blackboxtesting.md](decision-records/blackboxtesting.md); the
gate is `scripts/checksrctests.sh`).

Use unit tests for:

* Public invariants and contracts.
* Small algorithms reached through a public entry point.
* Validation logic and error variants.
* Boundary conditions reachable through the public interface.
* Local state transitions observable through the public interface.

A behavior that cannot be reached from the public interface is not tested;
either a legitimate public entry point is designed for it, or the unreachable
logic is deleted as dead code.

Keep fixtures small and deterministic.

### 6.2 Integration tests

Integration tests SHOULD exercise public interfaces and permitted crate interactions.

They SHOULD be used to verify:

* Public API behavior.
* Trait interoperability.
* Cross-crate data contracts.
* Error conversion at abstraction boundaries.
* Model loading and inference workflows.
* Checkpoint save-and-load workflows.
* Training and optimizer integration.

Integration tests MUST NOT become a mechanism for bypassing intended crate boundaries.

### 6.3 Shared test utilities

Shared fixtures and helpers MAY be placed in testkit when multiple test suites genuinely need them.

The testkit crate MUST NOT become a production dependency.

Test helpers SHOULD be small and explicit. Avoid building a large testing framework that hides the behavior under test.

### 6.4 Test data

Fixtures SHOULD be:

* Small enough for routine CI execution.
* Deterministic.
* Clearly named.
* Versioned when compatibility matters.
* Licensed appropriately when sourced externally.
* Documented when their binary structure is not self-explanatory.

Large models and datasets SHOULD NOT be required for every unit-test run.

Use small synthetic fixtures for routine tests and reserve larger artifacts for dedicated integration or release validation.

### 6.5 Avoid duplicated test ownership

A contract SHOULD have a clear primary owner.

For example, tensor should establish tensor shape semantics. A model integration test may verify that a model respects those semantics, but it need not reproduce every tensor boundary test.

---

## 7. Unit Test Requirements

Unit tests SHOULD verify one primary behavior at a time.

For each nontrivial function, consider:

* Typical valid input.
* Minimum and maximum supported values.
* Empty input, if permitted.
* Invalid input.
* Boundary transitions.
* Arithmetic overflow.
* Error propagation.
* State preservation after failure.

Not every function requires every category. Test cases SHOULD be selected according to the function’s contract.

### 7.1 Avoid oversized tests

A test that exercises dozens of unrelated behaviors is difficult to diagnose.

Split tests when they verify independent contracts or fail for unrelated reasons.

A test MAY cover a complete workflow when the interactions themselves are the intended subject.

### 7.2 Test names

Test names SHOULD describe the condition and expected result.

Prefer names such as:

```text
rejects_incompatible_dimensions
returns_error_when_element_count_overflows
preserves_parameters_after_failed_validation
loads_supported_checkpoint_version
```

Avoid vague names such as `test_basic`, `test_all`, or `works`.

### 7.3 Avoid tests that only mirror the implementation

A test should not simply repeat the production algorithm and compare the two results if both are likely to share the same bug.

Prefer direct expected values, independently derived references, invariants, or deliberately simpler algorithms.

---

## 8. Boundary and Adversarial Testing

Boundary tests are mandatory for components that process dimensions, indices, lengths, offsets, sizes, or externally supplied metadata.

### 8.1 Tensor boundaries

Where applicable, test:

* Empty shapes.
* Scalars and rank-one tensors.
* Supported minimum dimensions.
* Large dimensions near configured limits.
* Shape mismatch.
* Invalid indexing.
* Invalid permutations.
* Checked element-count arithmetic.
* Zero-sized dimensions if the API permits them.
* Empty reductions and broadcasting edge cases.
* Non-contiguous layouts, if supported.

Do not assume that zero-sized tensors are invalid or valid universally. The behavior MUST follow the documented tensor contract.

### 8.2 Input parsing boundaries

For parsers and deserializers, test:

* Empty input.
* Truncated input.
* Invalid headers.
* Invalid lengths and offsets.
* Unsupported versions.
* Inconsistent metadata.
* Unexpected trailing bytes, if relevant.
* Maximum permitted input sizes.
* Invalid encoding or malformed records.

Malformed input MUST produce the documented error rather than panic or uncontrolled resource consumption.

### 8.3 Integer arithmetic

Any calculation that can overflow and affect allocation, indexing, layout, or file access SHOULD receive explicit boundary coverage.

Examples include:

* Element count.
* Byte size.
* Strides.
* Offset calculations.
* Buffer lengths.
* Batch-size products.
* Checkpoint section boundaries.

Tests SHOULD exercise the overflow condition directly instead of relying on impractically large allocations.

### 8.4 Boundary coverage is contract-dependent

Do not introduce arbitrary behavior merely to satisfy a test checklist.

If a boundary is unsupported, the implementation SHOULD reject it explicitly and the tests SHOULD verify the rejection.

If a boundary is supported, tests SHOULD verify its behavior.

---

## 9. Numerical Correctness

Numerical correctness is a first-class requirement for tensor computation, neural-network operations, automatic differentiation, optimization, and model execution.

A numerically plausible result is not necessarily correct.

### 9.1 Reference implementations

Numerically nontrivial operators SHOULD be compared against an independent reference.

Suitable references may include:

* Hand-calculated small cases.
* A simple scalar algorithm.
* A higher-precision calculation.
* A trusted implementation with a documented contract.
* An independently derived mathematical identity.

The reference MUST match the intended mathematical operation, including shape, reduction, broadcasting, and edge-case semantics.

### 9.2 Absolute and relative tolerance

For floating-point results, a common comparison rule is:

$$
|a-b| \leq \epsilon_{\mathrm{abs}} + \epsilon_{\mathrm{rel}}\,|b|
$$

where:

* $a$ is the actual result.
* $b$ is the reference result.
* $\epsilon_{\mathrm{abs}}$ is the absolute tolerance.
* $\epsilon_{\mathrm{rel}}$ is the relative tolerance.

This formula is a starting point, not a universal policy.

Tolerance values MUST be chosen according to dtype, algorithm, reduction length, expected numerical conditioning, and backend behavior.

Tests MUST NOT use excessively loose tolerances merely to make incorrect implementations pass.

### 9.3 NaN and infinity

Comparisons involving NaN require explicit treatment because ordinary equality checks do not behave as they do for finite values.

Tests SHOULD define the expected behavior for:

* NaN inputs.
* Positive and negative infinity.
* Overflow and underflow.
* Division by zero.
* Degenerate reductions.
* Operations whose mathematical result is undefined for a particular input.

Whether these values are accepted, propagated, or rejected MUST follow the operator contract.

### 9.4 Accumulation order

Floating-point addition is not associative.

Different valid implementations may produce slightly different results because of:

* Vectorization.
* Parallel reductions.
* Different accumulation precision.
* Different operation ordering.
* Compiler and backend behavior.

Tests SHOULD validate the mathematical contract within justified tolerances rather than demand bitwise equality without a specific requirement.

Bitwise reproducibility MAY be required for explicitly deterministic operations or serialized artifacts. Such requirements MUST be documented separately.

### 9.5 Dtype coverage

Every supported dtype SHOULD have appropriate tests for its relevant behavior.

Do not claim support for a dtype merely because the operation compiles for it.

Where an operation is unsupported for a dtype, test the documented rejection behavior.

### 9.6 Numerical stability

Algorithms with known stability concerns SHOULD include tests for representative difficult inputs.

Examples include:

* Large and small magnitudes.
* Cancellation-prone sums.
* Extreme logits in softmax.
* Small variance in normalization.
* Nearly singular or ill-conditioned inputs.
* Long reductions.
* Gradients near zero.

These tests SHOULD target realistic failure modes rather than arbitrary random numbers alone.

---

## 10. Automatic Differentiation and Gradient Testing

Automatic differentiation requires verification of both forward values and gradient behavior.

### 10.1 Forward correctness

Each differentiable operator MUST have forward numerical tests covering its supported input domain.

### 10.2 Gradient correctness

Each new differentiable operator SHOULD be tested against an independent gradient reference.

Where practical, use finite differences:

$$
\frac{\partial f}{\partial x_i}
\approx
\frac{f(x+h e_i)-f(x-h e_i)}{2h}
$$

where $e_i$ is the unit direction for component $i$.

Finite differences are an approximation and can be inaccurate for poorly conditioned functions, inappropriate step sizes, or non-smooth points. Test inputs and step sizes MUST be chosen accordingly.

### 10.3 Required considerations

Gradient tests SHOULD cover:

* Representative inputs.
* Shape consistency.
* Gradient dimensions.
* Numerical agreement with a reference.
* Broadcasting behavior.
* Reduction behavior.
* Accumulation when multiple paths contribute to a gradient.
* Zero or boundary gradients where defined.
* Error behavior for unsupported inputs.

### 10.4 Non-differentiable points

For non-smooth functions, tests MUST follow the documented derivative convention or domain restriction.

Do not compare against a finite-difference result at a point where the derivative is undefined and then treat the disagreement as proof of an implementation defect.

### 10.5 No accidental state leakage

Tests SHOULD verify that repeated forward and backward operations do not unintentionally retain request-specific or graph-specific state in shared model parameters.

The expected lifetime of graph state, gradient buffers, and intermediate values MUST follow the autograd contract.

---

## 11. Operator Acceptance Standard

Every new tensor or compute operator MUST include:

1. Boundary tests — supported edges, invalid inputs, shape rules, and overflow conditions where applicable.
2. Numerical tests — comparison against a justified independent reference.
3. At least one benchmark — a reproducible baseline for a representative workload.

In addition, the operator SHOULD include:

* Dtype coverage.
* Layout coverage.
* Error-path tests.
* Property tests for applicable invariants.
* Gradient tests if the operator participates in automatic differentiation.
* Integration tests if it changes a public or cross-crate contract.

The benchmark requirement does not mean every new operator must demonstrate a performance improvement. It means its performance characteristics can be measured.

A new operator is not complete merely because its forward path produces one expected result.

---

## 12. Property-Based and Metamorphic Testing

Property-based tests SHOULD be used when an operation has useful general invariants over a broad input space.

Potential properties include:

* Shape preservation.
* Element-count consistency.
* Round-trip serialization.
* Permutation and inverse-permutation behavior.
* Identity operations.
* Algebraic identities within valid numerical tolerances.
* Consistency between equivalent layouts.
* Deterministic output for deterministic operations.

### 12.1 Generated inputs

Generated inputs MUST respect the preconditions of the property being tested.

If malformed input is intentionally generated, the expected behavior MUST be explicit.

Generators SHOULD remain bounded so that ordinary test runs do not allocate uncontrolled amounts of memory or run for unpredictable durations.

### 12.2 Metamorphic testing

Metamorphic tests verify relationships between related executions when a direct reference result is expensive or unavailable.

Examples include:

* Applying an inverse permutation restores the original tensor.
* Saving and loading a supported checkpoint preserves the documented contents.
* An identity transformation preserves the expected output.
* Equivalent input representations produce equivalent results where the contract guarantees equivalence.

The relationship itself MUST be mathematically and semantically justified.

### 12.3 Reproducibility

When generated test cases fail, the failure SHOULD preserve enough information to reproduce the case, including the seed or minimized input where supported by the chosen testing framework.

A failing generated case SHOULD be promoted to a focused regression test when it exposes a real defect that could otherwise be difficult to reproduce.

---

## 13. Regression Testing

Every reproducible bug fix MUST include a regression test unless a test is genuinely impractical or would not meaningfully detect recurrence.

When a regression test cannot be added, the change description MUST explain why and identify the alternative verification method.

A good regression test:

* Reproduces the original failure.
* Fails against the defective behavior.
* Passes against the corrected behavior.
* Verifies the intended contract rather than the exact patch implementation.
* Remains small enough to diagnose.

Avoid tests that pass because the affected code path is never exercised.

For numerical regressions, the test SHOULD identify the relevant input domain and expected tolerance.

For performance regressions, use the benchmark and measurement policy rather than treating ordinary test duration as proof of a regression.

---

## 14. Integration and End-to-End Testing

Integration tests verify that components interact correctly across public interfaces.

The workspace SHOULD maintain a small, reliable set of end-to-end tests for critical workflows.

Examples include:

### 14.1 Model loading and inference

* Load a supported small fixture.
* Validate model configuration.
* Run a deterministic or appropriately controlled inference case.
* Verify output structure and relevant numerical properties.
* Exercise a documented failure path.

### 14.2 Checkpoint round trip

* Construct a small known model state.
* Write a checkpoint.
* Read it back.
* Verify metadata, parameter names, shapes, dtypes, and values according to the format contract.
* Exercise malformed or incompatible checkpoint behavior.

### 14.3 Training workflow

* Construct a small deterministic dataset.
* Execute a bounded training workflow.
* Verify that the expected state changes occur.
* Check relevant optimizer or gradient invariants.
* Verify checkpoint or cancellation behavior where applicable.

### 14.4 Tokenizer and data pipeline

* Process a small known input.
* Verify token IDs, offsets, or decoded output according to the public contract.
* Exercise malformed input and relevant boundary cases.

End-to-end tests SHOULD use the smallest fixture that validates the interaction.

They SHOULD NOT download large models, require external network services, or consume excessive resources during routine local development unless explicitly classified as a separate integration suite.

---

## 15. Error-Path and Fault-Injection Testing

Error behavior is part of the contract and MUST be verified where relevant.

Tests SHOULD cover:

* Invalid input.
* Missing resources.
* Permission or I/O failures.
* Corrupt serialized data.
* Unsupported operations.
* Resource-limit rejection.
* Backend initialization failure.
* Cancellation and timeout.
* Partial writes or interrupted workflows.
* Retry exhaustion.
* Cleanup after failure.

### 15.1 Fault injection

When practical, use traits, controlled test doubles, temporary files, or other explicit seams to simulate failures.

Fault injection MUST remain test-focused and MUST NOT distort the production architecture solely to accommodate a test.

Do not introduce broad dependency-injection frameworks when a small trait or local test double is sufficient.

### 15.2 State consistency

For operations that mutate state, tests SHOULD verify what remains true after failure.

Examples include:

* A failed validation does not mutate model parameters.
* A failed checkpoint operation does not publish an incomplete artifact as committed.
* A cancelled request does not continue producing user-visible output indefinitely.
* A failed optimizer operation follows its documented partial-update semantics.

Do not assert rollback guarantees that the implementation does not actually provide.

---

## 16. Determinism and Test Isolation

Routine tests SHOULD be deterministic.

Tests MUST NOT rely on unspecified ordering, uncontrolled global mutable state, or accidental timing assumptions.

### 16.1 Randomness

Randomized tests SHOULD use controlled seeds when reproducibility matters.

The seed and relevant generator configuration SHOULD be reported when a failure depends on random input.

Do not fix a seed in a way that makes a test appear comprehensive while repeatedly exercising only one trivial case. A useful strategy may combine deterministic regression cases with bounded randomized exploration.

### 16.2 Parallel execution

Tests SHOULD remain correct when the Rust test harness runs them concurrently.

Avoid sharing mutable files, environment variables, ports, or global state without appropriate isolation.

Tests that require exclusive resources SHOULD use explicit isolation or be assigned to a separately controlled test target.

### 16.3 Time and concurrency

Avoid arbitrary sleeps as synchronization mechanisms.

Prefer explicit synchronization primitives, bounded deadlines, deterministic state transitions, and controlled test doubles.

Tests for concurrency and cancellation SHOULD use timeouts to avoid hanging the suite, but the timeout MUST be a safety bound rather than the primary proof of correctness.

### 16.4 Network access

Routine unit tests MUST NOT require external network access.

Tests that genuinely require network or external services SHOULD be isolated, clearly identified, and excluded from the default fast local test loop unless the project explicitly chooses otherwise.

### 16.5 Environment assumptions

Tests MUST document required external tools, hardware, environment variables, or operating-system features when those requirements cannot be avoided.

A test that cannot run in a particular environment MUST be reported as not executed or unsupported, not as passed.

---

## 17. Architecture and Dependency Tests

The workspace’s architectural rules SHOULD be enforced automatically wherever practical.

The checks SHOULD cover:

* Permitted crate dependency directions.
* Forbidden dependency cycles.
* Production dependencies on testkit.
* Production dependencies on benchmark-only components.
* Application-layer dependencies leaking into core crates.
* Accidental dependency additions.
* Prohibited unsafe usage.
* Repository-specific source and build constraints.

A check MAY use Cargo metadata, manifest inspection, source scanning, or another appropriately scoped mechanism.

### 17.1 Prefer structural checks

Dependency checks SHOULD inspect the resolved dependency graph rather than infer architecture solely from filenames or source imports.

Source scanning MAY supplement graph checks for requirements that dependency metadata cannot establish.

### 17.2 Avoid brittle policy checks

A policy script SHOULD have a documented purpose, deterministic output, and tests for its own behavior.

Do not create broad regular-expression checks that reject legitimate code while missing the prohibited case they were intended to detect.

### 17.3 Architecture changes

When a change modifies crate boundaries or introduces a dependency, reviewers MUST verify the actual dependency graph.

Documentation diagrams are not evidence that the resolved Cargo graph conforms.

---

## 18. Benchmarking and Performance Validation

Benchmarks are a separate source of evidence from correctness tests.

Every new operator MUST have at least one benchmark. Performance-sensitive changes SHOULD include before-and-after measurements when a valid baseline exists.

### 18.1 Benchmark requirements

A benchmark SHOULD document:

* Operation and workload.
* Input dimensions and dtype.
* Layout and relevant configuration.
* Compiler and Rust toolchain.
* Build profile and feature flags.
* Hardware and operating system.
* Warm-up and measurement strategy.
* Sample count or iteration policy.
* Reported statistic and variability.
* Allocation or memory behavior when relevant.

### 18.2 Benchmark isolation

Benchmarks SHOULD avoid unrelated work inside the timed region.

For example, a kernel benchmark SHOULD NOT accidentally include model loading, file I/O, or fixture generation unless those costs are intentionally part of the measured operation.

### 18.3 Correctness before performance

An implementation MUST satisfy its correctness contract before benchmark results can justify adoption.

A faster implementation that returns incorrect values is a regression, not an optimization.

### 18.4 Performance claims

Use the performance status vocabulary defined in docs/performance.md section 19.

Do not claim a performance improvement based solely on code inspection, intuition, or a single noisy observation.

See docs/performance.md for the full measurement and reporting policy.

---

## 19. Test Commands and Validation Levels

The exact commands MUST follow the workspace’s actual Cargo configuration, toolchain, supported targets, and feature policy.

The following are candidate baseline checks. They MUST be verified against the repository before being treated as mandatory CI commands.

### 19.1 Formatting

```sh
cargo fmt --all -- --check
```

### 19.2 Workspace compilation

```sh
cargo check --workspace --all-targets
```

### 19.3 Linting

```sh
cargo clippy --workspace --all-targets -- -D warnings
```

### 19.4 Unit and integration tests

```sh
cargo test --workspace
```

These commands may need additional feature, platform, or target-specific invocations.

A successful command verifies only the targets and configurations it actually covered.

### 19.5 Validation levels

The development workflow SHOULD use multiple validation levels.

Level 1 — Focused validation

Run the smallest relevant tests and checks for the changed component.

Level 2 — Crate validation

Run the affected crate’s tests and relevant integration tests.

Level 3 — Workspace validation

Run the applicable workspace compilation, lint, and test checks.

Level 4 — Specialized validation

Run numerical stress tests, architecture checks, fault injection, benchmarks, or hardware-specific tests when the change requires them.

Level 5 — Release validation

Run the project-defined release suite for compatibility, critical workflows, and supported configurations.

Not every edit requires every level. The selected validation scope MUST be justified by the change’s risk and contract.

---

## 20. CI and Continuous Validation

Continuous integration SHOULD provide reliable, repeatable evidence for each change.

The CI pipeline SHOULD prioritize fast, high-signal checks before expensive tests.

A recommended logical sequence is:

1. Formatting and static policy checks.
2. Compilation and linting.
3. Focused unit and contract tests.
4. Workspace integration tests.
5. Specialized numerical and architecture tests.
6. Scheduled or dedicated stress tests.
7. Benchmark collection and performance regression analysis.
8. Release validation.

The exact stages depend on repository size, available hardware, and CI capabilities.

### 20.1 Required versus optional checks

Each check SHOULD be classified as:

* Required for every change.
* Required for selected paths or crates.
* Required before release.
* Scheduled.
* Informational.

A required check MUST NOT silently become optional merely because it is inconvenient.

### 20.2 Failed infrastructure versus failed tests

CI MUST distinguish:

* A test assertion failure.
* A compilation failure.
* A timeout.
* An unavailable dependency or service.
* An unsupported platform.
* An infrastructure failure.

Infrastructure failure does not establish that the code is incorrect, but it also does not establish that the code is correct.

### 20.3 Flaky tests

A flaky test is a reliability defect in the verification system.

When a test is found to be flaky:

1. Preserve the failure evidence.
2. Identify the source of nondeterminism.
3. Fix the cause where practical.
4. Add isolation or bounded synchronization if needed.
5. Document any temporary quarantine.
6. Track the remediation.

Repeatedly rerunning a flaky test until it passes MUST NOT be treated as a reliable verification strategy.

---

## 21. Test Maintenance

Tests are production engineering assets and MUST be maintained alongside implementation changes.

### 21.1 Refactoring

A valid refactor SHOULD preserve the behavior established by existing contract tests.

When a test fails after refactoring, determine whether:

* The implementation introduced a regression.
* The test encoded an undocumented implementation detail.
* The intended contract changed.
* The reference implementation or test fixture is incorrect.

Do not automatically change the expected result to match the new implementation.

### 21.2 Removing tests

A test MAY be removed when its contract is obsolete, it is redundant with stronger evidence, or it cannot provide meaningful verification.

The removal MUST NOT leave a critical behavior untested without an explicit rationale.

### 21.3 Test performance

The default local test loop SHOULD remain fast enough to encourage frequent execution.

Expensive tests SHOULD be separated by scope or execution profile while retaining clear requirements for when they must run.

Test runtime SHOULD be measured when it becomes a significant engineering bottleneck.

### 21.4 Test helpers

Test helpers SHOULD reduce genuine duplication without hiding preconditions, side effects, or failure behavior.

Avoid helpers that silently substitute defaults for invalid fixture data or swallow setup failures.

---

## 22. AI-Assisted Testing Rules

AI coding agents MUST treat test output and repository state as evidence, not as assumptions.

An AI agent MUST NOT:

* Claim a test passed if it was not executed.
* Invent benchmark results.
* Claim complete coverage from a successful compilation.
* Remove a failing test merely to make CI green without understanding the failure.
* Weaken numerical tolerances without a mathematical justification.
* Replace a typed error assertion with a weaker assertion merely to bypass a failure.
* Add a test that never reaches the affected code path.
* Claim architecture compliance without checking the relevant dependency graph.
* Modify unrelated tests to hide a regression.

When test execution is unavailable, the agent MUST state which checks were not run and why.

When a test fails, the report SHOULD include the relevant failure, likely cause when supported by evidence, and the next verification step.

AI-generated tests MUST be reviewed for independence from the production implementation. A test that merely reproduces the same algorithm is not automatically a useful reference.

---

## 23. Definition of Done

A change is complete only when its applicable requirements have been satisfied.

General changes

* [ ] Acceptance criteria are explicit.
* [ ] Relevant unit or contract tests exist.
* [ ] Relevant failure paths are covered.
* [ ] Existing tests continue to pass within the verified scope.
* [ ] Public behavior and documentation agree.
* [ ] No unrelated changes are included.
* [ ] Test results and limitations are reported accurately.

Bug fixes

* [ ] The defect has a reproducible test when practical.
* [ ] The test verifies the original failure mode.
* [ ] The correction does not introduce an unrelated workaround.
* [ ] Relevant regression tests pass.

New operators

* [ ] Boundary tests exist.
* [ ] Numerical correctness tests exist.
* [ ] At least one benchmark exists.
* [ ] Supported dtype and layout behavior is documented and tested as applicable.
* [ ] Error behavior is tested where relevant.
* [ ] Gradient behavior is tested when applicable.
* [ ] Performance claims match the measurement evidence.

Architecture changes

* [ ] Dependency directions have been checked.
* [ ] No forbidden dependency cycle has been introduced.
* [ ] Public contracts and affected integration tests have been reviewed.
* [ ] Documentation reflects the intended architecture.
* [ ] The change remains independently testable.

Performance changes

* [ ] Correctness has been established.
* [ ] The baseline is appropriate and comparable.
* [ ] Benchmark methodology and environment are recorded.
* [ ] Variability and measurement limitations are considered.
* [ ] The reported conclusion matches the evidence.

---

## 24. Conformance and Adoption

This document defines the target testing standard. It does not assert that every existing crate currently satisfies it.

Adoption SHOULD proceed incrementally.

For each crate, record:

* Primary contracts.
* Existing test coverage.
* Known untested failure modes.
* Required boundary and numerical tests.
* Integration requirements.
* Architecture checks.
* Benchmark requirements.
* Verification status.

Use the following states:

* Proposed — the testing requirement is specified but not verified.
* Verified — the relevant implementation and tests provide the required evidence.
* Non-conforming — a known testing or behavior gap exists.
* Blocked — validation depends on an unresolved design, dependency, or environment issue.

A passing workspace test command MUST NOT be interpreted as proof that all contracts, error paths, supported platforms, and numerical edge cases have been validated.

---

## 25. Final Principle

A test suite exists to detect incorrect behavior, preserve contracts, and make engineering claims verifiable.

Test the contract. Test the boundaries. Test the failure paths. Measure performance separately. Report only what the evidence supports.

The objective is not to maximize test count. It is to maximize the amount of meaningful, independent confidence obtained from a maintainable verification system.