# Development Workflow and Execution Protocol

Status: Proposed
Scope: All development, refactoring, debugging, testing, optimization, and documentation changes in the llm/ workspace
Audience: Engineers, reviewers, maintainers, and AI coding agents

---

## 1. Mission

This document defines the mandatory execution procedure for engineering tasks in the llm/ workspace.

Its purpose is to make every change small enough to understand, precise enough to review, and rigorous enough to verify.

Development is treated as a controlled engineering process rather than a sequence of code-generation operations.

Every task MUST establish:

1. What problem is being solved.
2. Which existing contracts and dependencies are involved.
3. What may and may not change.
4. How correctness will be established.
5. How the change will be validated.
6. What evidence supports the completion claim.
7. What remains unresolved.

The workflow applies equally to human contributors and AI-assisted development.

The governing principle is:

Understand first. Plan narrowly. Change minimally. Verify independently. Report honestly.

---

## 2. Non-Negotiable Engineering Rules

Every task MUST comply with the following rules.

### 2.1 Understand Before Modifying

Before changing a crate, read its responsibilities, manifest, public API, relevant implementation, tests, and dependency relationships.

Do not infer implementation details from filenames or directory structure alone.

### 2.2 Preserve Dependency Direction

Do not introduce dependency cycles or bypass established crate boundaries.

Every new workspace dependency edge requires an explicit reason.

### 2.3 Preserve Explicit Error Handling

Production runtime errors MUST NOT be handled using panic!, unwrap(), or expect() as substitutes for proper error propagation. Recoverable failures follow the contracts defined in docs/error-handling.md.

### 2.4 Require Evidence for Performance Claims

No performance improvement may be claimed without comparable, reproducible benchmark data.

An optimization may be implemented experimentally, but its performance status must remain unverified until measured.

### 2.5 Prohibit Unsafe Code

The workspace prohibits unsafe code entirely, enforced by the workspace lint configuration (`unsafe_code = "deny"`).

New unsafe code MUST NOT be introduced. Safe Rust MUST be used even when unsafe would appear to simplify ownership or API design.

### 2.6 Keep Request State Isolated

Model weight objects MUST NOT contain request-scoped mutable state.

Inference changes must preserve the separation between shared model data and per-request execution state.

### 2.7 One Commit, One Independently Testable Stage

Each commit MUST represent one coherent, independently testable stage.

Unrelated features, refactors, formatting changes, and optimizations MUST NOT be combined merely to reduce the number of commits.

### 2.8 Documentation Must Match Reality

A design proposal is not an implementation.

A test that has not been executed is not a passing test.

An unmeasured optimization is not a demonstrated improvement.

---

## 3. Task Classification

Before execution, classify the task by its primary objective.

| Task type | Primary concern | Required evidence |
| --- | --- | --- |
| Documentation | Accuracy and consistency | Cross-document review |
| Bug fix | Reproduction and regression prevention | Failing-before/fixed-after test where feasible |
| New feature | Contract and behavior | Acceptance tests |
| New operator | Numerical correctness and boundaries | Reference tests and benchmark |
| Refactoring | Behavioral preservation | Relevant regression tests |
| Architecture change | Dependency and ownership integrity | Dependency graph and architectural review |
| Performance optimization | Measurable improvement | Comparable benchmark results |
| Safety-sensitive change | Memory and resource correctness | Safety argument and targeted validation |
| Dependency update | Compatibility and supply-chain risk | Build, tests, and dependency review |
| Release preparation | Integration and reproducibility | Full applicable release checks |

A task may affect multiple categories. Its primary category determines the main acceptance criteria, while every affected secondary category contributes additional requirements.

Classification MUST NOT be used to avoid required testing.

---

## 4. The Execution Lifecycle

Every implementation task follows the lifecycle below.

```text
Request
  |
  v
1. Understand the objective
  |
  v
2. Inspect repository and dependencies
  |
  v
3. Establish baseline
  |
  v
4. Write execution plan
  |
  v
5. Define acceptance criteria
  |
  v
6. Implement the smallest change
  |
  v
7. Validate behavior and boundaries
  |
  v
8. Review diff and architecture
  |
  v
9. Record evidence and limitations
  |
  v
10. Commit one completed stage
```

If a validation step fails, return to the smallest relevant earlier stage.

Do not continue adding features while a failure invalidates the assumptions required for the next stage.

Not every documentation-only change requires compiling the entire workspace. Validation depth must be appropriate to the change, but the reason for omitting a check must be clear.

---

## 5. Phase A — Understand the Objective

Before touching files, produce a concise problem statement.

The statement MUST identify:

* The current problem or missing capability.
* The intended observable outcome.
* The affected subsystem.
* Relevant architectural constraints.
* Explicit non-goals.

A task is insufficiently specified if its completion cannot be determined objectively.

### 5.1 Separate Requirements from Assumptions

Classify relevant information as:

* Confirmed: supported by repository contents, tests, or explicit requirements.
* Assumed: plausible but not yet verified.
* Unknown: requires inspection or a decision.
* Out of scope: deliberately excluded from the task.

Do not silently convert assumptions into implementation requirements.

If a missing decision could change correctness, safety, public API behavior, or dependency direction, resolve it before implementation.

### 5.2 Define the Smallest Useful Outcome

Prefer a narrow, observable result over a broad goal.

For example, replace:

“Improve inference performance.”

With:

“Establish a reproducible baseline for the existing CPU decode path, identify its dominant measured cost, and report the result without changing runtime behavior.”

The narrower task creates useful evidence without prematurely committing to an optimization.

---

## 6. Phase B — Repository Reconnaissance

Before modifying a target crate, inspect its actual implementation.

### 6.1 Required Inspection

At minimum, inspect:

1. Repository-level AGENTS.md.
2. Root Cargo.toml.
3. rust-toolchain.toml and relevant build configuration.
4. Target crate Cargo.toml.
5. Target crate public entry points and module structure.
6. Relevant source implementations.
7. Existing tests and benchmarks.
8. Direct workspace dependencies.
9. Relevant callers and downstream consumers.
10. Existing documentation and architectural decisions.

For a cross-crate change, inspect every affected crate rather than treating the initiating crate as the entire scope.

### 6.2 Establish the Actual Dependency Graph

Use Cargo metadata and dependency-tree inspection to identify the real graph.

```sh
cargo metadata --workspace --format-version 1
cargo tree --workspace
```

Inspect normal, build, development, optional, and feature-dependent relationships as appropriate.

Do not assume a dependency is harmless because it appears only in a manifest. Its feature activation and transitive closure may change the effective architecture.

### 6.3 Inspect the Working Tree

Before editing:

```sh
git status --short
git diff --check
git diff
```

These commands help identify existing user changes and formatting problems in the current diff.

Do not overwrite, discard, or reformat unrelated user modifications.

Do not use destructive Git operations to obtain a clean working tree without explicit authorization.

### 6.4 Reconnaissance Output

The reconnaissance stage MUST establish:

* Target crate and owned responsibilities.
* Relevant public contracts.
* Existing dependency paths.
* Important invariants.
* Existing tests and benchmarks.
* Potential regression risks.
* Unknowns that may block implementation.

If the repository is unavailable, explicitly state that source-level reconnaissance has not occurred.

---

## 7. Phase C — Establish the Baseline

Before changing behavior, determine the relevant starting state.

### 7.1 Correctness Baseline

Run the narrowest relevant existing tests.

For changes affecting shared contracts, public APIs, or multiple crates, broaden the test scope accordingly.

Record failures that existed before the change.

A pre-existing failure must not automatically be attributed to the current task.

### 7.2 Build Baseline

Verify the relevant toolchain and build configuration.

Use workspace checks when the scope warrants them:

```sh
cargo check --workspace --all-targets
```

For smaller changes, targeted checks may be sufficient during iteration, followed by broader checks before completion.

### 7.3 Performance Baseline

For performance work, establish the baseline before optimizing.

Record:

* Machine and CPU information.
* Operating system.
* Rust toolchain and relevant compiler settings.
* Build profile and enabled features.
* Input dimensions and workload.
* Warm-up and measurement procedures.
* Sample counts and statistical summaries.
* Memory consumption where relevant.

If a reliable baseline cannot be collected, the task may still proceed as exploratory work, but performance improvement cannot be claimed.

### 7.4 Baseline Report

Record whether each relevant check is:

* Passed.
* Failed.
* Not run.
* Blocked.

A failed command must not be described as successful merely because later commands pass.

---

## 8. Phase D — Write the Execution Plan

Every nontrivial task MUST have a written plan before implementation.

The plan MUST specify:

1. Objective.
2. Target files and crates.
3. Expected behavior change.
4. Expected dependency changes.
5. Relevant invariants.
6. Test strategy.
7. Acceptance criteria.
8. Rollback or recovery considerations where appropriate.

### 8.1 Change Budget

Define the smallest reasonable change set.

Prefer:

* One responsibility per stage.
* Narrow public API changes.
* Reuse of existing abstractions.
* Localized fixes.
* Explicit ownership transfer.

Avoid:

* Unrelated cleanup.
* Repository-wide renaming.
* Broad module restructuring without evidence.
* Premature generalization.
* Simultaneous replacement of several foundational abstractions.

The plan may change when new evidence appears. Any material change in scope MUST be recorded before proceeding.

### 8.2 Dependency Change Budget

If a task introduces a new dependency edge, state:

* Why the edge is needed.
* Which crate owns the required capability.
* Why an existing contract is insufficient.
* Whether the edge changes the dependency graph.
* How cycles and forbidden edges will be checked.

If a proposed change requires a reverse dependency, stop and redesign the interface.

---

## 9. Phase E — Define Acceptance Criteria

Acceptance criteria MUST be observable and testable.

Weak criterion:

“The tensor implementation is more robust.”

Strong criteria:

* Shape multiplication detects overflow.
* Invalid dimensions return a documented error.
* Existing valid inputs preserve their behavior.
* Boundary tests cover zero, minimum, and overflow-related cases where supported.
* Relevant tests pass under the configured toolchain.

### 9.1 Acceptance Categories

Select the categories relevant to the task:

* API behavior.
* Numerical correctness.
* Error propagation.
* Dependency direction.
* Ownership and lifetime.
* Concurrency and request isolation.
* Resource limits.
* Serialization compatibility.
* Performance measurement.
* Documentation consistency.

### 9.2 Negative Requirements

Explicitly state what must not change.

Examples:

* No new runtime dependency.
* No public API changes.
* No additional unsafe.
* No change to model weight layout.
* No change to inference output semantics.
* No change to unrelated formatting.

Negative requirements are important because a locally correct implementation may still violate system-level constraints.

---

## 10. Phase F — Implement the Smallest Correct Change

Implementation MUST follow the approved plan.

### 10.1 Preserve Existing Contracts

Before changing a public interface, identify all callers and their assumptions.

Do not silently change:

* Ownership behavior.
* Error semantics.
* Shape conventions.
* Numerical tolerances.
* Cancellation behavior.
* Resource-lifecycle guarantees.

If a contract must change, treat it as an explicit API or architecture task.

### 10.2 Prefer Composition

Use existing narrow traits and concrete types when they fit the requirement.

Do not add a generic framework when a concrete implementation is sufficient.

Do not add runtime polymorphism unless runtime selection or another concrete requirement justifies it.

Do not create a universal context object that absorbs unrelated state.

### 10.3 Error Handling

Production runtime paths MUST propagate recoverable errors explicitly. Error semantics, conversion, assertion policy, and input validation are defined by docs/error-handling.md.

### 10.4 Unsafe Changes

The workspace prohibits unsafe code entirely (see AGENTS.md section 6). New unsafe code MUST NOT be introduced.

If a requirement appears to need unsafe, redesign it with safe Rust, or report the limitation instead of weakening the prohibition.

### 10.5 Keep Changes Reviewable

Do not hide behavioral changes inside formatting-only commits.

Do not modify generated or unrelated files unless required by the build or explicitly included in the plan.

Review the diff continuously rather than waiting until the end of a large change.

---

## 11. Phase G — Validate in Increasing Scope

Validation should begin with cheap, targeted checks and expand according to the impact of the change.

Level 1: Local Feedback

* Compile the affected code.
* Run targeted unit tests.
* Check formatting of changed Rust code.
* Inspect the local diff.

Level 2: Contract Validation

* Test valid and invalid inputs.
* Verify documented error behavior.
* Test relevant shape and numeric boundaries.
* Verify ownership and lifecycle assumptions.

Level 3: Integration Validation

* Compile affected dependent crates.
* Run relevant integration tests.
* Verify public API compatibility.
* Check dependency direction and feature combinations where relevant.

Level 4: Workspace Validation

For changes with broad impact, run applicable workspace checks:

```sh
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

These commands are the proposed baseline. The actual workspace may require feature-specific, platform-specific, or toolchain-specific adjustments.

A check is complete only when the command actually runs and its result is observed.

Level 5: Specialized Validation

Add specialized checks when applicable:

* Architecture checks for dependency-boundary changes.
* Numerical reference tests for computational operators.
* Request-isolation tests for inference changes.
* Serialization compatibility tests for persistence changes.
* Reproducible benchmarks for new operators and performance work.

Passing a broader test suite does not eliminate a required specialized test.

---

## 12. Phase H — Review the Diff

Before declaring completion, review the final diff as an independent reviewer would.

Ask the following questions.

Correctness

* Does the change solve the stated problem?
* Are boundary conditions covered?
* Are error paths explicit?
* Could integer overflow, invalid layout, or resource exhaustion cause incorrect behavior?
* Are numerical assumptions documented?

Architecture

* Are crate responsibilities preserved?
* Has a dependency cycle been introduced?
* Are public interfaces unnecessarily broad?
* Is dynamic dispatch justified?
* Has a lower-level crate acquired higher-level knowledge?

State and Ownership

* Is mutable state owned by the correct object?
* Can two requests accidentally share mutable execution state?
* Are cancellation and early termination handled?
* Are resources released on failure?

Maintainability

* Is the solution simpler than the alternatives?
* Are new abstractions necessary?
* Are invariants visible in the code?
* Does documentation match the implementation?
* Have unrelated changes been removed?

Evidence

* Were the relevant checks executed?
* Are baseline failures distinguished from new failures?
* Are benchmark comparisons valid?
* Are limitations stated honestly?

Any unresolved correctness, safety, or architectural defect MUST be fixed or explicitly reported as blocking completion.

---

## 13. Phase I — Performance Work

Performance optimization requires its own execution discipline.

### 13.1 Optimization Procedure

1. Establish correctness.
2. Measure the baseline.
3. Identify a bottleneck from evidence.
4. Form a specific hypothesis.
5. Implement one meaningful change.
6. Re-run correctness tests.
7. Measure under comparable conditions.
8. Examine variability and regressions.
9. Record the outcome.

### 13.2 Claim Categories

Use the performance status vocabulary defined in docs/performance.md section 19.

Do not use “optimized” as a synonym for “changed.”

### 13.3 Report More Than a Single Number

Where relevant, report:

* Baseline and candidate results.
* Workload and environment.
* Sample count.
* Median or other suitable summary.
* Variability.
* Absolute and relative differences.
* Memory impact.
* Known limitations.

Do not select only favorable workloads while ignoring material regressions in the intended operating range.

---

## 14. Phase J — Documentation and Change Records

Update documentation in the same stage when implementation changes affect:

* Public APIs.
* Crate responsibilities.
* Dependency relationships.
* Error semantics.
* Resource ownership.
* Supported formats.
* Numerical behavior.
* Performance claims.

For significant architectural decisions, add a decision record.

Documentation must distinguish verified behavior from intended behavior.

Do not claim support for an unimplemented backend, operator, file format, or execution mode.

---

## 15. Commit Discipline

Each commit MUST complete one independently testable stage.

A commit should answer one primary question:

“What single coherent engineering outcome does this change establish?”

Examples of acceptable stages:

* Define and test checked tensor-size arithmetic.
* Add one CPU operator with correctness tests and a benchmark.
* Introduce a request-scoped KV Cache owner and isolation tests.
* Add a dependency-graph architecture check.
* Document an approved public API contract.

Examples of unacceptable combined stages:

* Add several operators, redesign the tensor API, and optimize SIMD.
* Rewrite inference, update checkpoint formats, and refactor the CLI.
* Reformat the workspace while changing unrelated runtime behavior.

A stage may contain multiple files and tests when they are necessary to establish the same coherent outcome.

Do not split changes artificially when doing so would leave intermediate commits invalid, unsafe, or misleading.

### 15.1 Commit Message

Commit messages SHOULD communicate the affected domain and the completed outcome.

Example:

tensor: validate shape size overflow

The message must describe the actual change, not an intended future feature.

### 15.2 Pre-Commit Checklist

Before committing:

* Confirm the intended files are the only changed files.
* Inspect the staged diff.
* Run applicable validation.
* Confirm no secrets, generated artifacts, or unrelated changes are included.
* Record unresolved limitations.
* Verify that the commit represents one completed stage.

Never claim a commit exists unless it was actually created.

---

## 16. Failure and Recovery Protocol

When a command or test fails:

1. Preserve the exact failure information.
2. Determine whether the failure predates the change.
3. Identify the smallest affected scope.
4. Reproduce the failure where feasible.
5. Fix the cause rather than suppressing the symptom.
6. Add a regression test when appropriate.
7. Re-run the failing check and related validation.
8. Report remaining failures explicitly.

Do not:

* Delete a failing test solely to obtain a passing suite.
* Weaken assertions without a documented contract change.
* Ignore compiler warnings that reveal relevant defects.
* Convert an error into success-shaped output.
* Claim success because unrelated checks pass.
* Hide environmental limitations.

If the failure is outside the task scope and cannot safely be fixed within it, record it as a blocker or follow-up task.

---

## 17. AI-Assisted Development Protocol

AI coding tools MUST follow the same repository rules as human contributors.

### 17.1 Before Editing

The assistant must inspect the relevant repository instructions, manifests, source files, tests, and dependency boundaries.

It must not invent files, public APIs, test results, benchmark measurements, or existing implementation details.

### 17.2 During Editing

The assistant must:

* Work within the approved change scope.
* Prefer minimal patches.
* Preserve unrelated user modifications.
* Avoid introducing unrequested dependencies.
* Avoid speculative refactoring.
* Separate independent stages.
* Explain material architectural decisions.

### 17.3 Before Completion

The assistant must:

* Inspect the final diff.
* Run applicable checks when execution access is available.
* Report exactly which checks ran and their results.
* Distinguish verified facts from assumptions.
* Identify untested behavior and environmental blockers.
* Avoid claiming that files were changed, tests passed, or commits were created without evidence.

If repository access or execution capability is unavailable, the assistant may provide a proposed patch or document, but must identify it as unverified.

### 17.4 Scope Expansion

If inspection reveals a larger architectural issue:

1. Identify the issue.
2. Explain its impact.
3. Determine whether it blocks the current task.
4. Keep unrelated changes out of the current stage.
5. Propose a separate task when appropriate.

Do not turn every local fix into an unsolicited repository-wide redesign.

---

## 18. Completion Report

Every completed engineering task MUST produce a concise completion report.

Use the following structure:

Objective

What the task was intended to establish.

Changes

Which files and contracts changed, and why.

Architecture

Which crate boundaries and dependencies were affected.

Validation

For each relevant check, record:

* Exact command or test.
* Execution status.
* Result.
* Relevant failure details.

Performance

If applicable, provide baseline and candidate measurements, environment, methodology, and limitations.

If not measured, state that no performance improvement has been established.

Safety and Resources

Describe applicable ownership and resource-lifecycle considerations.

Limitations

List unresolved issues, skipped checks, and known unsupported behavior.

Commit

Provide the actual commit identifier only if a commit was created.

---

## 19. Definition of Done

A task is complete only when:

1. Its objective and acceptance criteria are satisfied.
2. The change stays within the approved scope.
3. Existing architectural boundaries remain valid.
4. Relevant errors and failure modes are handled explicitly.
5. Applicable tests have actually run and passed.
6. Required specialized validation has been performed.
7. Public contracts and documentation are consistent.
8. The final diff contains no unexplained changes.
9. Performance claims are supported by evidence.
10. Remaining limitations are disclosed.

A blocked task is not a completed task.

A partially implemented feature must be described as partial.

An unmeasured optimization must not be described as a demonstrated improvement.

A passing build must not be presented as proof of full correctness.

---

## 20. Final Principle

The goal of this workflow is not to maximize process overhead.

It is to minimize the cost of misunderstanding, hidden coupling, regressions, and unsupported claims.

For every change, prefer the smallest implementation that satisfies the contract and can be independently verified.

Do not add complexity without a concrete requirement.

Do not weaken safety or correctness for convenience.

Do not confuse activity with progress.

A task is complete when its intended result is established by evidence, its boundaries remain intact, and its remaining limitations are explicit.