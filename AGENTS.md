# AGENTS.md — Engineering Constitution

## 1. Mission

Build a minimal, reliable, composable LLM system in Rust.

The system MUST prioritize:

1. Correctness over premature optimization.
2. Explicit ownership over implicit shared state.
3. Minimal interfaces over oversized abstractions.
4. Static dispatch over unnecessary dynamic dispatch.
5. One-way dependencies over cross-layer coupling.
6. Measurable evidence over unsupported claims.
7. Small, independently verifiable changes over large rewrites.
8. Resource safety and deterministic failure handling.
9. Documentation and tests as part of implementation.
10. Simplicity that preserves correctness, not simplicity that removes necessary safeguards.

The architectural foundation is a minimal kernel composed of static crates and narrow traits.

## 2. Mandatory Preflight

Before modifying any crate:

1. Read the repository-level instructions.
2. Read the workspace `Cargo.toml`, toolchain configuration, and relevant build configuration.
3. Read the target crate's `Cargo.toml` and public API.
4. Read the target crate's source code and tests.
5. Inspect direct and transitive workspace dependency relationships.
6. Identify ownership boundaries, error propagation, resource limits, and existing invariants.
7. Establish the existing test and performance baseline when relevant.
8. Write a minimal change plan with explicit acceptance criteria.

Do not assume that the directory structure accurately represents the implemented architecture.

Do not rewrite existing implementations before understanding their responsibilities and dependencies.

If repository contents are unavailable, document the uncertainty. Never claim that an audit, implementation, or test has been completed without evidence.

## 3. Architecture Rules

### 3.1 Dependency Direction

* `lib` defines shared foundational types and MUST NOT depend on other workspace crates.
* `tensor` owns tensor metadata, storage abstractions, shape validation, and tensor views.
* `api` defines backend and execution contracts.
* `cpu` implements supported execution contracts without introducing reverse dependencies into foundational crates.
* `autograd` owns differentiation mechanics.
* `nn` owns reusable neural network components and parameter abstractions.
* `models` owns concrete model architectures.
* `infer` owns generation workflows and request-scoped inference state.
* `train` owns training workflows.
* `checkpoint` owns persistence of model and training state.
* `format` owns external format parsing and validation.
* `tokenizer` owns tokenization and text-template behavior.
* `data` owns dataset and batching abstractions.
* `optim` owns optimization algorithms and optimizer state.
* `context` owns explicitly defined execution controls and resource-management contracts.
* `cli` and `serve` are application-facing layers.
* `testkit` is development and test infrastructure, not a production dependency.
* `bench` owns benchmark executables and measurement orchestration, and MUST NOT be a production dependency.

These are intended responsibilities. Only `crates/tensor` currently exists in the workspace; the other members are declared in the root `Cargo.toml` but not yet implemented. Actual dependency edges MUST be verified against source code and manifests.

### 3.2 Dependency Constraints

* Workspace dependency cycles MUST NOT be introduced.
* Foundational crates MUST NOT depend on application-facing crates.
* Backend implementations MUST NOT become dependencies of their own abstraction layer.
* Production crates MUST NOT depend on `testkit`.
* Dependencies MUST NOT be introduced solely for convenience when a smaller existing abstraction is sufficient.
* Cross-layer calls MUST use explicit interfaces and ownership rules.
* Shared types MUST NOT become a justification for moving unrelated responsibilities into `lib`.

Every new dependency edge requires a clear architectural justification.

### 3.3 Static Composition

* Prefer generic traits and compile-time composition when runtime selection is unnecessary.
* Use dynamic dispatch only when it solves a demonstrated runtime extensibility requirement.
* Keep traits narrow and responsibility-specific.
* Avoid universal backend traits that combine unrelated execution, memory, synchronization, and model responsibilities.
* Avoid premature plugin systems, service locators, global mutable registries, and unnecessary indirection.
* Keep implementation details private unless they are required by a supported public contract.

Static dispatch is a design preference, not an excuse to create excessively generic APIs.

## 4. Error Handling

Production runtime paths MUST NOT use `panic!`, `unwrap()`, or `expect()` to handle runtime errors. Recoverable failures MUST use `Result`, and each crate owns the errors for its own responsibilities.

The authoritative error-handling standard is docs/error-handling.md. This section states only the non-negotiable repository-level rule; the full taxonomy, conversion, retry, cancellation, validation, and diagnostics rules are defined there.

Do not introduce custom error frameworks when standard Rust error types are sufficient.

## 5. Ownership and Concurrency

* Model weights MUST NOT contain request-scoped mutable state.
* Request state MUST have an explicit owner and lifecycle.
* KV caches, sampling state, cancellation state, and generation counters MUST NOT be implicitly shared between independent requests.
* Shared immutable data SHOULD be preferred when safe and practical.
* Mutable shared state MUST have documented synchronization and ownership semantics.
* Resource acquisition and release MUST be explicit and testable.
* Cancellation, errors, and early termination MUST not leave request state incorrectly reusable.
* Avoid global mutable state unless a concrete system requirement justifies it.

Concurrency safety must follow from ownership, synchronization, and lifecycle design, not assumptions about call order.

## 6. Unsafe Code

The workspace prohibits `unsafe` code entirely. This prohibition is enforced by the workspace lint configuration (`unsafe_code = "deny"` in the root `Cargo.toml`).

* Do not introduce `unsafe` blocks, `unsafe fn`, `unsafe trait`, `unsafe impl`, `unsafe extern`, or any other `unsafe` construct.
* Do not add `#[allow(unsafe_code)]` or otherwise weaken the lint to admit `unsafe`.
* If a requirement appears to need `unsafe`, redesign it with safe Rust, or stop and report the limitation instead of weakening this rule.
* Do not claim that passing tests establish memory safety; the prohibition is absolute and independent of test evidence.

## 7. Operator Development

Every new computational operator MUST include:

1. A documented contract covering inputs, outputs, shapes, dtypes, layouts, and supported devices.
2. Explicit validation of invalid inputs and boundary conditions.
3. Numerical correctness tests against an independent reference where practical.
4. Tests for relevant degenerate sizes and floating-point edge cases.
5. Error-path tests.
6. At least one reproducible performance benchmark.
7. A documented benchmark configuration and baseline.
8. A clear statement of unsupported behavior.

A benchmark is required even when no optimization claim is made.

Do not claim performance improvement without comparable measurements under documented conditions.

## 8. Performance Engineering

* Establish a baseline before optimization.
* Change one meaningful variable at a time where practical.
* Keep benchmark inputs and compiler settings reproducible.
* Record hardware, operating system, Rust toolchain, build profile, workload dimensions, and measurement method.
* Distinguish latency, throughput, memory consumption, and startup cost.
* Separate microbenchmarks from end-to-end measurements.
* Investigate measurement noise and run-to-run variance.
* Avoid accepting a slower implementation solely because it is more complicated or theoretically faster.
* Do not introduce SIMD or parallelism without a clear correctness and measurement plan.

Performance statements MUST match the evidence collected.

## 9. Testing and Validation

Use the narrowest relevant test scope during development, followed by broader checks when appropriate.

Expected validation categories include:

* Unit tests.
* API and backend contract tests.
* Shape and boundary tests.
* Numerical reference tests.
* Integration tests.
* Request-isolation tests.
* Serialization and checkpoint round-trip tests.
* Concurrency and resource-lifecycle tests.
* Performance benchmarks.

Do not report a check as passing unless it was actually executed and its result was observed.

Distinguish between:

* Not run.
* Failed.
* Passed.
* Blocked by environment or missing prerequisites.

A passing compilation does not establish numerical correctness. Passing tests do not establish performance improvement.

## 10. Documentation

* Update documentation when a public contract or architectural boundary changes.
* Document ownership, errors, resource limits, and unsupported behavior.
* Keep examples consistent with the actual implementation.
* Record significant architectural decisions, alternatives, and consequences.
* Avoid duplicating normative rules across multiple documents.
* Identify the authoritative document for each policy.
* Mark proposed designs as proposed until verified against implementation.
* Record conformance verdicts in docs/conformance.md.

Documentation MUST NOT claim that unimplemented capabilities exist.

## 11. Change and Commit Discipline

Each commit MUST complete one independently testable stage.

A stage should have:

* One primary objective.
* A defined change boundary.
* Explicit acceptance criteria.
* Relevant tests.
* Recorded validation results.
* A clear explanation of remaining limitations.

Do not combine unrelated refactoring, feature development, performance optimization, and formatting changes into one commit.

Do not modify unrelated files merely to make a change appear more comprehensive.

If a change reveals a separate architectural problem, document it and create a separate task unless it blocks correctness or safety.

## 12. Linux and Rust Engineering Discipline

Use Linux kernel engineering principles as guidance for:

* Small functions and clear control flow.
* Explicit error paths.
* Narrow interfaces.
* Ownership of resources.
* Reviewable patches.
* Avoidance of unnecessary complexity.
* Clear invariants and failure handling.

Apply Rust-specific ownership, borrowing, lifetime, type-safety, and memory-safety principles rather than mechanically copying C kernel conventions.

Do not claim kernel compliance unless the relevant requirements have actually been evaluated.

## 13. Definition of Done

A task is complete only when all applicable conditions hold:

* The change meets its stated objective.
* Dependency direction remains valid.
* Public contracts and error behavior are documented.
* Relevant tests are present and have been run.
* Formatting and applicable lint checks pass.
* Performance claims, if any, are supported by reproducible measurements.
* No `unsafe` code has been introduced.
* No unrelated behavior has been silently changed.
* Remaining limitations are stated.
* The implementation and validation status are reported accurately.

When a requirement cannot be satisfied, report the reason and the exact remaining work instead of silently weakening the requirement.

All engineering tasks MUST follow docs/development-workflow.md. Repository-level instructions define the global constraints; this document defines the execution lifecycle, validation requirements, and completion criteria.

- 修改前先阅读目标 crate 的职责和依赖方向。
- 不得新增循环依赖。
- 运行时错误不得使用 panic、unwrap 或 expect 处理。
- 不得在没有基准数据时声称性能提升。
- 禁止引入任何 unsafe 代码（由 workspace lint `unsafe_code = "deny"` 强制）。
- 新算子必须包含边界测试、数值测试和至少一个性能基准。
- 不得在模型权重对象中保存请求级可变状态。
- 每个提交只完成一个可测试的阶段。
- 严格按照 Linux / Unix 内核编码规范、顶级系统工程设计思想,组合式开发
- 整体架构遵循极简内核设计哲学，静态 Crate + Trait
- 测试组织以 docs/testing.md 为准：`crates/*/src` 禁止任何 `#[cfg(test)]` 内联测试（由 `scripts/checksrctests.sh` 机械门禁强制），全部单元/集成/回归测试置于仓库根 `tests/`，仅通过 crate 的公共 `pub` 接口触发，不得为测试目的放宽可见性；基准置于仓库根 `benches/`（共享工作负载由 `crates/bench` 提供），共享夹具与参考实现放入 `testkit`。