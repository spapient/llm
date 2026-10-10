# Performance Engineering and Benchmarking

Status: Proposed
Scope: All performance-sensitive crates, operators, execution paths, and benchmarks in the llm/ workspace
Audience: Engineers, reviewers, maintainers, and AI coding agents

---

## 1. Purpose

This document defines the performance engineering methodology for the llm/ workspace.

The objective is to achieve measurable, reproducible performance improvements without sacrificing correctness, numerical integrity, architectural simplicity, safety, or maintainability.

Performance is a system property. It depends on more than the speed of an individual function or operator.

Relevant factors include:

* Algorithmic complexity.
* Tensor layout and memory access patterns.
* Allocation and buffer reuse.
* Data movement and copying.
* Cache locality.
* CPU vectorization and parallel execution.
* Thread scheduling and synchronization.
* Operator dispatch overhead.
* Model architecture and execution structure.
* Batch size and sequence length.
* Input and output processing.
* Compilation settings and target hardware.
* Resource contention and workload characteristics.

A local optimization is valuable only when its benefits are relevant to the intended workload and its costs are understood.

This document establishes how to identify bottlenecks, design benchmarks, evaluate changes, prevent regressions, and report performance claims honestly.

## 2. Normative Language

The terms MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY express engineering requirements.

MUST-level requirements may be waived only through an explicit, reviewed decision that documents the rationale, scope, risk, and alternative verification strategy.

## 3. Non-Negotiable Rules

1. Correctness MUST be established before an optimization is accepted.
2. Every new operator MUST include at least one benchmark.
3. Performance improvements MUST be supported by comparable measurements.
4. Benchmarks MUST identify the workload and the environment sufficiently to interpret the results.
5. Performance tests MUST NOT silently replace correctness tests.
6. Optimizations MUST preserve the documented semantic contract.
7. No performance claim may be based solely on code inspection or intuition.
8. Optimization complexity MUST be justified by measured benefit or a clearly documented architectural requirement.
9. Production runtime code MUST remain safe Rust; unsafe is prohibited by the workspace’s engineering constraints.
10. Optimizations MUST NOT introduce forbidden crate dependencies or circular dependencies.
11. Shared model weights MUST NOT contain request-level mutable state.
12. Performance-sensitive changes MUST include regression considerations.
13. Measurement noise MUST NOT be presented as a confirmed improvement.
14. Failed, inconclusive, or unavailable measurements MUST be reported accurately.
15. End-to-end performance MUST be considered whenever a local optimization could affect overall execution.

---

## 4. Performance Engineering Principles

### 4.1 Measure before optimizing

Every nontrivial optimization SHOULD begin with evidence that the affected operation contributes meaningfully to the workload’s cost.

Acceptable evidence may include:

* A repeatable benchmark.
* A CPU profile.
* Allocation measurements.
* Hardware performance counters.
* An execution trace.
* A representative end-to-end workload.
* A controlled experiment isolating the suspected bottleneck.

A suspected bottleneck is a hypothesis, not a verified fact.

### 4.2 Optimize the actual objective

Different workloads have different performance objectives.

For example:

* CLI startup may prioritize time to first useful output.
* Interactive inference may prioritize time to first token and inter-token latency.
* Offline generation may prioritize total throughput.
* Training may prioritize step time and useful work completed per second.
* Checkpointing may prioritize completion time, storage overhead, and recovery guarantees.
* Tensor operations may prioritize latency, throughput, memory traffic, or allocation overhead.

The intended objective MUST be stated before optimizing.

### 4.3 Prefer the simplest effective implementation

Prefer a simple implementation that meets the required performance target over a more complex implementation with an unproven or negligible advantage.

Do not introduce custom thread pools, complex dispatch systems, caching layers, specialized memory management, or additional dependencies without a clear justification.

### 4.4 Preserve abstraction boundaries

Performance work MUST respect docs/crate-boundaries.md.

A low-level optimization MUST NOT introduce a dependency on a higher-level crate merely to access configuration, model state, request context, or application behavior.

Use existing interfaces where they are sufficient. Change a trait or public interface only when the change is justified by a real contract or performance requirement.

### 4.5 Consider total cost

A faster kernel may increase allocation, memory usage, initialization time, code size, or downstream processing cost.

Evaluate relevant trade-offs rather than treating a single timing number as the complete result.

---

## 5. Performance Objectives and Metrics

Every significant performance investigation SHOULD identify its primary metric and relevant secondary metrics.

| Objective | Primary metric | Useful secondary metrics |
| --- | --- | --- |
| Operation latency | Time per operation | Variability, allocations |
| Throughput | Operations per second | CPU utilization, memory bandwidth |
| Model inference | End-to-end latency or tokens per second | Time to first token, inter-token latency |
| Training | Time per step or useful samples per second | Memory use, synchronization overhead |
| Memory efficiency | Peak or steady-state memory | Allocation count, retained capacity |
| Checkpointing | Time to save or load | Bytes processed, peak memory |
| Startup | Time to readiness | Initialization work, allocations |
| Scalability | Throughput or latency as load increases | Contention, tail latency |
| Compilation | Build or compile time | Incremental rebuild cost |

Not every metric applies to every operation.

The benchmark specification SHOULD distinguish the primary acceptance metric from diagnostic metrics.

### 5.1 Latency and throughput are different

Latency measures how long an operation takes. Throughput measures how much work is completed per unit of time.

A change can improve throughput while worsening individual-request latency, especially when batching or concurrency changes.

The metric MUST match the actual objective.

### 5.2 Average latency is insufficient for variable workloads

For systems with meaningful latency variability, report relevant percentiles such as p50, p95, and p99 when the sample size and measurement methodology support them.

Percentiles MUST NOT be treated as reliable estimates when the sample count is inadequate or the workload is not representative.

### 5.3 Memory metrics

Memory measurements SHOULD distinguish, where practical:

* Peak resident memory.
* Steady-state resident memory.
* Live tensor or buffer storage.
* Temporary allocation.
* Allocation count.
* Retained capacity.
* Device memory, if an accelerator backend exists.

A reduction in allocation count does not automatically imply lower peak memory or better end-to-end performance.

---

## 6. Benchmark Taxonomy

The workspace SHOULD maintain benchmarks at multiple levels.

### 6.1 Microbenchmarks

Microbenchmarks isolate a small operation or overhead.

Examples:

* Tensor indexing.
* Shape validation.
* Elementwise operations.
* Reduction kernels.
* Matrix multiplication.
* Allocation and buffer reuse.
* Data conversion.
* Trait dispatch overhead.

Microbenchmarks are useful for identifying local changes, but they do not establish end-to-end improvements.

### 6.2 Operator benchmarks

Operator benchmarks exercise a complete public or internal operator contract using representative shapes, dtypes, and layouts.

Every new operator MUST have at least one benchmark.

The benchmark SHOULD represent a useful workload rather than an artificially favorable single input.

### 6.3 Pipeline benchmarks

Pipeline benchmarks measure interactions among multiple operations.

Examples:

* Tokenization and encoding.
* Batch preparation.
* Forward and backward passes.
* Optimizer updates.
* Checkpoint serialization.
* Model loading and initialization.

These benchmarks help identify costs that individual operator benchmarks cannot reveal.

### 6.4 End-to-end benchmarks

End-to-end benchmarks measure a complete user-relevant workflow.

Examples:

* Loading a model and running inference.
* Generating a fixed number of tokens.
* Executing a training step.
* Saving and restoring a checkpoint.

End-to-end measurements SHOULD be included when the change could materially affect overall system performance.

### 6.5 Stress and scaling benchmarks

Stress benchmarks examine behavior under increasing workload, concurrency, or resource pressure.

They SHOULD be separated from routine microbenchmarks and used where the system’s operational requirements justify them.

A benchmark MUST NOT be labeled a stress test merely because it runs for a long time.

---

## 7. Benchmark Reproducibility

A benchmark result is meaningful only when the workload and measurement conditions are sufficiently documented.

### 7.1 Required benchmark metadata

Each maintained benchmark SHOULD record or make recoverable:

* Benchmark name and operation.
* Input dimensions and dtype.
* Tensor layout and strides, where relevant.
* Batch size and sequence length, where relevant.
* Relevant model or dataset fixture.
* Rust toolchain version.
* Compiler version and optimization configuration.
* Build profile.
* Enabled Cargo features.
* Target platform and architecture.
* CPU model and relevant hardware configuration.
* Operating system and kernel information where relevant.
* Benchmark framework and version.
* Warm-up policy.
* Measurement duration or iteration policy.
* Sample count.
* Reported statistic.
* Relevant workload-specific settings.

The exact metadata depends on the benchmark. Irrelevant fields need not be duplicated for every result if the environment is recorded centrally.

### 7.2 Control the environment

Where practical:

* Use optimized builds.
* Avoid unrelated CPU-intensive background work.
* Avoid simultaneous benchmarks competing for the same resources.
* Keep power and thermal conditions reasonably stable.
* Keep compiler flags and features consistent between comparisons.
* Avoid changing workload data between baseline and candidate measurements.
* Keep thread counts and affinity policies consistent when they affect the result.

Perfect environmental isolation is not always possible. Any material limitation MUST be recorded.

### 7.3 Warm-up

Warm-up SHOULD be used when first-run effects are not part of the intended metric.

Potential first-run effects include initialization, memory allocation, cache population, lazy setup, and runtime dispatch.

If startup or cold-cache performance is the actual objective, measure it explicitly rather than warming it away.

### 7.4 Repeated measurements

Use repeated measurements appropriate to the duration and variability of the workload.

Very short operations may require batching or repeated iterations to reduce timer overhead.

Long operations may require fewer samples, but their variability and environmental sensitivity still need consideration.

Do not choose the sample count after seeing which count makes the result appear favorable.

### 7.5 Record failed attempts

If a benchmark fails, times out, encounters resource exhaustion, or produces invalid output, preserve that result as a failed or incomplete run.

Do not silently discard inconvenient samples without a documented measurement rule.

---

## 8. Baselines and Before-and-After Comparisons

A performance improvement requires a meaningful comparison.

### 8.1 Baseline selection

A baseline SHOULD represent the previous correct implementation under the same or sufficiently comparable conditions.

Before accepting a comparison, verify that:

* The workload is equivalent.
* Correctness requirements are unchanged.
* Compiler and build settings are comparable.
* Hardware and environmental differences are understood.
* Relevant warm-up and measurement policies match.
* No unrelated code change dominates the result.

If these conditions cannot be established, the comparison MUST be qualified as limited or inconclusive.

### 8.2 Record the baseline before implementation

For planned performance work, record the baseline before making the optimization whenever practical.

This reduces the risk of choosing a favorable comparison after implementation.

If no valid baseline exists, establish one before claiming improvement.

### 8.3 Calculate relative change correctly

For a metric where lower is better, such as latency:

$$
\text{Reduction (\%)} =
\frac{T_{\mathrm{baseline}}-T_{\mathrm{candidate}}}
{T_{\mathrm{baseline}}}\times 100
$$

For a metric where higher is better, such as throughput:

$$
\text{Increase (\%)} =
\frac{P_{\mathrm{candidate}}-P_{\mathrm{baseline}}}
{P_{\mathrm{baseline}}}\times 100
$$

These calculations describe the observed relative change. They do not establish statistical significance, practical importance, or causal attribution by themselves.

### 8.4 Use consistent units

Always state units.

Examples:

* Microseconds per operation.
* Milliseconds per request.
* Tokens per second.
* Samples per second.
* MiB of peak memory.
* Bytes allocated per operation.

Avoid comparing measurements with different units or different definitions of the measured work.

---

## 9. Statistical Interpretation

Performance results vary. Interpretation MUST account for measurement uncertainty.

### 9.1 Report variability

Depending on the benchmark, useful summaries include:

* Median.
* Mean.
* Standard deviation.
* Interquartile range.
* Relevant percentiles.
* Confidence intervals.

No single statistic is universally appropriate.

For noisy or skewed latency distributions, the median and percentiles may be more informative than the mean. For throughput measurements with stable conditions, other summaries may be appropriate.

### 9.2 Distinguish observed differences from reliable improvements

A candidate that is slightly faster in one run is not necessarily faster in general.

Where variability is material, repeat measurements and use an appropriate comparison method.

The conclusion SHOULD distinguish:

* A clear and practically meaningful improvement.
* A clear regression.
* A difference too small to matter operationally.
* A result whose uncertainty prevents a conclusion.

### 9.3 Do not overstate precision

If a measurement supports only an approximate result, report an approximate result.

Do not claim a precise percentage improvement from unstable measurements or excessively rounded data.

### 9.4 Define practical significance

An optimization SHOULD have a stated reason to matter.

A statistically detectable difference may still be operationally irrelevant. Conversely, a small local improvement may matter when the operation dominates a frequently executed path.

The acceptance decision SHOULD consider both the evidence and the actual workload.

---

## 10. Optimization Workflow

Performance changes MUST follow a disciplined process.

Step 1: Define the objective

State the workload, primary metric, expected benefit, and relevant correctness constraints.

Avoid objectives such as “make the code faster” without a measurable target or an identified bottleneck.

Step 2: Establish the baseline

Run the existing correct implementation under a documented workload.

Record the relevant measurements and environment.

Step 3: Identify the bottleneck

Use measurements, profiling, or controlled experiments to identify the dominant cost.

Possible bottlenecks include:

* Excessive algorithmic work.
* Unnecessary allocations.
* Repeated data conversion.
* Poor memory locality.
* Redundant tensor copies.
* Synchronization overhead.
* Dispatch overhead.
* Inefficient batching.
* Unnecessary intermediate tensors.
* Excessive initialization.

Do not assume the bottleneck based only on source-code appearance.

Step 4: Form a testable hypothesis

Describe why the proposed change should improve the selected metric.

For example:

Reusing a correctly scoped temporary buffer is expected to reduce allocation overhead in repeated executions of this operator without changing its numerical result or public contract.

The hypothesis SHOULD identify the expected mechanism, affected workload, and possible trade-offs.

Step 5: Make the smallest effective change

Prefer a focused implementation change that tests the hypothesis.

Do not combine unrelated refactoring, API redesign, and performance tuning in one unreviewable patch.

Step 6: Validate correctness

Run the relevant unit, boundary, numerical, and integration tests before interpreting performance results.

A candidate that fails correctness requirements MUST NOT be accepted as an optimization.

Step 7: Measure the candidate

Run the same benchmark using comparable settings.

Collect enough evidence to understand the observed difference and its variability.

Step 8: Evaluate trade-offs

Review changes in:

* Latency.
* Throughput.
* Memory usage.
* Allocation behavior.
* Initialization cost.
* Code complexity.
* Build time and binary size, where relevant.
* Cross-crate coupling.
* Error handling.
* Numerical stability.

Step 9: Decide

Keep the change when the evidence supports the objective and the trade-offs are acceptable.

Revert or redesign it when the improvement is absent, unreliable, too small to justify the complexity, or accompanied by unacceptable regressions.

Step 10: Report the result

Record the benchmark method, results, limitations, and final conclusion.

Do not substitute a successful test run for performance evidence.

---

## 11. Tensor and CPU Kernel Optimization

Tensor and CPU operations often dominate computation, but local speed alone is insufficient.

### 11.1 Algorithmic complexity

Prefer an algorithm with an appropriate asymptotic cost before pursuing small constant-factor optimizations.

If the workload is dominated by an inefficient algorithm, instruction-level tuning may not address the principal bottleneck.

### 11.2 Memory layout

Tensor performance depends on shape, strides, contiguity, and access patterns.

Optimization work SHOULD consider:

* Sequential versus irregular access.
* Cache locality.
* Reuse of loaded values.
* Temporary buffers.
* Copying and materialization.
* Stride-aware iteration.
* Layout conversion overhead.

Do not assume all tensors are contiguous unless the API explicitly guarantees contiguity.

### 11.3 Allocation and reuse

Repeated allocation can be expensive in hot paths.

Buffer reuse MAY be introduced when it preserves ownership, concurrency safety, and lifetime guarantees.

Buffer reuse MUST NOT create hidden shared mutable state in model weights or allow request-level data to leak between executions.

### 11.4 Vectorization

Compiler vectorization and architecture-specific optimization MAY be considered after profiling identifies a relevant bottleneck.

Do not assume that manually restructuring code for vectorization is faster than a simpler implementation. Verify the generated behavior or benchmark the result.

### 11.5 Parallelism

Parallel execution MAY improve throughput for sufficiently large workloads.

It can also introduce:

* Thread startup and scheduling overhead.
* Synchronization cost.
* Contention.
* Oversubscription.
* Unstable latency.
* Nondeterministic floating-point accumulation.
* Increased memory usage.

Parallelism SHOULD be introduced only when the workload and measurement evidence justify it.

### 11.6 No unnecessary complexity

Do not add specialized execution paths for every shape, dtype, or input size without evidence that dispatch and maintenance costs are justified.

A small number of well-motivated paths is preferable to an unmanageable collection of speculative special cases.

---

## 12. Model Inference Performance

Inference performance MUST be evaluated at the level appropriate to the intended use case.

### 12.1 Distinguish execution phases

Where relevant, measure separately:

* Model loading.
* Initialization.
* Input preparation.
* Tokenization.
* Prefill.
* Autoregressive generation.
* Output decoding.
* End-to-end request latency.

These phases have different bottlenecks. Improving one phase does not necessarily improve the total request time.

### 12.2 Generation metrics

For autoregressive generation, distinguish:

* Time to first token.
* Inter-token latency.
* Total generation time.
* Generated tokens per second.
* End-to-end request latency.

The benchmark MUST document whether prompt processing is included in the throughput calculation.

Tokens per second without a defined token-counting and timing convention are not a reproducible metric.

### 12.3 Fix the workload

For comparisons, control or record:

* Model artifact and configuration.
* Prompt length.
* Requested output length.
* Batch size.
* Sampling or decoding configuration.
* Tokenizer.
* Warm-up.
* Concurrency.
* Thread settings.
* Relevant hardware and build configuration.

Where generation is stochastic, use a deterministic workload when possible, or report the stochastic settings and limitations.

### 12.4 Avoid shared mutable request state

Inference optimization MUST NOT place request-specific mutable state inside shared model weights.

Caches, temporary tensors, generation state, and cancellation information MUST have explicit ownership and concurrency semantics.

Any cache optimization SHOULD define capacity, lifetime, eviction behavior, and isolation between requests.

### 12.5 Compare equivalent outputs

When an optimization changes execution order or numerical behavior, verify the required output contract before comparing speed.

For deterministic operations, output equality requirements MUST be documented.

For numerically approximate operations, use justified numerical tolerances and application-level acceptance criteria.

---

## 13. Training Performance

Training performance is influenced by both compute and state-management overhead.

Relevant costs may include:

* Data loading.
* Batch preparation.
* Forward computation.
* Backward computation.
* Gradient accumulation.
* Optimizer updates.
* Synchronization.
* Checkpointing.

### 13.1 Measure the intended unit of work

A training benchmark SHOULD state whether it measures:

* One forward pass.
* One backward pass.
* One optimizer update.
* A complete training step.
* A complete epoch.
* An end-to-end training workflow.

Do not compare a complete training step with a measurement that excludes part of the work.

### 13.2 Fix the workload

Record the model, batch size, sequence length, dataset fixture, optimizer configuration, and relevant numerical settings.

If a change alters the amount of work performed, raw elapsed time alone may be misleading.

### 13.3 Preserve training semantics

A faster training path MUST NOT silently skip required gradient computation, optimizer updates, validation, or state persistence.

Performance changes MUST preserve the documented training semantics, including relevant state-update and error behavior.

### 13.4 Checkpoint overhead

Measure checkpointing separately when it materially affects training throughput.

If asynchronous checkpointing is introduced, evaluate both apparent step-time improvement and the actual resource and completion costs.

---

## 14. Memory and Allocation Efficiency

Memory optimization SHOULD be based on a clear understanding of the resource being reduced.

### 14.1 Distinguish allocation from usage

Reducing allocation count is not equivalent to reducing peak memory.

Reusing a large buffer may reduce allocation overhead while increasing retained memory.

Measure the relevant quantity rather than assuming one metric represents all memory costs.

### 14.2 Temporary storage

Temporary buffers SHOULD have clear ownership and bounded lifetimes.

Avoid retaining large intermediates beyond their useful lifetime.

Where buffer reuse is introduced, test that previous request or operation contents cannot affect subsequent results.

### 14.3 Avoid premature pooling

Object or buffer pools introduce complexity, capacity decisions, and possible contention.

Introduce pooling only when measurements show that allocation or initialization cost is material and a simpler approach is insufficient.

### 14.4 Resource limits

Memory optimization MUST NOT bypass documented size limits or convert allocation failures into fabricated success.

Resource exhaustion MUST follow docs/error-handling.md.

---

## 15. Caching and Memoization

Caching MAY be useful when an expensive operation is repeated with the same effective inputs.

A cache MUST define:

* Key semantics.
* Validity conditions.
* Lifetime.
* Capacity or eviction policy.
* Memory cost.
* Concurrency behavior.
* Failure behavior.
* Invalidation requirements.

### 15.1 Correctness first

A cache MUST NOT return a result for inputs that are not equivalent under the operation’s actual semantics.

If a result depends on configuration, dtype, layout, model version, or other context, the cache key or invalidation policy MUST account for that dependency.

### 15.2 Isolation

Caching MUST NOT allow request-specific mutable state to leak between users or operations.

### 15.3 Evidence

Cache effectiveness SHOULD be measured using realistic hit and miss patterns.

A cache that improves a synthetic repeated-input benchmark may not help the real workload.

Memory overhead, miss cost, and invalidation complexity MUST be considered.

---

## 16. Benchmark Design and Implementation

Benchmarks SHOULD be kept separate from ordinary correctness tests.

### 16.1 Timed region

The timed region MUST include exactly the work described by the benchmark.

Exclude fixture construction, random data generation, model loading, or other setup when those costs are not part of the target metric.

Include them when they are intentionally part of the workload.

### 16.2 Input preparation

Prepare reusable input data outside the timed region when measuring repeated operation latency and input preparation is not part of the target.

If allocation is the subject of the benchmark, allocation MUST remain inside the timed region.

### 16.3 Result consumption

The benchmark MUST ensure the intended computation is actually executed and that the compiler cannot trivially eliminate the measured work.

Use the benchmark framework’s appropriate result-consumption mechanisms.

### 16.4 Avoid benchmark-only semantics

The benchmark MUST call the actual implementation being evaluated.

Do not create a separate simplified implementation for the benchmark and then present its result as the performance of the production path.

### 16.5 Benchmark framework

The workspace MAY use an established Rust benchmarking framework or a repository-specific harness.

The framework MUST be selected deliberately and used consistently enough to support comparable results.

The framework itself does not guarantee valid measurements. Benchmark design and environment control remain necessary.

---

## 17. Regression Detection

Performance regression checks SHOULD be introduced for stable, important workloads where the environment supports meaningful comparisons.

### 17.1 Regression thresholds

A threshold SHOULD account for:

* Historical variability.
* Measurement overhead.
* Workload importance.
* Cost of a false alarm.
* Cost of an undetected regression.

Do not choose a threshold merely because it is easy to configure.

### 17.2 Environment differences

Results from different hardware or compiler configurations MUST NOT be compared as though they were directly equivalent without an appropriate normalization or qualification.

### 17.3 Noisy CI

Performance CI may be noisy because of shared machines, thermal behavior, virtualized environments, and competing workloads.

A suspected regression SHOULD trigger confirmation under controlled conditions rather than an automatic claim that the implementation is slower.

### 17.4 Preserve historical evidence

For critical benchmarks, retain enough historical results to identify trends and distinguish a sudden regression from normal variability.

The retention mechanism MAY be a CI artifact, benchmark database, or another repository-approved system.

Do not introduce a complex performance data platform unless the project’s needs justify it.

---

## 18. Trade-offs and Acceptance Decisions

An optimization MUST be evaluated against its complete cost.

Reviewers SHOULD ask:

* Does the change solve a demonstrated bottleneck?
* Is the measured improvement reproducible?
* Does it improve the intended workload?
* Does it increase memory consumption?
* Does it affect other important workloads?
* Does it add a new dependency?
* Does it complicate error handling?
* Does it make numerical behavior less stable?
* Does it require a more complex API?
* Does it violate the intended crate boundaries?
* Can the improvement be maintained and reverified?

A small, stable improvement may be worthwhile in a critical hot path. A similarly small improvement in an insignificant path may not justify a substantial increase in complexity.

The decision MUST consider both measured value and maintenance cost.

---

## 19. Performance Reporting

Every significant performance change SHOULD include a concise report.

The report MUST distinguish implementation status, correctness validation, benchmark execution, and the conclusion supported by the measurements.

Recommended template:

Objective

Describe the workload and intended improvement.

Baseline

Describe the previous implementation, benchmark configuration, and measured result.

Change

Describe the actual implementation change and the hypothesis it tests.

Correctness

List the relevant tests executed and their outcomes.

Measurement

Include the candidate result, units, relevant variability, environment, and comparison method.

Trade-offs

Describe relevant changes in memory, initialization, complexity, numerical behavior, and other workloads.

Conclusion

Choose the appropriate evidence-based status:

* Implemented — the change exists.
* Correctness-validated — the required correctness checks passed.
* Benchmarked — the defined benchmark was executed.
* Improved — comparable measurements support an improvement.
* Regressed — comparable measurements support a slowdown.
* Inconclusive — the evidence is insufficient to determine the outcome.

These statuses are not interchangeable. A change can be implemented, correctness-validated, and benchmarked while its performance result remains inconclusive.

---

## 20. AI-Assisted Performance Engineering

AI coding agents MUST follow the same evidence requirements as human engineers.

An AI agent MUST NOT:

* Invent benchmark measurements.
* Claim speedups based on source-code appearance.
* Treat reduced line count as improved performance.
* Claim vectorization without evidence.
* Assume allocation reduction implies lower latency.
* Remove correctness checks solely to improve benchmark results.
* Change workloads between baseline and candidate without disclosing the difference.
* Introduce speculative complexity without a testable hypothesis.
* Report a benchmark as executed when it was not run.
* Claim an end-to-end improvement based only on a microbenchmark.

If repository access, benchmark execution, or suitable hardware is unavailable, the agent MUST state the limitation.

An optimization proposal without measurements MAY be submitted as a hypothesis. It MUST NOT be presented as a verified performance improvement.

---

## 21. Performance Review Checklist

Before accepting a performance-sensitive change, reviewers SHOULD verify the following.

Objective and evidence

* [ ] The target workload is explicit.
* [ ] The primary metric is appropriate.
* [ ] A baseline exists or the absence of one is documented.
* [ ] The suspected bottleneck is supported by evidence.
* [ ] The optimization hypothesis is testable.

Correctness

* [ ] Required correctness tests pass.
* [ ] Boundary behavior is preserved.
* [ ] Numerical behavior is verified where relevant.
* [ ] Error handling remains correct.
* [ ] Shared model weights contain no request-level mutable state.

Measurement

* [ ] Benchmark inputs and configuration are recorded.
* [ ] Baseline and candidate workloads are comparable.
* [ ] Relevant environment differences are documented.
* [ ] Measurement variability is considered.
* [ ] Units and calculation methods are clear.

Architecture and maintenance

* [ ] No forbidden dependency direction is introduced.
* [ ] No unsafe code is introduced.
* [ ] Complexity is justified by evidence.
* [ ] Relevant memory and initialization trade-offs are understood.
* [ ] Regression coverage is appropriate.
* [ ] The final performance claim matches the evidence.

---

## 22. Conformance and Adoption

This document defines the target performance engineering standard. It does not assert that existing implementations or benchmarks already conform.

Adoption SHOULD proceed incrementally.

For each performance-critical crate or workflow, record:

* Primary workload.
* Primary metric.
* Existing benchmark coverage.
* Baseline availability.
* Known bottlenecks.
* Correctness requirements.
* Memory and concurrency constraints.
* Regression detection strategy.
* Verification status.

Use the following conformance states:

* Proposed — the performance requirement or benchmark is specified but not verified.
* Verified — the relevant benchmark and validation evidence have been collected.
* Non-conforming — a known requirement or measurement gap exists.
* Blocked — validation requires an unresolved design decision, dependency, or suitable environment.

A crate MUST NOT be marked performance-verified merely because it compiles, passes unit tests, or contains benchmark source files.

A benchmark that has never been executed is not measured evidence.

---

## 23. Final Principle

Performance engineering is a process of hypothesis, measurement, verification, and disciplined trade-offs.

Measure the bottleneck. Preserve correctness. Change the minimum necessary. Compare equivalent workloads. Report only what the evidence supports.

The goal is not to produce the most complicated implementation or the most impressive benchmark number.

The goal is to build a system whose performance characteristics are understood, reproducible, maintainable, and appropriate for its intended workloads.