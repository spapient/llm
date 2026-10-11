//! Criterion microbenchmarks for the native Metal backend.
//!
//! macOS only: on other targets the `metal` crate is an empty library, so the
//! target compiles to a no-op `main`. The CPU executor measures every workload
//! in the same run, so each Metal number is reported against a live baseline
//! rather than a stale constant. Both paths are driven through the same
//! `api::compute` contract with identical `bench` operands, so the comparison is
//! reproducible.
//!
//! The workload set and the group names mirror `benches/cuda.rs`: the two
//! backends share semantics, not implementation, so their measurements are
//! comparable one workload at a time while each keeps its own kernel and
//! dispatch geometry.
//!
//! No improvement is claimed: every call's cost still includes the per-call
//! device context creation and run-time MSL compilation this stage performs
//! (docs/design.md §21.10 records that as the reason Metal is slower than CPU at
//! the one size measured so far). Establishing the baseline is required by
//! docs/engineering.md; replacing the per-call compilation is a listed review
//! condition (docs/design.md §21.11).
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(missing_docs)]

#[cfg(target_os = "macos")]
mod macos {
    use std::hint::black_box;

    use ::tensor::tensor;
    use api::compute;
    use bench::{
        ELEMENTWISE, EMBEDDIM, GATHER, HEADDIM, HEADS, MATMULDIM, NORMCOLS, NORMROWS, SEQLEN,
        SOFTMAXCOLS, SOFTMAXROWS, attentioninput, elementwiseinput, embeddinginput, matmulinput,
        norminput, ropeinput, softmaxinput,
    };
    use cpu::cpuexecutor;
    use criterion::{
        BenchmarkGroup, Criterion, Throughput, criterion_group, measurement::WallTime,
    };
    use lib::limits;
    use metal::metalexecutor;

    /// Resource budget that admits every workload without rejecting it.
    const LIMIT: limits = limits::new(1 << 28, 1 << 31, 1 << 20, 1 << 18);

    /// Epsilon used by both sides of the RMS-normalization comparison.
    const EPS: f32 = 1.0e-5;

    /// Rotary base used by both sides of the RoPE comparison.
    const BASE: f32 = 10_000.0;

    /// Measures the `cpu` and `metal` closures of one workload back to back.
    ///
    /// Both are timed in the same run so that a change in the environment moves
    /// both numbers together and the ratio stays meaningful (VF4). Each timed
    /// region returns the operator's output tensor, which `black_box` keeps
    /// observable so the work cannot be optimized away.
    fn pair<F, G>(
        group: &mut BenchmarkGroup<'_, WallTime>,
        name: &str,
        elements: usize,
        cpu: F,
        metal: G,
    ) where
        F: Fn() -> tensor<f32>,
        G: Fn() -> tensor<f32>,
    {
        group.throughput(Throughput::Elements(elements as u64));
        group.bench_function(format!("cpu{name}"), |b| b.iter(|| black_box(cpu())));
        group.bench_function(format!("metal{name}"), |b| b.iter(|| black_box(metal())));
    }

    fn matmul(c: &mut Criterion) {
        let (lhs, rhs) = matmulinput().unwrap();
        let (lv, rv) = (lhs.getview(), rhs.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        let mut group = c.benchmark_group("gpu/matmul");
        pair(
            &mut group,
            "256",
            MATMULDIM * MATMULDIM,
            || cpuexec.matmul(&lv, &rv).unwrap(),
            || metalexec.matmul(&lv, &rv).unwrap(),
        );
        group.finish();
    }

    fn elementwise(c: &mut Criterion) {
        let x = elementwiseinput().unwrap();
        let y = elementwiseinput().unwrap();
        let v = x.getview();
        let w = y.getview();
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        let mut group = c.benchmark_group("gpu/elementwise");
        pair(
            &mut group,
            "silu",
            ELEMENTWISE,
            || cpuexec.silu(&v).unwrap(),
            || metalexec.silu(&v).unwrap(),
        );
        pair(
            &mut group,
            "add",
            ELEMENTWISE,
            || cpuexec.add(&v, &w).unwrap(),
            || metalexec.add(&v, &w).unwrap(),
        );
        pair(
            &mut group,
            "mul",
            ELEMENTWISE,
            || cpuexec.mul(&v, &w).unwrap(),
            || metalexec.mul(&v, &w).unwrap(),
        );
        group.finish();
    }

    fn softmax(c: &mut Criterion) {
        let x = softmaxinput().unwrap();
        let v = x.getview();
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        let mut group = c.benchmark_group("gpu/softmax");
        pair(
            &mut group,
            "rows512cols4096",
            SOFTMAXROWS * SOFTMAXCOLS,
            || cpuexec.softmax(&v).unwrap(),
            || metalexec.softmax(&v).unwrap(),
        );
        group.finish();
    }

    fn rmsnorm(c: &mut Criterion) {
        let (x, w) = norminput().unwrap();
        let (xv, wv) = (x.getview(), w.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        let mut group = c.benchmark_group("gpu/rmsnorm");
        pair(
            &mut group,
            "rows512cols4096",
            NORMROWS * NORMCOLS,
            || cpuexec.rmsnorm(&xv, &wv, EPS).unwrap(),
            || metalexec.rmsnorm(&xv, &wv, EPS).unwrap(),
        );
        group.finish();
    }

    fn rope(c: &mut Criterion) {
        let x = ropeinput().unwrap();
        let v = x.getview();
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        let mut group = c.benchmark_group("gpu/rope");
        pair(
            &mut group,
            "seq128heads8dim64",
            SEQLEN * HEADS * HEADDIM,
            || cpuexec.rope(&v, HEADS, BASE).unwrap(),
            || metalexec.rope(&v, HEADS, BASE).unwrap(),
        );
        group.finish();
    }

    fn embedding(c: &mut Criterion) {
        let (table, indices) = embeddinginput().unwrap();
        let v = table.getview();
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        let mut group = c.benchmark_group("gpu/embedding");
        pair(
            &mut group,
            "gather128",
            GATHER * EMBEDDIM,
            || cpuexec.embedding(&v, &indices).unwrap(),
            || metalexec.embedding(&v, &indices).unwrap(),
        );
        group.finish();
    }

    fn attention(c: &mut Criterion) {
        let (q, k, v) = attentioninput().unwrap();
        let (qv, kv, vv) = (q.getview(), k.getview(), v.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let metalexec = metalexecutor::new(LIMIT);
        let mut group = c.benchmark_group("gpu/attention");
        pair(
            &mut group,
            "seq128heads8dim64",
            SEQLEN * HEADS * HEADDIM,
            || {
                cpuexec
                    .attention(&qv, &kv, &vv, HEADS, HEADS, true)
                    .unwrap()
            },
            || {
                metalexec
                    .attention(&qv, &kv, &vv, HEADS, HEADS, true)
                    .unwrap()
            },
        );
        group.finish();
    }

    criterion_group!(
        benches,
        matmul,
        elementwise,
        softmax,
        rmsnorm,
        rope,
        embedding,
        attention
    );
}

#[cfg(target_os = "macos")]
criterion::criterion_main!(macos::benches);

#[cfg(not(target_os = "macos"))]
fn main() {}
