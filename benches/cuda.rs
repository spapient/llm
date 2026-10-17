//! Criterion microbenchmarks for the native CUDA backend.
//!
//! Linux only: on other targets the `cuda` crate is an empty library, so the
//! target compiles to a no-op `main`. The CPU executor measures every workload
//! in the same run, so each CUDA number is reported against a live baseline
//! rather than a stale constant. Both paths are driven through the same
//! `api::compute` contract with identical `bench` operands, so the comparison
//! is reproducible.
//!
//! No improvement is claimed: these numbers are a baseline (E1, RTX 2080 Ti /
//! CUDA 13.0 / `rustc 1.99.0`, 2026-10-12; docs/design.md §21.10.3), not a
//! comparison against an earlier revision. The per-call NVRTC compilation this
//! baseline measured no longer exists: kernels are compiled once per
//! `(source, entry)` and cached (docs/design.md §21.10.3 item 7, 2026-10-12),
//! and the cuBLAS/cuDNN handles are cached as well (2026-10-17).
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(missing_docs)]

#[cfg(target_os = "linux")]
mod linux {
    use std::hint::black_box;

    use ::tensor::tensor;
    use api::compute;
    use bench::{
        ELEMENTWISE, EMBEDDIM, GATHER, HEADDIM, HEADS, MATMULDIM, NORMCOLS, NORMROWS, SEQLEN,
        SOFTMAXCOLS, SOFTMAXROWS, attentioninput, elementwiseinput, embeddinginput,
        matmulbatchinput, matmulinput, norminput, ropeinput, softmaxinput,
    };
    use cpu::cpuexecutor;
    use criterion::{
        BenchmarkGroup, Criterion, Throughput, criterion_group, measurement::WallTime,
    };
    use cuda::cudaexecutor;
    use lib::limits;

    /// Resource budget that admits every workload without rejecting it.
    const LIMIT: limits = limits::new(1 << 28, 1 << 31, 1 << 20, 1 << 18);

    /// Epsilon used by both sides of the RMS-normalization comparison.
    const EPS: f32 = 1.0e-5;

    /// Rotary base used by both sides of the RoPE comparison.
    const BASE: f32 = 10_000.0;

    /// Measures the `cpu` and `cuda` closures of one workload back to back.
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
        cuda: G,
    ) where
        F: Fn() -> tensor<f32>,
        G: Fn() -> tensor<f32>,
    {
        group.throughput(Throughput::Elements(elements as u64));
        group.bench_function(format!("cpu{name}"), |b| b.iter(|| black_box(cpu())));
        group.bench_function(format!("cuda{name}"), |b| b.iter(|| black_box(cuda())));
    }

    fn matmul(c: &mut Criterion) {
        let (lhs, rhs) = matmulinput().unwrap();
        let (lv, rv) = (lhs.getview(), rhs.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let cudaexec = cudaexecutor::new(LIMIT);
        let mut group = c.benchmark_group("gpu/matmul");
        pair(
            &mut group,
            "256",
            MATMULDIM * MATMULDIM,
            || cpuexec.matmul(&lv, &rv).unwrap(),
            || cudaexec.matmul(&lv, &rv).unwrap(),
        );
        group.finish();
    }

    fn matmul_batch(c: &mut Criterion) {
        let (lhs, rhs) = matmulbatchinput().unwrap();
        let (lv, rv) = (lhs.getview(), rhs.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let cudaexec = cudaexecutor::new(LIMIT);
        let mut group = c.benchmark_group("gpu/matmulbatch");
        pair(
            &mut group,
            "8x256",
            lhs.getlen(),
            || cpuexec.matmul(&lv, &rv).unwrap(),
            || cudaexec.matmul(&lv, &rv).unwrap(),
        );
        group.finish();
    }

    fn elementwise(c: &mut Criterion) {
        let x = elementwiseinput().unwrap();
        let y = elementwiseinput().unwrap();
        let v = x.getview();
        let w = y.getview();
        let cpuexec = cpuexecutor::new(LIMIT);
        let cudaexec = cudaexecutor::new(LIMIT);
        let mut group = c.benchmark_group("gpu/elementwise");
        pair(
            &mut group,
            "silu",
            ELEMENTWISE,
            || cpuexec.silu(&v).unwrap(),
            || cudaexec.silu(&v).unwrap(),
        );
        pair(
            &mut group,
            "add",
            ELEMENTWISE,
            || cpuexec.add(&v, &w).unwrap(),
            || cudaexec.add(&v, &w).unwrap(),
        );
        pair(
            &mut group,
            "mul",
            ELEMENTWISE,
            || cpuexec.mul(&v, &w).unwrap(),
            || cudaexec.mul(&v, &w).unwrap(),
        );
        group.finish();
    }

    fn softmax(c: &mut Criterion) {
        let x = softmaxinput().unwrap();
        let v = x.getview();
        let cpuexec = cpuexecutor::new(LIMIT);
        let cudaexec = cudaexecutor::new(LIMIT);
        let mut group = c.benchmark_group("gpu/softmax");
        pair(
            &mut group,
            "rows512cols4096",
            SOFTMAXROWS * SOFTMAXCOLS,
            || cpuexec.softmax(&v).unwrap(),
            || cudaexec.softmax(&v).unwrap(),
        );
        group.finish();
    }

    /// The two softmax implementations side by side: the cuDNN default and
    /// the in-repo NVRTC kernel, so the per-call handle-creation cost the
    /// NVRTC path avoids is visible on the same device in the same run.
    fn softmaxnvrtc(c: &mut Criterion) {
        let x = softmaxinput().unwrap();
        let v = x.getview();
        let cudaexec = cudaexecutor::new(LIMIT);
        let nvrtcexec = cudaexecutor::newsoftmaxnvrtc(LIMIT);
        let mut group = c.benchmark_group("gpu/softmaxpath");
        group.throughput(Throughput::Elements((SOFTMAXROWS * SOFTMAXCOLS) as u64));
        group.bench_function("cudnn", |b| {
            b.iter(|| black_box(cudaexec.softmax(&v).unwrap()));
        });
        group.bench_function("nvrtc", |b| {
            b.iter(|| black_box(nvrtcexec.softmax(&v).unwrap()));
        });
        group.finish();
    }

    fn rmsnorm(c: &mut Criterion) {
        let (x, w) = norminput().unwrap();
        let (xv, wv) = (x.getview(), w.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let cudaexec = cudaexecutor::new(LIMIT);
        let mut group = c.benchmark_group("gpu/rmsnorm");
        pair(
            &mut group,
            "rows512cols4096",
            NORMROWS * NORMCOLS,
            || cpuexec.rmsnorm(&xv, &wv, EPS).unwrap(),
            || cudaexec.rmsnorm(&xv, &wv, EPS).unwrap(),
        );
        group.finish();
    }

    fn rope(c: &mut Criterion) {
        let x = ropeinput().unwrap();
        let v = x.getview();
        let cpuexec = cpuexecutor::new(LIMIT);
        let cudaexec = cudaexecutor::new(LIMIT);
        let mut group = c.benchmark_group("gpu/rope");
        pair(
            &mut group,
            "seq128heads8dim64",
            SEQLEN * HEADS * HEADDIM,
            || cpuexec.rope(&v, HEADS, BASE).unwrap(),
            || cudaexec.rope(&v, HEADS, BASE).unwrap(),
        );
        group.finish();
    }

    fn embedding(c: &mut Criterion) {
        let (table, indices) = embeddinginput().unwrap();
        let v = table.getview();
        let cpuexec = cpuexecutor::new(LIMIT);
        let cudaexec = cudaexecutor::new(LIMIT);
        let mut group = c.benchmark_group("gpu/embedding");
        pair(
            &mut group,
            "gather128",
            GATHER * EMBEDDIM,
            || cpuexec.embedding(&v, &indices).unwrap(),
            || cudaexec.embedding(&v, &indices).unwrap(),
        );
        group.finish();
    }

    fn attention(c: &mut Criterion) {
        let (q, k, v) = attentioninput().unwrap();
        let (qv, kv, vv) = (q.getview(), k.getview(), v.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let cudaexec = cudaexecutor::new(LIMIT);
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
                cudaexec
                    .attention(&qv, &kv, &vv, HEADS, HEADS, true)
                    .unwrap()
            },
        );
        group.finish();
    }

    criterion_group!(
        benches,
        matmul,
        matmul_batch,
        elementwise,
        softmax,
        softmaxnvrtc,
        rmsnorm,
        rope,
        embedding,
        attention
    );
}

#[cfg(target_os = "linux")]
criterion::criterion_main!(linux::benches);

#[cfg(not(target_os = "linux"))]
fn main() {}
