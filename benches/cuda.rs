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
//! and the cuBLAS handle is cached as well (2026-10-17).
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(missing_docs)]

#[cfg(target_os = "linux")]
mod linux {
    use std::hint::black_box;

    use ::bench::harness::{report, run};
    use ::tensor::tensor;
    use api::compute;
    use bench::{
        ELEMENTWISE, EMBEDDIM, GATHER, HEADDIM, HEADS, MATMULDIM, NORMCOLS, NORMROWS, SEQLEN,
        SOFTMAXCOLS, SOFTMAXROWS, attentioninput, elementwiseinput, embeddinginput,
        matmulbatchinput, matmulinput, norminput, ropeinput, softmaxinput,
    };
    use cpu::cpuexecutor;
    use gpu::cuda::cudaexecutor;
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
    fn pair<F, G>(prefix: &str, name: &str, cpu: F, cuda: G)
    where
        F: Fn() -> tensor<f32>,
        G: Fn() -> tensor<f32>,
    {
        report(&run(&format!("{prefix}/cpu{name}"), || black_box(cpu())));
        report(&run(&format!("{prefix}/cuda{name}"), || black_box(cuda())));
    }

    pub fn matmul() {
        let (lhs, rhs) = matmulinput().unwrap();
        let (lv, rv) = (lhs.getview(), rhs.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let cudaexec = cudaexecutor::new(LIMIT);
        pair(
            "gpu/matmul",
            "256",
            || cpuexec.matmul(&lv, &rv).unwrap(),
            || cudaexec.matmul(&lv, &rv).unwrap(),
        );
    }

    pub fn matmul_batch() {
        let (lhs, rhs) = matmulbatchinput().unwrap();
        let (lv, rv) = (lhs.getview(), rhs.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let cudaexec = cudaexecutor::new(LIMIT);
        pair(
            "gpu/matmulbatch",
            "8x256",
            || cpuexec.matmul(&lv, &rv).unwrap(),
            || cudaexec.matmul(&lv, &rv).unwrap(),
        );
    }

    pub fn elementwise() {
        let x = elementwiseinput().unwrap();
        let y = elementwiseinput().unwrap();
        let v = x.getview();
        let w = y.getview();
        let cpuexec = cpuexecutor::new(LIMIT);
        let cudaexec = cudaexecutor::new(LIMIT);
        pair(
            "gpu/elementwise",
            "silu",
            || cpuexec.silu(&v).unwrap(),
            || cudaexec.silu(&v).unwrap(),
        );
        pair(
            "gpu/elementwise",
            "add",
            || cpuexec.add(&v, &w).unwrap(),
            || cudaexec.add(&v, &w).unwrap(),
        );
        pair(
            "gpu/elementwise",
            "mul",
            || cpuexec.mul(&v, &w).unwrap(),
            || cudaexec.mul(&v, &w).unwrap(),
        );
    }

    pub fn softmax() {
        let x = softmaxinput().unwrap();
        let v = x.getview();
        let cpuexec = cpuexecutor::new(LIMIT);
        let cudaexec = cudaexecutor::new(LIMIT);
        pair(
            "gpu/softmax",
            "rows512cols4096",
            || cpuexec.softmax(&v).unwrap(),
            || cudaexec.softmax(&v).unwrap(),
        );
    }

    pub fn rmsnorm() {
        let (x, w) = norminput().unwrap();
        let (xv, wv) = (x.getview(), w.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let cudaexec = cudaexecutor::new(LIMIT);
        pair(
            "gpu/rmsnorm",
            "rows512cols4096",
            || cpuexec.rmsnorm(&xv, &wv, EPS).unwrap(),
            || cudaexec.rmsnorm(&xv, &wv, EPS).unwrap(),
        );
    }

    pub fn rope() {
        let x = ropeinput().unwrap();
        let v = x.getview();
        let cpuexec = cpuexecutor::new(LIMIT);
        let cudaexec = cudaexecutor::new(LIMIT);
        pair(
            "gpu/rope",
            "seq128heads8dim64",
            || cpuexec.rope(&v, HEADS, BASE).unwrap(),
            || cudaexec.rope(&v, HEADS, BASE).unwrap(),
        );
    }

    pub fn embedding() {
        let (table, indices) = embeddinginput().unwrap();
        let v = table.getview();
        let cpuexec = cpuexecutor::new(LIMIT);
        let cudaexec = cudaexecutor::new(LIMIT);
        pair(
            "gpu/embedding",
            "gather128",
            || cpuexec.embedding(&v, &indices).unwrap(),
            || cudaexec.embedding(&v, &indices).unwrap(),
        );
    }

    pub fn attention() {
        let (q, k, v) = attentioninput().unwrap();
        let (qv, kv, vv) = (q.getview(), k.getview(), v.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let cudaexec = cudaexecutor::new(LIMIT);
        pair(
            "gpu/attention",
            "seq128heads8dim64",
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
    }
}

#[cfg(target_os = "linux")]
fn main() {
    linux::matmul();
    linux::matmul_batch();
    linux::elementwise();
    linux::softmax();
    linux::rmsnorm();
    linux::rope();
    linux::embedding();
    linux::attention();
}

#[cfg(not(target_os = "linux"))]
fn main() {}
