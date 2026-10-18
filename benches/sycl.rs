//! Criterion microbenchmarks for the Intel oneAPI SYCL backend.
//!
//! Linux and Windows only: on other targets the `sycl` module is compiled out,
//! so the target compiles to a no-op `main`. The CPU executor measures every
//! workload in the same run, so each SYCL number is reported against a live
//! baseline rather than a stale constant. Both paths are driven through the
//! same `api::compute` contract with identical `bench` operands, so the
//! comparison is reproducible.
//!
//! The matmul numbers measure whichever path the `gpu` crate was built with:
//! `crates/gpu/build.rs` enables oneMKL (`GRID_MKL`) only when the headers and
//! all four libraries are present, and falls back to the portable tiled kernel
//! otherwise. Record the toolkit presence alongside any number taken here.
//!
//! No improvement is claimed: these numbers are a baseline recorded on the
//! validation host (Intel UHD Graphics 750) and documented in
//! `docs/design.md`. A SYCL number is only comparable to another SYCL number
//! taken on the same host, toolchain, and MKL configuration.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(missing_docs)]

#[cfg(all(any(target_os = "linux", target_os = "windows"), sycl_toolkit))]
mod device {
    use std::hint::black_box;

    use ::tensor::{shape, tensor};
    use api::{compute, operand, submission};
    use bench::{MATMULDIM, matmulbatchinput, matmulinput};
    use cpu::cpuexecutor;
    use criterion::{
        BenchmarkGroup, Criterion, Throughput, criterion_group, measurement::WallTime,
    };
    use gpu::sycl::executor as syclexecutor;
    use lib::limits;

    /// Resource budget that admits every workload without rejecting it.
    const LIMIT: limits = limits::new(1 << 28, 1 << 31, 1 << 20, 1 << 18);

    /// Measures the `cpu` and `sycl` closures of one workload back to back.
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
        sycl: G,
    ) where
        F: Fn() -> tensor<f32>,
        G: Fn() -> tensor<f32>,
    {
        group.throughput(Throughput::Elements(elements as u64));
        group.bench_function(format!("cpu{name}"), |b| b.iter(|| black_box(cpu())));
        group.bench_function(format!("sycl{name}"), |b| b.iter(|| black_box(sycl())));
    }

    fn matmul(c: &mut Criterion) {
        let (lhs, rhs) = matmulinput().unwrap();
        let (lv, rv) = (lhs.getview(), rhs.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let syclexec = syclexecutor::new(LIMIT);
        let mut group = c.benchmark_group("sycl/matmul");
        pair(
            &mut group,
            "256",
            MATMULDIM * MATMULDIM,
            || cpuexec.matmul(&lv, &rv).unwrap(),
            || syclexec.matmul(&lv, &rv).unwrap(),
        );
        group.finish();
    }

    fn matmul_batch(c: &mut Criterion) {
        let (lhs, rhs) = matmulbatchinput().unwrap();
        let (lv, rv) = (lhs.getview(), rhs.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let syclexec = syclexecutor::new(LIMIT);
        let mut group = c.benchmark_group("sycl/matmulbatch");
        pair(
            &mut group,
            "8x256",
            lhs.getlen(),
            || cpuexec.matmul(&lv, &rv).unwrap(),
            || syclexec.matmul(&lv, &rv).unwrap(),
        );
        group.finish();
    }

    /// Device-resident matmul chain: eight 1024x1024 contractions per
    /// iteration, with both operands uploaded once and every intermediate
    /// result kept in a device slot. Only the final `apply` synchronises and
    /// downloads, so the measurement is the GPU path plus submission, without
    /// the per-call host round trip the compute API pays.
    fn matmul_chain(c: &mut Criterion) {
        const SIDE: usize = 1024;
        const STEPS: usize = 8;
        let elements = SIDE * SIDE;
        let lhs = tensor::new(
            shape::new(vec![SIDE, SIDE]).unwrap(),
            vec![0.5_f32; elements],
        )
        .unwrap();
        let rhs = tensor::new(
            shape::new(vec![SIDE, SIDE]).unwrap(),
            vec![0.25_f32; elements],
        )
        .unwrap();
        let (lv, rv) = (lhs.getview(), rhs.getview());
        let syclexec = syclexecutor::new(LIMIT);
        let mut group = c.benchmark_group("sycl/matmulchain");
        group.throughput(Throughput::Elements((elements * STEPS) as u64));
        group.bench_function("sycl1024x8", |b| {
            b.iter(|| {
                let mut sub = syclexec.newsubmission().unwrap();
                let mut step = sub
                    .recordmatmul(operand::input(lv), operand::input(rv))
                    .unwrap();
                for _ in 1..STEPS {
                    step = sub
                        .recordmatmul(operand::result(step), operand::input(rv))
                        .unwrap();
                }
                black_box(sub.apply(&[step]).unwrap())
            })
        });
        group.finish();
    }

    /// A 32-step elementwise chain recorded through the accumulative
    /// submission (one synchronise at `apply`) against the same chain run as
    /// 32 separate compute calls (one synchronise each), on both backends.
    fn submissionchain(criterion: &mut Criterion) {
        let steps = 32usize;
        let elements = 64 * 64;
        let (x, y) = (
            tensor::new(shape::new(vec![64, 64]).unwrap(), vec![0.5_f32; elements]).unwrap(),
            tensor::new(shape::new(vec![64, 64]).unwrap(), vec![0.25_f32; elements]).unwrap(),
        );
        let (xv, yv) = (x.getview(), y.getview());
        let cpuexec = cpuexecutor::new(LIMIT);
        let syclexec = syclexecutor::new(LIMIT);
        let mut group = criterion.benchmark_group("sycl/submissionchain");
        group.throughput(Throughput::Elements((elements * steps) as u64));
        group.bench_function("cpucompute32", |b| {
            b.iter(|| {
                let mut acc = x.clone();
                for _ in 0..steps {
                    acc = cpuexec.add(&acc.getview(), &yv).unwrap();
                }
                black_box(acc)
            })
        });
        group.bench_function("syclcompute32", |b| {
            b.iter(|| {
                let mut acc = x.clone();
                for _ in 0..steps {
                    acc = syclexec.add(&acc.getview(), &yv).unwrap();
                }
                black_box(acc)
            })
        });
        group.bench_function("syclsubmission32", |b| {
            b.iter(|| {
                let mut sub = syclexec.newsubmission().unwrap();
                let mut current = operand::input(xv);
                let mut last = None;
                for _ in 0..steps {
                    last = Some(sub.recordadd(current, operand::input(yv)).unwrap());
                    current = operand::result(last.unwrap());
                }
                black_box(sub.apply(&[last.unwrap()]).unwrap())
            })
        });
        group.finish();
    }

    criterion_group!(benches, matmul, matmul_batch, matmul_chain, submissionchain);
}

#[cfg(all(any(target_os = "linux", target_os = "windows"), sycl_toolkit))]
criterion::criterion_main!(device::benches);

#[cfg(not(all(any(target_os = "linux", target_os = "windows"), sycl_toolkit)))]
fn main() {}
