//! Criterion microbenchmark for the native Metal `matmul` kernel.
//!
//! macOS only: on other targets the `metal` crate is an empty library, so the
//! target compiles to a no-op `main`. The CPU executor measures the same
//! workload in the same run, so the Metal number is reported against a live
//! baseline rather than a stale constant. Both paths are measured with the
//! identical `bench` operands so the comparison is reproducible.
//!
//! No improvement is claimed: the per-call cost includes the run-time MSL
//! compilation this stage performs (see docs/design.md §21). Establishing the
//! baseline is required by docs/engineering.md; the comparison is reported in
//! docs/design.md §21.10.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(missing_docs)]

#[cfg(target_os = "macos")]
mod macos {
    use std::hint::black_box;

    use api::compute;
    use bench::{MATMULDIM, matmulinput};
    use cpu::cpuexecutor;
    use criterion::{Criterion, Throughput, criterion_group};
    use lib::limits;
    use metal::metalexecutor;

    /// Resource budget that admits the matmul workload without rejecting it.
    const LIMIT: limits = limits::new(1 << 28, 1 << 31, 1 << 20, 1 << 18);

    fn matmul(c: &mut Criterion) {
        let (lhs, rhs) = matmulinput().unwrap();
        let lv = lhs.getview();
        let rv = rhs.getview();

        let mut group = c.benchmark_group("gpu/matmul");
        group.throughput(Throughput::Elements((MATMULDIM * MATMULDIM) as u64));

        let cpuexec = cpuexecutor::new(LIMIT);
        group.bench_function("cpu256", |b| {
            b.iter(|| black_box(cpuexec.matmul(black_box(&lv), black_box(&rv)).unwrap()));
        });

        let metalexec = metalexecutor::new(LIMIT);
        group.bench_function("metal256", |b| {
            b.iter(|| black_box(metalexec.matmul(black_box(&lv), black_box(&rv)).unwrap()));
        });

        group.finish();
    }

    criterion_group!(benches, matmul);
}

#[cfg(target_os = "macos")]
criterion::criterion_main!(macos::benches);

#[cfg(not(target_os = "macos"))]
fn main() {}
