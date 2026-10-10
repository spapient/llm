//! Criterion benchmarks for the CPU kernels.
//!
//! Every workload fixes its dimensions and dtype so results are reproducible.
//! These measurements establish the initial baseline required by
//! `docs/engineering.md`; no improvement is claimed and no allocation is
//! performed inside the timed region beyond the output the kernel contract
//! returns.
//!
//! The workload constructors live in the `bench` library so that this target
//! stays a thin measurement harness.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(missing_docs)]

use std::hint::black_box;

use api::compute;
use bench::{
    ELEMENTWISE, SOFTMAXCOLS, SOFTMAXROWS, attentioninput, cpuexec, elementwiseinput,
    embeddinginput, matmulinput, norminput, ropeinput, softmaxinput,
};
use cpu::cpuop;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

fn elementwise(c: &mut Criterion) {
    let exec = cpuexec();
    let t = elementwiseinput().unwrap();
    let view = t.getview();

    let mut group = c.benchmark_group("elementwise");
    group.throughput(Throughput::Elements(ELEMENTWISE as u64));
    for op in [cpuop::relu, cpuop::silu, cpuop::gelutanh] {
        group.bench_function(op.getname(), |b| {
            b.iter(|| black_box(exec.apply(op, black_box(&view)).unwrap()));
        });
    }
    group.finish();
}

fn softmax(c: &mut Criterion) {
    let exec = cpuexec();
    let t = softmaxinput().unwrap();
    let view = t.getview();

    let mut group = c.benchmark_group("softmax");
    group.throughput(Throughput::Elements((SOFTMAXROWS * SOFTMAXCOLS) as u64));
    group.bench_function("rows512cols4096", |b| {
        b.iter(|| black_box(exec.apply(cpuop::softmax, black_box(&view)).unwrap()));
    });
    group.finish();
}

fn binary(c: &mut Criterion) {
    let exec = cpuexec();
    let (a, b) = matmulinput().unwrap();
    let (av, bv) = (a.getview(), b.getview());

    let mut group = c.benchmark_group("executor/binary");
    group.throughput(Throughput::Elements(
        (bench::MATMULDIM * bench::MATMULDIM) as u64,
    ));
    for op in [cpuop::matmul, cpuop::add, cpuop::mul] {
        group.bench_function(op.getname(), |bench| {
            bench.iter(|| {
                black_box(
                    exec.applybinary(op, black_box(&av), black_box(&bv))
                        .unwrap(),
                )
            });
        });
    }
    group.finish();
}

fn weighted(c: &mut Criterion) {
    let exec = cpuexec();
    let (x, w) = norminput().unwrap();
    let (xv, wv) = (x.getview(), w.getview());

    let mut group = c.benchmark_group("executor/weighted");
    group.throughput(Throughput::Elements(
        (bench::NORMROWS * bench::NORMCOLS) as u64,
    ));
    group.bench_function("rmsnorm", |b| {
        b.iter(|| {
            black_box(
                exec.applyweighted(cpuop::rmsnorm, black_box(&xv), black_box(&wv))
                    .unwrap(),
            )
        });
    });
    group.finish();
}

fn gather(c: &mut Criterion) {
    let exec = cpuexec();
    let (table, indices) = embeddinginput().unwrap();
    let view = table.getview();

    let mut group = c.benchmark_group("executor/gather");
    group.throughput(Throughput::Elements(
        (bench::GATHER * bench::EMBEDDIM) as u64,
    ));
    group.bench_function("embedding", |b| {
        b.iter(|| {
            black_box(
                exec.applygather(cpuop::embedding, black_box(&view), black_box(&indices))
                    .unwrap(),
            )
        });
    });
    group.finish();
}

fn compute_paths(c: &mut Criterion) {
    let exec = cpuexec();
    let r = ropeinput().unwrap();
    let rv = r.getview();
    let (q, k, v) = attentioninput().unwrap();
    let (qv, kv, vv) = (q.getview(), k.getview(), v.getview());

    let mut group = c.benchmark_group("executor/compute");
    group.throughput(Throughput::Elements(
        (bench::SEQLEN * bench::HEADS * bench::HEADDIM) as u64,
    ));
    group.bench_function("rope", |b| {
        b.iter(|| black_box(compute::rope(&exec, black_box(&rv), bench::HEADS, 10000.0).unwrap()));
    });
    group.bench_function("attention", |b| {
        b.iter(|| {
            black_box(
                compute::attention(
                    &exec,
                    black_box(&qv),
                    black_box(&kv),
                    black_box(&vv),
                    bench::HEADS,
                    bench::HEADS,
                    true,
                )
                .unwrap(),
            )
        });
    });
    group.finish();
}

criterion_group!(
    benches,
    elementwise,
    softmax,
    binary,
    weighted,
    gather,
    compute_paths
);
criterion_main!(benches);
