//! Criterion microbenchmarks for the raw CPU kernels.
//!
//! These benchmarks call the kernel types directly through the frozen `api`
//! kernel contracts, without the executor's budget checks. The end-to-end
//! executor path is measured separately by the `cpuops` target. Every workload
//! fixes its dimensions and dtype so results are reproducible. These
//! measurements establish the baseline required by `docs/performance.md`; no
//! improvement is claimed and no allocation is performed inside the timed region
//! beyond the output the kernel contract returns.
//!
//! Shared workload generators live in the `bench` library so that every target
//! measures identical inputs.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(missing_docs)]

use std::hint::black_box;

use ::tensor::{shape, tensor};
use api::{binarykernel, gatherkernel, kernel, weightedkernel};
use bench::{
    GATHER, HEADDIM, HEADS, MATMULDIM, NORMCOLS, NORMROWS, SEQLEN, attentioninput, embeddinginput,
    matmulinput, norminput, ropeinput,
};
use cpu::{
    cpuadd, cpuattention, cpuembedding, cpugelutanh, cpumatmul, cpumul, cpurelu, cpurmsnorm,
    cpurope, cpusilu, cpusoftmax,
};
use criterion::{Criterion, Throughput, criterion_group, criterion_main};

/// Element count for the elementwise activation workloads.
const ELEMENTWISE: usize = 1 << 20;

/// Row count for the softmax workload.
const SOFTMAXROWS: usize = 512;

/// Row width for the softmax workload.
const SOFTMAXCOLS: usize = 4096;

fn elementwise(c: &mut Criterion) {
    let t = tensor::new(
        shape::new(vec![ELEMENTWISE]).unwrap(),
        vec![0.5_f32; ELEMENTWISE],
    )
    .unwrap();
    let view = t.getview();

    let mut group = c.benchmark_group("kernel/elementwise");
    group.throughput(Throughput::Elements(ELEMENTWISE as u64));
    group.bench_function("relu", |b| {
        b.iter(|| black_box(cpurelu.apply(black_box(&view)).unwrap()));
    });
    group.bench_function("silu", |b| {
        b.iter(|| black_box(cpusilu.apply(black_box(&view)).unwrap()));
    });
    group.bench_function("gelutanh", |b| {
        b.iter(|| black_box(cpugelutanh.apply(black_box(&view)).unwrap()));
    });
    group.finish();
}

fn softmax(c: &mut Criterion) {
    let elements = SOFTMAXROWS * SOFTMAXCOLS;
    let t = tensor::new(
        shape::new(vec![SOFTMAXROWS, SOFTMAXCOLS]).unwrap(),
        vec![0.25_f32; elements],
    )
    .unwrap();
    let view = t.getview();

    let mut group = c.benchmark_group("kernel/softmax");
    group.throughput(Throughput::Elements(elements as u64));
    group.bench_function("rows512cols4096", |b| {
        b.iter(|| black_box(cpusoftmax.apply(black_box(&view)).unwrap()));
    });
    group.finish();
}

fn binary(c: &mut Criterion) {
    let (a, b) = matmulinput().unwrap();
    let (av, bv) = (a.getview(), b.getview());

    let mut group = c.benchmark_group("kernel/binary");
    group.throughput(Throughput::Elements((MATMULDIM * MATMULDIM) as u64));
    group.bench_function("matmul256", |bch| {
        bch.iter(|| black_box(cpumatmul.apply(black_box(&av), black_box(&bv)).unwrap()));
    });
    group.bench_function("add", |bch| {
        bch.iter(|| black_box(cpuadd.apply(black_box(&av), black_box(&bv)).unwrap()));
    });
    group.bench_function("mul", |bch| {
        bch.iter(|| black_box(cpumul.apply(black_box(&av), black_box(&bv)).unwrap()));
    });
    group.finish();
}

fn rmsnorm(c: &mut Criterion) {
    let (x, w) = norminput().unwrap();
    let (xv, wv) = (x.getview(), w.getview());
    let kernel = cpurmsnorm::new(cpurmsnorm::DEFAULTEPS);

    let mut group = c.benchmark_group("kernel/rmsnorm");
    group.throughput(Throughput::Elements((NORMROWS * NORMCOLS) as u64));
    group.bench_function("rows512cols4096", |b| {
        b.iter(|| black_box(kernel.apply(black_box(&xv), black_box(&wv)).unwrap()));
    });
    group.finish();
}

fn rope(c: &mut Criterion) {
    let t = ropeinput().unwrap();
    let view = t.getview();
    let kernel = cpurope::new(HEADS, 10000.0);

    let mut group = c.benchmark_group("kernel/rope");
    group.throughput(Throughput::Elements((SEQLEN * HEADS * HEADDIM) as u64));
    group.bench_function("seq128heads8dim64", |b| {
        b.iter(|| black_box(kernel.apply(black_box(&view)).unwrap()));
    });
    group.finish();
}

fn embedding(c: &mut Criterion) {
    let (table, indices) = embeddinginput().unwrap();
    let view = table.getview();

    let mut group = c.benchmark_group("kernel/embedding");
    group.throughput(Throughput::Elements((GATHER * bench::EMBEDDIM) as u64));
    group.bench_function("gather128", |b| {
        b.iter(|| {
            black_box(
                cpuembedding
                    .apply(black_box(&view), black_box(&indices))
                    .unwrap(),
            )
        });
    });
    group.finish();
}

fn attention(c: &mut Criterion) {
    let (q, k, v) = attentioninput().unwrap();
    let (qv, kv, vv) = (q.getview(), k.getview(), v.getview());

    let mut group = c.benchmark_group("kernel/attention");
    group.throughput(Throughput::Elements((SEQLEN * HEADS * HEADDIM) as u64));
    group.bench_function("seq128heads8dim64", |b| {
        b.iter(|| {
            black_box(
                cpuattention
                    .apply(
                        black_box(&qv),
                        black_box(&kv),
                        black_box(&vv),
                        HEADS,
                        HEADS,
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
    rmsnorm,
    rope,
    embedding,
    attention
);
criterion_main!(benches);
