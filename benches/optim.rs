//! Criterion benchmark for the `optim` update rules and gradient clipping.
//!
//! Each workload is a single synthetic parameter block of shape
//! `[ROWS, COLS]` `f32` with a same-shaped gradient. Setup builds a fresh
//! optimizer and fresh buffers so every timed iteration performs exactly one
//! update with no cross-iteration state bleed, and the dimensions, dtype, and
//! hyperparameters are fixed so the measurement is reproducible. This
//! establishes a baseline; no improvement is claimed.
//!
//! Environment: see the completion report accompanying this file (operating
//! system, toolchain, and build profile are recorded there).
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(missing_docs)]

use std::hint::black_box;

use ::tensor::{shape, tensor};
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use optim::{adamw, adamwconfig, clipconfig, clipgradients, optimizer, sgd, sgdconfig};

/// Row count of the benchmark parameter block.
const ROWS: usize = 1024;

/// Column count of the benchmark parameter block.
const COLS: usize = 1024;

/// Total element count of the benchmark parameter block.
const ELEMENTS: u64 = (ROWS * COLS) as u64;

fn dense(value: f32) -> tensor<f32> {
    tensor::new(
        shape::new(vec![ROWS, COLS]).unwrap(),
        vec![value; ROWS * COLS],
    )
    .unwrap()
}

fn bench_adamw(c: &mut Criterion) {
    let mut group = c.benchmark_group("optim/adamw");
    group.throughput(Throughput::Elements(ELEMENTS));
    group.bench_function("step", |b| {
        b.iter_batched(
            || {
                let config = adamwconfig::new(1e-3, 0.9, 0.999, 1e-8, 0.01).unwrap();
                let opt = adamw::new(config);
                let params = vec![dense(0.1)];
                let grads = vec![dense(0.01)];
                (opt, params, grads)
            },
            |(mut opt, mut params, mut grads)| {
                opt.update(
                    black_box(params.as_mut_slice()),
                    black_box(grads.as_mut_slice()),
                )
                .unwrap();
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

fn bench_sgd(c: &mut Criterion) {
    let mut group = c.benchmark_group("optim/sgd");
    group.throughput(Throughput::Elements(ELEMENTS));
    group.bench_function("step", |b| {
        b.iter_batched(
            || {
                let config = sgdconfig::new(1e-3, 0.9, 0.01).unwrap();
                let opt = sgd::new(config);
                let params = vec![dense(0.1)];
                let grads = vec![dense(0.01)];
                (opt, params, grads)
            },
            |(mut opt, mut params, mut grads)| {
                opt.update(
                    black_box(params.as_mut_slice()),
                    black_box(grads.as_mut_slice()),
                )
                .unwrap();
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

fn bench_clip(c: &mut Criterion) {
    let mut group = c.benchmark_group("optim/clipgradients");
    group.throughput(Throughput::Elements(ELEMENTS));
    group.bench_function("bynorm", |b| {
        b.iter_batched(
            || vec![dense(0.5)],
            |mut grads| {
                clipgradients(
                    black_box(grads.as_mut_slice()),
                    clipconfig::bynorm(1.0).unwrap(),
                );
            },
            BatchSize::LargeInput,
        );
    });
    group.finish();
}

criterion_group!(benches, bench_adamw, bench_sgd, bench_clip);
criterion_main!(benches);
