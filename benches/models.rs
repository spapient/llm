//! Criterion benchmark for the end-to-end Llama forward pass.
//!
//! The model is built once from deterministic random weights; only the forward
//! pass is timed. The dimensions are fixed so the measurement is reproducible.
//! This is the correctness-first baseline: it recomputes the full sequence and
//! uses no KV cache, so a later cached decode phase has a measured target to
//! beat. No improvement is claimed.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(missing_docs)]

use std::hint::black_box;

use cpu::cpuexecutor;
use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use lib::{limits, seed};
use models::{causallm, llamaconfig, llamamodel, llamaweights};

/// Vocabulary size.
const VOCAB: usize = 256;

/// Model dimension.
const DIM: usize = 64;

/// Number of transformer layers.
const LAYERS: usize = 2;

/// Query head count.
const NHEADS: usize = 4;

/// Key/value head count.
const KVHEADS: usize = 2;

/// Feed-forward hidden dimension.
const HIDDEN: usize = 170;

/// Sequence length.
const SEQ: usize = 16;

fn model() -> llamamodel {
    let c = llamaconfig::new(
        VOCAB, DIM, LAYERS, NHEADS, KVHEADS, HIDDEN, 1.0e-5, 10_000.0, 64,
    )
    .unwrap();
    let w = llamaweights::random(&c, seed::new(42)).unwrap();
    llamamodel::new(c, w).unwrap()
}

fn forward(c: &mut Criterion) {
    let exec = cpuexecutor::new(limits::new(1 << 24, 1 << 28, 1 << 20, 1 << 16));
    let m = model();
    let tokens: Vec<usize> = (0..SEQ).map(|i| i % VOCAB).collect();
    let mut group = c.benchmark_group("models/llama");
    group.throughput(Throughput::Elements(SEQ as u64));
    group.bench_function("forward", |b| {
        b.iter(|| m.forward(&exec, black_box(&tokens)).unwrap());
    });
    group.finish();
}

criterion_group!(benches, forward);
criterion_main!(benches);
