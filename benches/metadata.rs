//! Criterion microbenchmarks for the tensor metadata layer.
//!
//! These benchmarks cover the O(rank) metadata operations that every tensor
//! creation and view passes through: shape validation, element-count checking,
//! contiguous-layout construction, and tensor ownership. They establish a
//! baseline required by `docs/performance.md`; no improvement is claimed.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(missing_docs)]

use std::hint::black_box;

use ::tensor::{layout, shape, tensor};
use criterion::{Criterion, criterion_group, criterion_main};

/// Side length used for the square metadata workloads.
const SIDE: usize = 1 << 10;

fn shapeops(c: &mut Criterion) {
    let s = shape::new(vec![SIDE, SIDE]).unwrap();
    c.bench_function("metadata/shapenew", |b| {
        b.iter(|| black_box(shape::new(vec![SIDE, SIDE]).unwrap()));
    });
    c.bench_function("metadata/elementcount", |b| {
        b.iter(|| black_box(s.getelementcount().unwrap()));
    });
}

fn layoutops(c: &mut Criterion) {
    let s = shape::new(vec![SIDE, SIDE]).unwrap();
    c.bench_function("metadata/contiguouslayout", |b| {
        b.iter(|| black_box(layout::contiguous(&s)));
    });
}

fn tensorops(c: &mut Criterion) {
    let data = vec![0.5_f32; SIDE * SIDE];
    c.bench_function("metadata/tensornew", |b| {
        b.iter(|| {
            let s = shape::new(vec![SIDE, SIDE]).unwrap();
            black_box(tensor::new(s, data.clone()).unwrap());
        });
    });
}

criterion_group!(benches, shapeops, layoutops, tensorops);
criterion_main!(benches);
