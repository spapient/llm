//! Benchmark for the `optim` update rules and gradient clipping.
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

use ::bench::harness::{report, run};
use ::tensor::{shape, tensor};
use optim::{adamw, adamwconfig, clipconfig, clipgradients, optimizer, sgd, sgdconfig};

/// Row count of the benchmark parameter block.
const ROWS: usize = 1024;

/// Column count of the benchmark parameter block.
const COLS: usize = 1024;

fn dense(value: f32) -> tensor<f32> {
    tensor::new(
        shape::new(vec![ROWS, COLS]).unwrap(),
        vec![value; ROWS * COLS],
    )
    .unwrap()
}

fn bench_adamw() {
    report(&run("optim/adamw/step", || {
        let config = adamwconfig::new(1e-3, 0.9, 0.999, 1e-8, 0.01).unwrap();
        let mut opt = adamw::new(config);
        let mut params = vec![dense(0.1)];
        let mut grads = vec![dense(0.01)];
        opt.update(
            black_box(params.as_mut_slice()),
            black_box(grads.as_mut_slice()),
        )
        .unwrap();
    }));
}

fn bench_sgd() {
    report(&run("optim/sgd/step", || {
        let config = sgdconfig::new(1e-3, 0.9, 0.01).unwrap();
        let mut opt = sgd::new(config);
        let mut params = vec![dense(0.1)];
        let mut grads = vec![dense(0.01)];
        opt.update(
            black_box(params.as_mut_slice()),
            black_box(grads.as_mut_slice()),
        )
        .unwrap();
    }));
}

fn bench_clip() {
    report(&run("optim/clipgradients/bynorm", || {
        let mut grads = vec![dense(0.5)];
        clipgradients(
            black_box(grads.as_mut_slice()),
            clipconfig::bynorm(1.0).unwrap(),
        );
    }));
}

fn main() {
    bench_adamw();
    bench_sgd();
    bench_clip();
}
