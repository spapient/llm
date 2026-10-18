//! Benchmark for the autograd reverse pass.
//!
//! The workload is a small but representative chain — matrix multiply, RMS
//! normalization, and softmax — built once per sample so that gradient
//! accumulation does not bleed across iterations. Only the reverse pass is
//! timed; the forward graph is constructed in the setup phase. The
//! dimensions and dtype are fixed so the measurement is reproducible. This
//! establishes a baseline; no improvement is claimed.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(missing_docs)]

use std::hint::black_box;

use ::bench::harness::{report, run};
use ::tensor::{shape, tensor};
use autograd::{graph, variableid};
use cpu::cpuexecutor;
use lib::limits;

/// Row count of the benchmark input.
const ROWS: usize = 64;

/// Feature width of the benchmark input.
const WIDTH: usize = 128;

fn engine() -> cpuexecutor {
    cpuexecutor::new(limits::new(1 << 22, 1 << 26, 1 << 16, 1 << 8))
}

fn dense(dims: Vec<usize>, value: f32) -> tensor<f32> {
    let elements = dims.iter().product();
    tensor::new(shape::new(dims).unwrap(), vec![value; elements]).unwrap()
}

/// Builds the forward graph and returns it with its output handle.
fn build() -> (graph<f32>, variableid) {
    let engine = engine();
    let mut g = graph::new();
    let x = g.leaf(dense(vec![ROWS, WIDTH], 0.05), true);
    let w = g.leaf(dense(vec![WIDTH, WIDTH], 0.01), true);
    let gamma = g.leaf(dense(vec![WIDTH], 1.0), true);
    let h = g.matmul(&engine, x, w).unwrap();
    let n = g.rmsnorm(&engine, h, gamma, 1.0e-5).unwrap();
    let p = g.softmax(&engine, n).unwrap();
    (g, p)
}

fn reverse() {
    report(&run("autograd/backward/matmul_rmsnorm_softmax", || {
        let (mut g, out) = build();
        g.backward(black_box(out)).unwrap();
    }));
}

fn main() {
    reverse();
}
