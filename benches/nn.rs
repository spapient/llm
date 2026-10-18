//! Criterion benchmarks for the composed `nn` layers.
//!
//! Each layer is built once outside the timed section and then run through the
//! CPU `api::compute` provider. Dimensions are fixed so the measurements are
//! reproducible. These figures are baselines; no improvement is claimed, and
//! no fused or blocked kernel is introduced here (the layers compose the
//! existing `cpu` operators).
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(missing_docs)]

use std::hint::black_box;

use ::bench::harness::{report, run};
use ::tensor::{shape, tensor};
use cpu::cpuexecutor;
use lib::limits;
use nn::{attention, linear, mlp, parameter};

/// Row count of the benchmark input.
const ROWS: usize = 128;

/// Model dimension.
const DIM: usize = 512;

/// Feed-forward hidden dimension.
const HIDDEN: usize = 1365;

/// Head count.
const HEADS: usize = 8;

/// Head dimension.
const HEADDIM: usize = 64;

fn engine() -> cpuexecutor {
    cpuexecutor::new(limits::new(1 << 24, 1 << 28, 1 << 20, 1 << 16))
}

fn dense(dims: Vec<usize>, value: f32) -> tensor<f32> {
    let elements = dims.iter().product();
    tensor::new(shape::new(dims).unwrap(), vec![value; elements]).unwrap()
}

fn projection(name: &str, in_features: usize, out_features: usize) -> linear {
    linear::new(
        parameter::new(name, dense(vec![in_features, out_features], 0.02)),
        None,
    )
    .unwrap()
}

fn linearforward() {
    let exec = engine();
    let layer = projection("weight", DIM, DIM);
    let input = dense(vec![ROWS, DIM], 0.05);
    report(&run("nn/linear/forward", || {
        layer.forward(&exec, black_box(&input)).unwrap()
    }));
}

fn mlpforward() {
    let exec = engine();
    let layer = mlp::new(
        projection("weight", DIM, HIDDEN),
        projection("weight", DIM, HIDDEN),
        projection("weight", HIDDEN, DIM),
    )
    .unwrap();
    let input = dense(vec![ROWS, DIM], 0.05);
    report(&run("nn/mlp/forward", || {
        layer.forward(&exec, black_box(&input)).unwrap()
    }));
}

fn attentionforward() {
    let exec = engine();
    let layer = attention::new(HEADS, HEADS, true).unwrap();
    let q = dense(vec![ROWS, HEADS * HEADDIM], 0.05);
    let k = dense(vec![ROWS, HEADS * HEADDIM], 0.05);
    let v = dense(vec![ROWS, HEADS * HEADDIM], 0.05);
    report(&run("nn/attention/forward", || {
        layer
            .forward(&exec, black_box(&q), black_box(&k), black_box(&v))
            .unwrap()
    }));
}

fn main() {
    linearforward();
    mlpforward();
    attentionforward();
}
