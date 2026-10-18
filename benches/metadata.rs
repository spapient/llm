//! Microbenchmarks for the tensor metadata layer.
//!
//! These benchmarks cover the O(rank) metadata operations that every tensor
//! creation and view passes through: shape validation, element-count checking,
//! contiguous-layout construction, and tensor ownership. They establish a
//! baseline required by `docs/engineering.md`; no improvement is claimed.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(missing_docs)]

use std::hint::black_box;

use ::bench::harness::{report, run};
use ::tensor::{layout, shape, tensor};

/// Side length used for the square metadata workloads.
const SIDE: usize = 1 << 10;

fn shapeops() {
    let s = shape::new(vec![SIDE, SIDE]).unwrap();
    report(&run("metadata/shapenew", || {
        black_box(shape::new(vec![SIDE, SIDE]).unwrap());
    }));
    report(&run("metadata/elementcount", || {
        black_box(s.getelementcount().unwrap());
    }));
}

fn layoutops() {
    let s = shape::new(vec![SIDE, SIDE]).unwrap();
    report(&run("metadata/contiguouslayout", || {
        black_box(layout::contiguous(&s));
    }));
}

fn tensorops() {
    let data = vec![0.5_f32; SIDE * SIDE];
    report(&run("metadata/tensornew", || {
        let s = shape::new(vec![SIDE, SIDE]).unwrap();
        black_box(tensor::new(s, data.clone()).unwrap());
    }));
}

fn main() {
    shapeops();
    layoutops();
    tensorops();
}
