//! Zero-copy byte view benchmarks: element-wise decode vs reinterpret + copy.
//!
//! Both paths produce the same owned [`tensor`] from the same bytes. The
//! element-wise path (`tensor::tryfrombytes`) decodes every element through
//! `from_ne_bytes`; the fast path (`tensor::trybyteview` + `to_vec`) validates
//! length and alignment once, reinterprets the bytes as a typed view, and
//! copies once. The third group measures the view construction alone to show
//! that the reinterpret itself costs nothing (it aliases the source bytes).
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![allow(missing_docs)]

use std::hint::black_box;

use ::bench::harness::{report, run};
use ::tensor::{shape, tensor};

fn workload() -> (shape, Vec<u8>) {
    let n = 4 * 1024 * 1024; // 4 Mi elements = 16 MiB of f32.
    let s = shape::new(vec![n]).unwrap();
    // Deterministic pseudo-random bits, every bit pattern valid for f32.
    let bytes: Vec<u8> = (0..n * 4)
        .map(|i| ((i * 31 + (i >> 9) * 7) % 251) as u8)
        .collect();
    (s, bytes)
}

fn element_wise(s: &shape, bytes: &[u8]) -> usize {
    let t: tensor<f32> = tensor::tryfrombytes(s.clone(), bytes).unwrap();
    t.getlen()
}

fn view_copy(s: &shape, bytes: &[u8]) -> usize {
    let view = ::tensor::trybyteview::<f32>(bytes).unwrap();
    let t = tensor::new(s.clone(), view.to_vec()).unwrap();
    t.getlen()
}

fn view_only(bytes: &[u8]) -> usize {
    ::tensor::trybyteview::<f32>(bytes).unwrap().len()
}

fn run_benches() {
    let (s, bytes) = workload();
    report(&run("byteview/element_wise_from_ne_bytes", || {
        black_box(element_wise(&s, &bytes))
    }));
    report(&run("byteview/reinterpret_then_copy", || {
        black_box(view_copy(&s, &bytes))
    }));
    report(&run("byteview/view_only_alias", || {
        black_box(view_only(&bytes))
    }));
}

fn main() {
    run_benches();
}
