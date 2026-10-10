//! Cross-crate contract test.
//!
//! This suite exercises `lib`, `tensor`, `api`, and `cpu` together through
//! their public interfaces only, and cross-checks every CPU kernel against the
//! independent reference implementations in `testkit::reference`. It is the
//! end-to-end integration test for the currently implemented layers.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use crate::common::provider;
use api::{backend, device};
use cpu::{cpubackend, cpuop};
use lib::limits;
use tensor::dtype;
use testkit::{assertions::assertallclose, fixtures, reference};

/// Absolute tolerance for elementwise kernels on `f32`.
const EWABS: f32 = 1e-5;

/// Relative tolerance for elementwise kernels on `f32`.
const EWREL: f32 = 1e-5;

/// Tolerance for softmax rows on `f32`.
const SOFTMAXTOL: f32 = 1e-6;

#[test]
fn cpu_backend_declares_cpu_identity_and_f32_only() {
    let b = cpubackend::new(limits::new(1024, 4096, 128, 8));
    assert_eq!(b.getname(), "cpu");
    assert!(b.getdevice().iscpu());
    assert_eq!(b.getdevice().getkind(), device::cpu().getkind());

    let caps = b.getcapabilities();
    assert!(caps.supports(dtype::f32));
    assert!(!caps.supports(dtype::f16));
    assert_eq!(caps.getdtypes(), &[dtype::f32]);
}

#[test]
fn relu_matches_reference() {
    let input = fixtures::ramp(64).unwrap();
    let expected = reference::map(input.getdata(), reference::relu);
    let out = provider().apply(cpuop::relu, &input.getview()).unwrap();
    assertallclose(out.getdata(), &expected, EWABS, EWREL);
}

#[test]
fn silu_matches_reference() {
    let input = fixtures::ramp(64).unwrap();
    let expected = reference::map(input.getdata(), reference::silu);
    let out = provider().apply(cpuop::silu, &input.getview()).unwrap();
    assertallclose(out.getdata(), &expected, EWABS, EWREL);
}

#[test]
fn gelutanh_matches_reference() {
    let input = fixtures::ramp(64).unwrap();
    let expected = reference::map(input.getdata(), reference::gelutanh);
    let out = provider().apply(cpuop::gelutanh, &input.getview()).unwrap();
    assertallclose(out.getdata(), &expected, EWABS, EWREL);
}

#[test]
fn softmax_matches_reference_per_row() {
    let rows = 8;
    let cols = 16;
    let input = fixtures::sequential(rows, cols).unwrap();
    let out = provider().apply(cpuop::softmax, &input.getview()).unwrap();

    assert_eq!(out.getshape().getdims(), &[rows, cols]);
    for r in 0..rows {
        let start = r * cols;
        let end = start + cols;
        let expected = reference::softmax(&input.getdata()[start..end]);
        assertallclose(
            &out.getdata()[start..end],
            &expected,
            SOFTMAXTOL,
            SOFTMAXTOL,
        );
    }
}

#[test]
fn softmax_rows_sum_to_one() {
    let input = fixtures::matrix(4, 8, 0.3).unwrap();
    let out = provider().apply(cpuop::softmax, &input.getview()).unwrap();
    for row in out.getdata().chunks(8) {
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() <= SOFTMAXTOL, "row sum {sum}");
    }
}
