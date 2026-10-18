//! Contract tests for the persistent device-input primitives a device KV
//! cache builds on: `recordpersist` (register caller-owned data as a
//! device-resident step), `recordappend` (copy a new row into the persistent
//! buffer without a host round trip), and `recordview` (retag the leading
//! rows with the window shape).
//!
//! The CPU submission is the reference implementation: it snapshots the
//! registered data, appends in place, and copies the leading elements for a
//! view. These tests pin the semantics every backend must reproduce — the
//! `forwardsubmission_cached_device` path reads `cached` rows through
//! `operand::result` of a persistent step and appends through
//! `recordappend`, so a wrong CPU implementation fails every backend.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use crate::support::{build, wide};
use api::{backenderror, operand, submission};
use cpu::{cpuexecutor, cpusubmission};
use lib::limits;
use tensor::shape;

/// A persistent buffer round-trips its registered data unchanged.
#[test]
fn recordpersist_returns_the_registered_data() {
    let exec = cpuexecutor::new(wide());
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let s = shape::new(vec![2, 3]).unwrap();

    let mut sub = cpusubmission::new(exec);
    let handle = sub.recordpersist(&data, s).unwrap();
    let out = sub.apply(&[handle]).unwrap();

    assert_eq!(out.len(), 1);
    assert_eq!(out[0].getshape().getdims(), &[2, 3]);
    assert_eq!(out[0].getdata(), &data[..]);
}

/// `recordpersist` rejects a data length that does not match the shape.
#[test]
fn recordpersist_rejects_mismatched_length() {
    let exec = cpuexecutor::new(wide());
    let s = shape::new(vec![2, 3]).unwrap();
    let mut sub = cpusubmission::new(exec);
    let err = sub.recordpersist(&[1.0, 2.0], s).unwrap_err();
    assert!(matches!(err, backenderror::invalidparameter { op: "persist" }));
}

/// Appending rows into a persistent buffer keeps the earlier rows and extends
/// the window the caller reads through `recordview`.
#[test]
fn recordappend_writes_rows_in_place() {
    let exec = cpuexecutor::new(wide());
    let maxseq = 4;
    let width = 2;
    let mut sub = cpusubmission::new(exec);
    let cached = sub
        .recordpersist(
            &vec![0.0; maxseq * width],
            shape::new(vec![maxseq, width]).unwrap(),
        )
        .unwrap();
    let row0 = build(vec![1, width], vec![1.5, -2.5]);
    sub.recordappend(operand::result(cached), operand::input(row0.getview()), 0)
        .unwrap();
    let row1 = build(vec![1, width], vec![3.25, 4.75]);
    sub.recordappend(operand::result(cached), operand::input(row1.getview()), 1)
        .unwrap();
    let window = sub
        .recordview(cached, shape::new(vec![2, width]).unwrap())
        .unwrap();
    let out = sub.apply(&[window]).unwrap();
    assert_eq!(out[0].getdata(), &[1.5, -2.5, 3.25, 4.75]);
}

/// Appending past the registered buffer is rejected.
#[test]
fn recordappend_rejects_out_of_bounds_row() {
    let exec = cpuexecutor::new(wide());
    let maxseq = 2;
    let width = 2;
    let mut sub = cpusubmission::new(exec);
    let cached = sub
        .recordpersist(
            &vec![0.0; maxseq * width],
            shape::new(vec![maxseq, width]).unwrap(),
        )
        .unwrap();
    let row = build(vec![1, width], vec![1.0, 1.0]);
    let err = sub
        .recordappend(operand::result(cached), operand::input(row.getview()), 2)
        .unwrap_err();
    assert!(matches!(err, backenderror::invalidparameter { op: "append" }));
}

/// Appending into a caller input (not a recorded step) is rejected.
#[test]
fn recordappend_rejects_input_cached() {
    let exec = cpuexecutor::new(wide());
    let maxseq = 2;
    let width = 2;
    let mut sub = cpusubmission::new(exec);
    let cached = build(vec![maxseq, width], vec![0.0; maxseq * width]);
    let row = build(vec![1, width], vec![1.0, 1.0]);
    let err = sub
        .recordappend(
            operand::input(cached.getview()),
            operand::input(row.getview()),
            0,
        )
        .unwrap_err();
    assert!(matches!(err, backenderror::invalidparameter { op: "append" }));
}

/// A view narrower than the buffer reads the leading rows only.
#[test]
fn recordview_reads_the_leading_rows() {
    let exec = cpuexecutor::new(wide());
    let mut sub = cpusubmission::new(exec);
    let base = sub
        .recordpersist(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            shape::new(vec![3, 2]).unwrap(),
        )
        .unwrap();
    let prefix = sub.recordview(base, shape::new(vec![2, 2]).unwrap()).unwrap();
    let out = sub.apply(&[prefix]).unwrap();
    assert_eq!(out[0].getdata(), &[1.0, 2.0, 3.0, 4.0]);
}

/// A view larger than the base buffer is rejected.
#[test]
fn recordview_rejects_oversized_shape() {
    let exec = cpuexecutor::new(wide());
    let mut sub = cpusubmission::new(exec);
    let base = sub
        .recordpersist(&[1.0, 2.0, 3.0, 4.0], shape::new(vec![2, 2]).unwrap())
        .unwrap();
    let err = sub
        .recordview(base, shape::new(vec![3, 2]).unwrap())
        .unwrap_err();
    assert!(matches!(err, backenderror::invalidparameter { op: "view" }));
}
