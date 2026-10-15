//! Contract tests for the accumulative `submission` boundary through the CPU
//! reference implementation, exercised only through public APIs.
//!
//! The CPU submission replays `compute` in order, so these tests pin the one
//! property the contract exists for: a chain driven through `submission`
//! produces exactly the bits a chain driven one operator at a time through
//! `compute` produces, and a declaration failure is the same failure `compute`
//! would report for the same rejected operand.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use crate::support::{build, wide};
use api::{backenderror, compute, operand, stepref, submission};
use cpu::{cpuexecutor, cpusubmission};
use lib::limits;

#[test]
fn submission_reproduces_a_compute_chain_bit_for_bit() {
    let exec = cpuexecutor::new(wide());
    let a = build(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = build(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let c = build(vec![2, 2], vec![0.5, -0.5, 1.5, -1.5]);

    let mut sub = cpusubmission::new(exec.clone());
    let matmul = sub
        .recordmatmul(operand::input(a.getview()), operand::input(b.getview()))
        .unwrap();
    let sum = sub
        .recordadd(operand::result(matmul), operand::input(c.getview()))
        .unwrap();
    let silu = sub.recordsilu(operand::result(sum)).unwrap();
    let got = sub.apply(&[silu]).unwrap();

    let want_matmul = exec.matmul(&a.getview(), &b.getview()).unwrap();
    let want_sum = exec.add(&want_matmul.getview(), &c.getview()).unwrap();
    let want_silu = exec.silu(&want_sum.getview()).unwrap();

    assert_eq!(got.len(), 1);
    assert_eq!(got[0].getdata(), want_silu.getdata());
}

#[test]
fn apply_returns_requested_outputs_in_order() {
    let exec = cpuexecutor::new(wide());
    let a = build(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = build(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);

    let mut sub = cpusubmission::new(exec.clone());
    let sum = sub
        .recordadd(operand::input(a.getview()), operand::input(b.getview()))
        .unwrap();
    let prod = sub
        .recordmul(operand::input(a.getview()), operand::input(b.getview()))
        .unwrap();
    let got = sub.apply(&[prod, sum]).unwrap();

    let want_sum = exec.add(&a.getview(), &b.getview()).unwrap();
    let want_prod = exec.mul(&a.getview(), &b.getview()).unwrap();
    assert_eq!(got.len(), 2);
    assert_eq!(got[0].getdata(), want_prod.getdata());
    assert_eq!(got[1].getdata(), want_sum.getdata());
}

#[test]
fn empty_submission_returns_no_outputs() {
    let exec = cpuexecutor::new(wide());
    let sub = cpusubmission::new(exec);
    let got = sub.apply(&[]).unwrap();
    assert!(got.is_empty());
}

#[test]
fn single_step_submission_matches_compute() {
    let exec = cpuexecutor::new(wide());
    let a = build(vec![1, 3], vec![1.0, 2.0, 3.0]);

    let mut sub = cpusubmission::new(exec.clone());
    let step = sub.recordsoftmax(operand::input(a.getview())).unwrap();
    let got = sub.apply(&[step]).unwrap();

    let want = exec.softmax(&a.getview()).unwrap();
    assert_eq!(got[0].getdata(), want.getdata());
}

#[test]
fn record_rejects_incompatible_shapes_like_compute() {
    let exec = cpuexecutor::new(wide());
    let a = build(vec![2, 3], vec![1.0; 6]);
    let b = build(vec![3, 2], vec![1.0; 6]);

    let via_compute = exec.add(&a.getview(), &b.getview()).unwrap_err();

    let mut sub = cpusubmission::new(exec.clone());
    let via_sub = sub
        .recordadd(operand::input(a.getview()), operand::input(b.getview()))
        .unwrap_err();

    assert_eq!(via_sub, via_compute);
}

#[test]
fn record_rejects_budget_like_compute() {
    let tight = limits::new(3, usize::MAX, usize::MAX, usize::MAX);
    let exec = cpuexecutor::new(tight);
    let a = build(vec![2, 2], vec![1.0; 4]);

    let via_compute = exec.add(&a.getview(), &a.getview()).unwrap_err();

    let mut sub = cpusubmission::new(exec.clone());
    let via_sub = sub
        .recordadd(operand::input(a.getview()), operand::input(a.getview()))
        .unwrap_err();

    assert_eq!(via_sub, via_compute);
}

#[test]
fn record_rejects_an_unrecorded_result_reference() {
    let exec = cpuexecutor::new(wide());
    let a = build(vec![2, 2], vec![1.0; 4]);
    let bogus = stepref::fromindex(99);

    let mut sub = cpusubmission::new(exec);
    let err = sub
        .recordadd(operand::result(bogus), operand::input(a.getview()))
        .unwrap_err();

    assert_eq!(err, backenderror::indexoutofrange { index: 99, len: 0 });
}

#[test]
fn apply_rejects_an_unrecorded_output() {
    let exec = cpuexecutor::new(wide());
    let sub = cpusubmission::new(exec);
    let bogus = stepref::fromindex(5);

    let err = sub.apply(&[bogus]).unwrap_err();

    assert_eq!(err, backenderror::indexoutofrange { index: 5, len: 0 });
}
