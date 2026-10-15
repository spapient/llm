//! Deterministic test that a full forward pass recorded through `submission`
//! reproduces the `compute`-driven forward bit for bit.
//!
//! The CPU submission replays `compute` in order, so this pins the one property
//! the submission forward exists for: inlining the Llama operator chain into a
//! submission produces exactly the logits the `causallm::forward` path produces.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use crate::common::{model, provider};
use api::submission;
use cpu::cpusubmission;
use models::causallm;

#[test]
fn forward_submitted_matches_forward_bit_for_bit() {
    let exec = provider();
    let model = model(42);
    let tokens: Vec<usize> = vec![0, 1, 2, 3];

    let mut sub = cpusubmission::new(exec.clone());
    let logits = model.forward_submitted(&mut sub, &tokens).unwrap();
    let got = sub.apply(&[logits]).unwrap();

    let want = model.forward(&exec, &tokens).unwrap();

    assert_eq!(got.len(), 1);
    assert_eq!(got[0].getdata(), want.getdata());
}

#[test]
fn forward_submitted_matches_forward_for_a_single_token() {
    let exec = provider();
    let model = model(7);
    let tokens: Vec<usize> = vec![1];

    let mut sub = cpusubmission::new(exec.clone());
    let logits = model.forward_submitted(&mut sub, &tokens).unwrap();
    let got = sub.apply(&[logits]).unwrap();

    let want = model.forward(&exec, &tokens).unwrap();

    assert_eq!(got.len(), 1);
    assert_eq!(got[0].getdata(), want.getdata());
}
