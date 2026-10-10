//! Cross-crate integration tests for the public `optim` API.
//!
//! These tests compose an optimizer with gradient clipping, a learning-rate
//! schedule, and the `checkpoint` tensor contract through their public
//! interfaces. The checkpoint edge is a development-only dependency; it does
//! not enter `optim`'s production dependency closure.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use crate::support::build;
use ::tensor::dtype;
use checkpoint::checkpoint;
use optim::{
    adamw, adamwconfig, clipconfig, lrschedule, optimizer, optimizerstate, sgd, sgdconfig,
    statetensor,
};

/// Persists a snapshot through the checkpoint tensor contract and reads it back.
fn roundtrip(state: &optimizerstate) -> optimizerstate {
    let mut artifact = checkpoint::new(1);
    for entry in state.gettensors() {
        artifact
            .addtensor(
                entry.getname().to_string(),
                dtype::f32,
                entry.getshape().clone(),
                &entry.tobytes(),
            )
            .unwrap();
    }
    artifact.addmetadata("optim.step".to_string(), state.getstep().to_string());
    artifact.addmetadata("optim.lr".to_string(), state.getlr().to_string());
    artifact.addmetadata(
        "optim.params".to_string(),
        state.getparamcount().to_string(),
    );

    let bytes = artifact.encode().unwrap();
    let decoded = checkpoint::decode(&bytes).unwrap();

    let mut tensors = Vec::new();
    for entry in decoded.getmanifest().gettensors() {
        let raw = decoded.readtensor(entry.getname()).unwrap();
        tensors
            .push(statetensor::frombytes(entry.getname(), entry.getshape().clone(), &raw).unwrap());
    }
    let metadata = decoded.getmanifest().getmetadata();
    let step: u64 = metadata.get("optim.step").unwrap().parse().unwrap();
    let lr: f32 = metadata.get("optim.lr").unwrap().parse().unwrap();
    let params: usize = metadata.get("optim.params").unwrap().parse().unwrap();
    optimizerstate::new(step, lr, params, tensors)
}

#[test]
fn adamw_state_survives_a_checkpoint_roundtrip() {
    let cfg = adamwconfig::new(0.05, 0.9, 0.98, 1e-8, 0.01).unwrap();
    let mut original = adamw::new(cfg);
    let mut params = [build([3], vec![1.0, -2.0, 0.5])];
    let mut grads = [build([3], vec![0.2, -0.1, 0.3])];
    for _ in 0..5 {
        original.update(&mut params, &mut grads).unwrap();
    }

    let restored_state = roundtrip(&original.getstate());
    let mut restored = adamw::new(cfg);
    restored.restore(&restored_state).unwrap();
    assert_eq!(restored.getstep(), original.getstep());
    assert_eq!(restored.getlr(), original.getlr());

    // Continue both optimizers from the same post-update state.
    let mut a_params = [params[0].clone()];
    let mut a_grads = [grads[0].clone()];
    let mut b_params = [params[0].clone()];
    let mut b_grads = [grads[0].clone()];
    for _ in 0..5 {
        original.update(&mut a_params, &mut a_grads).unwrap();
        restored.update(&mut b_params, &mut b_grads).unwrap();
    }
    assert_eq!(restored.getstep(), original.getstep());
    assert_eq!(b_params[0].getdata(), a_params[0].getdata());
}

#[test]
fn sgd_state_survives_a_checkpoint_roundtrip() {
    let cfg = sgdconfig::new(0.1, 0.9, 0.0).unwrap();
    let mut original = sgd::new(cfg);
    let mut params = [build([2, 2], vec![0.1, 0.2, 0.3, 0.4])];
    let mut grads = [build([2, 2], vec![0.4, 0.3, 0.2, 0.1])];
    original.update(&mut params, &mut grads).unwrap();

    let restored_state = roundtrip(&original.getstate());
    let mut restored = sgd::new(cfg);
    restored.restore(&restored_state).unwrap();

    let mut a_params = [params[0].clone()];
    let mut a_grads = [grads[0].clone()];
    let mut b_params = [params[0].clone()];
    let mut b_grads = [grads[0].clone()];
    original.update(&mut a_params, &mut a_grads).unwrap();
    restored.update(&mut b_params, &mut b_grads).unwrap();
    assert_eq!(b_params[0].getdata(), a_params[0].getdata());
}

#[test]
fn a_schedule_drives_the_optimizer_learning_rate() {
    let schedule = lrschedule::linearwarmupcosine(0.0, 0.1, 0.0, 2, 6).unwrap();
    let cfg = adamwconfig::new(0.1, 0.9, 0.999, 1e-8, 0.0)
        .unwrap()
        .withclip(clipconfig::byvalue(-1.0, 1.0).unwrap());
    let mut opt = adamw::new(cfg);
    let mut params = [build([1], vec![0.0])];
    let mut grads = [build([1], vec![1.0])];
    for step in 0..4_u64 {
        opt.setlr(schedule.getlr(step)).unwrap();
        opt.update(&mut params, &mut grads).unwrap();
        assert_eq!(opt.getstep(), step + 1);
    }
    // The optimizer now carries the rate for the step after the warmup peak.
    assert_eq!(opt.getlr(), schedule.getlr(3));
    assert!(schedule.getlr(1) < schedule.getlr(2));
    assert!(schedule.getlr(3) < schedule.getlr(2));
}
