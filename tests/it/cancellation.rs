#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Cancellation semantics: a cancelled run must not advance training or
//! optimizer state, whether cancellation is observed between steps or in the
//! middle of a gradient-accumulation window.

use crate::support::build;
use core::cell::Cell;

use ::tensor::tensor;
use context::cancellation;
use lib::seed;
use optim::{optimizer, sgd, sgdconfig};
use train::{trainconfig, trainer, trainerror};

fn optimizer() -> sgd {
    sgd::new(sgdconfig::new(0.1, 0.0, 0.0).unwrap())
}

fn quadratic(params: &[tensor<f32>]) -> Result<(f32, Vec<tensor<f32>>), trainerror> {
    let value = params[0].getdata()[0];
    Ok((value * value, vec![build(vec![1], vec![2.0 * value])]))
}

#[test]
fn cancellation_between_steps_does_not_advance_state() {
    let token = cancellation::new();
    let config = trainconfig::single(0, seed::new(1)).unwrap();
    let mut trainer = trainer::new(vec![build(vec![1], vec![4.0])], optimizer(), config)
        .unwrap()
        .withcancellation(token.clone());

    trainer.step(quadratic).unwrap();
    trainer.step(quadratic).unwrap();
    let step = trainer.getstep();
    let optimstep = trainer.getoptimizer().getstep();
    let snapshot = trainer.getparams()[0].getdata().to_vec();

    token.cancel();
    assert!(matches!(
        trainer.step(quadratic),
        Err(trainerror::cancelled)
    ));
    assert_eq!(trainer.getstep(), step);
    assert_eq!(trainer.getoptimizer().getstep(), optimstep);
    assert_eq!(trainer.getparams()[0].getdata(), snapshot.as_slice());
    assert!(trainer.getaccumulator().isempty());
}

#[test]
fn cancellation_mid_window_discards_the_window() {
    let token = cancellation::new();
    let config = trainconfig::new(3, 0, None, seed::new(2)).unwrap();
    let mut trainer = trainer::new(vec![build(vec![1], vec![4.0])], optimizer(), config)
        .unwrap()
        .withcancellation(token.clone());

    let calls = Cell::new(0_usize);
    let trigger = token.clone();
    let result = trainer.step(move |params| {
        calls.set(calls.get() + 1);
        if calls.get() == 2 {
            trigger.cancel();
        }
        quadratic(params)
    });

    assert!(matches!(result, Err(trainerror::cancelled)));
    assert_eq!(trainer.getstep(), 0);
    assert_eq!(trainer.getoptimizer().getstep(), 0);
    assert!(trainer.getaccumulator().isempty());
    assert!((trainer.getparams()[0].getdata()[0] - 4.0).abs() < 1e-6);
}
