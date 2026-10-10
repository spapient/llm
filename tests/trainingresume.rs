#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Resume orchestration: round-trip persistence of parameters, optimizer state,
//! and progress, and continued-training equivalence.

use std::path::PathBuf;

use ::tensor::{shape, tensor};
use lib::seed;
use optim::{adamw, adamwconfig, optimizer};
use train::{loadresume, saveresume, trainconfig, trainer, trainerror};

fn build(dims: Vec<usize>, data: Vec<f32>) -> tensor<f32> {
    tensor::new(shape::new(dims).unwrap(), data).unwrap()
}

fn path(label: &str) -> PathBuf {
    std::env::temp_dir().join(format!("train-{}-{label}.ckpt", std::process::id()))
}

fn optimizer() -> adamw {
    adamw::new(adamwconfig::new(0.05, 0.9, 0.999, 1e-8, 0.0).unwrap())
}

fn quadratic(params: &[tensor<f32>]) -> Result<(f32, Vec<tensor<f32>>), trainerror> {
    let loss = params[0].getdata().iter().map(|value| value * value).sum();
    let grad = build(
        vec![2],
        params[0]
            .getdata()
            .iter()
            .map(|value| 2.0 * value)
            .collect(),
    );
    Ok((loss, vec![grad]))
}

#[test]
fn resume_restores_params_state_and_progress() {
    let config = trainconfig::single(0, seed::new(11)).unwrap();
    let mut original =
        trainer::new(vec![build(vec![2], vec![1.0, -2.0])], optimizer(), config).unwrap();
    for _ in 0..5 {
        original.step(quadratic).unwrap();
    }
    original.setepoch(3);

    let file = path("roundtrip");
    saveresume(&original, &file).unwrap();
    let savedparams = original.getparams()[0].getdata().to_vec();
    let savedstate = original.getoptimizer().getstate();
    let savedstep = original.getstep();

    let mut restored =
        trainer::new(vec![build(vec![2], vec![0.0, 0.0])], optimizer(), config).unwrap();
    let state = loadresume(&mut restored, &file).unwrap();

    assert_eq!(state.getstep(), savedstep);
    assert_eq!(state.getepoch(), 3);
    assert_eq!(restored.getstep(), savedstep);
    assert_eq!(restored.getepoch(), 3);
    assert_eq!(restored.getparams()[0].getdata(), savedparams.as_slice());
    assert_eq!(restored.getoptimizer().getstate(), savedstate);

    // Continuing both from the same checkpoint must produce identical updates.
    for _ in 0..4 {
        let a = original.step(quadratic).unwrap();
        let b = restored.step(quadratic).unwrap();
        assert_eq!(a.getloss(), b.getloss());
        assert_eq!(a.getstep(), b.getstep());
        assert_eq!(a.getlearningrate(), b.getlearningrate());
    }
    assert_eq!(
        original.getparams()[0].getdata(),
        restored.getparams()[0].getdata()
    );

    std::fs::remove_file(&file).ok();
}

#[test]
fn a_shape_mismatch_is_rejected_without_mutation() {
    let config = trainconfig::single(0, seed::new(3)).unwrap();
    let mut writer =
        trainer::new(vec![build(vec![2], vec![1.0, 1.0])], optimizer(), config).unwrap();
    writer.step(quadratic).unwrap();
    let file = path("mismatch");
    saveresume(&writer, &file).unwrap();

    let mut reader = trainer::new(
        vec![build(vec![3], vec![0.0, 0.0, 0.0])],
        optimizer(),
        config,
    )
    .unwrap();
    let before = reader.getparams()[0].getdata().to_vec();
    assert!(loadresume(&mut reader, &file).is_err());
    assert_eq!(reader.getstep(), 0);
    assert_eq!(reader.getparams()[0].getdata(), before.as_slice());

    std::fs::remove_file(&file).ok();
}

#[test]
fn a_missing_checkpoint_is_reported() {
    let config = trainconfig::single(0, seed::new(5)).unwrap();
    let mut reader = trainer::new(vec![build(vec![1], vec![0.0])], optimizer(), config).unwrap();
    let missing = path("does-not-exist");
    assert!(loadresume(&mut reader, &missing).is_err());
}

#[test]
fn a_truncated_checkpoint_is_rejected() {
    let config = trainconfig::single(0, seed::new(7)).unwrap();
    let mut writer =
        trainer::new(vec![build(vec![2], vec![1.0, 1.0])], optimizer(), config).unwrap();
    writer.step(quadratic).unwrap();
    let file = path("truncated");
    saveresume(&writer, &file).unwrap();
    let bytes = std::fs::read(&file).unwrap();
    let short = path("truncated-short");
    std::fs::write(&short, &bytes[..bytes.len() / 2]).unwrap();

    let mut reader =
        trainer::new(vec![build(vec![2], vec![0.0, 0.0])], optimizer(), config).unwrap();
    assert!(loadresume(&mut reader, &short).is_err());
    std::fs::remove_file(&file).ok();
    std::fs::remove_file(&short).ok();
}
