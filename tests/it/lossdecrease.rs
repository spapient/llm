#![allow(clippy::unwrap_used, clippy::expect_used)]

//! Genuine loss-decrease tests.
//!
//! Both tests run a real optimization that must decrease the measured metric.
//! The first drives an `autograd` graph built only from operators that have a
//! backward path today (`matmul`, `add`, `mul`) on a convex least-squares
//! problem with the known optimum `w = 2`. The second drives the crate's own
//! cross-entropy loss and its analytic gradient on a separable classification
//! problem.

use ::tensor::{shape, tensor};
use autograd::graph;
use cpu::cpuexecutor;
use lib::{limits, seed};
use optim::{sgd, sgdconfig};
use train::{crossentropy, trainconfig, trainer, trainerror};

fn build(dims: Vec<usize>, data: Vec<f32>) -> tensor<f32> {
    tensor::new(shape::new(dims).unwrap(), data).unwrap()
}

#[test]
fn autograd_least_squares_decreases_the_loss() {
    let executor = cpuexecutor::new(limits::new(1 << 24, 1 << 28, 1 << 20, 1 << 16));
    // The data lie exactly on `y = 2x`, so the least-squares optimum is `w = 2`.
    let x = build(vec![3, 1], vec![1.0, 2.0, 3.0]);
    let negy = build(vec![3, 1], vec![-2.0, -4.0, -6.0]);
    let ones = build(vec![1, 3], vec![1.0, 1.0, 1.0]);

    let params = vec![build(vec![1, 1], vec![0.0])];
    let optimizer = sgd::new(sgdconfig::new(0.01, 0.0, 0.0).unwrap());
    let config = trainconfig::single(0, seed::new(2024)).unwrap();
    let mut trainer = trainer::new(params, optimizer, config).unwrap();

    let mut losses = Vec::new();
    for _ in 0..300 {
        let result = trainer
            .step(|params| {
                let mut g = graph::<f32>::new();
                let w = g.leaf(params[0].clone(), true);
                let xc = g.constant(x.clone());
                let pred = g
                    .matmul(&executor, xc, w)
                    .map_err(|error| trainerror::state(error.to_string()))?;
                let yc = g.constant(negy.clone());
                let residual = g
                    .add(&executor, pred, yc)
                    .map_err(|error| trainerror::state(error.to_string()))?;
                let squared = g
                    .mul(&executor, residual, residual)
                    .map_err(|error| trainerror::state(error.to_string()))?;
                let onesc = g.constant(ones.clone());
                let loss = g
                    .matmul(&executor, onesc, squared)
                    .map_err(|error| trainerror::state(error.to_string()))?;
                g.backward(loss)
                    .map_err(|error| trainerror::state(error.to_string()))?;
                let value = g
                    .getvalue(loss)
                    .ok_or_else(|| trainerror::state("loss value missing"))?
                    .getdata()[0];
                let grad = g
                    .getgrad(w)
                    .ok_or_else(|| trainerror::state("gradient missing"))?
                    .clone();
                Ok((value, vec![grad]))
            })
            .unwrap();
        losses.push(result.getloss());
    }

    let first = losses[0];
    let last = *losses.last().unwrap();
    let weight = trainer.getparams()[0].getdata()[0];
    println!("least-squares: first={first} last={last} weight={weight}");

    // Analytical initial loss: 14 * (0 - 2)^2 = 56.
    assert!(first > 50.0, "initial loss should be near 56, was {first}");
    assert!(
        last < first * 1e-3,
        "loss did not decrease: {first} -> {last}"
    );
    assert!(last < 1e-2, "final loss too large: {last}");
    assert!((weight - 2.0).abs() < 1e-3, "weight {weight} not near 2");
}

fn crossentropyloss(params: &[tensor<f32>]) -> Result<(f32, Vec<tensor<f32>>), trainerror> {
    let result = crossentropy(&params[0], &[0_usize, 1_usize])?;
    let (loss, grad) = result.intoparts();
    Ok((loss, vec![grad]))
}

#[test]
fn cross_entropy_descent_decreases_the_loss() {
    // Two separable rows over three classes; the target classes are reachable.
    let params = vec![build(vec![2, 3], vec![0.0; 6])];
    let optimizer = sgd::new(sgdconfig::new(0.3, 0.0, 0.0).unwrap());
    let config = trainconfig::single(0, seed::new(7)).unwrap();
    let mut trainer = trainer::new(params, optimizer, config).unwrap();

    let mut losses = Vec::new();
    for _ in 0..500 {
        let result = trainer.step(crossentropyloss).unwrap();
        losses.push(result.getloss());
    }

    let first = losses[0];
    let last = *losses.last().unwrap();
    println!("cross-entropy: first={first} last={last}");

    // Uniform logits give ln(3) per row.
    assert!(
        (first - (3.0_f32).ln()).abs() < 1e-5,
        "unexpected initial loss {first}"
    );
    assert!(last < first, "loss did not decrease: {first} -> {last}");
    assert!(last < 0.1, "final loss too large: {last}");
}
