//! Cross-crate numerical-gradient tests for `autograd`.
//!
//! Every differentiable operator is checked against a central finite
//! difference of a scalar loss `L = <output, cotangent>`. The analytic
//! vector-Jacobian product comes from [`graph::backwardwith`]; the numeric one
//! perturbs each input element in turn and recomputes the forward value through
//! the `cpu` backend. This proves the reverse pass implements the derivatives
//! the forward operators actually compute.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use crate::support::build;
use ::tensor::tensor;
use autograd::{autoggraderror, graph, variableid};
use cpu::cpuexecutor;
use lib::limits;

fn engine() -> cpuexecutor {
    cpuexecutor::new(limits::new(1 << 20, 1 << 24, 1 << 16, 1 << 8))
}

/// Recomputes the scalar loss `sum(output * cotangent)` with one input element
/// perturbed by `delta`.
fn perturbedloss(
    inputs: &[tensor<f32>],
    index: usize,
    element: usize,
    delta: f32,
    cotangent: &tensor<f32>,
    buildgraph: &impl Fn(&mut graph<f32>, &[variableid]) -> Result<variableid, autoggraderror>,
) -> f32 {
    let mut g = graph::new();
    let mut ids = Vec::with_capacity(inputs.len());
    for (i, input) in inputs.iter().enumerate() {
        let mut data = input.getdata().to_vec();
        if i == index {
            data[element] += delta;
        }
        let value = tensor::new(input.getshape().clone(), data).unwrap();
        ids.push(g.leaf(value, true));
    }
    let out = buildgraph(&mut g, &ids).unwrap();
    let y = g.getvalue(out).unwrap();
    y.getdata()
        .iter()
        .zip(cotangent.getdata())
        .map(|(&a, &b)| a * b)
        .sum()
}

/// Asserts that the analytic gradients of `buildgraph` match a central finite
/// difference at every input element.
fn checkvjp(
    inputs: &[tensor<f32>],
    cotangent: &tensor<f32>,
    buildgraph: impl Fn(&mut graph<f32>, &[variableid]) -> Result<variableid, autoggraderror>,
) {
    let mut g = graph::new();
    let ids: Vec<variableid> = inputs.iter().map(|t| g.leaf(t.clone(), true)).collect();
    let out = buildgraph(&mut g, &ids).unwrap();
    g.backwardwith(out, Some(cotangent.clone())).unwrap();
    let analytic: Vec<Vec<f32>> = ids
        .iter()
        .map(|id| {
            g.getgrad(*id)
                .map_or_else(Vec::new, |t| t.getdata().to_vec())
        })
        .collect();

    let h = 1.0e-3_f32;
    for (i, input) in inputs.iter().enumerate() {
        for (p, &got) in analytic[i].iter().enumerate().take(input.getlen()) {
            let plus = perturbedloss(inputs, i, p, h, cotangent, &buildgraph);
            let minus = perturbedloss(inputs, i, p, -h, cotangent, &buildgraph);
            let numeric = (plus - minus) / (2.0 * h);
            let tol = 2.0e-3 + 2.0e-2 * numeric.abs();
            assert!(
                (got - numeric).abs() <= tol,
                "input {i} index {p}: analytic {got} numeric {numeric} tol {tol}"
            );
        }
    }
}

#[test]
fn matmul_vjp_matches_finite_difference() {
    let engine = engine();
    let a = build(vec![2, 3], vec![0.5, -0.25, 1.0, 0.75, 0.1, -0.4]);
    let b = build(vec![3, 2], vec![0.3, -0.6, 0.9, 0.2, -0.8, 0.4]);
    let cot = build(vec![2, 2], vec![0.5, -1.0, 2.0, 0.25]);
    checkvjp(&[a, b], &cot, |g, ids| g.matmul(&engine, ids[0], ids[1]));
}

#[test]
fn add_vjp_matches_finite_difference() {
    let engine = engine();
    let a = build(vec![2, 2], vec![0.5, -1.5, 2.0, 0.25]);
    let b = build(vec![2, 2], vec![1.0, 0.5, -0.5, 3.0]);
    let cot = build(vec![2, 2], vec![1.0, -2.0, 0.5, 0.75]);
    checkvjp(&[a, b], &cot, |g, ids| g.add(&engine, ids[0], ids[1]));
}

#[test]
fn mul_vjp_matches_finite_difference() {
    let engine = engine();
    let a = build(vec![2, 2], vec![0.5, -1.5, 2.0, 0.25]);
    let b = build(vec![2, 2], vec![1.0, 0.5, -0.5, 3.0]);
    let cot = build(vec![2, 2], vec![1.0, -2.0, 0.5, 0.75]);
    checkvjp(&[a, b], &cot, |g, ids| g.mul(&engine, ids[0], ids[1]));
}

#[test]
fn rmsnorm_vjp_matches_finite_difference() {
    let engine = engine();
    let x = build(vec![2, 3], vec![0.5, -0.75, 1.25, 1.0, 0.25, -0.5]);
    let w = build(vec![3], vec![1.0, 0.5, 1.5]);
    let cot = build(vec![2, 3], vec![0.5, -1.0, 0.25, 1.5, 0.75, -0.5]);
    checkvjp(&[x, w], &cot, |g, ids| {
        g.rmsnorm(&engine, ids[0], ids[1], 1.0e-5)
    });
}

#[test]
fn softmax_vjp_matches_finite_difference() {
    let engine = engine();
    let x = build(vec![2, 3], vec![0.5, -0.75, 1.25, 1.0, 0.25, -0.5]);
    let cot = build(vec![2, 3], vec![0.5, -1.0, 0.25, 1.5, 0.75, -0.5]);
    checkvjp(&[x], &cot, |g, ids| g.softmax(&engine, ids[0]));
}

#[test]
fn relu_vjp_matches_finite_difference() {
    let x = build(vec![4], vec![-1.5, -0.25, 0.75, 2.0]);
    let cot = build(vec![4], vec![0.5, -1.0, 2.0, 0.25]);
    checkvjp(&[x], &cot, |g, ids| g.relu(ids[0]));
}

#[test]
fn silu_vjp_matches_finite_difference() {
    let engine = engine();
    let x = build(vec![4], vec![-1.5, -0.25, 0.75, 2.0]);
    let cot = build(vec![4], vec![0.5, -1.0, 2.0, 0.25]);
    checkvjp(&[x], &cot, |g, ids| g.silu(&engine, ids[0]));
}

#[test]
fn gelutanh_vjp_matches_finite_difference() {
    let x = build(vec![4], vec![-1.5, -0.25, 0.75, 2.0]);
    let cot = build(vec![4], vec![0.5, -1.0, 2.0, 0.25]);
    checkvjp(&[x], &cot, |g, ids| g.gelutanh(ids[0]));
}

#[test]
fn accumulation_sums_contributions_from_every_consumer() {
    let engine = engine();
    let a = build(vec![3], vec![0.5, -1.0, 2.0]);
    let cot = build(vec![3], vec![1.0, 0.5, -2.0]);
    checkvjp(&[a], &cot, |g, ids| g.mul(&engine, ids[0], ids[0]));
}

#[test]
fn nograd_scope_detaches_outputs() {
    let engine = engine();
    let mut g = graph::new();
    let a = g.leaf(build(vec![2], vec![0.5, -1.0]), true);
    let detached = {
        let mut guard = g.nograd();
        guard.mul(&engine, a, a).unwrap()
    };
    assert!(!g.requiresgrad(detached).unwrap());
    assert!(g.get(detached).unwrap().getproducer().isleaf());
    g.backward(detached).unwrap();
    assert!(g.getgrad(a).is_none());
}

#[test]
fn detach_stops_gradient_flow() {
    let engine = engine();
    let mut g = graph::new();
    let a = g.leaf(build(vec![2], vec![0.5, -1.0]), true);
    let stopped = g.detach(a).unwrap();
    let out = g.mul(&engine, stopped, a).unwrap();
    g.backward(out).unwrap();
    assert!(g.getgrad(a).is_some());
    assert!(g.getgrad(stopped).is_none());
}

#[test]
fn cleargradients_resets_before_a_second_pass() {
    let engine = engine();
    let mut g = graph::new();
    let a = g.leaf(build(vec![2], vec![0.5, -1.0]), true);
    let out = g.mul(&engine, a, a).unwrap();
    g.backward(out).unwrap();
    let first = g.getgrad(a).unwrap().getdata().to_vec();
    g.cleargradients();
    g.backward(out).unwrap();
    assert_eq!(g.getgrad(a).unwrap().getdata(), first.as_slice());
}

#[test]
fn unsupported_shapes_are_reported() {
    let engine = engine();
    let mut g = graph::new();
    let a = g.leaf(build(vec![2, 3], vec![0.0; 6]), true);
    let b = g.leaf(build(vec![2, 2], vec![0.0; 4]), true);
    assert!(g.matmul(&engine, a, b).is_err());
    let c = g.leaf(build(vec![3], vec![0.0; 3]), true);
    assert!(g.add(&engine, a, c).is_err());
}
