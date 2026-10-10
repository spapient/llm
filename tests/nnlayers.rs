//! Cross-crate layer tests for `nn` through its public API.
//!
//! Each layer is exercised against an independent scalar reference from
//! `testkit` with the CPU backend as the `api::compute` provider. The tests
//! also pin the shape-validation errors a caller must be able to rely on.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use ::tensor::{shape, tensor};
use cpu::cpuexecutor;
use lib::{limits, seed};
use nn::{
    DEFAULTBASE, DEFAULTEPS, attention, embedding, linear, mlp, module, nnerror, parameter,
    rmsnorm, rotary,
};
use testkit::{assertions::assertallclose, reference};

fn build(dims: Vec<usize>, data: Vec<f32>) -> tensor<f32> {
    tensor::new(shape::new(dims).unwrap(), data).unwrap()
}

fn param(name: &str, dims: Vec<usize>, data: Vec<f32>) -> parameter {
    parameter::new(name, build(dims, data))
}

fn wide() -> limits {
    limits::new(1 << 24, 1 << 28, 1 << 20, 1 << 16)
}

fn exec() -> cpuexecutor {
    cpuexecutor::new(wide())
}

#[test]
fn linear_forward_matches_the_reference_with_bias() {
    // weight [in=3, out=2], bias [2].
    let w = param("weight", vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = param("bias", vec![2], vec![0.5, -0.5]);
    let layer = linear::new(w, Some(b)).unwrap();

    let input = build(vec![2, 3], vec![1.0, 0.0, -1.0, 2.0, 1.0, 0.0]);
    let out = layer.forward(&exec(), &input).unwrap();
    assert_eq!(out.getshape().getdims(), &[2, 2]);

    let mut want = reference::matmul(
        input.getdata(),
        layer.getweight().getvalue().getdata(),
        2,
        3,
        2,
    );
    for row in want.chunks_mut(2) {
        row[0] += 0.5;
        row[1] -= 0.5;
    }
    assertallclose(out.getdata(), &want, 1e-5, 1e-5);
}

#[test]
fn linear_preserves_leading_batch_dimensions() {
    let w = param("weight", vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let layer = linear::new(w, None).unwrap();
    let input = build(vec![2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let out = layer.forward(&exec(), &input).unwrap();
    assert_eq!(out.getshape().getdims(), &[2, 2, 2]);
    assertallclose(out.getdata(), input.getdata(), 1e-6, 1e-6);
}

#[test]
fn linear_rejects_a_mismatched_last_dimension() {
    let w = param("weight", vec![3, 2], vec![0.0; 6]);
    let layer = linear::new(w, None).unwrap();
    let input = build(vec![2, 4], vec![0.0; 8]);
    assert!(matches!(
        layer.forward(&exec(), &input),
        Err(nnerror::shapemismatch {
            layer: "linear",
            ..
        })
    ));
}

#[test]
fn embedding_forward_matches_the_reference() {
    let weight = param("weight", vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let layer = embedding::new(weight).unwrap();
    let indices = [2usize, 0];
    let out = layer.forward(&exec(), &indices).unwrap();
    assert_eq!(out.getshape().getdims(), &[2, 2]);
    let want = reference::embedding(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, &indices);
    assertallclose(out.getdata(), &want, 1e-6, 1e-6);
}

#[test]
fn embedding_rejects_an_out_of_range_index() {
    let weight = param("weight", vec![2, 2], vec![0.0; 4]);
    let layer = embedding::new(weight).unwrap();
    assert!(matches!(
        layer.forward(&exec(), &[5]),
        Err(nnerror::backend(_))
    ));
}

#[test]
fn rmsnorm_forward_matches_the_reference() {
    let weight = param("weight", vec![3], vec![1.0, 2.0, 3.0]);
    let layer = rmsnorm::new(weight, DEFAULTEPS).unwrap();
    let input = build(vec![2, 3], vec![1.0, 2.0, 3.0, -1.0, 0.0, 1.0]);
    let out = layer.forward(&exec(), &input).unwrap();

    let w = layer.getweight().getvalue().getdata();
    let mut want = Vec::new();
    for row in input.getdata().chunks(3) {
        want.extend(reference::rmsnorm(row, w, DEFAULTEPS));
    }
    assertallclose(out.getdata(), &want, 1e-5, 1e-5);
}

#[test]
fn rotary_forward_matches_the_reference() {
    let layer = rotary::new(2, DEFAULTBASE).unwrap();
    // seq = 2, heads = 2, headdim = 2.
    let input = build(vec![2, 4], vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, -0.5, 0.5]);
    let out = layer.forward(&exec(), &input).unwrap();
    assert_eq!(out.getshape().getdims(), &[2, 4]);

    let mut want = Vec::new();
    for (s, row) in input.getdata().chunks(4).enumerate() {
        want.extend(reference::rope(row, s, 2, 2, DEFAULTBASE));
    }
    assertallclose(out.getdata(), &want, 1e-5, 1e-5);
}

/// Scalar reference for grouped-query causal attention.
fn attend(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    heads: usize,
    kvheads: usize,
    headdim: usize,
) -> Vec<f32> {
    let stride = heads * headdim;
    let kstride = kvheads * headdim;
    let group = heads / kvheads;
    let scale = 1.0 / (headdim as f32).sqrt();
    let mut out = vec![0.0_f32; seq * stride];
    for head in 0..heads {
        let kvhead = head / group;
        for s in 0..seq {
            let qrow = &q[s * stride + head * headdim..][..headdim];
            let mut logits = vec![f32::NEG_INFINITY; seq];
            for (t, slot) in logits.iter_mut().enumerate() {
                if t <= s {
                    let krow = &k[t * kstride + kvhead * headdim..][..headdim];
                    *slot = qrow.iter().zip(krow).map(|(&a, &b)| a * b).sum::<f32>() * scale;
                }
            }
            let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            for (t, &p) in exps.iter().enumerate() {
                let vrow = &v[t * kstride + kvhead * headdim..][..headdim];
                for (o, &x) in vrow.iter().enumerate() {
                    out[s * stride + head * headdim + o] += (p / sum) * x;
                }
            }
        }
    }
    out
}

#[test]
fn attention_forward_matches_a_scalar_reference() {
    // seq = 2, heads = 2, kvheads = 1, headdim = 2.
    let q = build(vec![2, 4], vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 1.0, -1.0]);
    let k = build(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let v = build(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let layer = attention::new(2, 1, true).unwrap();
    let out = layer.forward(&exec(), &q, &k, &v).unwrap();
    assert_eq!(out.getshape().getdims(), &[2, 4]);

    let want = attend(q.getdata(), k.getdata(), v.getdata(), 2, 2, 1, 2);
    assertallclose(out.getdata(), &want, 1e-5, 1e-5);
}

#[test]
fn attention_rejects_inconsistent_key_width() {
    let q = build(vec![1, 2], vec![0.0; 2]);
    let k = build(vec![1, 3], vec![0.0; 3]);
    let v = build(vec![1, 3], vec![0.0; 3]);
    let layer = attention::new(1, 1, true).unwrap();
    assert!(matches!(
        layer.forward(&exec(), &q, &k, &v),
        Err(nnerror::shapemismatch {
            layer: "attention",
            ..
        })
    ));
}

#[test]
fn mlp_forward_matches_the_swiglu_composition() {
    let dim = 3;
    let hidden = 2;
    let gate = linear::new(
        param(
            "weight",
            vec![dim, hidden],
            vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6],
        ),
        None,
    )
    .unwrap();
    let up = linear::new(
        param(
            "weight",
            vec![dim, hidden],
            vec![0.7, 0.8, -0.9, 1.0, 0.2, -0.3],
        ),
        None,
    )
    .unwrap();
    let down = linear::new(
        param(
            "weight",
            vec![hidden, dim],
            vec![1.0, 0.0, -1.0, 0.5, 0.5, 0.5],
        ),
        None,
    )
    .unwrap();
    let layer = mlp::new(gate, up, down).unwrap();

    let input = build(vec![2, dim], vec![1.0, 2.0, 3.0, -1.0, 0.0, 1.0]);
    let out = layer.forward(&exec(), &input).unwrap();
    assert_eq!(out.getshape().getdims(), &[2, dim]);

    let gw = layer.getgate().getweight().getvalue().getdata();
    let uw = layer.getup().getweight().getvalue().getdata();
    let dw = layer.getdown().getweight().getvalue().getdata();
    let g = reference::matmul(input.getdata(), gw, 2, dim, hidden);
    let u = reference::matmul(input.getdata(), uw, 2, dim, hidden);
    let gated: Vec<f32> = g
        .iter()
        .zip(&u)
        .map(|(&gv, &uv)| reference::silu(gv) * uv)
        .collect();
    let want = reference::matmul(&gated, dw, 2, hidden, dim);
    assertallclose(out.getdata(), &want, 1e-5, 1e-5);
}

#[test]
fn mlp_collects_prefixed_parameter_names() {
    let dim = 2;
    let hidden = 4;
    let make = |name: &str, in_features: usize, out_features: usize| {
        linear::new(
            param(
                name,
                vec![in_features, out_features],
                vec![0.0; in_features * out_features],
            ),
            None,
        )
        .unwrap()
    };
    let layer = mlp::new(
        make("weight", dim, hidden),
        make("weight", dim, hidden),
        make("weight", hidden, dim),
    )
    .unwrap();
    let dict = layer.getstatedict().unwrap();
    let names: Vec<&str> = dict.names().collect();
    assert_eq!(names, vec!["down.weight", "gate.weight", "up.weight"]);
}

#[test]
fn initialization_is_reproducible_across_layers() {
    let s = shape::new(vec![4, 4]).unwrap();
    let build_layer = |sd: seed| {
        let w = parameter::new("weight", nn::initxavier(s.clone(), sd).unwrap());
        linear::new(w, None).unwrap()
    };
    let a = build_layer(seed::new(2026));
    let b = build_layer(seed::new(2026));
    let c = build_layer(seed::new(2027));
    assert_eq!(a.getstatedict().unwrap(), b.getstatedict().unwrap());
    assert_ne!(a.getstatedict().unwrap(), c.getstatedict().unwrap());
}
