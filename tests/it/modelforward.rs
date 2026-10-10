//! Deterministic forward test for the `models` Llama implementation.
//!
//! A tiny one-layer model is built with deterministic random weights and run
//! through the CPU backend. The result is cross-checked against an independent
//! scalar reference written with `testkit::reference`, so the test does not
//! merely restate the implementation. Determinism is pinned by rebuilding the
//! model from the same seed and comparing logits.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use cpu::cpuexecutor;
use lib::{limits, seed};
use models::{causallm, llamaconfig, llamamodel, llamaweights};
use testkit::{assertions::assertallclose, reference};

const VOCAB: usize = 5;
const DIM: usize = 4;
const NHEADS: usize = 1;
const KVHEADS: usize = 1;
const HEADDIM: usize = 4;
const HIDDEN: usize = 8;
const RMSEPS: f32 = 1.0e-5;
const ROPEBASE: f32 = 10_000.0;

fn config() -> llamaconfig {
    llamaconfig::new(VOCAB, DIM, 1, NHEADS, KVHEADS, HIDDEN, RMSEPS, ROPEBASE, 8).unwrap()
}

fn model(seedvalue: u64) -> llamamodel {
    let c = config();
    let w = llamaweights::random(&c, seed::new(seedvalue)).unwrap();
    llamamodel::new(c, w).unwrap()
}

fn engine() -> cpuexecutor {
    cpuexecutor::new(limits::new(1 << 20, 1 << 24, 1 << 12, 1 << 8))
}

/// RMS-normalizes every row of a flat `[rows, width]` slice.
fn rmsrows(data: &[f32], weight: &[f32], width: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len());
    for row in data.chunks(width) {
        out.extend(reference::rmsnorm(row, weight, RMSEPS));
    }
    out
}

/// Applies rotary to every row of a flat `[rows, width]` slice.
fn roperows(data: &[f32], width: usize, heads: usize, headdim: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(data.len());
    for (s, row) in data.chunks(width).enumerate() {
        out.extend(reference::rope(row, s, heads, headdim, ROPEBASE));
    }
    out
}

/// Scalar grouped-query causal attention.
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

/// An independent scalar reference for the whole tiny forward pass.
fn referencelogits(m: &llamamodel, tokens: &[usize]) -> Vec<f32> {
    let seq = tokens.len();
    let w = m.getweights();
    let layer = &w.getlayers()[0];

    // Embedding.
    let embedded = reference::embedding(
        w.getembedding().getweight().getvalue().getdata(),
        DIM,
        tokens,
    );

    // Attention branch.
    let attn_norm = rmsrows(
        &embedded,
        layer.getattnnorm().getweight().getvalue().getdata(),
        DIM,
    );
    let q = reference::matmul(
        &attn_norm,
        layer.getwq().getweight().getvalue().getdata(),
        seq,
        DIM,
        DIM,
    );
    let k = reference::matmul(
        &attn_norm,
        layer.getwk().getweight().getvalue().getdata(),
        seq,
        DIM,
        KVHEADS * HEADDIM,
    );
    let v = reference::matmul(
        &attn_norm,
        layer.getwv().getweight().getvalue().getdata(),
        seq,
        DIM,
        KVHEADS * HEADDIM,
    );
    let q = roperows(&q, DIM, NHEADS, HEADDIM);
    let k = roperows(&k, KVHEADS * HEADDIM, KVHEADS, HEADDIM);
    let attended = attend(&q, &k, &v, seq, NHEADS, KVHEADS, HEADDIM);
    let projected = reference::matmul(
        &attended,
        layer.getwo().getweight().getvalue().getdata(),
        seq,
        DIM,
        DIM,
    );
    let residual: Vec<f32> = embedded
        .iter()
        .zip(&projected)
        .map(|(&a, &b)| a + b)
        .collect();

    // Feed-forward branch.
    let ffn_norm = rmsrows(
        &residual,
        layer.getffnnorm().getweight().getvalue().getdata(),
        DIM,
    );
    let gate = reference::matmul(
        &ffn_norm,
        layer.getmlp().getgate().getweight().getvalue().getdata(),
        seq,
        DIM,
        HIDDEN,
    );
    let up = reference::matmul(
        &ffn_norm,
        layer.getmlp().getup().getweight().getvalue().getdata(),
        seq,
        DIM,
        HIDDEN,
    );
    let gated: Vec<f32> = gate
        .iter()
        .zip(&up)
        .map(|(&g, &u)| reference::silu(g) * u)
        .collect();
    let down = reference::matmul(
        &gated,
        layer.getmlp().getdown().getweight().getvalue().getdata(),
        seq,
        HIDDEN,
        DIM,
    );
    let hidden: Vec<f32> = residual.iter().zip(&down).map(|(&a, &b)| a + b).collect();

    // Final normalization and output projection.
    let normalized = rmsrows(&hidden, w.getnorm().getweight().getvalue().getdata(), DIM);
    reference::matmul(
        &normalized,
        w.getoutput().getweight().getvalue().getdata(),
        seq,
        DIM,
        VOCAB,
    )
}

#[test]
fn tiny_forward_matches_an_independent_scalar_reference() {
    let m = model(99);
    let tokens = [0usize, 1];
    let logits = m.forward(&engine(), &tokens).unwrap();
    assert_eq!(logits.getshape().getdims(), &[2, VOCAB]);

    let want = referencelogits(&m, &tokens);
    assertallclose(logits.getdata(), &want, 1e-4, 1e-4);
}

#[test]
fn forward_is_deterministic_for_a_fixed_seed() {
    let tokens = [3usize, 1, 4];
    let a = model(7).forward(&engine(), &tokens).unwrap();
    let b = model(7).forward(&engine(), &tokens).unwrap();
    let c = model(8).forward(&engine(), &tokens).unwrap();
    assert_eq!(a.getdata(), b.getdata());
    assert_ne!(a.getdata(), c.getdata());
    assert_eq!(a.getshape().getdims(), &[3, VOCAB]);
}

#[test]
fn every_position_produces_finite_logits() {
    let tokens = [0usize, 4, 2, 3];
    let logits = model(3).forward(&engine(), &tokens).unwrap();
    assert_eq!(logits.getshape().getdims(), &[4, VOCAB]);
    assert!(logits.getdata().iter().all(|v| v.is_finite()));
}

#[test]
fn an_out_of_range_token_is_rejected() {
    let m = model(1);
    assert!(m.forward(&engine(), &[VOCAB + 10]).is_err());
}

#[test]
fn the_first_token_uses_only_its_own_value_for_causal_attention() {
    // With a one-token input there is nothing to attend to but position zero.
    let m = model(5);
    let single = m.forward(&engine(), &[2]).unwrap();
    let multi = m.forward(&engine(), &[2, 0]).unwrap();
    // The first position of the longer sequence sees the same single position.
    assertallclose(&multi.getdata()[..VOCAB], single.getdata(), 1e-4, 1e-4);
}
