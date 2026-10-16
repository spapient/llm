//! Contract tests for the `metal` GPU backend, exercised only through its
//! public API from outside the crate.
//!
//! macOS only: on every other target `metal` compiles to an empty library, so
//! the whole suite is compiled out there.
//!
//! # Environment assumptions
//!
//! Every dispatch case needs access to the system GPU. Where the process
//! cannot reach the Metal device the calls surface as
//! `backenderror::devicenotavailable` (distinct from a compile or execution
//! failure); this was observed reproducibly under the agent sandbox, and
//! running the same test binary outside it passed. Run this suite with GPU
//! access available. The shape, budget, and error-path cases are settled on
//! the host before any dispatch, so those hold with or without a device.
#![cfg(target_os = "macos")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use crate::support::{build, wide};
use ::tensor::{dtype, tensor};
use api::{backend, backenderror, compute, operand, stepref, submission};
use cpu::cpuexecutor;
use lib::limits;
use metal::{METALMAXHEADDIM, METALMAXRANK, metalbackend, metalexecutor};
use testkit::assertions::assertallclose;
use testkit::reference;

use crate::gpucontract;

/// Scalar reference for grouped-query scaled dot-product attention.
///
/// Kept local because only this suite needs it; the shape semantics it encodes
/// are the ones `crates/cpu/src/attention.rs` documents, restated here so the
/// expectation is not derived from the code under test.
fn attend(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    kvheads: usize,
    headdim: usize,
    causal: bool,
) -> Vec<f32> {
    let stride = heads * headdim;
    let kstride = kvheads * headdim;
    let qseq = q.len() / stride;
    let kvseq = k.len() / kstride;
    let offset = kvseq - qseq;
    let group = heads / kvheads;
    let scale = 1.0 / (headdim as f32).sqrt();
    let mut out = vec![0.0_f32; qseq * stride];
    for head in 0..heads {
        let kvhead = head / group;
        for s in 0..qseq {
            let qrow = &q[s * stride + head * headdim..][..headdim];
            let limit = if causal && offset + s + 1 < kvseq {
                offset + s + 1
            } else {
                kvseq
            };
            let mut logits = vec![f32::NEG_INFINITY; limit];
            for (t, slot) in logits.iter_mut().enumerate() {
                let krow = &k[t * kstride + kvhead * headdim..][..headdim];
                *slot = qrow.iter().zip(krow).map(|(&a, &b)| a * b).sum::<f32>() * scale;
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
fn backend_exposes_identity_device_and_f32_capability() {
    let b = metalbackend::new(limits::new(16, 64, 8, 2));
    assert_eq!(b.getname(), "metal");
    assert!(b.getdevice().ismetal());
    assert_eq!(b.getdevice(), api::device::metal());
    let caps = b.getcapabilities();
    assert_eq!(caps.getmaxrank(), METALMAXRANK);
    assert_eq!(caps.getdtypes(), &[dtype::f32]);
    assert!(caps.supports(dtype::f32));
    assert!(!caps.supports(dtype::f16));
    assert!(!caps.supports(dtype::i32));
}

// ---------------------------------------------------------------------------
// matmul
// ---------------------------------------------------------------------------

#[test]
fn matmul_matches_the_scalar_reference() {
    let exec = metalexecutor::new(wide());
    let a: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let b: Vec<f32> = (1..=12).map(|x| 0.5 * x as f32).collect();
    let lhs = build(vec![3, 4], a.clone());
    let rhs = build(vec![4, 3], b.clone());

    let out = exec.matmul(&lhs.getview(), &rhs.getview()).unwrap();

    assert_eq!(out.getshape().getdims(), &[3, 3]);
    assert_eq!(out.getdtype(), dtype::f32);
    let want = reference::matmul(&a, &b, 3, 4, 3);
    assertallclose(out.getdata(), &want, 1.0e-4, 1.0e-4);
}

#[test]
fn matmul_handles_a_leading_batch_dimension() {
    let exec = metalexecutor::new(wide());
    let a: Vec<f32> = (1..=24).map(|x| x as f32).collect();
    let b: Vec<f32> = (1..=24).map(|x| 0.25 * x as f32).collect();
    let lhs = build(vec![2, 3, 4], a.clone());
    let rhs = build(vec![2, 4, 3], b.clone());

    let out = exec.matmul(&lhs.getview(), &rhs.getview()).unwrap();

    assert_eq!(out.getshape().getdims(), &[2, 3, 3]);
    let mut want = reference::matmul(&a[0..12], &b[0..12], 3, 4, 3);
    want.extend_from_slice(&reference::matmul(&a[12..24], &b[12..24], 3, 4, 3));
    assertallclose(out.getdata(), &want, 1.0e-4, 1.0e-4);
}

#[test]
fn matmul_rejects_incompatible_shapes() {
    let exec = metalexecutor::new(wide());
    let lhs = build(vec![2, 3], vec![0.0; 6]);
    let rhs = build(vec![4, 2], vec![0.0; 8]);
    assert_eq!(
        exec.matmul(&lhs.getview(), &rhs.getview()).map(|_| ()),
        Err(backenderror::shapemismatch { op: "matmul" })
    );
}

#[test]
fn matmul_rejects_operands_beyond_the_element_budget() {
    let exec = metalexecutor::new(limits::new(1, 1, 1, 1));
    let lhs = build(vec![2, 3], vec![1.0; 6]);
    let rhs = build(vec![3, 2], vec![1.0; 6]);
    assert_eq!(
        exec.matmul(&lhs.getview(), &rhs.getview()).map(|_| ()),
        Err(backenderror::resourceexhausted)
    );
}

// ---------------------------------------------------------------------------
// Elementwise: add, mul, silu
// ---------------------------------------------------------------------------

#[test]
fn add_matches_the_elementwise_sum() {
    let exec = metalexecutor::new(wide());
    let a = [1.0_f32, -2.0, 3.5, 4.0, 0.0, -1.0];
    let b = [0.5_f32, 2.0, -1.5, -4.0, 7.0, 1.0];
    let lhs = build(vec![2, 3], a.to_vec());
    let rhs = build(vec![2, 3], b.to_vec());

    let out = exec.add(&lhs.getview(), &rhs.getview()).unwrap();

    assert_eq!(out.getshape().getdims(), &[2, 3]);
    let want: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
    assertallclose(out.getdata(), &want, 0.0, 0.0);
}

#[test]
fn mul_matches_the_elementwise_product() {
    let exec = metalexecutor::new(wide());
    let a = [1.0_f32, -2.0, 3.5, 4.0, 0.5, -1.0];
    let b = [0.5_f32, 2.0, -1.5, -4.0, 7.0, 1.0];
    let lhs = build(vec![2, 3], a.to_vec());
    let rhs = build(vec![2, 3], b.to_vec());

    let out = exec.mul(&lhs.getview(), &rhs.getview()).unwrap();

    assert_eq!(out.getshape().getdims(), &[2, 3]);
    let want: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).collect();
    assertallclose(out.getdata(), &want, 0.0, 0.0);
}

#[test]
fn silu_matches_the_scalar_reference() {
    let exec = metalexecutor::new(wide());
    let input = [-3.0_f32, -1.0, 0.0, 0.5, 2.0, 8.0];
    let x = build(vec![2, 3], input.to_vec());

    let out = exec.silu(&x.getview()).unwrap();

    assert_eq!(out.getshape().getdims(), &[2, 3]);
    let want = reference::map(&input, reference::silu);
    assertallclose(out.getdata(), &want, 1.0e-6, 1.0e-6);
}

#[test]
fn elementwise_rejects_mismatched_shapes() {
    let exec = metalexecutor::new(wide());
    let lhs = build(vec![2, 3], vec![0.0; 6]);
    let rhs = build(vec![3, 2], vec![0.0; 6]);
    assert_eq!(
        exec.add(&lhs.getview(), &rhs.getview()).map(|_| ()),
        Err(backenderror::shapemismatch { op: "add" })
    );
    assert_eq!(
        exec.mul(&lhs.getview(), &rhs.getview()).map(|_| ()),
        Err(backenderror::shapemismatch { op: "mul" })
    );
}

#[test]
fn elementwise_rejects_operands_beyond_the_element_budget() {
    let exec = metalexecutor::new(limits::new(1, 1, 1, 1));
    let lhs = build(vec![4], vec![1.0; 4]);
    let rhs = build(vec![4], vec![1.0; 4]);
    assert_eq!(
        exec.add(&lhs.getview(), &rhs.getview()).map(|_| ()),
        Err(backenderror::resourceexhausted)
    );
    assert_eq!(
        exec.mul(&lhs.getview(), &rhs.getview()).map(|_| ()),
        Err(backenderror::resourceexhausted)
    );
}

#[test]
fn empty_elementwise_inputs_produce_empty_output() {
    // The launch helpers return before opening the device when the operand is
    // empty, so this is device-free and pins the empty-shape behaviour.
    let exec = metalexecutor::new(wide());
    let lhs = build(vec![0, 3], vec![]);
    let rhs = build(vec![0, 3], vec![]);
    let sum = exec.add(&lhs.getview(), &rhs.getview()).unwrap();
    assert_eq!(sum.getshape().getdims(), &[0, 3]);
    assert!(sum.getdata().is_empty());
    let scaled = exec.silu(&lhs.getview()).unwrap();
    assert_eq!(scaled.getshape().getdims(), &[0, 3]);
}

// ---------------------------------------------------------------------------
// softmax
// ---------------------------------------------------------------------------

#[test]
fn softmax_matches_the_scalar_reference_per_row() {
    let exec = metalexecutor::new(wide());
    let input = [1.0_f32, 2.0, 3.0, -1.0, 0.0, 1.0];
    let x = build(vec![2, 3], input.to_vec());

    let out = exec.softmax(&x.getview()).unwrap();

    assert_eq!(out.getshape().getdims(), &[2, 3]);
    let mut want = reference::softmax(&input[0..3]);
    want.extend(reference::softmax(&input[3..6]));
    assertallclose(out.getdata(), &want, 1.0e-6, 1.0e-6);
}

#[test]
fn softmax_is_numerically_stable_for_extreme_logits() {
    let exec = metalexecutor::new(wide());
    let x = build(vec![1, 3], vec![1000.0, 1000.0, 1000.0]);

    let out = exec.softmax(&x.getview()).unwrap();

    // A naive softmax would overflow here; the stable form returns thirds.
    assertallclose(out.getdata(), &[1.0 / 3.0; 3], 1.0e-6, 1.0e-6);
}

#[test]
fn softmax_rejects_an_empty_last_axis() {
    let exec = metalexecutor::new(wide());
    let x = build(vec![2, 0], vec![]);
    assert_eq!(
        exec.softmax(&x.getview()).map(|_| ()),
        Err(backenderror::unsupportedoperation { op: "softmax" })
    );
}

#[test]
fn softmax_over_zero_rows_yields_an_empty_output() {
    // `rows == 0` is settled on the host before the device is opened.
    let exec = metalexecutor::new(wide());
    let x = build(vec![0, 3], vec![]);
    let out = exec.softmax(&x.getview()).unwrap();
    assert_eq!(out.getshape().getdims(), &[0, 3]);
    assert!(out.getdata().is_empty());
}

// ---------------------------------------------------------------------------
// rmsnorm
// ---------------------------------------------------------------------------

#[test]
fn rmsnorm_matches_the_scalar_reference_per_row() {
    let exec = metalexecutor::new(wide());
    let input = [3.0_f32, 4.0, 0.0, 5.0];
    let weight = [1.0_f32, 2.0];
    let x = build(vec![2, 2], input.to_vec());
    let w = build(vec![2], weight.to_vec());

    let out = exec.rmsnorm(&x.getview(), &w.getview(), 0.0).unwrap();

    assert_eq!(out.getshape().getdims(), &[2, 2]);
    let mut want = reference::rmsnorm(&input[0..2], &weight, 0.0);
    want.extend(reference::rmsnorm(&input[2..4], &weight, 0.0));
    assertallclose(out.getdata(), &want, 1.0e-5, 1.0e-5);
}

#[test]
fn rmsnorm_rejects_an_empty_last_axis() {
    let exec = metalexecutor::new(wide());
    let x = build(vec![2, 0], vec![]);
    let w = build(vec![0], vec![]);
    assert_eq!(
        exec.rmsnorm(&x.getview(), &w.getview(), 1.0e-5).map(|_| ()),
        Err(backenderror::unsupportedoperation { op: "rmsnorm" })
    );
}

#[test]
fn rmsnorm_rejects_a_weight_of_the_wrong_length() {
    let exec = metalexecutor::new(wide());
    let x = build(vec![2, 3], vec![0.0; 6]);
    let w = build(vec![2], vec![1.0; 2]);
    assert_eq!(
        exec.rmsnorm(&x.getview(), &w.getview(), 1.0e-5).map(|_| ()),
        Err(backenderror::shapemismatch { op: "rmsnorm" })
    );
}

#[test]
fn rmsnorm_accepts_a_one_row_weight() {
    // The scalar contract compares element counts, not shapes, so `[width]`
    // and `[1, width]` weights are both legal.
    let exec = metalexecutor::new(wide());
    let x = build(vec![1, 2], vec![3.0, 4.0]);
    let row = build(vec![2], vec![1.0, 1.0]);
    let matrix = build(vec![1, 2], vec![1.0, 1.0]);
    let from_row = exec.rmsnorm(&x.getview(), &row.getview(), 0.0).unwrap();
    let from_matrix = exec.rmsnorm(&x.getview(), &matrix.getview(), 0.0).unwrap();
    assertallclose(from_row.getdata(), from_matrix.getdata(), 0.0, 0.0);
}

// ---------------------------------------------------------------------------
// rope
// ---------------------------------------------------------------------------

#[test]
fn rope_matches_the_scalar_reference_per_position() {
    let heads = 2;
    let headdim = 4;
    let width = heads * headdim;
    let seq = 3;
    let input: Vec<f32> = (0..seq * width).map(|i| i as f32 * 0.1 - 1.0).collect();
    let x = build(vec![seq, width], input.clone());
    let exec = metalexecutor::new(wide());

    let out = exec.rope(&x.getview(), heads, 10_000.0).unwrap();

    assert_eq!(out.getshape().getdims(), &[seq, width]);
    let mut want = Vec::new();
    for s in 0..seq {
        want.extend(reference::rope(
            &input[s * width..(s + 1) * width],
            s,
            heads,
            headdim,
            10_000.0,
        ));
    }
    assertallclose(out.getdata(), &want, 1.0e-5, 1.0e-5);
}

#[test]
fn rope_rejects_a_zero_head_count_or_an_odd_head_dimension() {
    let exec = metalexecutor::new(wide());
    let x = build(vec![2, 4], vec![0.0; 8]);
    assert_eq!(
        exec.rope(&x.getview(), 0, 10_000.0).map(|_| ()),
        Err(backenderror::shapemismatch { op: "rope" })
    );
    // One head over a width of three gives an odd head dimension.
    let odd = build(vec![2, 3], vec![0.0; 6]);
    assert_eq!(
        exec.rope(&odd.getview(), 1, 10_000.0).map(|_| ()),
        Err(backenderror::shapemismatch { op: "rope" })
    );
    // A rank-one operand has no position axis.
    let flat = build(vec![4], vec![0.0; 4]);
    assert_eq!(
        exec.rope(&flat.getview(), 1, 10_000.0).map(|_| ()),
        Err(backenderror::shapemismatch { op: "rope" })
    );
}

// ---------------------------------------------------------------------------
// embedding
// ---------------------------------------------------------------------------

#[test]
fn embedding_matches_the_scalar_reference() {
    let exec = metalexecutor::new(wide());
    let table = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
    let t = build(vec![3, 2], table.to_vec());

    let out = exec.embedding(&t.getview(), &[2, 0]).unwrap();

    assert_eq!(out.getshape().getdims(), &[2, 2]);
    let want = reference::embedding(&table, 2, &[2, 0]);
    assertallclose(out.getdata(), &want, 0.0, 0.0);
}

#[test]
fn embedding_rejects_an_out_of_range_index() {
    let exec = metalexecutor::new(wide());
    let t = build(vec![3, 2], vec![0.0; 6]);
    assert_eq!(
        exec.embedding(&t.getview(), &[3]).map(|_| ()),
        Err(backenderror::indexoutofrange { index: 3, len: 3 })
    );
}

#[test]
fn embedding_rejects_a_gather_beyond_the_element_budget() {
    // The output size comes from the index list, not from any operand shape, so
    // it has its own budget check. Repeated indices let the gather exceed the
    // budget while the table itself still fits.
    let exec = metalexecutor::new(limits::new(8, 1 << 20, 8, 2));
    let t = build(vec![2, 2], vec![1.0; 4]);
    assert_eq!(
        exec.embedding(&t.getview(), &[0; 8]).map(|_| ()),
        Err(backenderror::resourceexhausted)
    );
}

#[test]
fn embedding_with_no_indices_yields_an_empty_output() {
    let exec = metalexecutor::new(wide());
    let t = build(vec![2, 3], vec![1.0; 6]);
    let out = exec.embedding(&t.getview(), &[]).unwrap();
    assert_eq!(out.getshape().getdims(), &[0, 3]);
    assert!(out.getdata().is_empty());
}

// ---------------------------------------------------------------------------
// attention
// ---------------------------------------------------------------------------

#[test]
fn scan_matches_the_scalar_reference() {
    let exec = metalexecutor::new(wide());
    let n = 8;
    let d = 3;
    let statewidth = 4;
    let input: Vec<f32> = (0..n * d).map(|i| (i as f32 * 0.37).sin()).collect();
    let init: Vec<f32> = (0..d * statewidth)
        .map(|i| (i as f32 * 0.13).cos())
        .collect();
    let a: Vec<f32> = (0..d * statewidth)
        .map(|i| 0.5 + 0.5 * (i as f32 * 0.71).cos().abs())
        .collect();
    let b: Vec<f32> = (0..d * statewidth)
        .map(|i| (i as f32 * 0.29).sin())
        .collect();
    let c: Vec<f32> = (0..d * statewidth)
        .map(|i| (i as f32 * 0.19).cos())
        .collect();
    let skip: Vec<f32> = (0..d).map(|i| (i as f32 * 0.11).sin()).collect();

    let input_t = build(vec![n, d], input.clone());
    let init_t = build(vec![d, statewidth], init.clone());
    let a_t = build(vec![d, statewidth], a.clone());
    let b_t = build(vec![d, statewidth], b.clone());
    let c_t = build(vec![d, statewidth], c.clone());
    let skip_t = build(vec![d], skip.clone());

    let (out, state) = exec
        .scan(
            &input_t.getview(),
            &init_t.getview(),
            &a_t.getview(),
            &b_t.getview(),
            &c_t.getview(),
            &skip_t.getview(),
        )
        .unwrap();

    let (want_out, want_state) =
        reference::scan(&input, &init, &a, &b, &c, &skip, n, d, statewidth);
    assertallclose(out.getdata(), &want_out, 1e-5, 1e-5);
    assertallclose(state.getdata(), &want_state, 1e-5, 1e-5);
}

#[test]
fn scan_rejects_a_spectral_violation() {
    let exec = metalexecutor::new(wide());
    let input = build(vec![2, 2], vec![1.0; 4]);
    let init = build(vec![2, 1], vec![0.0, 0.0]);
    let a = build(vec![2, 1], vec![0.5, -1.5]);
    let b = build(vec![2, 1], vec![1.0, 1.0]);
    let c = build(vec![2, 1], vec![1.0, 1.0]);
    let skip = build(vec![2], vec![0.0, 0.0]);
    assert_eq!(
        exec.scan(
            &input.getview(),
            &init.getview(),
            &a.getview(),
            &b.getview(),
            &c.getview(),
            &skip.getview(),
        )
        .map(|_| ()),
        Err(backenderror::invalidparameter { op: "scan" })
    );
}

#[test]
fn attention_matches_a_scalar_reference() {
    // seq = 2, heads = 2, kvheads = 1, headdim = 2.
    let q = build(vec![2, 4], vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 1.0, -1.0]);
    let k = build(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let v = build(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let exec = metalexecutor::new(wide());

    let out = exec
        .attention(&q.getview(), &k.getview(), &v.getview(), 2, 1, true)
        .unwrap();

    assert_eq!(out.getshape().getdims(), &[2, 4]);
    let want = attend(q.getdata(), k.getdata(), v.getdata(), 2, 1, 2, true);
    assertallclose(out.getdata(), &want, 1.0e-5, 1.0e-5);
}

#[test]
fn attention_non_causal_matches_a_scalar_reference() {
    let q = build(
        vec![3, 4],
        vec![
            0.5, -1.0, 1.5, 0.25, -0.5, 0.75, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0,
        ],
    );
    let k = build(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5]);
    let v = build(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5]);
    let exec = metalexecutor::new(wide());

    let out = exec
        .attention(&q.getview(), &k.getview(), &v.getview(), 2, 1, false)
        .unwrap();

    assert_eq!(out.getshape().getdims(), &[3, 4]);
    let want = attend(q.getdata(), k.getdata(), v.getdata(), 2, 1, 2, false);
    assertallclose(out.getdata(), &want, 1.0e-5, 1.0e-5);
}

#[test]
fn attention_single_position_returns_the_value() {
    let q = build(vec![1, 2], vec![1.0, 0.0]);
    let k = build(vec![1, 2], vec![1.0, 0.0]);
    let v = build(vec![1, 2], vec![3.0, 4.0]);
    let exec = metalexecutor::new(wide());

    let out = exec
        .attention(&q.getview(), &k.getview(), &v.getview(), 1, 1, true)
        .unwrap();

    assertallclose(out.getdata(), &[3.0, 4.0], 1.0e-6, 1.0e-6);
}

#[test]
fn attention_accepts_a_shorter_query_than_the_keys() {
    let q = build(vec![1, 4], vec![0.5, -1.0, 1.5, 0.25]);
    let k = build(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5]);
    let v = build(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5]);
    let exec = metalexecutor::new(wide());

    let out = exec
        .attention(&q.getview(), &k.getview(), &v.getview(), 2, 1, true)
        .unwrap();

    assert_eq!(out.getshape().getdims(), &[1, 4]);
    let want = attend(q.getdata(), k.getdata(), v.getdata(), 2, 1, 2, true);
    assertallclose(out.getdata(), &want, 1.0e-5, 1.0e-5);
}

#[test]
fn attention_rejects_inconsistent_operands() {
    let exec = metalexecutor::new(wide());
    let q = build(vec![2, 4], vec![0.0; 8]);
    let k = build(vec![2, 2], vec![0.0; 4]);
    let v = build(vec![2, 2], vec![0.0; 4]);

    // A rank-one query has no sequence axis.
    let flat = build(vec![4], vec![0.0; 4]);
    assert_eq!(
        exec.attention(&flat.getview(), &k.getview(), &v.getview(), 1, 1, true)
            .map(|_| ()),
        Err(backenderror::shapemismatch { op: "attention" })
    );
    // Key and value widths must agree.
    let wide_k = build(vec![2, 3], vec![0.0; 6]);
    assert_eq!(
        exec.attention(&q.getview(), &wide_k.getview(), &v.getview(), 1, 1, true)
            .map(|_| ()),
        Err(backenderror::shapemismatch { op: "attention" })
    );
    // The head split must be exact: four query columns over three heads.
    assert_eq!(
        exec.attention(&q.getview(), &k.getview(), &v.getview(), 3, 1, true)
            .map(|_| ()),
        Err(backenderror::shapemismatch { op: "attention" })
    );
    // The key head width is `kvheads * headdim`, so two heads over two columns
    // cannot serve a two-column query split into two heads of width two.
    assert_eq!(
        exec.attention(&q.getview(), &k.getview(), &v.getview(), 2, 2, true)
            .map(|_| ()),
        Err(backenderror::shapemismatch { op: "attention" })
    );
    // A key/value head count that does not divide the query head count.
    assert_eq!(
        exec.attention(&q.getview(), &k.getview(), &v.getview(), 2, 0, true)
            .map(|_| ()),
        Err(backenderror::shapemismatch { op: "attention" })
    );
}

#[test]
fn attention_rejects_a_head_dimension_beyond_the_device_limit() {
    // The kernel keeps one accumulator slot per head dimension in threadgroup
    // memory, so this is a device limit rather than a contract limit and is
    // reported as resource exhaustion. It is settled on the host, before any
    // launch, so it holds without a GPU.
    let exec = metalexecutor::new(wide());
    let width = METALMAXHEADDIM + 1;
    let q = build(vec![1, width], vec![0.0; width]);
    let k = build(vec![1, width], vec![0.0; width]);
    let v = build(vec![1, width], vec![0.0; width]);
    assert_eq!(
        exec.attention(&q.getview(), &k.getview(), &v.getview(), 1, 1, true)
            .map(|_| ()),
        Err(backenderror::resourceexhausted)
    );
}

// ---------------------------------------------------------------------------
// Shared device-free contract vectors
// ---------------------------------------------------------------------------

#[test]
fn device_free_contract_vectors_conform() {
    // Every case is settled on the host before a device call, so this runs
    // without a GPU and pins the classification the `api` contract requires.
    gpucontract::assertconforms("metal", metalexecutor::new);
}

#[test]
fn device_free_contract_vectors_match_the_cpu_backend() {
    // The contract is device-independent: the same operand must produce the
    // same public error on the host and on the GPU.
    assert_eq!(
        gpucontract::run(&metalexecutor::new),
        gpucontract::run(&cpuexecutor::new),
        "the metal and cpu backends disagree on a device-free contract vector"
    );
}

// ---------------------------------------------------------------------------
// Numerical coverage beyond the first fixture size
// ---------------------------------------------------------------------------

#[test]
fn matmul_with_distinct_extents_matches_the_reference() {
    // Extents that are neither square nor equal to the threadgroup edge, so the
    // dispatch covers several threadgroups in both directions with a partial
    // edge in each.
    let (m, k, n) = (17, 19, 13);
    let a: Vec<f32> = (1..=m * k).map(|x| x as f32 * 0.01).collect();
    let b: Vec<f32> = (1..=k * n).map(|x| x as f32 * -0.02).collect();
    let lhs = build(vec![m, k], a.clone());
    let rhs = build(vec![k, n], b.clone());
    let exec = metalexecutor::new(wide());

    let out = exec.matmul(&lhs.getview(), &rhs.getview()).unwrap();

    assert_eq!(out.getshape().getdims(), &[m, n]);
    assertallclose(
        out.getdata(),
        &reference::matmul(&a, &b, m, k, n),
        1.0e-4,
        1.0e-4,
    );
}

#[test]
fn matmul_of_a_single_element_matches_the_reference() {
    let lhs = build(vec![1, 1], vec![0.5]);
    let rhs = build(vec![1, 1], vec![2.0]);
    let exec = metalexecutor::new(wide());

    let out = exec.matmul(&lhs.getview(), &rhs.getview()).unwrap();

    assertallclose(out.getdata(), &[1.0], 0.0, 0.0);
}

#[test]
fn rmsnorm_handles_a_row_wider_than_one_threadgroup() {
    // The reduction threadgroup is 256 wide; a row of 1000 exercises the
    // strided pre-pass and a width that is not a power of two.
    let width = 1000;
    let rows = 3;
    let input: Vec<f32> = (0..rows * width)
        .map(|i| ((i % 37) as f32 - 18.0) * 0.25)
        .collect();
    let weight: Vec<f32> = (0..width).map(|i| 1.0 + (i % 5) as f32 * 0.1).collect();
    let x = build(vec![rows, width], input.clone());
    let w = build(vec![width], weight.clone());
    let exec = metalexecutor::new(wide());

    let out = exec.rmsnorm(&x.getview(), &w.getview(), 1.0e-5).unwrap();

    assert_eq!(out.getshape().getdims(), &[rows, width]);
    let mut want = Vec::new();
    for r in 0..rows {
        want.extend(reference::rmsnorm(
            &input[r * width..(r + 1) * width],
            &weight,
            1.0e-5,
        ));
    }
    assertallclose(out.getdata(), &want, 1.0e-5, 1.0e-5);
}

#[test]
fn softmax_handles_a_row_wider_than_one_threadgroup() {
    let width = 1000;
    let rows = 3;
    let input: Vec<f32> = (0..rows * width)
        .map(|i| ((i % 53) as f32 - 26.0) * 0.3)
        .collect();
    let x = build(vec![rows, width], input.clone());
    let exec = metalexecutor::new(wide());

    let out = exec.softmax(&x.getview()).unwrap();

    assert_eq!(out.getshape().getdims(), &[rows, width]);
    let mut want = Vec::new();
    for r in 0..rows {
        want.extend(reference::softmax(&input[r * width..(r + 1) * width]));
    }
    assertallclose(out.getdata(), &want, 1.0e-6, 1.0e-6);
    for r in 0..rows {
        let sum: f32 = out.getdata()[r * width..(r + 1) * width].iter().sum();
        assert!((sum - 1.0).abs() <= 1.0e-5, "row {r} sums to {sum}");
    }
}

#[test]
fn softmax_of_a_negative_infinity_row_is_not_a_number() {
    // Pinned because the shift is undefined: subtracting a row maximum of
    // negative infinity from itself is NaN, not a finite underflow, so the row
    // propagates NaN. The kernel's `total > 0` guard cannot rescue it, because
    // that comparison is false for NaN as well. `cpu` behaves identically.
    let x = build(vec![1, 3], vec![f32::NEG_INFINITY; 3]);
    let exec = metalexecutor::new(wide());

    let out = exec.softmax(&x.getview()).unwrap();

    assert!(
        out.getdata().iter().all(|value| value.is_nan()),
        "expected NaN propagation, got {:?}",
        out.getdata()
    );
}

#[test]
fn attention_with_equal_head_counts_matches_the_reference() {
    // `heads == kvheads` is the group size one case, where every query head
    // reads its own key/value head.
    let (seq, heads, kvheads, headdim) = (3, 2, 2, 4);
    let width = heads * headdim;
    let q: Vec<f32> = (0..seq * width).map(|i| (i as f32 * 0.1) - 1.2).collect();
    let k: Vec<f32> = (0..seq * width).map(|i| 0.9 - i as f32 * 0.07).collect();
    let v: Vec<f32> = (0..seq * width).map(|i| (i as f32 * 0.3) - 2.0).collect();
    let query = build(vec![seq, width], q.clone());
    let key = build(vec![seq, width], k.clone());
    let value = build(vec![seq, width], v.clone());
    let exec = metalexecutor::new(wide());

    let out = exec
        .attention(
            &query.getview(),
            &key.getview(),
            &value.getview(),
            heads,
            kvheads,
            true,
        )
        .unwrap();

    assert_eq!(out.getshape().getdims(), &[seq, width]);
    let want = attend(&q, &k, &v, heads, kvheads, headdim, true);
    assertallclose(out.getdata(), &want, 1.0e-5, 1.0e-5);
}

#[test]
fn rope_of_a_single_position_is_the_identity_rotation() {
    // Position zero has angle zero, so the rotation must return the input
    // unchanged to within the cosine's rounding.
    let heads = 2;
    let headdim = 4;
    let width = heads * headdim;
    let input: Vec<f32> = (0..width).map(|i| i as f32 * 0.5 - 2.0).collect();
    let x = build(vec![1, width], input.clone());
    let exec = metalexecutor::new(wide());

    let out = exec.rope(&x.getview(), heads, 10_000.0).unwrap();

    assertallclose(out.getdata(), &input, 1.0e-6, 1.0e-6);
}

#[test]
fn embedding_accepts_the_first_and_last_row() {
    let table = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
    let t = build(vec![3, 2], table.to_vec());
    let exec = metalexecutor::new(wide());

    let out = exec.embedding(&t.getview(), &[0, 2, 2]).unwrap();

    assert_eq!(out.getshape().getdims(), &[3, 2]);
    let want = reference::embedding(&table, 2, &[0, 2, 2]);
    assertallclose(out.getdata(), &want, 0.0, 0.0);
}

// ---------------------------------------------------------------------------
// Runtime compilation cache and concurrency
// ---------------------------------------------------------------------------

/// Runs one call of every operator, so the executor must hold one compiled
/// pipeline per distinct `(source, entry)` pair the crate ships.
fn runeverykernel(exec: &metalexecutor) {
    let a = build(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let b = build(vec![2, 2], vec![5.0, 6.0, 7.0, 8.0]);
    let weight = build(vec![2], vec![1.0, 1.0]);
    let table = build(vec![3, 2], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let q = build(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let k = build(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let v = build(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);

    exec.matmul(&a.getview(), &b.getview()).unwrap();
    exec.add(&a.getview(), &b.getview()).unwrap();
    exec.mul(&a.getview(), &b.getview()).unwrap();
    exec.silu(&a.getview()).unwrap();
    exec.softmax(&a.getview()).unwrap();
    exec.rmsnorm(&a.getview(), &weight.getview(), 1.0e-5)
        .unwrap();
    exec.rope(&a.getview(), 1, 10_000.0).unwrap();
    exec.embedding(&table.getview(), &[2, 0]).unwrap();
    exec.attention(&q.getview(), &k.getview(), &v.getview(), 1, 1, true)
        .unwrap();
}

#[test]
fn every_kernel_is_compiled_once_and_then_reused() {
    // The nine operators map onto nine distinct `(source, entry)` pairs, so the
    // count after one call of each is the exact number of kernels in the crate.
    // Repeating the pass must not compile anything: that is the machine-checkable
    // form of "the run-time compilation is cached", independent of timing.
    let exec = metalexecutor::new(wide());

    runeverykernel(&exec);
    assert_eq!(
        exec.getcachedpipelinecount(),
        9,
        "one pass over the operator set must compile exactly one pipeline per kernel"
    );

    runeverykernel(&exec);
    assert_eq!(
        exec.getcachedpipelinecount(),
        9,
        "a second pass recompiled a kernel instead of reusing its pipeline"
    );
}

#[test]
fn a_clone_shares_the_compiled_pipelines() {
    let exec = metalexecutor::new(wide());
    let a = build(vec![2, 2], vec![1.0; 4]);

    exec.silu(&a.getview()).unwrap();
    let clone = exec.clone();
    assert_eq!(clone.getcachedpipelinecount(), 1);

    clone.add(&a.getview(), &a.getview()).unwrap();

    assert_eq!(
        exec.getcachedpipelinecount(),
        2,
        "a clone must share the cache rather than compile into its own"
    );
}

#[test]
fn the_executor_is_shareable_across_threads() {
    // The concurrency claim in docs/design.md §21.6, asserted rather than
    // assumed: the executor is `Send + Sync`, and concurrent calls on one value
    // return identical results without compiling the kernel more than once.
    fn assertsendandsync<T: Send + Sync>() {}
    assertsendandsync::<metalexecutor>();

    let exec = metalexecutor::new(wide());
    let x = build(
        vec![64, 64],
        (0..64 * 64).map(|i| (i % 17) as f32 * 0.25).collect(),
    );
    let want = exec.softmax(&x.getview()).unwrap();

    let got: Vec<tensor<f32>> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| scope.spawn(|| exec.softmax(&x.getview()).unwrap()))
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect()
    });

    assert_eq!(got.len(), 4);
    for one in &got {
        assertallclose(one.getdata(), want.getdata(), 0.0, 0.0);
    }
    assert_eq!(
        exec.getcachedpipelinecount(),
        1,
        "concurrent first calls must not each compile their own pipeline"
    );
}

#[test]
fn submission_reproduces_a_compute_chain_bit_for_bit() {
    let exec = metalexecutor::new(wide());
    let a = build(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = build(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let c = build(vec![2, 2], vec![0.5, -0.5, 1.5, -1.5]);

    let mut sub = exec.newsubmission().unwrap();
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
fn submission_rejects_an_unrecorded_result_reference() {
    let exec = metalexecutor::new(wide());
    let mut sub = exec.newsubmission().unwrap();
    let a = build(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let bogus = stepref::fromindex(99);

    let err = sub
        .recordadd(operand::result(bogus), operand::input(a.getview()))
        .unwrap_err();

    assert_eq!(err, backenderror::indexoutofrange { index: 99, len: 0 });
}
