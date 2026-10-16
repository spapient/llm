//! Contract tests for the `musa` GPU backend, exercised only through its
//! public API from outside the crate.
//!
//! Linux only: on every other target `musa` compiles to an empty library, so
//! the whole suite is compiled out there.
//!
//! # Environment assumptions
//!
//! The numerical suites require a Moore Threads MUSA device and driver
//! (`musaInfo` from the MUSA toolkit). Per docs/engineering.md, a test that
//! cannot run in its environment must be reported as *not executed*, never as
//! passed:
//!
//! - The device probe is `musaInfo`, shared with every GPU backend through
//!   [`testkit::probe`]. When it is absent or reports no device, the suite
//!   prints `E4 未执行` and returns without asserting the numerical result.
//!   That printed line is the record; the harness still reports the test as
//!   `ok`, which is why the next point exists.
//! - Setting `LLM_MUSA_REQUIRE_DEVICE=1` turns the same condition into a hard
//!   failure, so a missing device can never be mistaken for a passing
//!   numerical check.
//!
//! # What runs without a device
//!
//! Every check that the executor settles before it touches a device runs
//! unconditionally: identity and capability declarations, rank/shape/index
//! rejection, the element and gather budgets, and the MUSA-specific
//! head-dimension limit. These are pure host-side validation and hold on a
//! machine with no GPU.
//!
//! # MUSA coverage beyond the CUDA contract
//!
//! The MUSA backend serves all thirteen `api::compute` operators from
//! build-time-compiled kernels (docs/musa-backend-plan.md §5.3): `rope_at`
//! and `attention_cached` run dedicated kernels (the `cuda` backend pads and
//! concatenates on the host), and `scan_cached` is supported (the `cuda`
//! backend reports `unsupportedoperation`). The numerical cases below
//! therefore cover those operators against the same scalar references.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use crate::support::{build, wide};
use ::tensor::dtype;
use api::{backend, backenderror, compute, kvpair, operand, stepref, submission};
use cpu::cpuexecutor;
use lib::limits;
use musa::{MUSAMAXHEADDIM, MUSAMAXRANK, musabackend, musaexecutor};
use testkit::assertions::assertallclose;
use testkit::reference;

use crate::gpucontract;

/// Reports whether a MUSA device is reachable, failing when it is required.
///
/// Returns `true` when a kernel may be dispatched. When no device is reachable
/// it returns `false` after applying the shared environment gate, so a caller
/// must return early instead of asserting a numerical result.
///
/// The probe protocol (command, `E4` record, `LLM_MUSA_REQUIRE_DEVICE` hard
/// gate) is the shared one from [`testkit::probe`].
fn hasdevice(test: &str) -> bool {
    testkit::probe::requiredevice(test, testkit::probe::gpuprobe::musa)
}

/// Scalar reference for grouped-query causal attention.
///
/// Kept local because only this suite needs it; the shape semantics it encodes
/// are the ones `crates/cpu/src/attention.rs` documents, restated here so the
/// expectation is not derived from the code under test. Position `s` attends
/// only to `t <= s` under the causal mask; the max-subtraction order is the
/// stable one the kernels share.
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
    let qseq = seq;
    let kvseq = k.len() / kstride;
    let offset = kvseq - qseq;
    let mut out = vec![0.0_f32; qseq * stride];
    for position in 0..qseq {
        for head in 0..heads {
            let kvhead = head / group;
            let qrow =
                &q[position * stride + head * headdim..position * stride + (head + 1) * headdim];
            let tmax = (position + offset).min(kvseq);
            let mut scores = vec![0.0_f32; tmax];
            let mut maxscore = f32::NEG_INFINITY;
            for t in 0..tmax {
                let krow = &k[t * kstride + kvhead * headdim..t * kstride + (kvhead + 1) * headdim];
                let mut score = 0.0_f32;
                for i in 0..headdim {
                    score += qrow[i] * krow[i];
                }
                score *= scale;
                scores[t] = score;
                if score > maxscore {
                    maxscore = score;
                }
            }
            let mut weights = vec![0.0_f32; tmax];
            let mut denom = 0.0_f32;
            for t in 0..tmax {
                let weight = (scores[t] - maxscore).exp();
                weights[t] = weight;
                denom += weight;
            }
            for t in 0..tmax {
                let vrow = &v[t * kstride + kvhead * headdim..t * kstride + (kvhead + 1) * headdim];
                let weight = weights[t] / denom;
                for i in 0..headdim {
                    out[position * stride + head * headdim + i] += weight * vrow[i];
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Identity and capability declarations (host-side, no device required)
// ---------------------------------------------------------------------------

/// The backend declares its identity, device, and `f32` capability.
#[test]
fn backend_exposes_identity_device_and_f32_capability() {
    let b = musabackend::new(limits::new(16, 64, 8, 2));
    assert_eq!(b.getname(), "musa");
    assert!(b.getdevice().ismusa());
    assert_eq!(b.getdevice(), api::device::musa());
    let caps = b.getcapabilities();
    assert_eq!(caps.dtypes, vec![dtype::f32]);
    assert_eq!(caps.maxrank, MUSAMAXRANK);
}

/// The MUSA backend declares the same rank bound as the CUDA backend.
#[test]
fn rank_bound_matches_cuda() {
    assert_eq!(MUSAMAXRANK, cuda::CUDAMAXRANK);
}

/// The MUSA attention head-dimension bound matches the CUDA derivation.
#[test]
fn headdim_bound_matches_cuda() {
    assert_eq!(MUSAMAXHEADDIM, cuda::CUDAMAXHEADDIM);
}

// ---------------------------------------------------------------------------
// Numerical suites (require a MUSA device)
// ---------------------------------------------------------------------------

/// `matmul` matches the scalar reference on a non-square case.
#[test]
fn matmul_matches_the_scalar_reference() {
    if !hasdevice("matmul_matches_the_scalar_reference") {
        return;
    }
    let exec = musaexecutor::new(wide());
    let lhs = build(
        vec![3, 4],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );
    let rhs = build(
        vec![4, 2],
        vec![0.5, -1.0, 2.0, 1.5, -0.25, 0.75, 1.0, -2.0],
    );
    let out = exec.matmul(&lhs.getview(), &rhs.getview()).unwrap();
    let want = reference::matmul(lhs.getdata(), rhs.getdata(), 3, 4, 2);
    assertallclose(out.getdata(), &want, 1.0e-4, 1.0e-4);
}

/// `matmul` handles a leading batch dimension.
#[test]
fn matmul_handles_a_leading_batch_dimension() {
    if !hasdevice("matmul_handles_a_leading_batch_dimension") {
        return;
    }
    let exec = musaexecutor::new(wide());
    let lhs = build(
        vec![2, 2, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, 0.0, 2.0, 1.0, 1.0, 1.0],
    );
    let rhs = build(vec![2, 3, 1], vec![1.0, 0.0, -1.0, 2.0, 1.0, 3.0]);
    let out = exec.matmul(&lhs.getview(), &rhs.getview()).unwrap();
    let want = reference::matmul(lhs.getdata(), rhs.getdata(), 2, 3, 1);
    assertallclose(out.getdata(), &want, 1.0e-4, 1.0e-4);
}

/// `matmul` with a zero contraction yields the additive identity.
#[test]
fn matmul_with_a_zero_contraction_yields_the_additive_identity() {
    if !hasdevice("matmul_with_a_zero_contraction_yields_the_additive_identity") {
        return;
    }
    let exec = musaexecutor::new(wide());
    let lhs = build(vec![2, 0], vec![]);
    let rhs = build(vec![0, 3], vec![]);
    let out = exec.matmul(&lhs.getview(), &rhs.getview()).unwrap();
    assert_eq!(out.getdata(), &vec![0.0_f32; 6]);
}

/// `matmul` rejects operands beyond the element budget.
#[test]
fn matmul_rejects_operands_beyond_the_element_budget() {
    let exec = musaexecutor::new(limits::new(4, 64, 8, 2));
    let lhs = build(vec![4, 4], vec![1.0; 16]);
    let rhs = build(vec![4, 4], vec![1.0; 16]);
    assert!(matches!(
        exec.matmul(&lhs.getview(), &rhs.getview()),
        Err(backenderror::resourceexhausted)
    ));
}

/// `add` matches the elementwise sum.
#[test]
fn add_matches_the_elementwise_sum() {
    if !hasdevice("add_matches_the_elementwise_sum") {
        return;
    }
    let exec = musaexecutor::new(wide());
    let lhs = build(vec![2, 3], vec![1.0, -2.0, 0.5, 4.0, 8.0, -1.0]);
    let rhs = build(vec![2, 3], vec![0.25, 1.5, -0.5, 2.0, -8.0, 1.0]);
    let out = exec.add(&lhs.getview(), &rhs.getview()).unwrap();
    let want: Vec<f32> = lhs
        .getdata()
        .iter()
        .zip(rhs.getdata())
        .map(|(a, b)| a + b)
        .collect();
    assertallclose(out.getdata(), &want, 1.0e-6, 1.0e-6);
}

/// `mul` matches the elementwise product.
#[test]
fn mul_matches_the_elementwise_product() {
    if !hasdevice("mul_matches_the_elementwise_product") {
        return;
    }
    let exec = musaexecutor::new(wide());
    let lhs = build(vec![3], vec![1.5, -2.0, 0.0]);
    let rhs = build(vec![3], vec![2.0, 0.5, -7.0]);
    let out = exec.mul(&lhs.getview(), &rhs.getview()).unwrap();
    let want: Vec<f32> = lhs
        .getdata()
        .iter()
        .zip(rhs.getdata())
        .map(|(a, b)| a * b)
        .collect();
    assertallclose(out.getdata(), &want, 1.0e-6, 1.0e-6);
}

/// `silu` matches the scalar reference.
#[test]
fn silu_matches_the_scalar_reference() {
    if !hasdevice("silu_matches_the_scalar_reference") {
        return;
    }
    let exec = musaexecutor::new(wide());
    let input = build(vec![4], vec![-2.0, -0.5, 0.0, 3.0]);
    let out = exec.silu(&input.getview()).unwrap();
    let want = reference::map(input.getdata(), reference::silu);
    assertallclose(out.getdata(), &want, 1.0e-6, 1.0e-6);
}

/// `softmax` matches the scalar reference per row.
#[test]
fn softmax_matches_the_scalar_reference_per_row() {
    if !hasdevice("softmax_matches_the_scalar_reference_per_row") {
        return;
    }
    let exec = musaexecutor::new(wide());
    let input = build(vec![2, 3], vec![1.0, 2.0, 3.0, 0.0, -1.0, -2.0]);
    let out = exec.softmax(&input.getview()).unwrap();
    let want = reference::softmax(input.getdata());
    assertallclose(out.getdata(), &want, 1.0e-6, 1.0e-6);
}

/// `softmax` is numerically stable for extreme logits.
#[test]
fn softmax_is_numerically_stable_for_extreme_logits() {
    if !hasdevice("softmax_is_numerically_stable_for_extreme_logits") {
        return;
    }
    let exec = musaexecutor::new(wide());
    let input = build(vec![1, 3], vec![1000.0, 0.0, -1000.0]);
    let out = exec.softmax(&input.getview()).unwrap();
    assert!(out.getdata()[0] > 0.9999);
    assert!(out.getdata()[2] < 1.0e-6);
}

/// `rmsnorm` matches the scalar reference per row.
#[test]
fn rmsnorm_matches_the_scalar_reference_per_row() {
    if !hasdevice("rmsnorm_matches_the_scalar_reference_per_row") {
        return;
    }
    let exec = musaexecutor::new(wide());
    let input = build(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -3.0]);
    let weight = build(vec![4], vec![0.5, 1.0, -0.25, 2.0]);
    let out = exec
        .rmsnorm(&input.getview(), &weight.getview(), 1.0e-5)
        .unwrap();
    let want: Vec<f32> = input
        .getdata()
        .chunks(4)
        .flat_map(|row| reference::rmsnorm(row, weight.getdata(), 1.0e-5))
        .collect();
    assertallclose(out.getdata(), &want, 1.0e-5, 1.0e-5);
}

/// `embedding` gathers rows from the table by index.
#[test]
fn embedding_matches_the_scalar_reference() {
    if !hasdevice("embedding_matches_the_scalar_reference") {
        return;
    }
    let exec = musaexecutor::new(wide());
    let table = build(
        vec![4, 3],
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );
    let indices = vec![0_usize, 2, 3, 1];
    let out = exec.embedding(&table.getview(), &indices).unwrap();
    let want = reference::embedding(table.getdata(), 3, &indices);
    assertallclose(out.getdata(), &want, 1.0e-6, 1.0e-6);
}

/// `embedding` rejects an index outside the table.
#[test]
fn embedding_rejects_an_index_outside_the_table() {
    let exec = musaexecutor::new(wide());
    let table = build(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    assert!(matches!(
        exec.embedding(&table.getview(), &[0, 2]),
        Err(backenderror::indexoutofrange { .. })
    ));
}

/// `rope` matches the scalar reference per position.
#[test]
fn rope_matches_the_scalar_reference_per_position() {
    if !hasdevice("rope_matches_the_scalar_reference_per_position") {
        return;
    }
    let exec = musaexecutor::new(wide());
    let heads = 2;
    let headdim = 4;
    let rows = 3;
    let seqaxis = 3;
    let input = build(
        vec![seqaxis, heads * headdim],
        (0..rows * heads * headdim)
            .map(|i| (i as f32 * 0.31).cos())
            .collect(),
    );
    let out = exec.rope(&input.getview(), heads, 10000.0).unwrap();
    let mut want = Vec::new();
    for position in 0..rows {
        want.extend(reference::rope(
            &input.getdata()[position * heads * headdim..(position + 1) * heads * headdim],
            position,
            heads,
            headdim,
            10000.0,
        ));
    }
    assertallclose(out.getdata(), &want, 1.0e-4, 1.0e-4);
}

/// `rope_at` rotates a single token at its absolute position.
#[test]
fn rope_at_matches_the_scalar_reference_at_the_absolute_position() {
    if !hasdevice("rope_at_matches_the_scalar_reference_at_the_absolute_position") {
        return;
    }
    let exec = musaexecutor::new(wide());
    let heads = 2;
    let headdim = 4;
    let input = build(
        vec![1, heads * headdim],
        vec![0.3, -0.7, 1.2, 0.5, -0.9, 0.1, 0.8, -0.2],
    );
    let position = 5;
    let out = exec
        .rope_at(&input.getview(), position, heads, 10000.0)
        .unwrap();
    let want = reference::rope(input.getdata(), position, heads, headdim, 10000.0);
    assertallclose(out.getdata(), &want, 1.0e-4, 1.0e-4);
}

/// `attention` matches the scalar reference with a causal mask.
#[test]
fn attention_matches_the_scalar_reference() {
    if !hasdevice("attention_matches_the_scalar_reference") {
        return;
    }
    let exec = musaexecutor::new(wide());
    let heads = 2;
    let kvheads = 1;
    let headdim = 4;
    let seq = 3;
    let q = build(
        vec![seq, heads * headdim],
        (0..seq * heads * headdim)
            .map(|i| (i as f32 * 0.17).sin())
            .collect(),
    );
    let k = build(
        vec![seq, kvheads * headdim],
        (0..seq * kvheads * headdim)
            .map(|i| (i as f32 * 0.23).cos())
            .collect(),
    );
    let v = build(
        vec![seq, kvheads * headdim],
        (0..seq * kvheads * headdim)
            .map(|i| (i as f32 * 0.41).sin())
            .collect(),
    );
    let out = exec
        .attention(
            &q.getview(),
            &k.getview(),
            &v.getview(),
            heads,
            kvheads,
            true,
        )
        .unwrap();
    let want = attend(
        q.getdata(),
        k.getdata(),
        v.getdata(),
        seq,
        heads,
        kvheads,
        headdim,
    );
    assertallclose(out.getdata(), &want, 1.0e-4, 1.0e-4);
}

/// `attention_cached` equals the scalar attention over cached+new.
#[test]
fn attention_cached_matches_attention_over_the_concatenated_history() {
    if !hasdevice("attention_cached_matches_attention_over_the_concatenated_history") {
        return;
    }
    let exec = musaexecutor::new(wide());
    let heads = 2;
    let kvheads = 1;
    let headdim = 4;
    let pos = 2;
    let q = build(
        vec![1, heads * headdim],
        (0..heads * headdim)
            .map(|i| (i as f32 * 0.11).sin())
            .collect(),
    );
    let keycached = build(
        vec![pos, kvheads * headdim],
        (0..pos * kvheads * headdim)
            .map(|i| (i as f32 * 0.19).cos())
            .collect(),
    );
    let keynew = build(
        vec![1, kvheads * headdim],
        (0..kvheads * headdim)
            .map(|i| (i as f32 * 0.53).sin())
            .collect(),
    );
    let valuecached = build(
        vec![pos, kvheads * headdim],
        (0..pos * kvheads * headdim)
            .map(|i| (i as f32 * 0.29).cos())
            .collect(),
    );
    let valuenew = build(
        vec![1, kvheads * headdim],
        (0..kvheads * headdim)
            .map(|i| (i as f32 * 0.61).sin())
            .collect(),
    );
    let out = exec
        .attention_cached(
            &q.getview(),
            kvpair {
                cached: keycached.getview(),
                new: keynew.getview(),
            },
            kvpair {
                cached: valuecached.getview(),
                new: valuenew.getview(),
            },
            heads,
            kvheads,
            true,
        )
        .unwrap();
    let mut k = keycached.getdata().to_vec();
    k.extend_from_slice(keynew.getdata());
    let mut v = valuecached.getdata().to_vec();
    v.extend_from_slice(valuenew.getdata());
    let want = attend(q.getdata(), &k, &v, 1, heads, kvheads, headdim);
    assertallclose(out.getdata(), &want, 1.0e-4, 1.0e-4);
}

/// `scan` matches the scalar reference.
#[test]
fn scan_matches_the_scalar_reference() {
    if !hasdevice("scan_matches_the_scalar_reference") {
        return;
    }
    let exec = musaexecutor::new(wide());
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

/// `scan_cached` continues the serial scan for the decode step.
#[test]
fn scan_cached_continues_the_serial_scan() {
    if !hasdevice("scan_cached_continues_the_serial_scan") {
        return;
    }
    let exec = musaexecutor::new(wide());
    let d = 2;
    let statewidth = 3;
    let n = 4;
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
    // Serial continuation: h' = A h + B x, y = C h' + D x for the new token.
    let (_, fullstate) = reference::scan(&input, &init, &a, &b, &c, &skip, n, d, statewidth);
    let x = input[n * d - d];
    let mut wantstate = fullstate.clone();
    let mut wantout = vec![0.0_f32; d];
    for i in 0..d {
        let mut acc = 0.0_f32;
        for j in 0..statewidth {
            let idx = i * statewidth + j;
            wantstate[idx] = a[idx] * wantstate[idx] + b[idx] * x;
            acc += c[idx] * wantstate[idx];
        }
        wantout[i] = acc + skip[i] * x;
    }
    let input_t = build(vec![1, d], vec![x]);
    let state_t = build(vec![d, statewidth], fullstate);
    let a_t = build(vec![d, statewidth], a.clone());
    let b_t = build(vec![d, statewidth], b.clone());
    let c_t = build(vec![d, statewidth], c.clone());
    let skip_t = build(vec![d], skip);
    let (out, newstate) = exec
        .scan_cached(
            &input_t.getview(),
            &state_t.getview(),
            &a_t.getview(),
            &b_t.getview(),
            &c_t.getview(),
            &skip_t.getview(),
        )
        .unwrap();
    assertallclose(out.getdata(), &wantout, 1e-5, 1e-5);
    assertallclose(newstate.getdata(), &wantstate, 1e-5, 1e-5);
}

/// `scan` rejects a spectral violation.
#[test]
fn scan_rejects_a_spectral_violation() {
    let exec = musaexecutor::new(wide());
    let input = build(vec![2, 1], vec![1.0, 2.0]);
    let init = build(vec![1, 1], vec![0.0]);
    let a = build(vec![1, 1], vec![1.5]); // |a| > 1
    let b = build(vec![1, 1], vec![1.0]);
    let c = build(vec![1, 1], vec![1.0]);
    let skip = build(vec![1], vec![0.0]);
    assert!(matches!(
        exec.scan(
            &input.getview(),
            &init.getview(),
            &a.getview(),
            &b.getview(),
            &c.getview(),
            &skip.getview(),
        ),
        Err(backenderror::invalidparameter { .. })
    ));
}

// ---------------------------------------------------------------------------
// Host-side rejection (no device required)
// ---------------------------------------------------------------------------

/// `attention` rejects a head dimension above the MUSA bound as exhaustion.
#[test]
fn attention_rejects_a_head_dimension_above_the_musa_bound() {
    let exec = musaexecutor::new(wide());
    let headdim = MUSAMAXHEADDIM + 2;
    let q = build(vec![1, 2 * headdim], vec![0.0; 2 * headdim]);
    let k = build(vec![1, 2 * headdim], vec![0.0; 2 * headdim]);
    let v = build(vec![1, 2 * headdim], vec![0.0; 2 * headdim]);
    assert!(matches!(
        exec.attention(&q.getview(), &k.getview(), &v.getview(), 2, 2, true),
        Err(backenderror::resourceexhausted)
    ));
}

/// Elementwise operators reject mismatched shapes.
#[test]
fn elementwise_rejects_mismatched_shapes() {
    let exec = musaexecutor::new(wide());
    let a = build(vec![2, 2], vec![1.0; 4]);
    let b = build(vec![4], vec![1.0; 4]);
    assert!(exec.add(&a.getview(), &b.getview()).is_err());
    assert!(exec.mul(&a.getview(), &b.getview()).is_err());
}

// ---------------------------------------------------------------------------
// Submission contract
// ---------------------------------------------------------------------------

/// A recorded matmul chain applies and downloads the requested output.
#[test]
fn submission_runs_a_recorded_chain() {
    if !hasdevice("submission_runs_a_recorded_chain") {
        return;
    }
    let exec = musaexecutor::new(wide());
    let mut sub = exec.newsubmission();
    let a = build(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = build(vec![3, 2], vec![0.5, -1.0, 2.0, 1.5, -0.25, 0.75]);
    let r1 = sub
        .recordmatmul(operand::input(a.getview()), operand::input(b.getview()))
        .unwrap();
    let r2 = sub.recordsilu(operand::result(r1)).unwrap();
    let out = sub.apply(&[r2]).unwrap();
    let want1 = reference::matmul(a.getdata(), b.getdata(), 2, 3, 2);
    let want = reference::map(&want1, reference::silu);
    assertallclose(out[0].getdata(), &want, 1.0e-4, 1.0e-4);
}

/// The submission rejects scan as unsupported, mirroring the CUDA contract.
#[test]
fn submission_rejects_scan_as_unsupported() {
    let exec = musaexecutor::new(wide());
    let mut sub = exec.newsubmission();
    let input = build(vec![2, 1], vec![1.0, 2.0]);
    let init = build(vec![1, 1], vec![0.0]);
    let a = build(vec![1, 1], vec![0.5]);
    let b = build(vec![1, 1], vec![1.0]);
    let c = build(vec![1, 1], vec![1.0]);
    let skip = build(vec![1], vec![0.0]);
    assert!(matches!(
        sub.recordscan(
            operand::input(input.getview()),
            operand::input(init.getview()),
            operand::input(a.getview()),
            operand::input(b.getview()),
            operand::input(c.getview()),
            operand::input(skip.getview()),
        ),
        Err(backenderror::unsupportedoperation { .. })
    ));
}

// ---------------------------------------------------------------------------
// Shared device-free contract vectors
// ---------------------------------------------------------------------------

/// Every device-free contract vector settles identically on MUSA and the host.
#[test]
fn device_free_contract_vectors_conform() {
    gpucontract::assertconforms("musa", musaexecutor::new);
}

/// The contract is device-independent: the same operand must produce the same
/// public error on the host and on the MUSA backend.
#[test]
fn device_free_contract_vectors_match_the_cpu_backend() {
    assert_eq!(
        gpucontract::run(&musaexecutor::new),
        gpucontract::run(&cpuexecutor::new),
        "the musa and cpu backends disagree on a device-free contract vector"
    );
}
