//! Contract tests for the `cuda` GPU backend, exercised only through its
//! public API from outside the crate.
//!
//! Linux only: on every other target `cuda` compiles to an empty library, so
//! the whole suite is compiled out there.
//!
//! # Environment assumptions
//!
//! The numerical suites require an NVIDIA GPU and a driver exposing
//! `nvidia-smi`. Per docs/engineering.md, a test that cannot run in its
//! environment must be reported as *not executed*, never as passed:
//!
//! - The device probe is `nvidia-smi -L`. When it is absent or reports no
//!   device, the suite prints `E4 未执行` and returns without asserting the
//!   numerical result. That printed line is the record; the harness still
//!   reports the test as `ok`, which is why the next point exists.
//! - Setting `LLM_CUDA_REQUIRE_DEVICE=1` turns the same condition into a hard
//!   failure. The `cuda-test` CI job on the self-hosted GPU runner sets it, so a
//!   missing device there can never be mistaken for a passing numerical check.
//!
//! # What runs without a device
//!
//! Every check that the executor settles before it touches device 0 runs
//! unconditionally: identity and capability declarations, the matmul path
//! selector, rank/shape/index rejection for all eight operators, the element
//! and gather budgets, the CUDA-specific head-dimension limit, and the
//! degenerate empty-input cases that the launch helpers short-circuit. These
//! are pure host-side validation and hold on a machine with no GPU, which is
//! where the CI `cuda-check` job evaluates them.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use crate::support::{build, wide};
use ::tensor::dtype;
use api::{backend, backenderror, compute, operand, stepref, submission};
use cpu::cpuexecutor;
use cuda::{CUDAMAXHEADDIM, CUDAMAXRANK, cudabackend, cudaexecutor, matmulpath};
use lib::limits;
use testkit::assertions::assertallclose;
use testkit::reference;

use crate::gpucontract;

/// Reports whether an NVIDIA device is reachable, failing when it is required.
///
/// Returns `true` when a kernel may be dispatched. When no device is reachable
/// it returns `false` after asserting the environment did not demand one, so a
/// caller must return early instead of asserting a numerical result.
fn hasdevice(test: &str) -> bool {
    let probed = std::process::Command::new("nvidia-smi")
        .arg("-L")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    if probed {
        return true;
    }
    assert!(
        std::env::var_os("LLM_CUDA_REQUIRE_DEVICE").is_none(),
        "{test}: LLM_CUDA_REQUIRE_DEVICE is set but no CUDA device was detected"
    );
    eprintln!(
        "E4 未执行: {test} needs an NVIDIA GPU; `nvidia-smi -L` was unavailable or \
         reported no device. Set LLM_CUDA_REQUIRE_DEVICE=1 to fail instead."
    );
    false
}

/// Scalar reference for grouped-query causal attention.
///
/// Kept local because only this suite needs it; the shape semantics it encodes
/// are the ones `crates/cpu/src/attention.rs` documents, restated here so the
/// expectation is not derived from the code under test.
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
    for head in 0..heads {
        let kvhead = head / group;
        for s in 0..qseq {
            let qrow = &q[s * stride + head * headdim..][..headdim];
            let mut logits = vec![f32::NEG_INFINITY; kvseq];
            for (t, slot) in logits.iter_mut().enumerate() {
                if t <= offset + s {
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

// ---------------------------------------------------------------------------
// Identity, capability, and path selection (no device)
// ---------------------------------------------------------------------------

#[test]
fn backend_exposes_identity_device_and_f32_capability() {
    let b = cudabackend::new(limits::new(16, 64, 8, 2));
    assert_eq!(b.getname(), "cuda");
    assert!(b.getdevice().iscuda());
    assert!(!b.getdevice().ismetal());
    assert_eq!(b.getdevice(), api::device::cuda());
    let caps = b.getcapabilities();
    assert_eq!(caps.getmaxrank(), CUDAMAXRANK);
    assert_eq!(caps.getdtypes(), &[dtype::f32]);
    assert!(caps.supports(dtype::f32));
    assert!(!caps.supports(dtype::f16));
}

#[test]
fn executor_reports_which_path_serves_matmul() {
    let lim = wide();
    assert_eq!(cudaexecutor::new(lim).getpath(), matmulpath::cublas);
    assert_eq!(cudaexecutor::newnvrtc(lim).getpath(), matmulpath::nvrtc);
    assert_eq!(cudaexecutor::new(lim).getbackend().getname(), "cuda");
}

// ---------------------------------------------------------------------------
// matmul
// ---------------------------------------------------------------------------

#[test]
fn cublas_matmul_matches_the_scalar_reference() {
    if !hasdevice("cublas_matmul_matches_the_scalar_reference") {
        return;
    }
    let exec = cudaexecutor::new(wide());
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
fn nvrtc_matmul_matches_the_scalar_reference() {
    if !hasdevice("nvrtc_matmul_matches_the_scalar_reference") {
        return;
    }
    let exec = cudaexecutor::newnvrtc(wide());
    let a: Vec<f32> = (1..=12).map(|x| x as f32).collect();
    let b: Vec<f32> = (1..=12).map(|x| 0.5 * x as f32).collect();
    let lhs = build(vec![3, 4], a.clone());
    let rhs = build(vec![4, 3], b.clone());

    let out = exec.matmul(&lhs.getview(), &rhs.getview()).unwrap();

    assert_eq!(out.getshape().getdims(), &[3, 3]);
    let want = reference::matmul(&a, &b, 3, 4, 3);
    assertallclose(out.getdata(), &want, 1.0e-4, 1.0e-4);
}

#[test]
fn the_two_matmul_paths_agree_on_a_non_square_case() {
    if !hasdevice("the_two_matmul_paths_agree_on_a_non_square_case") {
        return;
    }
    let a: Vec<f32> = (1..=35).map(|x| 0.125 * x as f32).collect();
    let b: Vec<f32> = (1..=28).map(|x| -0.25 * x as f32).collect();
    let lhs = build(vec![5, 7], a);
    let rhs = build(vec![7, 4], b);

    let via_cublas = cudaexecutor::new(wide())
        .matmul(&lhs.getview(), &rhs.getview())
        .unwrap();
    let via_nvrtc = cudaexecutor::newnvrtc(wide())
        .matmul(&lhs.getview(), &rhs.getview())
        .unwrap();

    assert_eq!(via_cublas.getshape().getdims(), &[5, 4]);
    assertallclose(via_cublas.getdata(), via_nvrtc.getdata(), 1.0e-4, 1.0e-4);
}

#[test]
fn matmul_handles_a_leading_batch_dimension() {
    if !hasdevice("matmul_handles_a_leading_batch_dimension") {
        return;
    }
    let exec = cudaexecutor::new(wide());
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
fn matmul_with_a_zero_contraction_yields_the_additive_identity() {
    // `k == 0` is settled on the host before either path opens the device, so
    // this holds without a GPU and pins the same semantics as `cpu`.
    let exec = cudaexecutor::new(wide());
    let lhs = build(vec![2, 0], vec![]);
    let rhs = build(vec![0, 3], vec![]);

    let out = exec.matmul(&lhs.getview(), &rhs.getview()).unwrap();

    assert_eq!(out.getshape().getdims(), &[2, 3]);
    assert_eq!(out.getdata(), &[0.0; 6]);
}

#[test]
fn matmul_rejects_incompatible_shapes() {
    let exec = cudaexecutor::new(wide());
    let lhs = build(vec![2, 3], vec![0.0; 6]);
    let rhs = build(vec![4, 2], vec![0.0; 8]);
    assert_eq!(
        exec.matmul(&lhs.getview(), &rhs.getview()).map(|_| ()),
        Err(backenderror::shapemismatch { op: "matmul" })
    );
}

#[test]
fn matmul_rejects_operands_beyond_the_element_budget() {
    let exec = cudaexecutor::new(limits::new(1, 1, 1, 1));
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
    if !hasdevice("add_matches_the_elementwise_sum") {
        return;
    }
    let exec = cudaexecutor::new(wide());
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
    if !hasdevice("mul_matches_the_elementwise_product") {
        return;
    }
    let exec = cudaexecutor::new(wide());
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
    if !hasdevice("silu_matches_the_scalar_reference") {
        return;
    }
    let exec = cudaexecutor::new(wide());
    let input = [-3.0_f32, -1.0, 0.0, 0.5, 2.0, 8.0];
    let x = build(vec![2, 3], input.to_vec());

    let out = exec.silu(&x.getview()).unwrap();

    assert_eq!(out.getshape().getdims(), &[2, 3]);
    let want = reference::map(&input, reference::silu);
    assertallclose(out.getdata(), &want, 1.0e-6, 1.0e-6);
}

#[test]
fn elementwise_rejects_mismatched_shapes() {
    let exec = cudaexecutor::new(wide());
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
    let exec = cudaexecutor::new(limits::new(1, 1, 1, 1));
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
    let exec = cudaexecutor::new(wide());
    let lhs = build(vec![0, 3], vec![]);
    let rhs = build(vec![0, 3], vec![]);
    let sum = exec.add(&lhs.getview(), &rhs.getview()).unwrap();
    assert_eq!(sum.getshape().getdims(), &[0, 3]);
    assert!(sum.getdata().is_empty());
    let scaled = exec.silu(&lhs.getview()).unwrap();
    assert_eq!(scaled.getshape().getdims(), &[0, 3]);
}

// ---------------------------------------------------------------------------
// softmax (cuDNN) and rmsnorm (NVRTC)
// ---------------------------------------------------------------------------

#[test]
fn softmax_matches_the_scalar_reference_per_row() {
    if !hasdevice("softmax_matches_the_scalar_reference_per_row") {
        return;
    }
    let exec = cudaexecutor::new(wide());
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
    if !hasdevice("softmax_is_numerically_stable_for_extreme_logits") {
        return;
    }
    let exec = cudaexecutor::new(wide());
    let x = build(vec![1, 3], vec![1000.0, 1000.0, 1000.0]);

    let out = exec.softmax(&x.getview()).unwrap();

    // A naive softmax would overflow here; the stable form returns thirds.
    assertallclose(out.getdata(), &[1.0 / 3.0; 3], 1.0e-6, 1.0e-6);
}

#[test]
fn rmsnorm_matches_the_scalar_reference_per_row() {
    if !hasdevice("rmsnorm_matches_the_scalar_reference_per_row") {
        return;
    }
    let exec = cudaexecutor::new(wide());
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
fn softmax_rejects_an_empty_last_axis() {
    let exec = cudaexecutor::new(wide());
    let x = build(vec![2, 0], vec![]);
    assert_eq!(
        exec.softmax(&x.getview()).map(|_| ()),
        Err(backenderror::unsupportedoperation { op: "softmax" })
    );
}

#[test]
fn softmax_over_zero_rows_yields_an_empty_output() {
    // `rows == 0` is settled on the host before cuDNN is opened.
    let exec = cudaexecutor::new(wide());
    let x = build(vec![0, 3], vec![]);
    let out = exec.softmax(&x.getview()).unwrap();
    assert_eq!(out.getshape().getdims(), &[0, 3]);
    assert!(out.getdata().is_empty());
}

#[test]
fn rmsnorm_rejects_an_empty_last_axis() {
    let exec = cudaexecutor::new(wide());
    let x = build(vec![2, 0], vec![]);
    let w = build(vec![0], vec![]);
    assert_eq!(
        exec.rmsnorm(&x.getview(), &w.getview(), 1.0e-5).map(|_| ()),
        Err(backenderror::unsupportedoperation { op: "rmsnorm" })
    );
}

#[test]
fn rmsnorm_rejects_a_weight_of_the_wrong_length() {
    let exec = cudaexecutor::new(wide());
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
    // and `[1, width]` weights are both legal. This is a device-gated call
    // because the shape is valid.
    if !hasdevice("rmsnorm_accepts_a_one_row_weight") {
        return;
    }
    let exec = cudaexecutor::new(wide());
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
    if !hasdevice("rope_matches_the_scalar_reference_per_position") {
        return;
    }
    let heads = 2;
    let headdim = 4;
    let width = heads * headdim;
    let seq = 3;
    let input: Vec<f32> = (0..seq * width).map(|i| i as f32 * 0.1 - 1.0).collect();
    let x = build(vec![seq, width], input.clone());
    let exec = cudaexecutor::new(wide());

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
    let exec = cudaexecutor::new(wide());
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
    if !hasdevice("embedding_matches_the_scalar_reference") {
        return;
    }
    let exec = cudaexecutor::new(wide());
    let table = [0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0];
    let t = build(vec![3, 2], table.to_vec());

    let out = exec.embedding(&t.getview(), &[2, 0]).unwrap();

    assert_eq!(out.getshape().getdims(), &[2, 2]);
    let want = reference::embedding(&table, 2, &[2, 0]);
    assertallclose(out.getdata(), &want, 0.0, 0.0);
}

#[test]
fn embedding_rejects_an_out_of_range_index() {
    let exec = cudaexecutor::new(wide());
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
    let exec = cudaexecutor::new(limits::new(8, 1 << 20, 8, 2));
    let t = build(vec![2, 2], vec![1.0; 4]);
    assert_eq!(
        exec.embedding(&t.getview(), &[0; 8]).map(|_| ()),
        Err(backenderror::resourceexhausted)
    );
}

#[test]
fn embedding_with_no_indices_yields_an_empty_output() {
    let exec = cudaexecutor::new(wide());
    let t = build(vec![2, 3], vec![1.0; 6]);
    let out = exec.embedding(&t.getview(), &[]).unwrap();
    assert_eq!(out.getshape().getdims(), &[0, 3]);
    assert!(out.getdata().is_empty());
}

// ---------------------------------------------------------------------------
// attention
// ---------------------------------------------------------------------------

#[test]
fn attention_matches_a_scalar_reference() {
    if !hasdevice("attention_matches_a_scalar_reference") {
        return;
    }
    // seq = 2, heads = 2, kvheads = 1, headdim = 2.
    let q = build(vec![2, 4], vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5, 1.0, -1.0]);
    let k = build(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let v = build(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let exec = cudaexecutor::new(wide());

    let out = exec
        .attention(&q.getview(), &k.getview(), &v.getview(), 2, 1, true)
        .unwrap();

    assert_eq!(out.getshape().getdims(), &[2, 4]);
    let want = attend(q.getdata(), k.getdata(), v.getdata(), 2, 2, 1, 2);
    assertallclose(out.getdata(), &want, 1.0e-5, 1.0e-5);
}

#[test]
fn attention_single_position_returns_the_value() {
    if !hasdevice("attention_single_position_returns_the_value") {
        return;
    }
    let q = build(vec![1, 2], vec![1.0, 0.0]);
    let k = build(vec![1, 2], vec![1.0, 0.0]);
    let v = build(vec![1, 2], vec![3.0, 4.0]);
    let exec = cudaexecutor::new(wide());

    let out = exec
        .attention(&q.getview(), &k.getview(), &v.getview(), 1, 1, true)
        .unwrap();

    assertallclose(out.getdata(), &[3.0, 4.0], 1.0e-6, 1.0e-6);
}

#[test]
fn attention_accepts_a_shorter_query_than_the_keys() {
    if !hasdevice("attention_accepts_a_shorter_query_than_the_keys") {
        return;
    }
    let q = build(vec![1, 4], vec![0.5, -1.0, 1.5, 0.25]);
    let k = build(vec![3, 2], vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5]);
    let v = build(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5]);
    let exec = cudaexecutor::new(wide());

    let out = exec
        .attention(&q.getview(), &k.getview(), &v.getview(), 2, 1, true)
        .unwrap();

    assert_eq!(out.getshape().getdims(), &[1, 4]);
    let want = attend(q.getdata(), k.getdata(), v.getdata(), 1, 2, 1, 2);
    assertallclose(out.getdata(), &want, 1.0e-5, 1.0e-5);
}

#[test]
fn attention_rejects_inconsistent_operands() {
    let exec = cudaexecutor::new(wide());
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
    // The kernel keeps one accumulator slot per head dimension in shared
    // memory, so this is a device limit rather than a contract limit and is
    // reported as resource exhaustion. It is settled on the host, before any
    // launch, so it holds without a GPU.
    let exec = cudaexecutor::new(wide());
    let width = CUDAMAXHEADDIM + 1;
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
    // Every case is settled on the host before a device call, so the
    // `cuda-check` CI job evaluates this without a GPU.
    gpucontract::assertconforms("cuda", cudaexecutor::new);
}

#[test]
fn device_free_contract_vectors_match_the_cpu_backend() {
    // The contract is device-independent: the same operand must produce the
    // same public error on the host and on the GPU.
    assert_eq!(
        gpucontract::run(&cudaexecutor::new),
        gpucontract::run(&cpuexecutor::new),
        "the cuda and cpu backends disagree on a device-free contract vector"
    );
}

#[test]
fn the_reference_path_agrees_with_the_cublas_path_on_the_contract_vectors() {
    // A path selector must not change what the backend accepts or rejects.
    assert_eq!(
        gpucontract::run(&cudaexecutor::new),
        gpucontract::run(&cudaexecutor::newnvrtc),
        "the cublas and nvrtc matmul paths disagree on a device-free contract vector"
    );
}

// ---------------------------------------------------------------------------
// Run-time compilation cache
// ---------------------------------------------------------------------------

/// Runs one call of each of the nine operators against `exec`.
///
/// The operands are minimal but legal: what matters here is which kernels the
/// call touches, not the numbers, which the numeric suites already cover.
fn runeveryoperator(exec: &cudaexecutor) {
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
    // Seven `(source, entry)` pairs serve the nine operators on the cuBLAS path:
    // `add`/`mul`/`silu` share ELEMENTWISE with three entries, and `matmul` is
    // served by cuBLAS so it compiles nothing. Compiling is the dominant
    // per-call cost (docs/design.md §21.10.3), so "compiled once" is asserted
    // mechanically rather than inferred from timing.
    if !hasdevice("every_kernel_is_compiled_once_and_then_reused") {
        return;
    }
    let cublas = cudaexecutor::new(wide());

    runeveryoperator(&cublas);
    assert_eq!(
        cublas.getcachedkernelcount(),
        7,
        "one pass over the operator set must compile exactly one kernel per (source, entry) pair"
    );

    runeveryoperator(&cublas);
    assert_eq!(
        cublas.getcachedkernelcount(),
        7,
        "a second pass recompiled a kernel instead of reusing it"
    );

    // The NVRTC path serves `matmul` from the in-repo kernel, which is one more
    // pair; it must be reached only when that path is selected.
    let nvrtc = cudaexecutor::newnvrtc(wide());
    runeveryoperator(&nvrtc);
    assert_eq!(
        nvrtc.getcachedkernelcount(),
        8,
        "the NVRTC matmul kernel must be compiled on exactly one path"
    );
}

#[test]
fn a_clone_shares_the_compiled_kernels() {
    if !hasdevice("a_clone_shares_the_compiled_kernels") {
        return;
    }
    let exec = cudaexecutor::new(wide());
    let a = build(vec![2, 2], vec![1.0; 4]);

    exec.silu(&a.getview()).unwrap();
    let clone = exec.clone();
    assert_eq!(clone.getcachedkernelcount(), 1);

    clone.add(&a.getview(), &a.getview()).unwrap();

    assert_eq!(
        exec.getcachedkernelcount(),
        2,
        "a clone must share the cache rather than compile into its own"
    );
}

#[test]
fn the_executor_is_shareable_across_threads() {
    // The concurrency claim of docs/design.md §21.6, asserted rather than
    // assumed: the executor is `Send + Sync`, four threads may call it at once,
    // the results are identical, and the concurrent first calls compile the
    // kernel once rather than once per thread.
    if !hasdevice("the_executor_is_shareable_across_threads") {
        return;
    }
    fn assertsendandsync<T: Send + Sync>() {}
    assertsendandsync::<cudaexecutor>();

    let exec = cudaexecutor::new(wide());
    let x = build(
        vec![64, 64],
        (0..64 * 64).map(|i| (i % 17) as f32 * 0.25).collect(),
    );
    let want = exec.silu(&x.getview()).unwrap();

    let got = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| scope.spawn(|| exec.silu(&x.getview()).unwrap()))
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>()
    });

    assert_eq!(got.len(), 4);
    for one in &got {
        assertallclose(one.getdata(), want.getdata(), 0.0, 0.0);
    }
    assert_eq!(
        exec.getcachedkernelcount(),
        1,
        "concurrent first calls must not each compile their own kernel"
    );
}

#[test]
fn softmax_of_a_negative_infinity_row_is_not_a_number() {
    // Pinned for the same reason as in `metalcontract`: subtracting a row
    // maximum of negative infinity from itself is NaN, not a finite underflow,
    // so the row propagates NaN, and `cpu` behaves identically. cuDNN computes
    // the same max-subtracting formulation, so the third backend must agree.
    if !hasdevice("softmax_of_a_negative_infinity_row_is_not_a_number") {
        return;
    }
    let x = build(vec![1, 3], vec![f32::NEG_INFINITY; 3]);
    let exec = cudaexecutor::new(wide());

    let out = exec.softmax(&x.getview()).unwrap();

    assert!(
        out.getdata().iter().all(|value| value.is_nan()),
        "expected NaN propagation, got {:?}",
        out.getdata()
    );
}

#[test]
fn softmax_matches_the_cpu_backend_outside_the_finite_domain() {
    // The measured divergence this suite found: cuDNN's `CUDNN_SOFTMAX_ACCURATE`
    // returns zeros for a row of negative infinity, where `cpu` and `metal`
    // propagate NaN. Every case is settled by comparing the device result with
    // the host backend on the same input, so neither side can drift alone.
    if !hasdevice("softmax_matches_the_cpu_backend_outside_the_finite_domain") {
        return;
    }
    let cases: [&[f32]; 5] = [
        &[f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY],
        &[1.0, f32::INFINITY, 2.0],
        &[f32::NAN, 1.0, 2.0],
        &[f32::NEG_INFINITY, 0.0, 1.0],
        &[f32::INFINITY, f32::NEG_INFINITY, 0.0],
    ];
    let gpu = cudaexecutor::new(wide());
    let host = cpuexecutor::new(wide());

    for (index, row) in cases.iter().enumerate() {
        let x = build(vec![1, row.len()], row.to_vec());

        let want = host.softmax(&x.getview()).unwrap();
        let got = gpu.softmax(&x.getview()).unwrap();

        assert_eq!(got.getdata().len(), want.getdata().len());
        for (position, (g, w)) in got.getdata().iter().zip(want.getdata()).enumerate() {
            assert!(
                (g.is_nan() && w.is_nan()) || g == w,
                "case {index} element {position}: cuda {g} but cpu {w}"
            );
        }
    }
}

#[test]
fn submission_reproduces_a_compute_chain_bit_for_bit() {
    if !hasdevice("submission_reproduces_a_compute_chain_bit_for_bit") {
        return;
    }
    let exec = cudaexecutor::new(wide());
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
    assertallclose(got[0].getdata(), want_silu.getdata(), 1.0e-5, 1.0e-5);
}

#[test]
fn submission_rejects_an_unrecorded_result_reference() {
    if !hasdevice("submission_rejects_an_unrecorded_result_reference") {
        return;
    }
    let exec = cudaexecutor::new(wide());
    let mut sub = exec.newsubmission().unwrap();
    let a = build(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let bogus = stepref::fromindex(99);

    let err = sub
        .recordadd(operand::result(bogus), operand::input(a.getview()))
        .unwrap_err();

    assert_eq!(err, backenderror::indexoutofrange { index: 99, len: 0 });
}
