//! Cross-crate contract tests for `cpu` through its public API.
//!
//! These tests prove that the CPU backend composes with the frozen `api`
//! contracts and with `tensor` from outside the crate: a caller constructs a
//! tensor, takes a view, and dispatches an operator through the executor.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use crate::support::{build, wide};
use ::tensor::dtype;
use api::{backend, binarykernel, compute, gatherkernel, kernel, weightedkernel};
use cpu::{
    CPUMAXRANK, cpuadd, cpuattention, cpubackend, cpuembedding, cpuexecutor, cpugelutanh,
    cpumatmul, cpumul, cpuop, cpurelu, cpurmsnorm, cpurope, cpusilu, cpusoftmax,
};
use lib::{limiterror, limits};
use testkit::{assertions::assertallclose, reference};

#[test]
fn backend_exposes_identity_device_and_f32_capability() {
    let b = cpubackend::new(limits::new(16, 64, 8, 2));
    assert_eq!(b.getname(), "cpu");
    assert!(b.getdevice().iscpu());
    assert_eq!(b.getdevice(), api::device::cpu());
    let caps = b.getcapabilities();
    assert_eq!(caps.getmaxrank(), CPUMAXRANK);
    assert_eq!(caps.getdtypes(), &[dtype::f32]);
    assert!(caps.supports(dtype::f32));
    assert!(!caps.supports(dtype::f16));
    assert!(!caps.supports(dtype::i32));
}

#[test]
fn kernels_implement_the_frozen_kernel_contract() {
    assert_eq!(kernel::<f32>::getname(&cpurelu), "relu");
    assert_eq!(kernel::<f32>::getname(&cpusilu), "silu");
    assert_eq!(kernel::<f32>::getname(&cpugelutanh), "gelutanh");
    assert_eq!(kernel::<f32>::getname(&cpusoftmax), "softmax");

    let t = build(vec![2, 2], vec![-1.0, 2.0, -3.0, 4.0]);
    let out = cpurelu.apply(&t.getview()).unwrap();
    assert_eq!(out.getdata(), &[0.0, 2.0, 0.0, 4.0]);
    assert_eq!(out.getshape().getdims(), &[2, 2]);
}

#[test]
fn executor_dispatches_every_operator_end_to_end() {
    let exec = cpuexecutor::new(wide());
    let t = build(vec![2, 3], vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
    let v = t.getview();

    for op in [cpuop::relu, cpuop::silu, cpuop::gelutanh, cpuop::softmax] {
        let out = exec.apply(op, &v).unwrap();
        assert_eq!(out.getshape().getdims(), &[2, 3]);
        assert_eq!(out.getdtype(), dtype::f32);
    }

    let soft = exec.apply(cpuop::softmax, &v).unwrap();
    let first: f32 = soft.getdata()[..3].iter().sum();
    let second: f32 = soft.getdata()[3..].iter().sum();
    assert!((first - 1.0).abs() <= 1.0e-6);
    assert!((second - 1.0).abs() <= 1.0e-6);
}

#[test]
fn executor_rejects_rank_beyond_declared_maximum() {
    let exec = cpuexecutor::new(wide());
    let t = build(vec![1, 1, 1, 1, 1, 1, 1, 1, 1], vec![0.0]);
    assert_eq!(
        exec.apply(cpuop::relu, &t.getview()).map(|_| ()),
        Err(cpu::cpuerror::rankexceeded {
            rank: CPUMAXRANK + 1,
            max: CPUMAXRANK,
        })
    );
}

#[test]
fn executor_rejects_budget_violations_with_limiterror() {
    let tight = cpuexecutor::new(limits::new(2, 8, 8, 2));
    let t = build(vec![4], vec![1.0; 4]);
    assert_eq!(
        tight.apply(cpuop::relu, &t.getview()).map(|_| ()),
        Err(cpu::cpuerror::limit(limiterror::maxelementsexceeded {
            requested: 4,
            limit: 2,
        }))
    );
}

#[test]
fn executor_reports_softmax_rejection_as_backend_error() {
    let exec = cpuexecutor::new(wide());
    let t = build(vec![3, 0], vec![]);
    let err = exec.apply(cpuop::softmax, &t.getview()).unwrap_err();
    assert_eq!(
        err,
        cpu::cpuerror::backend(api::backenderror::unsupportedoperation { op: "softmax" })
    );
    assert_eq!(err.to_string(), "unsupported operation: softmax");
}

#[test]
fn operator_names_match_the_enum() {
    for op in [cpuop::relu, cpuop::silu, cpuop::gelutanh, cpuop::softmax] {
        assert!(!op.getname().is_empty());
    }
    assert_eq!(cpuop::relu.getname(), "relu");
    assert_eq!(cpuop::softmax.getname(), "softmax");
}

#[test]
fn binary_kernels_match_the_scalar_reference() {
    let a = build(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = build(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let product = cpumatmul.apply(&a.getview(), &b.getview()).unwrap();
    assertallclose(
        product.getdata(),
        &reference::matmul(a.getdata(), b.getdata(), 2, 3, 2),
        1.0e-5,
        1.0e-6,
    );

    let sum = cpuadd.apply(&a.getview(), &a.getview()).unwrap();
    assert_eq!(sum.getdata(), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    let prod = cpumul.apply(&a.getview(), &a.getview()).unwrap();
    assert_eq!(prod.getdata(), &[1.0, 4.0, 9.0, 16.0, 25.0, 36.0]);

    assert_eq!(binarykernel::<f32>::getname(&cpumatmul), "matmul");
    assert_eq!(binarykernel::<f32>::getname(&cpuadd), "add");
    assert_eq!(binarykernel::<f32>::getname(&cpumul), "mul");
}

#[test]
fn matmul_rejects_incompatible_shapes() {
    let a = build(vec![2, 3], vec![0.0; 6]);
    let b = build(vec![2, 2], vec![0.0; 4]);
    assert_eq!(
        cpumatmul.apply(&a.getview(), &b.getview()).map(|_| ()),
        Err(api::backenderror::shapemismatch { op: "matmul" })
    );
}

/// The blocked packed path (`m >= 32`) agrees with the scalar reference
/// across panel and chunk boundaries: the first case spans two inner chunks
/// (`k = 512 + 88`) and two column panels (`n = 256 + 44`), the second leaves
/// a one-element inner chunk, a one-column panel, and a partial row block.
#[test]
fn matmul_blocked_path_matches_the_scalar_reference() {
    for (m, k, n) in [(32usize, 600usize, 300usize), (100, 513, 257)] {
        let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.13).sin()).collect();
        let rhs: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.17).cos()).collect();
        let a = build(vec![m, k], lhs.clone());
        let b = build(vec![k, n], rhs.clone());
        let out = cpumatmul.apply(&a.getview(), &b.getview()).unwrap();
        assertallclose(
            out.getdata(),
            &reference::matmul(&lhs, &rhs, m, k, n),
            1.0e-4,
            1.0e-4,
        );
    }
}

/// The blocked packed path is bitwise deterministic for fixed inputs.
#[test]
fn matmul_blocked_path_is_deterministic() {
    let m = 64usize;
    let k = 1024usize;
    let n = 512usize;
    let a = build(
        vec![m, k],
        (0..m * k).map(|i| (i as f32 * 0.11).sin()).collect(),
    );
    let b = build(
        vec![k, n],
        (0..k * n).map(|i| (i as f32 * 0.19).cos()).collect(),
    );
    let first = cpumatmul.apply(&a.getview(), &b.getview()).unwrap();
    let second = cpumatmul.apply(&a.getview(), &b.getview()).unwrap();
    assert_eq!(first.getdata(), second.getdata());
}

/// The row-count switch between the direct path (`m < 32`) and the blocked
/// packed path (`m >= 32`) does not change the result beyond tolerance.
#[test]
fn matmul_path_switch_matches_the_scalar_reference() {
    let k = 64usize;
    let n = 48usize;
    for m in [31usize, 32usize, 33usize] {
        let lhs: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.29).sin()).collect();
        let rhs: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.31).cos()).collect();
        let a = build(vec![m, k], lhs.clone());
        let b = build(vec![k, n], rhs.clone());
        let out = cpumatmul.apply(&a.getview(), &b.getview()).unwrap();
        assertallclose(
            out.getdata(),
            &reference::matmul(&lhs, &rhs, m, k, n),
            1.0e-4,
            1.0e-4,
        );
    }
}

/// The blocked packed path handles a leading batch dimension as per-slice
/// multiplications.
#[test]
fn matmul_blocked_path_handles_a_leading_batch_dimension() {
    let (batch, m, k, n) = (2usize, 64usize, 128usize, 64usize);
    let lhs: Vec<f32> = (0..batch * m * k)
        .map(|i| (i as f32 * 0.23).sin())
        .collect();
    let rhs: Vec<f32> = (0..batch * k * n)
        .map(|i| (i as f32 * 0.37).cos())
        .collect();
    let a = build(vec![batch, m, k], lhs.clone());
    let b = build(vec![batch, k, n], rhs.clone());
    let out = cpumatmul.apply(&a.getview(), &b.getview()).unwrap();
    let mut want = reference::matmul(&lhs[0..m * k], &rhs[0..k * n], m, k, n);
    want.extend_from_slice(&reference::matmul(&lhs[m * k..], &rhs[k * n..], m, k, n));
    assertallclose(out.getdata(), &want, 1.0e-4, 1.0e-4);
}

/// The blocked packed path is bit-for-bit identical across worker counts:
/// every output element's fused accumulation sequence does not depend on the
/// thread layout.
#[test]
fn matmul_threaded_matches_the_single_threaded_result() {
    let (m, k, n) = (200usize, 600usize, 300usize);
    let a = build(
        vec![m, k],
        (0..m * k).map(|i| (i as f32 * 0.17).sin()).collect(),
    );
    let b = build(
        vec![k, n],
        (0..k * n).map(|i| (i as f32 * 0.21).cos()).collect(),
    );
    let serial = cpuexecutor::new(wide());
    let wanted = serial
        .applybinary(cpuop::matmul, &a.getview(), &b.getview())
        .unwrap();
    for workers in [2usize, 4, 8] {
        let exec = cpuexecutor::newwithworkers(wide(), workers).unwrap();
        let got = exec
            .applybinary(cpuop::matmul, &a.getview(), &b.getview())
            .unwrap();
        assert_eq!(got.getdata(), wanted.getdata());
    }
}

/// The executor rejects a zero worker budget instead of silently serialising.
#[test]
fn executor_rejects_zero_workers() {
    assert!(cpuexecutor::newwithworkers(wide(), 0).is_err());
}

/// The rmsnorm kernel is bit-for-bit identical across worker counts: every
/// row's arithmetic sequence does not depend on the thread layout.
#[test]
fn rmsnorm_threaded_matches_the_single_threaded_result() {
    let rows = 40usize;
    let width = 600usize;
    let x = build(
        vec![rows, width],
        (0..rows * width).map(|i| (i as f32 * 0.27).sin()).collect(),
    );
    let w = build(vec![width], vec![1.0_f32; width]);
    let serial = cpuexecutor::new(wide());
    let wanted = serial
        .applyweighted(cpuop::rmsnorm, &x.getview(), &w.getview())
        .unwrap();
    for workers in [2usize, 4, 8] {
        let exec = cpuexecutor::newwithworkers(wide(), workers).unwrap();
        let got = exec
            .applyweighted(cpuop::rmsnorm, &x.getview(), &w.getview())
            .unwrap();
        assert_eq!(got.getdata(), wanted.getdata());
    }
}

#[test]
fn weighted_kernel_matches_the_scalar_reference() {
    let x = build(vec![2, 2], vec![3.0, 4.0, 0.0, 5.0]);
    let w = build(vec![2], vec![1.0, 2.0]);
    let out = cpurmsnorm::new(0.0)
        .apply(&x.getview(), &w.getview())
        .unwrap();
    let mut expected = reference::rmsnorm(&[3.0, 4.0], &[1.0, 2.0], 0.0);
    expected.extend(reference::rmsnorm(&[0.0, 5.0], &[1.0, 2.0], 0.0));
    assertallclose(out.getdata(), &expected, 1.0e-5, 1.0e-6);
    assert_eq!(
        weightedkernel::<f32>::getname(&cpurmsnorm::default()),
        "rmsnorm"
    );
}

#[test]
fn gather_kernel_matches_the_scalar_reference() {
    let table = build(vec![3, 2], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    let out = cpuembedding.apply(&table.getview(), &[2, 0]).unwrap();
    assertallclose(
        out.getdata(),
        &reference::embedding(table.getdata(), 2, &[2, 0]),
        0.0,
        0.0,
    );
    assert_eq!(gatherkernel::<f32>::getname(&cpuembedding), "embedding");
    assert_eq!(
        cpuembedding.apply(&table.getview(), &[3]).map(|_| ()),
        Err(api::backenderror::indexoutofrange { index: 3, len: 3 })
    );
}

#[test]
fn rope_matches_the_scalar_reference_per_position() {
    let heads = 2;
    let headdim = 4;
    let width = heads * headdim;
    let seq = 3;
    let input: Vec<f32> = (0..seq * width).map(|i| (i as f32) * 0.1 - 1.0).collect();
    let x = build(vec![seq, width], input.clone());
    let out = cpurope::new(heads, 10000.0).apply(&x.getview()).unwrap();
    let mut expected = Vec::new();
    for s in 0..seq {
        expected.extend(reference::rope(
            &input[s * width..(s + 1) * width],
            s,
            heads,
            headdim,
            10000.0,
        ));
    }
    assertallclose(out.getdata(), &expected, 1.0e-5, 1.0e-5);
}

#[test]
fn attention_single_position_returns_the_value() {
    let q = build(vec![1, 2], vec![1.0, 0.0]);
    let k = build(vec![1, 2], vec![1.0, 0.0]);
    let v = build(vec![1, 2], vec![3.0, 4.0]);
    let out = cpuattention
        .apply(&q.getview(), &k.getview(), &v.getview(), 1, 1, true)
        .unwrap();
    assertallclose(out.getdata(), &[3.0, 4.0], 1.0e-6, 1.0e-6);
}

/// Scalar attention reference restating the shape semantics `cpuattention`
/// documents: position `s` attends to `t <= offset + s` under the causal mask
/// (or to every key without it), and the max subtraction is the stable one the
/// kernel shares. Kept local because only this suite needs it, so the
/// expectation is not derived from the code under test.
#[allow(clippy::too_many_arguments)]
fn attend(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    qseq: usize,
    heads: usize,
    kvheads: usize,
    headdim: usize,
    causal: bool,
) -> Vec<f32> {
    let stride = heads * headdim;
    let kstride = kvheads * headdim;
    let group = heads / kvheads;
    let scale = 1.0 / (headdim as f32).sqrt();
    let kvseq = k.len() / kstride;
    let offset = kvseq - qseq;
    let mut out = vec![0.0_f32; qseq * stride];
    for position in 0..qseq {
        for head in 0..heads {
            let kvhead = head / group;
            let qrow =
                &q[position * stride + head * headdim..position * stride + (head + 1) * headdim];
            let tmax = if causal {
                (position + offset + 1).min(kvseq)
            } else {
                kvseq
            };
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
            let mut denom = 0.0_f32;
            for &score in &scores {
                denom += (score - maxscore).exp();
            }
            for (t, &score) in scores.iter().enumerate() {
                let vrow = &v[t * kstride + kvhead * headdim..t * kstride + (kvhead + 1) * headdim];
                let weight = (score - maxscore).exp() / denom;
                for i in 0..headdim {
                    out[position * stride + head * headdim + i] += weight * vrow[i];
                }
            }
        }
    }
    out
}

/// The blocked kernel matches the scalar reference for a causal grouped-query
/// call that spans several key tiles, query blocks, and partial boundary rows:
/// the sequence length crosses the 256-key tile boundary twice, the head
/// dimension is not a multiple of four, and two query heads share each key
/// head.
#[test]
fn attention_blocked_matches_the_scalar_reference() {
    let heads = 4usize;
    let kvheads = 2usize;
    let headdim = 6usize;
    let seq = 600usize;
    let q = build(
        vec![seq, heads * headdim],
        (0..seq * heads * headdim)
            .map(|i| (i as f32 * 0.13).sin())
            .collect(),
    );
    let k = build(
        vec![seq, kvheads * headdim],
        (0..seq * kvheads * headdim)
            .map(|i| (i as f32 * 0.17).cos())
            .collect(),
    );
    let v = build(
        vec![seq, kvheads * headdim],
        (0..seq * kvheads * headdim)
            .map(|i| (i as f32 * 0.19).sin())
            .collect(),
    );
    let out = cpuattention
        .apply(
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
        true,
    );
    assertallclose(out.getdata(), &want, 1.0e-4, 1.0e-4);
}

/// The blocked kernel matches the scalar reference without the causal mask.
#[test]
fn attention_blocked_non_causal_matches_the_scalar_reference() {
    let heads = 2usize;
    let kvheads = 1usize;
    let headdim = 4usize;
    let seq = 100usize;
    let q = build(
        vec![seq, heads * headdim],
        (0..seq * heads * headdim)
            .map(|i| (i as f32 * 0.23).sin())
            .collect(),
    );
    let k = build(
        vec![seq, kvheads * headdim],
        (0..seq * kvheads * headdim)
            .map(|i| (i as f32 * 0.29).cos())
            .collect(),
    );
    let v = build(
        vec![seq, kvheads * headdim],
        (0..seq * kvheads * headdim)
            .map(|i| (i as f32 * 0.31).sin())
            .collect(),
    );
    let out = cpuattention
        .apply(
            &q.getview(),
            &k.getview(),
            &v.getview(),
            heads,
            kvheads,
            false,
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
        false,
    );
    assertallclose(out.getdata(), &want, 1.0e-4, 1.0e-4);
}

/// The blocked kernel is bitwise deterministic for fixed inputs.
#[test]
fn attention_blocked_is_deterministic() {
    let heads = 2usize;
    let kvheads = 1usize;
    let headdim = 4usize;
    let seq = 300usize;
    let q = build(
        vec![seq, heads * headdim],
        (0..seq * heads * headdim)
            .map(|i| (i as f32 * 0.37).sin())
            .collect(),
    );
    let k = build(
        vec![seq, kvheads * headdim],
        (0..seq * kvheads * headdim)
            .map(|i| (i as f32 * 0.41).cos())
            .collect(),
    );
    let v = build(
        vec![seq, kvheads * headdim],
        (0..seq * kvheads * headdim)
            .map(|i| (i as f32 * 0.43).sin())
            .collect(),
    );
    let first = cpuattention
        .apply(
            &q.getview(),
            &k.getview(),
            &v.getview(),
            heads,
            kvheads,
            true,
        )
        .unwrap();
    let second = cpuattention
        .apply(
            &q.getview(),
            &k.getview(),
            &v.getview(),
            heads,
            kvheads,
            true,
        )
        .unwrap();
    assert_eq!(first.getdata(), second.getdata());
}

/// The blocked kernel is bit-for-bit identical across worker counts: every
/// row's arithmetic sequence does not depend on the thread layout.
#[test]
fn attention_threaded_matches_the_single_threaded_result() {
    let heads = 4usize;
    let kvheads = 2usize;
    let headdim = 6usize;
    let seq = 600usize;
    let q = build(
        vec![seq, heads * headdim],
        (0..seq * heads * headdim)
            .map(|i| (i as f32 * 0.13).sin())
            .collect(),
    );
    let k = build(
        vec![seq, kvheads * headdim],
        (0..seq * kvheads * headdim)
            .map(|i| (i as f32 * 0.17).cos())
            .collect(),
    );
    let v = build(
        vec![seq, kvheads * headdim],
        (0..seq * kvheads * headdim)
            .map(|i| (i as f32 * 0.19).sin())
            .collect(),
    );
    let serial = cpuexecutor::new(wide());
    let wanted = serial
        .attention(
            &q.getview(),
            &k.getview(),
            &v.getview(),
            heads,
            kvheads,
            true,
        )
        .unwrap();
    for workers in [2usize, 4, 8] {
        let exec = cpuexecutor::newwithworkers(wide(), workers).unwrap();
        let got = exec
            .attention(
                &q.getview(),
                &k.getview(),
                &v.getview(),
                heads,
                kvheads,
                true,
            )
            .unwrap();
        assert_eq!(got.getdata(), wanted.getdata());
    }
}

#[test]
fn executor_dispatches_binary_weighted_and_gather() {
    let exec = cpuexecutor::new(wide());
    let a = build(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let id = build(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let product = exec
        .applybinary(cpuop::matmul, &a.getview(), &id.getview())
        .unwrap();
    assert_eq!(product.getdata(), &[1.0, 2.0, 3.0, 4.0]);

    let w = build(vec![2], vec![1.0, 1.0]);
    assert!(
        exec.applyweighted(cpuop::rmsnorm, &a.getview(), &w.getview())
            .is_ok()
    );

    let table = build(vec![2, 2], vec![0.0, 1.0, 2.0, 3.0]);
    let gathered = exec
        .applygather(cpuop::embedding, &table.getview(), &[1])
        .unwrap();
    assert_eq!(gathered.getdata(), &[2.0, 3.0]);
}

#[test]
fn compute_contract_routes_through_the_executor() {
    let exec = cpuexecutor::new(wide());
    let a = build(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let id = build(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
    let product = exec.matmul(&a.getview(), &id.getview()).unwrap();
    assert_eq!(product.getdata(), &[1.0, 2.0, 3.0, 4.0]);

    let tight = cpuexecutor::new(limits::new(2, 1 << 20, 8, 2));
    let big = build(vec![4], vec![1.0; 4]);
    assert_eq!(
        compute::silu(&tight, &big.getview()).map(|_| ()),
        Err(api::backenderror::resourceexhausted)
    );
}
