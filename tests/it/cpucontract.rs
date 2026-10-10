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
