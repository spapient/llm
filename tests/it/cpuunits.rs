//! Relocated unit tests for the `cpu` crate.
//!
//! Formerly inline `#[cfg(test)] mod tests` blocks, moved here so that
//! production sources under `crates/cpu/src` hold no test code. Each source
//! module's tests are wrapped in a nested module to keep helper and test names
//! isolated. Every assertion reaches the crate through its public interface
//! (see `docs/engineering.md`).
#![allow(clippy::unwrap_used, clippy::expect_used, dead_code)]

/// Tests relocated from `crates/cpu/src/activation.rs`.
mod activation {
    use crate::support::close;
    use ::tensor::{shape, tensor};
    use api::kernel;
    use proptest::collection;
    use proptest::prelude::*;
    use {cpu::cpugelutanh, cpu::cpurelu, cpu::cpusilu};

    fn build(data: Vec<f32>) -> tensor<f32> {
        tensor::new(shape::new(vec![data.len()]).unwrap(), data).unwrap()
    }

    #[test]
    fn relu_clamps_negatives_and_keeps_non_negatives() {
        let t = build(vec![-2.0, -0.0, 0.0, 3.5]);
        let out = cpurelu.apply(&t.getview()).unwrap();
        assert_eq!(out.getdata(), &[0.0, 0.0, 0.0, 3.5]);
        assert_eq!(out.getshape().getdims(), &[4]);
    }

    #[test]
    fn relu_propagates_nan() {
        let t = build(vec![f32::NAN]);
        let out = cpurelu.apply(&t.getview()).unwrap();
        assert!(out.getdata()[0].is_nan());
    }

    #[test]
    fn relu_handles_empty_input() {
        let t = build(vec![]);
        let out = cpurelu.apply(&t.getview()).unwrap();
        assert!(out.isempty());
        assert_eq!(out.getshape().getdims(), &[0]);
    }

    #[test]
    fn silu_matches_reference_values() {
        let cases = [
            (0.0_f32, 0.0_f32),
            (1.0, 0.731_058_6),
            (-1.0, -0.268_941_42),
            (2.0, 1.761_594_2),
            (0.5, 0.311_229_66),
        ];
        let t = build(cases.iter().map(|&(x, _)| x).collect());
        let out = cpusilu.apply(&t.getview()).unwrap();
        for (got, (_, want)) in out.getdata().iter().zip(cases.iter()) {
            assert!(close(*got, *want), "silu mismatch: {got} vs {want}");
        }
    }

    #[test]
    fn silu_saturates_without_overflow() {
        let t = build(vec![-1.0e30, 0.0, 1.0e30]);
        let out = cpusilu.apply(&t.getview()).unwrap();
        assert_eq!(out.getdata()[0], 0.0);
        assert_eq!(out.getdata()[1], 0.0);
        assert!(close(out.getdata()[2], 1.0e30));
    }

    #[test]
    fn gelutanh_matches_reference_values() {
        let cases = [
            (0.0_f32, 0.0_f32),
            (1.0, 0.841_192),
            (-1.0, -0.158_808_01),
            (2.0, 1.954_597_7),
            (0.5, 0.345_714),
            (-3.0, -0.003_637_392_3),
        ];
        let t = build(cases.iter().map(|&(x, _)| x).collect());
        let out = cpugelutanh.apply(&t.getview()).unwrap();
        for (got, (_, want)) in out.getdata().iter().zip(cases.iter()) {
            assert!(close(*got, *want), "gelutanh mismatch: {got} vs {want}");
        }
    }

    #[test]
    fn gelutanh_saturates_at_the_tails() {
        let t = build(vec![-1.0e30, 1.0e30]);
        let out = cpugelutanh.apply(&t.getview()).unwrap();
        assert_eq!(out.getdata()[0], 0.0);
        assert!(out.getdata()[1] > 0.0);
    }

    #[test]
    fn kernels_report_stable_names() {
        assert_eq!(kernel::<f32>::getname(&cpurelu), "relu");
        assert_eq!(kernel::<f32>::getname(&cpusilu), "silu");
        assert_eq!(kernel::<f32>::getname(&cpugelutanh), "gelutanh");
    }

    proptest! {
        #[test]
        fn every_activation_preserves_shape_and_finiteness(
            data in collection::vec(-1.0e6f32..1.0e6, 1..=64)
        ) {
            let t = build(data);
            for out in [
                cpurelu.apply(&t.getview()).unwrap(),
                cpusilu.apply(&t.getview()).unwrap(),
                cpugelutanh.apply(&t.getview()).unwrap(),
            ] {
                prop_assert_eq!(out.getshape().getdims(), t.getshape().getdims());
                prop_assert!(out.getdata().iter().all(|x| x.is_finite()));
            }
        }

        #[test]
        fn relu_is_nonnegative_and_idempotent(
            data in collection::vec(-1.0e6f32..1.0e6, 1..=64)
        ) {
            let t = build(data);
            let once = cpurelu.apply(&t.getview()).unwrap();
            prop_assert!(once.getdata().iter().all(|&x| x >= 0.0));
            let twice = cpurelu.apply(&once.getview()).unwrap();
            prop_assert_eq!(once.getdata(), twice.getdata());
        }

        #[test]
        fn silu_and_gelutanh_are_bounded_by_abs_input(
            data in collection::vec(-1.0e3f32..1.0e3, 1..=64)
        ) {
            let t = build(data.clone());
            for out in [
                cpusilu.apply(&t.getview()).unwrap(),
                cpugelutanh.apply(&t.getview()).unwrap(),
            ] {
                for (got, &x) in out.getdata().iter().zip(data.iter()) {
                    prop_assert!(got.abs() <= x.abs() + 1.0e-3);
                }
            }
        }
    }
}

/// Tests relocated from `crates/cpu/src/attention.rs`.
mod attention {
    use crate::support::{build, close};
    use api::backenderror;
    use cpu::cpuattention;
    use rstest::rstest;

    #[test]
    fn single_position_returns_the_value() {
        let q = build(vec![1, 2], vec![1.0, 0.0]);
        let k = build(vec![1, 2], vec![1.0, 0.0]);
        let v = build(vec![1, 2], vec![3.0, 4.0]);
        let out = cpuattention
            .apply(&q.getview(), &k.getview(), &v.getview(), 1, 1, true)
            .unwrap();
        assert!(close(out.getdata()[0], 3.0));
        assert!(close(out.getdata()[1], 4.0));
    }

    #[test]
    fn causal_mask_excludes_future_positions() {
        let q = build(vec![2, 1], vec![1.0, 1.0]);
        let k = build(vec![2, 1], vec![1.0, 1.0]);
        let v = build(vec![2, 1], vec![1.0, 2.0]);
        let out = cpuattention
            .apply(&q.getview(), &k.getview(), &v.getview(), 1, 1, true)
            .unwrap();
        assert!(close(out.getdata()[0], 1.0));
        assert!(close(out.getdata()[1], 1.5));
    }

    #[test]
    fn non_causal_attends_to_every_position() {
        let q = build(vec![2, 1], vec![1.0, 1.0]);
        let k = build(vec![2, 1], vec![1.0, 1.0]);
        let v = build(vec![2, 1], vec![1.0, 3.0]);
        let out = cpuattention
            .apply(&q.getview(), &k.getview(), &v.getview(), 1, 1, false)
            .unwrap();
        assert!(close(out.getdata()[0], 2.0));
        assert!(close(out.getdata()[1], 2.0));
    }

    #[test]
    fn grouped_query_heads_share_key_values() {
        // heads = 2, kvheads = 1, headdim = 1.
        let q = build(vec![1, 2], vec![1.0, 1.0]);
        let k = build(vec![1, 1], vec![2.0]);
        let v = build(vec![1, 1], vec![5.0]);
        let out = cpuattention
            .apply(&q.getview(), &k.getview(), &v.getview(), 2, 1, false)
            .unwrap();
        assert_eq!(out.getshape().getdims(), &[1, 2]);
        assert!(close(out.getdata()[0], 5.0));
        assert!(close(out.getdata()[1], 5.0));
    }

    /// Every shape or head-count violation must be reported as the same
    /// `shapemismatch` for the `attention` operator.
    #[rstest]
    #[case::head_count(&[1, 4], &[1, 2], &[1, 2], 2, 3)]
    #[case::sequence_length(&[2, 2], &[1, 2], &[1, 2], 1, 1)]
    #[case::key_width(&[1, 2], &[1, 3], &[1, 3], 1, 1)]
    #[case::zero_kvheads(&[1, 1], &[1, 1], &[1, 1], 1, 0)]
    fn mismatches_are_rejected(
        #[case] qdims: &[usize],
        #[case] kdims: &[usize],
        #[case] vdims: &[usize],
        #[case] heads: usize,
        #[case] kvheads: usize,
    ) {
        let q = build(qdims, vec![0.0; qdims.iter().product()]);
        let k = build(kdims, vec![0.0; kdims.iter().product()]);
        let v = build(vdims, vec![0.0; vdims.iter().product()]);
        assert_eq!(
            cpuattention
                .apply(
                    &q.getview(),
                    &k.getview(),
                    &v.getview(),
                    heads,
                    kvheads,
                    true
                )
                .map(|_| ()),
            Err(backenderror::shapemismatch { op: "attention" })
        );
    }

    #[test]
    fn empty_sequence_produces_empty_output() {
        let q = build(vec![0, 2], vec![]);
        let k = build(vec![0, 2], vec![]);
        let v = build(vec![0, 2], vec![]);
        let out = cpuattention
            .apply(&q.getview(), &k.getview(), &v.getview(), 1, 1, true)
            .unwrap();
        assert_eq!(out.getshape().getdims(), &[0, 2]);
    }
}

/// Tests relocated from `crates/cpu/src/backend.rs`.
mod backend {
    use ::tensor::dtype;
    use api::{backend, device};
    use lib::limits;
    use {cpu::CPUMAXRANK, cpu::cpubackend};

    fn cpucaps() -> cpubackend {
        cpubackend::new(limits::new(1024, 4096, 128, 8))
    }

    #[test]
    fn declares_cpu_identity_and_device() {
        let b = cpucaps();
        assert_eq!(b.getname(), "cpu");
        assert!(b.getdevice().iscpu());
        assert_eq!(b.getdevice().getkind(), device::cpu().getkind());
    }

    #[test]
    fn declares_f32_support_only() {
        let b = cpucaps();
        let caps = b.getcapabilities();
        assert!(caps.supports(dtype::f32));
        assert!(!caps.supports(dtype::f16));
        assert!(!caps.supports(dtype::bf16));
        assert!(!caps.supports(dtype::i32));
        assert_eq!(caps.getdtypes(), &[dtype::f32]);
    }

    #[test]
    fn declares_rank_bound_and_echoes_limits() {
        let b = cpucaps();
        let caps = b.getcapabilities();
        assert_eq!(caps.getmaxrank(), CPUMAXRANK);
        assert_eq!(caps.getlimits().getmaxelements(), 1024);
        assert_eq!(caps.getlimits().getmaxbytes(), 4096);
        assert_eq!(caps.getlimits().getmaxsequencelen(), 128);
        assert_eq!(caps.getlimits().getmaxbatch(), 8);
    }
}

/// Tests relocated from `crates/cpu/src/elementwise.rs`.
mod elementwise {
    use crate::support::build;
    use api::{backenderror, binarykernel};
    use {cpu::cpuadd, cpu::cpumul};

    #[test]
    fn add_is_elementwise() {
        let a = build(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = build(vec![2, 2], vec![10.0, 20.0, 30.0, 40.0]);
        let out = cpuadd.apply(&a.getview(), &b.getview()).unwrap();
        assert_eq!(out.getdata(), &[11.0, 22.0, 33.0, 44.0]);
        assert_eq!(out.getshape().getdims(), &[2, 2]);
    }

    #[test]
    fn mul_is_elementwise() {
        let a = build(vec![3], vec![1.0, 2.0, 3.0]);
        let b = build(vec![3], vec![4.0, 5.0, 6.0]);
        let out = cpumul.apply(&a.getview(), &b.getview()).unwrap();
        assert_eq!(out.getdata(), &[4.0, 10.0, 18.0]);
    }

    #[test]
    fn add_propagates_nan_and_infinity() {
        let a = build(vec![2], vec![f32::NAN, f32::INFINITY]);
        let b = build(vec![2], vec![1.0, 1.0]);
        let out = cpuadd.apply(&a.getview(), &b.getview()).unwrap();
        assert!(out.getdata()[0].is_nan());
        assert_eq!(out.getdata()[1], f32::INFINITY);
    }

    #[test]
    fn shape_mismatch_is_rejected() {
        let a = build(vec![2, 3], vec![0.0; 6]);
        let b = build(vec![3, 2], vec![0.0; 6]);
        assert_eq!(
            cpuadd.apply(&a.getview(), &b.getview()).map(|_| ()),
            Err(backenderror::shapemismatch { op: "add" })
        );
        assert_eq!(
            cpumul.apply(&a.getview(), &b.getview()).map(|_| ()),
            Err(backenderror::shapemismatch { op: "mul" })
        );
    }

    #[test]
    fn empty_operands_produce_empty_output() {
        let a = build(vec![0, 3], vec![]);
        let b = build(vec![0, 3], vec![]);
        assert!(cpuadd.apply(&a.getview(), &b.getview()).unwrap().isempty());
    }

    #[test]
    fn reports_stable_names() {
        assert_eq!(binarykernel::<f32>::getname(&cpuadd), "add");
        assert_eq!(binarykernel::<f32>::getname(&cpumul), "mul");
    }
}

/// Tests relocated from `crates/cpu/src/embedding.rs`.
mod embedding {
    use crate::support::build;
    use api::{backenderror, gatherkernel};
    use cpu::cpuembedding;

    #[test]
    fn gathers_rows_in_index_order() {
        let table = build(vec![3, 2], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
        let out = cpuembedding.apply(&table.getview(), &[2, 0, 2]).unwrap();
        assert_eq!(out.getshape().getdims(), &[3, 2]);
        assert_eq!(out.getdata(), &[4.0, 5.0, 0.0, 1.0, 4.0, 5.0]);
    }

    #[test]
    fn rank_one_table_gathers_scalars() {
        let table = build(vec![4], vec![10.0, 20.0, 30.0, 40.0]);
        let out = cpuembedding.apply(&table.getview(), &[3, 1]).unwrap();
        assert_eq!(out.getshape().getdims(), &[2]);
        assert_eq!(out.getdata(), &[40.0, 20.0]);
    }

    #[test]
    fn empty_index_list_yields_empty_output() {
        let table = build(vec![2, 3], vec![1.0; 6]);
        let out = cpuembedding.apply(&table.getview(), &[]).unwrap();
        assert_eq!(out.getshape().getdims(), &[0, 3]);
        assert!(out.isempty());
    }

    #[test]
    fn out_of_range_index_is_rejected() {
        let table = build(vec![2, 3], vec![1.0; 6]);
        assert_eq!(
            cpuembedding.apply(&table.getview(), &[0, 2]).map(|_| ()),
            Err(backenderror::indexoutofrange { index: 2, len: 2 })
        );
    }

    #[test]
    fn reports_stable_name() {
        assert_eq!(gatherkernel::<f32>::getname(&cpuembedding), "embedding");
    }
}

/// Tests relocated from `crates/cpu/src/error.rs`.
mod error {
    use api::backenderror;
    use cpu::cpuerror;
    use lib::limiterror;

    #[test]
    fn converts_from_backenderror() {
        let e: cpuerror = backenderror::sizeoverflow.into();
        assert_eq!(e, cpuerror::backend(backenderror::sizeoverflow));
        assert_eq!(e.to_string(), "size overflow");
    }

    #[test]
    fn converts_from_limiterror() {
        let e: cpuerror = limiterror::maxelementsexceeded {
            requested: 9,
            limit: 8,
        }
        .into();
        assert_eq!(
            e,
            cpuerror::limit(limiterror::maxelementsexceeded {
                requested: 9,
                limit: 8,
            })
        );
        assert_eq!(e.to_string(), "requested element count 9 exceeds limit 8");
    }

    #[test]
    fn rank_variant_carries_both_bounds() {
        let e = cpuerror::rankexceeded { rank: 9, max: 8 };
        assert_eq!(e.to_string(), "input rank 9 exceeds backend maximum 8");
        assert_ne!(e, cpuerror::rankexceeded { rank: 9, max: 9 });
    }

    #[test]
    fn variants_are_distinguishable() {
        assert_ne!(
            cpuerror::backend(backenderror::executionfailed),
            cpuerror::limit(limiterror::maxbatchexceeded {
                requested: 1,
                limit: 0,
            })
        );
    }
}

/// Tests relocated from `crates/cpu/src/executor.rs`.
mod executor {
    use crate::support::build;
    use api::{backend, backenderror, compute};
    use cpu::cpuerror;
    use lib::{limiterror, limits};
    use rstest::rstest;
    use {cpu::cpuexecutor, cpu::cpuop};

    fn wide() -> cpuexecutor {
        cpuexecutor::new(limits::new(1 << 24, 1 << 28, 1 << 20, 1 << 16))
    }

    #[test]
    fn operator_names_are_stable() {
        assert_eq!(cpuop::relu.getname(), "relu");
        assert_eq!(cpuop::silu.getname(), "silu");
        assert_eq!(cpuop::gelutanh.getname(), "gelutanh");
        assert_eq!(cpuop::softmax.getname(), "softmax");
        assert_eq!(cpuop::matmul.getname(), "matmul");
        assert_eq!(cpuop::add.getname(), "add");
        assert_eq!(cpuop::mul.getname(), "mul");
        assert_eq!(cpuop::rmsnorm.getname(), "rmsnorm");
        assert_eq!(cpuop::embedding.getname(), "embedding");
    }

    #[test]
    fn dispatches_each_unary_operator() {
        let exec = wide();
        let t = build(vec![2], vec![-1.0, 2.0]);
        let v = t.getview();
        assert_eq!(exec.apply(cpuop::relu, &v).unwrap().getdata(), &[0.0, 2.0]);
        assert!(exec.apply(cpuop::silu, &v).unwrap().getdata()[1] > 1.0);
        assert!(exec.apply(cpuop::gelutanh, &v).unwrap().getdata()[1] > 1.0);
        let s = exec.apply(cpuop::softmax, &v).unwrap();
        assert!((s.getdata()[0] + s.getdata()[1] - 1.0).abs() <= 1.0e-6);
    }

    #[test]
    fn unary_dispatch_rejects_non_unary_operators() {
        let exec = wide();
        let t = build(vec![2], vec![1.0, 2.0]);
        assert_eq!(
            exec.apply(cpuop::matmul, &t.getview()).map(|_| ()),
            Err(cpuerror::backend(backenderror::unsupportedoperation {
                op: "matmul",
            }))
        );
    }

    #[test]
    fn dispatches_binary_operators() {
        let exec = wide();
        let a = build(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = build(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
        let out = exec
            .applybinary(cpuop::matmul, &a.getview(), &b.getview())
            .unwrap();
        assert_eq!(out.getdata(), &[1.0, 2.0, 3.0, 4.0]);
        let sum = exec
            .applybinary(cpuop::add, &a.getview(), &a.getview())
            .unwrap();
        assert_eq!(sum.getdata(), &[2.0, 4.0, 6.0, 8.0]);
        let prod = exec
            .applybinary(cpuop::mul, &a.getview(), &a.getview())
            .unwrap();
        assert_eq!(prod.getdata(), &[1.0, 4.0, 9.0, 16.0]);
    }

    #[test]
    fn dispatches_weighted_and_gather_operators() {
        let exec = wide();
        let x = build(vec![2], vec![3.0, 4.0]);
        let w = build(vec![2], vec![1.0, 1.0]);
        let norm = exec
            .applyweighted(cpuop::rmsnorm, &x.getview(), &w.getview())
            .unwrap();
        assert!(norm.getdata()[0] > 0.8 && norm.getdata()[0] < 0.9);
        let table = build(vec![2, 2], vec![0.0, 1.0, 2.0, 3.0]);
        let gathered = exec
            .applygather(cpuop::embedding, &table.getview(), &[1, 0])
            .unwrap();
        assert_eq!(gathered.getdata(), &[2.0, 3.0, 0.0, 1.0]);
    }

    /// Each case lowers one resource ceiling below the operator's demand and
    /// pins the exact budget error the executor must report.
    #[rstest]
    #[case::rank(
        limits::new(1024, 1 << 20, 64, 8),
        &[1, 1, 1, 1, 1, 1, 1, 1, 1],
        cpuerror::rankexceeded { rank: 9, max: 8 }
    )]
    #[case::elements(
        limits::new(2, 1 << 20, 64, 8),
        &[4],
        cpuerror::limit(limiterror::maxelementsexceeded { requested: 4, limit: 2 })
    )]
    #[case::bytes(
        limits::new(1024, 8, 64, 8),
        &[4],
        cpuerror::limit(limiterror::maxbytesexceeded { requested: 16, limit: 8 })
    )]
    fn enforces_the_budget(
        #[case] lim: limits,
        #[case] dims: &[usize],
        #[case] expected: cpuerror,
    ) {
        let exec = cpuexecutor::new(lim);
        let t = build(dims, vec![0.0; dims.iter().product()]);
        assert_eq!(
            exec.apply(cpuop::relu, &t.getview()).map(|_| ()),
            Err(expected)
        );
    }

    #[test]
    fn propagates_kernel_rejection() {
        let exec = wide();
        let t = build(vec![2, 0], vec![]);
        assert_eq!(
            exec.apply(cpuop::softmax, &t.getview()).map(|_| ()),
            Err(cpuerror::backend(backenderror::unsupportedoperation {
                op: "softmax",
            }))
        );
    }

    #[test]
    fn compute_matmul_and_add_route_through_budget() {
        let exec = wide();
        let a = build(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let id = build(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
        let out = exec.matmul(&a.getview(), &id.getview()).unwrap();
        assert_eq!(out.getdata(), &[1.0, 2.0, 3.0, 4.0]);
        let sum = compute::add(&exec, &a.getview(), &a.getview()).unwrap();
        assert_eq!(sum.getdata(), &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn compute_reports_resource_exhaustion_from_budget() {
        let tight = cpuexecutor::new(limits::new(2, 1 << 20, 64, 8));
        let t = build(vec![4], vec![1.0; 4]);
        assert_eq!(
            compute::silu(&tight, &t.getview()).map(|_| ()),
            Err(backenderror::resourceexhausted)
        );
    }

    #[test]
    fn compute_rope_and_attention_dispatch() {
        let exec = wide();
        let q = build(vec![1, 2], vec![1.0, 0.0]);
        let out = compute::rope(&exec, &q.getview(), 1, 10000.0).unwrap();
        assert_eq!(out.getdata(), &[1.0, 0.0]);
        let attn = compute::attention(&exec, &q.getview(), &q.getview(), &q.getview(), 1, 1, true)
            .unwrap();
        assert!((attn.getdata()[0] - 1.0).abs() <= 1.0e-6);
    }

    #[test]
    fn exposes_its_backend_capabilities() {
        let exec = cpuexecutor::new(limits::new(16, 64, 8, 2));
        assert_eq!(
            exec.getbackend()
                .getcapabilities()
                .getlimits()
                .getmaxelements(),
            16
        );
    }
}

/// Tests relocated from `crates/cpu/src/matmul.rs`.
mod matmul {
    use crate::support::build;
    use api::{backenderror, binarykernel};
    use cpu::cpumatmul;

    #[test]
    fn two_by_two_identity_multiplies_to_self() {
        let a = build(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let id = build(vec![2, 2], vec![1.0, 0.0, 0.0, 1.0]);
        let out = cpumatmul.apply(&a.getview(), &id.getview()).unwrap();
        assert_eq!(out.getdata(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(out.getshape().getdims(), &[2, 2]);
    }

    #[test]
    fn rectangular_product_matches_hand_computation() {
        // [2,3] * [3,2]
        let a = build(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = build(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let out = cpumatmul.apply(&a.getview(), &b.getview()).unwrap();
        // row0: [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // row1: [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert_eq!(out.getdata(), &[58.0, 64.0, 139.0, 154.0]);
        assert_eq!(out.getshape().getdims(), &[2, 2]);
    }

    #[test]
    fn batched_product_applies_per_batch() {
        let a = build(vec![2, 1, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = build(vec![2, 2, 1], vec![1.0, 1.0, 10.0, 10.0]);
        let out = cpumatmul.apply(&a.getview(), &b.getview()).unwrap();
        // batch0: [1+2]=3 ; batch1: [30+40]=70
        assert_eq!(out.getshape().getdims(), &[2, 1, 1]);
        assert_eq!(out.getdata(), &[3.0, 70.0]);
    }

    #[test]
    fn inner_dimension_mismatch_is_rejected() {
        let a = build(vec![2, 3], vec![0.0; 6]);
        let b = build(vec![2, 2], vec![0.0; 4]);
        assert_eq!(
            cpumatmul.apply(&a.getview(), &b.getview()).map(|_| ()),
            Err(backenderror::shapemismatch { op: "matmul" })
        );
    }

    #[test]
    fn rank_below_two_is_rejected() {
        let a = build(vec![3], vec![0.0; 3]);
        let b = build(vec![3], vec![0.0; 3]);
        assert_eq!(
            cpumatmul.apply(&a.getview(), &b.getview()).map(|_| ()),
            Err(backenderror::shapemismatch { op: "matmul" })
        );
    }

    #[test]
    fn batch_dimension_mismatch_is_rejected() {
        let a = build(vec![2, 2, 2], vec![0.0; 8]);
        let b = build(vec![3, 2, 2], vec![0.0; 12]);
        assert_eq!(
            cpumatmul.apply(&a.getview(), &b.getview()).map(|_| ()),
            Err(backenderror::shapemismatch { op: "matmul" })
        );
    }

    #[test]
    fn zero_inner_dimension_yields_additive_identity() {
        let a = build(vec![2, 0], vec![]);
        let b = build(vec![0, 3], vec![]);
        let out = cpumatmul.apply(&a.getview(), &b.getview()).unwrap();
        assert_eq!(out.getshape().getdims(), &[2, 3]);
        assert_eq!(out.getdata(), &[0.0; 6]);
    }

    #[test]
    fn reports_stable_name() {
        assert_eq!(binarykernel::<f32>::getname(&cpumatmul), "matmul");
    }
}

/// Tests relocated from `crates/cpu/src/normalization.rs`.
mod normalization {
    use crate::support::{build, close};
    use api::{backenderror, weightedkernel};
    use cpu::cpurmsnorm;

    #[test]
    fn matches_hand_computed_unit_weight() {
        let x = build(vec![2], vec![3.0, 4.0]);
        let w = build(vec![2], vec![1.0, 1.0]);
        let k = cpurmsnorm::new(0.0);
        let out = k.apply(&x.getview(), &w.getview()).unwrap();
        // mean square = 12.5, inv = 0.2828427
        assert!(close(out.getdata()[0], 0.848_528_1));
        assert!(close(out.getdata()[1], 1.131_370_8));
    }

    #[test]
    fn weight_scales_each_element() {
        let x = build(vec![2], vec![3.0, 4.0]);
        let w = build(vec![2], vec![2.0, 3.0]);
        let k = cpurmsnorm::new(0.0);
        let out = k.apply(&x.getview(), &w.getview()).unwrap();
        assert!(close(out.getdata()[0], 1.697_056_2));
        assert!(close(out.getdata()[1], 3.394_112_5));
    }

    #[test]
    fn normalizes_each_row_independently() {
        let x = build(vec![2, 2], vec![3.0, 4.0, 0.0, 5.0]);
        let w = build(vec![2], vec![1.0, 1.0]);
        let out = cpurmsnorm::new(0.0)
            .apply(&x.getview(), &w.getview())
            .unwrap();
        assert!(close(out.getdata()[0], 0.848_528_1));
        assert!(close(out.getdata()[1], 1.131_370_8));
        assert!(close(out.getdata()[2], 0.0));
        assert!(close(out.getdata()[3], core::f32::consts::SQRT_2));
    }

    #[test]
    fn epsilon_stabilizes_zero_row() {
        let x = build(vec![2], vec![0.0, 0.0]);
        let w = build(vec![2], vec![1.0, 1.0]);
        let out = cpurmsnorm::new(1.0e-5)
            .apply(&x.getview(), &w.getview())
            .unwrap();
        assert!(out.getdata().iter().all(|v| v.is_finite()));
        assert!(close(out.getdata()[0], 0.0));
    }

    #[test]
    fn weight_length_mismatch_is_rejected() {
        let x = build(vec![3], vec![1.0, 2.0, 3.0]);
        let w = build(vec![2], vec![1.0, 1.0]);
        assert_eq!(
            cpurmsnorm::new(0.0)
                .apply(&x.getview(), &w.getview())
                .map(|_| ()),
            Err(backenderror::shapemismatch { op: "rmsnorm" })
        );
    }

    #[test]
    fn zero_length_last_axis_is_rejected() {
        let x = build(vec![2, 0], vec![]);
        let w = build(vec![0], vec![]);
        assert_eq!(
            cpurmsnorm::new(0.0)
                .apply(&x.getview(), &w.getview())
                .map(|_| ()),
            Err(backenderror::unsupportedoperation { op: "rmsnorm" })
        );
    }

    #[test]
    fn default_matches_explicit_default_epsilon() {
        assert_eq!(cpurmsnorm::default().geteps(), cpurmsnorm::DEFAULTEPS);
        assert_eq!(cpurmsnorm::new(2.0).geteps(), 2.0);
    }

    #[test]
    fn reports_stable_name() {
        assert_eq!(
            weightedkernel::<f32>::getname(&cpurmsnorm::default()),
            "rmsnorm"
        );
    }
}

/// Tests relocated from `crates/cpu/src/rotary.rs`.
mod rotary {
    use crate::support::{build, close};
    use api::{backenderror, kernel};
    use cpu::cpurope;
    use rstest::rstest;

    #[test]
    fn position_zero_is_identity() {
        let x = build(vec![1, 4], vec![1.0, 2.0, 3.0, 4.0]);
        let out = cpurope::new(2, 10000.0).apply(&x.getview()).unwrap();
        assert_eq!(out.getdata(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn single_head_rotates_by_position() {
        // heads = 1, headdim = 2, freq = base^0 = 1.
        let x = build(vec![2, 2], vec![1.0, 0.0, 1.0, 0.0]);
        let out = cpurope::new(1, 10000.0).apply(&x.getview()).unwrap();
        assert!(close(out.getdata()[0], 1.0));
        assert!(close(out.getdata()[1], 0.0));
        assert!(close(out.getdata()[2], 1.0_f32.cos()));
        assert!(close(out.getdata()[3], 1.0_f32.sin()));
    }

    #[test]
    fn heads_rotate_independently_with_same_frequency() {
        let x = build(vec![2, 4], vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
        let out = cpurope::new(2, 10000.0).apply(&x.getview()).unwrap();
        // Position 0 rows are unchanged.
        assert!(close(out.getdata()[0], 1.0));
        assert!(close(out.getdata()[1], 0.0));
        assert!(close(out.getdata()[2], 1.0));
        assert!(close(out.getdata()[3], 0.0));
        // Position 1 rotates each head by the same angle.
        for base in [4usize, 6] {
            assert!(close(out.getdata()[base], 1.0_f32.cos()));
            assert!(close(out.getdata()[base + 1], 1.0_f32.sin()));
        }
    }

    #[test]
    fn zero_length_sequence_preserves_shape() {
        let x = build(vec![0, 4], vec![]);
        let out = cpurope::new(2, 10000.0).apply(&x.getview()).unwrap();
        assert_eq!(out.getshape().getdims(), &[0, 4]);
        assert!(out.isempty());
    }

    /// Every malformed rotary input must be reported as the same
    /// `shapemismatch` for the `rope` operator.
    #[rstest]
    #[case::odd_head_dimension(&[2, 3], 1)]
    #[case::indivisible_last_axis(&[2, 3], 2)]
    #[case::rank_one(&[4], 1)]
    fn malformed_inputs_are_rejected(#[case] dims: &[usize], #[case] heads: usize) {
        let x = build(dims, vec![0.0; dims.iter().product()]);
        assert_eq!(
            cpurope::new(heads, 10000.0).apply(&x.getview()).map(|_| ()),
            Err(backenderror::shapemismatch { op: "rope" })
        );
    }

    #[test]
    fn reports_configuration_and_name() {
        let k = cpurope::new(3, 500.0);
        assert_eq!(k.getheads(), 3);
        assert_eq!(k.getbase(), 500.0);
        assert_eq!(kernel::<f32>::getname(&k), "rope");
    }
}

/// Tests relocated from `crates/cpu/src/softmax.rs`.
mod softmax {
    use crate::support::build;
    use api::{backenderror, kernel};
    use cpu::cpusoftmax;
    use proptest::collection;
    use proptest::prelude::*;

    fn run(dims: Vec<usize>, data: Vec<f32>) -> Vec<f32> {
        let t = build(dims, data);
        cpusoftmax.apply(&t.getview()).unwrap().getdata().to_vec()
    }

    fn close(got: f32, want: f32) -> bool {
        (got - want).abs() <= 1.0e-6 * (1.0 + want.abs())
    }

    #[test]
    fn matches_hand_computed_reference() {
        let out = run(vec![3], vec![1.0, 2.0, 3.0]);
        let want = [0.090_030_57_f32, 0.244_728_47, 0.665_240_9];
        for (got, w) in out.iter().zip(want.iter()) {
            assert!(close(*got, *w), "softmax mismatch: {got} vs {w}");
        }
    }

    #[test]
    fn normalizes_each_row_independently() {
        let out = run(vec![2, 3], vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0]);
        assert!(close(out[0], 0.090_030_57));
        assert!(close(out[1], 0.244_728_47));
        assert!(close(out[2], 0.665_240_9));
        for got in &out[3..] {
            assert!(close(*got, 1.0 / 3.0), "uniform row mismatch: {got}");
        }
    }

    #[test]
    fn is_stable_for_extreme_shifted_logits() {
        let hot = run(vec![2], vec![1000.0, 1000.0]);
        assert!(close(hot[0], 0.5));
        assert!(close(hot[1], 0.5));
        let cold = run(vec![2], vec![-1000.0, -1000.0]);
        assert!(close(cold[0], 0.5));
        assert!(close(cold[1], 0.5));
    }

    #[test]
    fn uniform_input_gives_uniform_distribution() {
        let out = run(vec![4], vec![1.0; 4]);
        for got in &out {
            assert!(close(*got, 0.25), "uniform mismatch: {got}");
        }
    }

    #[test]
    fn rejects_zero_length_last_axis() {
        let t = build(vec![2, 0], vec![]);
        assert_eq!(
            cpusoftmax.apply(&t.getview()).map(|_| ()),
            Err(backenderror::unsupportedoperation { op: "softmax" })
        );
    }

    #[test]
    fn empty_rows_produce_empty_output() {
        let t = build(vec![0, 3], vec![]);
        let out = cpusoftmax.apply(&t.getview()).unwrap();
        assert!(out.isempty());
        assert_eq!(out.getshape().getdims(), &[0, 3]);
    }

    #[test]
    fn reports_stable_name() {
        assert_eq!(kernel::<f32>::getname(&cpusoftmax), "softmax");
    }

    proptest! {
        #[test]
        fn rows_form_a_probability_distribution(
            cols in 1usize..=16,
            rows in 1usize..=8,
            raw in collection::vec(-80.0f32..80.0, 1..=128),
        ) {
            let len = rows * cols;
            let data: Vec<f32> = (0..len).map(|i| raw[i % raw.len()]).collect();
            let out = run(vec![rows, cols], data);
            for row in out.chunks(cols) {
                let sum: f32 = row.iter().sum();
                prop_assert!((sum - 1.0).abs() <= 1.0e-5, "row sum was {sum}");
                prop_assert!(row.iter().all(|&x| (0.0..=1.0).contains(&x)));
            }
        }

        #[test]
        fn shifting_logits_preserves_output(
            shift in -50.0f32..50.0,
            data in collection::vec(-30.0f32..30.0, 1..=64),
        ) {
            let shifted: Vec<f32> = data.iter().map(|x| x + shift).collect();
            let base = run(vec![data.len()], data);
            let moved = run(vec![shifted.len()], shifted);
            for (a, b) in base.iter().zip(moved.iter()) {
                prop_assert!((a - b).abs() <= 1.0e-4, "{a} vs {b}");
            }
        }
    }
}
