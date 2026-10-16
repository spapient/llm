//! Contract tests for the `api` capability, device, error, and extended-kernel
//! surface.
//!
//! This suite is the black-box companion to `tests/apicontract.rs`. It exercises
//! `capabilities`, `device`, `backenderror`, and the additive
//! `binarykernel`/`weightedkernel`/`gatherkernel`/`compute` contracts through the
//! crate's public interface only. It replaces the former `#[cfg(test)]` unit
//! modules (see `docs/engineering.md`).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use ::tensor::{dtype, element, shape, tensor, tensorview};
use api::{
    backenderror, binarykernel, capabilities, compute, device, devicekind, gatherkernel, kvpair,
    weightedkernel,
};
use lib::limits;

fn caps() -> capabilities {
    capabilities {
        dtypes: vec![dtype::f32, dtype::i32],
        maxrank: 4,
        limits: limits::new(1024, 4096, 128, 8),
    }
}

#[test]
fn supports_reports_only_declared_dtypes() {
    let c = caps();
    assert!(c.supports(dtype::f32));
    assert!(c.supports(dtype::i32));
    assert!(!c.supports(dtype::f16));
    assert!(!c.supports(dtype::bf16));
}

#[test]
fn getters_report_declared_capabilities() {
    let c = caps();
    assert_eq!(c.getdtypes().len(), 2);
    assert_eq!(c.getdtypes()[0], dtype::f32);
    assert_eq!(c.getdtypes()[1], dtype::i32);
    assert_eq!(c.getmaxrank(), 4);
    assert_eq!(c.getlimits().getmaxelements(), 1024);
    assert_eq!(c.getlimits().getmaxbytes(), 4096);
    assert_eq!(c.getlimits().getmaxsequencelen(), 128);
    assert_eq!(c.getlimits().getmaxbatch(), 8);
}

#[test]
fn empty_dtype_set_supports_nothing() {
    let c = capabilities {
        dtypes: Vec::new(),
        maxrank: 0,
        limits: limits::new(0, 0, 0, 0),
    };
    assert!(!c.supports(dtype::f32));
    assert!(c.getdtypes().is_empty());
    assert_eq!(c.getmaxrank(), 0);
}

#[test]
fn cpu_constructor_sets_cpu_kind() {
    let d = device::cpu();
    assert_eq!(d.getkind(), devicekind::cpu);
    assert!(d.iscpu());
}

#[test]
fn devices_compare_by_kind() {
    assert_eq!(device::cpu(), device::cpu());
}

#[test]
fn dtype_variant_carries_dtype() {
    let e = backenderror::unsupporteddtype { dtype: dtype::f16 };
    assert_eq!(e, backenderror::unsupporteddtype { dtype: dtype::f16 });
    assert_ne!(e, backenderror::unsupporteddtype { dtype: dtype::f32 });
}

#[test]
fn operation_variant_carries_name() {
    let e = backenderror::unsupportedoperation { op: "matmul" };
    assert_eq!(e, backenderror::unsupportedoperation { op: "matmul" });
    assert_ne!(e, backenderror::unsupportedoperation { op: "conv" });
}

#[test]
fn invalidparameter_carries_op_name() {
    let e = backenderror::invalidparameter { op: "scan" };
    assert_eq!(e, backenderror::invalidparameter { op: "scan" });
    assert_ne!(e, backenderror::invalidparameter { op: "attention" });
    assert_eq!(e.to_string(), "invalid parameter for scan");
}

#[test]
fn unit_variants_are_distinct() {
    assert_eq!(backenderror::sizeoverflow, backenderror::sizeoverflow);
    assert_ne!(backenderror::sizeoverflow, backenderror::executionfailed);
}

#[test]
fn display_is_stable() {
    assert_eq!(
        backenderror::executionfailed.to_string(),
        "execution failed"
    );
    assert_eq!(backenderror::sizeoverflow.to_string(), "size overflow");
    assert_eq!(
        backenderror::unsupportedoperation { op: "conv" }.to_string(),
        "unsupported operation: conv"
    );
}

#[test]
fn shape_and_index_variants_carry_their_values() {
    let s = backenderror::shapemismatch { op: "matmul" };
    assert_eq!(s, backenderror::shapemismatch { op: "matmul" });
    assert_ne!(s, backenderror::shapemismatch { op: "add" });
    assert_eq!(s.to_string(), "incompatible operand shapes for matmul");

    let i = backenderror::indexoutofrange { index: 5, len: 3 };
    assert_eq!(i, backenderror::indexoutofrange { index: 5, len: 3 });
    assert_ne!(i, backenderror::indexoutofrange { index: 4, len: 3 });
    assert_eq!(i.to_string(), "index 5 out of range for length 3");
}

#[test]
fn resource_exhaustion_is_distinct() {
    assert_eq!(
        backenderror::resourceexhausted,
        backenderror::resourceexhausted
    );
    assert_ne!(
        backenderror::resourceexhausted,
        backenderror::executionfailed
    );
    assert_eq!(
        backenderror::resourceexhausted.to_string(),
        "resource limit exceeded"
    );
}

/// A binary kernel that refuses every element type, used only to type-check
/// the `binarykernel` signature.
struct refusebinary;

impl<T: element> binarykernel<T> for refusebinary {
    fn getname(&self) -> &'static str {
        "refusebinary"
    }

    fn apply(
        &self,
        _lhs: &tensorview<'_, T>,
        _rhs: &tensorview<'_, T>,
    ) -> Result<tensor<T>, backenderror> {
        Err(backenderror::unsupporteddtype { dtype: T::DTYPE })
    }
}

struct refuseweighted;

impl<T: element> weightedkernel<T> for refuseweighted {
    fn getname(&self) -> &'static str {
        "refuseweighted"
    }

    fn apply(
        &self,
        _input: &tensorview<'_, T>,
        _weight: &tensorview<'_, T>,
    ) -> Result<tensor<T>, backenderror> {
        Err(backenderror::unsupporteddtype { dtype: T::DTYPE })
    }
}

struct refusegather;

impl<T: element> gatherkernel<T> for refusegather {
    fn getname(&self) -> &'static str {
        "refusegather"
    }

    fn apply(
        &self,
        _table: &tensorview<'_, T>,
        _indices: &[usize],
    ) -> Result<tensor<T>, backenderror> {
        Err(backenderror::unsupporteddtype { dtype: T::DTYPE })
    }
}

/// A compute provider that refuses every operation, used to type-check the
/// `compute` contract without a concrete backend.
struct refusecompute;

impl compute for refusecompute {
    fn matmul(
        &self,
        _lhs: &tensorview<'_, f32>,
        _rhs: &tensorview<'_, f32>,
    ) -> Result<tensor<f32>, backenderror> {
        Err(backenderror::unsupportedoperation { op: "matmul" })
    }

    fn add(
        &self,
        _lhs: &tensorview<'_, f32>,
        _rhs: &tensorview<'_, f32>,
    ) -> Result<tensor<f32>, backenderror> {
        Err(backenderror::unsupportedoperation { op: "add" })
    }

    fn mul(
        &self,
        _lhs: &tensorview<'_, f32>,
        _rhs: &tensorview<'_, f32>,
    ) -> Result<tensor<f32>, backenderror> {
        Err(backenderror::unsupportedoperation { op: "mul" })
    }

    fn rmsnorm(
        &self,
        _input: &tensorview<'_, f32>,
        _weight: &tensorview<'_, f32>,
        _eps: f32,
    ) -> Result<tensor<f32>, backenderror> {
        Err(backenderror::unsupportedoperation { op: "rmsnorm" })
    }

    fn softmax(&self, _input: &tensorview<'_, f32>) -> Result<tensor<f32>, backenderror> {
        Err(backenderror::unsupportedoperation { op: "softmax" })
    }

    fn silu(&self, _input: &tensorview<'_, f32>) -> Result<tensor<f32>, backenderror> {
        Err(backenderror::unsupportedoperation { op: "silu" })
    }

    fn embedding(
        &self,
        _table: &tensorview<'_, f32>,
        _indices: &[usize],
    ) -> Result<tensor<f32>, backenderror> {
        Err(backenderror::unsupportedoperation { op: "embedding" })
    }

    fn rope(
        &self,
        _input: &tensorview<'_, f32>,
        _heads: usize,
        _base: f32,
    ) -> Result<tensor<f32>, backenderror> {
        Err(backenderror::unsupportedoperation { op: "rope" })
    }

    fn rope_at(
        &self,
        _input: &tensorview<'_, f32>,
        _position: usize,
        _heads: usize,
        _base: f32,
    ) -> Result<tensor<f32>, backenderror> {
        Err(backenderror::unsupportedoperation { op: "rope_at" })
    }

    fn attention(
        &self,
        _query: &tensorview<'_, f32>,
        _key: &tensorview<'_, f32>,
        _value: &tensorview<'_, f32>,
        _heads: usize,
        _kvheads: usize,
        _causal: bool,
    ) -> Result<tensor<f32>, backenderror> {
        Err(backenderror::unsupportedoperation { op: "attention" })
    }

    fn attention_cached(
        &self,
        _query: &tensorview<'_, f32>,
        _key: kvpair<'_>,
        _value: kvpair<'_>,
        _heads: usize,
        _kvheads: usize,
        _causal: bool,
    ) -> Result<tensor<f32>, backenderror> {
        Err(backenderror::unsupportedoperation {
            op: "attention_cached",
        })
    }

    fn scan(
        &self,
        _input: &tensorview<'_, f32>,
        _init: &tensorview<'_, f32>,
        _a: &tensorview<'_, f32>,
        _b: &tensorview<'_, f32>,
        _c: &tensorview<'_, f32>,
        _skip: &tensorview<'_, f32>,
    ) -> Result<(tensor<f32>, tensor<f32>), backenderror> {
        Err(backenderror::unsupportedoperation { op: "scan" })
    }

    fn scan_cached(
        &self,
        _input: &tensorview<'_, f32>,
        _state: &tensorview<'_, f32>,
        _a: &tensorview<'_, f32>,
        _b: &tensorview<'_, f32>,
        _c: &tensorview<'_, f32>,
        _skip: &tensorview<'_, f32>,
    ) -> Result<(tensor<f32>, tensor<f32>), backenderror> {
        Err(backenderror::unsupportedoperation { op: "scan_cached" })
    }
}

#[test]
fn binarykernel_is_generically_implementable() {
    assert_eq!(
        <refusebinary as binarykernel<f32>>::getname(&refusebinary),
        "refusebinary"
    );
    assert_eq!(
        <refusebinary as binarykernel<i64>>::getname(&refusebinary),
        "refusebinary"
    );
}

#[test]
fn weighted_and_gather_kernels_are_generically_implementable() {
    assert_eq!(
        <refuseweighted as weightedkernel<f32>>::getname(&refuseweighted),
        "refuseweighted"
    );
    assert_eq!(
        <refusegather as gatherkernel<f32>>::getname(&refusegather),
        "refusegather"
    );
}

#[test]
fn compute_contract_reports_unsupported_operations() {
    let t = tensor::new(shape::new(vec![2, 2]).unwrap(), vec![0.0_f32; 4]).unwrap();
    let v = t.getview();
    assert_eq!(
        refusecompute.matmul(&v, &v).map(|_| ()),
        Err(backenderror::unsupportedoperation { op: "matmul" })
    );
    assert_eq!(
        refusecompute.silu(&v).map(|_| ()),
        Err(backenderror::unsupportedoperation { op: "silu" })
    );
    assert_eq!(
        refusecompute.attention(&v, &v, &v, 1, 1, true).map(|_| ()),
        Err(backenderror::unsupportedoperation { op: "attention" })
    );
    assert_eq!(
        refusecompute.scan(&v, &v, &v, &v, &v, &v).map(|_| ()),
        Err(backenderror::unsupportedoperation { op: "scan" })
    );
    assert_eq!(
        refusecompute
            .scan_cached(&v, &v, &v, &v, &v, &v)
            .map(|_| ()),
        Err(backenderror::unsupportedoperation { op: "scan_cached" })
    );
}

#[test]
fn unsupported_dtype_variant_uses_element_constant() {
    let t = tensor::new(shape::new(vec![1]).unwrap(), vec![1.0_f32]).unwrap();
    assert_eq!(
        refusebinary.apply(&t.getview(), &t.getview()).map(|_| ()),
        Err(backenderror::unsupporteddtype { dtype: dtype::f32 })
    );
}
