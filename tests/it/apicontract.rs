//! Integration contract tests for the frozen `api` public surface.
//!
//! These tests exercise `api` from outside the crate, including a minimal
//! `backend` implementation and a generic `kernel` implementation, to prove the
//! contracts are usable by downstream (Layer 1) crates.
//!
//! The `kernel` implementation is deliberately not invoked: building a
//! `tensorview` requires the `tensor` constructor API, which is outside the
//! frozen `api` boundary. Implementing the trait still type-checks the full
//! `apply` signature.

use ::tensor::{dtype, element, tensor, tensorview};
use api::{backend, backenderror, capabilities, device, kernel};
use lib::limits;

struct testbackend {
    caps: capabilities,
}

impl backend for testbackend {
    fn getname(&self) -> &'static str {
        "testbackend"
    }

    fn getdevice(&self) -> device {
        device::cpu()
    }

    fn getcapabilities(&self) -> &capabilities {
        &self.caps
    }
}

struct refusekernel;

impl<T: element> kernel<T> for refusekernel {
    fn getname(&self) -> &'static str {
        "refusekernel"
    }

    fn apply(&self, _input: &tensorview<'_, T>) -> Result<tensor<T>, backenderror> {
        Err(backenderror::unsupporteddtype { dtype: T::DTYPE })
    }
}

fn caps() -> capabilities {
    capabilities {
        dtypes: vec![dtype::f32, dtype::i64],
        maxrank: 3,
        limits: limits::new(16, 64, 8, 2),
    }
}

#[test]
fn backend_reports_identity_device_and_capabilities() {
    let b = testbackend { caps: caps() };
    assert_eq!(b.getname(), "testbackend");

    let d = b.getdevice();
    assert!(d.iscpu());
    assert_eq!(d.getkind(), device::cpu().getkind());

    let c = b.getcapabilities();
    assert!(c.supports(dtype::f32));
    assert!(c.supports(dtype::i64));
    assert!(!c.supports(dtype::u8));
    assert_eq!(c.getdtypes().len(), 2);
    assert_eq!(c.getmaxrank(), 3);
    assert_eq!(c.getlimits().getmaxelements(), 16);
    assert_eq!(c.getlimits().getmaxbatch(), 2);
}

#[test]
fn kernel_trait_is_generically_implementable() {
    assert_eq!(
        <refusekernel as kernel<f32>>::getname(&refusekernel),
        "refusekernel"
    );
    assert_eq!(
        <refusekernel as kernel<i32>>::getname(&refusekernel),
        "refusekernel"
    );
}

#[test]
fn device_default_kind_is_cpu() {
    assert_eq!(device::cpu().getkind(), api::devicekind::cpu);
}
