//! Contract tests for the `musa` GPU backend, exercised only through its
//! public API from outside the crate.
//!
//! Linux only: on every other target `musa` compiles to an empty library, so
//! the whole suite is compiled out there.
//!
//! # Environment assumptions
//!
//! The numerical suites require a Moore Threads MUSA device and driver
//! (`mthreads-gmi`). Per docs/engineering.md, a test that cannot run in its
//! environment must be reported as *not executed*, never as passed:
//!
//! - The device probe is `mthreads-gmi`. When it is absent or reports no
//!   device, the suite prints `E4 未执行` and returns without asserting the
//!   numerical result. That printed line is the record; the harness still
//!   reports the test as `ok`.
//! - Setting `LLM_MUSA_REQUIRE_DEVICE=1` turns the same condition into a hard
//!   failure, so a missing device can never be mistaken for a passing
//!   numerical check.
//!
//! # What runs without a device
//!
//! Every check that the backend settles before it touches a device runs
//! unconditionally: identity and capability declarations, the rank and element
//! budgets, and the MUSA-specific head-dimension limit. These are pure
//! host-side validation and hold on a machine with no GPU.
#![cfg(target_os = "linux")]
#![allow(clippy::unwrap_used, clippy::expect_used)]

use ::tensor::dtype;
use api::backend;
use lib::limits;
use musa::{MUSAMAXHEADDIM, MUSAMAXRANK, musabackend};

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
