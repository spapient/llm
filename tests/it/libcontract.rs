//! Contract tests for the `lib` foundational types.
//!
//! This suite exercises `limits`, `limiterror`, `precision`, `requestid`,
//! `seed`, and `tokenid` through the crate's public interface only. It replaces
//! the former `#[cfg(test)]` unit modules so that every assertion in the
//! workspace runs through the same public boundary (see
//! `docs/engineering.md`).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use lib::limiterror;
use lib::limits;
use lib::precision;
use lib::requestid;
use lib::seed;
use lib::tokenid;

/// The bound set shared by the `limits` assertions: 8 elements, 64 bytes,
/// 16 sequence length, 4 batch.
fn lim() -> limits {
    limits::new(8, 64, 16, 4)
}

#[test]
fn getters_report_configured_bounds() {
    let l = lim();
    assert_eq!(l.getmaxelements(), 8);
    assert_eq!(l.getmaxbytes(), 64);
    assert_eq!(l.getmaxsequencelen(), 16);
    assert_eq!(l.getmaxbatch(), 4);
}

#[test]
fn element_check_accepts_bound_and_rejects_overflow() {
    let l = lim();
    assert_eq!(l.checkelements(8), Ok(()));
    assert_eq!(
        l.checkelements(9),
        Err(limiterror::maxelementsexceeded {
            requested: 9,
            limit: 8,
        })
    );
}

#[test]
fn byte_check_accepts_bound_and_rejects_overflow() {
    let l = lim();
    assert_eq!(l.checkbytes(64), Ok(()));
    assert_eq!(
        l.checkbytes(65),
        Err(limiterror::maxbytesexceeded {
            requested: 65,
            limit: 64,
        })
    );
}

#[test]
fn sequence_check_accepts_bound_and_rejects_overflow() {
    let l = lim();
    assert_eq!(l.checksequencelen(16), Ok(()));
    assert_eq!(
        l.checksequencelen(17),
        Err(limiterror::maxsequencelenexceeded {
            requested: 17,
            limit: 16,
        })
    );
}

#[test]
fn batch_check_accepts_bound_and_rejects_overflow() {
    let l = lim();
    assert_eq!(l.checkbatch(4), Ok(()));
    assert_eq!(
        l.checkbatch(5),
        Err(limiterror::maxbatchexceeded {
            requested: 5,
            limit: 4,
        })
    );
}

#[test]
fn zero_limits_accept_zero_and_reject_one() {
    let l = limits::new(0, 0, 0, 0);
    assert_eq!(l.checkelements(0), Ok(()));
    assert_eq!(
        l.checkelements(1),
        Err(limiterror::maxelementsexceeded {
            requested: 1,
            limit: 0,
        })
    );
    assert_eq!(l.checkbytes(0), Ok(()));
    assert_eq!(l.checksequencelen(0), Ok(()));
    assert_eq!(l.checkbatch(0), Ok(()));
}

#[test]
fn precision_sizeinbytes_matches_storage_width() {
    assert_eq!(precision::f32.sizeinbytes(), 4);
    assert_eq!(precision::f16.sizeinbytes(), 2);
    assert_eq!(precision::bf16.sizeinbytes(), 2);
}

#[test]
fn tokenid_new_and_get_roundtrip() {
    assert_eq!(tokenid::new(42).get(), 42);
}

#[test]
fn tokenid_boundary_values_roundtrip() {
    assert_eq!(tokenid::new(0).get(), 0);
    assert_eq!(tokenid::new(u32::MAX).get(), u32::MAX);
}

#[test]
fn seed_new_and_get_roundtrip() {
    assert_eq!(seed::new(1234).get(), 1234);
}

#[test]
fn seed_boundary_values_roundtrip() {
    assert_eq!(seed::new(0).get(), 0);
    assert_eq!(seed::new(u64::MAX).get(), u64::MAX);
}

#[test]
fn requestid_new_and_get_roundtrip() {
    assert_eq!(requestid::new(7).get(), 7);
}

#[test]
fn requestid_boundary_values_roundtrip() {
    assert_eq!(requestid::new(0).get(), 0);
    assert_eq!(requestid::new(u64::MAX).get(), u64::MAX);
}
