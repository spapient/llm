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
use rstest::rstest;

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

/// Each case pairs one bounded check with its ceiling: the value exactly at the
/// ceiling is accepted, and the next value is rejected with the field-specific
/// error. Every check keeps its own case name, so a failure names the field.
#[rstest]
#[case::elements(
    limits::checkelements,
    8,
    9,
    limiterror::maxelementsexceeded { requested: 9, limit: 8 }
)]
#[case::bytes(
    limits::checkbytes,
    64,
    65,
    limiterror::maxbytesexceeded { requested: 65, limit: 64 }
)]
#[case::sequence(
    limits::checksequencelen,
    16,
    17,
    limiterror::maxsequencelenexceeded { requested: 17, limit: 16 }
)]
#[case::batch(
    limits::checkbatch,
    4,
    5,
    limiterror::maxbatchexceeded { requested: 5, limit: 4 }
)]
fn checks_accept_the_ceiling_and_reject_above(
    #[case] check: fn(&limits, usize) -> Result<(), limiterror>,
    #[case] ceiling: usize,
    #[case] over: usize,
    #[case] expected: limiterror,
) {
    let l = lim();
    assert_eq!(check(&l, ceiling), Ok(()));
    assert_eq!(check(&l, over), Err(expected));
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

/// Every id newtype returns exactly the value it was constructed with,
/// including the zero and maximum boundary values. Each assertion is
/// independent, so a failure points at the specific constructor and value.
#[test]
fn id_newtypes_roundtrip_at_boundaries() {
    assert_eq!(tokenid::new(42).get(), 42);
    assert_eq!(seed::new(1234).get(), 1234);
    assert_eq!(requestid::new(7).get(), 7);
    assert_eq!(tokenid::new(0).get(), 0);
    assert_eq!(tokenid::new(u32::MAX).get(), u32::MAX);
    assert_eq!(seed::new(0).get(), 0);
    assert_eq!(seed::new(u64::MAX).get(), u64::MAX);
    assert_eq!(requestid::new(0).get(), 0);
    assert_eq!(requestid::new(u64::MAX).get(), u64::MAX);
}
