//! Tolerance-based comparison helpers for numerical tests.
//!
//! These helpers panic on mismatch. That is intentional: they are test
//! infrastructure, where failing fast with a descriptive message is the correct
//! behavior. They are never used on a production runtime path.

/// Asserts that `actual` and `expected` agree within a combined absolute and
/// relative tolerance.
///
/// The comparison passes when, for every index `i`,
///
/// ```text
/// |actual[i] - expected[i]| <= abstol + reltol * |expected[i]|
/// ```
///
/// The relative term is scaled by `|expected[i]|`, so values near zero are
/// governed by `abstol` and large values by `reltol`. Both tolerances must be
/// finite and non-negative; a non-finite or negative tolerance is rejected.
///
/// # Panics
///
/// Panics if the lengths differ, if a tolerance is negative or non-finite, or
/// on the first value that exceeds its tolerance. The panic message names the
/// index, both values, and the tolerance that was exceeded.
pub fn assertallclose(actual: &[f32], expected: &[f32], abstol: f32, reltol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "length mismatch: actual {} vs expected {}",
        actual.len(),
        expected.len()
    );
    assert!(
        abstol.is_finite() && abstol >= 0.0,
        "abstol must be finite and non-negative, got {abstol}"
    );
    assert!(
        reltol.is_finite() && reltol >= 0.0,
        "reltol must be finite and non-negative, got {reltol}"
    );
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let tol = abstol + reltol * e.abs();
        assert!(
            (a - e).abs() <= tol,
            "mismatch at index {i}: actual {a} expected {e} tolerance {tol}"
        );
    }
}
