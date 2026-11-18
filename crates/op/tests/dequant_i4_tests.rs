use super::*;

crate::dequant::test_support::dequant_packed_tests!(
    sample,
    reference,
    forward,
    SCALE,
    ZERO,
    2,
    16.0,
    -1.0,
    f32::INFINITY
);
