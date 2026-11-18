use super::*;

crate::dequant::test_support::dequant_packed_tests!(
    sample,
    reference,
    forward,
    SCALE,
    ZERO,
    4,
    4.0,
    -0.5,
    f32::NAN
);
