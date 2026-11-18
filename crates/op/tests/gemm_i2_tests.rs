use super::*;

#[test]
fn gemm_i2_matches_the_scalar_reference() {
    gemm::matches_reference::<GemmI2>();
}
