use super::*;

#[test]
fn gemm_i8_matches_the_scalar_reference() {
    gemm::matches_reference::<GemmI8>();
}
