use super::*;

#[test]
fn gemm_i4_matches_the_scalar_reference() {
    gemm::matches_reference::<GemmI4>();
}
