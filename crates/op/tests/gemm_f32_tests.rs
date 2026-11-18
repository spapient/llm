use super::*;

#[test]
fn gemm_matches_the_scalar_reference() {
    gemm::matches_reference::<Dense>();
}
