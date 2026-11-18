use super::*;

#[test]
fn gemm_matches_the_scalar_reference() {
    crate::gemm::matches_reference::<Dense>();
}
