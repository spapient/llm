use super::*;

/// 对拍形状矩阵：单行单列、多行单列、单行多列、常规矩阵。
const SHAPES: &[(usize, usize)] = &[(1, 1), (5, 1), (1, 7), (16, 12)];

#[test]
fn gemm_i8_matches_the_scalar_reference() {
    gemm::matches_reference::<GemmI8>();
}

#[test]
fn gemm_i8_matches_over_the_shape_matrix() {
    gemm::matches_reference_over_shapes::<GemmI8>(SHAPES);
}

#[test]
fn gemm_i8_matches_the_hand_closed_dot() {
    gemm::hand_closed::<GemmI8>();
}
