/// 对拍形状矩阵：单行单列、多行单列、单行多列、常规矩阵，以及跨列块的三档。
///
/// 后三档是 tile 的接缝：`n` 恰为一整块、多一列、两块多两列，用来钉住
/// [`super::TILE_N`] 的边界收口（块宽不是序列长的倍数时最容易被漏写或重复写）。
pub(crate) const SHAPES: &[(usize, usize)] = &[
    (1, 1),
    (5, 1),
    (1, 7),
    (16, 12),
    (16, super::TILE_N),
    (16, super::TILE_N + 1),
    (3, 2 * super::TILE_N + 2),
];

/// 打包量化格式共有的三则对拍。
macro_rules! gemm_packed_tests {
    ($ty:ty) => {
        #[test]
        fn matches_the_scalar_reference() {
            crate::gemm::matches_reference::<$ty>();
        }

        #[test]
        fn matches_over_the_shape_matrix() {
            crate::gemm::matches_reference_over_shapes::<$ty>($crate::gemm::test_support::SHAPES);
        }

        #[test]
        fn matches_the_hand_closed_dot() {
            crate::gemm::hand_closed::<$ty>();
        }
    };
}

/// 稠密 f32 格式：整数闭式点积不适用，故改用 **f64 重算**的独立闭式校验。
///
/// 不沿用标量参考对拍：`f32` 的 `mul` 就是 `matmul`，而 `reference` 也走 `matmul`，
/// 两者逐位相等是构造出来的，那种对拍的证据为零。
macro_rules! gemm_dense_tests {
    ($ty:ty) => {
        #[test]
        fn matches_an_independent_f64_closure() {
            crate::gemm::dense_closed::<$ty>();
        }
    };
}

/// 前向误差界常数 `γ_n = n·u/(1−n·u)`，`u = 2^-24` 为 f32 单位舍入。
pub(crate) fn gamma(n: usize) -> f64 {
    let u = f64::from(f32::EPSILON) * 0.5;
    let nu = n as f64 * u;
    nu / (1.0 - nu)
}
pub(crate) use gemm_dense_tests;
pub(crate) use gemm_packed_tests;
