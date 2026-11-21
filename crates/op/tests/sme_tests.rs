use super::super::tests::check_gemm;
use super::gemm;

/// 直接调用 SME 内核（越过 `gemm_f32` 的分派），用刚过规模门槛的小形状覆盖：
/// 多行块 × 多列组、单列组、单行块三种铺法。
fn check(m: usize, n: usize, k: usize) {
    check_gemm(
        m,
        n,
        k,
        |a, b, m, n, k, out| assert!(gemm(a, b, m, n, k, out), "SME 应接住整铺形状"),
        "SME 与标量参考容差内一致",
    );
}

#[test]
fn sme_matches_on_tile_multiples() {
    check(128, 128, 128);
}

#[test]
fn sme_matches_on_single_col_group() {
    check(128, 64, 128);
}

#[test]
fn sme_matches_on_single_row_block() {
    check(16, 64, 1024);
}
