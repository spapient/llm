use super::super::tests::check_gemm;
use super::gemm;

/// 直接调用 SME 内核（越过 `gemm_f32` 的分派），用刚过规模门槛的小形状覆盖：
/// 多行块 × 多列组、单列组、单行块三种铺法。三种形状都必须落在 `MR = 32` / `NR = 32`
/// 的整铺上，否则内核会拒绝而用例变成在测分派（F6）。
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
    check(128, 32, 256);
}

#[test]
fn sme_matches_on_single_row_block() {
    check(32, 32, 1024);
}

/// 多线程铺法：`t = min(线程数, m/32, n/32)` 由形状决定，三个单线程用例的 `t` 都是 1，
/// 线程化分支（`packed` 按行块切分 + 屏障同步后外积）从未被执行。256×256 给出 `t = 8`。
#[test]
fn sme_matches_when_threaded() {
    check_threaded(256, 256, 256);
}

/// 基准里 1024³ 的 `t = min(线程数, 32, 32)`，在 18 线程的机器上就是 18——线程数不再是
/// 形状的瓶颈。576 = 18×32 给出同样的整铺最细切分，确认每个线程只写自己的行块、互不重叠。
#[test]
fn sme_matches_at_full_thread_count() {
    check_threaded(576, 1024, 64);
}

/// 线程化用例的前置断言：本机必须真的切出 ≥2 个线程，否则用例名在撒谎（F6：看起来测了线程化，
/// 实际 `t = 1` 又回到单线程）。单核机器上这条会红——那是正确行为，它说明线程化分支无人覆盖。
fn check_threaded(m: usize, n: usize, k: usize) {
    let t = super::threads_for(m, n);
    assert!(t >= 2, "{m}×{n}×{k} 只切出 {t} 个线程，线程化分支未被覆盖");
    check(m, n, k);
}
