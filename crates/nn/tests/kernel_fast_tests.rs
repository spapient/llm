use super::*;
use crate::matmul::matmul_scalar;

fn sample_i8(len: usize, salt: u32) -> Vec<i8> {
    (0..len)
        .map(|i| {
            let x = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(salt.wrapping_mul(40503));
            ((x % 251) as i32 - 125) as i8
        })
        .collect()
}

fn check_i8(m: usize, n: usize, k: usize) {
    let a = sample_i8(m * k, 1);
    let b = sample_i8(k * n, 2);
    let mut want = vec![0i32; m * n];
    crate::matmul_i8::forward_tiled(&a, &b, m, n, k, &mut want).unwrap();
    let mut got = vec![0i32; m * n];
    gemm_i8(&a, &b, m, n, k, &mut got);
    assert_eq!(got, want, "gemm_i8 与参考逐元素相等（整数无舍入）");
}

#[test]
fn i8_matches_on_panel_multiples() {
    check_i8(64, 64, 96);
}

#[test]
fn i8_matches_on_ragged_tails() {
    check_i8(37, 19, 33);
}

#[test]
fn i8_matches_on_prefill_shape() {
    check_i8(256, 256, 4096);
}

#[test]
fn i8_concurrent_calls_are_isolated() {
    // 多槽池：并发调用各租一组空闲槽，必须互不干扰且不挂死。
    let shapes = [(64usize, 64usize, 96usize), (80, 32, 128), (96, 48, 64)];
    thread::scope(|s| {
        for (m, n, k) in shapes {
            let a = sample_i8(m * k, 1);
            let b = sample_i8(k * n, 2);
            let mut want = vec![0i32; m * n];
            crate::matmul_i8::forward_tiled(&a, &b, m, n, k, &mut want).unwrap();
            s.spawn(move || {
                for _ in 0..200 {
                    let mut got = vec![0i32; m * n];
                    gemm_i8(&a, &b, m, n, k, &mut got);
                    assert_eq!(got, want);
                }
            });
        }
    });
}

pub(super) fn sample(len: usize, salt: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(salt.wrapping_mul(40503));
            ((x >> 8) as f32) / 256.0 - 32.0
        })
        .collect()
}

/// 生成确定性输入、跑标量参考、调用 `compute` 后与参考对拍。
/// `sme` 测试复用同一骨架，仅替换计算入口与断言文案。
///
/// 断言的不是"容差内一致"，而是**可证的舍入上界**。重排求和改变的不是数学结果，
/// 只是每一部分和的舍入；Higham《Accuracy and Stability》式 (3.3) 给出
/// `|fl(Σ aᵢbᵢ) − Σ aᵢbᵢ| ≤ γ_k · Σ|aᵢbᵢ|`，`γ_n = n·u/(1−n·u)`，`u = 2⁻²⁴`。
/// 该界与求和树形状无关（任何 n 叶二叉树的深度 ≤ n−1），所以分块、多累加链、
/// FMA 融合都落在同一个界内——这正是"路径自由"的形式化说法。
pub(super) fn check_gemm<F>(m: usize, n: usize, k: usize, compute: F, msg: &str)
where
    F: FnOnce(&[f32], &[f32], usize, usize, usize, &mut [f32]),
{
    let a = sample(m * k, 1);
    let b = sample(k * n, 2);
    let mut want = vec![0.0f32; m * n];
    matmul_scalar(&a, &b, m, n, k, &mut want);
    let mut got = vec![0.0f32; m * n];
    compute(&a, &b, m, n, k, &mut got);
    for row in 0..m {
        for col in 0..n {
            let slack = f64::from(got[row * n + col]) - f64::from(want[row * n + col]);
            let scale: f64 = (0..k)
                .map(|i| f64::from(a[row * k + i].abs()) * f64::from(b[i * n + col].abs()))
                .sum();
            let bound = gamma(k) * scale;
            assert!(
                slack.abs() <= bound,
                "{msg}：[{row},{col}] 偏差 {slack} 超过 γ_{k}·Σ|aᵢbᵢ| = {bound}（重排求和的舍入上界）"
            );
        }
    }
}

/// `γ_n = n·u/(1 − n·u)`，f32 就近舍入取 `u = 2⁻²⁴`。
fn gamma(n: usize) -> f64 {
    let nu = n as f64 * f64::from(2.0f32.powi(-24));
    nu / (1.0 - nu)
}

fn check_fast_gemm(m: usize, n: usize, k: usize) {
    check_gemm(m, n, k, gemm_f32, "fast gemm 与标量参考容差内一致");
}

#[test]
fn matches_scalar_on_tile_multiples() {
    check_fast_gemm(64, 64, 96);
}

#[test]
fn matches_scalar_on_ragged_edges() {
    check_fast_gemm(100, 70, 4093);
}

#[test]
fn matches_scalar_trivial() {
    check_fast_gemm(1, 1, 1);
    check_fast_gemm(7, 3, 5);
}

#[test]
fn matches_scalar_on_prefill_shape() {
    check_fast_gemm(256, 256, 4096);
}

/// 吞吐基线：`cargo test --release -p llm-op --lib fast::bench_throughput -- --ignored --nocapture`
#[test]
#[ignore = "吞吐基线，手动跑"]
fn bench_throughput() {
    use std::time::Instant;
    let shapes = [
        (256usize, 256usize, 4096usize),
        (512, 512, 512),
        (1024, 1024, 1024),
    ];
    println!(
        "threads={}",
        std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(1)
    );
    let g = |ns: f64, m: usize, n: usize, k: usize| 2.0 * m as f64 * n as f64 * k as f64 / ns;
    for (m, n, k) in shapes {
        let a = sample_i8(m * k, 1);
        let b = sample_i8(k * n, 2);
        let mut out = vec![0i32; m * n];
        gemm_i8(&a, &b, m, n, k, &mut out);
        let mut best = f64::INFINITY;
        for _ in 0..20 {
            let t = Instant::now();
            gemm_i8(&a, &b, m, n, k, &mut out);
            best = best.min(t.elapsed().as_nanos() as f64);
        }
        let mt = g(best, m, n, k);
        gemm_i8_single(&a, &b, m, n, k, &mut out);
        let mut best1 = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            gemm_i8_single(&a, &b, m, n, k, &mut out);
            best1 = best1.min(t.elapsed().as_nanos() as f64);
        }
        println!(
            "  i8 {m}x{n}x{k}: fast {mt:.1} GF/s | fast单 {:.1} GF/s | 单/多 {:.2}",
            g(best1, m, n, k),
            g(best1, m, n, k) / mt
        );
    }
    for (m, n, k) in shapes {
        let a: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| ((i as f32) * 0.007).cos()).collect();
        let mut out = vec![0.0f32; m * n];
        gemm_f32(&a, &b, m, n, k, &mut out);
        let mut best = f64::INFINITY;
        for _ in 0..20 {
            let t = Instant::now();
            gemm_f32(&a, &b, m, n, k, &mut out);
            best = best.min(t.elapsed().as_nanos() as f64);
        }
        let mt = g(best, m, n, k);
        gemm_f32_single(&a, &b, m, n, k, &mut out);
        let mut best1 = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            gemm_f32_single(&a, &b, m, n, k, &mut out);
            best1 = best1.min(t.elapsed().as_nanos() as f64);
        }
        println!(
            "  f32 {m}x{n}x{k}: fast {mt:.1} GF/s | fast单 {:.1} GF/s | 单/多 {:.2}",
            g(best1, m, n, k),
            g(best1, m, n, k) / mt
        );
    }
}

/// 探针清单是「能力有没有」的唯一入口：条目随工具链变，但每条都必须有正数分子。
///
/// 这条测试盯的是**不泄漏 `cfg`**：调用方（测量容器）拿到的是一张清单，不是
/// `cfg(nightly)`——后者是 crate 私有 cfg，谁写谁报 `unexpected_cfgs`。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
#[test]
fn probe_list_matches_the_toolchain() {
    let probes = probes();
    assert!(probes.iter().all(|p| p.ops_per_round > 0), "{probes:?}");
    #[cfg(nightly)]
    assert!(probes.iter().any(|p| p.group == "sme_peak"));
    #[cfg(not(nightly))]
    assert!(probes.iter().all(|p| p.group != "sme_peak"));
    assert!(probes.iter().any(|p| p.group == "i8_peak"));
}

use super::with_panels;

/// 同一线程两次借用拿到同一块地址：复用生效，且旧内容不会越界泄漏到新长度之外。
#[test]
fn a_second_borrow_reuses_the_same_buf() {
    let first = with_panels(1024, |buf| buf.as_ptr() as usize);
    let second = with_panels(512, |buf| (buf.as_ptr() as usize, buf.len()));
    assert_eq!(first, second.0);
    assert_eq!(second.1, 512);
}

/// 扩容时新增段清零：面板补零列依赖这一点。
#[test]
fn growing_the_buf_clears_the_new_tail() {
    with_panels(16, |buf| buf.fill(1.0));
    with_panels(64, |buf| {
        assert!(buf[..16].iter().all(|v| *v == 1.0) || buf[..16].iter().all(|v| *v == 0.0));
        assert!(buf[16..].iter().all(|v| *v == 0.0), "新增段必须清零");
    });
}

/// 重入：借用期间再借一次不 panic，内层拿到的是另一块。
#[test]
fn a_nested_borrow_falls_back_to_a_temporary() {
    let outer = with_panels(8, |buf| {
        buf.fill(3.0);
        let inner = with_panels(8, |inner| (inner.as_ptr() as usize, inner.to_vec()));
        assert_ne!(buf.as_ptr() as usize, inner.0, "重入必须换一块");
        assert!(inner.1.iter().all(|v| *v == 0.0));
        buf[0]
    });
    assert_eq!(outer, 3.0);
}
