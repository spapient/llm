//! FMA / sdot 微内核的算术本体：scalar 跨平台参考与 fallback，NEON/dotprod、AVX2/FMA
//! 向量化实现。校验、缓冲几何与特性探测由上层安全桥负责，本文件只收纯计算与 unsafe
//! intrinsic 调用，不触碰切片所有权之外的索引。
#![allow(unsafe_code)]

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(all(feature = "bench", target_arch = "aarch64"))]
use super::gemm_micro::SDOT_CHAINS;
use super::gemm_micro::{FMA_MR, FMA_NR};
#[cfg(target_arch = "aarch64")]
use super::gemm_micro::{SDOT_MR, SDOT_NR};

/// scalar f32 瓦片：x86 无 AVX2 与其他架构的生产 fallback，aarch64 仅作对拍 oracle。
#[cfg(any(test, not(target_arch = "aarch64")))]
#[allow(clippy::too_many_arguments)]
pub(super) fn fma_scalar(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    row: usize,
    col: usize,
    n: usize,
    kb: usize,
    first: bool,
) {
    for r in 0..FMA_MR {
        for c in 0..FMA_NR {
            let slot = (row + r) * n + col + c;
            let mut acc = if first { 0.0 } else { out[slot] };
            for p in 0..kb {
                acc += a[p * FMA_MR + r] * b[p * FMA_NR + c];
            }
            out[slot] = acc;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fma_tile_neon(acc: &mut [float32x4_t; 16], a: *const f32, b: *const f32, kb: usize) {
    unsafe {
        for p in 0..kb {
            let av0 = vld1q_f32(a.add(p * FMA_MR));
            let av1 = vld1q_f32(a.add(p * FMA_MR + 4));
            let bv0 = vld1q_f32(b.add(p * FMA_NR));
            let bv1 = vld1q_f32(b.add(p * FMA_NR + 4));
            acc[0] = vfmaq_laneq_f32::<0>(acc[0], bv0, av0);
            acc[1] = vfmaq_laneq_f32::<0>(acc[1], bv1, av0);
            acc[2] = vfmaq_laneq_f32::<1>(acc[2], bv0, av0);
            acc[3] = vfmaq_laneq_f32::<1>(acc[3], bv1, av0);
            acc[4] = vfmaq_laneq_f32::<2>(acc[4], bv0, av0);
            acc[5] = vfmaq_laneq_f32::<2>(acc[5], bv1, av0);
            acc[6] = vfmaq_laneq_f32::<3>(acc[6], bv0, av0);
            acc[7] = vfmaq_laneq_f32::<3>(acc[7], bv1, av0);
            acc[8] = vfmaq_laneq_f32::<0>(acc[8], bv0, av1);
            acc[9] = vfmaq_laneq_f32::<0>(acc[9], bv1, av1);
            acc[10] = vfmaq_laneq_f32::<1>(acc[10], bv0, av1);
            acc[11] = vfmaq_laneq_f32::<1>(acc[11], bv1, av1);
            acc[12] = vfmaq_laneq_f32::<2>(acc[12], bv0, av1);
            acc[13] = vfmaq_laneq_f32::<2>(acc[13], bv1, av1);
            acc[14] = vfmaq_laneq_f32::<3>(acc[14], bv0, av1);
            acc[15] = vfmaq_laneq_f32::<3>(acc[15], bv1, av1);
        }
    }
}

/// NEON f32 瓦片入口：`first` 从零累加，否则读回输出续算，结果写回瓦片。
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn fma_neon(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    row: usize,
    col: usize,
    n: usize,
    kb: usize,
    first: bool,
) {
    unsafe {
        let mut acc = [vdupq_n_f32(0.0); 16];
        if !first {
            for (r, lane) in acc.iter_mut().enumerate() {
                let slot = (row + r / 2) * n + col + (r % 2) * 4;
                *lane = vld1q_f32(out.as_ptr().add(slot));
            }
        }
        fma_tile_neon(&mut acc, a.as_ptr(), b.as_ptr(), kb);
        for (r, lane) in acc.iter().enumerate() {
            let slot = (row + r / 2) * n + col + (r % 2) * 4;
            vst1q_f32(out.as_mut_ptr().add(slot), *lane);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_load_acc(
    out: *const f32,
    row: usize,
    col: usize,
    n: usize,
    acc: &mut [__m256; FMA_MR],
) {
    unsafe {
        for (r, lane) in acc.iter_mut().enumerate() {
            *lane = _mm256_loadu_ps(out.add((row + r) * n + col));
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_fma_block(a: *const f32, b: *const f32, p: usize, acc: &mut [__m256; FMA_MR]) {
    unsafe {
        let bv = _mm256_loadu_ps(b.add(p * FMA_NR));
        for (r, lane) in acc.iter_mut().enumerate() {
            let broadcast = _mm256_set1_ps(*a.add(p * FMA_MR + r));
            *lane = _mm256_fmadd_ps(broadcast, bv, *lane);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn avx2_store_acc(out: *mut f32, row: usize, col: usize, n: usize, acc: &[__m256; FMA_MR]) {
    unsafe {
        for (r, lane) in acc.iter().enumerate() {
            _mm256_storeu_ps(out.add((row + r) * n + col), *lane);
        }
    }
}

/// AVX2/FMA f32 瓦片入口：行广播乘加，输出按行连续布局。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn fma_avx2(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    row: usize,
    col: usize,
    n: usize,
    kb: usize,
    first: bool,
) {
    unsafe {
        let mut acc = [_mm256_setzero_ps(); FMA_MR];
        if !first {
            avx2_load_acc(out.as_ptr(), row, col, n, &mut acc);
        }
        for p in 0..kb {
            avx2_fma_block(a.as_ptr(), b.as_ptr(), p, &mut acc);
        }
        avx2_store_acc(out.as_mut_ptr(), row, col, n, &acc);
    }
}

/// scalar i8 sdot 参考实现：仅用于 dotprod 瓦片对拍；无 dotprod 平台由上层标量兜底。
#[cfg(all(test, target_arch = "aarch64"))]
fn sdot_scalar(
    a: &[i8],
    bpack: &[i8],
    out: &mut [i32],
    row: usize,
    blk: usize,
    n: usize,
    k: usize,
) {
    for r in 0..SDOT_MR {
        for nb in 0..SDOT_NR {
            let jj = blk + nb;
            if jj >= n {
                continue;
            }
            let mut sum = 0i32;
            for p in 0..k {
                sum += (a[(row + r) * k + p] as i32) * (bpack[jj * k + p] as i32);
            }
            out[(row + r) * n + jj] = sum;
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dotprod_zero(acc: &mut [[int32x4_t; SDOT_MR]; SDOT_NR]) {
    for row in acc.iter_mut() {
        for lane in row.iter_mut() {
            *lane = vdupq_n_s32(0);
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn dotprod_vdot_block(
    ap: *const i8,
    bp: *const i8,
    p: usize,
    k: usize,
    acc: &mut [[int32x4_t; SDOT_MR]; SDOT_NR],
) {
    unsafe {
        let av = [
            vld1q_s8(ap.add(p)),
            vld1q_s8(ap.add(k + p)),
            vld1q_s8(ap.add(2 * k + p)),
            vld1q_s8(ap.add(3 * k + p)),
        ];
        for (nb, row) in acc.iter_mut().enumerate() {
            let bv = vld1q_s8(bp.add(nb * k + p));
            for (r, lane) in row.iter_mut().enumerate() {
                *lane = vdotq_s32(*lane, av[r], bv);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
unsafe fn sdot_tile_dotprod(
    ap: *const i8,
    bp: *const i8,
    k: usize,
    kvec: usize,
    acc: &mut [[int32x4_t; SDOT_MR]; SDOT_NR],
) {
    unsafe {
        dotprod_zero(acc);
        let mut p = 0;
        while p < kvec {
            dotprod_vdot_block(ap, bp, p, k, acc);
            p += 16;
        }
    }
}

/// dotprod sdot 瓦片入口：16 字节段向量化累乘，横向求和后接标量尾部段。
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "dotprod")]
#[allow(clippy::too_many_arguments)]
pub(super) unsafe fn sdot_dotprod(
    a: &[i8],
    bpack: &[i8],
    out: &mut [i32],
    row: usize,
    blk: usize,
    n: usize,
    k: usize,
    kvec: usize,
) {
    unsafe {
        let ap = a.as_ptr().add(row * k);
        let mut acc = [[vdupq_n_s32(0); SDOT_MR]; SDOT_NR];
        sdot_tile_dotprod(ap, bpack.as_ptr().add(blk * k), k, kvec, &mut acc);
        for (nb, lane_row) in acc.iter().enumerate() {
            let jj = blk + nb;
            if jj >= n {
                continue;
            }
            for (r, lane) in lane_row.iter().enumerate() {
                let mut sum = vaddvq_s32(*lane);
                let mut p = kvec;
                while p < k {
                    let av = *ap.add(r * k + p) as i32;
                    let bv = *bpack.as_ptr().add(jj * k + p) as i32;
                    sum += av * bv;
                    p += 1;
                }
                out[(row + r) * n + jj] = sum;
            }
        }
    }
}

/// FMA 纯端口探针：环内无内存访问，acc 跨轮累加后经 black_box 阻止死代码消除。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
pub(super) unsafe fn fma_pure_probe(rounds: usize) {
    let mut acc = [vdupq_n_f32(0.0); 16];
    let av = vdupq_n_f32(1.000_000_1f32);
    for _ in 0..rounds {
        for acc in acc.iter_mut() {
            *acc = vfmaq_f32(*acc, av, av);
        }
    }
    for acc in acc.iter_mut() {
        core::hint::black_box(acc);
    }
}

/// FMA 取数探针：acc 跨轮复用同一瓦片，与生产 NEON 瓦片内核一致。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
#[target_feature(enable = "neon")]
pub(super) unsafe fn fma_block_probe(rounds: usize, a: *const f32, b: *const f32, k: usize) {
    unsafe {
        let mut acc = [vdupq_n_f32(0.0); 16];
        for _ in 0..rounds {
            fma_tile_neon(&mut acc, a, b, k);
        }
        for acc in acc.iter_mut() {
            core::hint::black_box(acc);
        }
    }
}

/// sdot 纯端口探针：固定 3×5 链上累乘，acc 跨轮累加后经 black_box 阻止消除。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn sdot_pure_probe(rounds: usize) {
    let a = vdupq_n_s8(3);
    let b = vdupq_n_s8(5);
    let mut acc = [vdupq_n_s32(0); SDOT_CHAINS];
    for _ in 0..rounds {
        for acc in acc.iter_mut() {
            *acc = vdotq_s32(*acc, a, b);
        }
    }
    for acc in acc.iter_mut() {
        core::hint::black_box(acc);
    }
}

/// sdot 取数探针：每轮独立瓦片（tile 内部清零），与生产 dotprod 瓦片内核一致。
#[cfg(all(feature = "bench", target_arch = "aarch64"))]
#[target_feature(enable = "dotprod")]
pub(super) unsafe fn sdot_block_probe(rounds: usize, a: *const i8, b: *const i8, k: usize) {
    unsafe {
        let mut acc = [[vdupq_n_s32(0); SDOT_MR]; SDOT_NR];
        for _ in 0..rounds {
            sdot_tile_dotprod(a, b, k, k, &mut acc);
        }
        for row in acc.iter_mut() {
            for acc in row.iter_mut() {
                core::hint::black_box(acc);
            }
        }
    }
}

#[cfg(test)]
fn fma_fixture(k: usize) -> (Vec<f32>, Vec<f32>) {
    let a = (0..FMA_MR * k)
        .map(|i| (i % 7) as f32 * 0.25 - 0.5)
        .collect();
    let b = (0..FMA_NR * k)
        .map(|i| (i % 5) as f32 * 0.5 - 1.0)
        .collect();
    (a, b)
}

#[cfg(test)]
fn assert_fma_matches(scalar: &[f32], accelerated: &[f32]) {
    for (scalar, accelerated) in scalar.iter().zip(accelerated) {
        let tolerance = 1e-4 * scalar.abs().max(1.0);
        assert!(
            (scalar - accelerated).abs() <= tolerance,
            "scalar={scalar} accelerated={accelerated}"
        );
    }
}

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::{FMA_MR, FMA_NR, SDOT_MR, SDOT_NR};

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_fma_matches_scalar() {
        let k = 32usize;
        let (a, b) = super::fma_fixture(k);
        let mut scalar = vec![0.0f32; FMA_MR * FMA_NR];
        let mut neon = vec![0.0f32; FMA_MR * FMA_NR];
        super::fma_scalar(&a, &b, &mut scalar, 0, 0, FMA_NR, k, true);
        unsafe {
            super::fma_neon(&a, &b, &mut neon, 0, 0, FMA_NR, k, true);
        }
        super::assert_fma_matches(&scalar, &neon);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn dotprod_sdot_matches_scalar() {
        if !std::arch::is_aarch64_feature_detected!("dotprod") {
            return;
        }
        let k = 32usize;
        let a: Vec<i8> = (0..SDOT_MR * k).map(|i| (i % 13) as i8 - 6).collect();
        let b: Vec<i8> = (0..SDOT_NR * k).map(|i| (i % 11) as i8 - 5).collect();
        let mut scalar = vec![0i32; SDOT_MR * SDOT_NR];
        let mut dot = vec![0i32; SDOT_MR * SDOT_NR];
        super::sdot_scalar(&a, &b, &mut scalar, 0, 0, SDOT_NR, k);
        unsafe {
            super::sdot_dotprod(&a, &b, &mut dot, 0, 0, SDOT_NR, k, k);
        }
        assert_eq!(scalar, dot, "dotprod 与标量结果不一致");
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests_x86 {
    use super::{FMA_MR, FMA_NR, assert_fma_matches, fma_avx2, fma_fixture, fma_scalar};

    #[test]
    fn avx2_fma_matches_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2")
            || !std::arch::is_x86_feature_detected!("fma")
        {
            return;
        }
        let k = 32usize;
        let (a, b) = fma_fixture(k);
        let mut scalar = vec![0.0f32; FMA_MR * FMA_NR];
        let mut avx2 = vec![0.0f32; FMA_MR * FMA_NR];
        fma_scalar(&a, &b, &mut scalar, 0, 0, FMA_NR, k, true);
        unsafe {
            fma_avx2(&a, &b, &mut avx2, 0, 0, FMA_NR, k, true);
        }
        assert_fma_matches(&scalar, &avx2);
    }
}
