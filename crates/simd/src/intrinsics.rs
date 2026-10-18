//! The registered `unsafe` island of the CPU SIMD path.
//!
//! Every `#[target_feature]` kernel of the workspace's CPU backend lives here
//! and nowhere else. The dispatch functions below are safe: they take the
//! caller's already-detected [`Isa`] and cross the feature boundary once per
//! kernel invocation inside an `unsafe` block guarded by a `SAFETY:` comment.
//!
//! The kernel bodies are bit-for-bit the former `archmage`
//! `#[arcane(import_intrinsics)]` implementations with the token parameter
//! removed; the numerical invariants (fused accumulation order, lane-tree
//! reductions, degree-7 Taylor exp2) are unchanged.
#![allow(unsafe_code)]
#![allow(unsafe_op_in_unsafe_fn)]

use crate::{AttentionShape, Isa};

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

// ---------------------------------------------------------------------------
// Dispatch: safe public entry points, one per operator.
// ---------------------------------------------------------------------------

/// RMS-normalizes one `row` into `out`, scaled by `weight`, with epsilon `eps`.
pub fn rmsnormrow(isa: Isa, row: &[f32], out: &mut [f32], weight: &[f32], eps: f32) {
    match isa {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Isa::Avx2 => {
            // SAFETY: `isa` was produced by `detect()`, so the current CPU
            // implements AVX2+FMA before this arm is reachable; `out` is a
            // distinct slice from `row`/`weight`, and the kernel performs only
            // arithmetic on those buffers.
            unsafe { rmsnormrow_avx2_kernel(row, out, weight, eps) }
        }
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => {
            // SAFETY: `isa` was produced by `detect()`, so the current CPU
            // implements NEON before this arm is reachable; `out` is distinct
            // from `row`/`weight`, and the kernel performs only arithmetic.
            unsafe { rmsnormrow_neon_kernel(row, out, weight, eps) }
        }
        _ => crate::scalar::rmsnormrow(row, out, weight, eps),
    }
}

/// Computes the reciprocal RMS scale `1 / sqrt(mean(x^2) + eps)` of one row.
pub fn rowinv(isa: Isa, row: &[f32], eps: f32) -> f32 {
    match isa {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Isa::Avx2 => {
            // SAFETY: as in `rmsnormrow`; the kernel only reads `row`.
            unsafe { rowinv_avx2_kernel(row, eps) }
        }
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => {
            // SAFETY: as in `rmsnormrow`; the kernel only reads `row`.
            unsafe { rowinv_neon_kernel(row, eps) }
        }
        _ => crate::scalar::rowinv(row, eps),
    }
}

/// Softmax-normalizes rows `[start, end)` of `src` into `data`, width `width`.
pub fn softmaxrowspan(
    isa: Isa,
    src: &[f32],
    data: &mut [f32],
    width: usize,
    start: usize,
    end: usize,
) {
    match isa {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Isa::Avx2 => {
            // SAFETY: as in `rmsnormrow`; `data` is distinct from `src`.
            unsafe { softmaxrowspan_avx2_kernel(src, data, width, start, end) }
        }
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => {
            // SAFETY: as in `rmsnormrow`; `data` is distinct from `src`.
            unsafe { softmaxrowspan_neon_kernel(src, data, width, start, end) }
        }
        _ => crate::scalar::softmaxrowspan(src, data, width, start, end),
    }
}

/// Dot product of one query row and one key row.
pub fn attentiondot(isa: Isa, a: &[f32], b: &[f32], ntile: usize, len: usize) -> f32 {
    match isa {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Isa::Avx2 => {
            // SAFETY: as in `rmsnormrow`; the kernel only reads `a` and `b`.
            unsafe { attentiondot_avx2_kernel(a, b, ntile, len) }
        }
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => {
            // SAFETY: as in `rmsnormrow`; the kernel only reads `a` and `b`.
            unsafe { attentiondot_neon_kernel(a, b, ntile, len) }
        }
        _ => crate::scalar::attentiondot(a, b, ntile, len),
    }
}

/// Evaluates one query block `[qs, qe)` with online-rescaled softmax.
#[allow(clippy::too_many_arguments)]
pub fn attentionqblock(
    isa: Isa,
    plan: &AttentionShape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    part: &mut [f32],
    qs: usize,
    qe: usize,
) {
    match isa {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Isa::Avx2 => {
            // SAFETY: as in `rmsnormrow`; `part` is distinct from `q`/`k`/`v`.
            unsafe { attentionqblock_avx2_kernel(plan, q, k, v, part, qs, qe) }
        }
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => {
            // SAFETY: as in `rmsnormrow`; `part` is distinct from `q`/`k`/`v`.
            unsafe { attentionqblock_neon_kernel(plan, q, k, v, part, qs, qe) }
        }
        _ => crate::scalar::attentionqblock(plan, q, k, v, part, qs, qe),
    }
}

/// Direct register-blocked matmul micro-kernel.
#[allow(clippy::too_many_arguments)]
pub fn matmuldirect(
    isa: Isa,
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    rowscale: Option<&[f32]>,
) {
    match isa {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Isa::Avx2 => {
            // SAFETY: as in `rmsnormrow`; `out` is distinct from `a`/`b`.
            unsafe { matmuldirect_avx2_kernel(a, b, out, m, k, n, rowscale) }
        }
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => {
            // SAFETY: as in `rmsnormrow`; `out` is distinct from `a`/`b`.
            unsafe { matmuldirect_neon_kernel(a, b, out, m, k, n, rowscale) }
        }
        _ => crate::scalar::matmuldirect(a, b, out, m, k, n, rowscale),
    }
}

/// Blocked matmul micro-kernel over a contiguous packed tile.
#[allow(clippy::too_many_arguments)]
pub fn matmultile(
    isa: Isa,
    a: &[f32],
    b: &[f32],
    packed: &[f32],
    out: &mut [f32],
    i0: usize,
    rows: usize,
    jp: usize,
    panel: usize,
    kc0: usize,
    kchunk: usize,
    k: usize,
    n: usize,
    rowscale: Option<&[f32]>,
    last: bool,
) {
    match isa {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        Isa::Avx2 => {
            // SAFETY: as in `rmsnormrow`; `out` is distinct from `a`/`b`/
            // `packed`.
            unsafe {
                matmultile_avx2_kernel(
                    a, b, packed, out, i0, rows, jp, panel, kc0, kchunk, k, n, rowscale, last,
                )
            }
        }
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => {
            // SAFETY: as in `rmsnormrow`; `out` is distinct from `a`/`b`/
            // `packed`.
            unsafe {
                matmultile_neon_kernel(
                    a, b, packed, out, i0, rows, jp, panel, kc0, kchunk, k, n, rowscale, last,
                )
            }
        }
        _ => crate::scalar::matmultile(
            a, b, packed, out, i0, rows, jp, panel, kc0, kchunk, k, n, rowscale, last,
        ),
    }
}

// ---------------------------------------------------------------------------
// AVX2 kernels (x86_64 / x86), eight `f32` lanes.
// ---------------------------------------------------------------------------

/// AVX2+FMA RMS-normalization of one row.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
// SAFETY: only reachable when the caller verified AVX2+FMA via `detect()`;
// the kernel is pure arithmetic over the disjoint `row`, `out`, `weight`
// slices, with the tail handled scalar so every element is written exactly
// once.
unsafe fn rmsnormrow_avx2_kernel(row: &[f32], out: &mut [f32], weight: &[f32], eps: f32) {
    let n = row.len();
    let (chunks, tail) = row.as_chunks::<8>();
    let (wchunks, _) = weight.as_chunks::<8>();
    let mut acc = _mm256_setzero_ps();
    for x in chunks {
        let xv = _mm256_loadu_ps(x.as_ptr());
        acc = _mm256_fmadd_ps(xv, xv, acc);
    }
    let mut lanes = [0.0_f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut sumsq = lanes.iter().sum::<f32>();
    for &x in tail {
        sumsq += x * x;
    }
    let inv = 1.0 / (sumsq / n as f32 + eps).sqrt();
    let invv = _mm256_set1_ps(inv);
    let (ochunks, _) = out.as_chunks_mut::<8>();
    for (o, (&x, &w)) in ochunks.iter_mut().zip(chunks.iter().zip(wchunks.iter())) {
        let xv = _mm256_loadu_ps(x.as_ptr());
        let wv = _mm256_loadu_ps(w.as_ptr());
        _mm256_storeu_ps(o.as_mut_ptr(), _mm256_mul_ps(_mm256_mul_ps(xv, invv), wv));
    }
    for i in n - tail.len()..n {
        out[i] = row[i] * inv * weight[i];
    }
}

/// AVX2+FMA reciprocal RMS of one row.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
// SAFETY: only reachable when the caller verified AVX2+FMA via `detect()`;
// the kernel only reads `row`.
unsafe fn rowinv_avx2_kernel(row: &[f32], eps: f32) -> f32 {
    let n = row.len();
    let (chunks, tail) = row.as_chunks::<8>();
    let mut acc = _mm256_setzero_ps();
    for x in chunks {
        let xv = _mm256_loadu_ps(x.as_ptr());
        acc = _mm256_fmadd_ps(xv, xv, acc);
    }
    let mut lanes = [0.0_f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), acc);
    let mut sumsq = lanes.iter().sum::<f32>();
    for &x in tail {
        sumsq += x * x;
    }
    1.0 / (sumsq / n as f32 + eps).sqrt()
}

/// AVX2+FMA softmax over rows `[start, end)`, width `width`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
// SAFETY: only reachable when the caller verified AVX2+FMA via `detect()`;
// `data` is a distinct slice from `src`; every row is normalized with the
// same max/subtract/exp/scale sequence as the scalar reference.
unsafe fn softmaxrowspan_avx2_kernel(
    src: &[f32],
    data: &mut [f32],
    width: usize,
    start: usize,
    end: usize,
) {
    let ntile = width / 8 * 8;
    for row in start..end {
        let inrow = &src[row * width..(row + 1) * width];
        let outrow = &mut data[row * width..(row + 1) * width];
        let mut m = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut finiteall = _mm256_set1_ps(-0.0);
        let mut i = 0;
        while i < ntile {
            let mut lane = [0.0_f32; 8];
            lane.copy_from_slice(&inrow[i..i + 8]);
            let x = _mm256_loadu_ps(lane.as_ptr());
            m = _mm256_max_ps(m, x);
            finiteall = _mm256_and_ps(finiteall, _mm256_cmp_ps(x, x, _CMP_ORD_Q));
            i += 8;
        }
        let mut allfinite = _mm256_movemask_ps(finiteall) == 0xFF;
        let mut tailmax = f32::NEG_INFINITY;
        for &x in &inrow[ntile..width] {
            if !x.is_finite() {
                allfinite = false;
            }
            tailmax = f32::max(tailmax, x);
        }
        if !allfinite {
            crate::scalar::softmaxrow(inrow, outrow);
            continue;
        }
        let maxv = f32::max(hmax256_avx2_kernel(m), tailmax);
        let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
        let maxx = _mm256_set1_ps(maxv);
        let mut s = _mm256_setzero_ps();
        let mut i = 0;
        while i < ntile {
            let mut lane = [0.0_f32; 8];
            lane.copy_from_slice(&inrow[i..i + 8]);
            let x = _mm256_loadu_ps(lane.as_ptr());
            let e = exp2approx_avx2_kernel(_mm256_sub_ps(x, maxx), log2e);
            let mut el = [0.0_f32; 8];
            _mm256_storeu_ps(el.as_mut_ptr(), e);
            outrow[i..i + 8].copy_from_slice(&el);
            s = _mm256_add_ps(s, e);
            i += 8;
        }
        let mut sum = hsum256_avx2_kernel(s);
        for (d, &x) in outrow[ntile..width].iter_mut().zip(&inrow[ntile..width]) {
            let e = (x - maxv).exp();
            *d = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        let invv = _mm256_set1_ps(inv);
        let mut i = 0;
        while i < ntile {
            let mut el = [0.0_f32; 8];
            el.copy_from_slice(&outrow[i..i + 8]);
            let e = _mm256_loadu_ps(el.as_ptr());
            let mut ol = [0.0_f32; 8];
            _mm256_storeu_ps(ol.as_mut_ptr(), _mm256_mul_ps(e, invv));
            outrow[i..i + 8].copy_from_slice(&ol);
            i += 8;
        }
        for d in &mut outrow[ntile..width] {
            *d *= inv;
        }
    }
}

/// AVX2+FMA dot product of `a` and `b` over `len` elements (`ntile` vectorized
/// in ascending 8-lane chunks, tail scalar).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
// SAFETY: only reachable when the caller verified AVX2+FMA via `detect()`;
// the kernel only reads `a` and `b`, whose combined length is `len`.
unsafe fn attentiondot_avx2_kernel(a: &[f32], b: &[f32], ntile: usize, len: usize) -> f32 {
    let mut acc = _mm256_setzero_ps();
    for i in (0..ntile).step_by(8) {
        let mut al = [0.0_f32; 8];
        let mut bl = [0.0_f32; 8];
        al.copy_from_slice(&a[i..i + 8]);
        bl.copy_from_slice(&b[i..i + 8]);
        acc = _mm256_fmadd_ps(
            _mm256_loadu_ps(al.as_ptr()),
            _mm256_loadu_ps(bl.as_ptr()),
            acc,
        );
    }
    let low = _mm256_castps256_ps128(acc);
    let high = _mm256_extractf128_ps::<1>(acc);
    let q = _mm_add_ps(low, high);
    let shuf = _mm_movehdup_ps(q);
    let sums = _mm_add_ps(q, shuf);
    let shuf = _mm_movehl_ps(shuf, sums);
    let mut dot = _mm_cvtss_f32(_mm_add_ss(sums, shuf));
    for i in ntile..len {
        dot += a[i] * b[i];
    }
    dot
}

/// AVX2+FMA direct register-blocked matmul micro-kernel.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
// SAFETY: only reachable when the caller verified AVX2+FMA via `detect()`;
// `out` is distinct from `a`/`b`; row indices are bounded by `m`, `k`, `n`
// as the caller's matmul plan dictates.
unsafe fn matmuldirect_avx2_kernel(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    rowscale: Option<&[f32]>,
) {
    let ntile = n / 8 * 8;
    for i0 in (0..m).step_by(BLOCKROWS) {
        let rows = (m - i0).min(BLOCKROWS);
        for j0 in (0..ntile).step_by(8) {
            let mut acc = [_mm256_setzero_ps(); BLOCKROWS];
            for p in 0..k {
                let mut bl = [0.0_f32; 8];
                bl.copy_from_slice(&b[p * n + j0..p * n + j0 + 8]);
                let brow = _mm256_loadu_ps(bl.as_ptr());
                for r in 0..rows {
                    acc[r] = _mm256_fmadd_ps(brow, _mm256_set1_ps(a[(i0 + r) * k + p]), acc[r]);
                }
            }
            for r in 0..rows {
                let mut outl = [0.0_f32; 8];
                _mm256_storeu_ps(outl.as_mut_ptr(), acc[r]);
                if let Some(scales) = rowscale {
                    let s = scales[i0 + r];
                    for lane in outl.iter_mut() {
                        *lane *= s;
                    }
                }
                out[(i0 + r) * n + j0..(i0 + r) * n + j0 + 8].copy_from_slice(&outl);
            }
        }
        for r in 0..rows {
            for j in ntile..n {
                let v = scalardot(a, b, i0 + r, j, k, n);
                out[(i0 + r) * n + j] = match rowscale {
                    Some(scales) => v * scales[i0 + r],
                    None => v,
                };
            }
        }
    }
}

/// AVX2+FMA horizontal maximum of one 256-bit register.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
// SAFETY: as in `softmaxrowspan_avx2_kernel`; purely reads `m`.
unsafe fn hmax256_avx2_kernel(m: __m256) -> f32 {
    let low = _mm256_castps256_ps128(m);
    let high = _mm256_extractf128_ps::<1>(m);
    let q = _mm_max_ps(low, high);
    let s1 = _mm_max_ps(q, _mm_movehdup_ps(q));
    let s2 = _mm_max_ps(_mm_movehl_ps(_mm_movehdup_ps(q), s1), s1);
    _mm_cvtss_f32(s2)
}

/// AVX2+FMA horizontal sum of one 256-bit register.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
// SAFETY: as in `softmaxrowspan_avx2_kernel`; purely reads `s`.
unsafe fn hsum256_avx2_kernel(s: __m256) -> f32 {
    let low = _mm256_castps256_ps128(s);
    let high = _mm256_extractf128_ps::<1>(s);
    let q = _mm_add_ps(low, high);
    let shuf = _mm_movehdup_ps(q);
    let sums = _mm_add_ps(q, shuf);
    let shuf = _mm_movehl_ps(sums, shuf);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf))
}

/// AVX2+FMA `exp(x)` for `x <= 0`, degree-7 Taylor in `2^(x * log2e)`.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
// SAFETY: as in `softmaxrowspan_avx2_kernel`; purely reads `x` and `log2e`.
unsafe fn exp2approx_avx2_kernel(x: __m256, log2e: __m256) -> __m256 {
    let y = _mm256_mul_ps(x, log2e);
    let yi = _mm256_floor_ps(y);
    let yf = _mm256_sub_ps(y, yi);
    let mut p = _mm256_set1_ps(1.525_39e-5);
    p = _mm256_fmadd_ps(p, yf, _mm256_set1_ps(1.540_353e-4));
    p = _mm256_fmadd_ps(p, yf, _mm256_set1_ps(1.333_355_8e-3));
    p = _mm256_fmadd_ps(p, yf, _mm256_set1_ps(9.618_129e-3));
    p = _mm256_fmadd_ps(p, yf, _mm256_set1_ps(5.550_411e-2));
    p = _mm256_fmadd_ps(p, yf, _mm256_set1_ps(2.402_265e-1));
    p = _mm256_fmadd_ps(p, yf, _mm256_set1_ps(std::f32::consts::LN_2));
    p = _mm256_fmadd_ps(p, yf, _mm256_set1_ps(1.0));
    let bias = _mm256_add_ps(yi, _mm256_set1_ps(127.0));
    let bits = _mm256_cvtps_epi32(bias);
    let pos = _mm256_cmpgt_epi32(bits, _mm256_set1_epi32(0));
    let lt = _mm256_cmpgt_epi32(_mm256_set1_epi32(255), bits);
    let valid = _mm256_and_si256(pos, lt);
    let scale = _mm256_castsi256_ps(_mm256_and_si256(_mm256_slli_epi32(bits, 23), valid));
    _mm256_mul_ps(p, scale)
}

// ---------------------------------------------------------------------------
// NEON kernels (AArch64), four `f32` lanes.
// ---------------------------------------------------------------------------

/// NEON RMS-normalization of one row.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
// SAFETY: only reachable when the caller verified NEON via `detect()`; the
// kernel is pure arithmetic over the disjoint `row`, `out`, `weight` slices.
unsafe fn rmsnormrow_neon_kernel(row: &[f32], out: &mut [f32], weight: &[f32], eps: f32) {
    let n = row.len();
    let (chunks, tail) = row.as_chunks::<4>();
    let (wchunks, _) = weight.as_chunks::<4>();
    let mut acc = vdupq_n_f32(0.0);
    for x in chunks {
        let xv = vld1q_f32(x.as_ptr());
        acc = vfmaq_f32(acc, xv, xv);
    }
    let mut sumsq = vaddvq_f32(acc);
    for &x in tail {
        sumsq += x * x;
    }
    let inv = 1.0 / (sumsq / n as f32 + eps).sqrt();
    let invv = vdupq_n_f32(inv);
    let (ochunks, _) = out.as_chunks_mut::<4>();
    for (o, (&x, &w)) in ochunks.iter_mut().zip(chunks.iter().zip(wchunks.iter())) {
        let xv = vld1q_f32(x.as_ptr());
        let wv = vld1q_f32(w.as_ptr());
        vst1q_f32(o.as_mut_ptr(), vmulq_f32(vmulq_f32(xv, invv), wv));
    }
    for i in n - tail.len()..n {
        out[i] = row[i] * inv * weight[i];
    }
}

/// NEON reciprocal RMS of one row.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
// SAFETY: only reachable when the caller verified NEON via `detect()`;
// the kernel only reads `row`.
unsafe fn rowinv_neon_kernel(row: &[f32], eps: f32) -> f32 {
    let n = row.len();
    let (chunks, tail) = row.as_chunks::<4>();
    let mut acc = vdupq_n_f32(0.0);
    for x in chunks {
        let xv = vld1q_f32(x.as_ptr());
        acc = vfmaq_f32(acc, xv, xv);
    }
    let mut sumsq = vaddvq_f32(acc);
    for &x in tail {
        sumsq += x * x;
    }
    1.0 / (sumsq / n as f32 + eps).sqrt()
}

/// NEON softmax over rows `[start, end)`, width `width`.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
// SAFETY: only reachable when the caller verified NEON via `detect()`;
// `data` is a distinct slice from `src`.
unsafe fn softmaxrowspan_neon_kernel(
    src: &[f32],
    data: &mut [f32],
    width: usize,
    start: usize,
    end: usize,
) {
    let ntile = width / 4 * 4;
    for row in start..end {
        let inrow = &src[row * width..(row + 1) * width];
        let outrow = &mut data[row * width..(row + 1) * width];
        let mut m = vdupq_n_f32(f32::NEG_INFINITY);
        let mut finiteall = vdupq_n_u32(u32::MAX);
        let mut i = 0;
        while i < ntile {
            let mut lane = [0.0_f32; 4];
            lane.copy_from_slice(&inrow[i..i + 4]);
            let x = vld1q_f32(lane.as_ptr());
            m = vmaxq_f32(m, x);
            finiteall = vandq_u32(finiteall, vceqq_f32(x, x));
            i += 4;
        }
        let mut allfinite = vmaxvq_u32(finiteall) == u32::MAX;
        let mut tailmax = f32::NEG_INFINITY;
        for &x in &inrow[ntile..width] {
            if !x.is_finite() {
                allfinite = false;
            }
            tailmax = f32::max(tailmax, x);
        }
        if !allfinite {
            crate::scalar::softmaxrow(inrow, outrow);
            continue;
        }
        let maxv = f32::max(vmaxvq_f32(m), tailmax);
        let log2e = vdupq_n_f32(std::f32::consts::LOG2_E);
        let maxx = vdupq_n_f32(maxv);
        let mut s = vdupq_n_f32(0.0);
        let mut i = 0;
        while i < ntile {
            let mut lane = [0.0_f32; 4];
            lane.copy_from_slice(&inrow[i..i + 4]);
            let x = vld1q_f32(lane.as_ptr());
            let e = exp2approx_neon_kernel(vsubq_f32(x, maxx), log2e);
            let mut el = [0.0_f32; 4];
            vst1q_f32(el.as_mut_ptr(), e);
            outrow[i..i + 4].copy_from_slice(&el);
            s = vaddq_f32(s, e);
            i += 4;
        }
        let mut sum = vaddvq_f32(s);
        for (d, &x) in outrow[ntile..width].iter_mut().zip(&inrow[ntile..width]) {
            let e = (x - maxv).exp();
            *d = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        let invv = vdupq_n_f32(inv);
        let mut i = 0;
        while i < ntile {
            let mut el = [0.0_f32; 4];
            el.copy_from_slice(&outrow[i..i + 4]);
            let e = vld1q_f32(el.as_ptr());
            let mut ol = [0.0_f32; 4];
            vst1q_f32(ol.as_mut_ptr(), vmulq_f32(e, invv));
            outrow[i..i + 4].copy_from_slice(&ol);
            i += 4;
        }
        for d in &mut outrow[ntile..width] {
            *d *= inv;
        }
    }
}

/// NEON dot product of `a` and `b` over `len` elements.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
// SAFETY: only reachable when the caller verified NEON via `detect()`;
// the kernel only reads `a` and `b`.
unsafe fn attentiondot_neon_kernel(a: &[f32], b: &[f32], ntile: usize, len: usize) -> f32 {
    let mut acc = vdupq_n_f32(0.0);
    for i in (0..ntile).step_by(4) {
        let mut al = [0.0_f32; 4];
        let mut bl = [0.0_f32; 4];
        al.copy_from_slice(&a[i..i + 4]);
        bl.copy_from_slice(&b[i..i + 4]);
        acc = vfmaq_f32(acc, vld1q_f32(al.as_ptr()), vld1q_f32(bl.as_ptr()));
    }
    let mut dot = vaddvq_f32(acc);
    for i in ntile..len {
        dot += a[i] * b[i];
    }
    dot
}

/// NEON direct register-blocked matmul micro-kernel.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
// SAFETY: only reachable when the caller verified NEON via `detect()`;
// `out` is distinct from `a`/`b`; indices bounded by the caller's plan.
unsafe fn matmuldirect_neon_kernel(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    rowscale: Option<&[f32]>,
) {
    let ntile = n / 4 * 4;
    for i0 in (0..m).step_by(BLOCKROWS) {
        let rows = (m - i0).min(BLOCKROWS);
        for j0 in (0..ntile).step_by(4) {
            let mut acc = [vdupq_n_f32(0.0); BLOCKROWS];
            for p in 0..k {
                let mut bl = [0.0_f32; 4];
                bl.copy_from_slice(&b[p * n + j0..p * n + j0 + 4]);
                let brow = vld1q_f32(bl.as_ptr());
                for r in 0..rows {
                    acc[r] = vfmaq_f32(acc[r], brow, vdupq_n_f32(a[(i0 + r) * k + p]));
                }
            }
            for r in 0..rows {
                let mut outl = [0.0_f32; 4];
                vst1q_f32(outl.as_mut_ptr(), acc[r]);
                if let Some(scales) = rowscale {
                    let s = scales[i0 + r];
                    for lane in outl.iter_mut() {
                        *lane *= s;
                    }
                }
                out[(i0 + r) * n + j0..(i0 + r) * n + j0 + 4].copy_from_slice(&outl);
            }
        }
        for r in 0..rows {
            for j in ntile..n {
                let v = scalardot(a, b, i0 + r, j, k, n);
                out[(i0 + r) * n + j] = match rowscale {
                    Some(scales) => v * scales[i0 + r],
                    None => v,
                };
            }
        }
    }
}

/// NEON `exp(x)` for `x <= 0`, degree-7 Taylor in `2^(x * log2e)`.
#[allow(clippy::excessive_precision)]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
// SAFETY: as in `softmaxrowspan_neon_kernel`; purely reads `x` and `log2e`.
unsafe fn exp2approx_neon_kernel(x: float32x4_t, log2e: float32x4_t) -> float32x4_t {
    let y = vmulq_f32(x, log2e);
    let yi = vrndmq_f32(y);
    let yf = vsubq_f32(y, yi);
    let mut p = vdupq_n_f32(1.525_39e-5);
    p = vfmaq_f32(vdupq_n_f32(1.540_353_0e-4), p, yf);
    p = vfmaq_f32(vdupq_n_f32(1.333_355_8e-3), p, yf);
    p = vfmaq_f32(vdupq_n_f32(9.618_129e-3), p, yf);
    p = vfmaq_f32(vdupq_n_f32(5.550_410_9e-2), p, yf);
    p = vfmaq_f32(vdupq_n_f32(2.402_265_1e-1), p, yf);
    p = vfmaq_f32(vdupq_n_f32(std::f32::consts::LN_2), p, yf);
    p = vfmaq_f32(vdupq_n_f32(1.0), p, yf);
    let bias = vaddq_f32(yi, vdupq_n_f32(127.0));
    let bits = vcvtq_s32_f32(bias);
    let pos = vcgtq_s32(bits, vdupq_n_s32(0));
    let lt = vcltq_s32(bits, vdupq_n_s32(255));
    let valid = vandq_u32(pos, lt);
    let scale = vreinterpretq_f32_u32(vandq_u32(
        vreinterpretq_u32_s32(vshlq_n_s32(bits, 23)),
        valid,
    ));
    vmulq_f32(p, scale)
}

// ---------------------------------------------------------------------------
// Shared scalar helpers (safe; used by SIMD kernels and the scalar tier).
// ---------------------------------------------------------------------------

/// The row-block size of the direct matmul micro-kernels.
const BLOCKROWS: usize = 8;

/// Query row-block size of the blocked attention kernel.
const QBLOCK: usize = 32;

/// Key tile size of the online-rescaled attention softmax.
const KVBLOCK: usize = 256;

/// Returns how many key positions a query position may attend to.
///
/// Causal attention lets query position `s` see keys up to `offset + s`,
/// where `offset = kvseq - qseq`; keys after that are excluded. Non-causal
/// attention attends to every key.
fn attentionlimit(causal: bool, s: usize, kvseq: usize, qseq: usize) -> usize {
    if causal {
        let offset = kvseq - qseq;
        (offset + s + 1).min(kvseq)
    } else {
        kvseq
    }
}

/// Scalar dot product of row `i` of `a` with column `j` of `b`, for the tail.
/// Plain left-to-right accumulation (not fma), matching the source kernel's
/// `scalardot` exactly.
fn scalardot(a: &[f32], b: &[f32], i: usize, j: usize, k: usize, n: usize) -> f32 {
    let mut acc = 0.0_f32;
    for p in 0..k {
        acc += a[i * k + p] * b[p * n + j];
    }
    acc
}

/// AVX2+FMA grouped-query attention query block `[qs, qe)` with online
/// rescaled softmax over key tiles of [`KVBLOCK`] positions.
#[allow(clippy::too_many_arguments)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
// SAFETY: only reachable when the caller verified AVX2+FMA via `detect()`;
// `part` is a distinct slice from `q`/`k`/`v`, and all indices derive from
// the caller's validated `plan`.
unsafe fn attentionqblock_avx2_kernel(
    plan: &AttentionShape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    part: &mut [f32],
    qs: usize,
    qe: usize,
) {
    let headdim = plan.headdim;
    let ntile = headdim / 8 * 8;
    let rows = qe - qs;
    let mut maxv = vec![0.0_f32; QBLOCK * plan.heads];
    let mut sumv = vec![0.0_f32; QBLOCK * plan.heads];
    let mut acc = vec![0.0_f32; QBLOCK * plan.heads * headdim];
    let mut weights = [0.0_f32; KVBLOCK];
    for i in 0..rows {
        for head in 0..plan.heads {
            maxv[i * plan.heads + head] = f32::NEG_INFINITY;
            sumv[i * plan.heads + head] = 0.0;
            for d in 0..headdim {
                acc[(i * plan.heads + head) * headdim + d] = 0.0;
            }
        }
    }
    for ks in (0..plan.kvseq).step_by(KVBLOCK) {
        let ke = (ks + KVBLOCK).min(plan.kvseq);
        if plan.causal && plan.offset + qe <= ks {
            continue;
        }
        for kvhead in 0..plan.kvheads {
            for head in kvhead * plan.group..(kvhead + 1) * plan.group {
                for i in 0..rows {
                    let s = qs + i;
                    let end = attentionlimit(plan.causal, s, plan.kvseq, plan.qseq).min(ke);
                    if end <= ks {
                        continue;
                    }
                    let state = i * plan.heads + head;
                    let qrow = &q[s * plan.qstride + head * headdim..][..headdim];
                    let mut rowmax = f32::NEG_INFINITY;
                    for t in ks..end {
                        let krow = &k[t * plan.kstride + kvhead * headdim..][..headdim];
                        let score =
                            attentiondot_avx2_kernel(qrow, krow, ntile, headdim) * plan.scale;
                        weights[t - ks] = score;
                        if score > rowmax {
                            rowmax = score;
                        }
                    }
                    let newmax = if rowmax > maxv[state] {
                        rowmax
                    } else {
                        maxv[state]
                    };
                    let alpha = (maxv[state] - newmax).exp();
                    let mut tilesum = 0.0_f32;
                    for t in ks..end {
                        let w = (weights[t - ks] - newmax).exp();
                        weights[t - ks] = w;
                        tilesum += w;
                    }
                    let alphav = _mm256_set1_ps(alpha);
                    for d in (0..ntile).step_by(8) {
                        let mut al = [0.0_f32; 8];
                        al.copy_from_slice(&acc[state * headdim + d..state * headdim + d + 8]);
                        let a8 = _mm256_loadu_ps(al.as_ptr());
                        let mut outl = [0.0_f32; 8];
                        _mm256_storeu_ps(outl.as_mut_ptr(), _mm256_mul_ps(a8, alphav));
                        acc[state * headdim + d..state * headdim + d + 8].copy_from_slice(&outl);
                    }
                    for d in ntile..headdim {
                        acc[state * headdim + d] *= alpha;
                    }
                    for t in ks..end {
                        let vrow = &v[t * plan.vstride + kvhead * headdim..][..headdim];
                        let w = weights[t - ks];
                        let wv = _mm256_set1_ps(w);
                        for d in (0..ntile).step_by(8) {
                            let mut al = [0.0_f32; 8];
                            let mut vl = [0.0_f32; 8];
                            al.copy_from_slice(&acc[state * headdim + d..state * headdim + d + 8]);
                            vl.copy_from_slice(&vrow[d..d + 8]);
                            let a8 = _mm256_loadu_ps(al.as_ptr());
                            let v8 = _mm256_loadu_ps(vl.as_ptr());
                            let mut outl = [0.0_f32; 8];
                            _mm256_storeu_ps(outl.as_mut_ptr(), _mm256_fmadd_ps(v8, wv, a8));
                            acc[state * headdim + d..state * headdim + d + 8]
                                .copy_from_slice(&outl);
                        }
                        for d in ntile..headdim {
                            acc[state * headdim + d] += w * vrow[d];
                        }
                    }
                    maxv[state] = newmax;
                    sumv[state] = sumv[state] * alpha + tilesum;
                }
            }
        }
    }
    for i in 0..rows {
        let outrow = &mut part[i * plan.qstride..(i + 1) * plan.qstride];
        for head in 0..plan.heads {
            let state = i * plan.heads + head;
            if sumv[state] > 0.0 {
                for d in 0..headdim {
                    outrow[head * headdim + d] = acc[state * headdim + d] / sumv[state];
                }
            }
        }
    }
}

/// NEON grouped-query attention query block `[qs, qe)`.
#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
// SAFETY: only reachable when the caller verified NEON via `detect()`;
// `part` is a distinct slice from `q`/`k`/`v`, and all indices derive from
// the caller's validated `plan`.
unsafe fn attentionqblock_neon_kernel(
    plan: &AttentionShape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    part: &mut [f32],
    qs: usize,
    qe: usize,
) {
    let headdim = plan.headdim;
    let ntile = headdim / 4 * 4;
    let rows = qe - qs;
    let mut maxv = vec![0.0_f32; QBLOCK * plan.heads];
    let mut sumv = vec![0.0_f32; QBLOCK * plan.heads];
    let mut acc = vec![0.0_f32; QBLOCK * plan.heads * headdim];
    let mut weights = [0.0_f32; KVBLOCK];
    for i in 0..rows {
        for head in 0..plan.heads {
            maxv[i * plan.heads + head] = f32::NEG_INFINITY;
            sumv[i * plan.heads + head] = 0.0;
            for d in 0..headdim {
                acc[(i * plan.heads + head) * headdim + d] = 0.0;
            }
        }
    }
    for ks in (0..plan.kvseq).step_by(KVBLOCK) {
        let ke = (ks + KVBLOCK).min(plan.kvseq);
        if plan.causal && plan.offset + qe <= ks {
            continue;
        }
        for kvhead in 0..plan.kvheads {
            for head in kvhead * plan.group..(kvhead + 1) * plan.group {
                for i in 0..rows {
                    let s = qs + i;
                    let end = attentionlimit(plan.causal, s, plan.kvseq, plan.qseq).min(ke);
                    if end <= ks {
                        continue;
                    }
                    let state = i * plan.heads + head;
                    let qrow = &q[s * plan.qstride + head * headdim..][..headdim];
                    let mut rowmax = f32::NEG_INFINITY;
                    for t in ks..end {
                        let krow = &k[t * plan.kstride + kvhead * headdim..][..headdim];
                        let score =
                            attentiondot_neon_kernel(qrow, krow, ntile, headdim) * plan.scale;
                        weights[t - ks] = score;
                        if score > rowmax {
                            rowmax = score;
                        }
                    }
                    let newmax = if rowmax > maxv[state] {
                        rowmax
                    } else {
                        maxv[state]
                    };
                    let alpha = (maxv[state] - newmax).exp();
                    let mut tilesum = 0.0_f32;
                    for t in ks..end {
                        let w = (weights[t - ks] - newmax).exp();
                        weights[t - ks] = w;
                        tilesum += w;
                    }
                    let alphav = vdupq_n_f32(alpha);
                    for d in (0..ntile).step_by(4) {
                        let mut al = [0.0_f32; 4];
                        al.copy_from_slice(&acc[state * headdim + d..state * headdim + d + 4]);
                        let a4 = vld1q_f32(al.as_ptr());
                        let mut outl = [0.0_f32; 4];
                        vst1q_f32(outl.as_mut_ptr(), vmulq_f32(a4, alphav));
                        acc[state * headdim + d..state * headdim + d + 4].copy_from_slice(&outl);
                    }
                    for d in ntile..headdim {
                        acc[state * headdim + d] *= alpha;
                    }
                    for t in ks..end {
                        let vrow = &v[t * plan.vstride + kvhead * headdim..][..headdim];
                        let w = weights[t - ks];
                        let wv = vdupq_n_f32(w);
                        for d in (0..ntile).step_by(4) {
                            let mut al = [0.0_f32; 4];
                            let mut vl = [0.0_f32; 4];
                            al.copy_from_slice(&acc[state * headdim + d..state * headdim + d + 4]);
                            vl.copy_from_slice(&vrow[d..d + 4]);
                            let a4 = vld1q_f32(al.as_ptr());
                            let v4 = vld1q_f32(vl.as_ptr());
                            let mut outl = [0.0_f32; 4];
                            vst1q_f32(outl.as_mut_ptr(), vfmaq_f32(a4, v4, wv));
                            acc[state * headdim + d..state * headdim + d + 4]
                                .copy_from_slice(&outl);
                        }
                        for d in ntile..headdim {
                            acc[state * headdim + d] += w * vrow[d];
                        }
                    }
                    maxv[state] = newmax;
                    sumv[state] = sumv[state] * alpha + tilesum;
                }
            }
        }
    }
    for i in 0..rows {
        let outrow = &mut part[i * plan.qstride..(i + 1) * plan.qstride];
        for head in 0..plan.heads {
            let state = i * plan.heads + head;
            if sumv[state] > 0.0 {
                for d in 0..headdim {
                    outrow[head * headdim + d] = acc[state * headdim + d] / sumv[state];
                }
            }
        }
    }
}

/// AVX2+FMA blocked matmul micro-kernel: accumulates one packed inner chunk
/// into `part[0..rows * n]` (the caller's local row-major block).
#[allow(clippy::too_many_arguments)]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
// SAFETY: only reachable when the caller verified AVX2+FMA via `detect()`;
// `part` is distinct from `a`/`b`/`packed`, and every index is bounded by
// the caller's matmul plan.
unsafe fn matmultile_avx2_kernel(
    a: &[f32],
    b: &[f32],
    packed: &[f32],
    part: &mut [f32],
    i0base: usize,
    rows: usize,
    jp: usize,
    panel: usize,
    kc0: usize,
    kchunk: usize,
    k: usize,
    n: usize,
    rowscale: Option<&[f32]>,
    lastchunk: bool,
) {
    let ntile = panel / 8 * 8;
    for j0 in (0..ntile).step_by(8) {
        let mut acc = [_mm256_setzero_ps(); BLOCKROWS];
        for r in 0..rows {
            let mut pl = [0.0_f32; 8];
            pl.copy_from_slice(&part[r * n + jp + j0..r * n + jp + j0 + 8]);
            acc[r] = _mm256_loadu_ps(pl.as_ptr());
        }
        for p in 0..kchunk {
            let mut bl = [0.0_f32; 8];
            bl.copy_from_slice(&packed[p * panel + j0..p * panel + j0 + 8]);
            let brow = _mm256_loadu_ps(bl.as_ptr());
            for r in 0..rows {
                acc[r] =
                    _mm256_fmadd_ps(brow, _mm256_set1_ps(a[(i0base + r) * k + kc0 + p]), acc[r]);
            }
        }
        for r in 0..rows {
            let mut outl = [0.0_f32; 8];
            _mm256_storeu_ps(outl.as_mut_ptr(), acc[r]);
            if lastchunk && let Some(scales) = rowscale {
                let s = scales[i0base + r];
                for lane in outl.iter_mut() {
                    *lane *= s;
                }
            }
            part[r * n + jp + j0..r * n + jp + j0 + 8].copy_from_slice(&outl);
        }
    }
    if lastchunk {
        for r in 0..rows {
            for j in ntile..panel {
                let v = scalardot(a, b, i0base + r, jp + j, k, n);
                part[r * n + jp + j] = match rowscale {
                    Some(scales) => v * scales[i0base + r],
                    None => v,
                };
            }
        }
    } else {
        for r in 0..rows {
            for j in ntile..panel {
                part[r * n + jp + j] = scalardot(a, b, i0base + r, jp + j, k, n);
            }
        }
    }
}

/// NEON blocked matmul micro-kernel: accumulates one packed inner chunk.
#[allow(clippy::too_many_arguments)]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
// SAFETY: only reachable when the caller verified NEON via `detect()`;
// `part` is distinct from `a`/`b`/`packed`, and every index is bounded by
// the caller's matmul plan.
unsafe fn matmultile_neon_kernel(
    a: &[f32],
    b: &[f32],
    packed: &[f32],
    part: &mut [f32],
    i0base: usize,
    rows: usize,
    jp: usize,
    panel: usize,
    kc0: usize,
    kchunk: usize,
    k: usize,
    n: usize,
    rowscale: Option<&[f32]>,
    lastchunk: bool,
) {
    let ntile = panel / 4 * 4;
    for j0 in (0..ntile).step_by(4) {
        let mut acc = [vdupq_n_f32(0.0); BLOCKROWS];
        for r in 0..rows {
            let mut pl = [0.0_f32; 4];
            pl.copy_from_slice(&part[r * n + jp + j0..r * n + jp + j0 + 4]);
            acc[r] = vld1q_f32(pl.as_ptr());
        }
        for p in 0..kchunk {
            let mut bl = [0.0_f32; 4];
            bl.copy_from_slice(&packed[p * panel + j0..p * panel + j0 + 4]);
            let brow = vld1q_f32(bl.as_ptr());
            for r in 0..rows {
                acc[r] = vfmaq_f32(acc[r], brow, vdupq_n_f32(a[(i0base + r) * k + kc0 + p]));
            }
        }
        for r in 0..rows {
            let mut outl = [0.0_f32; 4];
            vst1q_f32(outl.as_mut_ptr(), acc[r]);
            if lastchunk && let Some(scales) = rowscale {
                let s = scales[i0base + r];
                for lane in outl.iter_mut() {
                    *lane *= s;
                }
            }
            part[r * n + jp + j0..r * n + jp + j0 + 4].copy_from_slice(&outl);
        }
    }
    if lastchunk {
        for r in 0..rows {
            for j in ntile..panel {
                let v = scalardot(a, b, i0base + r, jp + j, k, n);
                part[r * n + jp + j] = match rowscale {
                    Some(scales) => v * scales[i0base + r],
                    None => v,
                };
            }
        }
    } else {
        for r in 0..rows {
            for j in ntile..panel {
                part[r * n + jp + j] = scalardot(a, b, i0base + r, jp + j, k, n);
            }
        }
    }
}
