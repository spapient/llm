use ::tensor::{tensor, tensorview};
use api::{backenderror, kernel};
use archmage::prelude::*;

use crate::support::finish;

/// The stable name reported by [`cpusoftmax`].
const SOFTMAXOP: &str = "softmax";

/// Numerically stable softmax over the last axis.
///
/// Each row is one contiguous last-axis slice of the input. The kernel
/// subtracts the row maximum before exponentiating, so a row such as
/// `[1000, 1000]` yields `[0.5, 0.5]` rather than overflowing.
///
/// A zero-length last dimension has no defined softmax and is rejected with
/// [`backenderror::unsupportedoperation`]. An input that is empty for another
/// reason, such as shape `[0, 3]`, has no rows and produces an empty output of
/// the same shape.
///
/// A row whose shifted exponents sum to zero is left as all zeros rather than
/// divided by zero. That branch is unreachable for a finite row, because the
/// maximum's own shifted exponential is one; it is also why a row of negative
/// infinity is *not* a zero row: subtracting the maximum from itself yields
/// NaN, and the same guard propagates it instead of masking it. Non-finite
/// inputs are neither clamped nor rejected; they follow ordinary
/// floating-point propagation.
///
/// # SIMD strategy
///
/// Fully finite rows take a vectorized fast path (AVX2 8-wide / NEON 4-wide /
/// scalar): a vector maximum reduction, a degree-7 Taylor `2^x` approximation
/// of the shifted exponential (relative error below `~5e-7` on `[-inf, 0]`,
/// inside the contract tolerance), a vector sum reduction, and a
/// multiplication by the reciprocal instead of per-element division. Any row
/// containing a non-finite value takes the scalar reference path unchanged, so
/// the documented non-finite propagation (including the `-inf` row's `NaN`
/// guard) is bit-exact. The `_mm256_max_ps` NaN asymmetry that `f32::max`
/// avoids is thus irrelevant: the fast path never sees a NaN.
#[derive(Clone, Copy, Debug, Default)]
pub struct cpusoftmax;

impl kernel<f32> for cpusoftmax {
    fn getname(&self) -> &'static str {
        SOFTMAXOP
    }

    fn apply(&self, input: &tensorview<'_, f32>) -> Result<tensor<f32>, backenderror> {
        let width = match input.getshape().getdims().last() {
            Some(&w) if w > 0 => w,
            _ => return Err(backenderror::unsupportedoperation { op: SOFTMAXOP }),
        };
        let src = input.getdata();
        let mut data = vec![0.0_f32; src.len()];
        softmaxrows(src, &mut data, width);
        finish(input.getshape(), data)
    }
}

/// Normalizes every row of `src` into `data`, selecting the widest CPU tier
/// available. The row loop lives inside the dispatched variant, so the
/// `#[target_feature]` boundary is crossed once per call.
fn softmaxrows(src: &[f32], data: &mut [f32], width: usize) {
    let rows = src.len() / width;
    incant!(
        softmaxrowspan(src, data, width, 0, rows),
        [v3, neon, scalar]
    );
}

/// AVX2 8-wide variant.
#[arcane(import_intrinsics)]
fn softmaxrowspan_v3(
    token: X64V3Token,
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
        // One pass: accumulate the maximum and a finite-flag mask together.
        let mut m = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut finiteall = _mm256_set1_ps(-0.0);
        let mut i = 0;
        while i < ntile {
            let mut lane = [0.0_f32; 8];
            lane.copy_from_slice(&inrow[i..i + 8]);
            let x = _mm256_loadu_ps(&lane);
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
            softmaxrow_scalar(inrow, outrow);
            continue;
        }
        let maxv = f32::max(hmax256(token, m), tailmax);
        // Shifted exponential via 2^(x - max) = 2^floor * 2^frac.
        let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
        let maxx = _mm256_set1_ps(maxv);
        let mut s = _mm256_setzero_ps();
        let mut i = 0;
        while i < ntile {
            let mut lane = [0.0_f32; 8];
            lane.copy_from_slice(&inrow[i..i + 8]);
            let x = _mm256_loadu_ps(&lane);
            let e = exp2approx_v3(token, _mm256_sub_ps(x, maxx), log2e);
            let mut el = [0.0_f32; 8];
            _mm256_storeu_ps(&mut el, e);
            outrow[i..i + 8].copy_from_slice(&el);
            s = _mm256_add_ps(s, e);
            i += 8;
        }
        let mut sum = hsum256(token, s);
        for (d, &x) in outrow[ntile..width].iter_mut().zip(&inrow[ntile..width]) {
            let e = (x - maxv).exp();
            *d = e;
            sum += e;
        }
        // Finite rows have sum >= 1 (the maximum's own exponential is one), so
        // the division guard never triggers; multiplying by the reciprocal is
        // inside tolerance and avoids 8-wide divides.
        let inv = 1.0 / sum;
        let invv = _mm256_set1_ps(inv);
        let mut i = 0;
        while i < ntile {
            let mut el = [0.0_f32; 8];
            el.copy_from_slice(&outrow[i..i + 8]);
            let e = _mm256_loadu_ps(&el);
            let mut ol = [0.0_f32; 8];
            _mm256_storeu_ps(&mut ol, _mm256_mul_ps(e, invv));
            outrow[i..i + 8].copy_from_slice(&ol);
            i += 8;
        }
        for d in &mut outrow[ntile..width] {
            *d *= inv;
        }
    }
}

/// AArch64 4-wide variant.
#[arcane(import_intrinsics)]
fn softmaxrowspan_neon(
    token: NeonToken,
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
            let x = vld1q_f32(&lane);
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
            softmaxrow_scalar(inrow, outrow);
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
            let x = vld1q_f32(&lane);
            let e = exp2approx_neon(token, vsubq_f32(x, maxx), log2e);
            let mut el = [0.0_f32; 4];
            vst1q_f32(&mut el, e);
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
            let e = vld1q_f32(&el);
            let mut ol = [0.0_f32; 4];
            vst1q_f32(&mut ol, vmulq_f32(e, invv));
            outrow[i..i + 4].copy_from_slice(&ol);
            i += 4;
        }
        for d in &mut outrow[ntile..width] {
            *d *= inv;
        }
    }
}

/// Scalar fallback: the original reference path, bit-exact for every input.
#[arcane(import_intrinsics)]
fn softmaxrowspan_scalar(
    _: ScalarToken,
    src: &[f32],
    data: &mut [f32],
    width: usize,
    start: usize,
    end: usize,
) {
    for row in start..end {
        softmaxrow_scalar(
            &src[row * width..(row + 1) * width],
            &mut data[row * width..(row + 1) * width],
        );
    }
}

/// The scalar softmax row: `f32::max` fold (skips NaN), libm `exp`, and the
/// zero-sum guard that propagates the `-inf` row's NaN.
fn softmaxrow_scalar(row: &[f32], dst: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for (d, &x) in dst.iter_mut().zip(row.iter()) {
        let e = (x - max).exp();
        *d = e;
        sum += e;
    }
    if sum > 0.0 {
        for d in dst.iter_mut() {
            *d /= sum;
        }
    }
}

/// Approximates `exp(x)` for `x <= 0` as `2^(x * log2e)`.
///
/// `2^yf` on `[0, 1)` is a degree-7 Taylor polynomial (remainder below
/// `2.3e-7` absolute, about `1.2e-7` relative); `2^yi` for the integer part is
/// packed into the IEEE exponent field. Exponents outside `[-126, 127]` (an
/// `x - max` below about `-87`) are flushed to zero, whose absolute error is
/// below `2^-126` and far inside tolerance. The fused multiply-add keeps the
/// polynomial's rounding to a few ulps, for a total relative error around
/// `5e-7`.
#[arcane(import_intrinsics)]
fn exp2approx_v3(_: X64V3Token, x: __m256, log2e: __m256) -> __m256 {
    let y = _mm256_mul_ps(x, log2e);
    let yi = _mm256_floor_ps(y);
    let yf = _mm256_sub_ps(y, yi);
    // 2^yf, Horner from the degree-7 coefficient down.
    let mut p = _mm256_set1_ps(1.525_39e-5);
    p = _mm256_fmadd_ps(p, yf, _mm256_set1_ps(1.540_353_0e-4));
    p = _mm256_fmadd_ps(p, yf, _mm256_set1_ps(1.333_355_8e-3));
    p = _mm256_fmadd_ps(p, yf, _mm256_set1_ps(9.618_129e-3));
    p = _mm256_fmadd_ps(p, yf, _mm256_set1_ps(5.550_410_9e-2));
    p = _mm256_fmadd_ps(p, yf, _mm256_set1_ps(2.402_265_1e-1));
    p = _mm256_fmadd_ps(p, yf, _mm256_set1_ps(6.931_471_8e-1));
    p = _mm256_fmadd_ps(p, yf, _mm256_set1_ps(1.0));
    // 2^yi: exponent field (yi + 127) << 23, masked to the valid range.
    let bias = _mm256_add_ps(yi, _mm256_set1_ps(127.0));
    let bits = _mm256_cvtps_epi32(bias);
    let pos = _mm256_cmpgt_epi32(bits, _mm256_set1_epi32(0));
    // archmage wraps `_mm256_cmpgt_epi32` but not its less-than twin; the
    // swapped comparison is the same predicate (255 > bits).
    let lt = _mm256_cmpgt_epi32(_mm256_set1_epi32(255), bits);
    let valid = _mm256_and_si256(pos, lt);
    let scale = _mm256_castsi256_ps(_mm256_and_si256(_mm256_slli_epi32(bits, 23), valid));
    _mm256_mul_ps(p, scale)
}

/// NEON 4-wide variant of [`exp2approx_v3`].
#[allow(clippy::excessive_precision)]
#[arcane(import_intrinsics)]
fn exp2approx_neon(_: NeonToken, x: float32x4_t, log2e: float32x4_t) -> float32x4_t {
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

/// Horizontal maximum of one 256-bit register.
#[arcane(import_intrinsics)]
fn hmax256(_: X64V3Token, m: __m256) -> f32 {
    let low = _mm256_castps256_ps128(m);
    let high = _mm256_extractf128_ps::<1>(m);
    let q = _mm_max_ps(low, high);
    let s1 = _mm_max_ps(q, _mm_movehdup_ps(q));
    let s2 = _mm_max_ps(_mm_movehl_ps(_mm_movehdup_ps(q), s1), s1);
    _mm_cvtss_f32(s2)
}

/// Horizontal sum of one 256-bit register.
#[arcane(import_intrinsics)]
fn hsum256(_: X64V3Token, s: __m256) -> f32 {
    let low = _mm256_castps256_ps128(s);
    let high = _mm256_extractf128_ps::<1>(s);
    let q = _mm_add_ps(low, high);
    let shuf = _mm_movehdup_ps(q);
    let sums = _mm_add_ps(q, shuf);
    let shuf = _mm_movehl_ps(sums, shuf);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf))
}
