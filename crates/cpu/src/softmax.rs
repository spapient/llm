use ::tensor::{tensor, tensorview};
use api::{backenderror, kernel};

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
/// available. The tier is detected once per call; the `#[target_feature]`
/// boundary is crossed inside `simd`.
fn softmaxrows(src: &[f32], data: &mut [f32], width: usize) {
    let rows = src.len() / width;
    simd::softmaxrowspan(simd::detect(), src, data, width, 0, rows);
}
