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
/// A row whose shifted exponents all underflow to zero, such as a row of
/// negative infinity, is left as all zeros instead of dividing by zero.
/// Non-finite inputs are neither clamped nor rejected; they follow ordinary
/// floating-point propagation.
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
        for (dst, row) in data.chunks_mut(width).zip(src.chunks(width)) {
            softmaxrow(row, dst);
        }
        finish(input.getshape(), data)
    }
}

fn softmaxrow(row: &[f32], dst: &mut [f32]) {
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
