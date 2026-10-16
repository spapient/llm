use ::tensor::{tensor, tensorview};
use api::{backenderror, weightedkernel};
use fearless_simd::{Level, Simd, dispatch, f32x4, prelude::*};

use crate::support::finish;

/// The stable name reported by [`cpurmsnorm`].
const RMSNORMOP: &str = "rmsnorm";

/// RMS normalization over the last axis, scaled by a learnable weight.
///
/// For each row `x` of the last axis and weight `w` of the same length,
///
/// ```text
/// rms     = sqrt(mean(x_i^2) + eps)
/// out_i   = (x_i / rms) * w_i
/// ```
///
/// The epsilon is carried by the kernel instance so that no per-model value is
/// hidden in a global. `eps` MUST be finite and non-negative; a negative
/// epsilon can drive the denominator to zero and is not validated here because
/// it is a caller configuration precondition, not external input.
#[derive(Clone, Copy, Debug)]
pub struct cpurmsnorm {
    eps: f32,
}

impl cpurmsnorm {
    /// The default epsilon used by [`Default`].
    pub const DEFAULTEPS: f32 = 1.0e-5;

    /// Creates an RMS normalization kernel with the given epsilon.
    #[must_use]
    pub const fn new(eps: f32) -> Self {
        Self { eps }
    }

    /// Returns the configured epsilon.
    #[must_use]
    pub const fn geteps(self) -> f32 {
        self.eps
    }
}

impl Default for cpurmsnorm {
    fn default() -> Self {
        Self::new(Self::DEFAULTEPS)
    }
}

impl weightedkernel<f32> for cpurmsnorm {
    fn getname(&self) -> &'static str {
        RMSNORMOP
    }

    fn apply(
        &self,
        input: &tensorview<'_, f32>,
        weight: &tensorview<'_, f32>,
    ) -> Result<tensor<f32>, backenderror> {
        let Some(&width) = input.getshape().getdims().last() else {
            return Err(backenderror::unsupportedoperation { op: RMSNORMOP });
        };
        if width == 0 {
            return Err(backenderror::unsupportedoperation { op: RMSNORMOP });
        }
        if weight.getdata().len() != width {
            return Err(backenderror::shapemismatch { op: RMSNORMOP });
        }
        let src = input.getdata();
        let w = weight.getdata();
        let mut data = vec![0.0_f32; src.len()];
        let level = Level::new();
        dispatch!(level, simd => rmsnormrows(simd, src, &mut data, w, self.eps, width));
        finish(input.getshape(), data)
    }
}

/// Normalizes every row of `src` into `data`, dispatched through
/// `fearless_simd`.
#[inline(always)]
fn rmsnormrows<S: Simd>(
    simd: S,
    src: &[f32],
    data: &mut [f32],
    weight: &[f32],
    eps: f32,
    width: usize,
) {
    for (row, out) in src.chunks(width).zip(data.chunks_mut(width)) {
        rmsnormrow(simd, row, out, weight, eps);
    }
}

/// RMS-normalizes one `row` with a vectorized square-sum and a vectorized
/// scale pass.
#[inline(always)]
fn rmsnormrow<S: Simd>(simd: S, row: &[f32], out: &mut [f32], weight: &[f32], eps: f32) {
    let n = row.len();
    let ntile = n / 4 * 4;
    let mut acc = f32x4::<S>::splat(simd, 0.0);
    for i in (0..ntile).step_by(4) {
        let x = f32x4::<S>::from_slice(simd, &row[i..i + 4]);
        acc = x.mul_add(x, acc);
    }
    let mut sumsq = acc.reduce_sum();
    for &x in &row[ntile..n] {
        sumsq += x * x;
    }
    let inv = 1.0 / (sumsq / n as f32 + eps).sqrt();

    let invv = f32x4::<S>::splat(simd, inv);
    for i in (0..ntile).step_by(4) {
        let x = f32x4::<S>::from_slice(simd, &row[i..i + 4]);
        let w = f32x4::<S>::from_slice(simd, &weight[i..i + 4]);
        (x * invv * w).store_slice(&mut out[i..i + 4]);
    }
    for i in ntile..n {
        out[i] = row[i] * inv * weight[i];
    }
}
