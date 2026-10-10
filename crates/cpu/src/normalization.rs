use ::tensor::{tensor, tensorview};
use api::{backenderror, weightedkernel};

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
        for (row, out) in src.chunks(width).zip(data.chunks_mut(width)) {
            rmsnormrow(row, out, w, self.eps);
        }
        finish(input.getshape(), data)
    }
}

fn rmsnormrow(row: &[f32], out: &mut [f32], weight: &[f32], eps: f32) {
    let sumsq: f32 = row.iter().map(|&x| x * x).sum();
    let mean = sumsq / row.len() as f32;
    let inv = 1.0 / (mean + eps).sqrt();
    for ((dst, &x), &w) in out.iter_mut().zip(row.iter()).zip(weight.iter()) {
        *dst = x * inv * w;
    }
}
