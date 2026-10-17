use ::tensor::{tensor, tensorview};
use api::{backenderror, kernel};

use crate::support::finish;

/// Square root of `2 / pi`, the scale factor of the tanh GELU approximation.
const GELUTANHSCALE: f32 = 0.797_884_6;

/// Cubic coefficient of the tanh GELU approximation.
const GELUTANHCUBIC: f32 = 0.044_715;

/// Rectified linear unit: `relu(x) = max(0, x)`.
///
/// Implemented as `if x < 0.0 { 0.0 } else { x }`, so `NaN` propagates and
/// negative zero is preserved (it compares equal to `0.0`).
#[derive(Clone, Copy, Debug, Default)]
pub struct cpurelu;

/// Sigmoid linear unit: `silu(x) = x / (1 + exp(-x))`.
///
/// `NaN` propagates. For large negative `x` the exponential saturates to
/// positive infinity and the quotient underflows toward negative zero.
///
/// The transcendental stays on the scalar `exp` path: the operator contract
/// fixes exact IEEE-754 propagation semantics, and the libm exponential is the
/// only value that keeps every non-finite case bit-exact across SIMD tiers.
#[derive(Clone, Copy, Debug, Default)]
pub struct cpusilu;

/// Tanh-approximation GELU:
/// `gelutanh(x) = 0.5 * x * (1 + tanh(c * (x + 0.044715 * x^3)))`
/// with `c = sqrt(2 / pi)`.
///
/// This is the tanh formulation, not the erf formulation; the two agree to
/// roughly `1e-3`, and the choice is part of the operator contract.
#[derive(Clone, Copy, Debug, Default)]
pub struct cpugelutanh;

impl kernel<f32> for cpurelu {
    fn getname(&self) -> &'static str {
        "relu"
    }

    fn apply(&self, input: &tensorview<'_, f32>) -> Result<tensor<f32>, backenderror> {
        let mut data = vec![0.0_f32; input.getdata().len()];
        reluspan(input.getdata(), &mut data);
        finish(input.getshape(), data)
    }
}

/// Rectified linear unit over one span.
///
/// The `if x < 0.0` form is kept deliberately instead of a `max` intrinsic:
/// the compiler's vectorized `select` preserves the scalar contract exactly —
/// `NaN` propagates (the comparison is false) and negative zero is passed
/// through unchanged rather than flushed to `+0`.
fn reluspan(data: &[f32], out: &mut [f32]) {
    for (o, &x) in out.iter_mut().zip(data.iter()) {
        *o = if x < 0.0 { 0.0 } else { x };
    }
}

impl kernel<f32> for cpusilu {
    fn getname(&self) -> &'static str {
        "silu"
    }

    fn apply(&self, input: &tensorview<'_, f32>) -> Result<tensor<f32>, backenderror> {
        let data = input.getdata().iter().copied().map(siluvalue).collect();
        finish(input.getshape(), data)
    }
}

impl kernel<f32> for cpugelutanh {
    fn getname(&self) -> &'static str {
        "gelutanh"
    }

    fn apply(&self, input: &tensorview<'_, f32>) -> Result<tensor<f32>, backenderror> {
        let data = input.getdata().iter().copied().map(gelutanhvalue).collect();
        finish(input.getshape(), data)
    }
}

fn siluvalue(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn gelutanhvalue(x: f32) -> f32 {
    let inner = GELUTANHSCALE * (x + GELUTANHCUBIC * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}
