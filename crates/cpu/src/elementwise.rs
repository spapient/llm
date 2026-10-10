use ::tensor::{tensor, tensorview};
use api::{backenderror, binarykernel};

use crate::support::finish;

/// The stable name reported by [`cpuadd`].
const ADDOP: &str = "add";

/// The stable name reported by [`cpumul`].
const MULOP: &str = "mul";

/// Elementwise addition of two `f32` tensors of identical shape.
///
/// No broadcasting is performed: the two operands must have exactly the same
/// dimensions. `NaN` and infinities propagate through ordinary floating-point
/// addition.
#[derive(Clone, Copy, Debug, Default)]
pub struct cpuadd;

/// Elementwise multiplication of two `f32` tensors of identical shape.
///
/// No broadcasting is performed: the two operands must have exactly the same
/// dimensions. `NaN` and infinities propagate through ordinary floating-point
/// multiplication.
#[derive(Clone, Copy, Debug, Default)]
pub struct cpumul;

impl binarykernel<f32> for cpuadd {
    fn getname(&self) -> &'static str {
        ADDOP
    }

    fn apply(
        &self,
        lhs: &tensorview<'_, f32>,
        rhs: &tensorview<'_, f32>,
    ) -> Result<tensor<f32>, backenderror> {
        let data = zipmap(lhs, rhs, ADDOP, |a, b| a + b)?;
        finish(lhs.getshape(), data)
    }
}

impl binarykernel<f32> for cpumul {
    fn getname(&self) -> &'static str {
        MULOP
    }

    fn apply(
        &self,
        lhs: &tensorview<'_, f32>,
        rhs: &tensorview<'_, f32>,
    ) -> Result<tensor<f32>, backenderror> {
        let data = zipmap(lhs, rhs, MULOP, |a, b| a * b)?;
        finish(lhs.getshape(), data)
    }
}

/// Zips two equal-shaped operands through `op`, rejecting shape mismatches.
fn zipmap(
    lhs: &tensorview<'_, f32>,
    rhs: &tensorview<'_, f32>,
    op: &'static str,
    f: impl Fn(f32, f32) -> f32,
) -> Result<Vec<f32>, backenderror> {
    if lhs.getshape().getdims() != rhs.getshape().getdims() {
        return Err(backenderror::shapemismatch { op });
    }
    Ok(lhs
        .getdata()
        .iter()
        .zip(rhs.getdata().iter())
        .map(|(&a, &b)| f(a, b))
        .collect())
}
