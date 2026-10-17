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
        if lhs.getshape().getdims() != rhs.getshape().getdims() {
            return Err(backenderror::shapemismatch { op: ADDOP });
        }
        let mut data = vec![0.0_f32; lhs.getdata().len()];
        addspans(lhs.getdata(), rhs.getdata(), &mut data);
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
        if lhs.getshape().getdims() != rhs.getshape().getdims() {
            return Err(backenderror::shapemismatch { op: MULOP });
        }
        let mut data = vec![0.0_f32; lhs.getdata().len()];
        mulspans(lhs.getdata(), rhs.getdata(), &mut data);
        finish(lhs.getshape(), data)
    }
}

/// Elementwise addition over equal-length spans.
///
/// The branch-free scalar loop is auto-vectorized by the compiler on every
/// tier. Plain `+` keeps every IEEE-754 property of the scalar kernel:
/// `NaN`/infinity propagation, signed zero, and rounding are unchanged, so the
/// result is bit-identical to the former `zipmap` implementation.
fn addspans(a: &[f32], b: &[f32], out: &mut [f32]) {
    for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = x + y;
    }
}

/// Elementwise multiplication over equal-length spans.
///
/// Same contract as [`addspans`]; `*` is likewise bit-identical to scalar.
fn mulspans(a: &[f32], b: &[f32], out: &mut [f32]) {
    for ((o, &x), &y) in out.iter_mut().zip(a.iter()).zip(b.iter()) {
        *o = x * y;
    }
}
