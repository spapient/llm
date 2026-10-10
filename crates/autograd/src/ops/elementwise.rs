use ::tensor::tensor;
use api::compute;

use crate::arena::graph;
use crate::error::autoggraderror;
use crate::node::{op, variableid};
use crate::ops::{accumulate, sameshape};

/// The stable name reported by the elementwise addition operator.
const ADDOP: &str = "add";

/// The stable name reported by the elementwise multiplication operator.
const MULOP: &str = "mul";

impl graph<f32> {
    /// Differentiable elementwise addition of two equal-shaped operands.
    ///
    /// # Errors
    ///
    /// Returns [`autoggraderror::unknownvariable`] for a missing operand,
    /// [`autoggraderror::unsupportedshape`] when the shapes differ, or the
    /// provider's error when execution fails.
    pub fn add<C: compute>(
        &mut self,
        engine: &C,
        lhs: variableid,
        rhs: variableid,
    ) -> Result<variableid, autoggraderror> {
        let a = self.valueof(lhs)?.clone();
        let b = self.valueof(rhs)?.clone();
        if !sameshape(&a, &b) {
            return Err(autoggraderror::unsupportedshape {
                op: ADDOP,
                dims: a.getshape().getdims().to_vec(),
            });
        }
        let out = engine.add(&a.getview(), &b.getview())?;
        let requires = self.record(&[lhs, rhs]);
        let producer = if requires {
            op::add { lhs, rhs }
        } else {
            op::leaf
        };
        Ok(self.push(out, requires, producer))
    }

    /// Differentiable elementwise multiplication of two equal-shaped operands.
    ///
    /// # Errors
    ///
    /// Returns [`autoggraderror::unknownvariable`] for a missing operand,
    /// [`autoggraderror::unsupportedshape`] when the shapes differ, or the
    /// provider's error when execution fails.
    pub fn mul<C: compute>(
        &mut self,
        engine: &C,
        lhs: variableid,
        rhs: variableid,
    ) -> Result<variableid, autoggraderror> {
        let a = self.valueof(lhs)?.clone();
        let b = self.valueof(rhs)?.clone();
        if !sameshape(&a, &b) {
            return Err(autoggraderror::unsupportedshape {
                op: MULOP,
                dims: a.getshape().getdims().to_vec(),
            });
        }
        let out = engine.mul(&a.getview(), &b.getview())?;
        let requires = self.record(&[lhs, rhs]);
        let producer = if requires {
            op::mul { lhs, rhs }
        } else {
            op::leaf
        };
        Ok(self.push(out, requires, producer))
    }
}

/// Adds the incoming gradient to both operands unchanged.
pub(crate) fn backwardadd(
    g: &mut graph<f32>,
    lhs: variableid,
    rhs: variableid,
    gy: &tensor<f32>,
) -> Result<(), autoggraderror> {
    if g.requiresgrad(lhs)? {
        accumulate(g, lhs, gy.clone())?;
    }
    if g.requiresgrad(rhs)? {
        accumulate(g, rhs, gy.clone())?;
    }
    Ok(())
}

/// Multiplies the incoming gradient by the opposite operand for each operand.
pub(crate) fn backwardmul(
    g: &mut graph<f32>,
    lhs: variableid,
    rhs: variableid,
    gy: &tensor<f32>,
) -> Result<(), autoggraderror> {
    let a = g.valueof(lhs)?.clone();
    let b = g.valueof(rhs)?.clone();
    if !sameshape(&a, &b) || !sameshape(&a, gy) {
        return Err(autoggraderror::unsupportedshape {
            op: MULOP,
            dims: a.getshape().getdims().to_vec(),
        });
    }
    if g.requiresgrad(lhs)? {
        let data: Vec<f32> = gy
            .getdata()
            .iter()
            .zip(b.getdata())
            .map(|(&gv, &bv)| gv * bv)
            .collect();
        accumulate(g, lhs, tensor::new(gy.getshape().clone(), data)?)?;
    }
    if g.requiresgrad(rhs)? {
        let data: Vec<f32> = gy
            .getdata()
            .iter()
            .zip(a.getdata())
            .map(|(&gv, &av)| gv * av)
            .collect();
        accumulate(g, rhs, tensor::new(gy.getshape().clone(), data)?)?;
    }
    Ok(())
}
