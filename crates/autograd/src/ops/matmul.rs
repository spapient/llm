use ::tensor::{shape, tensor};
use api::compute;

use crate::arena::graph;
use crate::error::autoggraderror;
use crate::node::{op, variableid};
use crate::ops::accumulate;

/// The stable name reported by the matrix-multiplication operator.
const MATMULOP: &str = "matmul";

impl graph<f32> {
    /// Differentiable matrix multiplication of two rank-two operands.
    ///
    /// `lhs` is `[m, k]` and `rhs` is `[k, n]`; the result is `[m, n]`. Batched
    /// operands are not supported by this operator.
    ///
    /// # Errors
    ///
    /// Returns [`autoggraderror::unknownvariable`] for a missing operand,
    /// [`autoggraderror::unsupportedshape`] when either operand is not rank two,
    /// or the provider's [`autoggraderror::backend`] error when the shapes are
    /// incompatible or execution fails.
    pub fn matmul<C: compute>(
        &mut self,
        engine: &C,
        lhs: variableid,
        rhs: variableid,
    ) -> Result<variableid, autoggraderror> {
        let a = self.valueof(lhs)?.clone();
        let b = self.valueof(rhs)?.clone();
        if a.getshape().getrank() != 2 || b.getshape().getrank() != 2 {
            return Err(autoggraderror::unsupportedshape {
                op: MATMULOP,
                dims: a.getshape().getdims().to_vec(),
            });
        }
        let out = engine.matmul(&a.getview(), &b.getview())?;
        let requires = self.record(&[lhs, rhs]);
        let producer = if requires {
            op::matmul { lhs, rhs }
        } else {
            op::leaf
        };
        Ok(self.push(out, requires, producer))
    }
}

/// Accumulates the vector-Jacobian products of a matrix multiplication.
///
/// For `Y = A * B`, `dA = dY * B^T` and `dB = A^T * dY`. Both are computed with
/// plain `f32` accumulation, matching the forward operator's order.
pub(crate) fn backward(
    g: &mut graph<f32>,
    lhs: variableid,
    rhs: variableid,
    gy: &tensor<f32>,
) -> Result<(), autoggraderror> {
    let a = g.valueof(lhs)?.clone();
    let b = g.valueof(rhs)?.clone();
    let ad = a.getshape().getdims();
    let bd = b.getshape().getdims();
    if ad.len() != 2 || bd.len() != 2 {
        return Err(autoggraderror::unsupportedshape {
            op: MATMULOP,
            dims: ad.to_vec(),
        });
    }
    let (m, k, n) = (ad[0], ad[1], bd[1]);
    if gy.getshape().getdims() != [m, n].as_slice() {
        return Err(autoggraderror::unsupportedshape {
            op: MATMULOP,
            dims: gy.getshape().getdims().to_vec(),
        });
    }
    let mk = m
        .checked_mul(k)
        .ok_or(autoggraderror::sizeoverflow { op: MATMULOP })?;
    let kn = k
        .checked_mul(n)
        .ok_or(autoggraderror::sizeoverflow { op: MATMULOP })?;
    let mut da = vec![0.0_f32; mk];
    let mut db = vec![0.0_f32; kn];
    let gd = gy.getdata();
    let av = a.getdata();
    let bv = b.getdata();
    for i in 0..m {
        for j in 0..n {
            let gij = gd[i * n + j];
            for p in 0..k {
                da[i * k + p] += gij * bv[p * n + j];
                db[p * n + j] += av[i * k + p] * gij;
            }
        }
    }
    if g.requiresgrad(lhs)? {
        accumulate(g, lhs, tensor::new(shape::new(vec![m, k])?, da)?)?;
    }
    if g.requiresgrad(rhs)? {
        accumulate(g, rhs, tensor::new(shape::new(vec![k, n])?, db)?)?;
    }
    Ok(())
}
