use ::tensor::tensor;
use api::compute;

use crate::arena::graph;
use crate::error::autoggraderror;
use crate::node::{op, variableid};
use crate::ops::accumulate;

/// The stable name reported by the softmax operator.
const SOFTMAXOP: &str = "softmax";

impl graph<f32> {
    /// Differentiable softmax over the last axis of `input`.
    ///
    /// # Errors
    ///
    /// Returns [`autoggraderror::unknownvariable`] for a missing input, or the
    /// provider's error (for example an empty last axis) when execution fails.
    pub fn softmax<C: compute>(
        &mut self,
        engine: &C,
        input: variableid,
    ) -> Result<variableid, autoggraderror> {
        let x = self.valueof(input)?.clone();
        let out = engine.softmax(&x.getview())?;
        let requires = self.record(&[input]);
        let producer = if requires {
            op::softmax { input }
        } else {
            op::leaf
        };
        Ok(self.push(out, requires, producer))
    }
}

/// Accumulates the vector-Jacobian product of a softmax.
///
/// With output `y` and upstream gradient `dY`, the row-wise product is
/// `dX_j = y_j * (dY_j - sum_i dY_i * y_i)`, so the row's gradients sum to zero
/// as a softmax Jacobian requires.
pub(crate) fn backward(
    g: &mut graph<f32>,
    input: variableid,
    gy: &tensor<f32>,
    y: &tensor<f32>,
) -> Result<(), autoggraderror> {
    let x = g.valueof(input)?.clone();
    if x.getshape().getdims() != gy.getshape().getdims()
        || x.getshape().getdims() != y.getshape().getdims()
    {
        return Err(autoggraderror::unsupportedshape {
            op: SOFTMAXOP,
            dims: gy.getshape().getdims().to_vec(),
        });
    }
    let dims = x.getshape().getdims();
    let width = *dims.last().ok_or(autoggraderror::unsupportedshape {
        op: SOFTMAXOP,
        dims: dims.to_vec(),
    })?;
    if width == 0 {
        return Err(autoggraderror::unsupportedshape {
            op: SOFTMAXOP,
            dims: dims.to_vec(),
        });
    }
    let gv = gy.getdata();
    let yv = y.getdata();
    let mut dx = vec![0.0_f32; x.getlen()];
    for ((out, grown), yrow) in dx
        .chunks_mut(width)
        .zip(gv.chunks(width))
        .zip(yv.chunks(width))
    {
        let dot: f32 = grown.iter().zip(yrow).map(|(&gv_i, &y_i)| gv_i * y_i).sum();
        for ((slot, &gv_i), &y_i) in out.iter_mut().zip(grown).zip(yrow) {
            *slot = y_i * (gv_i - dot);
        }
    }
    if g.requiresgrad(input)? {
        accumulate(g, input, tensor::new(x.getshape().clone(), dx)?)?;
    }
    Ok(())
}
