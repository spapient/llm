//! Differentiable operators.
//!
//! Each submodule adds forward operators to [`graph`](crate::graph) and a
//! reverse-pass function the backward walk calls. The forward methods run
//! through an [`api::compute`](api::compute) provider; the reverse functions use
//! plain `f32` arithmetic on the recorded values.

pub(crate) mod activation;
pub(crate) mod elementwise;
pub(crate) mod matmul;
pub(crate) mod normalization;
pub(crate) mod softmax;

use ::tensor::tensor;

use crate::arena::graph;
use crate::error::autoggraderror;
use crate::node::variableid;

/// Adds `delta` to the accumulated gradient of `id`, creating it if absent.
///
/// Gradient accumulation is what lets one value feed several consumers: each
/// consumer's vector-Jacobian product is summed into the same slot.
pub(crate) fn accumulate(
    g: &mut graph<f32>,
    id: variableid,
    delta: tensor<f32>,
) -> Result<(), autoggraderror> {
    match g.getgrad(id).cloned() {
        Some(existing) => {
            if existing.getshape().getdims() != delta.getshape().getdims() {
                return Err(autoggraderror::unsupportedshape {
                    op: "accumulate",
                    dims: delta.getshape().getdims().to_vec(),
                });
            }
            let mut data = existing.getdata().to_vec();
            for (slot, &addend) in data.iter_mut().zip(delta.getdata()) {
                *slot += addend;
            }
            let sum = tensor::new(existing.getshape().clone(), data)?;
            g.setgrad(id, sum)
        }
        None => g.setgrad(id, delta),
    }
}

/// Builds a tensor of `1.0` with the same shape as `source`.
pub(crate) fn oneslike(source: &tensor<f32>) -> Result<tensor<f32>, autoggraderror> {
    let count = source.getshape().getelementcount()?;
    Ok(tensor::new(
        source.getshape().clone(),
        vec![1.0_f32; count],
    )?)
}

/// Returns whether two operands have exactly equal shapes.
pub(crate) fn sameshape(a: &tensor<f32>, b: &tensor<f32>) -> bool {
    a.getshape().getdims() == b.getshape().getdims()
}
