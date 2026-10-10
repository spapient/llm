use ::tensor::tensor;
use api::compute;

use crate::arena::graph;
use crate::error::autoggraderror;
use crate::node::{op, variableid};
use crate::ops::accumulate;

/// The stable name reported by the RMS normalization operator.
const RMSNORMOP: &str = "rmsnorm";

impl graph<f32> {
    /// Differentiable RMS normalization of `input` by `weight`.
    ///
    /// `input` is normalized over its last axis and scaled by `weight`, whose
    /// length must equal that axis. `eps` is the numerical-stability epsilon.
    ///
    /// # Errors
    ///
    /// Returns [`autoggraderror::unknownvariable`] for a missing operand, or the
    /// provider's error when the weight length or last axis is invalid.
    pub fn rmsnorm<C: compute>(
        &mut self,
        engine: &C,
        input: variableid,
        weight: variableid,
        eps: f32,
    ) -> Result<variableid, autoggraderror> {
        let x = self.valueof(input)?.clone();
        let w = self.valueof(weight)?.clone();
        let out = engine.rmsnorm(&x.getview(), &w.getview(), eps)?;
        let requires = self.record(&[input, weight]);
        let producer = if requires {
            op::rmsnorm { input, weight, eps }
        } else {
            op::leaf
        };
        Ok(self.push(out, requires, producer))
    }
}

/// Accumulates the vector-Jacobian products of an RMS normalization.
///
/// With `inv = 1 / sqrt(mean(x^2) + eps)`, output `y_i = x_i * inv * w_i`, and
/// `s = sum_i dY_i * w_i * x_i`, the input and weight gradients are
///
/// ```text
/// dX_j = inv * (dY_j * w_j) - inv^3 * s * x_j / n
/// dW_j = dY_j * x_j * inv
/// ```
pub(crate) fn backward(
    g: &mut graph<f32>,
    input: variableid,
    weight: variableid,
    eps: f32,
    gy: &tensor<f32>,
) -> Result<(), autoggraderror> {
    let x = g.valueof(input)?.clone();
    let w = g.valueof(weight)?.clone();
    let dims = x.getshape().getdims();
    let width = *dims.last().ok_or(autoggraderror::unsupportedshape {
        op: RMSNORMOP,
        dims: dims.to_vec(),
    })?;
    if width == 0 || w.getlen() != width || gy.getshape().getdims() != dims {
        return Err(autoggraderror::unsupportedshape {
            op: RMSNORMOP,
            dims: dims.to_vec(),
        });
    }
    let n = width as f32;
    let xv = x.getdata();
    let wv = w.getdata();
    let gv = gy.getdata();
    let mut dx = vec![0.0_f32; x.getlen()];
    let mut dw = vec![0.0_f32; width];
    for (row, (xrow, grow)) in xv.chunks(width).zip(gv.chunks(width)).enumerate() {
        let sumsq: f32 = xrow.iter().map(|&v| v * v).sum();
        let inv = 1.0 / (sumsq / n + eps).sqrt();
        let s: f32 = xrow
            .iter()
            .zip(wv)
            .zip(grow)
            .map(|((&x_i, &w_i), &g_i)| g_i * w_i * x_i)
            .sum();
        let base = row * width;
        for j in 0..width {
            dx[base + j] = inv * (grow[j] * wv[j]) - inv * inv * inv * s * xrow[j] / n;
            dw[j] += grow[j] * xrow[j] * inv;
        }
    }
    if g.requiresgrad(input)? {
        accumulate(g, input, tensor::new(x.getshape().clone(), dx)?)?;
    }
    if g.requiresgrad(weight)? {
        accumulate(g, weight, tensor::new(w.getshape().clone(), dw)?)?;
    }
    Ok(())
}
