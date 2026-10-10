use ::tensor::tensor;
use api::compute;

use crate::arena::graph;
use crate::error::autoggraderror;
use crate::node::{op, variableid};
use crate::ops::accumulate;

/// The stable name reported by the ReLU operator.
const RELUOP: &str = "relu";

/// The stable name reported by the SiLU operator.
const SILUOP: &str = "silu";

/// The stable name reported by the tanh-GELU operator.
const GELUTANHOP: &str = "gelutanh";

/// `sqrt(2 / pi)`, the tanh-GELU scale.
const SQRT2OVERPI: f32 = 0.797_884_6;

/// The cubic coefficient of the tanh GELU approximation.
const GELUCUBIC: f32 = 0.044_715;

impl graph<f32> {
    /// Differentiable rectified linear unit.
    ///
    /// # Errors
    ///
    /// Returns [`autoggraderror::unknownvariable`] for a missing input, or
    /// [`autoggraderror::tensor`] if the output tensor cannot be built.
    pub fn relu(&mut self, input: variableid) -> Result<variableid, autoggraderror> {
        let x = self.valueof(input)?.clone();
        let data: Vec<f32> = x.getdata().iter().map(|&v| v.max(0.0)).collect();
        let out = tensor::new(x.getshape().clone(), data)?;
        self.finishunary(input, out, |i| op::relu { input: i })
    }

    /// Differentiable sigmoid linear unit, forwarded to the provider.
    ///
    /// # Errors
    ///
    /// Returns [`autoggraderror::unknownvariable`] for a missing input, or the
    /// provider's error when execution fails.
    pub fn silu<C: compute>(
        &mut self,
        engine: &C,
        input: variableid,
    ) -> Result<variableid, autoggraderror> {
        let x = self.valueof(input)?.clone();
        let out = engine.silu(&x.getview())?;
        self.finishunary(input, out, |i| op::silu { input: i })
    }

    /// Differentiable tanh-approximation GELU.
    ///
    /// # Errors
    ///
    /// Returns [`autoggraderror::unknownvariable`] for a missing input, or
    /// [`autoggraderror::tensor`] if the output tensor cannot be built.
    pub fn gelutanh(&mut self, input: variableid) -> Result<variableid, autoggraderror> {
        let x = self.valueof(input)?.clone();
        let data: Vec<f32> = x.getdata().iter().map(|&v| gelutanhscalar(v)).collect();
        let out = tensor::new(x.getshape().clone(), data)?;
        self.finishunary(input, out, |i| op::gelutanh { input: i })
    }

    fn finishunary(
        &mut self,
        input: variableid,
        out: tensor<f32>,
        kind: impl FnOnce(variableid) -> op,
    ) -> Result<variableid, autoggraderror> {
        let requires = self.record(&[input]);
        let producer = if requires { kind(input) } else { op::leaf };
        Ok(self.push(out, requires, producer))
    }
}

/// Evaluates the tanh GELU scalar `0.5 * x * (1 + tanh(u))` with
/// `u = sqrt(2/pi) * (x + 0.044715 * x^3)`.
#[must_use]
fn gelutanhscalar(x: f32) -> f32 {
    let u = SQRT2OVERPI * (x + GELUCUBIC * x * x * x);
    0.5 * x * (1.0 + u.tanh())
}

/// Accumulates the ReLU vector-Jacobian product: `dX = dY * (x > 0)`.
pub(crate) fn backwardrelu(
    g: &mut graph<f32>,
    input: variableid,
    gy: &tensor<f32>,
) -> Result<(), autoggraderror> {
    let x = g.valueof(input)?.clone();
    checkgradshape(&x, gy, RELUOP)?;
    if !g.requiresgrad(input)? {
        return Ok(());
    }
    let data: Vec<f32> = gy
        .getdata()
        .iter()
        .zip(x.getdata())
        .map(|(&gv, &xv)| if xv > 0.0 { gv } else { 0.0 })
        .collect();
    accumulate(g, input, tensor::new(x.getshape().clone(), data)?)
}

/// Accumulates the SiLU vector-Jacobian product.
///
/// With `s = sigmoid(x)`, `silu'(x) = s * (1 + x * (1 - s))`.
pub(crate) fn backwardsilu(
    g: &mut graph<f32>,
    input: variableid,
    gy: &tensor<f32>,
) -> Result<(), autoggraderror> {
    let x = g.valueof(input)?.clone();
    checkgradshape(&x, gy, SILUOP)?;
    if !g.requiresgrad(input)? {
        return Ok(());
    }
    let data: Vec<f32> = gy
        .getdata()
        .iter()
        .zip(x.getdata())
        .map(|(&gv, &xv)| {
            let s = 1.0 / (1.0 + (-xv).exp());
            gv * s * (1.0 + xv * (1.0 - s))
        })
        .collect();
    accumulate(g, input, tensor::new(x.getshape().clone(), data)?)
}

/// Accumulates the tanh-GELU vector-Jacobian product.
pub(crate) fn backwardgelutanh(
    g: &mut graph<f32>,
    input: variableid,
    gy: &tensor<f32>,
) -> Result<(), autoggraderror> {
    let x = g.valueof(input)?.clone();
    checkgradshape(&x, gy, GELUTANHOP)?;
    if !g.requiresgrad(input)? {
        return Ok(());
    }
    let data: Vec<f32> = gy
        .getdata()
        .iter()
        .zip(x.getdata())
        .map(|(&gv, &xv)| {
            let u = SQRT2OVERPI * (xv + GELUCUBIC * xv * xv * xv);
            let t = u.tanh();
            let du = SQRT2OVERPI * (1.0 + 3.0 * GELUCUBIC * xv * xv);
            let d = 0.5 * (1.0 + t) + 0.5 * xv * (1.0 - t * t) * du;
            gv * d
        })
        .collect();
    accumulate(g, input, tensor::new(x.getshape().clone(), data)?)
}

/// Rejects a gradient whose shape differs from the input it flows into.
fn checkgradshape(
    x: &tensor<f32>,
    gy: &tensor<f32>,
    op: &'static str,
) -> Result<(), autoggraderror> {
    if x.getshape().getdims() != gy.getshape().getdims() {
        return Err(autoggraderror::unsupportedshape {
            op,
            dims: gy.getshape().getdims().to_vec(),
        });
    }
    Ok(())
}
