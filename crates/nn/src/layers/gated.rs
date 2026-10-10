use ::tensor::tensor;
use api::compute;

use crate::dictionary::{joinname, statedict};
use crate::error::nnerror;
use crate::layers::linear;
use crate::modules::module;

/// A SwiGLU feed-forward network: `down(silu(gate(x)) * up(x))`.
///
/// The three projections are stored as [`linear`] layers. The gating is the
/// composition of `api::compute::silu` and `api::compute::mul`, so the layer
/// introduces no operator of its own. The shape agreement between the three
/// projections is validated once at construction rather than on every forward
/// pass.
#[derive(Clone, Debug, PartialEq)]
pub struct mlp {
    gate: linear,
    up: linear,
    down: linear,
}

impl mlp {
    /// Builds a SwiGLU network from its gate, up, and down projections.
    ///
    /// # Errors
    ///
    /// Returns [`nnerror::invalidconfig`] unless `gate` and `up` share the same
    /// input and output features, `down` consumes the gate's output features,
    /// and `down` produces the gate's input features.
    pub fn new(gate: linear, up: linear, down: linear) -> Result<Self, nnerror> {
        let dim = gate.getinfeatures();
        let hidden = gate.getoutfeatures();
        if up.getinfeatures() != dim || up.getoutfeatures() != hidden {
            return Err(nnerror::config(
                "up",
                "up features must match the gate projection",
            ));
        }
        if down.getinfeatures() != hidden || down.getoutfeatures() != dim {
            return Err(nnerror::config(
                "down",
                "down features must invert the gate projection",
            ));
        }
        Ok(Self { gate, up, down })
    }

    /// Returns the model dimension.
    #[must_use]
    pub fn getfeatures(&self) -> usize {
        self.gate.getinfeatures()
    }

    /// Returns the hidden dimension.
    #[must_use]
    pub fn gethidden(&self) -> usize {
        self.gate.getoutfeatures()
    }

    /// Returns the gate projection.
    #[must_use]
    pub fn getgate(&self) -> &linear {
        &self.gate
    }

    /// Returns the up projection.
    #[must_use]
    pub fn getup(&self) -> &linear {
        &self.up
    }

    /// Returns the down projection.
    #[must_use]
    pub fn getdown(&self) -> &linear {
        &self.down
    }

    /// Applies the network to `input`.
    ///
    /// # Errors
    ///
    /// Forwards the shape and provider errors of the three projections and the
    /// `silu`/`mul` operators.
    pub fn forward<C: compute>(
        &self,
        engine: &C,
        input: &tensor<f32>,
    ) -> Result<tensor<f32>, nnerror> {
        let g = self.gate.forward(engine, input)?;
        let u = self.up.forward(engine, input)?;
        let s = engine.silu(&g.getview())?;
        let gated = engine.mul(&s.getview(), &u.getview())?;
        self.down.forward(engine, &gated)
    }
}

impl module for mlp {
    fn getname(&self) -> &'static str {
        "mlp"
    }

    fn collect(&self, prefix: &str, dict: &mut statedict) -> Result<(), nnerror> {
        self.gate.collect(&joinname(prefix, "gate"), dict)?;
        self.up.collect(&joinname(prefix, "up"), dict)?;
        self.down.collect(&joinname(prefix, "down"), dict)
    }
}
