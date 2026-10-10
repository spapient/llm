use ::tensor::tensor;
use api::compute;
use nn::{attention, rotary};

use crate::error::modelerror;
use crate::llama::config::llamaconfig;
use crate::llama::weights::llamalayerweights;

/// The configuration-derived operators of one Llama transformer block.
///
/// A block's attention configuration and rotary embeddings depend only on the
/// architecture, not on the weights, so they are built once and shared by every
/// layer. The per-layer weights are passed to [`Self::forward`], keeping the
/// immutable weights separate from the operator configuration (OW4).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct llamablock {
    attention: attention,
    rotaryq: rotary,
    rotaryk: rotary,
}

impl llamablock {
    /// Builds the block operators from `config`.
    ///
    /// # Errors
    ///
    /// Returns a [`modelerror`] when the head layout or rotary base is invalid
    /// for the configuration.
    pub fn new(config: &llamaconfig) -> Result<Self, modelerror> {
        Ok(Self {
            attention: attention::new(config.getnheads(), config.getnkvheads(), true)?,
            rotaryq: rotary::new(config.getnheads(), config.getropebase())?,
            rotaryk: rotary::new(config.getnkvheads(), config.getropebase())?,
        })
    }

    /// Runs one pre-norm transformer block over `x` of shape `[seq, dim]`.
    ///
    /// # Errors
    ///
    /// Returns a [`modelerror`] when `x` is not rank two with the model
    /// dimension, or when a projection, normalization, rotary, attention, or
    /// residual operation fails.
    pub fn forward<C: compute>(
        &self,
        engine: &C,
        weights: &llamalayerweights,
        x: &tensor<f32>,
    ) -> Result<tensor<f32>, modelerror> {
        let dims = x.getshape().getdims();
        if dims.len() != 2 {
            return Err(modelerror::config(
                "block",
                "input must be a rank-two [seq, dim] tensor",
            ));
        }
        let normalized = weights.getattnnorm().forward(engine, x)?;
        let q = weights.getwq().forward(engine, &normalized)?;
        let k = weights.getwk().forward(engine, &normalized)?;
        let v = weights.getwv().forward(engine, &normalized)?;
        let q = self.rotaryq.forward(engine, &q)?;
        let k = self.rotaryk.forward(engine, &k)?;
        let attended = self.attention.forward(engine, &q, &k, &v)?;
        let projected = weights.getwo().forward(engine, &attended)?;
        let residual = engine.add(&x.getview(), &projected.getview())?;
        let normalized = weights.getffnnorm().forward(engine, &residual)?;
        let feedforward = weights.getmlp().forward(engine, &normalized)?;
        Ok(engine.add(&residual.getview(), &feedforward.getview())?)
    }
}
