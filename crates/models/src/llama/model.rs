use ::tensor::tensor;
use api::compute;

use crate::causal::causallm;
use crate::error::modelerror;
use crate::llama::block::llamablock;
use crate::llama::config::llamaconfig;
use crate::llama::weights::llamaweights;

/// A Llama-family causal language model: configuration, immutable weights, and
/// the shared block operators.
///
/// The model owns its weights and holds no request-scoped state. A forward pass
/// embeds the tokens, runs every block in order, applies the final
/// normalization, and projects to vocabulary logits. The block operators are
/// built once from the configuration, so a forward pass allocates only the
/// activation tensors the operators produce.
#[derive(Clone, Debug)]
pub struct llamamodel {
    config: llamaconfig,
    weights: llamaweights,
    block: llamablock,
}

impl llamamodel {
    /// Builds a model from a configuration and weights, validating agreement.
    ///
    /// # Errors
    ///
    /// Returns a [`modelerror`] when the weights' layer count, embedding, final
    /// normalization, or output projection disagree with `config`, or when the
    /// configuration-derived block operators cannot be built.
    pub fn new(config: llamaconfig, weights: llamaweights) -> Result<Self, modelerror> {
        if weights.getlayers().len() != config.getnlayers() {
            return Err(modelerror::config(
                "layers",
                "weight layer count must equal nlayers",
            ));
        }
        if weights.getembedding().getvocab() != config.getvocab()
            || weights.getembedding().getwidth() != config.getdim()
        {
            return Err(modelerror::shapemismatch {
                context: "embedding",
                expected: vec![config.getvocab(), config.getdim()],
                actual: vec![
                    weights.getembedding().getvocab(),
                    weights.getembedding().getwidth(),
                ],
            });
        }
        if weights.getoutput().getinfeatures() != config.getdim()
            || weights.getoutput().getoutfeatures() != config.getvocab()
        {
            return Err(modelerror::shapemismatch {
                context: "output",
                expected: vec![config.getdim(), config.getvocab()],
                actual: vec![
                    weights.getoutput().getinfeatures(),
                    weights.getoutput().getoutfeatures(),
                ],
            });
        }
        let block = llamablock::new(&config)?;
        Ok(Self {
            config,
            weights,
            block,
        })
    }

    /// Returns the immutable weights.
    #[must_use]
    pub fn getweights(&self) -> &llamaweights {
        &self.weights
    }
}

impl causallm for llamamodel {
    fn getconfig(&self) -> &llamaconfig {
        &self.config
    }

    fn forward<C: compute>(&self, engine: &C, tokens: &[usize]) -> Result<tensor<f32>, modelerror> {
        let mut x = self.weights.getembedding().forward(engine, tokens)?;
        for layer in self.weights.getlayers() {
            x = self.block.forward(engine, layer, &x)?;
        }
        let normalized = self.weights.getnorm().forward(engine, &x)?;
        Ok(self.weights.getoutput().forward(engine, &normalized)?)
    }
}
