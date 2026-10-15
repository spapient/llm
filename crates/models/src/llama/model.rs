use ::tensor::tensor;
use api::{compute, operand, stepref, submission};

use crate::causal::{causallm, kvcache};
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

    /// Records a full forward pass into a submission, mirroring
    /// [`causallm::forward`] operator for operator, and returns the output
    /// projection's handle.
    ///
    /// The caller submits the whole chain once and resolves the returned
    /// [`stepref`]; every intermediate activation stays on the backend's device.
    pub fn forwardsubmission<S: submission>(
        &self,
        sub: &mut S,
        tokens: &[usize],
    ) -> Result<stepref, modelerror> {
        let mut x = sub.recordembedding(
            operand::input(self.weights.getembedding().getweight().getvalue().getview()),
            tokens,
        )?;
        for layer in self.weights.getlayers() {
            x = self.block.forwardsubmission(sub, layer, x)?;
        }
        let normalized = sub.recordrmsnorm(
            operand::result(x),
            operand::input(self.weights.getnorm().getweight().getvalue().getview()),
            self.weights.getnorm().geteps(),
        )?;
        Ok(sub.recordmatmul(
            operand::result(normalized),
            operand::input(self.weights.getoutput().getweight().getvalue().getview()),
        )?)
    }

    /// Runs one cache-aware forward pass for a single new token, reading and
    /// appending `cache`, and returns the next-token logits.
    ///
    /// The token sits at absolute position `cache.getlen()`; every layer
    /// projects only that token, attends over the cached keys/values plus its
    /// own, and appends its new key/value. The caller must prefill the cache by
    /// processing the prompt tokens in order before the first decode step.
    pub fn forward_cached<C: compute, K: kvcache>(
        &self,
        provider: &C,
        cache: &mut K,
        token: usize,
    ) -> Result<Vec<f32>, modelerror> {
        let position = cache.getlen();
        let mut x = provider.embedding(
            &self.weights.getembedding().getweight().getvalue().getview(),
            &[token],
        )?;
        for (layer, layerweights) in self.weights.getlayers().iter().enumerate() {
            x = self
                .block
                .forward_cached(provider, cache, layerweights, &x, layer, position)?;
        }
        let normalized = provider.rmsnorm(
            &x.getview(),
            &self.weights.getnorm().getweight().getvalue().getview(),
            self.weights.getnorm().geteps(),
        )?;
        let logits = provider.matmul(
            &normalized.getview(),
            &self.weights.getoutput().getweight().getvalue().getview(),
        )?;
        let vocab = logits
            .getshape()
            .getdims()
            .last()
            .copied()
            .ok_or_else(|| modelerror::config("output", "empty logits"))?;
        let data = logits.getdata();
        let start = data
            .len()
            .checked_sub(vocab)
            .ok_or_else(|| modelerror::config("output", "degenerate logits"))?;
        Ok(data[start..].to_vec())
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

    fn forward_cached<C: compute, K: kvcache>(
        &self,
        engine: &C,
        cache: &mut K,
        token: usize,
    ) -> Result<Vec<f32>, modelerror> {
        llamamodel::forward_cached(self, engine, cache, token)
    }
}
