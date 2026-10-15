use ::tensor::tensor;
use api::compute;

use crate::error::modelerror;
use crate::llama::llamaconfig;

/// A causal language model: an architecture that maps token ids to next-token
/// logits.
///
/// A model owns its weights immutably and holds no request-scoped state. A
/// forward pass takes only a compute provider and the token ids, so the same
/// model value can serve independent requests without sharing mutable state
/// (OW2). The returned tensor has shape `[tokens.len(), vocab]`.
pub trait causallm {
    /// Returns the architecture configuration.
    fn getconfig(&self) -> &llamaconfig;

    /// Runs one forward pass over `tokens`.
    ///
    /// # Errors
    ///
    /// Returns a [`modelerror`] when a token is out of range, an operand shape
    /// is inconsistent with the configuration, or a provider operation fails.
    fn forward<C: compute>(&self, engine: &C, tokens: &[usize]) -> Result<tensor<f32>, modelerror>;

    /// Runs one cache-aware forward pass for a single new token.
    ///
    /// The token sits at absolute position [`kvcache::getlen`], and the call
    /// appends that position's key and value to `cache`. The caller prefill's
    /// the cache by processing the prompt tokens in order before the first
    /// decode step. Returns the next-token logits.
    ///
    /// # Errors
    ///
    /// Returns a [`modelerror`] when a token is out of range, an operand shape
    /// is inconsistent with the configuration, or a provider operation fails.
    fn forward_cached<C: compute, K: kvcache>(
        &self,
        engine: &C,
        cache: &mut K,
        token: usize,
    ) -> Result<Vec<f32>, modelerror>;
}

/// A key/value cache for incremental decoding, owned by `models` and
/// implemented by the inference layer.
///
/// The contract is defined here so `models` owns the cache abstraction it will
/// consume, while the request-scoped buffer and its lifecycle belong to the
/// caller that implements it (`infer`, P7). Implementations hold one key and one
/// value tensor per layer, each of shape `[cached_len, kv_heads * head_dim]`.
///
/// This stage defines the contract only: the deterministic forward pass
/// recomputes the full sequence and does not read or write a cache yet. The
/// cache is the measured optimization a later stage adds against that baseline.
pub trait kvcache {
    /// Returns the number of transformer layers the cache holds.
    fn getlayers(&self) -> usize;

    /// Returns the number of cached positions.
    fn getlen(&self) -> usize;

    /// Returns the maximum number of positions the cache can hold.
    fn getmaxseq(&self) -> usize;

    /// Appends one position's key and value to `layer`.
    ///
    /// # Errors
    ///
    /// Returns a [`modelerror`] when `layer` is out of range, a tensor shape is
    /// inconsistent with the layer, or the cache is full.
    fn append(
        &mut self,
        layer: usize,
        key: &tensor<f32>,
        value: &tensor<f32>,
    ) -> Result<(), modelerror>;

    /// Returns the accumulated keys of `layer`, or `None` when it is empty.
    fn keys(&self, layer: usize) -> Option<&tensor<f32>>;

    /// Returns the accumulated values of `layer`, or `None` when it is empty.
    fn values(&self, layer: usize) -> Option<&tensor<f32>>;
}
