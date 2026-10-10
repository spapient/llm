use ::tensor::tensor;
use api::compute;

use crate::dictionary::statedict;
use crate::error::nnerror;
use crate::modules::module;

/// Multi-head scaled dot-product attention with grouped-query heads.
///
/// The layer owns no parameters: it is the configuration of the attention
/// operation, not a projection. The query is `[seq, heads * headdim]`; the key
/// and value are `[seq, kvheads * headdim]`. The grouped-query, causal, and
/// scaling behavior is the composition performed by
/// `api::compute::attention` (matrix multiply, softmax, causal mask, and
/// grouped heads); this layer validates the head layout and forwards the call
/// rather than reimplementing the same math beside the backend.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct attention {
    heads: usize,
    kvheads: usize,
    causal: bool,
}

impl attention {
    /// Builds an attention configuration.
    ///
    /// # Errors
    ///
    /// Returns [`nnerror::invalidconfig`] when `heads` or `kvheads` is zero or
    /// `heads` is not a multiple of `kvheads`.
    pub fn new(heads: usize, kvheads: usize, causal: bool) -> Result<Self, nnerror> {
        if heads == 0 || kvheads == 0 {
            return Err(nnerror::config(
                "heads",
                "heads and kvheads must be non-zero",
            ));
        }
        if !heads.is_multiple_of(kvheads) {
            return Err(nnerror::config(
                "heads",
                "heads must be a multiple of kvheads",
            ));
        }
        Ok(Self {
            heads,
            kvheads,
            causal,
        })
    }

    /// Returns the query head count.
    #[must_use]
    pub fn getheads(&self) -> usize {
        self.heads
    }

    /// Returns the key/value head count.
    #[must_use]
    pub fn getkvheads(&self) -> usize {
        self.kvheads
    }

    /// Returns whether the causal mask is applied.
    #[must_use]
    pub fn getcausal(&self) -> bool {
        self.causal
    }

    /// Applies attention to a query, key, and value.
    ///
    /// # Errors
    ///
    /// Returns [`nnerror::shapemismatch`] when any operand is not rank two, the
    /// sequence lengths differ, the head counts are inconsistent with the
    /// operands, or the query last axis is not a multiple of `heads`; or
    /// forwards the provider's error.
    pub fn forward<C: compute>(
        &self,
        engine: &C,
        query: &tensor<f32>,
        key: &tensor<f32>,
        value: &tensor<f32>,
    ) -> Result<tensor<f32>, nnerror> {
        let qd = query.getshape().getdims();
        let kd = key.getshape().getdims();
        let vd = value.getshape().getdims();
        if qd.len() != 2 || kd.len() != 2 || vd.len() != 2 {
            return Err(nnerror::shapemismatch {
                layer: "attention",
                expected: vec![qd.first().copied().unwrap_or(0), self.heads * 2],
                actual: qd.to_vec(),
            });
        }
        if kd != vd || kd[0] != qd[0] || !qd[1].is_multiple_of(self.heads) {
            return Err(nnerror::shapemismatch {
                layer: "attention",
                expected: vec![qd[0], self.heads * 2],
                actual: qd.to_vec(),
            });
        }
        let headdim = qd[1] / self.heads;
        let kwidth = self
            .kvheads
            .checked_mul(headdim)
            .ok_or_else(|| nnerror::config("kvheads", "head dimension overflow"))?;
        if headdim == 0 || kd[1] != kwidth {
            return Err(nnerror::shapemismatch {
                layer: "attention",
                expected: vec![qd[0], kwidth],
                actual: kd.to_vec(),
            });
        }
        Ok(engine.attention(
            &query.getview(),
            &key.getview(),
            &value.getview(),
            self.heads,
            self.kvheads,
            self.causal,
        )?)
    }
}

impl module for attention {
    fn getname(&self) -> &'static str {
        "attention"
    }

    fn collect(&self, _prefix: &str, _dict: &mut statedict) -> Result<(), nnerror> {
        Ok(())
    }
}
