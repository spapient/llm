use ::tensor::tensor;
use api::compute;

use crate::dictionary::statedict;
use crate::error::nnerror;
use crate::modules::module;
use crate::parameters::parameter;

/// An embedding table that maps token indices to dense vectors.
///
/// The table is `[vocab, width]`. `forward` gathers the rows named by the
/// indices through `api::compute::embedding` and returns `[indices.len(),
/// width]`. Out-of-range indices are rejected by the provider rather than
/// clamped, so an invalid token surfaces as an error.
#[derive(Clone, Debug, PartialEq)]
pub struct embedding {
    weight: parameter,
}

impl embedding {
    /// Builds an embedding layer from a rank-two table.
    ///
    /// # Errors
    ///
    /// Returns [`nnerror::shapemismatch`] when the table is not rank two.
    pub fn new(weight: parameter) -> Result<Self, nnerror> {
        let wd = weight.getshape().getdims();
        if wd.len() != 2 {
            return Err(nnerror::shapemismatch {
                layer: "embedding",
                expected: vec![wd.first().copied().unwrap_or(0), 0],
                actual: wd.to_vec(),
            });
        }
        Ok(Self { weight })
    }

    /// Returns the vocabulary size.
    #[must_use]
    pub fn getvocab(&self) -> usize {
        self.weight.getshape().getdims()[0]
    }

    /// Returns the embedding width.
    #[must_use]
    pub fn getwidth(&self) -> usize {
        self.weight.getshape().getdims()[1]
    }

    /// Returns the table parameter.
    #[must_use]
    pub fn getweight(&self) -> &parameter {
        &self.weight
    }

    /// Gathers the rows named by `indices`.
    ///
    /// # Errors
    ///
    /// Forwards the provider's [`nnerror::backend`] error when an index is out
    /// of range or execution fails.
    pub fn forward<C: compute>(
        &self,
        engine: &C,
        indices: &[usize],
    ) -> Result<tensor<f32>, nnerror> {
        Ok(engine.embedding(&self.weight.getvalue().getview(), indices)?)
    }
}

impl module for embedding {
    fn getname(&self) -> &'static str {
        "embedding"
    }

    fn collect(&self, prefix: &str, dict: &mut statedict) -> Result<(), nnerror> {
        dict.add(prefix, self.weight.clone())
    }
}
