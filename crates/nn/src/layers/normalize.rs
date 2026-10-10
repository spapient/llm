use ::tensor::tensor;
use api::compute;

use crate::dictionary::statedict;
use crate::error::nnerror;
use crate::modules::module;
use crate::parameters::parameter;

/// The default numerical-stability epsilon for RMS normalization.
pub const DEFAULTEPS: f32 = 1.0e-5;

/// An RMS normalization layer over the last axis, scaled by a learned weight.
///
/// The weight is `[features]` and must equal the last dimension of the input.
/// The normalization and scaling are performed by `api::compute::rmsnorm`; this
/// layer owns the weight, the epsilon, and the input-shape validation.
#[derive(Clone, Debug, PartialEq)]
pub struct rmsnorm {
    weight: parameter,
    eps: f32,
}

impl rmsnorm {
    /// Builds an RMS normalization layer.
    ///
    /// # Errors
    ///
    /// Returns [`nnerror::shapemismatch`] when the weight is not rank one, or
    /// [`nnerror::invalidconfig`] when `eps` is negative or not finite.
    pub fn new(weight: parameter, eps: f32) -> Result<Self, nnerror> {
        let wd = weight.getshape().getdims();
        if wd.len() != 1 {
            return Err(nnerror::shapemismatch {
                layer: "rmsnorm",
                expected: vec![wd.first().copied().unwrap_or(0)],
                actual: wd.to_vec(),
            });
        }
        if !eps.is_finite() || eps < 0.0 {
            return Err(nnerror::config("eps", "must be finite and non-negative"));
        }
        Ok(Self { weight, eps })
    }

    /// Returns the feature count.
    #[must_use]
    pub fn getfeatures(&self) -> usize {
        self.weight.getshape().getdims()[0]
    }

    /// Returns the epsilon.
    #[must_use]
    pub fn geteps(&self) -> f32 {
        self.eps
    }

    /// Returns the weight parameter.
    #[must_use]
    pub fn getweight(&self) -> &parameter {
        &self.weight
    }

    /// Normalizes `input` over its last axis.
    ///
    /// # Errors
    ///
    /// Returns [`nnerror::shapemismatch`] when the last dimension is not the
    /// feature count, or forwards the provider's error.
    pub fn forward<C: compute>(
        &self,
        engine: &C,
        input: &tensor<f32>,
    ) -> Result<tensor<f32>, nnerror> {
        let dims = input.getshape().getdims();
        let features = self.getfeatures();
        if dims[dims.len() - 1] != features {
            let mut expected = dims.to_vec();
            expected[dims.len() - 1] = features;
            return Err(nnerror::shapemismatch {
                layer: "rmsnorm",
                expected,
                actual: dims.to_vec(),
            });
        }
        Ok(engine.rmsnorm(
            &input.getview(),
            &self.weight.getvalue().getview(),
            self.eps,
        )?)
    }
}

impl module for rmsnorm {
    fn getname(&self) -> &'static str {
        "rmsnorm"
    }

    fn collect(&self, prefix: &str, dict: &mut statedict) -> Result<(), nnerror> {
        dict.add(prefix, self.weight.clone())
    }
}
