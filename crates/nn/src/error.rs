use ::tensor::{shape, tensorerror};
use api::backenderror;
use thiserror::Error;

/// Failures produced while building or running a neural-network layer.
///
/// The enum separates the distinctions a caller of `nn` must act on: a
/// backend operator failure, a tensor-metadata failure, an operand shape the
/// layer does not accept, a layer configuration that cannot be built, and a
/// parameter name that is duplicated or missing. Underlying errors are carried
/// transparently so no context is lost (ER2).
#[derive(Debug, Error, PartialEq, Eq)]
pub enum nnerror {
    /// A backend operator failed during a forward pass.
    #[error(transparent)]
    backend(#[from] backenderror),

    /// Tensor metadata construction failed while building a layer or output.
    #[error(transparent)]
    tensor(#[from] tensorerror),

    /// A layer received an operand whose shape it does not accept.
    #[error("layer {layer} expected shape {expected:?} but received {actual:?}")]
    shapemismatch {
        /// The stable layer name.
        layer: &'static str,
        /// The shape the layer requires.
        expected: Vec<usize>,
        /// The shape that was received.
        actual: Vec<usize>,
    },

    /// A layer configuration value is invalid.
    #[error("invalid configuration for {field}: {detail}")]
    invalidconfig {
        /// The configuration field that was rejected.
        field: &'static str,
        /// A description of why the value is invalid.
        detail: String,
    },

    /// Two parameters were collected under the same name.
    #[error("duplicate parameter name: {name}")]
    duplicateparameter {
        /// The repeated parameter name.
        name: String,
    },

    /// A parameter name was requested but is not present.
    #[error("missing parameter name: {name}")]
    missingparameter {
        /// The absent parameter name.
        name: String,
    },
}

impl nnerror {
    /// Builds a shape mismatch from two shapes.
    #[must_use]
    pub fn shapes(layer: &'static str, expected: &shape, actual: &shape) -> Self {
        Self::shapemismatch {
            layer,
            expected: expected.getdims().to_vec(),
            actual: actual.getdims().to_vec(),
        }
    }

    /// Builds an invalid-configuration error.
    #[must_use]
    pub fn config(field: &'static str, detail: impl Into<String>) -> Self {
        Self::invalidconfig {
            field,
            detail: detail.into(),
        }
    }
}
