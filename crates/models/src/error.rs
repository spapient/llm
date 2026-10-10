use ::tensor::tensorerror;
use api::backenderror;
use nn::nnerror;
use thiserror::Error;

/// Failures produced while building or running a concrete model.
///
/// The enum carries the distinctions a caller must act on: a backend operator
/// failure, a tensor-metadata failure, a layer failure from `nn`, an invalid
/// architecture configuration, a weight whose shape does not match the
/// configuration, an unknown architecture name, and a weight that is missing.
/// Underlying errors are carried transparently so no context is lost (ER2).
#[derive(Debug, Error, PartialEq, Eq)]
pub enum modelerror {
    /// A backend operator failed during a forward pass.
    #[error(transparent)]
    backend(#[from] backenderror),

    /// Tensor metadata construction failed.
    #[error(transparent)]
    tensor(#[from] tensorerror),

    /// A neural-network layer failed.
    #[error(transparent)]
    nn(#[from] nnerror),

    /// An architecture configuration value is invalid.
    #[error("invalid configuration for {field}: {detail}")]
    invalidconfig {
        /// The configuration field that was rejected.
        field: &'static str,
        /// A description of why the value is invalid.
        detail: String,
    },

    /// A weight tensor has a shape the configuration does not allow.
    #[error("weight {context} expected shape {expected:?} but received {actual:?}")]
    shapemismatch {
        /// The weight name or role that was checked.
        context: &'static str,
        /// The shape the configuration requires.
        expected: Vec<usize>,
        /// The shape that was received.
        actual: Vec<usize>,
    },

    /// An architecture name does not correspond to a known architecture.
    #[error("unknown architecture: {name}")]
    unknownarchitecture {
        /// The unrecognized architecture name.
        name: String,
    },

    /// A required weight is absent.
    #[error("missing weight: {name}")]
    missingweight {
        /// The absent weight name.
        name: String,
    },
}

impl modelerror {
    /// Builds an invalid-configuration error.
    #[must_use]
    pub fn config(field: &'static str, detail: impl Into<String>) -> Self {
        Self::invalidconfig {
            field,
            detail: detail.into(),
        }
    }
}
