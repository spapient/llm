use ::tensor::{shape, tensorerror};
use api::backenderror;

/// Failures produced while building or running a neural-network layer.
///
/// The enum separates the distinctions a caller of `nn` must act on: a
/// backend operator failure, a tensor-metadata failure, an operand shape the
/// layer does not accept, a layer configuration that cannot be built, and a
/// parameter name that is duplicated or missing. Underlying errors are carried
/// transparently so no context is lost (ER2).
#[derive(Debug, PartialEq, Eq)]
pub enum nnerror {
    /// A backend operator failed during a forward pass.
    backend(backenderror),

    /// Tensor metadata construction failed while building a layer or output.
    tensor(tensorerror),

    /// A layer received an operand whose shape it does not accept.
    shapemismatch {
        /// The stable layer name.
        layer: &'static str,
        /// The shape the layer requires.
        expected: Vec<usize>,
        /// The shape that was received.
        actual: Vec<usize>,
    },

    /// A layer configuration value is invalid.
    invalidconfig {
        /// The configuration field that was rejected.
        field: &'static str,
        /// A description of why the value is invalid.
        detail: String,
    },

    /// Two parameters were collected under the same name.
    duplicateparameter {
        /// The repeated parameter name.
        name: String,
    },

    /// A parameter name was requested but is not present.
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

impl std::fmt::Display for nnerror {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::backend(f0) => std::fmt::Display::fmt(f0, f),
            Self::tensor(f0) => std::fmt::Display::fmt(f0, f),
            Self::shapemismatch {
                layer,
                expected,
                actual,
            } => write!(
                f,
                "layer {layer} expected shape {expected:?} but received {actual:?}"
            ),
            Self::invalidconfig { field, detail } => {
                write!(f, "invalid configuration for {field}: {detail}")
            }
            Self::duplicateparameter { name } => write!(f, "duplicate parameter name: {name}"),
            Self::missingparameter { name } => write!(f, "missing parameter name: {name}"),
        }
    }
}

impl std::error::Error for nnerror {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::backend(f0) => Some(f0),
            Self::tensor(f0) => Some(f0),
            _ => None,
        }
    }
}

impl From<backenderror> for nnerror {
    fn from(v: backenderror) -> Self {
        Self::backend(v)
    }
}

impl From<tensorerror> for nnerror {
    fn from(v: tensorerror) -> Self {
        Self::tensor(v)
    }
}
