use ::tensor::tensorerror;
use api::backenderror;
use nn::nnerror;

/// Failures produced while building or running a concrete model.
///
/// The enum carries the distinctions a caller must act on: a backend operator
/// failure, a tensor-metadata failure, a layer failure from `nn`, an invalid
/// architecture configuration, a weight whose shape does not match the
/// configuration, an unknown architecture name, and a weight that is missing.
/// Underlying errors are carried transparently so no context is lost (ER2).
#[derive(Debug, PartialEq, Eq)]
pub enum modelerror {
    /// A backend operator failed during a forward pass.
    backend(backenderror),

    /// Tensor metadata construction failed.
    tensor(tensorerror),

    /// A neural-network layer failed.
    nn(nnerror),

    /// An architecture configuration value is invalid.
    invalidconfig {
        /// The configuration field that was rejected.
        field: &'static str,
        /// A description of why the value is invalid.
        detail: String,
    },

    /// A weight tensor has a shape the configuration does not allow.
    shapemismatch {
        /// The weight name or role that was checked.
        context: &'static str,
        /// The shape the configuration requires.
        expected: Vec<usize>,
        /// The shape that was received.
        actual: Vec<usize>,
    },

    /// An architecture name does not correspond to a known architecture.
    unknownarchitecture {
        /// The unrecognized architecture name.
        name: String,
    },

    /// A required weight is absent.
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

impl std::fmt::Display for modelerror {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::backend(f0) => std::fmt::Display::fmt(f0, f),
            Self::tensor(f0) => std::fmt::Display::fmt(f0, f),
            Self::nn(f0) => std::fmt::Display::fmt(f0, f),
            Self::invalidconfig { field, detail } => {
                write!(f, "invalid configuration for {field}: {detail}")
            }
            Self::shapemismatch {
                context,
                expected,
                actual,
            } => write!(
                f,
                "weight {context} expected shape {expected:?} but received {actual:?}"
            ),
            Self::unknownarchitecture { name } => write!(f, "unknown architecture: {name}"),
            Self::missingweight { name } => write!(f, "missing weight: {name}"),
        }
    }
}

impl std::error::Error for modelerror {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::backend(f0) => Some(f0),
            Self::tensor(f0) => Some(f0),
            Self::nn(f0) => Some(f0),
            _ => None,
        }
    }
}

impl From<backenderror> for modelerror {
    fn from(v: backenderror) -> Self {
        Self::backend(v)
    }
}

impl From<tensorerror> for modelerror {
    fn from(v: tensorerror) -> Self {
        Self::tensor(v)
    }
}

impl From<nnerror> for modelerror {
    fn from(v: nnerror) -> Self {
        Self::nn(v)
    }
}
