use ::tensor::tensorerror;
use thiserror::Error;

/// Failures produced while configuring an optimizer, validating its operands,
/// or restoring its state.
///
/// The enum separates the distinctions a caller must act on: a tensor-metadata
/// failure propagated from `tensor`, an operand-count mismatch, an operand
/// shape mismatch, an inconsistent or incompatible optimizer state, and an
/// invalid hyperparameter. Underlying tensor errors are carried transparently
/// so no context is lost (ER2).
#[derive(Debug, Error, PartialEq, Eq)]
pub enum opterror {
    /// Tensor metadata construction or validation failed.
    #[error(transparent)]
    tensor(#[from] tensorerror),

    /// The number of parameter tensors differs from the number of gradients.
    #[error("received {params} parameters but {grads} gradients")]
    mismatchedcount {
        /// The parameter count.
        params: usize,
        /// The gradient count.
        grads: usize,
    },

    /// A gradient's shape differs from its parameter's shape.
    #[error("gradient {index} has shape {grad:?} but its parameter has shape {param:?}")]
    shapemismatch {
        /// The zero-based index of the offending parameter/gradient pair.
        index: usize,
        /// The parameter's dimensions.
        param: Vec<usize>,
        /// The gradient's dimensions.
        grad: Vec<usize>,
    },

    /// The optimizer state is inconsistent with itself or with the parameters.
    #[error("invalid optimizer state: {detail}")]
    invalidstate {
        /// Why the state was rejected.
        detail: String,
    },

    /// A hyperparameter is outside its supported domain.
    #[error("invalid hyperparameter {field}: {detail}")]
    invalidhyperparameter {
        /// The rejected hyperparameter name.
        field: &'static str,
        /// Why the value was rejected.
        detail: String,
    },
}

impl opterror {
    /// Builds an invalid-state error from a description.
    #[must_use]
    pub fn state(detail: impl Into<String>) -> Self {
        Self::invalidstate {
            detail: detail.into(),
        }
    }

    /// Builds an invalid-hyperparameter error.
    #[must_use]
    pub fn hyperparameter(field: &'static str, detail: impl Into<String>) -> Self {
        Self::invalidhyperparameter {
            field,
            detail: detail.into(),
        }
    }
}
