use ::tensor::dtype;
use ::tensor::tensorerror;
use optim::opterror;
use thiserror::Error;

/// Failures produced while orchestrating a training run.
///
/// The enum separates the distinctions a caller must act on: a metadata or
/// shape failure propagated from `tensor`, an update failure propagated from
/// `optim`, a persistence failure propagated from `checkpoint`, a cancelled or
/// exhausted run, an inconsistent local training state, and a loss or gradient
/// input that violates the cross-entropy contract. Underlying tensor, optimizer,
/// and checkpoint errors are carried transparently so no diagnostic context is
/// lost (`ER2`).
#[derive(Debug, Error)]
pub enum trainerror {
    /// Tensor metadata construction or validation failed.
    #[error(transparent)]
    tensor(#[from] tensorerror),

    /// An optimizer update or state operation failed.
    #[error(transparent)]
    optim(#[from] opterror),

    /// A checkpoint load, save, or compatibility check failed.
    #[error(transparent)]
    checkpoint(#[from] checkpoint::checkpointerror),

    /// The run was cancelled at a cancellation checkpoint.
    #[error("training was cancelled")]
    cancelled,

    /// The gradient-accumulation window size is not at least one.
    #[error("gradient accumulation must cover at least one micro-batch, found {found}")]
    invalidaccumulation {
        /// The rejected window size.
        found: usize,
    },

    /// No parameters were supplied to the trainer.
    #[error("a trainer requires at least one parameter")]
    emptyparameters,

    /// The configured step budget for the run has been exhausted.
    #[error("training step budget exhausted: {completed} of {max} steps completed")]
    steplimitreached {
        /// The number of completed optimizer steps.
        completed: u64,
        /// The configured maximum number of optimizer steps.
        max: u64,
    },

    /// The loss function returned a different number of gradients than parameters.
    #[error("loss function returned {actual} gradients for {expected} parameters")]
    gradientcountmismatch {
        /// The number of parameters the trainer holds.
        expected: usize,
        /// The number of gradients the loss function returned.
        actual: usize,
    },

    /// A returned gradient's shape differs from its parameter's shape.
    #[error("gradient {index} has shape {grad:?} but its parameter has shape {param:?}")]
    gradientshapemismatch {
        /// The zero-based index of the offending gradient.
        index: usize,
        /// The parameter's dimensions.
        param: Vec<usize>,
        /// The gradient's dimensions.
        grad: Vec<usize>,
    },

    /// The loss function returned a value that is not finite.
    #[error("loss at training step {step} is not finite: {loss}")]
    nonfiniteloss {
        /// The zero-based index of the step whose loss was rejected.
        step: u64,
        /// The rejected loss value.
        loss: f32,
    },

    /// A gradient was added to an already-full accumulation window.
    #[error("gradient accumulation window is already full")]
    accumulatorfull,

    /// The local training state is inconsistent with itself or a loaded artifact.
    #[error("invalid training state: {detail}")]
    invalidstate {
        /// Why the state was rejected.
        detail: String,
    },

    /// Cross-entropy received logits that are not a rank-two matrix.
    #[error("cross-entropy logits must be a rank-two matrix, found shape {dims:?}")]
    lossnotmatrix {
        /// The rejected logits dimensions.
        dims: Vec<usize>,
    },

    /// Cross-entropy received an empty row or class axis.
    #[error("cross-entropy requires at least one row and one class, found shape {dims:?}")]
    lossemptymatrix {
        /// The rejected logits dimensions.
        dims: Vec<usize>,
    },

    /// Cross-entropy received a different number of targets than logit rows.
    #[error("cross-entropy received {rows} rows but {targets} targets")]
    targetcountmismatch {
        /// The number of rows in the logits.
        rows: usize,
        /// The number of supplied targets.
        targets: usize,
    },

    /// A cross-entropy target is outside the class axis.
    #[error("cross-entropy target {index} is {label}, outside the {classes} classes")]
    targetoutofrange {
        /// The zero-based row index of the offending target.
        index: usize,
        /// The rejected target label.
        label: usize,
        /// The number of classes.
        classes: usize,
    },

    /// Cross-entropy produced a non-finite loss or gradient.
    #[error("cross-entropy produced a non-finite loss or gradient")]
    lossnonfinite,

    /// A checkpoint tensor that must hold `f32` data declares another dtype.
    #[error("checkpoint tensor {name:?} uses dtype {dtype:?} instead of f32")]
    unsupportedtensordtype {
        /// The offending tensor name.
        name: String,
        /// The unsupported storage dtype.
        dtype: dtype,
    },
}

impl trainerror {
    /// Builds an invalid-state error from a description.
    #[must_use]
    pub fn state(detail: impl Into<String>) -> Self {
        Self::invalidstate {
            detail: detail.into(),
        }
    }
}
