use ::tensor::dtype;
use ::tensor::tensorerror;
use optim::opterror;

/// Failures produced while orchestrating a training run.
///
/// The enum separates the distinctions a caller must act on: a metadata or
/// shape failure propagated from `tensor`, an update failure propagated from
/// `optim`, a persistence failure propagated from `checkpoint`, a cancelled or
/// exhausted run, an inconsistent local training state, and a loss or gradient
/// input that violates the cross-entropy contract. Underlying tensor, optimizer,
/// and checkpoint errors are carried transparently so no diagnostic context is
/// lost (`ER2`).
#[derive(Debug)]
pub enum trainerror {
    /// Tensor metadata construction or validation failed.
    tensor(tensorerror),

    /// An optimizer update or state operation failed.
    optim(opterror),

    /// A checkpoint load, save, or compatibility check failed.
    checkpoint(checkpoint::checkpointerror),

    /// The run was cancelled at a cancellation checkpoint.
    cancelled,

    /// The gradient-accumulation window size is not at least one.
    invalidaccumulation {
        /// The rejected window size.
        found: usize,
    },

    /// No parameters were supplied to the trainer.
    emptyparameters,

    /// The configured step budget for the run has been exhausted.
    steplimitreached {
        /// The number of completed optimizer steps.
        completed: u64,
        /// The configured maximum number of optimizer steps.
        max: u64,
    },

    /// The loss function returned a different number of gradients than parameters.
    gradientcountmismatch {
        /// The number of parameters the trainer holds.
        expected: usize,
        /// The number of gradients the loss function returned.
        actual: usize,
    },

    /// A returned gradient's shape differs from its parameter's shape.
    gradientshapemismatch {
        /// The zero-based index of the offending gradient.
        index: usize,
        /// The parameter's dimensions.
        param: Vec<usize>,
        /// The gradient's dimensions.
        grad: Vec<usize>,
    },

    /// The loss function returned a value that is not finite.
    nonfiniteloss {
        /// The zero-based index of the step whose loss was rejected.
        step: u64,
        /// The rejected loss value.
        loss: f32,
    },

    /// A gradient was added to an already-full accumulation window.
    accumulatorfull,

    /// The local training state is inconsistent with itself or a loaded artifact.
    invalidstate {
        /// Why the state was rejected.
        detail: String,
    },

    /// Cross-entropy received logits that are not a rank-two matrix.
    lossnotmatrix {
        /// The rejected logits dimensions.
        dims: Vec<usize>,
    },

    /// Cross-entropy received an empty row or class axis.
    lossemptymatrix {
        /// The rejected logits dimensions.
        dims: Vec<usize>,
    },

    /// Cross-entropy received a different number of targets than logit rows.
    targetcountmismatch {
        /// The number of rows in the logits.
        rows: usize,
        /// The number of supplied targets.
        targets: usize,
    },

    /// A cross-entropy target is outside the class axis.
    targetoutofrange {
        /// The zero-based row index of the offending target.
        index: usize,
        /// The rejected target label.
        label: usize,
        /// The number of classes.
        classes: usize,
    },

    /// Cross-entropy produced a non-finite loss or gradient.
    lossnonfinite,

    /// A checkpoint tensor that must hold `f32` data declares another dtype.
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

impl std::fmt::Display for trainerror {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::tensor(f0) => std::fmt::Display::fmt(f0, f),
            Self::optim(f0) => std::fmt::Display::fmt(f0, f),
            Self::checkpoint(f0) => std::fmt::Display::fmt(f0, f),
            Self::cancelled => f.write_str("training was cancelled"),
            Self::invalidaccumulation { found } => write!(
                f,
                "gradient accumulation must cover at least one micro-batch, found {found}"
            ),
            Self::emptyparameters => f.write_str("a trainer requires at least one parameter"),
            Self::steplimitreached { completed, max } => write!(
                f,
                "training step budget exhausted: {completed} of {max} steps completed"
            ),
            Self::gradientcountmismatch { expected, actual } => write!(
                f,
                "loss function returned {actual} gradients for {expected} parameters"
            ),
            Self::gradientshapemismatch { index, param, grad } => write!(
                f,
                "gradient {index} has shape {grad:?} but its parameter has shape {param:?}"
            ),
            Self::nonfiniteloss { step, loss } => {
                write!(f, "loss at training step {step} is not finite: {loss}")
            }
            Self::accumulatorfull => f.write_str("gradient accumulation window is already full"),
            Self::invalidstate { detail } => write!(f, "invalid training state: {detail}"),
            Self::lossnotmatrix { dims } => write!(
                f,
                "cross-entropy logits must be a rank-two matrix, found shape {dims:?}"
            ),
            Self::lossemptymatrix { dims } => write!(
                f,
                "cross-entropy requires at least one row and one class, found shape {dims:?}"
            ),
            Self::targetcountmismatch { rows, targets } => write!(
                f,
                "cross-entropy received {rows} rows but {targets} targets"
            ),
            Self::targetoutofrange {
                index,
                label,
                classes,
            } => write!(
                f,
                "cross-entropy target {index} is {label}, outside the {classes} classes"
            ),
            Self::lossnonfinite => {
                f.write_str("cross-entropy produced a non-finite loss or gradient")
            }
            Self::unsupportedtensordtype { name, dtype } => write!(
                f,
                "checkpoint tensor {name:?} uses dtype {dtype:?} instead of f32"
            ),
        }
    }
}

impl std::error::Error for trainerror {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::tensor(f0) => Some(f0),
            Self::optim(f0) => Some(f0),
            Self::checkpoint(f0) => Some(f0),
            _ => None,
        }
    }
}

impl From<tensorerror> for trainerror {
    fn from(v: tensorerror) -> Self {
        Self::tensor(v)
    }
}

impl From<opterror> for trainerror {
    fn from(v: opterror) -> Self {
        Self::optim(v)
    }
}

impl From<checkpoint::checkpointerror> for trainerror {
    fn from(v: checkpoint::checkpointerror) -> Self {
        Self::checkpoint(v)
    }
}
