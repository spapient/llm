use thiserror::Error;

/// Failures produced while enforcing execution controls.
///
/// Each variant distinguishes a cause a caller must act on differently:
/// cancellation is not a resource failure, a step budget is not a memory
/// budget, and an invalid configuration is a programming or wiring error rather
/// than a resource condition.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum contexterror {
    /// The execution was cancelled at a checkpoint.
    #[error("execution was cancelled")]
    cancelled,

    /// The configured number of checkpoints was exhausted.
    #[error("step budget exceeded: used {used}, maximum {max}")]
    stepbudgetexceeded {
        /// The number of checkpoints already consumed.
        used: usize,
        /// The configured maximum number of checkpoints.
        max: usize,
    },

    /// A memory reservation would exceed the configured capacity.
    #[error(
        "memory budget exceeded: requested {requested}, already reserved {reserved}, capacity {capacity}"
    )]
    budgetexceeded {
        /// The requested byte count.
        requested: usize,
        /// The byte count already reserved.
        reserved: usize,
        /// The configured capacity in bytes.
        capacity: usize,
    },

    /// A memory reservation would overflow the accounting counter.
    #[error("memory budget accounting overflow")]
    budgetoverflow,

    /// A configuration value is invalid.
    #[error("invalid execution configuration: {detail}")]
    invalidconfig {
        /// A static explanation of the rejected configuration.
        detail: &'static str,
    },
}
