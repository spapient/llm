/// Failures produced while enforcing execution controls.
///
/// Each variant distinguishes a cause a caller must act on differently:
/// cancellation is not a resource failure, a step budget is not a memory
/// budget, and an invalid configuration is a programming or wiring error rather
/// than a resource condition.
#[derive(Debug, PartialEq, Eq)]
pub enum contexterror {
    /// The execution was cancelled at a checkpoint.
    cancelled,

    /// The configured number of checkpoints was exhausted.
    stepbudgetexceeded {
        /// The number of checkpoints already consumed.
        used: usize,
        /// The configured maximum number of checkpoints.
        max: usize,
    },

    /// A memory reservation would exceed the configured capacity.
    budgetexceeded {
        /// The requested byte count.
        requested: usize,
        /// The byte count already reserved.
        reserved: usize,
        /// The configured capacity in bytes.
        capacity: usize,
    },

    /// A memory reservation would overflow the accounting counter.
    budgetoverflow,

    /// A configuration value is invalid.
    invalidconfig {
        /// A static explanation of the rejected configuration.
        detail: &'static str,
    },
}

impl std::fmt::Display for contexterror {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::cancelled => f.write_str("execution was cancelled"),
            Self::stepbudgetexceeded { used, max } => {
                write!(f, "step budget exceeded: used {used}, maximum {max}")
            }
            Self::budgetexceeded {
                requested,
                reserved,
                capacity,
            } => write!(
                f,
                "memory budget exceeded: requested {requested}, already reserved {reserved}, capacity {capacity}"
            ),
            Self::budgetoverflow => f.write_str("memory budget accounting overflow"),
            Self::invalidconfig { detail } => {
                write!(f, "invalid execution configuration: {detail}")
            }
        }
    }
}

impl std::error::Error for contexterror {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
