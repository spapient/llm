use serde::{Deserialize, Serialize};

/// A snapshot of training progress stored in a checkpoint.
///
/// The state is deliberately minimal and owned by this crate: it records the
/// completed optimizer step and epoch so a training run can resume. Optimizer
/// parameter state is not modeled here; that belongs to the optimizer crate and
/// is persisted as ordinary tensors through the same checkpoint container.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct trainingstate {
    step: u64,
    epoch: u64,
}

impl trainingstate {
    /// Creates a training-state snapshot.
    #[must_use]
    pub const fn new(step: u64, epoch: u64) -> Self {
        Self { step, epoch }
    }

    /// Returns the number of completed optimizer steps.
    #[must_use]
    pub const fn getstep(self) -> u64 {
        self.step
    }

    /// Returns the number of completed epochs.
    #[must_use]
    pub const fn getepoch(self) -> u64 {
        self.epoch
    }
}
