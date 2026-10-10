use optim::{lrschedule, optimizer};

use crate::error::trainerror;

/// The outcome of one completed optimizer step.
///
/// A step covers one full gradient-accumulation window. The reported `loss` is
/// the mean of the micro-batch losses that contributed to the update, the
/// `step` is the number of completed optimizer steps after this update, and the
/// `learningrate` is the rate that was actually applied.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct stepresult {
    loss: f32,
    step: u64,
    learningrate: f32,
    microbatches: usize,
}

impl stepresult {
    /// Builds a step result.
    #[must_use]
    pub(crate) const fn new(loss: f32, step: u64, learningrate: f32, microbatches: usize) -> Self {
        Self {
            loss,
            step,
            learningrate,
            microbatches,
        }
    }

    /// Returns the mean loss over the step's micro-batches.
    #[must_use]
    pub const fn getloss(&self) -> f32 {
        self.loss
    }

    /// Returns the number of completed optimizer steps after this update.
    #[must_use]
    pub const fn getstep(&self) -> u64 {
        self.step
    }

    /// Returns the learning rate applied during this update.
    #[must_use]
    pub const fn getlearningrate(&self) -> f32 {
        self.learningrate
    }

    /// Returns how many micro-batches this step averaged.
    #[must_use]
    pub const fn getmicrobatches(&self) -> usize {
        self.microbatches
    }
}

/// Resolves and applies the learning rate for a step.
///
/// With a schedule, the rate is `schedule.getlr(stepindex)` and is pushed into
/// the optimizer through [`optimizer::setlr`]; without one, the optimizer's
/// current rate is kept. The rate actually in effect is returned so the caller
/// can report it in a [`stepresult`].
pub(crate) fn applyschedule<O: optimizer>(
    optimizer: &mut O,
    schedule: Option<lrschedule>,
    stepindex: u64,
) -> Result<f32, trainerror> {
    let lr = match schedule {
        Some(schedule) => schedule.getlr(stepindex),
        None => optimizer.getlr(),
    };
    optimizer.setlr(lr)?;
    Ok(lr)
}
