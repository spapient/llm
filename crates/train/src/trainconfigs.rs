use lib::seed;
use optim::lrschedule;

use crate::error::trainerror;

/// Immutable configuration for a training run.
///
/// A configuration is plain validated data with no mutable state. It fixes how
/// many micro-batches are averaged into one optimizer step
/// (`accumulationsteps`), the maximum number of optimizer steps this run may
/// perform (`maxsteps`), an optional learning-rate schedule applied before each
/// update, and a reproducible seed.
///
/// The seed is carried for downstream initialization and data shuffling so a
/// whole run can be reproduced from one value; the core loss/step loop is
/// deterministic and does not consume randomness itself.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct trainconfig {
    accumulationsteps: usize,
    maxsteps: u64,
    schedule: Option<lrschedule>,
    seed: seed,
}

impl trainconfig {
    /// Creates a training configuration.
    ///
    /// # Errors
    ///
    /// Returns [`trainerror::invalidaccumulation`] when `accumulationsteps` is
    /// zero, since a step that averages no micro-batches cannot produce a
    /// gradient.
    pub fn new(
        accumulationsteps: usize,
        maxsteps: u64,
        schedule: Option<lrschedule>,
        seed: seed,
    ) -> Result<Self, trainerror> {
        if accumulationsteps == 0 {
            return Err(trainerror::invalidaccumulation { found: 0 });
        }
        Ok(Self {
            accumulationsteps,
            maxsteps,
            schedule,
            seed,
        })
    }

    /// Creates a single-micro-batch configuration with a constant learning rate
    /// and no step budget.
    ///
    /// This is a convenience for tests and simple runs; it performs no gradient
    /// accumulation and applies no schedule.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::new`].
    pub fn single(maxsteps: u64, seed: seed) -> Result<Self, trainerror> {
        Self::new(1, maxsteps, None, seed)
    }

    /// Returns the number of micro-batches averaged into one optimizer step.
    #[must_use]
    pub const fn getaccumulationsteps(&self) -> usize {
        self.accumulationsteps
    }

    /// Returns the maximum number of optimizer steps this run may perform.
    #[must_use]
    pub const fn getmaxsteps(&self) -> u64 {
        self.maxsteps
    }

    /// Returns the optional learning-rate schedule.
    #[must_use]
    pub const fn getschedule(&self) -> Option<lrschedule> {
        self.schedule
    }

    /// Returns the reproducible run seed.
    #[must_use]
    pub const fn getseed(&self) -> seed {
        self.seed
    }
}
