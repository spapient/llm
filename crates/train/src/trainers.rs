use ::tensor::tensor;
use context::cancellation;
use optim::optimizer;
use tracing::debug;

use crate::error::trainerror;
use crate::gradientaccumulation::gradientaccumulator;
use crate::step::{applyschedule, stepresult};
use crate::trainconfigs::trainconfig;

/// Orchestrates a training loop over explicit parameters and gradients.
///
/// A trainer binds an owned parameter set, an [`optimizer`], and a
/// [`trainconfig`] together with the local progress counters
/// (`step`, `epoch`) and a [`gradientaccumulator`]. It does not own a model, a
/// graph, or a forward pass: the caller supplies a loss function per step and
/// receives the gradients to apply, which keeps the trainer independent of any
/// particular architecture and lets `autograd`, `nn`, or a hand-written
/// gradient all drive the same loop.
///
/// # Failure and cancellation
///
/// State advances only on a completed optimizer update. If the loss function
/// fails, returns a non-finite value, violates the gradient contract, or the run
/// is cancelled, the buffered window is discarded, no optimizer update is
/// applied, and neither `step` nor the optimizer's own counter moves
/// (`OW3`). The optimizer itself validates before mutating, so a rejected update
/// leaves both the parameters and the optimizer state unchanged.
pub struct trainer<O: optimizer> {
    params: Vec<tensor<f32>>,
    optimizer: O,
    config: trainconfig,
    step: u64,
    epoch: u64,
    accumulator: gradientaccumulator,
    windowsum: f32,
    cancellation: Option<cancellation>,
}

impl<O: optimizer> trainer<O> {
    /// Binds parameters, an optimizer, and a configuration into a trainer.
    ///
    /// # Errors
    ///
    /// Returns [`trainerror::emptyparameters`] when `params` is empty and any
    /// error [`gradientaccumulator::new`] raises for the configuration's window.
    pub fn new(
        params: Vec<tensor<f32>>,
        optimizer: O,
        config: trainconfig,
    ) -> Result<Self, trainerror> {
        let accumulator = gradientaccumulator::new(&params, config.getaccumulationsteps())?;
        Ok(Self {
            params,
            optimizer,
            config,
            step: 0,
            epoch: 0,
            accumulator,
            windowsum: 0.0,
            cancellation: None,
        })
    }

    /// Attaches a cancellation token consulted before every micro-batch.
    #[must_use]
    pub fn withcancellation(mut self, cancellation: cancellation) -> Self {
        self.cancellation = Some(cancellation);
        self
    }

    /// Returns the parameter tensors in the optimizer's key order.
    #[must_use]
    pub fn getparams(&self) -> &[tensor<f32>] {
        &self.params
    }

    /// Returns the optimizer.
    #[must_use]
    pub const fn getoptimizer(&self) -> &O {
        &self.optimizer
    }

    /// Returns the run configuration.
    #[must_use]
    pub const fn getconfig(&self) -> &trainconfig {
        &self.config
    }

    /// Returns the number of completed optimizer steps.
    #[must_use]
    pub const fn getstep(&self) -> u64 {
        self.step
    }

    /// Returns the number of completed epochs recorded in the trainer.
    #[must_use]
    pub const fn getepoch(&self) -> u64 {
        self.epoch
    }

    /// Returns the gradient accumulator.
    #[must_use]
    pub const fn getaccumulator(&self) -> &gradientaccumulator {
        &self.accumulator
    }

    /// Returns `true` when the step budget still allows another optimizer step.
    ///
    /// A configured maximum of zero means the budget is unbounded.
    #[must_use]
    pub const fn canstep(&self) -> bool {
        self.config.getmaxsteps() == 0 || self.step < self.config.getmaxsteps()
    }

    /// Sets the epoch counter, for example after a resume.
    pub fn setepoch(&mut self, epoch: u64) {
        self.epoch = epoch;
    }

    /// Replaces the parameter tensors and rebuilds the accumulator to match.
    ///
    /// # Errors
    ///
    /// Returns [`trainerror::invalidstate`] when a gradient window is buffered,
    /// since discarding it silently would lose work, and otherwise the same
    /// errors as [`gradientaccumulator::new`].
    pub(crate) fn replaceparams(&mut self, params: Vec<tensor<f32>>) -> Result<(), trainerror> {
        if !self.accumulator.isempty() {
            return Err(trainerror::state(
                "cannot replace parameters while a gradient window is buffered",
            ));
        }
        let accumulator = gradientaccumulator::new(&params, self.config.getaccumulationsteps())?;
        self.params = params;
        self.accumulator = accumulator;
        self.windowsum = 0.0;
        Ok(())
    }

    /// Sets the completed-step counter, for example after a resume.
    pub(crate) fn setstep(&mut self, step: u64) {
        self.step = step;
    }

    /// Returns the optimizer for state restoration.
    pub(crate) fn getoptimizermut(&mut self) -> &mut O {
        &mut self.optimizer
    }

    /// Runs `accumulationsteps` micro-batches and applies one optimizer update.
    ///
    /// The closure receives the current parameters and returns the scalar loss
    /// of one micro-batch together with one gradient per parameter, in the same
    /// order. The window must be empty on entry; use [`Self::accumulatemicrobatch`]
    /// to drive it by hand.
    ///
    /// # Errors
    ///
    /// Returns [`trainerror::invalidstate`] when the window is not empty,
    /// [`trainerror::steplimitreached`] when the budget is exhausted, and any
    /// error [`Self::accumulatemicrobatch`] raises.
    pub fn step<F>(&mut self, mut loss: F) -> Result<stepresult, trainerror>
    where
        F: FnMut(&[tensor<f32>]) -> Result<(f32, Vec<tensor<f32>>), trainerror>,
    {
        if !self.accumulator.isempty() {
            return Err(trainerror::state(
                "the gradient window is not empty; complete or reset it before a full step",
            ));
        }
        if !self.canstep() {
            return Err(trainerror::steplimitreached {
                completed: self.step,
                max: self.config.getmaxsteps(),
            });
        }
        for _ in 0..self.config.getaccumulationsteps() {
            if let Some(result) = self.accumulatemicrobatch(&mut loss)? {
                return Ok(result);
            }
        }
        Err(trainerror::state(
            "the gradient window did not close within its capacity",
        ))
    }

    /// Runs one micro-batch, applying an update only when the window closes.
    ///
    /// Returns `None` while the window is filling and `Some` with the step
    /// result on the micro-batch that closes it. On the closing micro-batch the
    /// buffered gradients are averaged, the learning rate for the current step
    /// index is applied, and the optimizer performs exactly one update.
    ///
    /// # Errors
    ///
    /// Returns [`trainerror::steplimitreached`] when the budget is exhausted
    /// before the micro-batch, [`trainerror::cancelled`] when the token is set,
    /// [`trainerror::nonfiniteloss`] when the loss is not finite, and any
    /// gradient-, optimizer-, or schedule-level error. Every error discards the
    /// buffered window without advancing the step counter.
    pub fn accumulatemicrobatch<F>(&mut self, mut loss: F) -> Result<Option<stepresult>, trainerror>
    where
        F: FnMut(&[tensor<f32>]) -> Result<(f32, Vec<tensor<f32>>), trainerror>,
    {
        if !self.canstep() {
            self.resetwindow();
            return Err(trainerror::steplimitreached {
                completed: self.step,
                max: self.config.getmaxsteps(),
            });
        }
        if self.cancelled() {
            self.resetwindow();
            return Err(trainerror::cancelled);
        }
        let (value, grads) = match loss(&self.params) {
            Ok(pair) => pair,
            Err(error) => {
                self.resetwindow();
                return Err(error);
            }
        };
        if !value.is_finite() {
            self.resetwindow();
            return Err(trainerror::nonfiniteloss {
                step: self.step,
                loss: value,
            });
        }
        if let Err(error) = self.accumulator.accumulate(&grads) {
            self.resetwindow();
            return Err(error);
        }
        self.windowsum += value;
        if !self.accumulator.isfull() {
            return Ok(None);
        }
        if self.cancelled() {
            self.resetwindow();
            return Err(trainerror::cancelled);
        }
        let mut grads = match self.accumulator.take() {
            Ok(grads) => grads,
            Err(error) => {
                self.resetwindow();
                return Err(error);
            }
        };
        let microbatches = self.accumulator.getcapacity();
        let lr = match applyschedule(&mut self.optimizer, self.config.getschedule(), self.step) {
            Ok(lr) => lr,
            Err(error) => {
                self.windowsum = 0.0;
                return Err(error);
            }
        };
        if let Err(error) = self.optimizer.update(&mut self.params, &mut grads) {
            self.windowsum = 0.0;
            return Err(trainerror::optim(error));
        }
        let loss = self.windowsum / microbatches as f32;
        self.windowsum = 0.0;
        self.step += 1;
        debug!(
            step = self.step,
            loss, lr, microbatches, "optimizer update applied"
        );
        Ok(Some(stepresult::new(loss, self.step, lr, microbatches)))
    }

    fn cancelled(&self) -> bool {
        self.cancellation
            .as_ref()
            .is_some_and(cancellation::iscancelled)
    }

    fn resetwindow(&mut self) {
        self.accumulator.reset();
        self.windowsum = 0.0;
    }
}
