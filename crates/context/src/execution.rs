use core::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use lib::requestid;

use crate::budget::memorybudget;
use crate::cancel::cancellation;
use crate::error::contexterror;
use crate::metric::metrics;

/// Immutable execution configuration.
///
/// The configuration carries only validated bounds. It deliberately holds no
/// request data and no mutable state, so it can be constructed once and shared
/// by reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct executionconfig {
    memorybytes: usize,
    maxworkers: usize,
    maxsteps: usize,
}

impl executionconfig {
    /// Creates a configuration from a memory capacity, a worker count, and a
    /// step (checkpoint) budget.
    ///
    /// # Errors
    ///
    /// Returns [`contexterror::invalidconfig`] when `maxworkers` is zero, since
    /// an execution that can run no work is a wiring error rather than a valid
    /// resource limit.
    pub fn new(
        memorybytes: usize,
        maxworkers: usize,
        maxsteps: usize,
    ) -> Result<Self, contexterror> {
        if maxworkers == 0 {
            return Err(contexterror::invalidconfig {
                detail: "maxworkers must be at least 1",
            });
        }
        Ok(Self {
            memorybytes,
            maxworkers,
            maxsteps,
        })
    }

    /// Returns the configured memory capacity in bytes.
    #[must_use]
    pub fn getmemorybytes(&self) -> usize {
        self.memorybytes
    }

    /// Returns the configured maximum number of concurrent workers.
    #[must_use]
    pub fn getmaxworkers(&self) -> usize {
        self.maxworkers
    }

    /// Returns the configured maximum number of checkpoints.
    #[must_use]
    pub fn getmaxsteps(&self) -> usize {
        self.maxsteps
    }
}

/// The per-execution control context.
///
/// An `executioncontext` owns the request-scoped controls an execution needs:
/// its identity, its cancellation handle, its memory budget, its step counter,
/// and an optional metrics sink. Each execution creates its own context; the
/// context is not global, is not stored in shared weights, and its mutable
/// fields are atomics with documented ordering.
pub struct executioncontext {
    id: requestid,
    config: executionconfig,
    cancel: cancellation,
    budget: memorybudget,
    steps: AtomicUsize,
    sink: Option<Arc<dyn metrics>>,
}

impl executioncontext {
    /// Creates a context without a metrics sink.
    #[must_use]
    pub fn new(id: requestid, config: executionconfig) -> Self {
        Self::newwithsink(id, config, None)
    }

    /// Creates a context that reports to `sink`.
    #[must_use]
    pub fn withmetrics(id: requestid, config: executionconfig, sink: Arc<dyn metrics>) -> Self {
        Self::newwithsink(id, config, Some(sink))
    }

    fn newwithsink(id: requestid, config: executionconfig, sink: Option<Arc<dyn metrics>>) -> Self {
        Self {
            id,
            config,
            cancel: cancellation::new(),
            budget: memorybudget::new(config.getmemorybytes()),
            steps: AtomicUsize::new(0),
            sink,
        }
    }

    /// Returns the execution's request identity.
    #[must_use]
    pub fn getid(&self) -> requestid {
        self.id
    }

    /// Returns the execution configuration.
    #[must_use]
    pub fn getconfig(&self) -> &executionconfig {
        &self.config
    }

    /// Returns the cancellation handle.
    #[must_use]
    pub fn getcancellation(&self) -> &cancellation {
        &self.cancel
    }

    /// Returns the memory budget.
    #[must_use]
    pub fn getbudget(&self) -> &memorybudget {
        &self.budget
    }

    /// Returns the number of checkpoints consumed so far.
    #[must_use]
    pub fn getsteps(&self) -> usize {
        self.steps.load(Ordering::Acquire)
    }

    /// Advances one step and checks cancellation.
    ///
    /// A checkpoint is the single explicit progress guard: it fails when the
    /// execution has been cancelled or when the step budget is exhausted, and
    /// it consumes one step on success. Long-running loops must call it at safe
    /// points so that cancellation propagates and the step budget is enforced.
    ///
    /// # Errors
    ///
    /// Returns [`contexterror::cancelled`] when cancellation has been requested
    /// and [`contexterror::stepbudgetexceeded`] when the configured number of
    /// checkpoints has already been consumed.
    pub fn checkpoint(&self) -> Result<(), contexterror> {
        if self.cancel.iscancelled() {
            return Err(contexterror::cancelled);
        }
        let max = self.config.getmaxsteps();
        self.steps
            .try_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                if current >= max {
                    None
                } else {
                    Some(current + 1)
                }
            })
            .map_err(|current| contexterror::stepbudgetexceeded { used: current, max })?;
        Ok(())
    }

    /// Records a monotonically increasing counter in the metrics sink, if any.
    pub fn recordcounter(&self, name: &'static str, value: u64) {
        if let Some(sink) = &self.sink {
            sink.recordcounter(name, value);
        }
    }

    /// Records a gauge value in the metrics sink, if any.
    pub fn recordgauge(&self, name: &'static str, value: i64) {
        if let Some(sink) = &self.sink {
            sink.recordgauge(name, value);
        }
    }
}
