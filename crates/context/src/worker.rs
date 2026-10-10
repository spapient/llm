use core::fmt;
use core::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

use crate::error::contexterror;
use crate::execution::executioncontext;

/// A failure produced while coordinating workers.
///
/// The variant distinguishes a control failure (cancellation or step budget,
/// carried as [`contexterror`]), a job failure (the job's own error `E`), and a
/// worker panic. It is generic over the job error so that the coordination
/// primitive does not force callers to translate their own error types.
#[derive(Debug, PartialEq, Eq)]
pub enum workererror<E> {
    /// A checkpoint failed before a job could start.
    context {
        /// The control failure reported by the execution context.
        source: contexterror,
    },
    /// A job returned an error.
    job {
        /// The index of the failing job.
        index: usize,
        /// The job's own error.
        source: E,
    },
    /// A worker thread panicked.
    workerpanicked,
}

impl<E: fmt::Display> fmt::Display for workererror<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::context { source } => write!(formatter, "worker control error: {source}"),
            Self::job { index, source } => write!(formatter, "worker job {index} failed: {source}"),
            Self::workerpanicked => formatter.write_str("worker panicked"),
        }
    }
}

impl<E: fmt::Debug + fmt::Display> std::error::Error for workererror<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::context { source } => Some(source),
            Self::job { .. } | Self::workerpanicked => None,
        }
    }
}

/// A bounded worker-coordination primitive.
///
/// A `workergroup` runs a fixed number of indexed jobs across at most
/// `maxworkers` scoped threads. Before each job a worker takes an execution
/// checkpoint, so cancellation and the step budget are enforced at job
/// boundaries. Results are returned in job-index order; on the first observed
/// failure or cancellation, remaining workers stop taking new jobs and no
/// partial results are returned.
///
/// Which failure is reported is deterministic when the failure occurs before
/// concurrency can interleave (for example with a single worker); with multiple
/// workers the first failure observed may vary, so callers must not depend on a
/// specific job failing first.
pub struct workergroup {
    maxworkers: usize,
}

impl workergroup {
    /// Creates a group that runs at most `maxworkers` jobs concurrently.
    ///
    /// # Errors
    ///
    /// Returns [`contexterror::invalidconfig`] when `maxworkers` is zero.
    pub fn new(maxworkers: usize) -> Result<Self, contexterror> {
        if maxworkers == 0 {
            return Err(contexterror::invalidconfig {
                detail: "worker count must be at least 1",
            });
        }
        Ok(Self { maxworkers })
    }

    /// Returns the configured maximum number of concurrent workers.
    #[must_use]
    pub fn getmaxworkers(&self) -> usize {
        self.maxworkers
    }

    /// Runs `count` jobs, checking `exec` before each job.
    ///
    /// Returns the jobs' outputs ordered by job index.
    ///
    /// # Errors
    ///
    /// Returns [`workererror::context`] when an execution checkpoint fails,
    /// [`workererror::job`] when a job returns an error, and
    /// [`workererror::workerpanicked`] when a worker thread panics.
    pub fn run<T, E, F>(
        &self,
        count: usize,
        exec: &executioncontext,
        work: F,
    ) -> Result<Vec<T>, workererror<E>>
    where
        T: Send,
        E: Send,
        F: Fn(usize) -> Result<T, E> + Sync,
    {
        if count == 0 {
            return Ok(Vec::new());
        }
        let next = AtomicUsize::new(0);
        let stop = AtomicBool::new(false);
        let failure: Mutex<Option<workererror<E>>> = Mutex::new(None);
        let workers = self.maxworkers.min(count);
        let mut results: Vec<(usize, T)> = Vec::with_capacity(count);

        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(workers);
            for _ in 0..workers {
                let next = &next;
                let stop = &stop;
                let failure = &failure;
                let work = &work;
                handles.push(scope.spawn(move || {
                    let mut local: Vec<(usize, T)> = Vec::new();
                    loop {
                        if stop.load(Ordering::Acquire) {
                            break;
                        }
                        let index = next.fetch_add(1, Ordering::AcqRel);
                        if index >= count {
                            break;
                        }
                        if let Err(source) = exec.checkpoint() {
                            storefailure(failure, workererror::context { source });
                            stop.store(true, Ordering::Release);
                            break;
                        }
                        match work(index) {
                            Ok(value) => local.push((index, value)),
                            Err(source) => {
                                storefailure(failure, workererror::job { index, source });
                                stop.store(true, Ordering::Release);
                                break;
                            }
                        }
                    }
                    local
                }));
            }
            for handle in handles {
                match handle.join() {
                    Ok(mut local) => results.append(&mut local),
                    Err(_) => storefailure(&failure, workererror::workerpanicked),
                }
            }
        });

        if let Some(error) = failure.into_inner().unwrap_or_else(PoisonError::into_inner) {
            return Err(error);
        }
        results.sort_by_key(|(index, _)| *index);
        Ok(results.into_iter().map(|(_, value)| value).collect())
    }
}

/// Records `error` if no failure has been recorded yet.
fn storefailure<E>(slot: &Mutex<Option<workererror<E>>>, error: workererror<E>) {
    let mut guard = slot.lock().unwrap_or_else(PoisonError::into_inner);
    if guard.is_none() {
        *guard = Some(error);
    }
}
