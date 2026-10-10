use core::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::error::contexterror;

/// A cheaply cloneable cancellation handle.
///
/// The handle owns a reference-counted atomic flag. All clones observe the same
/// flag, so cancelling through any clone makes every clone report cancelled.
/// The first-created handle owns the flag's lifetime; clones borrow it. This is
/// an explicit synchronization primitive, not global state: a cancellation
/// handle must be created per request and must not be stored in shared model
/// weights or process-wide statics.
///
/// Visibility is established by the atomic itself: [`Self::cancel`] stores with
/// `Release` ordering and [`Self::iscancelled`] loads with `Acquire` ordering,
/// so a thread that observes cancellation also observes all writes that
/// preceded it.
#[derive(Clone, Debug)]
pub struct cancellation {
    flag: Arc<AtomicBool>,
}

impl cancellation {
    /// Creates a new, not-yet-cancelled handle.
    #[must_use]
    pub fn new() -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Marks the execution as cancelled.
    ///
    /// Idempotent: cancelling an already-cancelled execution is a no-op.
    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
    }

    /// Reports whether the execution has been cancelled.
    #[must_use]
    pub fn iscancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    /// Returns the cancellation checkpoint result.
    ///
    /// # Errors
    ///
    /// Returns [`contexterror::cancelled`] when the execution has been
    /// cancelled, and `Ok(())` otherwise.
    pub fn checkpoint(&self) -> Result<(), contexterror> {
        if self.iscancelled() {
            return Err(contexterror::cancelled);
        }
        Ok(())
    }
}

impl Default for cancellation {
    fn default() -> Self {
        Self::new()
    }
}
