use core::sync::atomic::{AtomicUsize, Ordering};

use crate::error::serveerror;

/// A bounded concurrency admission controller.
///
/// `admissions` owns a fixed ceiling and an atomic count of in-flight
/// admissions. It is the single synchronization primitive for the service's
/// concurrency limit: [`Self::tryadmit`] grants an [`admission`] only when the
/// count is below the ceiling, using compare-and-update semantics so concurrent
/// admissions can never exceed the ceiling. No global state is used; the
/// controller lives inside the shared [`serviceconfig`](crate::serviceconfig).
#[derive(Debug)]
pub struct admissions {
    limit: usize,
    inflight: AtomicUsize,
}

impl admissions {
    /// Creates a controller that admits at most `limit` concurrent requests.
    ///
    /// # Errors
    ///
    /// Returns [`serveerror::invalidconfig`] when `limit` is zero, since a
    /// service that can admit no request is a wiring error rather than a valid
    /// limit.
    pub fn new(limit: usize) -> Result<Self, serveerror> {
        if limit == 0 {
            return Err(serveerror::invalidconfig {
                detail: "maxconcurrent must be at least 1",
            });
        }
        Ok(Self {
            limit,
            inflight: AtomicUsize::new(0),
        })
    }

    /// Returns the configured concurrency ceiling.
    #[must_use]
    pub fn getlimit(&self) -> usize {
        self.limit
    }

    /// Returns the number of admissions currently held.
    #[must_use]
    pub fn getinflight(&self) -> usize {
        self.inflight.load(Ordering::Acquire)
    }

    /// Attempts to admit one request.
    ///
    /// # Errors
    ///
    /// Returns [`serveerror::overloaded`] when the ceiling is already reached.
    /// A rejected attempt reserves nothing.
    pub fn tryadmit(&self) -> Result<admission<'_>, serveerror> {
        self.inflight
            .try_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                if current >= self.limit {
                    None
                } else {
                    Some(current + 1)
                }
            })
            .map_err(|_| serveerror::overloaded { limit: self.limit })?;
        Ok(admission { control: self })
    }
}

/// An RAII guard for one granted admission.
///
/// Dropping the guard releases the admission in every termination path
/// (success, error, early return), so the ceiling is restored without a manual
/// release call (OW1). The guard borrows the controller, so it can never
/// outlive the owner that provided the ceiling.
#[derive(Debug)]
pub struct admission<'a> {
    control: &'a admissions,
}

impl Drop for admission<'_> {
    fn drop(&mut self) {
        self.control.inflight.fetch_sub(1, Ordering::AcqRel);
    }
}
