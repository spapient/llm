use core::sync::atomic::{AtomicUsize, Ordering};

use crate::error::contexterror;

/// An explicit, shared memory budget with byte-granular accounting.
///
/// The budget owns a capacity and an atomic count of currently reserved bytes.
/// Reservations are granted only when the requested byte count fits within the
/// remaining capacity, using checked arithmetic so that a request that would
/// overflow the counter is rejected rather than wrapping. The owner of the
/// budget decides its lifetime; clone-free sharing is achieved by holding
/// `&memorybudget` (which is `Sync`) or by wrapping it in an explicit shared
/// owner at the call site. No global or static budget exists.
///
/// A granted reservation is an RAII guard: dropping it releases exactly the
/// bytes it reserved, in every termination path (success, error, early return),
/// because release is implemented in `Drop`.
#[derive(Debug)]
pub struct memorybudget {
    capacity: usize,
    reserved: AtomicUsize,
}

impl memorybudget {
    /// Creates a budget with `capacity` bytes available.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            reserved: AtomicUsize::new(0),
        }
    }

    /// Returns the configured capacity in bytes.
    #[must_use]
    pub fn getcapacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of bytes currently reserved.
    #[must_use]
    pub fn getreserved(&self) -> usize {
        self.reserved.load(Ordering::Acquire)
    }

    /// Returns the number of bytes not currently reserved.
    #[must_use]
    pub fn getavailable(&self) -> usize {
        self.capacity.saturating_sub(self.getreserved())
    }

    /// Attempts to reserve `bytes` and returns a guard that releases them on drop.
    ///
    /// # Errors
    ///
    /// Returns [`contexterror::budgetexceeded`] when the reservation would
    /// exceed the capacity, and [`contexterror::budgetoverflow`] when the byte
    /// counter would overflow. A rejected request reserves nothing.
    pub fn tryreserve(&self, bytes: usize) -> Result<memoryreservation<'_>, contexterror> {
        self.reserved
            .try_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                let next = current.checked_add(bytes)?;
                if next > self.capacity {
                    None
                } else {
                    Some(next)
                }
            })
            .map_err(|current| {
                if current.checked_add(bytes).is_none() {
                    contexterror::budgetoverflow
                } else {
                    contexterror::budgetexceeded {
                        requested: bytes,
                        reserved: current,
                        capacity: self.capacity,
                    }
                }
            })?;
        Ok(memoryreservation {
            budget: self,
            bytes,
        })
    }
}

/// An RAII guard for bytes reserved from a [`memorybudget`].
///
/// Dropping the guard returns its bytes to the budget. The guard borrows the
/// budget, so it can never outlive the owner that provided the capacity.
#[derive(Debug)]
pub struct memoryreservation<'a> {
    budget: &'a memorybudget,
    bytes: usize,
}

impl memoryreservation<'_> {
    /// Returns the number of bytes held by this reservation.
    #[must_use]
    pub fn getbytes(&self) -> usize {
        self.bytes
    }
}

impl Drop for memoryreservation<'_> {
    fn drop(&mut self) {
        self.budget.reserved.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}
