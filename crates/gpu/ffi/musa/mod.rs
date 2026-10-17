//! The MUSA platform FFI island, split into four sibling modules.
//!
//! This file holds the declarations ([`cache`], the [`scalar`] / [`buf`]
//! argument descriptors, the ABI handle aliases, and [`dispatchresult`]);
//! the narrow unsafe driver boundary lives in [`drive`], the device-resident
//! [`session`] and the accumulative `*_record` functions in [`record`], and
//! the synchronous operators (matmul, elementwise, attention, scan) in
//! [`ops`].
//!
//! Authority: docs/design.md §21.8. The crate root keeps
//! `#![deny(unsafe_code)]`; the two edition-2024 lower-bound sites (the
//! `unsafe extern "C"` declaration and the single driver-handoff block) both
//! live in [`drive`], each directly preceded by a `// SAFETY:` comment.
mod drive;
mod ops;
mod record;

use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};

/// The MUSA driver result code; `MUSA_SUCCESS` is zero.
pub(crate) type muresult = i32;

/// The success code every `mu*` call must return.
const MUSA_SUCCESS: muresult = 0;

/// An opaque MUSA device handle: the address of the driver's opaque device
/// object, represented as `usize` so the cached state stays `Send + Sync`
/// (the two-site unsafe budget does not allow an `unsafe impl`; the address
/// is a plain integer, so no raw pointer crosses a thread boundary).
pub(crate) type mudevice = usize;

/// An opaque MUSA context handle (see [`mudevice`]).
pub(crate) type mucontext = usize;

/// An opaque MUSA module handle (see [`mudevice`]).
pub(crate) type mumodule = usize;

/// An opaque MUSA kernel handle (see [`mudevice`]).
pub(crate) type mufunction = usize;

/// A device memory pointer: a byte address into the device address space.
pub(crate) type mudevptr = usize;

/// The null stream: launches go to the default stream, which orders
/// synchronously in this backend's run-to-completion contract.
pub(crate) const MUNULLSTREAM: usize = 0;

/// The maximum number of buffers a pool bucket keeps per byte size before an
/// excess buffer is freed back to the driver. It bounds device memory to a
/// small multiple of the largest operand set while still absorbing the
/// per-dispatch allocation churn (docs/musa-backend-plan.md §5.5).
const POOLCAP: usize = 4;

// ---------------------------------------------------------------------------
// cache: the device handle, context, loaded modules, and resolved kernels.
// ---------------------------------------------------------------------------

/// The backend's device state, guarded by a single lock.
///
/// `cache` owns the opened device and context plus every loaded module and
/// resolved kernel function. It holds no operand or result, so one cache can
/// be shared immutably across requests (docs/design.md §21.6). The lock is
/// held across a dispatch: the default MUSA stream serialises launches
/// anyway, and the lazy open/module/function state must stay consistent with
/// the launches that use it.
pub(crate) struct cache {
    state: Mutex<cachestate>,
}

/// The mutable half of [`cache`].
struct cachestate {
    /// Whether [`open`](cachestate::ensureopen) has run.
    opened: bool,
    /// The opened device and context.
    device: Option<mudevice>,
    context: Option<mucontext>,
    /// Loaded modules by source id (`<stem>.cu`).
    modules: HashMap<&'static str, mumodule>,
    /// Resolved kernels by `(source id, entry)`.
    functions: HashMap<(&'static str, &'static str), mufunction>,
    /// Device buffers returned by earlier dispatches, keyed by exact byte
    /// size, reused by later dispatches instead of a fresh `muMemAlloc_v2`.
    /// The lock is held across a whole dispatch and the default stream
    /// serialises launches, so a pooled buffer is never checked out by two
    /// dispatches at once. The pool is bounded in practice by the fixed set
    /// of operand sizes the operators request, so it does not grow without
    /// limit for a fixed workload.
    pool: HashMap<usize, Vec<mudevptr>>,
    /// Device-resident buffers released by [`retiredevice`], waiting for the
    /// next driver handoff block to return them to the pool or free them.
    /// The queue keeps the release inside the single registered handoff, and
    /// its length is bounded by the live device-resident slots of one
    /// multi-stage operator.
    pendingfree: Vec<(mudevptr, usize)>,
}

impl cache {
    /// Creates a cache with nothing opened; the driver is initialised and the
    /// device/context opened lazily by the first dispatch.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(cachestate {
                opened: false,
                device: None,
                context: None,
                modules: HashMap::new(),
                functions: HashMap::new(),
                pool: HashMap::new(),
                pendingfree: Vec::new(),
            }),
        }
    }

    /// Returns how many kernels have been resolved so far.
    ///
    /// Kernels resolve lazily, once per `(source, entry)` pair, so the count
    /// rises to the number of distinct kernels a workload exercises and then
    /// stops. It is the observable that makes "resolved once" checkable.
    #[must_use]
    pub(crate) fn kernelcount(&self) -> usize {
        self.lock().functions.len()
    }

    /// Acquires the lock, recovering the contents if an earlier holder
    /// panicked (the panic would otherwise poison the shared state).
    fn lock(&self) -> std::sync::MutexGuard<'_, cachestate> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl core::fmt::Debug for cache {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Deliberately opaque: the handles are raw pointers with no useful
        // debug rendering, and only the kernel count is observable.
        f.debug_struct("cache")
            .field("kernels", &self.lock().functions.len())
            .finish()
    }
}

/// A scalar kernel argument. Every in-repo kernel takes its buffer operands
/// first and its scalar parameters last, so the launch argument list is
/// exactly the bound device pointers followed by these scalars.
#[derive(Clone, Copy, Debug)]
pub(crate) enum scalar {
    /// An `unsigned` kernel parameter.
    u32(u32),
    /// An `unsigned long long` kernel parameter.
    u64(u64),
    /// A `float` kernel parameter.
    f32(f32),
}

/// A bound operand of a dispatch.
///
/// Every `dispatch` is run-to-completion: host operands are uploaded, the
/// output is downloaded, and this dispatch's temporaries are returned to the
/// pool inside the single handoff block. The device-resident variants serve
/// multi-stage kernels (`scan_parallel` chains its passes): the producer hands
/// a buffer back with [`buf::outdevice`] or [`buf::hostdevice`], later passes
/// bind it with [`buf::device`], and the owner releases it through
/// [`retiredevice`], which queues the return into the next handoff block — no
/// second driver-handoff site exists for freeing.
#[derive(Clone, Copy, Debug)]
pub(crate) enum buf<'a> {
    /// Upload a host `f32` slice into a fresh device allocation.
    host(&'a [f32]),
    /// Upload a host `u64` slice into a fresh device allocation.
    hostu64(&'a [u64]),
    /// Upload a host `u64` slice and hand the allocation back as a device
    /// pointer instead of returning it to the pool, so a recorded step keeps
    /// its index buffer alive across the deferred synchronisation.
    hostu64device(&'a [u64]),
    /// Allocate the output buffer and download it after the kernel completes.
    out(usize),
    /// Bind an existing device allocation without uploading or returning it
    /// to the pool. The caller owns its lifetime and releases it through
    /// [`retiredevice`].
    device(mudevptr),
    /// Allocate the output buffer and hand it back as a device pointer
    /// instead of downloading it. The caller owns its lifetime and releases
    /// it through [`retiredevice`].
    outdevice(usize),
    /// Upload a host `f32` slice and hand the allocation back as a device
    /// pointer instead of returning it to the pool, so later passes reuse the
    /// upload. The caller owns its lifetime and releases it through
    /// [`retiredevice`].
    hostdevice(&'a [f32]),
    /// Allocate a fresh device buffer and copy `count` bytes into it from
    /// `src`, handing the allocation back to the caller. Used when an operand
    /// must be preserved across a kernel that updates it in place.
    copy(mudevptr, usize),
    /// Copy `count` bytes from `src` into an existing device buffer `dst`
    /// without allocating. Used to append a key/value row to a persistent
    /// cache buffer.
    copyinto(mudevptr, mudevptr, usize),
    /// Download an existing device allocation into a host vector without
    /// allocating or launching anything. Used by [`flush`], which runs the
    /// deferred synchronisation and the downloads inside the one registered
    /// handoff block.
    download(mudevptr, usize),
}

/// The two halves of one driver handoff: downloaded host outputs in buffer
/// order, and handed-back device buffers as `(pointer, byte size)` pairs for
/// every `device`-resident slot in buffer order.
pub(crate) struct dispatchresult {
    pub(crate) host: Vec<Vec<f32>>,
    pub(crate) device: Vec<(mudevptr, usize)>,
}

pub(crate) use drive::*;
pub(crate) use ops::*;
pub(crate) use record::*;
