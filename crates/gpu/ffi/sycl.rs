#![allow(unsafe_code)]
//! The Intel oneAPI SYCL island: the linked `sycl_*` C-ABI bridge and the
//! device-resident record session.
//!
//! Two execution forms share one driver handoff ([`handoff`]):
//!
//! * the **direct operator path** ([`drive`], expanded by `GENOPS!`) allocates
//!   every operand, submits the single kernel, waits, downloads the outputs,
//!   and releases the operands — run-to-completion, one kernel per call;
//! * the **recorded path** ([`session`]) keeps every operand and intermediate
//!   result in shared USM, submits each step to a session-owned in-order
//!   queue without waiting, and pays one `sycl_wait` plus the requested
//!   downloads in `apply`. No intermediate result crosses the host boundary,
//!   which is what makes a recorded chain cost one synchronisation instead of
//!   two per step.
//!
//! Both forms go through the single `unsafe` block in [`handoff`], so this
//! file keeps the registered `sycl = 2` site count: the `unsafe extern "C"`
//! declaration island and the driver handoff.
use super::gkernel::{self, buf, scalar};
use crate::catalog::sycl as source;
use crate::engine::gpuerror;
use crate::{GENOP, GENOPS};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

// SAFETY: Every declaration mirrors the C ABI of `grid/sycl.c` (this
// crate's own bridge): the queue is an opaque pointer to a `sycl::queue`,
// USM helpers return raw `malloc_shared` pointers, and every `sycl_*`
// entry takes the queue first, then the kernel's operands in signature
// order, with the same scalar layout the generator uses. All calls happen
// only inside the single `unsafe` block of [`handoff`]; C++ exceptions never
// cross the boundary because each entry catches them and reports through
// `sycl_take_error`.
unsafe extern "C" {
    /// Creates a queue on `device_ordinal` (-1 = default device); returns
    /// null when no device exists.
    fn sycl_queue_new(device_ordinal: i32) -> *mut core::ffi::c_void;
    /// Destroys a queue created by `sycl_queue_new`.
    fn sycl_queue_free(queue: *mut core::ffi::c_void);
    /// Allocates `bytes` of shared USM memory visible to host and device.
    fn sycl_malloc_shared(queue: *mut core::ffi::c_void, bytes: usize) -> *mut core::ffi::c_void;
    /// Frees a shared USM allocation.
    fn sycl_free_shared(queue: *mut core::ffi::c_void, ptr: *mut core::ffi::c_void);
    /// Returns and clears the last error (0 = ok).
    fn sycl_take_error() -> i32;
    /// Waits for every kernel submitted to `queue` so far.
    fn sycl_wait(queue: *mut core::ffi::c_void);
    #[allow(clippy::too_many_arguments)]
    fn sycl_attention(
        queue: *mut core::ffi::c_void,
        q: *const f32,
        k: *const f32,
        v: *const f32,
        out: *mut f32,
        qseq: u32,
        kvseq: u32,
        heads: u32,
        kvheads: u32,
        headdim: u32,
        causal: u32,
    );
    #[allow(clippy::too_many_arguments)]
    fn sycl_attention_cached(
        queue: *mut core::ffi::c_void,
        q: *const f32,
        ck: *const f32,
        cv: *const f32,
        nk: *const f32,
        nv: *const f32,
        out: *mut f32,
        kvseq: u32,
        heads: u32,
        kvheads: u32,
        headdim: u32,
        causal: u32,
    );
    fn sycl_add(queue: *mut core::ffi::c_void, a: *const f32, b: *const f32, c: *mut f32, n: u32);
    fn sycl_mul(queue: *mut core::ffi::c_void, a: *const f32, b: *const f32, c: *mut f32, n: u32);
    fn sycl_silu(queue: *mut core::ffi::c_void, a: *const f32, c: *mut f32, n: u32);
    fn sycl_embedding(
        queue: *mut core::ffi::c_void,
        table: *const f32,
        indices: *const u64,
        out: *mut f32,
        rows: u64,
        width: u64,
    );
    #[allow(clippy::too_many_arguments)]
    fn sycl_matmul(
        queue: *mut core::ffi::c_void,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: u32,
        k: u32,
        n: u32,
        batch: u32,
    );
    fn sycl_rmsnorm(
        queue: *mut core::ffi::c_void,
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        rows: u32,
        width: u32,
        eps: f32,
    );
    #[allow(clippy::too_many_arguments)]
    fn sycl_rope(
        queue: *mut core::ffi::c_void,
        x: *const f32,
        y: *mut f32,
        rows: u32,
        seqaxis: u32,
        heads: u32,
        headdim: u32,
        base: f32,
    );
    fn sycl_rope_at(
        queue: *mut core::ffi::c_void,
        x: *const f32,
        y: *mut f32,
        heads: u32,
        headdim: u32,
        position: u32,
        base: f32,
    );
    #[allow(clippy::too_many_arguments)]
    fn sycl_scan(
        queue: *mut core::ffi::c_void,
        input: *const f32,
        a: *const f32,
        b: *const f32,
        c: *const f32,
        skip: *const f32,
        state: *mut f32,
        out: *mut f32,
        n: u32,
        d: u32,
        nstate: u32,
    );
    fn sycl_softmax(
        queue: *mut core::ffi::c_void,
        x: *const f32,
        y: *mut f32,
        rows: u32,
        width: u32,
    );
}

/// The backend's device state, guarded by a single lock.
pub(crate) struct cache {
    state: Mutex<cachestate>,
}

struct cachestate {
    queue: *mut core::ffi::c_void,
}

impl cache {
    /// Creates a cache with no queue; the SYCL runtime and the default
    /// device are initialised lazily on the first dispatch.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(cachestate {
                queue: core::ptr::null_mut(),
            }),
        }
    }

    /// Acquires the lock, recovering the contents if an earlier holder
    /// panicked.
    fn lock(&self) -> std::sync::MutexGuard<'_, cachestate> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

impl core::fmt::Debug for cache {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("cache").finish_non_exhaustive()
    }
}

/// One shared-USM allocation (opaque; safe code never dereferences it).
type vptr = *mut core::ffi::c_void;

/// One driver handoff: the union of platform operations this island needs.
///
/// The direct path uses `alloc`/`upload`/`uploadu64`/`launch`/`wait`/
/// `download`/`free`; a record session additionally creates and destroys its
/// own queue with `newqueue`/`closequeue`.
#[derive(Clone, Copy)]
enum job<'a> {
    /// Creates a queue on the default device and returns its pointer.
    newqueue,
    /// Destroys the queue passed alongside the job.
    closequeue,
    /// Allocates `bytes` of shared USM.
    alloc(usize),
    /// Allocates and copies a host `f32` slice.
    upload(&'a [f32]),
    /// Allocates and copies a host `u64` slice.
    uploadu64(&'a [u64]),
    /// Submits one kernel on existing allocations.
    launch {
        /// The linked C entry point.
        entry: &'static str,
        /// The bound operand pointers, in kernel signature order.
        ptrs: &'a [vptr],
        /// The scalar arguments, already widened to the kernel ABI lane.
        scalars: &'a [u64],
    },
    /// Waits for every kernel submitted on the queue so far.
    wait,
    /// Copies `len` floats out of an allocation.
    download(vptr, usize),
    /// Releases one allocation.
    free(vptr),
}

/// The typed result of one handoff.
enum outcome {
    /// The operation produced no value.
    none,
    /// An allocation was created.
    ptr(vptr),
    /// Host data was copied out.
    host(Vec<f32>),
}

/// Runs one platform operation inside the island's single driver handoff.
///
/// A null `queue` selects the cache's lazily created shared queue (the direct
/// operator path); a non-null `queue` names a record session's own in-order
/// queue. The two paths never share a queue, so a recorded chain's single
/// `wait` does not observe another request's kernels.
///
/// # Errors
///
/// Returns [`gpuerror::nodevice`] when no device exists,
/// [`gpuerror::allocation`] when a USM allocation fails,
/// [`gpuerror::launch`] when the C++ side reports a submission or wait
/// failure, and [`gpuerror::module`] for an unknown entry.
fn handoff(cache: &cache, queue: vptr, job: job<'_>) -> Result<outcome, gpuerror> {
    // SAFETY: The queue is either created lazily here and owned by `cache`, or
    // a live session queue whose owner keeps it alive for this call. Every USM
    // allocation is a live `sycl_malloc_shared` buffer whose byte size matches
    // its operand, and every host slice outlives the call (it is borrowed for
    // the whole job). Kernel arguments are passed in signature order (queue,
    // operands, scalars). The C++ entries catch every exception and report
    // through `sycl_take_error`, so none crosses the boundary; asynchronous
    // failures are reported by `sycl_wait`.
    unsafe {
        if matches!(job, job::newqueue) {
            let created = sycl_queue_new(-1);
            if created.is_null() {
                return Err(gpuerror::nodevice);
            }
            return Ok(outcome::ptr(created));
        }
        if matches!(job, job::closequeue) {
            sycl_queue_free(queue);
            return Ok(outcome::none);
        }
        let queue = if queue.is_null() {
            let mut state = cache.lock();
            if state.queue.is_null() {
                state.queue = sycl_queue_new(-1);
                if state.queue.is_null() {
                    return Err(gpuerror::nodevice);
                }
            }
            state.queue
        } else {
            queue
        };
        match job {
            job::newqueue | job::closequeue => Ok(outcome::none),
            job::alloc(bytes) => {
                // A zero-byte shared allocation is implementation-defined and
                // may be null; allocate one element and let the caller treat
                // it as empty. Kernels are never launched over a zero range
                // with this pointer dereferenced.
                let ptr = sycl_malloc_shared(queue, bytes.max(4));
                if ptr.is_null() {
                    return Err(gpuerror::allocation);
                }
                Ok(outcome::ptr(ptr))
            }
            job::upload(data) => {
                let ptr = sycl_malloc_shared(queue, (data.len() * 4).max(4));
                if ptr.is_null() {
                    return Err(gpuerror::allocation);
                }
                if !data.is_empty() {
                    core::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut f32, data.len());
                }
                Ok(outcome::ptr(ptr))
            }
            job::uploadu64(data) => {
                let ptr = sycl_malloc_shared(queue, (data.len() * 8).max(8));
                if ptr.is_null() {
                    return Err(gpuerror::allocation);
                }
                if !data.is_empty() {
                    core::ptr::copy_nonoverlapping(data.as_ptr(), ptr as *mut u64, data.len());
                }
                Ok(outcome::ptr(ptr))
            }
            job::launch {
                entry,
                ptrs,
                scalars,
            } => {
                let s = |i: usize| scalars[i];
                match entry {
                    source::ADDENTRY => sycl_add(
                        queue,
                        ptrs[0] as *const f32,
                        ptrs[1] as *const f32,
                        ptrs[2] as *mut f32,
                        s(0) as u32,
                    ),
                    source::MULENTRY => sycl_mul(
                        queue,
                        ptrs[0] as *const f32,
                        ptrs[1] as *const f32,
                        ptrs[2] as *mut f32,
                        s(0) as u32,
                    ),
                    source::SILUENTRY => sycl_silu(
                        queue,
                        ptrs[0] as *const f32,
                        ptrs[1] as *mut f32,
                        s(0) as u32,
                    ),
                    source::EMBEDDINGENTRY => sycl_embedding(
                        queue,
                        ptrs[0] as *const f32,
                        ptrs[1] as *const u64,
                        ptrs[2] as *mut f32,
                        s(0),
                        s(1),
                    ),
                    source::MATMULENTRY => sycl_matmul(
                        queue,
                        ptrs[0] as *const f32,
                        ptrs[1] as *const f32,
                        ptrs[2] as *mut f32,
                        s(0) as u32,
                        s(1) as u32,
                        s(2) as u32,
                        s(3) as u32,
                    ),
                    source::RMSNORMENTRY => sycl_rmsnorm(
                        queue,
                        ptrs[0] as *const f32,
                        ptrs[1] as *const f32,
                        ptrs[2] as *mut f32,
                        s(0) as u32,
                        s(1) as u32,
                        f32::from_bits(s(2) as u32),
                    ),
                    source::ROPEENTRY => sycl_rope(
                        queue,
                        ptrs[0] as *const f32,
                        ptrs[1] as *mut f32,
                        s(0) as u32,
                        s(1) as u32,
                        s(2) as u32,
                        s(3) as u32,
                        f32::from_bits(s(4) as u32),
                    ),
                    source::ROPEATENTRY => sycl_rope_at(
                        queue,
                        ptrs[0] as *const f32,
                        ptrs[1] as *mut f32,
                        s(0) as u32,
                        s(1) as u32,
                        s(2) as u32,
                        f32::from_bits(s(3) as u32),
                    ),
                    source::SCANENTRY => sycl_scan(
                        queue,
                        ptrs[0] as *const f32,
                        ptrs[1] as *const f32,
                        ptrs[2] as *const f32,
                        ptrs[3] as *const f32,
                        ptrs[4] as *const f32,
                        ptrs[5] as *mut f32,
                        ptrs[6] as *mut f32,
                        s(0) as u32,
                        s(1) as u32,
                        s(2) as u32,
                    ),
                    source::SOFTMAXENTRY => sycl_softmax(
                        queue,
                        ptrs[0] as *const f32,
                        ptrs[1] as *mut f32,
                        s(0) as u32,
                        s(1) as u32,
                    ),
                    source::ATTENTIONENTRY => sycl_attention(
                        queue,
                        ptrs[0] as *const f32,
                        ptrs[1] as *const f32,
                        ptrs[2] as *const f32,
                        ptrs[3] as *mut f32,
                        s(0) as u32,
                        s(1) as u32,
                        s(2) as u32,
                        s(3) as u32,
                        s(4) as u32,
                        s(5) as u32,
                    ),
                    source::ATTENTIONCACHEDENTRY => sycl_attention_cached(
                        queue,
                        ptrs[0] as *const f32,
                        ptrs[1] as *const f32,
                        ptrs[2] as *const f32,
                        ptrs[3] as *const f32,
                        ptrs[4] as *const f32,
                        ptrs[5] as *mut f32,
                        s(0) as u32,
                        s(1) as u32,
                        s(2) as u32,
                        s(3) as u32,
                        s(4) as u32,
                    ),
                    _ => {
                        return Err(gpuerror::module {
                            kernel: entry.into(),
                            message: "unknown sycl entry".into(),
                        });
                    }
                }
                let reported = sycl_take_error();
                if reported != 0 {
                    return Err(gpuerror::launch {
                        message: format!("sycl {entry}: error {reported}"),
                    });
                }
                Ok(outcome::none)
            }
            job::wait => {
                sycl_wait(queue);
                let reported = sycl_take_error();
                if reported != 0 {
                    return Err(gpuerror::launch {
                        message: format!("sycl wait: error {reported}"),
                    });
                }
                Ok(outcome::none)
            }
            job::download(ptr, len) => {
                let mut host: Vec<f32> = vec![0.0_f32; len];
                core::ptr::copy_nonoverlapping(ptr as *const f32, host.as_mut_ptr(), len);
                Ok(outcome::host(host))
            }
            job::free(ptr) => {
                if !ptr.is_null() {
                    sycl_free_shared(queue, ptr);
                }
                Ok(outcome::none)
            }
        }
    }
}

/// Extracts an allocation pointer from a handoff result.
fn asptr(outcome: outcome) -> Result<vptr, gpuerror> {
    match outcome {
        outcome::ptr(ptr) => Ok(ptr),
        _ => Err(gpuerror::allocation),
    }
}

/// Creates an independent queue for one record session.
fn newqueue(cache: &cache) -> Result<vptr, gpuerror> {
    asptr(handoff(cache, core::ptr::null_mut(), job::newqueue)?)
}

/// Destroys a session queue.
fn closequeue(cache: &cache, queue: vptr) -> Result<(), gpuerror> {
    handoff(cache, queue, job::closequeue).map(|_| ())
}

/// Allocates `bytes` of shared USM.
fn alloc(cache: &cache, queue: vptr, bytes: usize) -> Result<vptr, gpuerror> {
    asptr(handoff(cache, queue, job::alloc(bytes))?)
}

/// Allocates and copies a host `f32` slice.
fn upload(cache: &cache, queue: vptr, data: &[f32]) -> Result<vptr, gpuerror> {
    asptr(handoff(cache, queue, job::upload(data))?)
}

/// Allocates and copies a host `u64` slice.
fn uploadu64(cache: &cache, queue: vptr, data: &[u64]) -> Result<vptr, gpuerror> {
    asptr(handoff(cache, queue, job::uploadu64(data))?)
}

/// Submits one kernel; completion is requested separately with [`wait`].
fn launch(
    cache: &cache,
    queue: vptr,
    entry: &'static str,
    ptrs: &[vptr],
    scalars: &[u64],
) -> Result<(), gpuerror> {
    handoff(
        cache,
        queue,
        job::launch {
            entry,
            ptrs,
            scalars,
        },
    )
    .map(|_| ())
}

/// Waits for every kernel submitted on `queue`.
fn wait(cache: &cache, queue: vptr) -> Result<(), gpuerror> {
    handoff(cache, queue, job::wait).map(|_| ())
}

/// Copies `len` floats out of `ptr`.
fn download(cache: &cache, queue: vptr, ptr: vptr, len: usize) -> Result<Vec<f32>, gpuerror> {
    match handoff(cache, queue, job::download(ptr, len))? {
        outcome::host(data) => Ok(data),
        _ => Err(gpuerror::transfer {
            message: "sycl download".into(),
        }),
    }
}

/// Drops one allocation.
fn dropallocation(cache: &cache, queue: vptr, ptr: vptr) -> Result<(), gpuerror> {
    handoff(cache, queue, job::free(ptr)).map(|_| ())
}

/// Widens a scalar argument to the kernel ABI lane.
fn encode(scalars: &[scalar]) -> Vec<u64> {
    scalars
        .iter()
        .map(|s| match s {
            scalar::u32(v) => *v as u64,
            scalar::u64(v) => *v,
            scalar::f32(v) => (*v).to_bits() as u64,
        })
        .collect()
}

/// Releases every temporary allocation of one direct dispatch.
///
/// A dispatch can fail after some operands were allocated; the arena frees the
/// live ones on the way out, so every error path releases exactly what it
/// acquired.
struct arena<'a> {
    cache: &'a cache,
    ptrs: Vec<vptr>,
}

impl<'a> arena<'a> {
    fn new(cache: &'a cache) -> Self {
        Self {
            cache,
            ptrs: Vec::new(),
        }
    }
}

impl Drop for arena<'_> {
    fn drop(&mut self) {
        for ptr in self.ptrs.drain(..) {
            let _ = dropallocation(self.cache, core::ptr::null_mut(), ptr);
        }
    }
}

/// The `rmsnorm` scalar group: the SYCL kernel takes an explicit row
/// count because the launch range is built from it.
fn rmsnormparams(rows: usize, width: usize, eps: f32) -> Vec<scalar> {
    vec![
        scalar::u32(rows as u32),
        scalar::u32(width as u32),
        scalar::f32(eps),
    ]
}

/// The `softmax` scalar group: the SYCL kernel takes an explicit row
/// count.
fn softmaxparams(rows: usize, width: usize) -> Vec<scalar> {
    vec![scalar::u32(rows as u32), scalar::u32(width as u32)]
}

/// Runs one kernel to completion through the single driver-handoff block.
///
/// Every operand is allocated in shared USM, uploaded, the kernel is
/// submitted and waited for, the outputs are copied back, and the arena
/// releases every allocation. The queue is the cache's shared queue, created
/// lazily on first use.
///
/// # Errors
///
/// Returns [`gpuerror::device`] when no device exists,
/// [`gpuerror::allocation`] when a USM allocation fails,
/// [`gpuerror::launch`] when the C++ side reports an error, and
/// [`gpuerror::module`] for an unknown entry.
fn drive(
    cache: &cache,
    entry: &'static str,
    buffers: &[buf<'_>],
    scalars: &[scalar],
    _spec: gkernel::launchspec,
) -> Result<Vec<Vec<f32>>, gpuerror> {
    let mut arena = arena::new(cache);
    for operand in buffers {
        let ptr = match operand {
            buf::host(data) => upload(cache, core::ptr::null_mut(), data)?,
            buf::hostu64(data) => uploadu64(cache, core::ptr::null_mut(), data)?,
            buf::inout(data) => upload(cache, core::ptr::null_mut(), data)?,
            buf::out(len) => alloc(cache, core::ptr::null_mut(), len * 4)?,
        };
        arena.ptrs.push(ptr);
    }
    launch(
        cache,
        core::ptr::null_mut(),
        entry,
        &arena.ptrs,
        &encode(scalars),
    )?;
    wait(cache, core::ptr::null_mut())?;
    let mut downloads: Vec<Vec<f32>> = Vec::new();
    for (index, operand) in buffers.iter().enumerate() {
        match operand {
            buf::out(len) => downloads.push(download(
                cache,
                core::ptr::null_mut(),
                arena.ptrs[index],
                *len,
            )?),
            buf::inout(data) => downloads.push(download(
                cache,
                core::ptr::null_mut(),
                arena.ptrs[index],
                data.len(),
            )?),
            _ => {}
        }
    }
    Ok(downloads)
}

// ---------------------------------------------------------------------------
// Device-resident record session
// ---------------------------------------------------------------------------

/// An in-order queue owned by one record session.
///
/// The queue is reference-counted: every allocation created on it holds a
/// handle, so the queue outlives every buffer and is destroyed only after the
/// last one is released.
pub(crate) struct queue {
    ptr: vptr,
    cache: Arc<cache>,
}

impl queue {
    fn new(cache: Arc<cache>) -> Result<Self, gpuerror> {
        let ptr = newqueue(&cache)?;
        Ok(Self { ptr, cache })
    }
}

impl Drop for queue {
    fn drop(&mut self) {
        let _ = closequeue(&self.cache, self.ptr);
    }
}

/// One shared-USM allocation with shared ownership.
///
/// Cloning the handle is cheap and keeps the allocation alive; the last clone
/// releases it. Recorded chains therefore pass intermediate results by handle
/// with no host round trip.
#[derive(Clone)]
pub(crate) struct buffer(Rc<allocation>);

struct allocation {
    ptr: vptr,
    bytes: usize,
    queue: Rc<queue>,
    cache: Arc<cache>,
}

impl buffer {
    fn new(ptr: vptr, bytes: usize, queue: Rc<queue>, cache: Arc<cache>) -> Self {
        Self(Rc::new(allocation {
            ptr,
            bytes,
            queue,
            cache,
        }))
    }

    /// The USM pointer of this allocation.
    fn ptr(&self) -> vptr {
        self.0.ptr
    }

    /// The number of `f32` elements this allocation holds.
    fn len(&self) -> usize {
        self.0.bytes / 4
    }
}

impl Drop for allocation {
    fn drop(&mut self) {
        let _ = dropallocation(&self.cache, self.queue.ptr, self.ptr);
    }
}

/// The recorded output of one step: a live allocation, no elements, or a run
/// of zeros produced without a kernel.
enum result {
    device(buffer),
    empty,
    zeros(usize),
}

/// A device-resident record chain.
///
/// Every step submits its kernel to the session's own in-order queue and
/// leaves its output in a shared-USM allocation. `apply` performs exactly one
/// `sycl_wait`, downloads the recorded outputs in slot order, and releases the
/// step allocations; uploaded caller inputs live in reference-counted handles
/// owned by the submission and survive `apply`, so a persistent session
/// reuses them across chains.
///
/// # Buffer lifetime
///
/// Shared USM has no implicit synchronisation on free, so an allocation must
/// not be released while a submitted kernel may still read it. Three rules
/// keep that true:
///
/// * recorded outputs stay in `results` until `apply` waited;
/// * kernel temporaries (the `embedding` index lanes) stay in `transients`
///   until `apply` waited;
/// * caller uploads are retained by the submission's input table for the
///   whole submission (`recordplatform::DEDUP_INPUTS` is true), never dropped
///   between a record step and `apply`.
pub(crate) struct session {
    cache: Arc<cache>,
    queue: Rc<queue>,
    results: Vec<result>,
    transients: Vec<buffer>,
}

impl session {
    /// Opens a record session with its own queue.
    ///
    /// # Errors
    ///
    /// Returns [`gpuerror::nodevice`] when no SYCL device is available.
    pub(crate) fn new(cache: Arc<cache>) -> Result<Self, gpuerror> {
        let queue = Rc::new(queue::new(cache.clone())?);
        Ok(Self {
            cache,
            queue,
            results: Vec::new(),
            transients: Vec::new(),
        })
    }

    /// Uploads a host slice once and returns its shared handle.
    ///
    /// # Errors
    ///
    /// Returns [`gpuerror::allocation`] when the USM allocation fails.
    pub(crate) fn upload(&self, data: &[f32]) -> Result<buffer, gpuerror> {
        let ptr = upload(&self.cache, self.queue.ptr, data)?;
        Ok(buffer::new(
            ptr,
            data.len() * 4,
            self.queue.clone(),
            self.cache.clone(),
        ))
    }

    /// Uploads a host `u64` slice into a transient allocation that stays live
    /// until the next `apply` (the `embedding` index lanes).
    fn uploadu64(&mut self, data: &[u64]) -> Result<buffer, gpuerror> {
        let ptr = uploadu64(&self.cache, self.queue.ptr, data)?;
        Ok(buffer::new(
            ptr,
            data.len() * 8,
            self.queue.clone(),
            self.cache.clone(),
        ))
    }

    /// Allocates the output of one recorded step, submits the kernel, and
    /// returns the output slot.
    fn launchresult(
        &mut self,
        entry: &'static str,
        operands: &[vptr],
        scalars: &[scalar],
        count: usize,
    ) -> Result<usize, gpuerror> {
        let bytes = count.checked_mul(4).ok_or(gpuerror::sizeoverflow)?;
        let ptr = alloc(&self.cache, self.queue.ptr, bytes)?;
        let output = buffer::new(ptr, bytes, self.queue.clone(), self.cache.clone());
        let mut ptrs: Vec<vptr> = Vec::with_capacity(operands.len() + 1);
        ptrs.extend_from_slice(operands);
        ptrs.push(ptr);
        launch(&self.cache, self.queue.ptr, entry, &ptrs, &encode(scalars))?;
        let index = self.results.len();
        self.results.push(result::device(output));
        Ok(index)
    }

    /// Records a zero-element output without touching the device.
    fn recordempty(&mut self) -> usize {
        let index = self.results.len();
        self.results.push(result::empty);
        index
    }

    /// Records an all-zero output of `count` elements without launching.
    fn recordzeros(&mut self, count: usize) -> usize {
        let index = self.results.len();
        self.results.push(result::zeros(count));
        index
    }

    /// Returns the allocation of device-resident slot `index`.
    pub(crate) fn getbuffer(&self, index: usize) -> Option<buffer> {
        match self.results.get(index) {
            Some(result::device(held)) => Some(held.clone()),
            _ => None,
        }
    }

    /// Returns the host data of a slot that never reached the device.
    pub(crate) fn gethostdata(&self, index: usize) -> Option<Vec<f32>> {
        match self.results.get(index) {
            Some(result::empty) => Some(Vec::new()),
            Some(result::zeros(count)) => Some(vec![0.0_f32; *count]),
            _ => None,
        }
    }

    /// Waits once, downloads every recorded output in slot order, and releases
    /// the step allocations.
    ///
    /// The uploads held by the submission are untouched, so a persistent
    /// session can record the next chain over the same device state.
    ///
    /// # Errors
    ///
    /// Returns [`gpuerror::launch`] when a submitted kernel failed and
    /// [`gpuerror::transfer`] when a download could not be copied.
    pub(crate) fn apply(&mut self) -> Result<Vec<Vec<f32>>, gpuerror> {
        let results = std::mem::take(&mut self.results);
        let waited = wait(&self.cache, self.queue.ptr);
        self.transients.clear();
        waited?;
        results
            .into_iter()
            .map(|result| match result {
                result::device(held) => {
                    download(&self.cache, self.queue.ptr, held.ptr(), held.len())
                }
                result::empty => Ok(Vec::new()),
                result::zeros(count) => Ok(vec![0.0_f32; count]),
            })
            .collect()
    }

    /// Releases the recorded state of the previous chain.
    ///
    /// A chain may be discarded before `apply`, so the queue is drained
    /// first: shared-USM `free` does not wait for in-flight kernels, and the
    /// buffers released below may still be read or written by them. `reset`
    /// has no error channel by contract and a discard has no recovery path,
    /// hence the ignored result — the same policy as [`Drop`].
    pub(crate) fn reset(&mut self) {
        let _ = wait(&self.cache, self.queue.ptr);
        self.results.clear();
        self.transients.clear();
    }
}

impl Drop for session {
    fn drop(&mut self) {
        // A chain dropped before `apply` may still have kernels in flight;
        // drain them before the field drops release their buffers.
        let _ = wait(&self.cache, self.queue.ptr);
    }
}

// ---------------------------------------------------------------------------
// Recorded operator steps
// ---------------------------------------------------------------------------

/// Records one elementwise `add` into a device-resident chain.
pub(crate) fn add_record(
    session: &mut session,
    lhs: &buffer,
    rhs: &buffer,
    count: usize,
) -> Result<usize, gpuerror> {
    if count == 0 {
        return Ok(session.recordempty());
    }
    session.launchresult(
        source::ADDENTRY,
        &[lhs.ptr(), rhs.ptr()],
        &[scalar::u32(count as u32)],
        count,
    )
}

/// Records one elementwise `mul` into a device-resident chain.
pub(crate) fn mul_record(
    session: &mut session,
    lhs: &buffer,
    rhs: &buffer,
    count: usize,
) -> Result<usize, gpuerror> {
    if count == 0 {
        return Ok(session.recordempty());
    }
    session.launchresult(
        source::MULENTRY,
        &[lhs.ptr(), rhs.ptr()],
        &[scalar::u32(count as u32)],
        count,
    )
}

/// Records one `silu` into a device-resident chain.
pub(crate) fn silu_record(
    session: &mut session,
    input: &buffer,
    count: usize,
) -> Result<usize, gpuerror> {
    if count == 0 {
        return Ok(session.recordempty());
    }
    session.launchresult(
        source::SILUENTRY,
        &[input.ptr()],
        &[scalar::u32(count as u32)],
        count,
    )
}

/// Records one `matmul` (including the batched form) into a device-resident
/// chain.
pub(crate) fn matmul_record(
    session: &mut session,
    lhs: &buffer,
    rhs: &buffer,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<usize, gpuerror> {
    let total = batch
        .checked_mul(m)
        .and_then(|value| value.checked_mul(n))
        .ok_or(gpuerror::sizeoverflow)?;
    if m == 0 || n == 0 || batch == 0 {
        return Ok(session.recordempty());
    }
    if k == 0 {
        return Ok(session.recordzeros(total));
    }
    session.launchresult(
        source::MATMULENTRY,
        &[lhs.ptr(), rhs.ptr()],
        &[
            scalar::u32(m as u32),
            scalar::u32(k as u32),
            scalar::u32(n as u32),
            scalar::u32(batch as u32),
        ],
        total,
    )
}

/// Records one RMS normalization into a device-resident chain.
pub(crate) fn rmsnorm_record(
    session: &mut session,
    input: &buffer,
    weight: &buffer,
    rows: usize,
    width: usize,
    eps: f32,
) -> Result<usize, gpuerror> {
    if rows == 0 {
        return Ok(session.recordempty());
    }
    let total = rows.checked_mul(width).ok_or(gpuerror::sizeoverflow)?;
    session.launchresult(
        source::RMSNORMENTRY,
        &[input.ptr(), weight.ptr()],
        &rmsnormparams(rows, width, eps),
        total,
    )
}

/// Records one softmax into a device-resident chain.
///
/// The direct operator path inspects its host input and defers a non-finite
/// row to the host scalar rule; the recorded contract has no host-side
/// finiteness scan, so the recorded step follows the kernel's finite-sentinel
/// semantics. This matches every device-recorded backend: the shared
/// `submission` layer never inspects operands.
pub(crate) fn softmax_record(
    session: &mut session,
    input: &buffer,
    rows: usize,
    width: usize,
) -> Result<usize, gpuerror> {
    if rows == 0 {
        return Ok(session.recordempty());
    }
    let total = rows.checked_mul(width).ok_or(gpuerror::sizeoverflow)?;
    session.launchresult(
        source::SOFTMAXENTRY,
        &[input.ptr()],
        &softmaxparams(rows, width),
        total,
    )
}

/// Records one embedding gather into a device-resident chain.
///
/// The `u64` index lanes are uploaded into a transient allocation owned by the
/// session and released after the next `apply` wait.
pub(crate) fn embedding_record(
    session: &mut session,
    table: &buffer,
    indices: &[u64],
    rows: usize,
    width: usize,
) -> Result<usize, gpuerror> {
    if rows == 0 {
        return Ok(session.recordempty());
    }
    let total = rows.checked_mul(width).ok_or(gpuerror::sizeoverflow)?;
    let lanes = session.uploadu64(indices)?;
    let lanep = lanes.ptr();
    session.transients.push(lanes);
    session.launchresult(
        source::EMBEDDINGENTRY,
        &[table.ptr(), lanep],
        &[scalar::u64(rows as u64), scalar::u64(width as u64)],
        total,
    )
}

/// Records one rotary position embedding into a device-resident chain.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rope_record(
    session: &mut session,
    input: &buffer,
    rows: usize,
    seqaxis: usize,
    heads: usize,
    headdim: usize,
    base: f32,
) -> Result<usize, gpuerror> {
    if rows == 0 {
        return Ok(session.recordempty());
    }
    let total = rows
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(headdim))
        .ok_or(gpuerror::sizeoverflow)?;
    session.launchresult(
        source::ROPEENTRY,
        &[input.ptr()],
        &[
            scalar::u32(rows as u32),
            scalar::u32(seqaxis as u32),
            scalar::u32(heads as u32),
            scalar::u32(headdim as u32),
            scalar::f32(base),
        ],
        total,
    )
}

/// Records one absolute-position rotary embedding into a device-resident
/// chain.
pub(crate) fn rope_at_record(
    session: &mut session,
    input: &buffer,
    position: usize,
    heads: usize,
    headdim: usize,
    base: f32,
) -> Result<usize, gpuerror> {
    let total = heads.checked_mul(headdim).ok_or(gpuerror::sizeoverflow)?;
    session.launchresult(
        source::ROPEATENTRY,
        &[input.ptr()],
        &[
            scalar::u32(heads as u32),
            scalar::u32(headdim as u32),
            scalar::u32(position as u32),
            scalar::f32(base),
        ],
        total,
    )
}

/// Records one multi-head attention into a device-resident chain.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_record(
    session: &mut session,
    query: &buffer,
    key: &buffer,
    value: &buffer,
    qseq: usize,
    kvseq: usize,
    heads: usize,
    kvheads: usize,
    headdim: usize,
    causal: bool,
) -> Result<usize, gpuerror> {
    if headdim > source::ATTNLIMIT as usize {
        return Err(gpuerror::sizeoverflow);
    }
    if qseq == 0 {
        return Ok(session.recordempty());
    }
    let total = qseq
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(headdim))
        .ok_or(gpuerror::sizeoverflow)?;
    session.launchresult(
        source::ATTENTIONENTRY,
        &[query.ptr(), key.ptr(), value.ptr()],
        &[
            scalar::u32(qseq as u32),
            scalar::u32(kvseq as u32),
            scalar::u32(heads as u32),
            scalar::u32(kvheads as u32),
            scalar::u32(headdim as u32),
            scalar::u32(causal as u32),
        ],
        total,
    )
}

/// Records one incremental-decode attention into a device-resident chain.
///
/// The operands are reordered to the kernel signature here: the record
/// contract lists the cached and new key/value pairs separately, while the C
/// entry takes the cached rows before the new rows.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_cached_record(
    session: &mut session,
    query: &buffer,
    keycached: &buffer,
    valuecached: &buffer,
    keynew: &buffer,
    valuenew: &buffer,
    kvseq: usize,
    heads: usize,
    kvheads: usize,
    headdim: usize,
    causal: bool,
) -> Result<usize, gpuerror> {
    if headdim > source::ATTNLIMIT as usize {
        return Err(gpuerror::sizeoverflow);
    }
    let total = heads.checked_mul(headdim).ok_or(gpuerror::sizeoverflow)?;
    session.launchresult(
        source::ATTENTIONCACHEDENTRY,
        &[
            query.ptr(),
            keycached.ptr(),
            valuecached.ptr(),
            keynew.ptr(),
            valuenew.ptr(),
        ],
        &[
            scalar::u32(kvseq as u32),
            scalar::u32(heads as u32),
            scalar::u32(kvheads as u32),
            scalar::u32(headdim as u32),
            scalar::u32(causal as u32),
        ],
        total,
    )
}

GENOPS!(
    add,
    mul,
    silu,
    matmul,
    rmsnorm,
    softmax,
    embedding,
    rope,
    rope_at,
    attention,
    attention_cached,
    scan,
    scan_cached
);

// ---------------------------------------------------------------------------
// cann island
// ---------------------------------------------------------------------------

// The Huawei CANN driver island (Ascend 910/310, AI Core).
//
// The CANN runtime is initialised lazily by the first dispatch; the fatbin
// for `grid/cann.c` (compiled by `ccec`/`bisheng` at build time, embedded
// through the build-time registry) is loaded with `aclrtCreateKernel` and
// every entry is resolved once and cached. CANN launches pack all kernel
// arguments (operands then scalars) into one 8-byte-aligned buffer, so the
// island builds the pack inside the single `unsafe` block of [`drive`];
// [`GENOPS`] expands the operator surface over it. The registered site
// count is `cann = 2`: the `unsafe extern "C"` declaration island and the
// driver-handoff block.
//
// The exact launch ABI varies across CANN releases; the declarations below
// follow CANN 8's `aclrtLaunchKernel` (kernel, blockDim, args, argsSize,
// stream) and are validated on a real toolkit host.
