#![allow(unsafe_code)]
//! The MUSA platform FFI island: every `mu*` driver call in the crate lives
//! here, and the crate root keeps `#![deny(unsafe_code)]`. Each `unsafe`
//! construct is directly preceded by a `// SAFETY:` comment; the registered
//! site count (two) is enforced by `scripts/checkunsafe.sh`.
//!
//! The two sites are the 2024-edition lower bound for a hand-written FFI
//! (docs/musa-backend-plan.md §5.0, §5.4):
//! 1. the `unsafe extern "C"` declaration island mirroring `musa.h`;
//! 2. the single driver-handoff block inside [`dispatch`], which is lazy by
//!    design: device open, module load, function resolution, allocation,
//!    transfers, launch, synchronisation, and freeing all happen inside that
//!    one block, so no second handoff site exists.
//!
//! Every safe function exposed to the executor validates shapes and budgets
//! first, then describes its operands as a [`buf`] list and forwards to
//! [`dispatch`] — a narrow run-to-completion interface like the `metal`
//! backend's `dispatch`, keeping the three GPU backends structurally aligned.

use std::collections::HashMap;
use std::ffi::CString;
use std::sync::Arc;
use std::sync::{Mutex, PoisonError};

use crate::error::musaerror;
use crate::registry::KERNELS;
use crate::source;

// ---------------------------------------------------------------------------
// MUSA driver ABI (musa.h, MUSA Toolkit 3.1.0). The opaque handles are
// one-field repr(C) wrappers; the device pointer is a byte address.
// ---------------------------------------------------------------------------

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

// SAFETY: Every declaration mirrors the C ABI of `musa.h` (MUSA Toolkit
// 3.1.0): the opaque handles are pointer-sized `repr(C)` structs, `MUdeviceptr`
// is a byte address, and `muLaunchKernel`'s `kernelParams` is a pointer to an
// array of pointers, one per kernel parameter, each pointing at a value whose
// type matches the kernel signature the caller mirrors. All calls happen only
// inside the single `unsafe` block of [`dispatch`], guarded by result-code
// checks, so no unvalidated driver call can escape this island.
unsafe extern "C" {
    /// Initialises the MUSA driver (idempotent; flags must be 0 per musa.h).
    fn muInit(flags: u32) -> muresult;
    /// Fills `device` with the `ordinal`-th device.
    fn muDeviceGet(device: *mut mudevice, ordinal: i32) -> muresult;
    /// Creates a context for `device`.
    fn muCtxCreate_v2(context: *mut mucontext, flags: u32, device: mudevice) -> muresult;
    /// Blocks until every preceding operation on `context` completes.
    fn muCtxSynchronize(context: mucontext) -> muresult;
    /// Frees a device allocation.
    fn muMemFree_v2(mem: mudevptr) -> muresult;
    /// Allocates `bytesize` device bytes into `mem`.
    fn muMemAlloc_v2(mem: *mut mudevptr, bytesize: usize) -> muresult;
    /// Copies `count` bytes from host `src` to device `dst`.
    fn muMemcpyHtoD_v2(dst: mudevptr, src: *const core::ffi::c_void, count: usize) -> muresult;
    /// Copies `count` bytes from device `src` to host `dst`.
    fn muMemcpyDtoH_v2(dst: *mut core::ffi::c_void, src: mudevptr, count: usize) -> muresult;
    /// Loads the binary fatbin `data` into `module`.
    fn muModuleLoadData(module: *mut mumodule, data: *const core::ffi::c_void) -> muresult;
    /// Resolves `name` in `module` into `function`.
    fn muModuleGetFunction(
        function: *mut mufunction,
        module: mumodule,
        name: *const core::ffi::c_char,
    ) -> muresult;
    /// Launches `function` on the grid/block geometry with `kernelParams`.
    #[allow(clippy::too_many_arguments)]
    fn muLaunchKernel(
        function: mufunction,
        gridx: u32,
        gridy: u32,
        gridz: u32,
        blockx: u32,
        blocky: u32,
        blockz: u32,
        sharedbytes: u32,
        stream: usize,
        kernelparams: *mut *mut core::ffi::c_void,
        extra: *mut *mut core::ffi::c_void,
    ) -> muresult;
}

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
    /// Loaded modules by source id (`kernels/<stem>.cu`).
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

// ---------------------------------------------------------------------------
// dispatch: the single driver-handoff block.
// ---------------------------------------------------------------------------

/// A scalar kernel argument. Every in-repo kernel takes its buffer operands
/// first and its scalar parameters last, so the launch argument list is
/// exactly the bound device pointers followed by these scalars.
#[derive(Clone, Copy, Debug)]
pub(crate) enum scalar {
    /// An `unsigned` kernel parameter.
    u32(u32),
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
    /// Download an existing device allocation into a host vector without
    /// allocating or launching anything. Used by [`flush`], which runs the
    /// deferred synchronisation and the downloads inside the one registered
    /// handoff block.
    download(mudevptr, usize),
}

/// The two halves of one driver handoff: downloaded host outputs in buffer
/// order, and handed-back device buffers as `(pointer, byte size)` pairs for
/// every `device`-resident slot in buffer order.
struct dispatchresult {
    host: Vec<Vec<f32>>,
    device: Vec<(mudevptr, usize)>,
}

/// Runs one kernel through the single driver-handoff block.
///
/// The block is lazy by design: the first call initialises the driver, opens
/// the device and context, loads the fatbin for `source_id` from the
/// build-time registry, and resolves `entry`; later calls reuse all of it.
/// Host operands are uploaded, the kernel is launched on the default stream,
/// the context is synchronised, outputs are downloaded, and this dispatch's
/// temporary allocations are freed — all inside the one `unsafe` block, so the
/// registered handoff count stays exactly one (docs/musa-backend-plan.md
/// §5.4).
///
/// # Errors
///
/// Returns [`musaerror::device`] when the driver cannot be initialised or no
/// device exists, [`musaerror::module`] when a fatbin cannot be loaded or an
/// entry resolved, [`musaerror::transfer`] when a copy fails,
/// [`musaerror::launch`] when the launch or synchronisation fails, and
/// [`musaerror::sizeoverflow`] when a byte count overflows.
/// Runs one kernel through the single driver-handoff block, downloading
/// every [`buf::out`] slot. Device-resident variants are accepted but their
/// handed-back pointers are discarded: callers of a mixed run use
/// [`dispatchdevice`] instead.
pub(crate) fn dispatch(
    cache: &cache,
    source_id: &'static str,
    entry: &'static str,
    buffers: &[buf<'_>],
    scalars: &[scalar],
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    sharedbytes: u32,
) -> Result<Vec<Vec<f32>>, musaerror> {
    Ok(drive(
        cache,
        Some((source_id, entry)),
        buffers,
        scalars,
        grid,
        block,
        sharedbytes,
        true,
    )?
    .host)
}

/// Runs one kernel through the single driver-handoff block, handing back
/// both halves: downloaded [`buf::out`] slots and every device-resident slot
/// as a `(pointer, byte size)` pair, each in buffer order. The caller owns
/// the device-resident buffers and releases them through [`retiredevice`].
pub(crate) fn dispatchmixed(
    cache: &cache,
    source_id: &'static str,
    entry: &'static str,
    buffers: &[buf<'_>],
    scalars: &[scalar],
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    sharedbytes: u32,
) -> Result<dispatchresult, musaerror> {
    drive(
        cache,
        Some((source_id, entry)),
        buffers,
        scalars,
        grid,
        block,
        sharedbytes,
        true,
    )
}

/// Queues a device-resident buffer for release at the next driver handoff.
///
/// The release itself (pool return or `muMemFree_v2`) happens inside the one
/// registered handoff block, so this function performs no driver call and adds
/// no `unsafe` site. A zero pointer or zero size is a no-op.
pub(crate) fn retiredevice(cache: &cache, mem: mudevptr, bytes: usize) {
    if mem != 0 && bytes != 0 {
        cache.lock().pendingfree.push((mem, bytes));
    }
}

/// Synchronises once and downloads the given device buffers, all inside the
/// one registered handoff block.
///
/// This is the submission path's `apply`: the per-step records ran with the
/// synchronisation deferred, and this call drains the whole chain with a
/// single `muCtxSynchronize` before copying the requested outputs out.
pub(crate) fn flush(
    cache: &cache,
    downloads: &[(mudevptr, usize)],
) -> Result<Vec<Vec<f32>>, musaerror> {
    let slots: Vec<buf<'_>> = downloads
        .iter()
        .map(|&(ptr, bytes)| buf::download(ptr, bytes))
        .collect();
    Ok(drive(cache, None, &slots, &[], (1, 1, 1), (1, 1, 1), 0, true)?.host)
}

/// A device-resident record chain: every step launches immediately on the
/// default stream with the synchronisation deferred, and [`session::apply`]
/// drains the chain with one synchronise plus the requested downloads.
pub(crate) struct session {
    cache: Arc<cache>,
    /// Every device buffer the session owns, in creation order: uploaded
    /// inputs and recorded outputs alike. `apply` frees them all through
    /// [`retiredevice`], whose queue drains inside the next handoff block.
    slots: Vec<(mudevptr, usize)>,
}

impl session {
    /// Creates an empty record chain over `cache`.
    #[must_use]
    pub(crate) fn new(cache: Arc<cache>) -> Self {
        Self {
            cache,
            slots: Vec::new(),
        }
    }

    /// Returns the device pointer of `slot`.
    pub(crate) fn slotptr(&self, slot: usize) -> mudevptr {
        self.slots[slot].0
    }

    /// Uploads a host slice once and returns its slot.
    pub(crate) fn upload(&mut self, data: &[f32]) -> Result<usize, musaerror> {
        let mut result = drive(
            &self.cache,
            None,
            &[buf::hostdevice(data)],
            &[],
            (1, 1, 1),
            (1, 1, 1),
            0,
            false,
        )?;
        let (ptr, bytes) = result.device.remove(0);
        self.slots.push((ptr, bytes));
        Ok(self.slots.len() - 1)
    }

    /// Records one kernel step without synchronising and returns the output
    /// slot. `hostdevice` inputs in `buffers` are uploaded once and kept;
    /// `device` inputs bind existing slots.
    pub(crate) fn record(
        &mut self,
        source_id: &'static str,
        entry: &'static str,
        buffers: &[buf<'_>],
        scalars: &[scalar],
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        sharedbytes: u32,
    ) -> Result<usize, musaerror> {
        let mut result = drive(
            &self.cache,
            Some((source_id, entry)),
            buffers,
            scalars,
            grid,
            block,
            sharedbytes,
            false,
        )?;
        for (ptr, bytes) in result.device {
            self.slots.push((ptr, bytes));
        }
        Ok(self.slots.len() - 1)
    }

    /// Drains one slot to host data mid-chain (an operator still on the
    /// host-replay path consuming a device-resident result). The slot is
    /// released immediately and marked empty.
    pub(crate) fn take(&mut self, slot: usize) -> Result<Vec<f32>, musaerror> {
        let (ptr, bytes) = self.slots[slot];
        let mut out = flush(&self.cache, &[(ptr, bytes)])?;
        retiredevice(&self.cache, ptr, bytes);
        self.slots[slot] = (0, 0);
        Ok(out.remove(0))
    }

    /// Synchronises once, downloads the requested slots in order, and releases
    /// every owned buffer through [`retiredevice`].
    pub(crate) fn apply(&mut self, downloads: &[usize]) -> Result<Vec<Vec<f32>>, musaerror> {
        let wanted: Vec<(mudevptr, usize)> =
            downloads.iter().map(|&index| self.slots[index]).collect();
        let out = flush(&self.cache, &wanted)?;
        for (ptr, bytes) in self.slots.drain(..) {
            retiredevice(&self.cache, ptr, bytes);
        }
        Ok(out)
    }
}

/// Records one elementwise `add` into a device-resident chain.
pub(crate) fn add_record(
    session: &mut session,
    lhs: buf<'_>,
    rhs: buf<'_>,
    count: usize,
) -> Result<usize, musaerror> {
    let grid = ((count as u32).div_ceil(source::ELEMENTWISETILE), 1, 1);
    let block = (source::ELEMENTWISETILE, 1, 1);
    session.record(
        "elementwise",
        source::ADDENTRY,
        &[lhs, rhs, buf::outdevice(count)],
        &[scalar::u32(count as u32)],
        grid,
        block,
        0,
    )
}

/// Records one elementwise `mul` into a device-resident chain.
pub(crate) fn mul_record(
    session: &mut session,
    lhs: buf<'_>,
    rhs: buf<'_>,
    count: usize,
) -> Result<usize, musaerror> {
    let grid = ((count as u32).div_ceil(source::ELEMENTWISETILE), 1, 1);
    let block = (source::ELEMENTWISETILE, 1, 1);
    session.record(
        "elementwise",
        source::MULENTRY,
        &[lhs, rhs, buf::outdevice(count)],
        &[scalar::u32(count as u32)],
        grid,
        block,
        0,
    )
}

/// Records one `silu` into a device-resident chain.
pub(crate) fn silu_record(
    session: &mut session,
    input: buf<'_>,
    count: usize,
) -> Result<usize, musaerror> {
    let grid = ((count as u32).div_ceil(source::ELEMENTWISETILE), 1, 1);
    let block = (source::ELEMENTWISETILE, 1, 1);
    session.record(
        "elementwise",
        source::SILUENTRY,
        &[input, buf::outdevice(count)],
        &[scalar::u32(count as u32)],
        grid,
        block,
        0,
    )
}

/// Records one RMS normalization into a device-resident chain.
pub(crate) fn rmsnorm_record(
    session: &mut session,
    input: buf<'_>,
    weight: buf<'_>,
    rows: usize,
    width: usize,
    eps: f32,
) -> Result<usize, musaerror> {
    let count = rows.checked_mul(width).ok_or(musaerror::sizeoverflow)?;
    let grid = (rows as u32, 1, 1);
    let block = (source::ROWTILE, 1, 1);
    session.record(
        "rmsnorm",
        source::RMSNORMENTRY,
        &[input, weight, buf::outdevice(count)],
        &[scalar::u32(width as u32), scalar::f32(eps)],
        grid,
        block,
        source::ROWTILE as u32 * 4,
    )
}

/// Records one softmax over the last axis into a device-resident chain.
pub(crate) fn softmax_record(
    session: &mut session,
    input: buf<'_>,
    rows: usize,
    width: usize,
) -> Result<usize, musaerror> {
    let count = rows.checked_mul(width).ok_or(musaerror::sizeoverflow)?;
    let grid = (rows as u32, 1, 1);
    let block = (source::ELEMENTWISETILE, 1, 1);
    session.record(
        "softmax",
        source::SOFTMAXENTRY,
        &[input, buf::outdevice(count)],
        &[scalar::u32(width as u32)],
        grid,
        block,
        source::ELEMENTWISETILE as u32 * 4,
    )
}

/// The single registered driver-handoff block behind [`dispatch`] and
/// [`dispatchdevice`].
///
/// The block is lazy by design: the first call initialises the driver, opens
/// the device and context, loads the fatbin for `source_id` from the
/// build-time registry, and resolves `entry`; later calls reuse all of it.
/// Host operands are uploaded, the kernel is launched on the default stream,
/// the context is synchronised, outputs are downloaded, and this dispatch's
/// temporary allocations are returned to the pool — all inside the one
/// `unsafe` block, so the registered handoff count stays exactly one
/// (docs/musa-backend-plan.md §5.4).
#[allow(clippy::too_many_arguments)]
fn drive(
    cache: &cache,
    kernel: Option<(&'static str, &'static str)>,
    buffers: &[buf<'_>],
    scalars: &[scalar],
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    sharedbytes: u32,
    sync: bool,
) -> Result<dispatchresult, musaerror> {
    // The state lock is held across the whole handoff: the lazy open and
    // function resolution mutate the shared caches, and the default stream
    // serialises launches anyway.
    let mut state = cache.lock();

    // SAFETY: All driver calls in this block share one precondition: every
    // handle and pointer is either (a) a handle produced by an earlier call in
    // this block or held in `state` from a previous dispatch, or (b) a live
    // value owned by this call (host slices, the fatbin bytes, the C string
    // entry name, the argument lists). Every allocation size is derived from
    // the host slice length or the output element count with checked
    // arithmetic, and every copy length equals the allocation it touches. The
    // launch geometry comes from the caller's validated shapes and the kernels
    // bounds-test every index they derive. `kernelparams` points at the local
    // argument arrays, which outlive the launch. Every call's result is
    // checked against `MUSA_SUCCESS` and converted to `musaerror`; no result
    // is ignored. The synchronisation orders the downloads after the kernel's
    // writes, and a buffer is returned to the pool only after its last use in
    // this handoff; the lock held across the whole block means no pooled
    // buffer is ever checked out by two dispatches at once, and the
    // device-resident variants bypass the pool return because the caller owns
    // them through [`retiredevice`].
    unsafe {
        // --- lazy open -----------------------------------------------------
        if !state.opened {
            if muInit(0) != MUSA_SUCCESS {
                return Err(musaerror::device {
                    message: "muInit failed".into(),
                });
            }
            let mut device: mudevice = 0;
            if muDeviceGet(&mut device, 0) != MUSA_SUCCESS {
                return Err(musaerror::device {
                    message: "muDeviceGet(0) failed: no MUSA device".into(),
                });
            }
            let mut context: mucontext = 0;
            if muCtxCreate_v2(&mut context, 0, device) != MUSA_SUCCESS {
                return Err(musaerror::device {
                    message: "muCtxCreate failed".into(),
                });
            }
            state.opened = true;
            state.device = Some(device);
            state.context = Some(context);
        }
        let context = state.context.ok_or(musaerror::device {
            message: "context not open".into(),
        })?;

        // --- deferred releases --------------------------------------------
        // Buffers released by `retiredevice` re-enter the pool (or are freed)
        // here, inside the one registered handoff block, before any new
        // allocation so a just-released slot can serve this very dispatch.
        let pending: Vec<(mudevptr, usize)> = state.pendingfree.drain(..).collect();
        for (mem, bytes) in pending {
            let bucket = state.pool.entry(bytes).or_default();
            if bucket.len() < POOLCAP {
                bucket.push(mem);
            } else if muMemFree_v2(mem) != MUSA_SUCCESS {
                return Err(musaerror::transfer {
                    message: "muMemFree failed for released device buffer".into(),
                });
            }
        }

        // --- lazy module load and kernel resolution ------------------------
        if let Some((source_id, entry)) = kernel {
            if !state.functions.contains_key(&(source_id, entry)) {
                let fatbin = KERNELS
                    .iter()
                    .find(|(id, _)| *id == source_id)
                    .map(|(_, bytes)| *bytes)
                    .ok_or_else(|| musaerror::module {
                        kernel: entry.to_string(),
                        message: format!(
                            "no fatbin for source `{source_id}` (was mcc available at build time?)"
                        ),
                    })?;
                let module = if let Some(module) = state.modules.get(source_id) {
                    *module
                } else {
                    let mut module: mumodule = 0;
                    if muModuleLoadData(&mut module, fatbin.as_ptr().cast()) != MUSA_SUCCESS {
                        return Err(musaerror::module {
                            kernel: entry.to_string(),
                            message: format!("muModuleLoadData failed for `{source_id}`"),
                        });
                    }
                    state.modules.insert(source_id, module);
                    module
                };
                let name = CString::new(entry).map_err(|_| musaerror::module {
                    kernel: entry.to_string(),
                    message: "entry name contains a NUL byte".into(),
                })?;
                let mut function: mufunction = 0;
                if muModuleGetFunction(&mut function, module, name.as_ptr()) != MUSA_SUCCESS {
                    return Err(musaerror::module {
                        kernel: entry.to_string(),
                        message: format!("muModuleGetFunction failed for `{entry}`"),
                    });
                }
                state.functions.insert((source_id, entry), function);
            }
        }
        let function = match kernel {
            Some((source_id, entry)) => {
                Some(*state.functions.get(&(source_id, entry)).ok_or_else(|| {
                    musaerror::module {
                        kernel: entry.to_string(),
                        message: "kernel not resolved".into(),
                    }
                })?)
            }
            None => None,
        };

        // --- allocate and upload ------------------------------------------
        // Byte sizes first (all checked), then allocate, then copy, so a
        // failed allocation leaves nothing half-uploaded. Device-resident
        // slots (`device`/`outdevice`/`hostdevice`) are marked `keep`: they
        // skip the pool return and their `(pointer, byte size)` pair is
        // handed back to the caller in buffer order.
        let mut sizes: Vec<usize> = Vec::with_capacity(buffers.len());
        let mut owned: Vec<usize> = Vec::with_capacity(buffers.len());
        let mut keep: Vec<bool> = Vec::with_capacity(buffers.len());
        for slot in buffers {
            match *slot {
                buf::host(data) | buf::hostdevice(data) => {
                    sizes.push(
                        data.len()
                            .checked_mul(core::mem::size_of::<f32>())
                            .ok_or(musaerror::sizeoverflow)?,
                    );
                    keep.push(matches!(*slot, buf::hostdevice(_)));
                    owned.push(0);
                }
                buf::hostu64(data) => {
                    sizes.push(
                        data.len()
                            .checked_mul(core::mem::size_of::<u64>())
                            .ok_or(musaerror::sizeoverflow)?,
                    );
                    keep.push(false);
                    owned.push(0);
                }
                buf::out(count) | buf::outdevice(count) => {
                    sizes.push(
                        count
                            .checked_mul(core::mem::size_of::<f32>())
                            .ok_or(musaerror::sizeoverflow)?,
                    );
                    keep.push(matches!(*slot, buf::outdevice(_)));
                    owned.push(0);
                }
                buf::device(mem) => {
                    sizes.push(0);
                    keep.push(true);
                    owned.push(mem);
                }
                buf::download(mem, bytes) => {
                    sizes.push(bytes);
                    keep.push(true);
                    owned.push(mem);
                }
            }
        }
        for (index, &size) in sizes.iter().enumerate() {
            if size == 0 {
                continue;
            }
            let mem = if let Some(pooled) = state.pool.get_mut(&size).and_then(Vec::pop) {
                pooled
            } else {
                let mut mem: mudevptr = 0;
                if muMemAlloc_v2(&mut mem, size) != MUSA_SUCCESS {
                    return Err(musaerror::transfer {
                        message: format!("muMemAlloc failed for {size} bytes"),
                    });
                }
                mem
            };
            owned[index] = mem;
        }
        for (index, slot) in buffers.iter().enumerate() {
            match *slot {
                buf::host(data) | buf::hostdevice(data) => {
                    if muMemcpyHtoD_v2(owned[index], data.as_ptr().cast(), sizes[index])
                        != MUSA_SUCCESS
                    {
                        return Err(musaerror::transfer {
                            message: "muMemcpyHtoD failed for f32 operand".into(),
                        });
                    }
                }
                buf::hostu64(data) => {
                    if muMemcpyHtoD_v2(owned[index], data.as_ptr().cast(), sizes[index])
                        != MUSA_SUCCESS
                    {
                        return Err(musaerror::transfer {
                            message: "muMemcpyHtoD failed for u64 operand".into(),
                        });
                    }
                }
                buf::out(_) | buf::outdevice(_) | buf::device(_) | buf::download(_, _) => {}
            }
        }
        // Buffer pointers: every bound operand is backed by an owned
        // allocation or a caller-owned device pointer. `ptrs` outlives the
        // launch.
        let mut ptrs: Vec<usize> = owned.clone();

        // --- launch on the default stream, then synchronise ----------------
        let mut rawargs: Vec<*mut core::ffi::c_void> =
            Vec::with_capacity(ptrs.len() + scalars.len());
        rawargs.extend(ptrs.iter_mut().map(|ptr| ptr as *mut usize as *mut _));
        let mut rawscalars: Vec<scalar> = scalars.to_vec();
        for value in rawscalars.iter_mut() {
            match value {
                scalar::u32(number) => rawargs.push(number as *mut u32 as *mut _),
                scalar::f32(number) => rawargs.push(number as *mut f32 as *mut _),
            }
        }
        if let Some(function) = function {
            if muLaunchKernel(
                function,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                sharedbytes,
                MUNULLSTREAM,
                rawargs.as_mut_ptr(),
                core::ptr::null_mut(),
            ) != MUSA_SUCCESS
            {
                return Err(musaerror::launch {
                    message: "muLaunchKernel failed".into(),
                });
            }
        }
        if sync && muCtxSynchronize(context) != MUSA_SUCCESS {
            return Err(musaerror::launch {
                message: "muCtxSynchronize failed".into(),
            });
        }

        // --- download outputs ---------------------------------------------
        let mut host: Vec<Vec<f32>> = Vec::with_capacity(buffers.len());
        let mut device: Vec<(mudevptr, usize)> = Vec::with_capacity(buffers.len());
        for (index, slot) in buffers.iter().enumerate() {
            match *slot {
                buf::out(count) => {
                    let mut out = Vec::new();
                    out.try_reserve_exact(count)
                        .map_err(|_| musaerror::sizeoverflow)?;
                    out.resize(count, 0.0_f32);
                    if muMemcpyDtoH_v2(out.as_mut_ptr().cast(), owned[index], sizes[index])
                        != MUSA_SUCCESS
                    {
                        return Err(musaerror::transfer {
                            message: "muMemcpyDtoH failed for output".into(),
                        });
                    }
                    host.push(out);
                }
                buf::outdevice(_) | buf::hostdevice(_) => {
                    device.push((owned[index], sizes[index]));
                }
                buf::device(_) => {}
                buf::download(ptr, bytes) => {
                    let mut out = Vec::new();
                    out.try_reserve_exact(bytes / core::mem::size_of::<f32>())
                        .map_err(|_| musaerror::sizeoverflow)?;
                    out.resize(bytes / core::mem::size_of::<f32>(), 0.0_f32);
                    if muMemcpyDtoH_v2(out.as_mut_ptr().cast(), ptr, bytes) != MUSA_SUCCESS {
                        return Err(musaerror::transfer {
                            message: "muMemcpyDtoH failed for flush download".into(),
                        });
                    }
                    host.push(out);
                }
                buf::host(_) | buf::hostu64(_) => {}
            }
        }

        // --- return this dispatch's buffers to the pool --------------------
        // Every non-zero allocation this dispatch checked out goes back by its
        // exact byte size, so the next dispatch of the same size reuses it
        // instead of paying the driver allocation cost. The pool never hands a
        // buffer to two dispatches at once: the lock is held across this whole
        // block and the default stream serialises launches. A bucket that has
        // reached `POOLCAP` frees its excess instead, bounding device memory.
        for (index, &mem) in owned.iter().enumerate() {
            if sizes[index] == 0 || keep[index] {
                continue;
            }
            let bucket = state.pool.entry(sizes[index]).or_default();
            if bucket.len() < POOLCAP {
                bucket.push(mem);
            } else if muMemFree_v2(mem) != MUSA_SUCCESS {
                return Err(musaerror::transfer {
                    message: "muMemFree failed for excess pooled buffer".into(),
                });
            }
        }
        Ok(dispatchresult { host, device })
    }
}

// ---------------------------------------------------------------------------
// The safe operator surface the executor calls.
// ---------------------------------------------------------------------------

/// Computes `[batch, m, k] * [batch, k, n]` with the reference FMA kernel.
///
/// The kernel is compiled at build time by `mcc`; this is the A1 path (no
/// runtime JIT). The batch dimension is the grid's third axis, so every slice
/// is uploaded, launched, and downloaded in one dispatch instead of one per
/// slice. The WMMA Tensor-Core kernel is a separate measurable stage.
pub(crate) fn matmul(
    cache: &cache,
    lhs: &[f32],
    rhs: &[f32],
    m: usize,
    k: usize,
    n: usize,
    batch: usize,
) -> Result<Vec<f32>, musaerror> {
    if batch == 0 {
        return Ok(Vec::new());
    }
    let total = m
        .checked_mul(n)
        .and_then(|value| value.checked_mul(batch))
        .ok_or(musaerror::sizeoverflow)?;
    let grid = (
        (n as u32).div_ceil(source::THREADEDGE),
        (m as u32).div_ceil(source::THREADEDGE),
        batch as u32,
    );
    let block = (source::THREADEDGE, source::THREADEDGE, 1);
    let mut results = dispatch(
        cache,
        "matmul",
        source::MATMULENTRY,
        &[buf::host(lhs), buf::host(rhs), buf::out(total)],
        &[
            scalar::u32(m as u32),
            scalar::u32(k as u32),
            scalar::u32(n as u32),
            scalar::u32(batch as u32),
        ],
        grid,
        block,
        0,
    )?;
    Ok(results.remove(0))
}

/// Elementwise `lhs + rhs`.
pub(crate) fn add(cache: &cache, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, musaerror> {
    let count = lhs.len();
    let grid = ((count as u32).div_ceil(source::ELEMENTWISETILE), 1, 1);
    let block = (source::ELEMENTWISETILE, 1, 1);
    let mut results = dispatch(
        cache,
        "elementwise",
        source::ADDENTRY,
        &[buf::host(lhs), buf::host(rhs), buf::out(count)],
        &[scalar::u32(count as u32)],
        grid,
        block,
        0,
    )?;
    Ok(results.remove(0))
}

/// Elementwise `lhs * rhs`.
pub(crate) fn mul(cache: &cache, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, musaerror> {
    let count = lhs.len();
    let grid = ((count as u32).div_ceil(source::ELEMENTWISETILE), 1, 1);
    let block = (source::ELEMENTWISETILE, 1, 1);
    let mut results = dispatch(
        cache,
        "elementwise",
        source::MULENTRY,
        &[buf::host(lhs), buf::host(rhs), buf::out(count)],
        &[scalar::u32(count as u32)],
        grid,
        block,
        0,
    )?;
    Ok(results.remove(0))
}

/// Sigmoid linear unit, elementwise.
pub(crate) fn silu(cache: &cache, input: &[f32]) -> Result<Vec<f32>, musaerror> {
    let count = input.len();
    let grid = ((count as u32).div_ceil(source::ELEMENTWISETILE), 1, 1);
    let block = (source::ELEMENTWISETILE, 1, 1);
    let mut results = dispatch(
        cache,
        "elementwise",
        source::SILUENTRY,
        &[buf::host(input), buf::out(count)],
        &[scalar::u32(count as u32)],
        grid,
        block,
        0,
    )?;
    Ok(results.remove(0))
}

/// RMS normalisation over the last axis.
pub(crate) fn rmsnorm(
    cache: &cache,
    input: &[f32],
    weight: &[f32],
    rows: usize,
    width: usize,
    eps: f32,
) -> Result<Vec<f32>, musaerror> {
    let count = rows.checked_mul(width).ok_or(musaerror::sizeoverflow)?;
    let grid = (rows as u32, 1, 1);
    let block = (source::ROWTILE, 1, 1);
    let mut results = dispatch(
        cache,
        "rmsnorm",
        source::RMSNORMENTRY,
        &[buf::host(input), buf::host(weight), buf::out(count)],
        &[scalar::u32(width as u32), scalar::f32(eps)],
        grid,
        block,
        source::ROWTILE as u32 * 4,
    )?;
    Ok(results.remove(0))
}

/// Numerically stable softmax over the last axis.
pub(crate) fn softmax(
    cache: &cache,
    input: &[f32],
    rows: usize,
    width: usize,
) -> Result<Vec<f32>, musaerror> {
    let count = rows.checked_mul(width).ok_or(musaerror::sizeoverflow)?;
    let grid = (rows as u32, 1, 1);
    let block = (source::ELEMENTWISETILE, 1, 1);
    let mut results = dispatch(
        cache,
        "softmax",
        source::SOFTMAXENTRY,
        &[buf::host(input), buf::out(count)],
        &[scalar::u32(width as u32)],
        grid,
        block,
        source::ELEMENTWISETILE as u32 * 4,
    )?;
    Ok(results.remove(0))
}

/// Rotary position embedding.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rope(
    cache: &cache,
    input: &[f32],
    rows: usize,
    seqaxis: usize,
    heads: usize,
    headdim: usize,
    base: f32,
) -> Result<Vec<f32>, musaerror> {
    let count = input.len();
    let grid = ((seqaxis as u32).div_ceil(source::ELEMENTWISETILE), 1, 1);
    let block = (source::ELEMENTWISETILE, 1, 1);
    let mut results = dispatch(
        cache,
        "rope",
        source::ROPEENTRY,
        &[buf::host(input), buf::out(count)],
        &[
            scalar::u32(rows as u32),
            scalar::u32(seqaxis as u32),
            scalar::u32(heads as u32),
            scalar::u32(headdim as u32),
            scalar::f32(base),
        ],
        grid,
        block,
        0,
    )?;
    Ok(results.remove(0))
}

/// Absolute-position rotary embedding.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rope_at(
    cache: &cache,
    input: &[f32],
    position: usize,
    heads: usize,
    headdim: usize,
    base: f32,
) -> Result<Vec<f32>, musaerror> {
    let count = input.len();
    let grid = (1, 1, 1);
    let block = (source::ELEMENTWISETILE, 1, 1);
    let mut results = dispatch(
        cache,
        "rope_at",
        source::ROPEATENTRY,
        &[buf::host(input), buf::out(count)],
        &[
            scalar::u32(heads as u32),
            scalar::u32(headdim as u32),
            scalar::u32(position as u32),
            scalar::f32(base),
        ],
        grid,
        block,
        0,
    )?;
    Ok(results.remove(0))
}

/// Gathers `rows` of `width` elements from `table` by `indices`.
pub(crate) fn embedding(
    cache: &cache,
    table: &[f32],
    indices: &[u64],
    width: usize,
) -> Result<Vec<f32>, musaerror> {
    let rows = indices.len();
    let count = rows.checked_mul(width).ok_or(musaerror::sizeoverflow)?;
    let grid = (rows as u32, 1, 1);
    let block = (source::ELEMENTWISETILE, 1, 1);
    let mut results = dispatch(
        cache,
        "embedding",
        source::EMBEDDINGENTRY,
        &[buf::host(table), buf::hostu64(indices), buf::out(count)],
        &[scalar::u32(rows as u32), scalar::u32(width as u32)],
        grid,
        block,
        0,
    )?;
    Ok(results.remove(0))
}

/// Multi-head scaled dot-product attention.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention(
    cache: &cache,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    qseq: usize,
    kvseq: usize,
    heads: usize,
    kvheads: usize,
    headdim: usize,
    causal: bool,
) -> Result<Vec<f32>, musaerror> {
    let count = qseq
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(headdim))
        .ok_or(musaerror::sizeoverflow)?;
    let grid = (
        qseq.checked_mul(heads).ok_or(musaerror::sizeoverflow)? as u32,
        1,
        1,
    );
    let block = (source::ATTENTIONLANES, 1, 1);
    let mut results = dispatch(
        cache,
        "attention",
        source::ATTENTIONENTRY,
        &[
            buf::host(query),
            buf::host(key),
            buf::host(value),
            buf::out(count),
        ],
        &[
            scalar::u32(qseq as u32),
            scalar::u32(kvseq as u32),
            scalar::u32(heads as u32),
            scalar::u32(kvheads as u32),
            scalar::u32(headdim as u32),
            scalar::u32(causal as u32),
        ],
        grid,
        block,
        (2 * source::ATTENTIONLANES as u32 + headdim as u32) * 4,
    )?;
    Ok(results.remove(0))
}

/// Incremental-decode attention over a cached history plus a new row.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_cached(
    cache: &cache,
    query: &[f32],
    keycached: &[f32],
    keynew: &[f32],
    valuecached: &[f32],
    valuenew: &[f32],
    kvseq: usize,
    heads: usize,
    kvheads: usize,
    headdim: usize,
    causal: bool,
) -> Result<Vec<f32>, musaerror> {
    let count = heads.checked_mul(headdim).ok_or(musaerror::sizeoverflow)?;
    let grid = (heads as u32, 1, 1);
    let block = (source::ATTENTIONLANES, 1, 1);
    let mut results = dispatch(
        cache,
        "attention_cached",
        source::ATTENTIONCACHEDENTRY,
        &[
            buf::host(query),
            buf::host(keycached),
            buf::host(valuecached),
            buf::host(keynew),
            buf::host(valuenew),
            buf::out(count),
        ],
        &[
            scalar::u32(kvseq as u32),
            scalar::u32(heads as u32),
            scalar::u32(kvheads as u32),
            scalar::u32(headdim as u32),
            scalar::u32(causal as u32),
        ],
        grid,
        block,
        (2 * source::ATTENTIONLANES as u32 + headdim as u32) * 4,
    )?;
    Ok(results.remove(0))
}

/// Serial per-channel semigroup scan (the bit-for-bit reference path).
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_serial(
    cache: &cache,
    input: &[f32],
    init: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    skip: &[f32],
    n: usize,
    d: usize,
    statewidth: usize,
) -> Result<(Vec<f32>, Vec<f32>), musaerror> {
    let outcount = n.checked_mul(d).ok_or(musaerror::sizeoverflow)?;
    let grid = (d as u32, 1, 1);
    let block = (source::ELEMENTWISETILE, 1, 1);
    let mut results = dispatch(
        cache,
        "scan",
        source::SCANENTRY,
        &[
            buf::host(input),
            buf::host(a),
            buf::host(b),
            buf::host(c),
            buf::host(skip),
            buf::host(init),
            buf::out(outcount),
            buf::out(statewidth.checked_mul(d).ok_or(musaerror::sizeoverflow)?),
        ],
        &[
            scalar::u32(n as u32),
            scalar::u32(d as u32),
            scalar::u32(statewidth as u32),
        ],
        grid,
        block,
        0,
    )?;
    let out = results.remove(0);
    let state = results.remove(0);
    Ok((out, state))
}

/// Parallel three-stage semigroup scan for state widths within
/// [`source::SCANMAXN`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_parallel(
    cache: &cache,
    input: &[f32],
    init: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    skip: &[f32],
    n: usize,
    d: usize,
    statewidth: usize,
) -> Result<(Vec<f32>, Vec<f32>), musaerror> {
    // The three passes chain on device-resident intermediates: pass 1 keeps
    // `dtotal` and the six shared operands on the device (uploaded once),
    // pass 2 binds them, keeps `dprefix` on the device and downloads only the
    // final state, and pass 3 binds the retained operands and downloads only
    // the output. The intermediates are released through `retiredevice`, whose
    // queue is drained inside the next handoff block, so the release adds no
    // driver call of its own (docs/musa-backend-plan.md §5.5).
    let blocks = n.div_ceil(source::SCANCHUNK as usize);
    let dntotal = d.checked_mul(statewidth).ok_or(musaerror::sizeoverflow)?;
    let pairbytes = |count: usize| {
        count
            .checked_mul(2)
            .and_then(|pairs| pairs.checked_mul(dntotal))
            .and_then(|elements| elements.checked_mul(core::mem::size_of::<f32>()))
            .ok_or(musaerror::sizeoverflow)
    };
    let aggsize = pairbytes(blocks)? / core::mem::size_of::<f32>();

    // Pass 1: per-block totals; every operand upload is retained.
    let mut totals = dispatchmixed(
        cache,
        "scan_parallel",
        source::SCANAGGENTRY,
        &[
            buf::hostdevice(input),
            buf::hostdevice(a),
            buf::hostdevice(b),
            buf::hostdevice(c),
            buf::hostdevice(skip),
            buf::hostdevice(init),
            buf::outdevice(aggsize),
        ],
        &[
            scalar::u32(n as u32),
            scalar::u32(d as u32),
            scalar::u32(statewidth as u32),
            scalar::u32(source::SCANCHUNK),
            scalar::u32(blocks as u32),
        ],
        (blocks as u32, (d as u32).div_ceil(source::SCANBLOCKDIM), 1),
        (source::SCANBLOCKDIM, 1, 1),
        0,
    )?;
    // Device slots in buffer order: input, a, b, c, skip, init, dtotal.
    let dinput = totals.device[0];
    let da = totals.device[1];
    let db = totals.device[2];
    let dc = totals.device[3];
    let dskip = totals.device[4];
    let dinit = totals.device[5];
    let dtotal = totals.device[6];

    // Pass 2: inter-block prefix plus the whole-sequence final state.
    let prefixcount = blocks.checked_add(1).ok_or(musaerror::sizeoverflow)?;
    let statecount = dntotal;
    let mut prefix = dispatchmixed(
        cache,
        "scan_parallel",
        source::SCANPREFIXENTRY,
        &[
            buf::device(dtotal.0),
            buf::device(dinit.0),
            buf::out(statecount),
            buf::outdevice(pairbytes(prefixcount)? / core::mem::size_of::<f32>()),
        ],
        &[
            scalar::u32(n as u32),
            scalar::u32(d as u32),
            scalar::u32(statewidth as u32),
            scalar::u32(source::SCANCHUNK),
            scalar::u32(blocks as u32),
        ],
        ((statecount as u32).min(1024), 1, 1),
        ((statecount as u32).min(1024), 1, 1),
        0,
    )?;
    let state = prefix.host.remove(0);
    // Only producer-side slots are collected; the bound pass-throughs
    // (`device`) are not, so the single collected entry is `dprefix`.
    let dprefix = prefix.device[0];
    retiredevice(cache, dtotal.0, dtotal.1);

    // Pass 3: per-block apply; only the output is downloaded.
    let outcount = n.checked_mul(d).ok_or(musaerror::sizeoverflow)?;
    let mut results = dispatch(
        cache,
        "scan_parallel",
        source::SCANAPPLYENTRY,
        &[
            buf::device(dinput.0),
            buf::device(da.0),
            buf::device(db.0),
            buf::device(dc.0),
            buf::device(dskip.0),
            buf::device(dinit.0),
            buf::device(dprefix.0),
            buf::out(outcount),
        ],
        &[
            scalar::u32(n as u32),
            scalar::u32(d as u32),
            scalar::u32(statewidth as u32),
            scalar::u32(source::SCANCHUNK),
            scalar::u32(blocks as u32),
        ],
        (blocks as u32, (d as u32).div_ceil(source::SCANBLOCKDIM), 1),
        (source::SCANBLOCKDIM, 1, 1),
        0,
    )?;
    let out = results.remove(0);
    retiredevice(cache, dinput.0, dinput.1);
    retiredevice(cache, da.0, da.1);
    retiredevice(cache, db.0, db.1);
    retiredevice(cache, dc.0, dc.1);
    retiredevice(cache, dskip.0, dskip.1);
    retiredevice(cache, dinit.0, dinit.1);
    retiredevice(cache, dprefix.0, dprefix.1);
    Ok((out, state))
}

/// Semigroup scan: parallel for state widths within [`source::SCANMAXN`],
/// serial (the bit-for-bit reference) above it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan(
    cache: &cache,
    input: &[f32],
    init: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    skip: &[f32],
    n: usize,
    d: usize,
    statewidth: usize,
) -> Result<(Vec<f32>, Vec<f32>), musaerror> {
    // A single chunk (`n <= SCANCHUNK`) degenerates the three-pass parallel
    // scan to one block: pass 1 aggregates the whole sequence, pass 2 is a
    // trivial identity prefix, and pass 3 recomputes it — three dispatches for
    // no parallelism over the serial kernel, which already runs one thread per
    // channel. Route the single-chunk case to the serial reference, which is
    // the bit-for-bit continuation of the same recurrence.
    if statewidth > source::SCANMAXN as usize || n <= source::SCANCHUNK as usize {
        return scan_serial(cache, input, init, a, b, c, skip, n, d, statewidth);
    }
    scan_parallel(cache, input, init, a, b, c, skip, n, d, statewidth)
}

/// Single-step incremental scan (decode form).
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_cached(
    cache: &cache,
    input: &[f32],
    state: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    skip: &[f32],
    d: usize,
    statewidth: usize,
) -> Result<(Vec<f32>, Vec<f32>), musaerror> {
    let outcount = d;
    let mut results = dispatch(
        cache,
        "scan",
        source::SCANENTRY,
        &[
            buf::host(input),
            buf::host(a),
            buf::host(b),
            buf::host(c),
            buf::host(skip),
            buf::host(state),
            buf::out(outcount),
            buf::out(statewidth.checked_mul(d).ok_or(musaerror::sizeoverflow)?),
        ],
        &[
            scalar::u32(1),
            scalar::u32(d as u32),
            scalar::u32(statewidth as u32),
        ],
        (d as u32, 1, 1),
        (source::ELEMENTWISETILE, 1, 1),
        0,
    )?;
    let out = results.remove(0);
    let newstate = results.remove(0);
    Ok((out, newstate))
}
