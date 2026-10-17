#![allow(unsafe_code)]
//! The MUSA driver boundary: the only file in this island that invokes
//! `mu*` FFI. Two registered sites — the `unsafe extern "C"` declaration
//! mirroring `musa.h`, and the single lazy driver-handoff block in
//! [`drive`] — each directly preceded by a `// SAFETY:` comment.
//!
//! The safe wrappers ([`dispatch`], [`dispatchmixed`], [`flush`],
//! [`retiredevice`]) funnel every call through [`drive`], so no third
//! handoff site exists.
use std::ffi::CString;

use super::*;
use crate::engine::gpuerror;
use crate::registry::KERNELS;

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
    /// Copies `count` bytes from device `src` to device `dst`.
    fn muMemcpyDtoD_v2(dst: mudevptr, src: mudevptr, count: usize) -> muresult;
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
// dispatch: the single driver-handoff block.
// ---------------------------------------------------------------------------

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
/// Returns [`gpuerror::device`] when the driver cannot be initialised or no
/// device exists, [`gpuerror::module`] when a fatbin cannot be loaded or an
/// entry resolved, [`gpuerror::transfer`] when a copy fails,
/// [`gpuerror::launch`] when the launch or synchronisation fails, and
/// [`gpuerror::sizeoverflow`] when a byte count overflows.
/// Runs one kernel through the single driver-handoff block, downloading
/// every [`buf::out`] slot. Device-resident variants are accepted but their
/// handed-back pointers are discarded: callers of a mixed run use
/// [`dispatchdevice`] instead.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch(
    cache: &cache,
    source_id: &'static str,
    entry: &'static str,
    buffers: &[buf<'_>],
    scalars: &[scalar],
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    sharedbytes: u32,
) -> Result<Vec<Vec<f32>>, gpuerror> {
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
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatchmixed(
    cache: &cache,
    source_id: &'static str,
    entry: &'static str,
    buffers: &[buf<'_>],
    scalars: &[scalar],
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    sharedbytes: u32,
) -> Result<dispatchresult, gpuerror> {
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
) -> Result<Vec<Vec<f32>>, gpuerror> {
    let slots: Vec<buf<'_>> = downloads
        .iter()
        .map(|&(ptr, bytes)| buf::download(ptr, bytes))
        .collect();
    Ok(drive(cache, None, &slots, &[], (1, 1, 1), (1, 1, 1), 0, true)?.host)
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
pub(super) fn drive(
    cache: &cache,
    kernel: Option<(&'static str, &'static str)>,
    buffers: &[buf<'_>],
    scalars: &[scalar],
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    sharedbytes: u32,
    sync: bool,
) -> Result<dispatchresult, gpuerror> {
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
    // checked against `MUSA_SUCCESS` and converted to `gpuerror`; no result
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
                return Err(gpuerror::device {
                    message: "muInit failed".into(),
                });
            }
            let mut device: mudevice = 0;
            if muDeviceGet(&mut device, 0) != MUSA_SUCCESS {
                return Err(gpuerror::device {
                    message: "muDeviceGet(0) failed: no MUSA device".into(),
                });
            }
            let mut context: mucontext = 0;
            if muCtxCreate_v2(&mut context, 0, device) != MUSA_SUCCESS {
                return Err(gpuerror::device {
                    message: "muCtxCreate failed".into(),
                });
            }
            state.opened = true;
            state.device = Some(device);
            state.context = Some(context);
        }
        let context = state.context.ok_or(gpuerror::device {
            message: "context not open".into(),
        })?;

        // --- deferred releases --------------------------------------------
        // Buffers released by `retiredevice` re-enter the pool (or are freed)
        // here, inside the one registered handoff block, before any new
        // allocation so a just-released slot can serve this very dispatch.
        let pending = std::mem::take(&mut state.pendingfree);
        for (mem, bytes) in pending {
            let bucket = state.pool.entry(bytes).or_default();
            if bucket.len() < POOLCAP {
                bucket.push(mem);
            } else if muMemFree_v2(mem) != MUSA_SUCCESS {
                return Err(gpuerror::transfer {
                    message: "muMemFree failed for released device buffer".into(),
                });
            }
        }

        // --- lazy module load and kernel resolution ------------------------
        if let Some((source_id, entry)) = kernel
            && !state.functions.contains_key(&(source_id, entry))
        {
            let fatbin = KERNELS
                .iter()
                .find(|(id, _)| *id == source_id)
                .map(|(_, bytes)| *bytes)
                .ok_or_else(|| gpuerror::module {
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
                    return Err(gpuerror::module {
                        kernel: entry.to_string(),
                        message: format!("muModuleLoadData failed for `{source_id}`"),
                    });
                }
                state.modules.insert(source_id, module);
                module
            };
            let name = CString::new(entry).map_err(|_| gpuerror::module {
                kernel: entry.to_string(),
                message: "entry name contains a NUL byte".into(),
            })?;
            let mut function: mufunction = 0;
            if muModuleGetFunction(&mut function, module, name.as_ptr()) != MUSA_SUCCESS {
                return Err(gpuerror::module {
                    kernel: entry.to_string(),
                    message: format!("muModuleGetFunction failed for `{entry}`"),
                });
            }
            state.functions.insert((source_id, entry), function);
        }
        let function = match kernel {
            Some((source_id, entry)) => Some(
                *state
                    .functions
                    .get(&(source_id, entry))
                    .ok_or_else(|| gpuerror::module {
                        kernel: entry.to_string(),
                        message: "kernel not resolved".into(),
                    })?,
            ),
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
        let mut allocate: Vec<bool> = Vec::with_capacity(buffers.len());
        for slot in buffers {
            match *slot {
                buf::host(data) | buf::hostdevice(data) => {
                    sizes.push(
                        data.len()
                            .checked_mul(core::mem::size_of::<f32>())
                            .ok_or(gpuerror::sizeoverflow)?,
                    );
                    keep.push(matches!(*slot, buf::hostdevice(_)));
                    allocate.push(true);
                    owned.push(0);
                }
                buf::hostu64(data) | buf::hostu64device(data) => {
                    sizes.push(
                        data.len()
                            .checked_mul(core::mem::size_of::<u64>())
                            .ok_or(gpuerror::sizeoverflow)?,
                    );
                    keep.push(matches!(*slot, buf::hostu64device(_)));
                    allocate.push(true);
                    owned.push(0);
                }
                buf::out(count) | buf::outdevice(count) => {
                    sizes.push(
                        count
                            .checked_mul(core::mem::size_of::<f32>())
                            .ok_or(gpuerror::sizeoverflow)?,
                    );
                    keep.push(matches!(*slot, buf::outdevice(_)));
                    allocate.push(true);
                    owned.push(0);
                }
                buf::device(mem) => {
                    sizes.push(0);
                    keep.push(true);
                    allocate.push(false);
                    owned.push(mem);
                }
                buf::download(mem, bytes) => {
                    sizes.push(bytes);
                    keep.push(true);
                    allocate.push(false);
                    owned.push(mem);
                }
                buf::copy(_, bytes) => {
                    sizes.push(bytes);
                    keep.push(true);
                    allocate.push(true);
                    owned.push(0);
                }
                buf::copyinto(_, _, bytes) => {
                    sizes.push(bytes);
                    keep.push(true);
                    allocate.push(false);
                    owned.push(0);
                }
            }
        }
        for (index, &size) in sizes.iter().enumerate() {
            if size == 0 || !allocate[index] {
                continue;
            }
            let mem = if let Some(pooled) = state.pool.get_mut(&size).and_then(Vec::pop) {
                pooled
            } else {
                let mut mem: mudevptr = 0;
                if muMemAlloc_v2(&mut mem, size) != MUSA_SUCCESS {
                    return Err(gpuerror::transfer {
                        message: format!("muMemAlloc failed for {size} bytes"),
                    });
                }
                mem
            };
            owned[index] = mem;
        }
        for (index, slot) in buffers.iter().enumerate() {
            if sizes[index] == 0 {
                continue;
            }
            match *slot {
                buf::host(data) | buf::hostdevice(data) => {
                    if muMemcpyHtoD_v2(owned[index], data.as_ptr().cast(), sizes[index])
                        != MUSA_SUCCESS
                    {
                        return Err(gpuerror::transfer {
                            message: "muMemcpyHtoD failed for f32 operand".into(),
                        });
                    }
                }
                buf::hostu64(data) | buf::hostu64device(data) => {
                    if muMemcpyHtoD_v2(owned[index], data.as_ptr().cast(), sizes[index])
                        != MUSA_SUCCESS
                    {
                        return Err(gpuerror::transfer {
                            message: "muMemcpyHtoD failed for u64 operand".into(),
                        });
                    }
                }
                buf::copy(src, _) => {
                    if muMemcpyDtoD_v2(owned[index], src, sizes[index]) != MUSA_SUCCESS {
                        return Err(gpuerror::transfer {
                            message: "muMemcpyDtoD failed for copy".into(),
                        });
                    }
                }
                buf::copyinto(dst, src, _) => {
                    if muMemcpyDtoD_v2(dst, src, sizes[index]) != MUSA_SUCCESS {
                        return Err(gpuerror::transfer {
                            message: "muMemcpyDtoD failed for copyinto".into(),
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
                scalar::u64(number) => rawargs.push(number as *mut u64 as *mut _),
                scalar::f32(number) => rawargs.push(number as *mut f32 as *mut _),
            }
        }
        if let Some(function) = function
            && muLaunchKernel(
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
            return Err(gpuerror::launch {
                message: "muLaunchKernel failed".into(),
            });
        }
        if sync && muCtxSynchronize(context) != MUSA_SUCCESS {
            return Err(gpuerror::launch {
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
                        .map_err(|_| gpuerror::sizeoverflow)?;
                    out.resize(count, 0.0_f32);
                    if count != 0
                        && muMemcpyDtoH_v2(out.as_mut_ptr().cast(), owned[index], sizes[index])
                            != MUSA_SUCCESS
                    {
                        return Err(gpuerror::transfer {
                            message: "muMemcpyDtoH failed for output".into(),
                        });
                    }
                    host.push(out);
                }
                buf::outdevice(_)
                | buf::hostdevice(_)
                | buf::hostu64device(_)
                | buf::copy(_, _) => {
                    device.push((owned[index], sizes[index]));
                }
                buf::device(_) | buf::copyinto(_, _, _) => {}
                buf::download(ptr, bytes) => {
                    let mut out = Vec::new();
                    out.try_reserve_exact(bytes / core::mem::size_of::<f32>())
                        .map_err(|_| gpuerror::sizeoverflow)?;
                    out.resize(bytes / core::mem::size_of::<f32>(), 0.0_f32);
                    if bytes != 0
                        && muMemcpyDtoH_v2(out.as_mut_ptr().cast(), ptr, bytes) != MUSA_SUCCESS
                    {
                        return Err(gpuerror::transfer {
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
                return Err(gpuerror::transfer {
                    message: "muMemFree failed for excess pooled buffer".into(),
                });
            }
        }
        Ok(dispatchresult { host, device })
    }
}
