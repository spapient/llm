#![allow(unsafe_code)]

use super::gkernel::{self, buf, scalar};
pub(crate) use super::session;
use crate::catalog::sycl as source;
use crate::engine::gpuerror;
use crate::{GENINTO, GENINTOOP, GENOP, GENOPS};
use std::sync::Mutex;

// SAFETY: Every declaration mirrors the C ABI of `grid/sycl.c` (this
// crate's own bridge): the queue is an opaque pointer to a `sycl::queue`,
// USM helpers return raw `malloc_shared` pointers, and every `sycl_*`
// entry takes the queue first, then the kernel's operands in signature
// order, with the same scalar layout the generator uses. All calls happen
// only inside the single `unsafe` block of `drive`; C++ exceptions never
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

    /// Returns how many kernels this island has; SYCL resolves no kernels
    /// (they are linked symbols), so the count is always zero.
    #[must_use]
    pub(crate) fn kernelcount(&self) -> usize {
        0
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

/// Runs one kernel through the single driver-handoff block.
///
/// The block is lazy by design: the first call creates the queue, and the
/// entry is dispatched by name onto the linked `sycl_*` symbol. Shared
/// USM allocations are host-visible, so upload and download are plain
/// copies inside the same block; the C++ side reports failures through
/// `sycl_take_error`. Every allocation and the queue are freed by the
/// crate's cache lifecycle; per-dispatch buffers are released at the end
/// of this block.
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
    let mut state = cache.lock();
    let mut downloads: Vec<Vec<f32>> = Vec::new();
    // SAFETY: The queue is created lazily here and owned by `cache`;
    // every USM allocation is a live `sycl_malloc_shared` buffer whose
    // byte size matches its operand; the scalar storage is alive for the
    // whole dispatch; the kernel arguments are passed in signature order
    // (queue, operands, scalars); and the C++ entry reports failures
    // through `sycl_take_error` so no exception ever crosses the boundary.
    unsafe {
        if state.queue.is_null() {
            state.queue = sycl_queue_new(-1);
            if state.queue.is_null() {
                return Err(gpuerror::nodevice);
            }
        }
        let queue = state.queue;
        let mut ptrs: Vec<*mut core::ffi::c_void> = Vec::new();
        let mut outslots: Vec<(usize, usize)> = Vec::new();
        for (index, operand) in buffers.iter().enumerate() {
            let bytes = match operand {
                buf::host(data) => data.len() * 4,
                buf::hostu64(data) => data.len() * 8,
                buf::inout(data) => data.len() * 4,
                buf::out(len) => *len * 4,
            };
            let devptr = sycl_malloc_shared(queue, bytes);
            if devptr.is_null() {
                for p in ptrs {
                    sycl_free_shared(queue, p);
                }
                return Err(gpuerror::allocation);
            }
            ptrs.push(devptr);
            if matches!(operand, buf::out(_) | buf::inout(_)) {
                outslots.push((index, bytes / 4));
            }
        }
        for (index, operand) in buffers.iter().enumerate() {
            match operand {
                buf::host(data) => {
                    core::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        ptrs[index] as *mut f32,
                        data.len(),
                    );
                }
                buf::hostu64(data) => {
                    core::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        ptrs[index] as *mut u64,
                        data.len(),
                    );
                }
                buf::inout(data) => {
                    core::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        ptrs[index] as *mut f32,
                        data.len(),
                    );
                }
                buf::out(_) => {}
            }
        }
        let storages: Vec<u64> = scalars
            .iter()
            .map(|s| match s {
                scalar::u32(v) => *v as u64,
                scalar::u64(v) => *v,
                scalar::f32(v) => (*v).to_bits() as u64,
            })
            .collect();
        let s = |i: usize| storages[i];
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
                for p in ptrs {
                    sycl_free_shared(queue, p);
                }
                return Err(gpuerror::module {
                    kernel: entry.into(),
                    message: "unknown sycl entry".into(),
                });
            }
        }
        let reported = sycl_take_error();
        if reported != 0 {
            for p in ptrs {
                sycl_free_shared(queue, p);
            }
            return Err(gpuerror::launch {
                message: format!("sycl {entry}: error {reported}"),
            });
        }
        for (index, len) in outslots {
            let mut host: Vec<f32> = vec![0.0f32; len];
            core::ptr::copy_nonoverlapping(ptrs[index] as *const f32, host.as_mut_ptr(), len);
            downloads.push(host);
        }
        for p in ptrs {
            sycl_free_shared(queue, p);
        }
    }
    Ok(downloads)
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
GENINTO!(
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
    attention_cached
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
