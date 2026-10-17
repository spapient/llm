#![allow(unsafe_code)]

use super::gkernel::{self, buf, launchspec, scalar};
pub(crate) use super::session;
use crate::catalog::mlu as source;
use crate::engine::gpuerror;
use crate::{GENINTO, GENINTOOP, GENOP, GENOPS};
use std::collections::HashMap;
use std::sync::Mutex;

/// `cnrtRet_t` (CNRT_RET_SUCCESS = 0).
type cnrtret = i32;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct cnrtfunction(*mut core::ffi::c_void);

// SAFETY: Every declaration mirrors the C ABI of `cnrt.h` (NeuWare
// 3.x/4.x): the function handle is an opaque pointer-sized `repr(C)`
// struct, the `cnrtMemcpy` directions are the documented integers, and
// `cnrtLaunchKernel`'s `args` is a caller-owned buffer of the kernel's
// arguments packed 8-byte aligned in signature order. All calls happen
// only inside the single `unsafe` block of `drive`, guarded by
// result-code checks, so no unvalidated driver call can escape this
// island.
unsafe extern "C" {
    /// Initialises the CNRT runtime (flags must be 0).
    fn cnrtInit(flags: u32) -> cnrtret;
    /// Selects `dev` for the calling thread.
    fn cnrtSetDevice(dev: i32) -> cnrtret;
    /// Allocates `size` device bytes into `ptr`.
    fn cnrtMalloc(ptr: *mut *mut core::ffi::c_void, size: usize) -> cnrtret;
    /// Frees a device allocation.
    fn cnrtFree(ptr: *mut core::ffi::c_void) -> cnrtret;
    /// Copies `size` bytes from `src` to `dst` under `dir`.
    fn cnrtMemcpy(
        dst: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        size: usize,
        dir: u32,
    ) -> cnrtret;
    /// Blocks until the device completes all preceding work.
    fn cnrtSyncDevice() -> cnrtret;
    /// Resolves the linked kernel named `kernel_name`.
    fn cnrtGetKernel(kernel: *mut cnrtfunction, kernel_name: *const core::ffi::c_char) -> cnrtret;
    /// Launches `kernel` on the grid/block geometry with the packed `args`.
    #[allow(clippy::too_many_arguments)]
    fn cnrtLaunchKernel(
        kernel: cnrtfunction,
        grid_x: u32,
        grid_y: u32,
        grid_z: u32,
        block_x: u32,
        block_y: u32,
        block_z: u32,
        args: *mut core::ffi::c_void,
        args_size: usize,
        queue: *mut core::ffi::c_void,
    ) -> cnrtret;
}

const CNRTMEMDIRHOST2DEVICE: u32 = 0;
const CNRTMEMDIRDEVICE2HOST: u32 = 1;

/// The backend's device state, guarded by a single lock.
pub(crate) struct cache {
    state: Mutex<cachestate>,
}

struct cachestate {
    opened: bool,
    functions: HashMap<&'static str, cnrtfunction>,
}

impl cache {
    /// Creates a cache with nothing opened; the runtime and kernels
    /// initialise lazily on the first dispatch.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(cachestate {
                opened: false,
                functions: HashMap::new(),
            }),
        }
    }

    /// Returns how many kernels have been resolved so far.
    #[must_use]
    pub(crate) fn kernelcount(&self) -> usize {
        self.lock().functions.len()
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
        f.debug_struct("cache")
            .field("kernels", &self.lock().functions.len())
            .finish()
    }
}

/// The `rmsnorm` scalar group: the BANG kernel takes an explicit row
/// count because it partitions rows across cores inside the kernel.
fn rmsnormparams(rows: usize, width: usize, eps: f32) -> Vec<scalar> {
    vec![
        scalar::u32(rows as u32),
        scalar::u32(width as u32),
        scalar::f32(eps),
    ]
}

/// The `softmax` scalar group: the BANG kernel takes an explicit row
/// count.
fn softmaxparams(rows: usize, width: usize) -> Vec<scalar> {
    vec![scalar::u32(rows as u32), scalar::u32(width as u32)]
}

/// Runs one kernel through the single driver-handoff block.
///
/// The block is lazy by design: the first call initialises the runtime,
/// and each entry is resolved with `cnrtGetKernel` once. CNRT packs every
/// kernel argument (operands then scalars) into one 8-byte-aligned
/// buffer, which the block builds, launches on the grid/block geometry,
/// synchronises, downloads, and frees — all inside the one `unsafe`
/// block, so the registered handoff count stays exactly one.
///
/// # Errors
///
/// Returns [`gpuerror::device`] when the runtime cannot be initialised or
/// no device exists, [`gpuerror::module`] when an entry cannot be
/// resolved, [`gpuerror::transfer`] when a copy fails,
/// [`gpuerror::launch`] when the launch or synchronisation fails, and
/// [`gpuerror::allocation`] when a device allocation fails.
fn drive(
    cache: &cache,
    entry: &'static str,
    buffers: &[buf<'_>],
    scalars: &[scalar],
    spec: launchspec,
) -> Result<Vec<Vec<f32>>, gpuerror> {
    let mut state = cache.lock();
    let mut allocations: Vec<*mut core::ffi::c_void> = Vec::new();
    let mut downloads: Vec<Vec<f32>> = Vec::new();
    let mut run: Result<(), gpuerror> = Ok(());
    // SAFETY: The runtime/kernel state is owned by `cache` and
    // initialised lazily here, every allocation is a live `cnrtMalloc`
    // buffer whose byte size matches its operand, the packed argument
    // buffer mirrors the kernel signature (operands then scalars, 8-byte
    // aligned), and every driver call is guarded by a result-code check
    // inside this block — no unvalidated driver call escapes the island.
    unsafe {
        if !state.opened {
            if cnrtInit(0) != 0 {
                run = Err(gpuerror::device {
                    message: "cnrtInit".into(),
                });
            } else if cnrtSetDevice(0) != 0 {
                run = Err(gpuerror::nodevice);
            } else {
                state.opened = true;
            }
        }
        let function = if run.is_ok() {
            match state.functions.get(entry) {
                Some(&f) => f,
                None => {
                    let mut function = cnrtfunction(core::ptr::null_mut());
                    let name = std::ffi::CString::new(entry).map_err(|_| gpuerror::module {
                        kernel: entry.into(),
                        message: "entry contains nul".into(),
                    });
                    match name {
                        Ok(n) => {
                            if cnrtGetKernel(&mut function, n.as_ptr()) != 0 {
                                run = Err(gpuerror::module {
                                    kernel: entry.into(),
                                    message: "cnrtGetKernel".into(),
                                });
                            }
                        }
                        Err(e) => run = Err(e),
                    }
                    state.functions.insert(entry, function);
                    function
                }
            }
        } else {
            cnrtfunction(core::ptr::null_mut())
        };
        if run.is_ok() {
            let mut params: Vec<*mut core::ffi::c_void> = Vec::new();
            let mut outslots: Vec<(usize, usize)> = Vec::new();
            for (index, operand) in buffers.iter().enumerate() {
                let (ptr, bytes) = match operand {
                    buf::host(data) => (data.as_ptr() as *const core::ffi::c_void, data.len() * 4),
                    buf::hostu64(data) => {
                        (data.as_ptr() as *const core::ffi::c_void, data.len() * 8)
                    }
                    buf::inout(data) => (data.as_ptr() as *const core::ffi::c_void, data.len() * 4),
                    buf::out(len) => (core::ptr::null(), *len * 4),
                };
                let mut devptr: *mut core::ffi::c_void = core::ptr::null_mut();
                if cnrtMalloc(&mut devptr, bytes) != 0 {
                    run = Err(gpuerror::allocation);
                    break;
                }
                allocations.push(devptr);
                if !ptr.is_null() && cnrtMemcpy(devptr, ptr, bytes, CNRTMEMDIRHOST2DEVICE) != 0 {
                    run = Err(gpuerror::transfer {
                        message: format!("upload operand {index}"),
                    });
                    break;
                }
                params.push(devptr);
                if matches!(operand, buf::out(_) | buf::inout(_)) {
                    outslots.push((index, bytes / 4));
                }
            }
            if run.is_ok() {
                // CNRT packs kernel arguments into one 8-byte-aligned
                // buffer: the device-pointer values first, then the
                // scalar values, in signature order.
                let mut argpack: Vec<u8> = Vec::with_capacity((params.len() + scalars.len()) * 8);
                for p in &params {
                    argpack.extend_from_slice(&(*p as usize as u64).to_ne_bytes());
                }
                for s in scalars {
                    let bits = match s {
                        scalar::u32(v) => *v as u64,
                        scalar::u64(v) => *v,
                        scalar::f32(v) => (*v).to_bits() as u64,
                    };
                    argpack.extend_from_slice(&bits.to_ne_bytes());
                }
                let (gx, gy, gz) = spec.grid;
                let (bx, by, bz) = spec.block;
                if cnrtLaunchKernel(
                    function,
                    gx,
                    gy,
                    gz,
                    bx,
                    by,
                    bz,
                    argpack.as_mut_ptr() as *mut core::ffi::c_void,
                    argpack.len(),
                    core::ptr::null_mut(),
                ) != 0
                {
                    run = Err(gpuerror::launch {
                        message: format!("cnrtLaunchKernel {entry}"),
                    });
                } else if cnrtSyncDevice() != 0 {
                    run = Err(gpuerror::launch {
                        message: "cnrtSyncDevice".into(),
                    });
                }
            }
            if run.is_ok() {
                for (index, len) in outslots {
                    let mut host: Vec<f32> = vec![0.0f32; len];
                    if cnrtMemcpy(
                        host.as_mut_ptr() as *mut core::ffi::c_void,
                        allocations[index],
                        len * 4,
                        CNRTMEMDIRDEVICE2HOST,
                    ) != 0
                    {
                        run = Err(gpuerror::transfer {
                            message: format!("download operand {index}"),
                        });
                        break;
                    }
                    downloads.push(host);
                }
            }
        }
        for ptr in allocations {
            let _ = cnrtFree(ptr);
        }
    }
    run?;
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
