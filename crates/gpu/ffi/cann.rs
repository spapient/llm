#![allow(unsafe_code)]

use super::gkernel::{self, buf, launchspec, scalar};
pub(crate) use super::session;
use crate::catalog::cann as source;
use crate::engine::gpuerror;
use crate::registry::KERNELS;
use crate::{GENINTO, GENINTOOP, GENOP, GENOPS};
use std::collections::HashMap;
use std::sync::Mutex;

/// `aclError` (ACL_SUCCESS = 0).
type aclerror = i32;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct aclrtcontext(*mut core::ffi::c_void);
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct aclrtkernel(*mut core::ffi::c_void);

// SAFETY: Every declaration mirrors the C ABI of `acl/acl_rt.h` (CANN
// 8.x): the contexts and kernels are opaque pointer-sized `repr(C)`
// structs, the `aclrtMemcpy` kinds are the documented integers, and
// `aclrtLaunchKernel`'s `args` is a caller-owned buffer of the kernel's
// arguments packed 8-byte aligned in signature order. All calls happen
// only inside the single `unsafe` block of `drive`, guarded by
// result-code checks, so no unvalidated driver call can escape this
// island.
unsafe extern "C" {
    /// Initialises the CANN runtime; `config_path` may be null.
    fn aclInit(config_path: *const core::ffi::c_char) -> aclerror;
    /// Selects `device_id` for the calling thread.
    fn aclrtSetDevice(device_id: i32) -> aclerror;
    /// Creates a context on `device_id`.
    fn aclrtCreateContext(context: *mut aclrtcontext, device_id: i32) -> aclerror;
    /// Allocates `size` device bytes into `dev_ptr` under `policy`.
    fn aclrtMalloc(dev_ptr: *mut *mut core::ffi::c_void, size: usize, policy: u32) -> aclerror;
    /// Frees a device allocation.
    fn aclrtFree(dev_ptr: *mut core::ffi::c_void) -> aclerror;
    /// Copies `count` bytes from `src` to `dst` under `kind`.
    #[allow(clippy::too_many_arguments)]
    fn aclrtMemcpy(
        dst: *mut core::ffi::c_void,
        dst_max: usize,
        src: *const core::ffi::c_void,
        count: usize,
        kind: u32,
    ) -> aclerror;
    /// Blocks until the device completes all preceding work.
    fn aclrtSynchronizeDevice() -> aclerror;
    /// Creates a kernel from the compiled binary `image`.
    fn aclrtCreateKernel(
        kernel: *mut aclrtkernel,
        image: *const core::ffi::c_void,
        image_size: usize,
        kernel_name: *const core::ffi::c_char,
    ) -> aclerror;
    /// Launches `kernel` with `block_dim` tasks and the packed `args`.
    fn aclrtLaunchKernel(
        kernel: aclrtkernel,
        block_dim: u32,
        args: *mut core::ffi::c_void,
        args_size: u32,
        stream: *mut core::ffi::c_void,
    ) -> aclerror;
}

const ACLMEMCPYHOSTTODEVICE: u32 = 1;
const ACLMEMCPYDEVICETOHOST: u32 = 2;

/// The backend's device state, guarded by a single lock.
pub(crate) struct cache {
    state: Mutex<cachestate>,
}

struct cachestate {
    opened: bool,
    kernels: HashMap<&'static str, aclrtkernel>,
}

impl cache {
    /// Creates a cache with nothing opened; the runtime and kernels
    /// initialise lazily on the first dispatch.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(cachestate {
                opened: false,
                kernels: HashMap::new(),
            }),
        }
    }

    /// Returns how many kernels have been created so far.
    #[must_use]
    pub(crate) fn kernelcount(&self) -> usize {
        self.lock().kernels.len()
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
            .field("kernels", &self.lock().kernels.len())
            .finish()
    }
}

/// The `rmsnorm` scalar group: the Ascend C kernel takes an explicit row
/// count because it partitions rows across AI cores inside the kernel.
fn rmsnormparams(rows: usize, width: usize, eps: f32) -> Vec<scalar> {
    vec![
        scalar::u32(rows as u32),
        scalar::u32(width as u32),
        scalar::f32(eps),
    ]
}

/// The `softmax` scalar group: the Ascend C kernel takes an explicit row
/// count.
fn softmaxparams(rows: usize, width: usize) -> Vec<scalar> {
    vec![scalar::u32(rows as u32), scalar::u32(width as u32)]
}

/// Runs one kernel through the single driver-handoff block.
///
/// The block is lazy by design: the first call initialises the runtime,
/// and each entry is turned into a kernel with `aclrtCreateKernel` from
/// the `grid/cann.c` fatbin once. CANN packs every kernel argument
/// (operands then scalars) into one 8-byte-aligned buffer, which the
/// block builds, launches with the row grid's task count, synchronises,
/// downloads, and frees — all inside the one `unsafe` block, so the
/// registered handoff count stays exactly one.
///
/// # Errors
///
/// Returns [`gpuerror::device`] when the runtime cannot be initialised or
/// no device exists, [`gpuerror::module`] when the fatbin cannot be
/// loaded or an entry created, [`gpuerror::transfer`] when a copy fails,
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
    // initialised lazily here, every allocation is a live `aclrtMalloc`
    // buffer whose byte size matches its operand, the packed argument
    // buffer mirrors the kernel signature (operands then scalars, 8-byte
    // aligned), and every driver call is guarded by a result-code check
    // inside this block — no unvalidated driver call escapes the island.
    unsafe {
        if !state.opened {
            if aclInit(core::ptr::null()) != 0 {
                run = Err(gpuerror::device {
                    message: "aclInit".into(),
                });
            } else if aclrtSetDevice(0) != 0 {
                run = Err(gpuerror::nodevice);
            } else {
                let mut context = aclrtcontext(core::ptr::null_mut());
                if aclrtCreateContext(&mut context, 0) != 0 {
                    run = Err(gpuerror::device {
                        message: "aclrtCreateContext".into(),
                    });
                }
                state.opened = true;
            }
        }
        let kernel = if run.is_ok() {
            match state.kernels.get(entry) {
                Some(&k) => k,
                None => {
                    let Some((_, bytes)) = KERNELS.iter().find(|(id, _)| *id == "cann") else {
                        return Err(gpuerror::module {
                            kernel: entry.into(),
                            message: "cann fatbin not in registry".into(),
                        });
                    };
                    let mut kernel = aclrtkernel(core::ptr::null_mut());
                    let name = std::ffi::CString::new(entry).unwrap_or_else(|_| {
                        run = Err(gpuerror::module {
                            kernel: entry.into(),
                            message: "entry contains nul".into(),
                        });
                        std::ffi::CString::new("").unwrap()
                    });
                    if run.is_ok()
                        && aclrtCreateKernel(
                            &mut kernel,
                            bytes.as_ptr() as *const core::ffi::c_void,
                            bytes.len(),
                            name.as_ptr(),
                        ) != 0
                    {
                        run = Err(gpuerror::module {
                            kernel: entry.into(),
                            message: "aclrtCreateKernel".into(),
                        });
                    }
                    state.kernels.insert(entry, kernel);
                    kernel
                }
            }
        } else {
            aclrtkernel(core::ptr::null_mut())
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
                if aclrtMalloc(&mut devptr, bytes, 0) != 0 {
                    run = Err(gpuerror::allocation);
                    break;
                }
                allocations.push(devptr);
                if !ptr.is_null()
                    && aclrtMemcpy(devptr, bytes, ptr, bytes, ACLMEMCPYHOSTTODEVICE) != 0
                {
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
                // CANN packs kernel arguments into one 8-byte-aligned
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
                if aclrtLaunchKernel(
                    kernel,
                    spec.block.0,
                    argpack.as_mut_ptr() as *mut core::ffi::c_void,
                    argpack.len() as u32,
                    core::ptr::null_mut(),
                ) != 0
                {
                    run = Err(gpuerror::launch {
                        message: format!("aclrtLaunchKernel {entry}"),
                    });
                } else if aclrtSynchronizeDevice() != 0 {
                    run = Err(gpuerror::launch {
                        message: "aclrtSynchronizeDevice".into(),
                    });
                }
            }
            if run.is_ok() {
                for (index, len) in outslots {
                    let mut host: Vec<f32> = vec![0.0f32; len];
                    if aclrtMemcpy(
                        host.as_mut_ptr() as *mut core::ffi::c_void,
                        len * 4,
                        allocations[index],
                        len * 4,
                        ACLMEMCPYDEVICETOHOST,
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
            let _ = aclrtFree(ptr);
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

// ---------------------------------------------------------------------------
// mlu island
// ---------------------------------------------------------------------------

// The Cambricon MLU driver island (Siyuan processors).
//
// The CNRT runtime is initialised lazily by the first dispatch; the
// `grid/mlu.c` kernels (compiled by `cncc` at build time and linked into the
// crate, so they are ordinary symbols) are resolved once per entry with
// `cnrtGetKernel` and cached. CNRT launches also pack all kernel arguments
// (operands then scalars) into one 8-byte-aligned buffer, built inside the
// single `unsafe` block of [`drive`]; [`GENOPS`] expands the operator
// surface over it. The registered site count is `mlu = 2`: the `unsafe
// extern "C"` declaration island and the driver-handoff block.
//
// The exact launch ABI varies across NeuWare releases; the declarations
// below follow the classic `cnrtLaunchKernel` (function, grid, block, args,
// argsSize, queue) and are validated on a real toolkit host.
