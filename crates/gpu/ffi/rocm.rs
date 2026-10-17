#![allow(unsafe_code)]

use super::gkernel::{self, buf, launchspec, scalar};
pub(crate) use super::session;
use crate::catalog::rocm as source;
use crate::engine::gpuerror;
use crate::registry::KERNELS;
use crate::{GENINTO, GENINTOOP, GENOP, GENOPS};
use core::ffi::{c_int, c_longlong};
use std::collections::HashMap;
use std::sync::Mutex;

/// `hipError_t` (hipSuccess = 0).
type hipresult = i32;
/// `hipblasStatus_t` (HIPBLAS_STATUS_SUCCESS = 0).
type hipblasstatus = i32;
/// `hipblasOperation_t` (`HIPBLAS_OP_N` = 0, `HIPBLAS_OP_T` = 1).
type hipblasoperation = i32;
/// `hipblasHandle_t` is an opaque pointer-sized handle.
type hipblashandle = *mut core::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct hipdevice(pub(crate) i32);
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct hipmodule(*mut core::ffi::c_void);
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct hipfunction(*mut core::ffi::c_void);

// SAFETY: Every declaration mirrors the C ABI of `hip_runtime_api.h`
// (ROCm 5.x/6.x) and `hipblas.h`: the opaque handles are pointer-sized
// `repr(C)` structs, the `hipMemcpy` kinds are the documented integers,
// `hipLaunchKernel`'s `kernelParams` is a pointer to an array of pointers,
// one per kernel parameter, each pointing at a value whose type matches
// the kernel signature the caller mirrors, and the hipBLAS handle is a
// pointer-sized opaque type with the enum-like status/operation integers
// below. All calls happen only inside the `unsafe` blocks of `drive` and
// `matmul`, guarded by result-code checks, so no unvalidated driver call
// can escape this island.
unsafe extern "C" {
    /// Initialises the HIP driver (flags must be 0).
    fn hipInit(flags: u32) -> hipresult;
    /// Fills `device` with the `ordinal`-th device.
    fn hipDeviceGet(device: *mut hipdevice, ordinal: i32) -> hipresult;
    /// Allocates `size` device bytes into `ptr`.
    fn hipMalloc(ptr: *mut *mut core::ffi::c_void, size: usize) -> hipresult;
    /// Frees a device allocation.
    fn hipFree(ptr: *mut core::ffi::c_void) -> hipresult;
    /// Copies `count` bytes from `src` to `dst` under `kind`.
    fn hipMemcpy(
        dst: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        count: usize,
        kind: i32,
    ) -> hipresult;
    /// Blocks until the device completes all preceding work.
    fn hipDeviceSynchronize() -> hipresult;
    /// Loads the fatbin `image` into `module`.
    fn hipModuleLoadData(module: *mut hipmodule, image: *const core::ffi::c_void) -> hipresult;
    /// Resolves `name` in `module` into `function`.
    fn hipModuleGetFunction(
        function: *mut hipfunction,
        module: hipmodule,
        name: *const core::ffi::c_char,
    ) -> hipresult;
    /// Launches `function` on the grid/block geometry with `kernelParams`.
    #[allow(clippy::too_many_arguments)]
    fn hipLaunchKernel(
        function: hipfunction,
        gridx: u32,
        gridy: u32,
        gridz: u32,
        blockx: u32,
        blocky: u32,
        blockz: u32,
        shared: u32,
        stream: *mut core::ffi::c_void,
        kernelparams: *mut *mut core::ffi::c_void,
        extra: *mut *mut core::ffi::c_void,
    ) -> hipresult;
    /// Creates a hipBLAS handle bound to the default stream.
    fn hipblasCreate(handle: *mut hipblashandle) -> hipblasstatus;
    /// Destroys a hipBLAS handle.
    fn hipblasDestroy(handle: hipblashandle) -> hipblasstatus;
    /// Strided-batched `C = op(A) * op(B)` for `f32`, column-major.
    #[allow(clippy::too_many_arguments)]
    fn hipblasGemmStridedBatched(
        handle: hipblashandle,
        transa: hipblasoperation,
        transb: hipblasoperation,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: *const f32,
        a: *const core::ffi::c_void,
        lda: c_int,
        stride_a: c_longlong,
        b: *const core::ffi::c_void,
        ldb: c_int,
        stride_b: c_longlong,
        beta: *const f32,
        c: *mut core::ffi::c_void,
        ldc: c_int,
        stride_c: c_longlong,
        batch: c_int,
    ) -> hipblasstatus;
}

const HIPMEMCPYHOSTTODEVICE: i32 = 0;
const HIPMEMCPYDEVICETOHOST: i32 = 1;

/// The backend's device state, guarded by a single lock.
pub(crate) struct cache {
    state: Mutex<cachestate>,
}

struct cachestate {
    opened: bool,
    module: Option<hipmodule>,
    functions: HashMap<&'static str, hipfunction>,
    /// The hipBLAS handle created on first use and reused by every
    /// [`matmul`]; bound to the default stream like the kernel launches.
    blas: Option<hipblashandle>,
}

impl cache {
    /// Creates a cache with nothing opened; the driver, module, entries,
    /// and hipBLAS handle initialise lazily on the first dispatch.
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(cachestate {
                opened: false,
                module: None,
                functions: HashMap::new(),
                blas: None,
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

/// The `rmsnorm` scalar group: HIP mirrors the shared CUDA-family
/// signature (the row count travels in the grid), so only `eps` is a
/// scalar parameter.
fn rmsnormparams(_rows: usize, _width: usize, eps: f32) -> Vec<scalar> {
    vec![scalar::f32(eps)]
}

/// The `softmax` scalar group: HIP mirrors the shared CUDA-family
/// signature, so there is no scalar parameter.
fn softmaxparams(_rows: usize, _width: usize) -> Vec<scalar> {
    Vec::new()
}

/// Runs one kernel through the single driver-handoff block.
///
/// The block is lazy by design: the first call initialises the driver,
/// loads the `rocm/rocm.c` fatbin from the build-time registry, and
/// resolves `entry`; later calls reuse all of it. Host operands are
/// uploaded, the kernel is launched, the device synchronised, outputs are
/// downloaded, and this dispatch's temporary allocations are freed — all
/// inside the one `unsafe` block, so the registered handoff count stays
/// exactly one.
///
/// # Errors
///
/// Returns [`gpuerror::device`] when the driver cannot be initialised or
/// no device exists, [`gpuerror::module`] when the fatbin cannot be loaded
/// or an entry resolved, [`gpuerror::transfer`] when a copy fails,
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
    // SAFETY: The driver/module/entry state is owned by `cache` and
    // initialised lazily here, every allocation is a live `hipMalloc`
    // buffer whose byte size matches its operand, the scalar storage is
    // alive for the whole launch, the kernel-parameter pointer array is
    // rebuilt per dispatch in signature order (buffers then scalars), and
    // every driver call is guarded by a result-code check inside this
    // block — no unvalidated driver call escapes the island.
    unsafe {
        if !state.opened {
            if hipInit(0) != 0 {
                run = Err(gpuerror::device {
                    message: "hipInit".into(),
                });
            } else {
                let mut device = hipdevice(0);
                if hipDeviceGet(&mut device, 0) != 0 {
                    run = Err(gpuerror::nodevice);
                }
                state.opened = true;
            }
        }
        if run.is_ok() && state.module.is_none() {
            let Some((_, bytes)) = KERNELS.iter().find(|(id, _)| *id == "rocm") else {
                return Err(gpuerror::module {
                    kernel: entry.into(),
                    message: "rocm fatbin not in registry".into(),
                });
            };
            if run.is_ok() {
                let mut module = hipmodule(core::ptr::null_mut());
                if hipModuleLoadData(&mut module, bytes.as_ptr() as *const core::ffi::c_void) != 0 {
                    run = Err(gpuerror::module {
                        kernel: entry.into(),
                        message: "hipModuleLoadData".into(),
                    });
                }
                state.module = Some(module);
            }
        }
        let function = if run.is_ok() {
            match state.functions.get(entry) {
                Some(&f) => f,
                None => {
                    let mut function = hipfunction(core::ptr::null_mut());
                    let name = std::ffi::CString::new(entry)
                        .map_err(|_| gpuerror::module {
                            kernel: entry.into(),
                            message: "entry contains nul".into(),
                        })
                        .and_then(|n| {
                            if hipModuleGetFunction(
                                &mut function,
                                state.module.unwrap(),
                                n.as_ptr(),
                            ) != 0
                            {
                                Err(gpuerror::module {
                                    kernel: entry.into(),
                                    message: "hipModuleGetFunction".into(),
                                })
                            } else {
                                Ok(())
                            }
                        });
                    if let Err(e) = name {
                        run = Err(e);
                    }
                    state.functions.insert(entry, function);
                    function
                }
            }
        } else {
            hipfunction(core::ptr::null_mut())
        };
        if run.is_ok() {
            // Bind the buffers: upload host operands, allocate outputs.
            let mut params: Vec<*mut core::ffi::c_void> = Vec::new();
            let mut outslots: Vec<(usize, usize)> = Vec::new(); // (buffer index, len)
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
                if hipMalloc(&mut devptr, bytes) != 0 {
                    run = Err(gpuerror::allocation);
                    break;
                }
                allocations.push(devptr);
                if !ptr.is_null() && hipMemcpy(devptr, ptr, bytes, HIPMEMCPYHOSTTODEVICE) != 0 {
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
                // The kernel takes its buffer operands first and its
                // scalar parameters last; each scalar is stored in an
                // 8-byte slot whose address is handed to the launcher.
                let mut storages: Vec<u64> = scalars
                    .iter()
                    .map(|s| match s {
                        scalar::u32(v) => *v as u64,
                        scalar::u64(v) => *v,
                        scalar::f32(v) => (*v).to_bits() as u64,
                    })
                    .collect();
                let mut argptrs: Vec<*mut core::ffi::c_void> = storages
                    .iter_mut()
                    .map(|s| s as *mut u64 as *mut core::ffi::c_void)
                    .collect();
                let mut allptrs = params;
                allptrs.append(&mut argptrs);
                let (gx, gy, gz) = spec.grid;
                let (bx, by, bz) = spec.block;
                if hipLaunchKernel(
                    function,
                    gx,
                    gy,
                    gz,
                    bx,
                    by,
                    bz,
                    spec.shared,
                    core::ptr::null_mut(),
                    allptrs.as_mut_ptr(),
                    core::ptr::null_mut(),
                ) != 0
                {
                    run = Err(gpuerror::launch {
                        message: format!("hipLaunchKernel {entry}"),
                    });
                } else if hipDeviceSynchronize() != 0 {
                    run = Err(gpuerror::launch {
                        message: "hipDeviceSynchronize".into(),
                    });
                }
            }
            if run.is_ok() {
                for (index, len) in outslots {
                    let mut host: Vec<f32> = vec![0.0f32; len];
                    if hipMemcpy(
                        host.as_mut_ptr() as *mut core::ffi::c_void,
                        allocations[index],
                        len * 4,
                        HIPMEMCPYDEVICETOHOST,
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
            let _ = hipFree(ptr);
        }
    }
    run?;
    Ok(downloads)
}

/// Computes `[m, k] * [k, n]` per batch with hipBLAS.
///
/// This is the ROCm fast path. The operands are row-major, so the call
/// asks hipBLAS for the column-major transpose product `C^T = B^T * A^T`,
/// which yields exactly the row-major `C = A * B` the contract requires.
/// One handle is created lazily and reused, and every call allocates and
/// releases its own buffers by scope on the default stream, exactly like
/// the kernel dispatches — the hand-written `matmul` replaces the shared
/// FMA reference kernel, which is an order of magnitude slower on real
/// AMD hardware.
///
/// # Errors
///
/// Returns [`gpuerror`] when the device cannot be opened, a dimension does
/// not fit `c_int`, an allocation or transfer fails, or hipBLAS rejects
/// the configuration.
pub(crate) fn matmul(
    cache: &cache,
    lhs: &[f32],
    rhs: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, gpuerror> {
    if m == 0 || n == 0 || batch == 0 {
        return Ok(Vec::new());
    }
    if k == 0 {
        return Ok(vec![0.0_f32; m * n * batch]);
    }
    let mi = c_int::try_from(m).map_err(|_| gpuerror::sizeoverflow)?;
    let ni = c_int::try_from(n).map_err(|_| gpuerror::sizeoverflow)?;
    let ki = c_int::try_from(k).map_err(|_| gpuerror::sizeoverflow)?;
    let batchi = c_int::try_from(batch).map_err(|_| gpuerror::sizeoverflow)?;
    let total = m
        .checked_mul(n)
        .and_then(|value| value.checked_mul(batch))
        .ok_or(gpuerror::sizeoverflow)?;
    let mut state = cache.lock();
    let mut run: Result<(), gpuerror> = Ok(());
    let mut out = vec![0.0_f32; total];
    // SAFETY: The handle is owned by `cache` and initialised lazily here;
    // the three allocations are live `hipMalloc` buffers whose byte sizes
    // match their operands, the host copies are complete before the gemm
    // is submitted on the same (default) stream, the strides are one
    // slice's element count per operand so every batch slice stays inside
    // its allocation, and every driver call is guarded by a result-code
    // check inside this block — no unvalidated driver call escapes the
    // island.
    unsafe {
        if !state.opened {
            if hipInit(0) != 0 {
                run = Err(gpuerror::device {
                    message: "hipInit".into(),
                });
            } else {
                let mut device = hipdevice(0);
                if hipDeviceGet(&mut device, 0) != 0 {
                    run = Err(gpuerror::nodevice);
                }
                state.opened = true;
            }
        }
        if run.is_ok() && state.blas.is_none() {
            let mut created = core::ptr::null_mut();
            if hipblasCreate(&mut created) != 0 {
                run = Err(gpuerror::blas {
                    message: "hipblasCreate".into(),
                });
            } else {
                state.blas = Some(created);
            }
        }
        let handle = state.blas.unwrap_or(core::ptr::null_mut());
        let (mut da, mut db, mut dc) = (
            core::ptr::null_mut(),
            core::ptr::null_mut(),
            core::ptr::null_mut(),
        );
        if run.is_ok() {
            let la = m
                .checked_mul(k)
                .and_then(|value| value.checked_mul(4))
                .ok_or(gpuerror::sizeoverflow)?;
            if hipMalloc(&mut da, la) != 0
                || hipMemcpy(
                    da,
                    lhs.as_ptr() as *const core::ffi::c_void,
                    la,
                    HIPMEMCPYHOSTTODEVICE,
                ) != 0
            {
                run = Err(gpuerror::transfer {
                    message: "hipMalloc/hipMemcpy lhs".into(),
                });
            }
        }
        if run.is_ok() {
            let lb = k
                .checked_mul(n)
                .and_then(|value| value.checked_mul(4))
                .ok_or(gpuerror::sizeoverflow)?;
            if hipMalloc(&mut db, lb) != 0
                || hipMemcpy(
                    db,
                    rhs.as_ptr() as *const core::ffi::c_void,
                    lb,
                    HIPMEMCPYHOSTTODEVICE,
                ) != 0
            {
                run = Err(gpuerror::transfer {
                    message: "hipMalloc/hipMemcpy rhs".into(),
                });
            }
        }
        if run.is_ok()
            && hipMalloc(&mut dc, total.checked_mul(4).ok_or(gpuerror::sizeoverflow)?) != 0
        {
            run = Err(gpuerror::transfer {
                message: "hipMalloc out".into(),
            });
        }
        if run.is_ok() {
            // Column-major view: the row-major `[m, k]` lhs reads as a
            // `k x m` matrix (leading dimension k), the row-major
            // `[k, n]` rhs as an `n x k` one (leading dimension n), and
            // the product is the column-major `n x m` matrix whose
            // row-major reading is the `[m, n]` result. The hipBLAS
            // extents are therefore (m = n, n = m, k = k); the strides
            // are one row-major slice's element count per operand.
            if hipblasGemmStridedBatched(
                handle,
                0, // HIPBLAS_OP_N
                0, // HIPBLAS_OP_N
                ni,
                mi,
                ki,
                &1.0_f32,
                db,
                ni,
                (k * n) as c_longlong,
                da,
                ki,
                (m * k) as c_longlong,
                &0.0_f32,
                dc,
                ni,
                (m * n) as c_longlong,
                batchi,
            ) != 0
            {
                run = Err(gpuerror::blas {
                    message: format!("hipblasGemmStridedBatched m={m} k={k} n={n} batch={batch}"),
                });
            }
        }
        if run.is_ok() && hipDeviceSynchronize() != 0 {
            run = Err(gpuerror::device {
                message: "hipDeviceSynchronize".into(),
            });
        }
        if run.is_ok()
            && hipMemcpy(
                out.as_mut_ptr() as *mut core::ffi::c_void,
                dc,
                total * 4,
                HIPMEMCPYDEVICETOHOST,
            ) != 0
        {
            run = Err(gpuerror::transfer {
                message: "hipMemcpy out".into(),
            });
        }
        if !da.is_null() {
            let _ = hipFree(da);
        }
        if !db.is_null() {
            let _ = hipFree(db);
        }
        if !dc.is_null() {
            let _ = hipFree(dc);
        }
    }
    run?;
    Ok(out)
}

// The hipBLAS path serves `matmul`; the FMA reference kernel stays in the
// shared translation unit but is no longer dispatched from this island.
GENOPS!(
    add,
    mul,
    silu,
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
// sycl island
// ---------------------------------------------------------------------------

// The Intel oneAPI SYCL driver island.
//
// SYCL has no runtime name lookup, so `grid/sycl.c` exports the operator set
// as 15 plain-C `sycl_*` entry points plus queue and USM helpers, compiled by
// `icpx -fsycl` at build time and linked into the crate (no fatbin
// registry). The Rust side holds the opaque `sycl::queue*` (created through
// `sycl_queue_new`) and passes it to every launch; each launch is
// synchronous (`wait_and_throw`) and reports through `sycl_take_error`, so
// the island never lets a C++ exception cross the C ABI. All driver calls
// live in the single `unsafe` block of [`drive`]; [`GENOPS`] expands the
// operator surface over it. The registered site count is `sycl = 2`: the
// `unsafe extern "C"` declaration island and the driver-handoff block.
