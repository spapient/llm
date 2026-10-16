#![allow(unsafe_code)]
//! The single platform-FFI island of the CUDA backend.
//!
//! Authority: docs/design.md §21.8. This is the only file in the crate that
//! declares `#![allow(unsafe_code)]`; the crate root keeps
//! `#![deny(unsafe_code)]`. Exactly three constructs are `unsafe` — one per
//! **class of platform interaction**, not one per operator — and each is
//! directly preceded by a `// SAFETY:` comment. `scripts/checkunsafe.sh`
//! enforces both the count and the comments.
//!
//! The three classes are:
//!
//! * [`submit`] — the NVRTC kernel launch. Every in-repo kernel reaches the
//!   driver through it, so the count does not grow with the kernel count.
//! * [`matmul_cublas`] — the cuBLAS `gemm` call.
//! * [`softmax`] — the cuDNN softmax call.
//!
//! The safe functions expose a narrow, run-to-completion interface: upload host
//! slices, run one kernel or one library call, download the result. Every stream
//! and library handle is created locally and released by scope, so no execution
//! state outlives a call. The only state that does survive is [`cache`] — the
//! retained device context and the compiled kernels — which the executor owns
//! and which holds no request-scoped value (docs/design.md §21.6).
//!
//! The launch helpers take the validated plans produced by [`api::geometry`]
//! rather than loose extents, so no number is re-derived after it was checked.

use core::ffi::c_int;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, sys::cublasOperation_t};
use cudarc::cudnn::{
    Cudnn, CudnnError, SoftmaxForward,
    sys::{cudnnSoftmaxAlgorithm_t, cudnnSoftmaxMode_t},
};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchArgs, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use crate::error::cudaerror;
use crate::source;
use api::geometry::{attentionshape, ropeshape};

/// The device state and compiled kernels one executor owns.
///
/// Retaining device 0's primary context costs about 1.5 ms and compiling one
/// kernel with NVRTC about 10.8 ms, both per call, before this cache existed
/// (docs/design.md §21.10.3): `gpu/embedding` measured 15.9 ms against 26 µs for
/// the CPU — a fixed cost independent of the work. The cache is owned by the
/// executor and shared by its clones, deliberately *not* kept in a process-wide
/// singleton: two independently constructed executors retain and compile
/// independently, and nothing here outlives the executor that owns it.
///
/// What the cache holds is device state and compiled code, never an operand, a
/// result, or any other request-scoped value, so sharing it across requests
/// cannot pass data between them. The execution stream is deliberately *not*
/// cached: one stream shared by concurrent calls would serialize their GPU work
/// (the same reason `metal` does not cache its command queue, docs/design.md
/// §21.6), so every call takes a fresh stream from the cached context and
/// releases it when it returns.
pub(crate) struct cache {
    state: Mutex<cachestate>,
}

/// The mutable half of [`cache`], guarded by a single lock that is never
/// acquired while another is held (the crate has exactly one).
#[derive(Default)]
struct cachestate {
    /// Device 0's retained primary context, opened on first use.
    ///
    /// `None` means "not opened yet or the last attempt failed", so an
    /// unavailable device is retried on the next call rather than latched.
    device: Option<Arc<CudaContext>>,
    /// One resolved kernel per `(source, entry)` pair.
    ///
    /// The key carries the kernel **source text**, so editing a kernel in
    /// [`crate::source`] can never reuse a stale module. The map is
    /// device-specific by construction: it lives beside the one device whose
    /// module it was resolved from, so no device belongs in the key. Every
    /// `CudaFunction` keeps its `CudaModule` — and through it the context —
    /// alive, so a cached kernel cannot outlive the device it was compiled for.
    functions: HashMap<(&'static str, &'static str), CudaFunction>,
}

impl cache {
    /// Creates an empty cache.
    ///
    /// Retaining the device is deferred to the first operator call, so
    /// construction cannot fail and the error still surfaces per call.
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(cachestate::default()),
        }
    }

    /// Returns how many kernels the cache currently holds.
    ///
    /// Exposed so that the executor can report it and the contract suite can
    /// assert that a second call reuses the compiled kernel instead of
    /// compiling it again.
    pub(crate) fn kernelcount(&self) -> usize {
        self.lock().functions.len()
    }

    /// Takes a fresh execution stream on the cached device.
    ///
    /// The stream carries its own reference to the context, so it stays valid
    /// for the whole call even though the caller holds no device handle.
    fn stream(&self) -> Result<Arc<CudaStream>, cudaerror> {
        let device = self.device()?;
        device.new_stream().map_err(|error| cudaerror::device {
            message: error.to_string(),
        })
    }

    /// Returns the retained device, opening it once.
    fn device(&self) -> Result<Arc<CudaContext>, cudaerror> {
        self.lock().device()
    }

    /// Returns the resolved kernel for `(text, entry)`, compiling it once.
    ///
    /// The lock is held across a miss, its NVRTC compilation, and the module
    /// load. That serializes concurrent first calls for the same kernel instead
    /// of compiling it several times over, and it is released before the caller
    /// moves any operand: no transfer, launch, or synchronization ever happens
    /// under the lock.
    fn function(&self, text: &'static str, entry: &'static str) -> Result<CudaFunction, cudaerror> {
        let mut state = self.lock();
        if let Some(compiled) = state.functions.get(&(text, entry)) {
            return Ok(compiled.clone());
        }
        let device = state.device()?;
        let function = compile(&device, text, entry)?;
        state.functions.insert((text, entry), function.clone());
        Ok(function)
    }

    /// Acquires the lock, recovering the contents if an earlier holder panicked.
    ///
    /// The map holds no invariant a panic could break — the worst case is a
    /// partially filled cache, which is still a valid one — so recovering is
    /// strictly better than turning a panic into a permanently unusable
    /// backend. This is also why no call site uses `unwrap`.
    fn lock(&self) -> std::sync::MutexGuard<'_, cachestate> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

impl cachestate {
    /// Returns the cached device, retaining it if absent.
    fn device(&mut self) -> Result<Arc<CudaContext>, cudaerror> {
        if let Some(device) = &self.device {
            return Ok(device.clone());
        }
        let opened = CudaContext::new(0).map_err(|error| cudaerror::device {
            message: error.to_string(),
        })?;
        self.device = Some(opened.clone());
        Ok(opened)
    }
}

impl core::fmt::Debug for cache {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // The count is the only property of the cache worth printing, and it
        // needs the lock; the platform handles themselves have no useful
        // rendering and are not exposed.
        f.debug_struct("cache")
            .field("functions", &self.lock().functions.len())
            .finish()
    }
}

/// Compiles `text` with NVRTC, loads the module, and resolves `entry`.
fn compile(device: &Arc<CudaContext>, text: &str, entry: &str) -> Result<CudaFunction, cudaerror> {
    let ptx = compile_ptx(text).map_err(|error| cudaerror::compile {
        kernel: entry.to_string(),
        message: error.to_string(),
    })?;
    let module = device.load_module(ptx).map_err(|error| cudaerror::module {
        kernel: entry.to_string(),
        message: error.to_string(),
    })?;
    module
        .load_function(entry)
        .map_err(|error| cudaerror::module {
            kernel: entry.to_string(),
            message: error.to_string(),
        })
}

/// Converts a validated extent into the `c_int` a host-side library takes.
fn asint(value: u32) -> Result<c_int, cudaerror> {
    c_int::try_from(value).map_err(|_| cudaerror::sizeoverflow)
}

/// Converts a validated extent into the `unsigned` a kernel parameter takes.
fn asunsigned(value: usize) -> Result<u32, cudaerror> {
    u32::try_from(value).map_err(|_| cudaerror::sizeoverflow)
}

/// Builds the one-dimensional launch configuration covering `count` elements.
fn flatconfig(count: usize) -> Result<LaunchConfig, cudaerror> {
    let tile = source::ELEMENTWISETILE;
    let grid = count.div_ceil(tile as usize);
    Ok(LaunchConfig {
        grid_dim: (asunsigned(grid)?, 1, 1),
        block_dim: (tile, 1, 1),
        shared_mem_bytes: 0,
    })
}

/// Copies a host slice to a fresh device allocation.
fn upload<T: DeviceRepr>(stream: &Arc<CudaStream>, src: &[T]) -> Result<CudaSlice<T>, cudaerror> {
    stream.clone_htod(src).map_err(|error| cudaerror::transfer {
        message: error.to_string(),
    })
}

/// Allocates `len` zeroed device elements.
fn alloc(stream: &Arc<CudaStream>, len: usize) -> Result<CudaSlice<f32>, cudaerror> {
    stream
        .alloc_zeros::<f32>(len)
        .map_err(|error| cudaerror::transfer {
            message: error.to_string(),
        })
}

/// Waits for `stream` and copies a device allocation back to the host.
fn syncanddownload(stream: &Arc<CudaStream>, src: &CudaSlice<f32>) -> Result<Vec<f32>, cudaerror> {
    stream.synchronize().map_err(|error| cudaerror::transfer {
        message: error.to_string(),
    })?;
    stream.clone_dtoh(src).map_err(|error| cudaerror::transfer {
        message: error.to_string(),
    })
}

/// Maps a cuDNN status into the crate's own error type.
fn cudnnfail(error: CudnnError) -> cudaerror {
    cudaerror::cudnn {
        message: error.to_string(),
    }
}

/// Submits a prepared kernel launch to the driver.
///
/// This is the crate's single kernel-launch site, and it is a function rather
/// than an inline block so the registered count does not grow with the number
/// of kernels: callers assemble the argument list with the safe
/// `LaunchArgs::arg` builder and only the submit itself is unchecked.
fn submit(args: &mut LaunchArgs<'_>, config: LaunchConfig) -> Result<(), cudaerror> {
    // SAFETY: `args` was built by `CudaStream::launch_builder` from a function
    // resolved out of a module compiled from the very source whose signature
    // the caller's argument list mirrors, and every argument is either a live
    // device allocation or a scalar borrowed from a local that outlives this
    // call. `config` is the caller's launch geometry: for the flat kernels it
    // covers at least the element count (the kernels bounds-test themselves),
    // and for the tiled kernels it matches the shared-memory demand the kernel
    // computes for itself.
    unsafe { args.launch(config) }
        .map(|_| ())
        .map_err(|error| cudaerror::launch {
            message: error.to_string(),
        })
}

/// Runs one cuBLAS `gemm`, the crate's single cuBLAS site.
///
/// This is a function rather than an inline block so the registered count stays
/// one: the single-operator [`matmul_cublas`] and the accumulative submission
/// path both call it, and neither adds an `unsafe` site of its own.
fn gemm(
    blas: &CudaBlas,
    config: GemmConfig<f32>,
    db: &CudaSlice<f32>,
    da: &CudaSlice<f32>,
    dc: &mut CudaSlice<f32>,
) -> Result<(), cudaerror> {
    // SAFETY: `da`, `db`, and `dc` are live device allocations. `da` holds the
    // `[m, k]` left operand and `db` the `[k, n]` right operand; read as
    // column-major they are the `k x m` and `n x k` matrices the `lda`/`ldb`
    // leading dimensions describe, and `dc` is written as an `n x m`
    // column-major matrix with leading dimension `ldc`, i.e. the row-major
    // `[m, n]` result. Every dimension is at least 1 and fits `c_int`, so no
    // out-of-bounds access is possible.
    unsafe {
        blas.gemm(config, db, da, dc)
            .map_err(|error| cudaerror::blas {
                message: error.to_string(),
            })?;
    }
    Ok(())
}

/// Computes `[m, k] * [k, n]` with the in-repo NVRTC kernel.
///
/// This is the independent reference path: the kernel is compiled from
/// [`MATMULCUDA`](crate::source::MATMULCUDA) at run time, so it shares no code
/// with cuBLAS.
///
/// # Errors
///
/// Returns [`cudaerror`] when the device cannot be opened, the source fails to
/// compile, the kernel cannot be resolved, a transfer fails, a dimension does
/// not fit its device type, or the launch fails.
pub(crate) fn matmul_nvrtc(
    cache: &cache,
    lhs: &[f32],
    rhs: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, cudaerror> {
    if m == 0 || n == 0 {
        return Ok(Vec::new());
    }
    if k == 0 {
        return Ok(vec![0.0_f32; m * n]);
    }
    let dims = devicedims::new(m, k, n)?;
    let stream = cache.stream()?;
    let function = cache.function(source::MATMULCUDA, source::MATMULENTRY)?;
    let da = upload(&stream, lhs)?;
    let db = upload(&stream, rhs)?;
    let mut dc = alloc(&stream, m * n)?;

    let mut args = stream.launch_builder(&function);
    args.arg(&da)
        .arg(&db)
        .arg(&mut dc)
        .arg(&dims.m)
        .arg(&dims.k)
        .arg(&dims.n);
    submit(&mut args, launchconfig(&dims))?;
    syncanddownload(&stream, &dc)
}

/// Computes `[m, k] * [k, n]` with cuBLAS.
///
/// This is the CUDA-specific fast path. The operands are row-major, so the call
/// asks cuBLAS to compute `C^T = B^T * A^T` in its native column-major layout,
/// which produces exactly the row-major `C = A * B` the contract requires.
///
/// # Errors
///
/// Returns [`cudaerror`] when the device cannot be opened, a transfer fails, a
/// dimension does not fit `c_int`, or the cuBLAS call fails.
pub(crate) fn matmul_cublas(
    cache: &cache,
    lhs: &[f32],
    rhs: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, cudaerror> {
    if m == 0 || n == 0 {
        return Ok(Vec::new());
    }
    if k == 0 {
        return Ok(vec![0.0_f32; m * n]);
    }
    let dims = devicedims::new(m, k, n)?;
    let stream = cache.stream()?;
    let blas = CudaBlas::new(stream.clone()).map_err(|error| cudaerror::blas {
        message: error.to_string(),
    })?;

    let da = upload(&stream, lhs)?;
    let db = upload(&stream, rhs)?;
    let mut dc = alloc(&stream, m * n)?;

    // cuBLAS is column-major, so the operands are handed over swapped: the
    // row-major `[m, k]` left operand reads as a column-major `k x m` matrix,
    // and the row-major `[k, n]` right operand as a column-major `n x k` one.
    // Their product is the column-major `n x m` matrix whose row-major reading
    // is the `[m, n]` result the caller asked for. `m` and `n` below are
    // therefore the cuBLAS extents, not the contract's.
    let m_int = asint(dims.n)?;
    let n_int = asint(dims.m)?;
    let k_int = asint(dims.k)?;
    let config = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: m_int,
        n: n_int,
        k: k_int,
        alpha: 1.0_f32,
        lda: m_int,
        ldb: k_int,
        beta: 0.0_f32,
        ldc: m_int,
    };

    gemm(&blas, config, &db, &da, &mut dc)?;
    syncanddownload(&stream, &dc)
}

/// Launches one cuDNN softmax, the crate's single cuDNN site.
///
/// This is a function rather than an inline block so the registered count stays
/// one: the single-operator [`softmax`] and the accumulative submission path
/// both call it, and neither adds an `unsafe` site of its own.
fn softmax_forward(
    forward: &SoftmaxForward<'_, f32, f32, f32>,
    dx: &CudaSlice<f32>,
    dy: &mut CudaSlice<f32>,
) -> Result<(), cudaerror> {
    // SAFETY: `dx` and `dy` are live device allocations, each holding the
    // `total` contiguous `f32` the descriptors name, and `forward`'s
    // `xdesc`/`ydesc` describe exactly that buffer shape with the strides it
    // already has, so cuDNN reads and writes within the allocations.
    // `CUDNN_SOFTMAX_ACCURATE` is the max-subtracting formulation, which is the
    // numerically stable variant the scalar contract requires. The scaling pair
    // is the neutral `(1, 0)`, so it changes no operand.
    unsafe {
        forward
            .launch(
                (1.0_f32, 0.0_f32),
                cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE,
                dx,
                dy,
            )
            .map_err(cudnnfail)?;
    }
    Ok(())
}

/// Numerically stable softmax over the last axis with cuDNN.
///
/// `input` is read as `[rows, width]`. cuDNN descriptors are `[N, C, H, W]`, so
/// the row axis becomes `N` and the normalised axis becomes `C`; with
/// `CUDNN_SOFTMAX_MODE_CHANNEL` cuDNN then normalises over `C` for every `N`,
/// which is exactly softmax over the last axis of each row. The strides are the
/// standard contiguous NCHW strides of that shape, so the descriptor describes
/// the buffer as it already lies in memory.
///
/// cuDNN computes the same max-subtracting formulation, so it agrees with the
/// scalar reference on finite rows. It does **not** agree outside that domain:
/// measured on an RTX 2080 Ti (E1), a row of negative infinity comes back as
/// zeros under `CUDNN_SOFTMAX_ACCURATE`, whereas `cpu` and `metal` propagate
/// NaN. The contract is device-independent, so an input containing any
/// non-finite value is computed by [`scalarsoftmax`] — the same formulation the
/// other two backends implement — instead of being delegated to cuDNN.
///
/// # Errors
///
/// Returns [`cudaerror`] when the device cannot be opened, cuDNN cannot be
/// loaded or set up, a transfer fails, a dimension does not fit `c_int`, or the
/// softmax call fails.
pub(crate) fn softmax(
    cache: &cache,
    input: &[f32],
    rows: usize,
    width: usize,
) -> Result<Vec<f32>, cudaerror> {
    if rows == 0 {
        return Ok(Vec::new());
    }
    let total = rows.checked_mul(width).ok_or(cudaerror::sizeoverflow)?;
    if input.iter().any(|value| !value.is_finite()) {
        return Ok(scalarsoftmax(input, width));
    }
    let stream = cache.stream()?;
    let handle = Cudnn::new(stream.clone()).map_err(cudnnfail)?;
    let operation = handle
        .create_softmax::<f32>(cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_CHANNEL)
        .map_err(cudnnfail)?;
    let dims = [asint(asunsigned(rows)?)?, asint(asunsigned(width)?)?, 1, 1];
    let strides = [asint(asunsigned(width)?)?, 1, 1, 1];
    let xdesc = handle
        .create_4d_tensor_ex::<f32>(dims, strides)
        .map_err(cudnnfail)?;
    let ydesc = handle
        .create_4d_tensor_ex::<f32>(dims, strides)
        .map_err(cudnnfail)?;

    let dx = upload(&stream, input)?;
    let mut dy = alloc(&stream, total)?;
    let forward = SoftmaxForward {
        softmax: &operation,
        x: &xdesc,
        y: &ydesc,
    };

    softmax_forward(&forward, &dx, &mut dy)?;
    syncanddownload(&stream, &dy)
}

/// The scalar max-subtracting softmax `cpu` and `metal` implement.
///
/// Private to this file: it exists so the cuDNN path can step aside for inputs
/// outside the finite domain, not as a second public softmax. The row rule is
/// the one `crates/cpu/src/softmax.rs` documents — subtract the row maximum,
/// exponentiate, and divide only when the shifted exponents sum above zero, so a
/// row of negative infinity propagates NaN instead of being masked to zeros.
fn scalarsoftmax(input: &[f32], width: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; input.len()];
    for (dst, row) in out.chunks_mut(width).zip(input.chunks(width)) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for (d, &x) in dst.iter_mut().zip(row.iter()) {
            let e = (x - max).exp();
            *d = e;
            sum += e;
        }
        if sum > 0.0 {
            for d in dst.iter_mut() {
                *d /= sum;
            }
        }
    }
    out
}

/// Adds two equal-length `f32` operands elementwise.
///
/// # Errors
///
/// Returns [`cudaerror`] when the device cannot be opened, the source fails to
/// compile, a transfer fails, the length does not fit `unsigned`, or the launch
/// fails.
pub(crate) fn add(cache: &cache, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, cudaerror> {
    elementwise2(cache, source::ADDENTRY, lhs, rhs)
}

/// Multiplies two equal-length `f32` operands elementwise.
///
/// # Errors
///
/// See [`add`].
pub(crate) fn mul(cache: &cache, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, cudaerror> {
    elementwise2(cache, source::MULENTRY, lhs, rhs)
}

/// Applies the sigmoid linear unit elementwise.
///
/// # Errors
///
/// See [`add`].
pub(crate) fn silu(cache: &cache, input: &[f32]) -> Result<Vec<f32>, cudaerror> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let count = asunsigned(input.len())?;
    let stream = cache.stream()?;
    let function = cache.function(source::ELEMENTWISE, source::SILUENTRY)?;
    let da = upload(&stream, input)?;
    let mut dc = alloc(&stream, input.len())?;
    let mut args = stream.launch_builder(&function);
    args.arg(&da).arg(&mut dc).arg(&count);
    submit(&mut args, flatconfig(input.len())?)?;
    syncanddownload(&stream, &dc)
}

/// Runs the two-operand elementwise kernel named `entry` in [`ELEMENTWISE`].
fn elementwise2(
    cache: &cache,
    entry: &'static str,
    lhs: &[f32],
    rhs: &[f32],
) -> Result<Vec<f32>, cudaerror> {
    if lhs.is_empty() {
        return Ok(Vec::new());
    }
    let count = asunsigned(lhs.len())?;
    let stream = cache.stream()?;
    let function = cache.function(source::ELEMENTWISE, entry)?;
    let da = upload(&stream, lhs)?;
    let db = upload(&stream, rhs)?;
    let mut dc = alloc(&stream, lhs.len())?;
    let mut args = stream.launch_builder(&function);
    args.arg(&da).arg(&db).arg(&mut dc).arg(&count);
    submit(&mut args, flatconfig(lhs.len())?)?;
    syncanddownload(&stream, &dc)
}

/// RMS-normalises `rows` rows of `width` elements and scales them by `weight`.
///
/// # Errors
///
/// Returns [`cudaerror`] when the device cannot be opened, the source fails to
/// compile, a transfer fails, an extent does not fit its device type, or the
/// launch fails.
pub(crate) fn rmsnorm(
    cache: &cache,
    input: &[f32],
    weight: &[f32],
    rows: usize,
    width: usize,
    eps: f32,
) -> Result<Vec<f32>, cudaerror> {
    if rows == 0 {
        return Ok(Vec::new());
    }
    let total = rows.checked_mul(width).ok_or(cudaerror::sizeoverflow)?;
    let stream = cache.stream()?;
    let function = cache.function(source::RMSNORMCUDA, source::RMSNORMENTRY)?;
    let dx = upload(&stream, input)?;
    let dw = upload(&stream, weight)?;
    let mut dy = alloc(&stream, total)?;
    let widthu = asunsigned(width)?;
    let config = LaunchConfig {
        grid_dim: (asunsigned(rows)?, 1, 1),
        block_dim: (source::ROWTILE, 1, 1),
        shared_mem_bytes: source::ROWTILE * 4,
    };
    let mut args = stream.launch_builder(&function);
    args.arg(&dx).arg(&dw).arg(&mut dy).arg(&widthu).arg(&eps);
    submit(&mut args, config)?;
    syncanddownload(&stream, &dy)
}

/// Rotates every `headdim` slice of `rows` rows by its position angle.
///
/// # Errors
///
/// Returns [`cudaerror`] when the device cannot be opened, the source fails to
/// compile, a transfer fails, an extent does not fit its device type, or the
/// launch fails.
pub(crate) fn rope(
    cache: &cache,
    input: &[f32],
    rows: usize,
    seqaxis: usize,
    heads: usize,
    headdim: usize,
    base: f32,
) -> Result<Vec<f32>, cudaerror> {
    if rows == 0 {
        return Ok(Vec::new());
    }
    let pairs = rows
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(headdim / 2))
        .ok_or(cudaerror::sizeoverflow)?;
    let stream = cache.stream()?;
    let function = cache.function(source::ROPECUDA, source::ROPEENTRY)?;
    let dx = upload(&stream, input)?;
    let mut dy = alloc(&stream, input.len())?;
    let rowsu = asunsigned(rows)?;
    let seqaxu = asunsigned(seqaxis)?;
    let headsu = asunsigned(heads)?;
    let dimu = asunsigned(headdim)?;
    let mut args = stream.launch_builder(&function);
    args.arg(&dx)
        .arg(&mut dy)
        .arg(&rowsu)
        .arg(&seqaxu)
        .arg(&headsu)
        .arg(&dimu)
        .arg(&base);
    submit(&mut args, flatconfig(pairs)?)?;
    syncanddownload(&stream, &dy)
}

/// Gathers `indices.len()` rows of `width` elements from `table`.
///
/// The indices arrive already validated against the table's row count; the
/// kernel itself performs no range check, which is why the conversion to the
/// device's `unsigned long long` happens here rather than at the call site.
///
/// # Errors
///
/// Returns [`cudaerror`] when the device cannot be opened, the source fails to
/// compile, a transfer fails, an extent does not fit its device type, or the
/// launch fails.
pub(crate) fn embedding(
    cache: &cache,
    table: &[f32],
    indices: &[usize],
    width: usize,
) -> Result<Vec<f32>, cudaerror> {
    let rows = indices.len();
    if rows == 0 {
        return Ok(Vec::new());
    }
    let total = rows.checked_mul(width).ok_or(cudaerror::sizeoverflow)?;
    let devindices: Vec<u64> = indices.iter().map(|&index| index as u64).collect();
    let stream = cache.stream()?;
    let function = cache.function(source::EMBEDDINGCUDA, source::EMBEDDINGENTRY)?;
    let dtable = upload(&stream, table)?;
    let dindices = upload(&stream, &devindices)?;
    let mut dout = alloc(&stream, total)?;
    let rowsu = rows as u64;
    let widthu = width as u64;
    let mut args = stream.launch_builder(&function);
    args.arg(&dtable)
        .arg(&dindices)
        .arg(&mut dout)
        .arg(&rowsu)
        .arg(&widthu);
    submit(&mut args, flatconfig(total)?)?;
    syncanddownload(&stream, &dout)
}

/// Applies grouped-query scaled dot-product attention to `geometry.seq` positions.
///
/// `query` is `[seq, heads * headdim]` and `key`/`value` are
/// `[seq, kvheads * headdim]`. The extents come from the validated
/// [`attentionshape`], so this function revalidates nothing. The shared-memory
/// request is `2 * ATTENTIONLANES + headdim` floats: the tile that holds one
/// window of attention weights, the reduction scratch, and the per-head-dimension
/// accumulator. It therefore scales with `headdim` and not with `seq`, so the
/// kernel needs no sequence-length cap.
///
/// # Errors
///
/// Returns [`cudaerror`] when the device cannot be opened, the source fails to
/// compile, a transfer fails, the launch geometry or the shared-memory request
/// overflows, or the launch fails.
pub(crate) fn attention(
    cache: &cache,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    geometry: &attentionshape,
    causal: bool,
) -> Result<Vec<f32>, cudaerror> {
    let qseq = geometry.qseq;
    if qseq == 0 {
        return Ok(Vec::new());
    }
    let queryelements = qseq
        .checked_mul(geometry.heads)
        .and_then(|rows| rows.checked_mul(geometry.headdim))
        .ok_or(cudaerror::sizeoverflow)?;
    let stream = cache.stream()?;
    let function = cache.function(source::ATTENTIONCUDA, source::ATTENTIONENTRY)?;
    let dquery = upload(&stream, query)?;
    let dkey = upload(&stream, key)?;
    let dvalue = upload(&stream, value)?;
    let mut dout = alloc(&stream, queryelements)?;
    let qseqv = asunsigned(qseq)?;
    let kvseqv = asunsigned(geometry.kvseq)?;
    let headsv = asunsigned(geometry.heads)?;
    let kvheadsv = asunsigned(geometry.kvheads)?;
    let dimv = asunsigned(geometry.headdim)?;
    let causalv: u32 = u32::from(causal);
    let grid = qseqv.checked_mul(headsv).ok_or(cudaerror::sizeoverflow)?;
    let shared = (source::ATTENTIONLANES * 2)
        .checked_add(dimv)
        .and_then(|floats| floats.checked_mul(4))
        .ok_or(cudaerror::sizeoverflow)?;
    let config = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (source::ATTENTIONLANES, 1, 1),
        shared_mem_bytes: shared,
    };
    let mut args = stream.launch_builder(&function);
    args.arg(&dquery)
        .arg(&dkey)
        .arg(&dvalue)
        .arg(&mut dout)
        .arg(&qseqv)
        .arg(&kvseqv)
        .arg(&headsv)
        .arg(&kvheadsv)
        .arg(&dimv)
        .arg(&causalv);
    submit(&mut args, config)?;
    syncanddownload(&stream, &dout)
}

/// Runs the serial per-channel scan kernel, returning the `[n, d]` output and
/// the `[d, N]` final state.
///
/// When the state width fits the parallel kernel's thread-local arrays, the
/// three-stage parallel scan ([`scan_parallel`]) is used; the serial kernel
/// remains the fallback for wider states.
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
) -> Result<(Vec<f32>, Vec<f32>), cudaerror> {
    if n == 0 {
        return Ok((Vec::new(), init.to_vec()));
    }
    if statewidth <= source::SCANMAXN as usize {
        return scan_parallel(cache, input, init, a, b, c, skip, n, d, statewidth);
    }
    let stream = cache.stream()?;
    let function = cache.function(source::SCANCUDA, source::SCANENTRY)?;
    let dinput = upload(&stream, input)?;
    let da = upload(&stream, a)?;
    let db = upload(&stream, b)?;
    let dc = upload(&stream, c)?;
    let dskip = upload(&stream, skip)?;
    let mut dstate = upload(&stream, init)?;
    let outelements = n.checked_mul(d).ok_or(cudaerror::sizeoverflow)?;
    let mut dout = alloc(&stream, outelements)?;
    let nv = asunsigned(n)?;
    let dv = asunsigned(d)?;
    let statewidthv = asunsigned(statewidth)?;
    let config = LaunchConfig {
        grid_dim: (dv, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut args = stream.launch_builder(&function);
    args.arg(&dinput)
        .arg(&da)
        .arg(&db)
        .arg(&dc)
        .arg(&dskip)
        .arg(&mut dstate)
        .arg(&mut dout)
        .arg(&nv)
        .arg(&dv)
        .arg(&statewidthv);
    submit(&mut args, config)?;
    let out = syncanddownload(&stream, &dout)?;
    let state = syncanddownload(&stream, &dstate)?;
    Ok((out, state))
}

/// Runs the three-stage parallel scan: per-block aggregation, inter-block
/// prefix, and per-block apply, mirroring the Metal implementation. The apply
/// pass reproduces the serial recurrence bit-for-bit; the block start states
/// differ by the reordered float associations of the inter-block prefix, whose
/// rounding error grows like `P·ε·‖B‖`.
#[allow(clippy::too_many_arguments)]
fn scan_parallel(
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
) -> Result<(Vec<f32>, Vec<f32>), cudaerror> {
    let stream = cache.stream()?;
    let dinput = upload(&stream, input)?;
    let da = upload(&stream, a)?;
    let db = upload(&stream, b)?;
    let dc = upload(&stream, c)?;
    let dskip = upload(&stream, skip)?;
    let dinit = upload(&stream, init)?;
    let dN = d.checked_mul(statewidth).ok_or(cudaerror::sizeoverflow)?;
    let blocks = n.div_ceil(source::SCANCHUNK as usize);
    let totalelements = blocks
        .checked_mul(2)
        .and_then(|twice| twice.checked_mul(dN))
        .ok_or(cudaerror::sizeoverflow)?;
    let mut dtotal = alloc(&stream, totalelements)?;
    let mut dstateout = alloc(&stream, dN)?;
    let prefixelements = blocks
        .checked_add(1)
        .and_then(|one_more| one_more.checked_mul(2))
        .and_then(|twice| twice.checked_mul(dN))
        .ok_or(cudaerror::sizeoverflow)?;
    let mut dprefix = alloc(&stream, prefixelements)?;
    let outelements = n.checked_mul(d).ok_or(cudaerror::sizeoverflow)?;
    let mut dout = alloc(&stream, outelements)?;
    let nv = asunsigned(n)?;
    let dv = asunsigned(d)?;
    let nw = asunsigned(statewidth)?;
    let cv = asunsigned(source::SCANCHUNK as usize)?;
    let bv = asunsigned(blocks)?;
    let block = source::SCANBLOCKDIM;
    let grid_y = asunsigned(d.div_ceil(block as usize))?;

    // Pass 1: block aggregation.
    let agg = cache.function(source::SCANPARALLELCUDA, source::SCANAGGENTRY)?;
    let mut args = stream.launch_builder(&agg);
    args.arg(&dinput)
        .arg(&da)
        .arg(&db)
        .arg(&dc)
        .arg(&dskip)
        .arg(&dinit)
        .arg(&mut dtotal)
        .arg(&nv)
        .arg(&dv)
        .arg(&nw)
        .arg(&cv)
        .arg(&bv);
    submit(
        &mut args,
        LaunchConfig {
            grid_dim: (bv, grid_y, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        },
    )?;

    // Pass 2: inter-block prefix plus the whole-sequence final state.
    let prefix = cache.function(source::SCANPARALLELCUDA, source::SCANPREFIXENTRY)?;
    let mut args = stream.launch_builder(&prefix);
    args.arg(&mut dtotal)
        .arg(&dinit)
        .arg(&mut dstateout)
        .arg(&mut dprefix)
        .arg(&nv)
        .arg(&dv)
        .arg(&nw)
        .arg(&cv)
        .arg(&bv);
    let prefixthreads = asunsigned(dN.min(1024))?;
    submit(
        &mut args,
        LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (prefixthreads, 1, 1),
            shared_mem_bytes: 0,
        },
    )?;

    // Pass 3: per-block apply writing `out`; the final state was computed by
    // Pass 2 into `dstateout`.
    let apply = cache.function(source::SCANPARALLELCUDA, source::SCANAPPLYENTRY)?;
    let mut args = stream.launch_builder(&apply);
    args.arg(&dinput)
        .arg(&da)
        .arg(&db)
        .arg(&dc)
        .arg(&dskip)
        .arg(&dinit)
        .arg(&mut dprefix)
        .arg(&mut dout)
        .arg(&nv)
        .arg(&dv)
        .arg(&nw)
        .arg(&cv)
        .arg(&bv);
    submit(
        &mut args,
        LaunchConfig {
            grid_dim: (bv, grid_y, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        },
    )?;

    let out = syncanddownload(&stream, &dout)?;
    let state = syncanddownload(&stream, &dstateout)?;
    Ok((out, state))
}

/// The matmul extents as device-side `u32` values.
struct devicedims {
    /// The left operand's row count, as the kernel's `unsigned`.
    m: u32,
    /// The shared contraction extent, as the kernel's `unsigned`.
    k: u32,
    /// The right operand's column count, as the kernel's `unsigned`.
    n: u32,
}

impl devicedims {
    /// Validates that the extents fit the kernel's `unsigned` parameters.
    fn new(m: usize, k: usize, n: usize) -> Result<Self, cudaerror> {
        Ok(Self {
            m: asunsigned(m)?,
            k: asunsigned(k)?,
            n: asunsigned(n)?,
        })
    }
}

/// Builds the 2-D launch configuration for the NVRTC matmul kernel.
fn launchconfig(dims: &devicedims) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (
            dims.n.div_ceil(source::THREADEDGE),
            dims.m.div_ceil(source::THREADEDGE),
            1,
        ),
        block_dim: (source::THREADEDGE, source::THREADEDGE, 1),
        shared_mem_bytes: 0,
    }
}

/// One accumulative submission: a single execution stream that many operators
/// share, plus the output slot of every recorded step.
///
/// A `session` is the CUDA side of the accumulative `api::submission` contract
/// (docs/design.md §21.13). The stream serialises the recorded launches in
/// order, so intermediate results stay on the device and [`apply`](Self::apply)
/// synchronises and downloads only once. It owns no model weights and carries no
/// request-scoped state beyond its own slices; compiled kernels and the retained
/// context still come from the executor-owned [`cache`].
pub(crate) struct session {
    stream: Arc<CudaStream>,
    outputs: Vec<slot>,
}

/// The recorded output of one step: a device slice to download, no elements, or
/// a host-side run of zeros.
enum slot {
    buffer(CudaSlice<f32>),
    empty,
    zeros(usize),
}

impl session {
    /// Opens a fresh stream for one submission.
    pub(crate) fn new(cache: &cache) -> Result<Self, cudaerror> {
        let stream = cache.stream()?;
        Ok(Self {
            stream,
            outputs: Vec::new(),
        })
    }

    /// Uploads host data into a fresh device slice.
    pub(crate) fn upload<T: DeviceRepr>(&self, data: &[T]) -> Result<CudaSlice<T>, cudaerror> {
        upload(&self.stream, data)
    }

    /// Allocates `len` zeroed device elements.
    pub(crate) fn alloc(&self, len: usize) -> Result<CudaSlice<f32>, cudaerror> {
        alloc(&self.stream, len)
    }

    /// Returns the device slice of slot `index`, when that slot holds one.
    pub(crate) fn getbuffer(&self, index: usize) -> Option<&CudaSlice<f32>> {
        match self.outputs.get(index) {
            Some(slot::buffer(buffer)) => Some(buffer),
            _ => None,
        }
    }

    /// Returns the host-side data a non-buffer slot names, when `index` is one.
    pub(crate) fn gethostdata(&self, index: usize) -> Option<Vec<f32>> {
        match self.outputs.get(index) {
            Some(slot::empty) => Some(Vec::new()),
            Some(slot::zeros(count)) => Some(vec![0.0_f32; *count]),
            _ => None,
        }
    }

    /// Records a zero-element output without touching the device.
    fn recordempty(&mut self) -> usize {
        self.record(slot::empty)
    }

    /// Records an all-zero output of `count` elements without touching the device.
    fn recordzeros(&mut self, count: usize) -> usize {
        self.record(slot::zeros(count))
    }

    /// Records an output slot and returns its index.
    fn record(&mut self, output: slot) -> usize {
        let index = self.outputs.len();
        self.outputs.push(output);
        index
    }

    /// Synchronises once and downloads every recorded output, in slot order.
    pub(crate) fn apply(self) -> Result<Vec<Vec<f32>>, cudaerror> {
        let stream = self.stream;
        stream.synchronize().map_err(|error| cudaerror::transfer {
            message: error.to_string(),
        })?;
        self.outputs
            .into_iter()
            .map(|output| match output {
                slot::buffer(buffer) => {
                    stream
                        .clone_dtoh(&buffer)
                        .map_err(|error| cudaerror::transfer {
                            message: error.to_string(),
                        })
                }
                slot::empty => Ok(Vec::new()),
                slot::zeros(count) => Ok(vec![0.0_f32; count]),
            })
            .collect()
    }
}

/// Encodes one `[m, k] * [k, n]` block with the in-repo NVRTC kernel.
pub(crate) fn matmul_nvrtc_into(
    cache: &cache,
    session: &mut session,
    lhs: &CudaSlice<f32>,
    rhs: &CudaSlice<f32>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<usize, cudaerror> {
    if m == 0 || n == 0 {
        return Ok(session.recordempty());
    }
    if k == 0 {
        return Ok(session.recordzeros(m * n));
    }
    let dims = devicedims::new(m, k, n)?;
    let function = cache.function(source::MATMULCUDA, source::MATMULENTRY)?;
    let mut dc = session.alloc(m * n)?;
    {
        let mut args = session.stream.launch_builder(&function);
        args.arg(lhs)
            .arg(rhs)
            .arg(&mut dc)
            .arg(&dims.m)
            .arg(&dims.k)
            .arg(&dims.n);
        submit(&mut args, launchconfig(&dims))?;
    }
    Ok(session.record(slot::buffer(dc)))
}

/// Encodes one `[m, k] * [k, n]` block with cuBLAS.
pub(crate) fn matmul_cublas_into(
    _cache: &cache,
    session: &mut session,
    lhs: &CudaSlice<f32>,
    rhs: &CudaSlice<f32>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<usize, cudaerror> {
    if m == 0 || n == 0 {
        return Ok(session.recordempty());
    }
    if k == 0 {
        return Ok(session.recordzeros(m * n));
    }
    let dims = devicedims::new(m, k, n)?;
    let blas = CudaBlas::new(session.stream.clone()).map_err(|error| cudaerror::blas {
        message: error.to_string(),
    })?;
    let mut dc = session.alloc(m * n)?;
    let m_int = asint(dims.n)?;
    let n_int = asint(dims.m)?;
    let k_int = asint(dims.k)?;
    let config = GemmConfig {
        transa: cublasOperation_t::CUBLAS_OP_N,
        transb: cublasOperation_t::CUBLAS_OP_N,
        m: m_int,
        n: n_int,
        k: k_int,
        alpha: 1.0_f32,
        lda: m_int,
        ldb: k_int,
        beta: 0.0_f32,
        ldc: m_int,
    };
    gemm(&blas, config, rhs, lhs, &mut dc)?;
    Ok(session.record(slot::buffer(dc)))
}

/// Encodes an elementwise `add` of `count` elements.
pub(crate) fn add_into(
    cache: &cache,
    session: &mut session,
    lhs: &CudaSlice<f32>,
    rhs: &CudaSlice<f32>,
    count: usize,
) -> Result<usize, cudaerror> {
    elementwise2_into(cache, session, source::ADDENTRY, lhs, rhs, count)
}

/// Encodes an elementwise `mul` of `count` elements.
pub(crate) fn mul_into(
    cache: &cache,
    session: &mut session,
    lhs: &CudaSlice<f32>,
    rhs: &CudaSlice<f32>,
    count: usize,
) -> Result<usize, cudaerror> {
    elementwise2_into(cache, session, source::MULENTRY, lhs, rhs, count)
}

/// Encodes the two-operand elementwise kernel named `entry`.
fn elementwise2_into(
    cache: &cache,
    session: &mut session,
    entry: &'static str,
    lhs: &CudaSlice<f32>,
    rhs: &CudaSlice<f32>,
    count: usize,
) -> Result<usize, cudaerror> {
    if count == 0 {
        return Ok(session.recordempty());
    }
    let countu = asunsigned(count)?;
    let function = cache.function(source::ELEMENTWISE, entry)?;
    let mut dc = session.alloc(count)?;
    {
        let mut args = session.stream.launch_builder(&function);
        args.arg(lhs).arg(rhs).arg(&mut dc).arg(&countu);
        submit(&mut args, flatconfig(count)?)?;
    }
    Ok(session.record(slot::buffer(dc)))
}

/// Encodes a `silu` of `count` elements.
pub(crate) fn silu_into(
    cache: &cache,
    session: &mut session,
    input: &CudaSlice<f32>,
    count: usize,
) -> Result<usize, cudaerror> {
    if count == 0 {
        return Ok(session.recordempty());
    }
    let countu = asunsigned(count)?;
    let function = cache.function(source::ELEMENTWISE, source::SILUENTRY)?;
    let mut dc = session.alloc(count)?;
    {
        let mut args = session.stream.launch_builder(&function);
        args.arg(input).arg(&mut dc).arg(&countu);
        submit(&mut args, flatconfig(count)?)?;
    }
    Ok(session.record(slot::buffer(dc)))
}

/// Encodes a `softmax` over `rows` rows of `width` with cuDNN.
///
/// The non-finite-input fallback of the single-operator [`softmax`] is not
/// replicated here: a submission forwards the recorded step to cuDNN
/// unconditionally, so a non-finite row is the caller's responsibility to keep
/// out of a recorded chain.
pub(crate) fn softmax_into(
    _cache: &cache,
    session: &mut session,
    input: &CudaSlice<f32>,
    rows: usize,
    width: usize,
) -> Result<usize, cudaerror> {
    if rows == 0 {
        return Ok(session.recordempty());
    }
    let total = rows.checked_mul(width).ok_or(cudaerror::sizeoverflow)?;
    let handle = Cudnn::new(session.stream.clone()).map_err(cudnnfail)?;
    let operation = handle
        .create_softmax::<f32>(cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_CHANNEL)
        .map_err(cudnnfail)?;
    let dims = [asint(asunsigned(rows)?)?, asint(asunsigned(width)?)?, 1, 1];
    let strides = [asint(asunsigned(width)?)?, 1, 1, 1];
    let xdesc = handle
        .create_4d_tensor_ex::<f32>(dims, strides)
        .map_err(cudnnfail)?;
    let ydesc = handle
        .create_4d_tensor_ex::<f32>(dims, strides)
        .map_err(cudnnfail)?;
    let mut dy = session.alloc(total)?;
    let forward = SoftmaxForward {
        softmax: &operation,
        x: &xdesc,
        y: &ydesc,
    };
    softmax_forward(&forward, input, &mut dy)?;
    Ok(session.record(slot::buffer(dy)))
}

/// Encodes an RMS normalization.
pub(crate) fn rmsnorm_into(
    cache: &cache,
    session: &mut session,
    input: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    rows: usize,
    width: usize,
    eps: f32,
) -> Result<usize, cudaerror> {
    if rows == 0 {
        return Ok(session.recordempty());
    }
    let total = rows.checked_mul(width).ok_or(cudaerror::sizeoverflow)?;
    let function = cache.function(source::RMSNORMCUDA, source::RMSNORMENTRY)?;
    let mut dy = session.alloc(total)?;
    let widthu = asunsigned(width)?;
    let config = LaunchConfig {
        grid_dim: (asunsigned(rows)?, 1, 1),
        block_dim: (source::ROWTILE, 1, 1),
        shared_mem_bytes: source::ROWTILE * 4,
    };
    {
        let mut args = session.stream.launch_builder(&function);
        args.arg(input)
            .arg(weight)
            .arg(&mut dy)
            .arg(&widthu)
            .arg(&eps);
        submit(&mut args, config)?;
    }
    Ok(session.record(slot::buffer(dy)))
}

/// Encodes rotary position embedding.
pub(crate) fn rope_into(
    cache: &cache,
    session: &mut session,
    input: &CudaSlice<f32>,
    geometry: &ropeshape,
    base: f32,
) -> Result<usize, cudaerror> {
    let rows = geometry.rows;
    if rows == 0 {
        return Ok(session.recordempty());
    }
    let pairs = rows
        .checked_mul(geometry.heads)
        .and_then(|value| value.checked_mul(geometry.headdim / 2))
        .ok_or(cudaerror::sizeoverflow)?;
    let total = rows
        .checked_mul(geometry.heads)
        .and_then(|value| value.checked_mul(geometry.headdim))
        .ok_or(cudaerror::sizeoverflow)?;
    let function = cache.function(source::ROPECUDA, source::ROPEENTRY)?;
    let mut dy = session.alloc(total)?;
    let rowsu = asunsigned(rows)?;
    let seqaxu = asunsigned(geometry.seqaxis)?;
    let headsu = asunsigned(geometry.heads)?;
    let dimu = asunsigned(geometry.headdim)?;
    {
        let mut args = session.stream.launch_builder(&function);
        args.arg(input)
            .arg(&mut dy)
            .arg(&rowsu)
            .arg(&seqaxu)
            .arg(&headsu)
            .arg(&dimu)
            .arg(&base);
        submit(&mut args, flatconfig(pairs)?)?;
    }
    Ok(session.record(slot::buffer(dy)))
}

/// Encodes absolute-position rotary embedding of a single token.
pub(crate) fn rope_at_into(
    cache: &cache,
    session: &mut session,
    input: &CudaSlice<f32>,
    position: usize,
    heads: usize,
    headdim: usize,
    base: f32,
) -> Result<usize, cudaerror> {
    let total = heads.checked_mul(headdim).ok_or(cudaerror::sizeoverflow)?;
    let function = cache.function(source::ROPEATCUDA, source::ROPEATENTRY)?;
    let mut dy = session.alloc(total)?;
    let headsu = asunsigned(heads)?;
    let dimu = asunsigned(headdim)?;
    let posu = asunsigned(position)?;
    let pairs = heads
        .checked_mul(headdim / 2)
        .ok_or(cudaerror::sizeoverflow)?;
    {
        let mut args = session.stream.launch_builder(&function);
        args.arg(input)
            .arg(&mut dy)
            .arg(&headsu)
            .arg(&dimu)
            .arg(&posu)
            .arg(&base);
        submit(&mut args, flatconfig(pairs)?)?;
    }
    Ok(session.record(slot::buffer(dy)))
}

/// Encodes an embedding gather.
pub(crate) fn embedding_into(
    cache: &cache,
    session: &mut session,
    table: &CudaSlice<f32>,
    indices: &[usize],
    width: usize,
) -> Result<usize, cudaerror> {
    let rows = indices.len();
    if rows == 0 {
        return Ok(session.recordempty());
    }
    let total = rows.checked_mul(width).ok_or(cudaerror::sizeoverflow)?;
    let devindices: Vec<u64> = indices.iter().map(|&index| index as u64).collect();
    let function = cache.function(source::EMBEDDINGCUDA, source::EMBEDDINGENTRY)?;
    let dindices = session.upload(&devindices)?;
    let mut dout = session.alloc(total)?;
    let rowsu = rows as u64;
    let widthu = width as u64;
    {
        let mut args = session.stream.launch_builder(&function);
        args.arg(table)
            .arg(&dindices)
            .arg(&mut dout)
            .arg(&rowsu)
            .arg(&widthu);
        submit(&mut args, flatconfig(total)?)?;
    }
    Ok(session.record(slot::buffer(dout)))
}

/// Encodes grouped-query attention.
pub(crate) fn attention_into(
    cache: &cache,
    session: &mut session,
    query: &CudaSlice<f32>,
    key: &CudaSlice<f32>,
    value: &CudaSlice<f32>,
    geometry: &attentionshape,
    causal: bool,
) -> Result<usize, cudaerror> {
    let qseq = geometry.qseq;
    if qseq == 0 {
        return Ok(session.recordempty());
    }
    let queryelements = qseq
        .checked_mul(geometry.heads)
        .and_then(|rows| rows.checked_mul(geometry.headdim))
        .ok_or(cudaerror::sizeoverflow)?;
    let function = cache.function(source::ATTENTIONCUDA, source::ATTENTIONENTRY)?;
    let mut dout = session.alloc(queryelements)?;
    let qseqv = asunsigned(qseq)?;
    let kvseqv = asunsigned(geometry.kvseq)?;
    let headsv = asunsigned(geometry.heads)?;
    let kvheadsv = asunsigned(geometry.kvheads)?;
    let dimv = asunsigned(geometry.headdim)?;
    let causalv: u32 = u32::from(causal);
    let grid = qseqv.checked_mul(headsv).ok_or(cudaerror::sizeoverflow)?;
    let shared = (source::ATTENTIONLANES * 2)
        .checked_add(dimv)
        .and_then(|floats| floats.checked_mul(4))
        .ok_or(cudaerror::sizeoverflow)?;
    let config = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (source::ATTENTIONLANES, 1, 1),
        shared_mem_bytes: shared,
    };
    {
        let mut args = session.stream.launch_builder(&function);
        args.arg(query)
            .arg(key)
            .arg(value)
            .arg(&mut dout)
            .arg(&qseqv)
            .arg(&kvseqv)
            .arg(&headsv)
            .arg(&kvheadsv)
            .arg(&dimv)
            .arg(&causalv);
        submit(&mut args, config)?;
    }
    Ok(session.record(slot::buffer(dout)))
}

/// A cached and a new device slice of an incremental attention operand.
pub(crate) struct bufferpair<'a> {
    /// The accumulated cached operand.
    pub(crate) cached: &'a CudaSlice<f32>,
    /// The new operand.
    pub(crate) new: &'a CudaSlice<f32>,
}

/// Encodes incremental-decode attention: one query token attends over a cached
/// key/value buffer plus one new row, without materialising the concatenation.
pub(crate) fn attention_cached_into(
    cache: &cache,
    session: &mut session,
    query: &CudaSlice<f32>,
    key: bufferpair<'_>,
    value: bufferpair<'_>,
    geometry: &attentionshape,
    causal: bool,
) -> Result<usize, cudaerror> {
    let kvseq = geometry.kvseq;
    if kvseq == 0 {
        return Ok(session.recordempty());
    }
    let queryelements = geometry
        .qseq
        .checked_mul(geometry.heads)
        .and_then(|rows| rows.checked_mul(geometry.headdim))
        .ok_or(cudaerror::sizeoverflow)?;
    let function = cache.function(source::ATTENTIONCACHEDCUDA, source::ATTENTIONCACHEDENTRY)?;
    let mut dout = session.alloc(queryelements)?;
    let kvseqv = asunsigned(kvseq)?;
    let headsv = asunsigned(geometry.heads)?;
    let kvheadsv = asunsigned(geometry.kvheads)?;
    let dimv = asunsigned(geometry.headdim)?;
    let causalv: u32 = u32::from(causal);
    let grid = headsv;
    let shared = (source::ATTENTIONLANES * 2)
        .checked_add(dimv)
        .and_then(|floats| floats.checked_mul(4))
        .ok_or(cudaerror::sizeoverflow)?;
    let config = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (source::ATTENTIONLANES, 1, 1),
        shared_mem_bytes: shared,
    };
    {
        let mut args = session.stream.launch_builder(&function);
        args.arg(query)
            .arg(key.cached)
            .arg(key.new)
            .arg(value.cached)
            .arg(value.new)
            .arg(&mut dout)
            .arg(&kvseqv)
            .arg(&headsv)
            .arg(&kvheadsv)
            .arg(&dimv)
            .arg(&causalv);
        submit(&mut args, config)?;
    }
    Ok(session.record(slot::buffer(dout)))
}
