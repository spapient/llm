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
//! The launch helpers take the validated plans produced by [`crate::geometry`]
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
use crate::geometry::attentionshape;
use crate::source;

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
        message: error.to_string(),
    })?;
    let module = device.load_module(ptx).map_err(|error| cudaerror::module {
        message: error.to_string(),
    })?;
    module
        .load_function(entry)
        .map_err(|error| cudaerror::module {
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

    // SAFETY: `da`, `db`, and `dc` are live device allocations. `da` holds the
    // `[m, k]` left operand and `db` the `[k, n]` right operand; read as
    // column-major they are the `k x m` and `n x k` matrices the `lda`/`ldb`
    // leading dimensions describe, and `dc` is written as an `n x m`
    // column-major matrix with leading dimension `ldc`, i.e. the row-major
    // `[m, n]` result. Every dimension is at least 1 and fits `c_int`, so no
    // out-of-bounds access is possible.
    unsafe {
        blas.gemm(config, &db, &da, &mut dc)
            .map_err(|error| cudaerror::blas {
                message: error.to_string(),
            })?;
    }
    syncanddownload(&stream, &dc)
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

    // SAFETY: `dx` and `dy` are live device allocations, each holding `total`
    // contiguous `f32`, and `xdesc`/`ydesc` describe exactly that buffer shape
    // with the strides it already has, so cuDNN reads and writes within the
    // allocations. `CUDNN_SOFTMAX_ACCURATE` is the max-subtracting formulation,
    // which is the numerically stable variant the scalar contract requires. The
    // scaling pair is the neutral `(1, 0)`, so it changes no operand.
    unsafe {
        forward
            .launch(
                (1.0_f32, 0.0_f32),
                cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE,
                &dx,
                &mut dy,
            )
            .map_err(cudnnfail)?;
    }
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
    let seq = geometry.seq;
    if seq == 0 {
        return Ok(Vec::new());
    }
    let queryelements = seq
        .checked_mul(geometry.heads)
        .and_then(|rows| rows.checked_mul(geometry.headdim))
        .ok_or(cudaerror::sizeoverflow)?;
    let stream = cache.stream()?;
    let function = cache.function(source::ATTENTIONCUDA, source::ATTENTIONENTRY)?;
    let dquery = upload(&stream, query)?;
    let dkey = upload(&stream, key)?;
    let dvalue = upload(&stream, value)?;
    let mut dout = alloc(&stream, queryelements)?;
    let seqv = asunsigned(seq)?;
    let headsv = asunsigned(geometry.heads)?;
    let kvheadsv = asunsigned(geometry.kvheads)?;
    let dimv = asunsigned(geometry.headdim)?;
    let causalv: u32 = u32::from(causal);
    let grid = seqv.checked_mul(headsv).ok_or(cudaerror::sizeoverflow)?;
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
        .arg(&seqv)
        .arg(&headsv)
        .arg(&kvheadsv)
        .arg(&dimv)
        .arg(&causalv);
    submit(&mut args, config)?;
    syncanddownload(&stream, &dout)
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
