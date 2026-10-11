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
//! slices, run one kernel or one library call, download the result. Every
//! device, stream, module, and handle object is created locally and released by
//! scope, so no device state outlives a call.
//!
//! The launch helpers take the validated plans produced by [`crate::geometry`]
//! rather than loose extents, so no number is re-derived after it was checked.

use core::ffi::c_int;
use std::sync::Arc;

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

/// Opens CUDA device 0 and returns its context with the default stream.
///
/// The stream carries its own `Arc` of the context, so a caller that only needs
/// the stream may drop the context immediately.
fn opendevice() -> Result<(Arc<CudaContext>, Arc<CudaStream>), cudaerror> {
    let ctx = CudaContext::new(0).map_err(|error| cudaerror::device {
        message: error.to_string(),
    })?;
    let stream = ctx.default_stream();
    Ok((ctx, stream))
}

/// Compiles `text` with NVRTC, loads the module, and resolves `entry`.
fn loadkernel(ctx: &Arc<CudaContext>, text: &str, entry: &str) -> Result<CudaFunction, cudaerror> {
    let ptx = compile_ptx(text).map_err(|error| cudaerror::compile {
        message: error.to_string(),
    })?;
    let module = ctx.load_module(ptx).map_err(|error| cudaerror::module {
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
    let (ctx, stream) = opendevice()?;
    let function = loadkernel(&ctx, source::MATMULCUDA, source::MATMULENTRY)?;
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
    let (_ctx, stream) = opendevice()?;
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
/// scalar reference on finite rows. The behavior of a row of negative infinity
/// under `CUDNN_SOFTMAX_ACCURATE` is not asserted: no NVIDIA GPU was reachable
/// where this was written, so the case is recorded as not executed (E4).
///
/// # Errors
///
/// Returns [`cudaerror`] when the device cannot be opened, cuDNN cannot be
/// loaded or set up, a transfer fails, a dimension does not fit `c_int`, or the
/// softmax call fails.
pub(crate) fn softmax(input: &[f32], rows: usize, width: usize) -> Result<Vec<f32>, cudaerror> {
    if rows == 0 {
        return Ok(Vec::new());
    }
    let total = rows.checked_mul(width).ok_or(cudaerror::sizeoverflow)?;
    let (_ctx, stream) = opendevice()?;
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

/// Adds two equal-length `f32` operands elementwise.
///
/// # Errors
///
/// Returns [`cudaerror`] when the device cannot be opened, the source fails to
/// compile, a transfer fails, the length does not fit `unsigned`, or the launch
/// fails.
pub(crate) fn add(lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, cudaerror> {
    elementwise2(source::ADDENTRY, lhs, rhs)
}

/// Multiplies two equal-length `f32` operands elementwise.
///
/// # Errors
///
/// See [`add`].
pub(crate) fn mul(lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, cudaerror> {
    elementwise2(source::MULENTRY, lhs, rhs)
}

/// Applies the sigmoid linear unit elementwise.
///
/// # Errors
///
/// See [`add`].
pub(crate) fn silu(input: &[f32]) -> Result<Vec<f32>, cudaerror> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let count = asunsigned(input.len())?;
    let (ctx, stream) = opendevice()?;
    let function = loadkernel(&ctx, source::ELEMENTWISE, source::SILUENTRY)?;
    let da = upload(&stream, input)?;
    let mut dc = alloc(&stream, input.len())?;
    let mut args = stream.launch_builder(&function);
    args.arg(&da).arg(&mut dc).arg(&count);
    submit(&mut args, flatconfig(input.len())?)?;
    syncanddownload(&stream, &dc)
}

/// Runs the two-operand elementwise kernel named `entry` in [`ELEMENTWISE`].
fn elementwise2(entry: &str, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, cudaerror> {
    if lhs.is_empty() {
        return Ok(Vec::new());
    }
    let count = asunsigned(lhs.len())?;
    let (ctx, stream) = opendevice()?;
    let function = loadkernel(&ctx, source::ELEMENTWISE, entry)?;
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
    let (ctx, stream) = opendevice()?;
    let function = loadkernel(&ctx, source::RMSNORMCUDA, source::RMSNORMENTRY)?;
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
    let (ctx, stream) = opendevice()?;
    let function = loadkernel(&ctx, source::ROPECUDA, source::ROPEENTRY)?;
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
    let (ctx, stream) = opendevice()?;
    let function = loadkernel(&ctx, source::EMBEDDINGCUDA, source::EMBEDDINGENTRY)?;
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
    let (ctx, stream) = opendevice()?;
    let function = loadkernel(&ctx, source::ATTENTIONCUDA, source::ATTENTIONENTRY)?;
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
