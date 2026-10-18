//! The safe CUDA operator and accumulative-session surface.
//!
//! Every direct operator runs through [`drive::drive`](drive::drive) and the
//! accumulative `*_into` steps through [`drive::submit`]; [`drive::gemm`] is
//! the cuBLAS matmul site. No direct platform `unsafe` calls live here, so
//! this file needs no `allow(unsafe_code)`.
use core::ffi::c_longlong;
use std::sync::Arc;

use cudarc::cublas::{CudaBlas, GemmConfig, StridedBatchedConfig, sys::cublasOperation_t};
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, LaunchConfig, PushKernelArg};

use super::cache;
use super::drive::{
    alloc, asint, asunsigned, drive, flatconfig, gemm, submit, syncanddownload, upload,
};
use crate::cuda::source;
use crate::engine::gpuerror;
use crate::ffi::gkernel::{self, buf, scalar};
use crate::{GENOP, GENOPS};
use api::geometry::{attentionshape, ropeshape};

/// The `rmsnorm` scalar group for the CUDA-family kernel: `(width, eps)`.
/// The row count is implicit in the grid, so it is not passed.
fn rmsnormparams(_rows: usize, width: usize, eps: f32) -> Vec<scalar> {
    vec![scalar::u32(width as u32), scalar::f32(eps)]
}

/// The `softmax` scalar group for the CUDA-family kernel: `(width)`.
fn softmaxparams(_rows: usize, width: usize) -> Vec<scalar> {
    vec![scalar::u32(width as u32)]
}

// Generates the operator surface over `drive`.
//
// CUDA keeps `matmul` on cuBLAS (`matmul_nvrtc` is the reference path) and
// keeps its parallel `scan`; `rope_at`, `attention_cached`, and
// `scan_cached` are served by the platform layer (host padding, session
// recording, or an explicit `unsupportedoperation`), so the list stops at
// the eight operators the engine actually dispatches here.
GENOPS!(add, mul, silu, rmsnorm, softmax, embedding, rope, attention);

/// Computes `[m, k] * [k, n]` with the in-repo NVRTC kernel.
///
/// This is the independent reference path: the kernel is compiled from
/// [`SOURCE`](crate::cuda::source::SOURCE) at run time, so it shares no code
/// with cuBLAS.
///
/// # Errors
///
/// Returns [`gpuerror`] when the device cannot be opened, the source fails to
/// compile, the kernel cannot be resolved, a transfer fails, a dimension does
/// not fit its device type, or the launch fails.
pub(crate) fn matmul_nvrtc(
    cache: &cache,
    lhs: &[f32],
    rhs: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, gpuerror> {
    if m == 0 || n == 0 {
        return Ok(Vec::new());
    }
    if k == 0 {
        return Ok(vec![0.0_f32; m * n]);
    }
    let dims = devicedims::new(m, k, n)?;
    let stream = cache.stream()?;
    let function = cache.function(source::MATMULENTRY)?;
    let da = upload(&stream, lhs)?;
    let db = upload(&stream, rhs)?;
    let mut dc = alloc(&stream, m * n)?;

    let mut args = stream.launch_builder(&function);
    args.arg(&da)
        .arg(&db)
        .arg(&mut dc)
        .arg(&dims.m)
        .arg(&dims.k)
        .arg(&dims.n)
        .arg(&1_u32);
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
/// Returns [`gpuerror`] when the device cannot be opened, a transfer fails, a
/// dimension does not fit `c_int`, or the cuBLAS call fails.
pub(crate) fn matmul_cublas(
    cache: &cache,
    lhs: &[f32],
    rhs: &[f32],
    m: usize,
    k: usize,
    n: usize,
    batch: usize,
) -> Result<Vec<f32>, gpuerror> {
    if m == 0 || n == 0 || batch == 0 {
        return Ok(Vec::new());
    }
    if k == 0 {
        return Ok(vec![0.0_f32; m * n * batch]);
    }
    let dims = devicedims::new(m, k, n)?;
    let (blas, stream) = cache.blas()?;

    let da = upload(&stream, lhs)?;
    let db = upload(&stream, rhs)?;
    let total = m
        .checked_mul(n)
        .and_then(|value| value.checked_mul(batch))
        .ok_or(gpuerror::sizeoverflow)?;
    let mut dc = alloc(&stream, total)?;

    // cuBLAS is column-major, so the operands are handed over swapped: the
    // row-major `[m, k]` left operand reads as a column-major `k x m` matrix,
    // and the row-major `[k, n]` right operand as a column-major `n x k` one.
    // Their product is the column-major `n x m` matrix whose row-major reading
    // is the `[m, n]` result the caller asked for. `m` and `n` below are
    // therefore the cuBLAS extents, not the contract's. The strides are one
    // slice's element count per operand, so the batch dimension walks the
    // row-major slices without any extra transfer or launch.
    let m_int = asint(dims.n)?;
    let n_int = asint(dims.m)?;
    let k_int = asint(dims.k)?;
    let config = StridedBatchedConfig {
        gemm: GemmConfig {
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
        },
        batch_size: asint(batch.try_into().map_err(|_| gpuerror::sizeoverflow)?)?,
        stride_a: (m * k) as c_longlong,
        stride_b: (k * n) as c_longlong,
        stride_c: (m * n) as c_longlong,
    };

    gemm(&blas, config, &db, &da, &mut dc)?;
    syncanddownload(&stream, &dc)
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
) -> Result<(Vec<f32>, Vec<f32>), gpuerror> {
    if n == 0 {
        return Ok((Vec::new(), init.to_vec()));
    }
    if statewidth <= source::SCANMAXN as usize {
        return scan_parallel(cache, input, init, a, b, c, skip, n, d, statewidth);
    }
    let stream = cache.stream()?;
    let function = cache.function(source::SCANENTRY)?;
    let dinput = upload(&stream, input)?;
    let da = upload(&stream, a)?;
    let db = upload(&stream, b)?;
    let dc = upload(&stream, c)?;
    let dskip = upload(&stream, skip)?;
    let mut dstate = upload(&stream, init)?;
    let outelements = n.checked_mul(d).ok_or(gpuerror::sizeoverflow)?;
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
) -> Result<(Vec<f32>, Vec<f32>), gpuerror> {
    let stream = cache.stream()?;
    let dinput = upload(&stream, input)?;
    let da = upload(&stream, a)?;
    let db = upload(&stream, b)?;
    let dc = upload(&stream, c)?;
    let dskip = upload(&stream, skip)?;
    let dinit = upload(&stream, init)?;
    let dN = d.checked_mul(statewidth).ok_or(gpuerror::sizeoverflow)?;
    let blocks = n.div_ceil(source::SCANCHUNK as usize);
    let totalelements = blocks
        .checked_mul(2)
        .and_then(|twice| twice.checked_mul(dN))
        .ok_or(gpuerror::sizeoverflow)?;
    let mut dtotal = alloc(&stream, totalelements)?;
    let mut dstateout = alloc(&stream, dN)?;
    let prefixelements = blocks
        .checked_add(1)
        .and_then(|one_more| one_more.checked_mul(2))
        .and_then(|twice| twice.checked_mul(dN))
        .ok_or(gpuerror::sizeoverflow)?;
    let mut dprefix = alloc(&stream, prefixelements)?;
    let outelements = n.checked_mul(d).ok_or(gpuerror::sizeoverflow)?;
    let mut dout = alloc(&stream, outelements)?;
    let nv = asunsigned(n)?;
    let dv = asunsigned(d)?;
    let nw = asunsigned(statewidth)?;
    let cv = asunsigned(source::SCANCHUNK as usize)?;
    let bv = asunsigned(blocks)?;
    let block = source::SCANBLOCKDIM;
    let grid_y = asunsigned(d.div_ceil(block as usize))?;

    // Pass 1: block aggregation.
    let agg = cache.function(source::SCANAGGENTRY)?;
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
    let prefix = cache.function(source::SCANPREFIXENTRY)?;
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
    let apply = cache.function(source::SCANAPPLYENTRY)?;
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
    fn new(m: usize, k: usize, n: usize) -> Result<Self, gpuerror> {
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
/// The slot-space marker for persistent device buffers registered through
/// [`session::persist`]. Persistent slots live in their own table (they must
/// survive every `apply` without being downloaded as step outputs), so their
/// index carries this high bit and `getbuffer` routes on it.
const PERSIST_BASE: usize = 1usize << 63;

pub(crate) struct session {
    stream: Arc<CudaStream>,
    outputs: Vec<slot>,
    /// Persistent device slices registered through [`session::persist`],
    /// retained across every apply so a device KV cache keeps its key/value
    /// rows resident.
    persisted: Vec<CudaSlice<f32>>,
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
    pub(crate) fn new(cache: &cache) -> Result<Self, gpuerror> {
        let stream = cache.stream()?;
        Ok(Self {
            stream,
            outputs: Vec::new(),
            persisted: Vec::new(),
        })
    }

    /// Uploads host data into a fresh device slice.
    pub(crate) fn upload<T: DeviceRepr>(&self, data: &[T]) -> Result<CudaSlice<T>, gpuerror> {
        upload(&self.stream, data)
    }

    /// Allocates `len` zeroed device elements.
    pub(crate) fn alloc(&self, len: usize) -> Result<CudaSlice<f32>, gpuerror> {
        alloc(&self.stream, len)
    }

    /// Registers `data` as a persistent device-resident slice.
    ///
    /// The slice is retained in the session's persistent table (never
    /// downloaded as a step output). The returned slot carries
    /// [`PERSIST_BASE`], so `getbuffer` routes it to the persistent table even
    /// after `apply` rebuilt the output list.
    pub(crate) fn persist(&mut self, data: &[f32]) -> Result<usize, gpuerror> {
        // WSL/NVIDIA virtual stacks: a `clone_htod` slice carries a write
        // event, and a later kernel launch that waits on that event silently
        // no-ops (verified on T600). All-zero persistence — the KV-cache
        // case — allocates without events so an append kernel can write into
        // the slot; non-zero payloads keep the upload path (their event is
        // harmless as a kernel *input*).
        let slice = if data.iter().all(|value| *value == 0.0) {
            alloc(&self.stream, data.len())?
        } else {
            upload(&self.stream, data)?
        };
        let index = self.persisted.len();
        self.persisted.push(slice);
        Ok(PERSIST_BASE | index)
    }

    /// Returns the device slice of slot `index`, when that slot holds one.
    pub(crate) fn getbuffer(&self, index: usize) -> Option<&CudaSlice<f32>> {
        if index & PERSIST_BASE != 0 {
            return self.persisted.get(index & !PERSIST_BASE);
        }
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
            _ => {
                // A persistent slot names a device-resident slice that is not
                // part of the per-apply download list; copy it out here on
                // demand so a leading-window view downloads the real prefix.
                // A failed device-to-host copy falls back to `None` (the
                // caller reports the slot as out of range) rather than
                // widening this slot's `Option` contract.
                if index & PERSIST_BASE != 0 {
                    let persist_index = index & !PERSIST_BASE;
                    let slice = self.persisted.get(persist_index)?;
                    return self
                        .stream
                        .clone_dtoh(slice)
                        .map_err(|error| gpuerror::transfer {
                            message: error.to_string(),
                        })
                        .ok();
                }
                None
            }
        }
    }

    /// Appends one `[1, width]` device row into the leading rows of a
    /// persistent cache buffer, in stream order (the same stream the kernels
    /// run on, ordered before `apply`'s synchronise).
    ///
    /// The copy is a launched kernel rather than `cuMemcpyDtoD` because the
    /// driver's D2D memcpy silently no-ops on WSL/NVIDIA virtual stacks
    /// (verified on T600; kernel launches on the same stream work), and the
    /// kernel keeps the append ordered with the surrounding compute.
    ///
    /// `cached` is the persistent slot index, and the copy writes directly
    /// into the resident allocation — never a `CudaSlice` copy, whose clone
    /// performs a deep device-to-device copy and would leave the resident
    /// buffer untouched.
    pub(crate) fn recordappend(
        &mut self,
        cache: &cache,
        cached: usize,
        new: &CudaSlice<f32>,
        position: usize,
    ) -> Result<(), gpuerror> {
        let elements = new.len();
        if elements == 0 {
            return Err(gpuerror::execution {
                message: "append of an empty row".into(),
            });
        }
        if cached & PERSIST_BASE == 0 {
            return Err(gpuerror::execution {
                message: "append target is not a persistent slot".into(),
            });
        }
        let base_index = cached & !PERSIST_BASE;
        let stream = self.stream.clone();
        let base = self
            .persisted
            .get_mut(base_index)
            .ok_or(gpuerror::execution {
                message: "append target slot out of range".into(),
            })?;
        let start = position
            .checked_mul(elements)
            .ok_or(gpuerror::sizeoverflow)?;
        let end = start.checked_add(elements).ok_or(gpuerror::sizeoverflow)?;
        if end > base.len() {
            return Err(gpuerror::execution {
                message: "append row out of the cache bounds".into(),
            });
        }
        let function = cache.function(source::COPYENTRY)?;
        let elementsu = asunsigned(elements)?;
        let offsetu = asunsigned(start)?;
        // WSL/NVIDIA virtual stacks: the stream event-wait that guards the
        // persistent slot's prior upload can stall a following launch, so
        // force the upload to complete before the copy is queued.
        stream.synchronize().map_err(|error| gpuerror::transfer {
            message: error.to_string(),
        })?;
        // The kernel addresses the target as `c[off + i]` where `off` is an
        // absolute element offset into the persistent allocation, so the view
        // must span the whole base buffer — slicing it to `start..end` first
        // would double-count the offset and write past the row.
        let mut view = base.slice_mut(0..base.len());
        let mut args = stream.launch_builder(&function);
        args.arg(new).arg(&mut view).arg(&elementsu).arg(&offsetu);
        submit(&mut args, flatconfig(elements)?)?;
        stream.synchronize().map_err(|error| gpuerror::transfer {
            message: error.to_string(),
        })?;
        Ok(())
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
    ///
    /// The session keeps its stream and starts a fresh slot list, so a
    /// persistent session can record the next chain into the same device
    /// state.
    pub(crate) fn apply(&mut self) -> Result<Vec<Vec<f32>>, gpuerror> {
        let stream = self.stream.clone();
        stream.synchronize().map_err(|error| gpuerror::transfer {
            message: error.to_string(),
        })?;
        core::mem::take(&mut self.outputs)
            .into_iter()
            .map(|output| match output {
                slot::buffer(buffer) => {
                    stream
                        .clone_dtoh(&buffer)
                        .map_err(|error| gpuerror::transfer {
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
) -> Result<usize, gpuerror> {
    if m == 0 || n == 0 {
        return Ok(session.recordempty());
    }
    if k == 0 {
        return Ok(session.recordzeros(m * n));
    }
    let dims = devicedims::new(m, k, n)?;
    let function = cache.function(source::MATMULENTRY)?;
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
) -> Result<usize, gpuerror> {
    if m == 0 || n == 0 {
        return Ok(session.recordempty());
    }
    if k == 0 {
        return Ok(session.recordzeros(m * n));
    }
    let dims = devicedims::new(m, k, n)?;
    let blas = CudaBlas::new(session.stream.clone()).map_err(|error| gpuerror::blas {
        message: error.to_string(),
    })?;
    let mut dc = session.alloc(m * n)?;
    let m_int = asint(dims.n)?;
    let n_int = asint(dims.m)?;
    let k_int = asint(dims.k)?;
    let config = StridedBatchedConfig {
        gemm: GemmConfig {
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
        },
        batch_size: 1,
        stride_a: (m * k) as c_longlong,
        stride_b: (k * n) as c_longlong,
        stride_c: (m * n) as c_longlong,
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
) -> Result<usize, gpuerror> {
    elementwise2_into(cache, session, source::ADDENTRY, lhs, rhs, count)
}

/// Encodes an elementwise `mul` of `count` elements.
pub(crate) fn mul_into(
    cache: &cache,
    session: &mut session,
    lhs: &CudaSlice<f32>,
    rhs: &CudaSlice<f32>,
    count: usize,
) -> Result<usize, gpuerror> {
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
) -> Result<usize, gpuerror> {
    if count == 0 {
        return Ok(session.recordempty());
    }
    let countu = asunsigned(count)?;
    let function = cache.function(entry)?;
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
) -> Result<usize, gpuerror> {
    if count == 0 {
        return Ok(session.recordempty());
    }
    let countu = asunsigned(count)?;
    let function = cache.function(source::SILUENTRY)?;
    let mut dc = session.alloc(count)?;
    {
        let mut args = session.stream.launch_builder(&function);
        args.arg(input).arg(&mut dc).arg(&countu);
        submit(&mut args, flatconfig(count)?)?;
    }
    Ok(session.record(slot::buffer(dc)))
}

/// Encodes a `softmax` over `rows` rows of `width` with the in-repo NVRTC
/// kernel.
///
/// The launch is enqueued on the submission's stream like every other recorded
/// step, so the intermediate stays on the device until `apply`. Unlike the
/// single-operator [`softmax`], no host-side finite check is possible here —
/// operands may already be device-resident — so a non-finite row takes the
/// kernel's own path. For an all-negative-infinity row that is zeros, the same
/// behavior the retired cuDNN submission path had; the single-operator path
/// still propagates NaN through [`scalarsoftmax`].
pub(crate) fn softmax_into(
    cache: &cache,
    session: &mut session,
    input: &CudaSlice<f32>,
    rows: usize,
    width: usize,
) -> Result<usize, gpuerror> {
    if rows == 0 {
        return Ok(session.recordempty());
    }
    let total = rows.checked_mul(width).ok_or(gpuerror::sizeoverflow)?;
    let function = cache.function(source::SOFTMAXENTRY)?;
    let mut dy = session.alloc(total)?;
    let widthu = asunsigned(width)?;
    let config = LaunchConfig {
        grid_dim: (asunsigned(rows)?, 1, 1),
        block_dim: (source::RMSNORMTILE, 1, 1),
        shared_mem_bytes: source::RMSNORMTILE * 4,
    };
    {
        let mut args = session.stream.launch_builder(&function);
        args.arg(input).arg(&mut dy).arg(&widthu);
        submit(&mut args, config)?;
    }
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
) -> Result<usize, gpuerror> {
    if rows == 0 {
        return Ok(session.recordempty());
    }
    let total = rows.checked_mul(width).ok_or(gpuerror::sizeoverflow)?;
    let function = cache.function(source::RMSNORMENTRY)?;
    let mut dy = session.alloc(total)?;
    let widthu = asunsigned(width)?;
    let config = LaunchConfig {
        grid_dim: (asunsigned(rows)?, 1, 1),
        block_dim: (source::RMSNORMTILE, 1, 1),
        shared_mem_bytes: source::RMSNORMTILE * 4,
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
) -> Result<usize, gpuerror> {
    let rows = geometry.rows;
    if rows == 0 {
        return Ok(session.recordempty());
    }
    let pairs = rows
        .checked_mul(geometry.heads)
        .and_then(|value| value.checked_mul(geometry.headdim / 2))
        .ok_or(gpuerror::sizeoverflow)?;
    let total = rows
        .checked_mul(geometry.heads)
        .and_then(|value| value.checked_mul(geometry.headdim))
        .ok_or(gpuerror::sizeoverflow)?;
    let function = cache.function(source::ROPEENTRY)?;
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
) -> Result<usize, gpuerror> {
    let total = heads.checked_mul(headdim).ok_or(gpuerror::sizeoverflow)?;
    let function = cache.function(source::ROPEATENTRY)?;
    let mut dy = session.alloc(total)?;
    let headsu = asunsigned(heads)?;
    let dimu = asunsigned(headdim)?;
    let posu = asunsigned(position)?;
    let pairs = heads
        .checked_mul(headdim / 2)
        .ok_or(gpuerror::sizeoverflow)?;
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
) -> Result<usize, gpuerror> {
    let rows = indices.len();
    if rows == 0 {
        return Ok(session.recordempty());
    }
    let total = rows.checked_mul(width).ok_or(gpuerror::sizeoverflow)?;
    let devindices: Vec<u64> = indices.iter().map(|&index| index as u64).collect();
    let function = cache.function(source::EMBEDDINGENTRY)?;
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
) -> Result<usize, gpuerror> {
    let qseq = geometry.qseq;
    if qseq == 0 {
        return Ok(session.recordempty());
    }
    let queryelements = qseq
        .checked_mul(geometry.heads)
        .and_then(|rows| rows.checked_mul(geometry.headdim))
        .ok_or(gpuerror::sizeoverflow)?;
    let function = cache.function(source::ATTENTIONENTRY)?;
    let mut dout = session.alloc(queryelements)?;
    let qseqv = asunsigned(qseq)?;
    let kvseqv = asunsigned(geometry.kvseq)?;
    let headsv = asunsigned(geometry.heads)?;
    let kvheadsv = asunsigned(geometry.kvheads)?;
    let dimv = asunsigned(geometry.headdim)?;
    let causalv: u32 = u32::from(causal);
    let grid = qseqv.checked_mul(headsv).ok_or(gpuerror::sizeoverflow)?;
    let shared = (source::ATTENTIONLANES * 2)
        .checked_add(dimv)
        .and_then(|floats| floats.checked_mul(4))
        .ok_or(gpuerror::sizeoverflow)?;
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
) -> Result<usize, gpuerror> {
    let kvseq = geometry.kvseq;
    if kvseq == 0 {
        return Ok(session.recordempty());
    }
    let queryelements = geometry
        .qseq
        .checked_mul(geometry.heads)
        .and_then(|rows| rows.checked_mul(geometry.headdim))
        .ok_or(gpuerror::sizeoverflow)?;
    let function = cache.function(source::ATTENTIONCACHEDENTRY)?;
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
        .ok_or(gpuerror::sizeoverflow)?;
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
