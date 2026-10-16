#![allow(unsafe_code)]
//! The single platform-FFI island of the Metal backend.
//!
//! Authority: docs/design.md §21.8. This is the only file in the crate that
//! declares `#![allow(unsafe_code)]`; the crate root keeps
//! `#![deny(unsafe_code)]`. Every platform call that the Objective-C bridge
//! marks `unsafe` is confined here, and each `unsafe` construct is directly
//! preceded by a `// SAFETY:` comment. The registered site count (three) is
//! enforced by `scripts/checkunsafe.sh`.
//!
//! The three sites are one per **driver handoff block**, not one per operator:
//! the synchronous `dispatch` block (upload, buffer binding, scalar binding,
//! launch, `endEncoding`/`commit`/`waitUntilCompleted`, and download in one
//! block), plus the accumulative `session`'s encode block and commit block.
//! Every kernel in [`crate::source`] reuses those same sites, so the count does
//! not grow with the operator count.
//!
//! The safe functions expose a narrow, run-to-completion interface: upload host
//! slices, dispatch one kernel, download the result. Buffers, command buffers,
//! and encoders are created locally and released by scope, so no operand or
//! result survives a call. The only state that does survive is [`cache`] — the
//! device and the compiled pipelines — which the executor owns and which holds
//! no request-scoped value (docs/design.md §21.6).

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};

use crate::error::metalerror;
use crate::source;
use api::geometry::{attentionshape, matmulshape, ropeshape};

/// The threadgroup edge used when dispatching the matmul kernel.
const THREADEDGE: usize = 16;

/// The device state and compiled pipelines one executor owns.
///
/// Compiling MSL at run time costs on the order of a millisecond, which
/// dominated every operator call this backend served before the cache existed:
/// `docs/design.md` §21.10.1 records `gpu/embedding` at 3.97 ms against 6.7 µs
/// for the CPU — a fixed cost independent of the work. The cache is owned by the
/// executor and shared by its clones, deliberately *not* kept in a process-wide
/// singleton: two independently constructed executors compile independently,
/// and nothing here outlives the executor that owns it.
///
/// What the cache holds is device state and compiled code, never an operand, a
/// result, or any other request-scoped value, so sharing it across requests
/// cannot pass data between them.
pub(crate) struct cache {
    state: Mutex<cachestate>,
}

/// The mutable half of [`cache`], guarded by a single lock that is never
/// acquired while another is held (the crate has exactly one).
#[derive(Default)]
struct cachestate {
    /// The system default device, opened on first use.
    ///
    /// `None` means "not opened yet or the last attempt failed", so an
    /// unavailable device is retried on the next call rather than latched.
    device: Option<Retained<ProtocolObject<dyn MTLDevice>>>,
    /// The device's command queue, opened on first use and reused by every
    /// dispatch.
    ///
    /// `newCommandQueue` is a driver round trip; caching it removes that fixed
    /// cost from every operator call while the command buffers themselves stay
    /// per-dispatch (Metal command buffers are single-use by contract). The
    /// queue lives beside the one device it was created for, so it can never
    /// outlive a device it does not belong to.
    queue: Option<Retained<ProtocolObject<dyn MTLCommandQueue>>>,
    /// One compiled compute pipeline per `(source, entry)` pair.
    ///
    /// The key carries the MSL **source text**, so editing a kernel in
    /// [`crate::source`] can never reuse a stale binary. The map is
    /// device-specific by construction: it lives beside the one device it was
    /// compiled for, so no device belongs in the key.
    pipelines: HashMap<
        (&'static str, &'static str),
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    >,
}

impl cache {
    /// Creates an empty cache.
    ///
    /// Opening the device is deferred to the first operator call, so
    /// construction cannot fail and the error still surfaces per call.
    pub(crate) fn new() -> Self {
        Self {
            state: Mutex::new(cachestate::default()),
        }
    }

    /// Returns the system default device, opening it once.
    fn device(&self) -> Result<Retained<ProtocolObject<dyn MTLDevice>>, metalerror> {
        let mut state = self.lock();
        state.device()
    }

    /// Returns how many compiled pipelines the cache currently holds.
    ///
    /// Exposed so that the executor can report it and the contract suite can
    /// assert that a second call reuses the binary instead of recompiling it.
    pub(crate) fn pipelinecount(&self) -> usize {
        self.lock().pipelines.len()
    }

    /// Returns the compiled pipeline for `(source, entry)`, compiling it once.
    ///
    /// The lock is held across a miss and its compilation. That serializes
    /// concurrent first calls for the same kernel instead of compiling it
    /// several times over, and it is released before the caller creates a
    /// command buffer, so no transfer or GPU wait happens under the lock.
    fn pipeline(
        &self,
        source: &'static str,
        entry: &'static str,
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, metalerror> {
        let mut state = self.lock();
        if let Some(compiled) = state.pipelines.get(&(source, entry)) {
            return Ok(compiled.clone());
        }
        let device = state.device()?;
        let compiled = compile(&device, source, entry)?;
        state.pipelines.insert((source, entry), compiled.clone());
        Ok(compiled)
    }

    /// Returns the device's command queue, opening and caching it once.
    fn queue(&self) -> Result<Retained<ProtocolObject<dyn MTLCommandQueue>>, metalerror> {
        let mut state = self.lock();
        if let Some(queue) = &state.queue {
            return Ok(queue.clone());
        }
        let device = state.device()?;
        let opened = device
            .newCommandQueue()
            .ok_or_else(|| metalerror::execution {
                message: "the device returned no command queue".to_string(),
            })?;
        state.queue = Some(opened.clone());
        Ok(opened)
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
    /// Returns the cached device, opening and caching it if absent.
    fn device(&mut self) -> Result<Retained<ProtocolObject<dyn MTLDevice>>, metalerror> {
        if let Some(device) = &self.device {
            return Ok(device.clone());
        }
        let opened = objc2_metal::MTLCreateSystemDefaultDevice().ok_or(metalerror::nodevice)?;
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
            .field("pipelines", &self.lock().pipelines.len())
            .finish()
    }
}

/// Computes `batch` blocks of `[m, k] * [k, n]` on the default Metal device.
///
/// `lhs` holds `batch * m * k` row-major elements and `rhs` holds `batch * k *
/// n`; the result holds `batch * m * n`. The call is synchronous: it returns
/// only after the kernel has completed and the result has been copied back to
/// the host.
///
/// The whole call is **one** dispatch whose grid depth is the block count. That
/// is not a micro-optimization: a command buffer round trip costs the same for
/// one dispatch as for many (docs/design.md §21.10.2), so dispatching a block at
/// a time would pay that cost once per block and multiply the call's latency by
/// the batch size.
///
/// # Errors
///
/// Returns [`metalerror`] when no device is available, a buffer cannot be
/// allocated, the MSL source fails to compile, the compute pipeline cannot be
/// built, a dimension does not fit a `u32`, or the kernel fails to execute.
pub(crate) fn matmul(
    cache: &cache,
    lhs: &[f32],
    rhs: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, metalerror> {
    let total = batch
        .checked_mul(m)
        .and_then(|value| value.checked_mul(n))
        .ok_or(metalerror::sizeoverflow)?;
    if m == 0 || n == 0 {
        return Ok(Vec::new());
    }
    if k == 0 {
        return Ok(vec![0.0_f32; total]);
    }
    let dims = Extents::new(m, k, n)?;

    let params = [dims.m, dims.k, dims.n];
    let mut results = dispatch(
        cache,
        source::MATMULMSL,
        source::MATMULENTRY,
        &[buf::host(lhs), buf::host(rhs), buf::out(total)],
        &params,
        MTLSize {
            width: dims.cols,
            height: dims.rows,
            depth: batch,
        },
        dims.threadgroup(),
    )?;
    Ok(results.remove(0))
}

/// The matmul extents as device-side `u32` values.
///
/// The host-side extents are kept alongside their `u32` counterparts because
/// the threadgroup shape is expressed in host units while the kernel parameters
/// are expressed in device units.
struct Extents {
    /// The left operand's row count, as the kernel's `uint`.
    m: u32,
    /// The shared contraction extent, as the kernel's `uint`.
    k: u32,
    /// The right operand's column count, as the kernel's `uint`.
    n: u32,
    /// The host-side row count.
    rows: usize,
    /// The host-side column count.
    cols: usize,
}

impl Extents {
    /// Validates that the extents fit the kernel's `uint` parameters.
    fn new(m: usize, k: usize, n: usize) -> Result<Self, metalerror> {
        Ok(Self {
            m: asu32(m)?,
            k: asu32(k)?,
            n: asu32(n)?,
            rows: m,
            cols: n,
        })
    }

    /// Returns the threadgroup edge, never exceeding the output extent.
    fn threadgroup(&self) -> MTLSize {
        MTLSize {
            width: THREADEDGE.min(self.cols),
            height: THREADEDGE.min(self.rows),
            depth: 1,
        }
    }
}

/// Adds two equal-length `f32` operands elementwise.
///
/// # Errors
///
/// Returns [`metalerror`] when no device is available, a buffer cannot be
/// allocated, the source fails to compile, the length does not fit `u32`, or
/// the kernel fails to execute.
pub(crate) fn add(cache: &cache, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, metalerror> {
    elementwise2(cache, source::ADDENTRY, lhs, rhs)
}

/// Multiplies two equal-length `f32` operands elementwise.
///
/// # Errors
///
/// See [`add`].
pub(crate) fn mul(cache: &cache, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, metalerror> {
    elementwise2(cache, source::MULENTRY, lhs, rhs)
}

/// Applies the sigmoid linear unit elementwise.
///
/// # Errors
///
/// See [`add`].
pub(crate) fn silu(cache: &cache, input: &[f32]) -> Result<Vec<f32>, metalerror> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let count = asu32(input.len())?;
    let params = [count];
    let mut results = dispatch(
        cache,
        source::ELEMENTWISE,
        source::SILUENTRY,
        &[buf::host(input), buf::out(input.len())],
        &params,
        flatgrid(input.len())?,
        flatthreads(input.len()),
    )?;
    Ok(results.remove(0))
}

/// Runs the two-operand elementwise kernel named `entry` in [`ELEMENTWISE`].
fn elementwise2(
    cache: &cache,
    entry: &'static str,
    lhs: &[f32],
    rhs: &[f32],
) -> Result<Vec<f32>, metalerror> {
    if lhs.is_empty() {
        return Ok(Vec::new());
    }
    let count = asu32(lhs.len())?;
    let params = [count];
    let mut results = dispatch(
        cache,
        source::ELEMENTWISE,
        entry,
        &[buf::host(lhs), buf::host(rhs), buf::out(lhs.len())],
        &params,
        flatgrid(lhs.len())?,
        flatthreads(lhs.len()),
    )?;
    Ok(results.remove(0))
}

/// Numerically stable softmax over the last axis.
///
/// `input` is read as `[rows, width]`. The row maximum is subtracted before
/// exponentiating, matching the scalar reference, and a row whose shifted
/// exponents sum to zero keeps the shifted exponential it already wrote rather
/// than dividing by zero. That branch is unreachable for a finite row, because
/// the maximum's own shifted exponential is one; a row of negative infinity is
/// not zeroed either, because the shift itself is undefined and yields NaN.
///
/// # Errors
///
/// Returns [`metalerror`] when no device is available, a buffer cannot be
/// allocated, the source fails to compile, an extent does not fit `u32`, or the
/// kernel fails to execute.
pub(crate) fn softmax(
    cache: &cache,
    input: &[f32],
    rows: usize,
    width: usize,
) -> Result<Vec<f32>, metalerror> {
    if rows == 0 {
        return Ok(Vec::new());
    }
    let total = rows.checked_mul(width).ok_or(metalerror::sizeoverflow)?;
    let params = [asu32(width)?];
    let mut results = dispatch(
        cache,
        source::SOFTMAXMSL,
        source::SOFTMAXENTRY,
        &[buf::host(input), buf::out(total)],
        &params,
        rowgrid(rows, source::SOFTMAXTILE)?,
        fullthreads(source::SOFTMAXTILE),
    )?;
    Ok(results.remove(0))
}

/// RMS-normalises `rows` rows of `width` elements and scales them by `weight`.
///
/// # Errors
///
/// Returns [`metalerror`] when no device is available, a buffer cannot be
/// allocated, the source fails to compile, an extent does not fit `u32`, or the
/// kernel fails to execute.
pub(crate) fn rmsnorm(
    cache: &cache,
    input: &[f32],
    weight: &[f32],
    rows: usize,
    width: usize,
    eps: f32,
) -> Result<Vec<f32>, metalerror> {
    if rows == 0 {
        return Ok(Vec::new());
    }
    let total = rows.checked_mul(width).ok_or(metalerror::sizeoverflow)?;
    let params = [asu32(width)?, eps.to_bits()];
    let mut results = dispatch(
        cache,
        source::RMSNORMMSL,
        source::RMSNORMENTRY,
        &[buf::host(input), buf::host(weight), buf::out(total)],
        &params,
        rowgrid(rows, source::RMSNORMTILE)?,
        fullthreads(source::RMSNORMTILE),
    )?;
    Ok(results.remove(0))
}

/// Rotates every `headdim` slice of `rows` rows by its position angle.
///
/// # Errors
///
/// Returns [`metalerror`] when no device is available, a buffer cannot be
/// allocated, the source fails to compile, an extent does not fit `u32`, or the
/// kernel fails to execute.
pub(crate) fn rope(
    cache: &cache,
    input: &[f32],
    rows: usize,
    seqaxis: usize,
    heads: usize,
    headdim: usize,
    base: f32,
) -> Result<Vec<f32>, metalerror> {
    if rows == 0 {
        return Ok(Vec::new());
    }
    let pairs = rows
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(headdim / 2))
        .ok_or(metalerror::sizeoverflow)?;
    let params = [
        asu32(rows)?,
        asu32(seqaxis)?,
        asu32(heads)?,
        asu32(headdim)?,
        base.to_bits(),
    ];
    let mut results = dispatch(
        cache,
        source::ROPEMSL,
        source::ROPEENTRY,
        &[buf::host(input), buf::out(input.len())],
        &params,
        flatgrid(pairs)?,
        flatthreads(pairs),
    )?;
    Ok(results.remove(0))
}

/// Rotates one absolute-position row with the dedicated single-row kernel.
///
/// Unlike [`rope`] on a padded `[position + 1, width]` operand, the kernel
/// takes the position as a scalar, so only the row itself is uploaded and
/// rotated: no padded operand is materialized, transferred, or walked.
pub(crate) fn rope_at(
    cache: &cache,
    input: &[f32],
    position: usize,
    heads: usize,
    headdim: usize,
    base: f32,
) -> Result<Vec<f32>, metalerror> {
    let total = heads
        .checked_mul(headdim)
        .ok_or(metalerror::sizeoverflow)?;
    let pairs = total / 2;
    let params = [
        asu32(heads)?,
        asu32(headdim)?,
        asu32(position)?,
        base.to_bits(),
    ];
    let mut results = dispatch(
        cache,
        source::ROPEATMSL,
        source::ROPEATENTRY,
        &[buf::host(input), buf::out(total)],
        &params,
        flatgrid(pairs)?,
        flatthreads(pairs),
    )?;
    Ok(results.remove(0))
}

/// Gathers `indices.len()` rows of `width` elements from `table`.
///
/// The indices arrive already validated against the table's row count; the
/// kernel itself performs no range check, which is why the conversion to the
/// device's `ulong` happens here rather than at the call site.
///
/// # Errors
///
/// Returns [`metalerror`] when no device is available, a buffer cannot be
/// allocated, the source fails to compile, an extent does not fit `u32`, or the
/// kernel fails to execute.
pub(crate) fn embedding(
    cache: &cache,
    table: &[f32],
    indices: &[usize],
    width: usize,
) -> Result<Vec<f32>, metalerror> {
    let rows = indices.len();
    if rows == 0 {
        return Ok(Vec::new());
    }
    let total = rows.checked_mul(width).ok_or(metalerror::sizeoverflow)?;
    let devindices: Vec<u64> = indices.iter().map(|&index| index as u64).collect();
    let params = [rows as u64, width as u64];
    let mut results = dispatch(
        cache,
        source::EMBEDDINGMSL,
        source::EMBEDDINGENTRY,
        &[buf::host(table), buf::hostu64(&devindices), buf::out(total)],
        &params,
        flatgrid(total)?,
        flatthreads(total),
    )?;
    Ok(results.remove(0))
}

/// Applies grouped-query scaled dot-product attention to `geometry.seq`
/// positions.
///
/// `query` is `[seq, heads * headdim]` and `key`/`value` are
/// `[seq, kvheads * headdim]`. The extents come from the validated
/// [`attentionshape`], so this function revalidates nothing. The kernel keeps
/// its entire accumulator in threadgroup memory, so no sequence-length cap is
/// needed; `geometry` already rejected a head dimension beyond
/// [`METALMAXHEADDIM`](crate::METALMAXHEADDIM).
///
/// # Errors
///
/// Returns [`metalerror`] when no device is available, a buffer cannot be
/// allocated, the source fails to compile, an extent does not fit `u32`, or the
/// kernel fails to execute.
pub(crate) fn attention(
    cache: &cache,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    geometry: &attentionshape,
    causal: bool,
) -> Result<Vec<f32>, metalerror> {
    let qseq = geometry.qseq;
    if qseq == 0 {
        return Ok(Vec::new());
    }
    let queryelements = qseq
        .checked_mul(geometry.heads)
        .and_then(|rows| rows.checked_mul(geometry.headdim))
        .ok_or(metalerror::sizeoverflow)?;
    let groups = qseq
        .checked_mul(geometry.heads)
        .ok_or(metalerror::sizeoverflow)?;
    let params = [
        asu32(qseq)?,
        asu32(geometry.kvseq)?,
        asu32(geometry.heads)?,
        asu32(geometry.kvheads)?,
        asu32(geometry.headdim)?,
        u32::from(causal),
    ];
    let mut results = dispatch(
        cache,
        source::ATTENTIONMSL,
        source::ATTENTIONENTRY,
        &[
            buf::host(query),
            buf::host(key),
            buf::host(value),
            buf::out(queryelements),
        ],
        &params,
        rowgrid(groups, source::ATTENTIONLANES)?,
        fullthreads(source::ATTENTIONLANES),
    )?;
    Ok(results.remove(0))
}

/// Runs the serial per-channel scan kernel, returning the `[n, d]` output and
/// the `[d, N]` final state.
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
) -> Result<(Vec<f32>, Vec<f32>), metalerror> {
    if n == 0 {
        return Ok((Vec::new(), init.to_vec()));
    }
    // The parallel three-stage kernels keep one channel's state in a
    // compile-time-sized thread-local array; a wider state falls back to the
    // serial per-channel kernel (bit-for-bit reference).
    if statewidth > source::SCANMAXN {
        return scan_serial(cache, input, init, a, b, c, skip, n, d, statewidth);
    }
    let device = cache.device()?;
    // The inter-block prefix pass writes the final state into a separate
    // buffer so the uploaded `init` stays untouched for the apply pass.
    let statebytes = init
        .len()
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(metalerror::sizeoverflow)?;
    let dstateout = alloc(&device, statebytes)?;
    let outbytes = n
        .checked_mul(d)
        .and_then(|elements| elements.checked_mul(core::mem::size_of::<f32>()))
        .ok_or(metalerror::sizeoverflow)?;
    let dout = alloc(&device, outbytes)?;
    let blocks = n.div_ceil(source::SCANCHUNK);
    let dntotal = d.checked_mul(statewidth).ok_or(metalerror::sizeoverflow)?;
    let pairbytes = |count: usize| {
        count
            .checked_mul(2)
            .and_then(|pairs| pairs.checked_mul(dntotal))
            .and_then(|elements| elements.checked_mul(core::mem::size_of::<f32>()))
            .ok_or(metalerror::sizeoverflow)
    };
    let dtotal = alloc(&device, pairbytes(blocks)?)?;
    let prefixcount = blocks.checked_add(1).ok_or(metalerror::sizeoverflow)?;
    let dprefix = alloc(&device, pairbytes(prefixcount)?)?;
    let params = [
        asu32(n)?,
        asu32(d)?,
        asu32(statewidth)?,
        asu32(source::SCANCHUNK)?,
        asu32(blocks)?,
    ];
    // Pass 1: per-block totals. The block buffers are uploaded once and passed
    // through the three dispatches by device reference; only `dstateout` and
    // the final `out` are downloaded.
    dispatch(
        cache,
        source::SCANPARALLELMSL,
        source::SCANAGGENTRY,
        &[
            buf::host(input),
            buf::host(a),
            buf::host(b),
            buf::host(c),
            buf::host(skip),
            buf::host(init),
            buf::device(&dtotal),
        ],
        &params,
        scangrid(blocks, d)?,
        scantiles(d),
    )?;
    // Pass 2: inter-block prefix plus the whole-sequence final state, written
    // into the caller-held `dstateout` and read back here.
    let prefixthreads = (d * statewidth).min(1024);
    let mut state = dispatch(
        cache,
        source::SCANPARALLELMSL,
        source::SCANPREFIXENTRY,
        &[
            buf::device(&dtotal),
            buf::host(init),
            buf::readback(&dstateout, d * statewidth),
            buf::device(&dprefix),
        ],
        &params,
        flatgrid(prefixthreads)?,
        flatthreads(prefixthreads),
    )?;
    // Pass 3: per-block apply writing `out`; the final state was computed by
    // Pass 2 into `dstateout`.
    let mut results = dispatch(
        cache,
        source::SCANPARALLELMSL,
        source::SCANAPPLYENTRY,
        &[
            buf::host(input),
            buf::host(a),
            buf::host(b),
            buf::host(c),
            buf::host(skip),
            buf::host(init),
            buf::device(&dprefix),
            buf::readback(&dout, n * d),
        ],
        &params,
        scangrid(blocks, d)?,
        scantiles(d),
    )?;
    let out = results.remove(0);
    Ok((out, state.remove(0)))
}

/// Runs the serial per-channel scan kernel, returning the `[n, d]` output and
/// the `[d, N]` final state.
///
/// This is the bit-for-bit reference path and the fallback for state widths
/// above [`source::SCANMAXN`].
#[allow(clippy::too_many_arguments)]
fn scan_serial(
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
) -> Result<(Vec<f32>, Vec<f32>), metalerror> {
    let device = cache.device()?;
    // The output and the final state both stay on the device for one dispatch
    // and are read back together after it completes.
    let outbytes = n
        .checked_mul(d)
        .and_then(|elements| elements.checked_mul(core::mem::size_of::<f32>()))
        .ok_or(metalerror::sizeoverflow)?;
    let dout = alloc(&device, outbytes)?;
    let statebytes = init
        .len()
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(metalerror::sizeoverflow)?;
    let dstate = alloc(&device, statebytes)?;
    let params = [asu32(n)?, asu32(d)?, asu32(statewidth)?];
    let mut results = dispatch(
        cache,
        source::SCANMSL,
        source::SCANENTRY,
        &[
            buf::host(input),
            buf::host(a),
            buf::host(b),
            buf::host(c),
            buf::host(skip),
            buf::host(init),
            buf::readback(&dstate, d * statewidth),
            buf::readback(&dout, n * d),
        ],
        &params,
        flatgrid(d)?,
        flatthreads(d),
    )?;
    let state = results.remove(0);
    let out = results.remove(0);
    Ok((out, state))
}

/// Builds the two-dimensional scan thread grid: `blocks × d` threads, one per
/// sequence block along the x axis and one per channel along the y axis.
/// `dispatchThreads` treats this as the total thread count and partitions it
/// into threadgroups of [`source::FLATTILE`] threads automatically.
fn scangrid(blocks: usize, d: usize) -> Result<MTLSize, metalerror> {
    asu32(blocks)?;
    Ok(MTLSize {
        width: blocks,
        height: d,
        depth: 1,
    })
}

/// The threadgroup shape of the scan block kernels: one thread per channel,
/// `FLATTILE` threads per group.
fn scantiles(d: usize) -> MTLSize {
    MTLSize {
        width: 1,
        height: source::FLATTILE.min(d),
        depth: 1,
    }
}

/// Compiles `source` and builds the compute pipeline for `entry`.
fn compile(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
    entry: &str,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, metalerror> {
    let text = NSString::from_str(source);
    let library: Retained<ProtocolObject<dyn MTLLibrary>> = device
        .newLibraryWithSource_options_error(&text, None)
        .map_err(|error| metalerror::compile {
            kernel: entry.to_string(),
            message: describe(&error),
        })?;
    let name = NSString::from_str(entry);
    let function = library
        .newFunctionWithName(&name)
        .ok_or_else(|| metalerror::compile {
            kernel: entry.to_string(),
            message: format!("entry point `{entry}` is missing from the compiled library"),
        })?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| metalerror::pipeline {
            kernel: entry.to_string(),
            message: describe(&error),
        })
}

/// One buffer slot of a dispatch, in kernel binding order.
///
/// The three variants cover every kernel the crate ships: `host` uploads a
/// host slice into a freshly allocated buffer, `device` passes through a
/// buffer the caller allocated and still owns (the inter-pass buffers of the
/// three-stage scan), and `out` allocates a buffer whose contents are copied
/// back to the host after the dispatch completes.
pub(crate) enum buf<'a> {
    /// Upload a host f32 slice into a freshly allocated device buffer.
    host(&'a [f32]),
    /// Upload a host u64 slice into a freshly allocated device buffer — the
    /// embedding gather's index buffer, whose elements are row IDs.
    hostu64(&'a [u64]),
    /// Pass through a caller-owned device buffer.
    device(&'a ProtocolObject<dyn MTLBuffer>),
    /// Allocate a device buffer and copy its contents back after completion.
    out(usize),
    /// Pass through a caller-owned buffer and copy `count` f32 elements back
    /// after completion — the final state buffer of the three-stage scan,
    /// allocated once and written by the prefix pass, read by the apply pass,
    /// and downloaded by the caller.
    readback(&'a ProtocolObject<dyn MTLBuffer>, usize),
}

/// Compiles, binds, and runs one kernel, then waits and copies the outputs
/// back — the backend's single driver-handoff.
///
/// Every platform interaction this crate performs (uploading an operand,
/// binding buffers, copying the scalar parameters, launching the kernel,
/// waiting for completion, downloading the result) happens inside **one**
/// `unsafe` block, the crate's registered unsafe site. Everything around it —
/// geometry, budget checks, buffer allocation, the pipeline cache, the reused
/// command queue — is safe Rust, so the unsafe surface is exactly the handoff
/// itself, matching the `drive` design the MUSA backend uses
/// (docs/musa-backend-plan.md §5.4).
///
/// The caller establishes the SAFETY preconditions that the block documents:
/// `buffers` are bound to indices `0..buffers.len()` in order and `params` is
/// copied into the next index, matching the kernel's `[[buffer(i)]]`
/// attributes; `grid`/`threads` are validated by the caller's geometry helper;
/// every element index the kernels derive is bounds-tested by the kernels
/// themselves.
///
/// # Errors
///
/// Returns [`metalerror`] when compilation or pipeline creation fails, or when
/// the device, queue, or command buffer declines to hand out the resources the
/// dispatch needs. Output buffers are reserved before their contents are read,
/// so a host allocation failure surfaces as [`metalerror::allocation`] rather
/// than aborting.
fn dispatch<P>(
    cache: &cache,
    source: &'static str,
    entry: &'static str,
    buffers: &[buf<'_>],
    params: &P,
    grid: MTLSize,
    threads: MTLSize,
) -> Result<Vec<Vec<f32>>, metalerror> {
    // The device, the pipeline, and the queue all come from the same cache, so
    // the pipeline cannot be built for one device and used on another. The
    // queue is opened once and reused; only the command buffer is per-dispatch
    // (Metal command buffers are single-use by contract).
    let device = cache.device()?;
    let pipeline = cache.pipeline(source, entry)?;
    let queue = cache.queue()?;
    let command = queue.commandBuffer().ok_or_else(|| metalerror::execution {
        message: "the queue returned no command buffer".to_string(),
    })?;
    let encoder = command
        .computeCommandEncoder()
        .ok_or_else(|| metalerror::execution {
            message: "the command buffer returned no compute encoder".to_string(),
        })?;
    encoder.setComputePipelineState(&pipeline);

    // Allocate the device buffers this dispatch owns in one pass, then line up
    // the bound references in a second pass. Splitting the passes keeps the
    // borrows disjoint: nothing borrows `owned` while it is still being pushed
    // to. `owned` lives for the whole function, so no buffer is freed while a
    // kernel can still touch it.
    let mut allocbytes: Vec<usize> = Vec::with_capacity(buffers.len());
    let mut uploads: Vec<(usize, &[f32])> = Vec::new();
    let mut uploads64: Vec<(usize, &[u64])> = Vec::new();
    let mut outputs: Vec<(usize, usize)> = Vec::new();
    for (index, slot) in buffers.iter().enumerate() {
        match *slot {
            buf::host(data) => {
                allocbytes.push(
                    data.len()
                        .checked_mul(core::mem::size_of::<f32>())
                        .ok_or(metalerror::sizeoverflow)?,
                );
                uploads.push((index, data));
            }
            buf::hostu64(data) => {
                allocbytes.push(
                    data.len()
                        .checked_mul(core::mem::size_of::<u64>())
                        .ok_or(metalerror::sizeoverflow)?,
                );
                uploads64.push((index, data));
            }
            buf::out(count) => {
                let bytes = count
                    .checked_mul(core::mem::size_of::<f32>())
                    .ok_or(metalerror::sizeoverflow)?;
                allocbytes.push(bytes);
                outputs.push((index, count));
            }
            buf::readback(_, count) => {
                outputs.push((index, count));
            }
            buf::device(_) => {}
        }
    }
    let mut owned: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> =
        Vec::with_capacity(allocbytes.len());
    for bytes in allocbytes {
        owned.push(alloc(&device, bytes)?);
    }
    let mut refs: Vec<&ProtocolObject<dyn MTLBuffer>> = Vec::with_capacity(buffers.len());
    let mut owncursor = 0;
    for slot in buffers.iter() {
        match *slot {
            buf::host(_) | buf::hostu64(_) | buf::out(_) => {
                refs.push(&owned[owncursor]);
                owncursor += 1;
            }
            buf::device(passedbuf) | buf::readback(passedbuf, _) => refs.push(passedbuf),
        }
    }

    let mut results: Vec<Vec<f32>> = Vec::with_capacity(outputs.len());
    // SAFETY: `refs` is exactly the bound buffer list and every entry is a
    // live device buffer — caller-owned (device) or allocated above with the
    // byte length the kernel reads or writes — so binding indices
    // 0..refs.len() to the pipeline's `[[buffer(i)]]` attributes matches the
    // MSL signature, and the kernels bounds-test every element index they
    // derive. `params` is a live value whose address is valid for the call,
    // and `setBytes_length_atIndex` copies rather than retains it. The
    // downloaded pointers cover exactly the elements reserved in `results`,
    // and `waitUntilCompleted` (invoked inside the block) orders the reads
    // after the kernel writes. No index is dereferenced here.
    unsafe {
        // Upload: copy each host operand into its shared-storage buffer. The
        // buffers were allocated with exactly `size_of_val(data)` bytes, so
        // the copies are in bounds and the two pointers never overlap.
        for (index, data) in &uploads {
            let target = refs[*index];
            core::ptr::copy_nonoverlapping(
                data.as_ptr().cast::<u8>(),
                target.contents().as_ptr().cast::<u8>(),
                core::mem::size_of_val(&data[..]),
            );
        }
        for (index, data) in &uploads64 {
            let target = refs[*index];
            core::ptr::copy_nonoverlapping(
                data.as_ptr().cast::<u8>(),
                target.contents().as_ptr().cast::<u8>(),
                core::mem::size_of_val(&data[..]),
            );
        }
        // Bind: operand buffers to buffer indices 0..refs.len().
        for (index, buffer) in refs.iter().enumerate() {
            encoder.setBuffer_offset_atIndex(Some(*buffer), 0, index);
        }
        // Scalars: the kernel's parameter buffer at the next index.
        encoder.setBytes_length_atIndex(
            core::ptr::NonNull::from(params).cast(),
            core::mem::size_of_val(params),
            refs.len(),
        );
        // Launch, close the encoder, commit, and synchronize: the
        // grid/threadgroup shape was validated by the caller's geometry
        // helper, and completion orders the reads below. `endEncoding`,
        // `commit`, and `waitUntilCompleted` are safe methods invoked inside
        // the block so that the whole driver handoff stays one unsafe site.
        encoder.dispatchThreads_threadsPerThreadgroup(grid, threads);
        encoder.endEncoding();
        command.commit();
        command.waitUntilCompleted();
        // Download: copy each output buffer into a host vector reserved
        // beforehand, so an allocation failure surfaces as an error here.
        for (index, count) in &outputs {
            let mut out = Vec::new();
            out.try_reserve_exact(*count)
                .map_err(|_| metalerror::allocation)?;
            out.resize(*count, 0.0_f32);
            core::ptr::copy_nonoverlapping(
                refs[*index].contents().as_ptr().cast::<f32>(),
                out.as_mut_ptr(),
                *count,
            );
            results.push(out);
        }
    }
    Ok(results)
}

/// Allocates a shared-storage buffer of `bytes` bytes.
fn alloc(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, metalerror> {
    device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .ok_or(metalerror::allocation)
}

/// Converts a validated extent into the kernel's `uint` parameter.
fn asu32(value: usize) -> Result<u32, metalerror> {
    u32::try_from(value).map_err(|_| metalerror::sizeoverflow)
}

/// Builds the one-dimensional grid covering `count` elements.
///
/// The `uint` thread index a flat kernel reads cannot address more than
/// `u32::MAX`, so a larger count is rejected as a size overflow rather than
/// silently truncated.
fn flatgrid(count: usize) -> Result<MTLSize, metalerror> {
    asu32(count)?;
    Ok(MTLSize {
        width: count,
        height: 1,
        depth: 1,
    })
}

/// Builds a flat kernel's threadgroup, never exceeding `count`.
fn flatthreads(count: usize) -> MTLSize {
    MTLSize {
        width: source::FLATTILE.min(count),
        height: 1,
        depth: 1,
    }
}

/// Builds a reduction kernel's grid: one full `tile`-wide threadgroup per row.
///
/// Reduction kernels take `threadgroup_position_in_grid` as the row index and
/// require a full power-of-two threadgroup, so the grid covers `rows * tile`
/// threads with exactly `rows` threadgroups.
fn rowgrid(rows: usize, tile: usize) -> Result<MTLSize, metalerror> {
    let width = rows.checked_mul(tile).ok_or(metalerror::sizeoverflow)?;
    Ok(MTLSize {
        width,
        height: 1,
        depth: 1,
    })
}

/// Builds a reduction kernel's full threadgroup.
fn fullthreads(tile: usize) -> MTLSize {
    MTLSize {
        width: tile,
        height: 1,
        depth: 1,
    }
}

/// Renders an `NSError` as a diagnostic string.
fn describe(error: &NSError) -> String {
    error.localizedDescription().to_string()
}

/// The pipeline key and thread geometry of one dispatch, packed so a launch
/// reads as one argument rather than four.
struct dispatchspec {
    source: &'static str,
    entry: &'static str,
    grid: MTLSize,
    threads: MTLSize,
}

/// One buffer slot of an asynchronous session dispatch, in kernel binding
/// order. The last slot MUST be [`sbuf::out`], the output the slot records.
enum sbuf<'a> {
    /// Upload a host u64 slice into a freshly allocated device buffer — the
    /// embedding gather's index buffer, whose elements are row IDs. Host f32
    /// operands on the session path go through [`session::upload`] instead,
    /// which defers the device copy to the next dispatch's unsafe block.
    hostu64(&'a [u64]),
    /// Pass through a caller-owned device buffer.
    device(&'a ProtocolObject<dyn MTLBuffer>),
    /// Allocate the output buffer (not read back here; [`session::apply`] does).
    out(usize),
}

/// A deferred host upload queued by [`session::upload`]: the device buffer the
/// next dispatch's unsafe block copies into, plus the host data it copies.
type pendingupload = (Retained<ProtocolObject<dyn MTLBuffer>>, Vec<f32>);

/// One accumulative submission: a command buffer and compute encoder that many
/// operators share, plus the output slot of every recorded step.
///
/// A `session` is the Metal side of the accumulative `api::submission` contract
/// (docs/design.md §21.13). It owns its command buffer and encoder, so its
/// lifetime bounds the single `commit` + `waitUntilCompleted` that all recorded
/// dispatches share. It owns no model weights and carries no request-scoped
/// state beyond its own buffers; the compiled pipelines still come from the
/// executor-owned [`cache`], which holds no operand or result.
pub(crate) struct session {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    encoder: Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
    outputs: Vec<slot>,
    /// Host operands uploaded with [`session::upload`] whose device copies are
    /// flushed into their buffers by the next dispatch, inside that dispatch's
    /// single unsafe block. Holding the data here keeps the upload path free of
    /// its own unsafe site without pinning a copy lifetime on the caller.
    ///
    /// `RefCell` is the submission's only interior mutability: the API trait
    /// encodes operators as `&self` (operand resolution never mutates the
    /// submission's recorded shape list), so the pending queue mutates through
    /// a runtime borrow. A submission is request-scoped and never shared
    /// across threads, so the borrow is uncontended by construction.
    pending: RefCell<Vec<pendingupload>>,
    /// True once `endEncoding` has been called, so `Drop` ends an encoder that
    /// an error path dropped mid-submission exactly once.
    ended: bool,
}

/// The recorded output of one step: a device buffer to download, no elements,
/// or a host-side run of zeros.
enum slot {
    buffer(Retained<ProtocolObject<dyn MTLBuffer>>, usize),
    empty,
    zeros(usize),
}

impl session {
    /// Opens a device, queue, command buffer, and encoder for one submission.
    ///
    /// The command queue comes from the executor-owned [`cache`], so a
    /// submission pays `newCommandQueue` at most once per cache, not once per
    /// submission; only the command buffer is created here, because Metal
    /// command buffers are single-use by contract.
    pub(crate) fn new(cache: &cache) -> Result<Self, metalerror> {
        let device = cache.device()?;
        let queue = cache.queue()?;
        let command = queue.commandBuffer().ok_or_else(|| metalerror::execution {
            message: "the queue returned no command buffer".to_string(),
        })?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| metalerror::execution {
                message: "the command buffer returned no compute encoder".to_string(),
            })?;
        Ok(Self {
            device,
            command,
            encoder,
            outputs: Vec::new(),
            pending: RefCell::new(Vec::new()),
            ended: false,
        })
    }

    /// Uploads host data into a fresh device buffer.
    ///
    /// The device copy is deferred to the next dispatch, whose unsafe block
    /// flushes the pending operands — so the upload path itself has no unsafe
    /// site. The returned buffer is owned by the caller; the encoder records
    /// only a reference to it, so an input buffer may be dropped after the
    /// dispatch that reads it has been encoded.
    pub(crate) fn upload(
        &self,
        data: &[f32],
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, metalerror> {
        let buffer = alloc(&self.device, core::mem::size_of_val(data))?;
        self.pending
            .borrow_mut()
            .push((buffer.clone(), data.to_vec()));
        Ok(buffer)
    }

    /// Encodes one dispatch into the shared encoder and records its output slot.
    ///
    /// This is the asynchronous driver handoff: the whole encoding — uploading
    /// host operands, binding buffers, copying the scalar parameters, launching
    /// the kernel — happens inside one `unsafe` block, the session path's
    /// registered unsafe site. No synchronization happens here; the shared
    /// `commit` + `wait` lives in [`session::apply`], whose download block is
    /// that path's second and last unsafe site.
    ///
    /// `buffers` bind to indices `0..buffers.len()` in order and `params` to
    /// the next index; the **last** slot MUST be the output buffer, whose slot
    /// is recorded for [`session::apply`] to download. Every kernel this crate
    /// encodes follows that layout.
    fn dispatch<T>(
        &mut self,
        cache: &cache,
        spec: &dispatchspec,
        buffers: &[sbuf<'_>],
        params: &T,
        outlen: usize,
    ) -> Result<usize, metalerror> {
        let pipeline = cache.pipeline(spec.source, spec.entry)?;
        self.encoder.setComputePipelineState(&pipeline);

        // Allocate the buffers this dispatch owns in one pass, then line up
        // the bound references in a second pass, keeping the borrows disjoint.
        let mut allocbytes: Vec<usize> = Vec::with_capacity(buffers.len());
        let mut uploads64: Vec<(usize, &[u64])> = Vec::new();
        for (index, slot) in buffers.iter().enumerate() {
            match *slot {
                sbuf::hostu64(data) => {
                    allocbytes.push(
                        data.len()
                            .checked_mul(core::mem::size_of::<u64>())
                            .ok_or(metalerror::sizeoverflow)?,
                    );
                    uploads64.push((index, data));
                }
                sbuf::out(count) => {
                    let bytes = count
                        .checked_mul(core::mem::size_of::<f32>())
                        .ok_or(metalerror::sizeoverflow)?;
                    allocbytes.push(bytes);
                }
                sbuf::device(_) => {}
            }
        }
        let mut owned: Vec<Retained<ProtocolObject<dyn MTLBuffer>>> =
            Vec::with_capacity(allocbytes.len());
        for bytes in allocbytes {
            owned.push(alloc(&self.device, bytes)?);
        }
        let mut refs: Vec<&ProtocolObject<dyn MTLBuffer>> = Vec::with_capacity(buffers.len());
        let mut owncursor = 0;
        let mut out: Option<Retained<ProtocolObject<dyn MTLBuffer>>> = None;
        for slot in buffers.iter() {
            match *slot {
                sbuf::hostu64(_) | sbuf::out(_) => {
                    let buffer = &owned[owncursor];
                    if matches!(*slot, sbuf::out(_)) {
                        out = Some(buffer.clone());
                    }
                    refs.push(buffer);
                    owncursor += 1;
                }
                sbuf::device(passed) => refs.push(passed),
            }
        }
        let out = out.ok_or_else(|| metalerror::execution {
            message: "a session dispatch has no output slot".to_string(),
        })?;
        let index = self.outputs.len();
        self.outputs.push(slot::buffer(out, outlen));

        // SAFETY: `refs` is exactly the bound buffer list and every entry is a
        // live device buffer — caller-owned (device) or allocated above with
        // the byte length the kernel reads or writes — so binding indices
        // 0..refs.len() to the pipeline's `[[buffer(i)]]` attributes matches
        // the MSL signature, and the kernels bounds-test every element index
        // they derive. `params` is a live value whose address is valid for the
        // call, and `setBytes_length_atIndex` copies rather than retains it.
        // The upload targets (this dispatch's and the pending ones flushed
        // below) were allocated with exactly the byte length of their data, so
        // the copies are in bounds and never overlap. No index is dereferenced
        // here.
        unsafe {
            {
                let mut pending = self.pending.borrow_mut();
                for (buffer, data) in pending.iter() {
                    core::ptr::copy_nonoverlapping(
                        data.as_ptr().cast::<u8>(),
                        buffer.contents().as_ptr().cast::<u8>(),
                        core::mem::size_of_val(data.as_slice()),
                    );
                }
                pending.clear();
            }
            for (index, data) in &uploads64 {
                let target = refs[*index];
                core::ptr::copy_nonoverlapping(
                    data.as_ptr().cast::<u8>(),
                    target.contents().as_ptr().cast::<u8>(),
                    core::mem::size_of_val(&data[..]),
                );
            }
            for (index, buffer) in refs.iter().enumerate() {
                self.encoder
                    .setBuffer_offset_atIndex(Some(*buffer), 0, index);
            }
            self.encoder.setBytes_length_atIndex(
                core::ptr::NonNull::from(params).cast(),
                core::mem::size_of_val(params),
                refs.len(),
            );
            self.encoder
                .dispatchThreads_threadsPerThreadgroup(spec.grid, spec.threads);
        }
        Ok(index)
    }

    /// Records a zero-element output without touching the device.
    fn recordempty(&mut self) -> usize {
        let index = self.outputs.len();
        self.outputs.push(slot::empty);
        index
    }

    /// Records an all-zero output of `count` elements without touching the device.
    fn recordzeros(&mut self, count: usize) -> usize {
        let index = self.outputs.len();
        self.outputs.push(slot::zeros(count));
        index
    }

    /// Returns the device buffer of slot `index`, when that slot holds one.
    pub(crate) fn getbuffer(
        &self,
        index: usize,
    ) -> Option<&Retained<ProtocolObject<dyn MTLBuffer>>> {
        match self.outputs.get(index) {
            Some(slot::buffer(buffer, _)) => Some(buffer),
            _ => None,
        }
    }

    /// Returns the host-side data a non-buffer slot names, when `index` is one.
    ///
    /// An empty slot yields an empty vector and a zeros slot its zero run; a
    /// buffer slot yields `None`. This lets a later step read an empty or zero
    /// result back off the host rather than off the device.
    pub(crate) fn gethostdata(&self, index: usize) -> Option<Vec<f32>> {
        match self.outputs.get(index) {
            Some(slot::empty) => Some(Vec::new()),
            Some(slot::zeros(count)) => Some(vec![0.0_f32; *count]),
            _ => None,
        }
    }

    /// Ends encoding, commits, waits, then downloads every recorded output.
    ///
    /// The returned vectors are in slot order, so a slot index obtained from a
    /// `dispatch` / `recordempty` / `recordzeros` call indexes into this result.
    pub(crate) fn apply(mut self) -> Result<Vec<Vec<f32>>, metalerror> {
        self.encoder.endEncoding();
        self.ended = true;
        self.command.commit();
        // `session` implements `Drop`, so fields cannot be moved out of `self`;
        // `take` leaves an empty vector behind and hands the slots over.
        let outputs = core::mem::take(&mut self.outputs);
        let mut results: Vec<Vec<f32>> = Vec::with_capacity(outputs.len());
        // SAFETY: every `slot::buffer` was allocated with at least `len` f32
        // elements by the dispatch that recorded it, and the `waitUntilCompleted`
        // inside this block orders the reads after every encoded kernel wrote
        // the buffers, so each `contents` pointer yields exactly `len`
        // initialized elements, which the reserved `out` has room for.
        unsafe {
            self.command.waitUntilCompleted();
            for output in outputs {
                match output {
                    slot::buffer(buffer, len) => {
                        let mut out = Vec::new();
                        out.try_reserve_exact(len)
                            .map_err(|_| metalerror::allocation)?;
                        out.resize(len, 0.0_f32);
                        core::ptr::copy_nonoverlapping(
                            buffer.contents().as_ptr().cast::<f32>(),
                            out.as_mut_ptr(),
                            len,
                        );
                        results.push(out);
                    }
                    slot::empty => results.push(Vec::new()),
                    slot::zeros(count) => results.push(vec![0.0_f32; count]),
                }
            }
        }
        Ok(results)
    }
}

impl Drop for session {
    fn drop(&mut self) {
        // An error path can drop the session before `apply`; end the encoder
        // here so Metal never sees a compute encoder released mid-submission.
        if !self.ended {
            self.encoder.endEncoding();
        }
    }
}

/// Encodes a batched `matmul` into `session`, returning its output slot index.
///
/// `lhs` and `rhs` are already on the device; the result buffer is allocated
/// here and its slot recorded. The zero-element and zero-contraction cases are
/// recorded without a dispatch, matching [`matmul`].
pub(crate) fn matmul_into(
    cache: &cache,
    session: &mut session,
    lhs: &ProtocolObject<dyn MTLBuffer>,
    rhs: &ProtocolObject<dyn MTLBuffer>,
    geometry: &matmulshape,
) -> Result<usize, metalerror> {
    let batch = geometry.batch;
    let m = geometry.m;
    let k = geometry.k;
    let n = geometry.n;
    let total = batch
        .checked_mul(m)
        .and_then(|value| value.checked_mul(n))
        .ok_or(metalerror::sizeoverflow)?;
    if m == 0 || n == 0 {
        return Ok(session.recordempty());
    }
    if k == 0 {
        return Ok(session.recordzeros(total));
    }
    let dims = Extents::new(m, k, n)?;
    let buffers = [sbuf::device(lhs), sbuf::device(rhs), sbuf::out(total)];
    let params = [dims.m, dims.k, dims.n];
    let spec = dispatchspec {
        source: source::MATMULMSL,
        entry: source::MATMULENTRY,
        grid: MTLSize {
            width: dims.cols,
            height: dims.rows,
            depth: batch,
        },
        threads: dims.threadgroup(),
    };
    session.dispatch(cache, &spec, &buffers, &params, total)
}

/// Encodes an elementwise `add` of `count` elements, returning the slot index.
pub(crate) fn add_into(
    cache: &cache,
    session: &mut session,
    lhs: &ProtocolObject<dyn MTLBuffer>,
    rhs: &ProtocolObject<dyn MTLBuffer>,
    count: usize,
) -> Result<usize, metalerror> {
    elementwise2_into(cache, session, source::ADDENTRY, lhs, rhs, count)
}

/// Encodes an elementwise `mul` of `count` elements, returning the slot index.
pub(crate) fn mul_into(
    cache: &cache,
    session: &mut session,
    lhs: &ProtocolObject<dyn MTLBuffer>,
    rhs: &ProtocolObject<dyn MTLBuffer>,
    count: usize,
) -> Result<usize, metalerror> {
    elementwise2_into(cache, session, source::MULENTRY, lhs, rhs, count)
}

/// Encodes the two-operand elementwise kernel named `entry`.
fn elementwise2_into(
    cache: &cache,
    session: &mut session,
    entry: &'static str,
    lhs: &ProtocolObject<dyn MTLBuffer>,
    rhs: &ProtocolObject<dyn MTLBuffer>,
    count: usize,
) -> Result<usize, metalerror> {
    if count == 0 {
        return Ok(session.recordempty());
    }
    let buffers = [sbuf::device(lhs), sbuf::device(rhs), sbuf::out(count)];
    let params = [asu32(count)?];
    let spec = dispatchspec {
        source: source::ELEMENTWISE,
        entry,
        grid: flatgrid(count)?,
        threads: flatthreads(count),
    };
    session.dispatch(cache, &spec, &buffers, &params, count)
}

/// Encodes a `silu` of `count` elements, returning the slot index.
pub(crate) fn silu_into(
    cache: &cache,
    session: &mut session,
    input: &ProtocolObject<dyn MTLBuffer>,
    count: usize,
) -> Result<usize, metalerror> {
    if count == 0 {
        return Ok(session.recordempty());
    }
    let buffers = [sbuf::device(input), sbuf::out(count)];
    let params = [asu32(count)?];
    let spec = dispatchspec {
        source: source::ELEMENTWISE,
        entry: source::SILUENTRY,
        grid: flatgrid(count)?,
        threads: flatthreads(count),
    };
    session.dispatch(cache, &spec, &buffers, &params, count)
}

/// Encodes a `softmax` over `rows` rows of `width`, returning the slot index.
pub(crate) fn softmax_into(
    cache: &cache,
    session: &mut session,
    input: &ProtocolObject<dyn MTLBuffer>,
    rows: usize,
    width: usize,
) -> Result<usize, metalerror> {
    if rows == 0 {
        return Ok(session.recordempty());
    }
    let total = rows.checked_mul(width).ok_or(metalerror::sizeoverflow)?;
    let buffers = [sbuf::device(input), sbuf::out(total)];
    let params = [asu32(width)?];
    let spec = dispatchspec {
        source: source::SOFTMAXMSL,
        entry: source::SOFTMAXENTRY,
        grid: rowgrid(rows, source::SOFTMAXTILE)?,
        threads: fullthreads(source::SOFTMAXTILE),
    };
    session.dispatch(cache, &spec, &buffers, &params, total)
}

/// Encodes an RMS normalization, returning the slot index.
pub(crate) fn rmsnorm_into(
    cache: &cache,
    session: &mut session,
    input: &ProtocolObject<dyn MTLBuffer>,
    weight: &ProtocolObject<dyn MTLBuffer>,
    rows: usize,
    width: usize,
    eps: f32,
) -> Result<usize, metalerror> {
    if rows == 0 {
        return Ok(session.recordempty());
    }
    let total = rows.checked_mul(width).ok_or(metalerror::sizeoverflow)?;
    let buffers = [sbuf::device(input), sbuf::device(weight), sbuf::out(total)];
    let params = [asu32(width)?, eps.to_bits()];
    let spec = dispatchspec {
        source: source::RMSNORMMSL,
        entry: source::RMSNORMENTRY,
        grid: rowgrid(rows, source::RMSNORMTILE)?,
        threads: fullthreads(source::RMSNORMTILE),
    };
    session.dispatch(cache, &spec, &buffers, &params, total)
}

/// Encodes rotary position embedding, returning the slot index.
pub(crate) fn rope_into(
    cache: &cache,
    session: &mut session,
    input: &ProtocolObject<dyn MTLBuffer>,
    geometry: &ropeshape,
    base: f32,
) -> Result<usize, metalerror> {
    let rows = geometry.rows;
    let seqaxis = geometry.seqaxis;
    let heads = geometry.heads;
    let headdim = geometry.headdim;
    if rows == 0 {
        return Ok(session.recordempty());
    }
    let pairs = rows
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(headdim / 2))
        .ok_or(metalerror::sizeoverflow)?;
    let total = rows
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(headdim))
        .ok_or(metalerror::sizeoverflow)?;
    let buffers = [sbuf::device(input), sbuf::out(total)];
    let params = [
        asu32(rows)?,
        asu32(seqaxis)?,
        asu32(heads)?,
        asu32(headdim)?,
        base.to_bits(),
    ];
    let spec = dispatchspec {
        source: source::ROPEMSL,
        entry: source::ROPEENTRY,
        grid: flatgrid(pairs)?,
        threads: flatthreads(pairs),
    };
    session.dispatch(cache, &spec, &buffers, &params, total)
}

/// Encodes absolute-position rotary embedding of a single token.
pub(crate) fn rope_at_into(
    cache: &cache,
    session: &mut session,
    input: &ProtocolObject<dyn MTLBuffer>,
    position: usize,
    heads: usize,
    headdim: usize,
    base: f32,
) -> Result<usize, metalerror> {
    let total = heads.checked_mul(headdim).ok_or(metalerror::sizeoverflow)?;
    let buffers = [sbuf::device(input), sbuf::out(total)];
    let params = [
        asu32(heads)?,
        asu32(headdim)?,
        asu32(position)?,
        base.to_bits(),
    ];
    let spec = dispatchspec {
        source: source::ROPEATMSL,
        entry: source::ROPEATENTRY,
        grid: flatgrid(heads * headdim / 2)?,
        threads: flatthreads(heads * headdim / 2),
    };
    session.dispatch(cache, &spec, &buffers, &params, total)
}

/// Encodes an embedding gather, returning the slot index.
pub(crate) fn embedding_into(
    cache: &cache,
    session: &mut session,
    table: &ProtocolObject<dyn MTLBuffer>,
    indices: &[usize],
    width: usize,
) -> Result<usize, metalerror> {
    let rows = indices.len();
    if rows == 0 {
        return Ok(session.recordempty());
    }
    let total = rows.checked_mul(width).ok_or(metalerror::sizeoverflow)?;
    let devindices: Vec<u64> = indices.iter().map(|&index| index as u64).collect();
    let buffers = [
        sbuf::device(table),
        sbuf::hostu64(&devindices),
        sbuf::out(total),
    ];
    let params = [rows as u64, width as u64];
    let spec = dispatchspec {
        source: source::EMBEDDINGMSL,
        entry: source::EMBEDDINGENTRY,
        grid: flatgrid(total)?,
        threads: flatthreads(total),
    };
    session.dispatch(cache, &spec, &buffers, &params, total)
}

/// Encodes grouped-query attention, returning the slot index.
pub(crate) fn attention_into(
    cache: &cache,
    session: &mut session,
    query: &ProtocolObject<dyn MTLBuffer>,
    key: &ProtocolObject<dyn MTLBuffer>,
    value: &ProtocolObject<dyn MTLBuffer>,
    geometry: &attentionshape,
    causal: bool,
) -> Result<usize, metalerror> {
    let qseq = geometry.qseq;
    if qseq == 0 {
        return Ok(session.recordempty());
    }
    let queryelements = qseq
        .checked_mul(geometry.heads)
        .and_then(|rows| rows.checked_mul(geometry.headdim))
        .ok_or(metalerror::sizeoverflow)?;
    let groups = qseq
        .checked_mul(geometry.heads)
        .ok_or(metalerror::sizeoverflow)?;
    let buffers = [
        sbuf::device(query),
        sbuf::device(key),
        sbuf::device(value),
        sbuf::out(queryelements),
    ];
    let params = [
        asu32(qseq)?,
        asu32(geometry.kvseq)?,
        asu32(geometry.heads)?,
        asu32(geometry.kvheads)?,
        asu32(geometry.headdim)?,
        u32::from(causal),
    ];
    let spec = dispatchspec {
        source: source::ATTENTIONMSL,
        entry: source::ATTENTIONENTRY,
        grid: rowgrid(groups, source::ATTENTIONLANES)?,
        threads: fullthreads(source::ATTENTIONLANES),
    };
    session.dispatch(cache, &spec, &buffers, &params, queryelements)
}

/// A cached and a new device buffer of an incremental attention operand.
pub(crate) struct bufferpair<'a> {
    /// The accumulated cached operand.
    pub(crate) cached: &'a ProtocolObject<dyn MTLBuffer>,
    /// The new operand.
    pub(crate) new: &'a ProtocolObject<dyn MTLBuffer>,
}

/// Encodes incremental-decode attention: one query token attends over a cached
/// key/value buffer plus one new row, without materialising the concatenation.
pub(crate) fn attention_cached_into(
    cache: &cache,
    session: &mut session,
    query: &ProtocolObject<dyn MTLBuffer>,
    key: bufferpair<'_>,
    value: bufferpair<'_>,
    geometry: &attentionshape,
    causal: bool,
) -> Result<usize, metalerror> {
    let kvseq = geometry.kvseq;
    if kvseq == 0 {
        return Ok(session.recordempty());
    }
    let queryelements = geometry
        .qseq
        .checked_mul(geometry.heads)
        .and_then(|rows| rows.checked_mul(geometry.headdim))
        .ok_or(metalerror::sizeoverflow)?;
    let buffers = [
        sbuf::device(query),
        sbuf::device(key.cached),
        sbuf::device(key.new),
        sbuf::device(value.cached),
        sbuf::device(value.new),
        sbuf::out(queryelements),
    ];
    let params = [
        asu32(kvseq)?,
        asu32(geometry.heads)?,
        asu32(geometry.kvheads)?,
        asu32(geometry.headdim)?,
        u32::from(causal),
    ];
    let spec = dispatchspec {
        source: source::ATTENTIONCACHEDMSL,
        entry: source::ATTENTIONCACHEDENTRY,
        grid: rowgrid(geometry.heads, source::ATTENTIONLANES)?,
        threads: fullthreads(source::ATTENTIONLANES),
    };
    session.dispatch(cache, &spec, &buffers, &params, queryelements)
}
