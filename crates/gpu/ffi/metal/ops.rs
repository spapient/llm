//! The safe Metal operator surface (synchronous and accumulative).
//!
//! Every operator builds [`dispatchspec`] geometry and runs through
//! [`drive::dispatch`](drive::dispatch) for the synchronous path or
//! [`session::dispatch`] for the accumulative `*_into` path. No direct
//! platform `unsafe` calls live here, so this file needs no
//! `allow(unsafe_code)`.
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;

use objc2_metal::{MTLBuffer, MTLSize};

use crate::engine::gpuerror;
use crate::metal::source;
use api::geometry::{attentionshape, matmulshape, ropeshape};

use super::drive::{alloc, dispatch, dispatchspec, sbuf};
use super::{Extents, asu32, buf, cache, flatgrid, flatthreads, fullthreads, rowgrid, session};

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
/// Returns [`gpuerror`] when no device is available, a buffer cannot be
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
) -> Result<Vec<f32>, gpuerror> {
    let total = batch
        .checked_mul(m)
        .and_then(|value| value.checked_mul(n))
        .ok_or(gpuerror::sizeoverflow)?;
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

/// Adds two equal-length `f32` operands elementwise.
///
/// # Errors
///
/// Returns [`gpuerror`] when no device is available, a buffer cannot be
/// allocated, the source fails to compile, the length does not fit `u32`, or
/// the kernel fails to execute.
pub(crate) fn add(cache: &cache, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, gpuerror> {
    elementwise2(cache, source::ADDENTRY, lhs, rhs)
}

/// Multiplies two equal-length `f32` operands elementwise.
///
/// # Errors
///
/// See [`add`].
pub(crate) fn mul(cache: &cache, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, gpuerror> {
    elementwise2(cache, source::MULENTRY, lhs, rhs)
}

/// Applies the sigmoid linear unit elementwise.
///
/// # Errors
///
/// See [`add`].
pub(crate) fn silu(cache: &cache, input: &[f32]) -> Result<Vec<f32>, gpuerror> {
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
) -> Result<Vec<f32>, gpuerror> {
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
/// Returns [`gpuerror`] when no device is available, a buffer cannot be
/// allocated, the source fails to compile, an extent does not fit `u32`, or the
/// kernel fails to execute.
pub(crate) fn softmax(
    cache: &cache,
    input: &[f32],
    rows: usize,
    width: usize,
) -> Result<Vec<f32>, gpuerror> {
    if rows == 0 {
        return Ok(Vec::new());
    }
    let total = rows.checked_mul(width).ok_or(gpuerror::sizeoverflow)?;
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
/// Returns [`gpuerror`] when no device is available, a buffer cannot be
/// allocated, the source fails to compile, an extent does not fit `u32`, or the
/// kernel fails to execute.
pub(crate) fn rmsnorm(
    cache: &cache,
    input: &[f32],
    weight: &[f32],
    rows: usize,
    width: usize,
    eps: f32,
) -> Result<Vec<f32>, gpuerror> {
    if rows == 0 {
        return Ok(Vec::new());
    }
    let total = rows.checked_mul(width).ok_or(gpuerror::sizeoverflow)?;
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
/// Returns [`gpuerror`] when no device is available, a buffer cannot be
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
) -> Result<Vec<f32>, gpuerror> {
    if rows == 0 {
        return Ok(Vec::new());
    }
    let pairs = rows
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(headdim / 2))
        .ok_or(gpuerror::sizeoverflow)?;
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
) -> Result<Vec<f32>, gpuerror> {
    let total = heads.checked_mul(headdim).ok_or(gpuerror::sizeoverflow)?;
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
/// Returns [`gpuerror`] when no device is available, a buffer cannot be
/// allocated, the source fails to compile, an extent does not fit `u32`, or the
/// kernel fails to execute.
pub(crate) fn embedding(
    cache: &cache,
    table: &[f32],
    indices: &[usize],
    width: usize,
) -> Result<Vec<f32>, gpuerror> {
    let rows = indices.len();
    if rows == 0 {
        return Ok(Vec::new());
    }
    let total = rows.checked_mul(width).ok_or(gpuerror::sizeoverflow)?;
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
/// [`METALMAXHEADDIM`](crate::metal::METALMAXHEADDIM).
///
/// # Errors
///
/// Returns [`gpuerror`] when no device is available, a buffer cannot be
/// allocated, the source fails to compile, an extent does not fit `u32`, or the
/// kernel fails to execute.
pub(crate) fn attention(
    cache: &cache,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    geometry: &attentionshape,
    causal: bool,
) -> Result<Vec<f32>, gpuerror> {
    if geometry.headdim > source::ATTNLIMIT {
        return Err(gpuerror::sizeoverflow);
    }
    let qseq = geometry.qseq;
    if qseq == 0 {
        return Ok(Vec::new());
    }
    let queryelements = qseq
        .checked_mul(geometry.heads)
        .and_then(|rows| rows.checked_mul(geometry.headdim))
        .ok_or(gpuerror::sizeoverflow)?;
    let groups = qseq
        .checked_mul(geometry.heads)
        .ok_or(gpuerror::sizeoverflow)?;
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

/// The host operands of one parallel scan invocation.
struct scanoperands<'a> {
    input: &'a [f32],
    a: &'a [f32],
    b: &'a [f32],
    c: &'a [f32],
    skip: &'a [f32],
    init: &'a [f32],
}

/// Encodes the per-block totals pass of the parallel scan.
fn scanaggregate(
    cache: &cache,
    operands: &scanoperands<'_>,
    totals: &Retained<ProtocolObject<dyn MTLBuffer>>,
    params: &[u32; 5],
    blocks: usize,
    d: usize,
) -> Result<(), gpuerror> {
    dispatch(
        cache,
        source::SCANPARALLELMSL,
        source::SCANAGGENTRY,
        &[
            buf::host(operands.input),
            buf::host(operands.a),
            buf::host(operands.b),
            buf::host(operands.c),
            buf::host(operands.skip),
            buf::host(operands.init),
            buf::device(totals),
        ],
        params,
        scangrid(blocks, d)?,
        scantiles(d),
    )?;
    Ok(())
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
) -> Result<(Vec<f32>, Vec<f32>), gpuerror> {
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
        .ok_or(gpuerror::sizeoverflow)?;
    let dstateout = alloc(&device, statebytes)?;
    let outbytes = n
        .checked_mul(d)
        .and_then(|elements| elements.checked_mul(core::mem::size_of::<f32>()))
        .ok_or(gpuerror::sizeoverflow)?;
    let dout = alloc(&device, outbytes)?;
    let blocks = n.div_ceil(source::SCANCHUNK);
    let dntotal = d.checked_mul(statewidth).ok_or(gpuerror::sizeoverflow)?;
    let pairbytes = |count: usize| {
        count
            .checked_mul(2)
            .and_then(|pairs| pairs.checked_mul(dntotal))
            .and_then(|elements| elements.checked_mul(core::mem::size_of::<f32>()))
            .ok_or(gpuerror::sizeoverflow)
    };
    let dtotal = alloc(&device, pairbytes(blocks)?)?;
    let prefixcount = blocks.checked_add(1).ok_or(gpuerror::sizeoverflow)?;
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
    let operands = scanoperands {
        input,
        a,
        b,
        c,
        skip,
        init,
    };
    scanaggregate(cache, &operands, &dtotal, &params, blocks, d)?;
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
) -> Result<(Vec<f32>, Vec<f32>), gpuerror> {
    let device = cache.device()?;
    // The output and the final state both stay on the device for one dispatch
    // and are read back together after it completes.
    let outbytes = n
        .checked_mul(d)
        .and_then(|elements| elements.checked_mul(core::mem::size_of::<f32>()))
        .ok_or(gpuerror::sizeoverflow)?;
    let dout = alloc(&device, outbytes)?;
    let statebytes = init
        .len()
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(gpuerror::sizeoverflow)?;
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
fn scangrid(blocks: usize, d: usize) -> Result<MTLSize, gpuerror> {
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
) -> Result<usize, gpuerror> {
    let batch = geometry.batch;
    let m = geometry.m;
    let k = geometry.k;
    let n = geometry.n;
    let total = batch
        .checked_mul(m)
        .and_then(|value| value.checked_mul(n))
        .ok_or(gpuerror::sizeoverflow)?;
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
) -> Result<usize, gpuerror> {
    elementwise2_into(cache, session, source::ADDENTRY, lhs, rhs, count)
}

/// Encodes an elementwise `mul` of `count` elements, returning the slot index.
pub(crate) fn mul_into(
    cache: &cache,
    session: &mut session,
    lhs: &ProtocolObject<dyn MTLBuffer>,
    rhs: &ProtocolObject<dyn MTLBuffer>,
    count: usize,
) -> Result<usize, gpuerror> {
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
) -> Result<usize, gpuerror> {
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
) -> Result<usize, gpuerror> {
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
) -> Result<usize, gpuerror> {
    if rows == 0 {
        return Ok(session.recordempty());
    }
    let total = rows.checked_mul(width).ok_or(gpuerror::sizeoverflow)?;
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
) -> Result<usize, gpuerror> {
    if rows == 0 {
        return Ok(session.recordempty());
    }
    let total = rows.checked_mul(width).ok_or(gpuerror::sizeoverflow)?;
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
) -> Result<usize, gpuerror> {
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
        .ok_or(gpuerror::sizeoverflow)?;
    let total = rows
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(headdim))
        .ok_or(gpuerror::sizeoverflow)?;
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
) -> Result<usize, gpuerror> {
    let total = heads.checked_mul(headdim).ok_or(gpuerror::sizeoverflow)?;
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
) -> Result<usize, gpuerror> {
    let rows = indices.len();
    if rows == 0 {
        return Ok(session.recordempty());
    }
    let total = rows.checked_mul(width).ok_or(gpuerror::sizeoverflow)?;
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
) -> Result<usize, gpuerror> {
    let qseq = geometry.qseq;
    if qseq == 0 {
        return Ok(session.recordempty());
    }
    let queryelements = qseq
        .checked_mul(geometry.heads)
        .and_then(|rows| rows.checked_mul(geometry.headdim))
        .ok_or(gpuerror::sizeoverflow)?;
    let groups = qseq
        .checked_mul(geometry.heads)
        .ok_or(gpuerror::sizeoverflow)?;
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
