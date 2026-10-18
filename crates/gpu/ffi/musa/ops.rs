//! The synchronous MUSA operator surface: the FMA/WMMA matmul family,
//! elementwise/normalisation/attention kernels, and the parallel scan.
//!
//! Every operator validates its shapes, builds [`buf`] / [`scalar`]
//! arguments, and forwards through [`dispatch`] / [`dispatchmixed`]; the
//! accumulative `*_record` counterparts live in [`record`]. No direct
//! platform `unsafe` calls live here.
use super::*;
use crate::engine::gpuerror;
use crate::musa::source;

// ---------------------------------------------------------------------------
// The safe operator surface the executor calls.
// ---------------------------------------------------------------------------

/// Computes `[batch, m, k] * [batch, k, n]` with the reference FMA kernel.
///
/// The kernel is compiled at build time by `mcc`; this is the A1 path (no
/// runtime JIT). The batch dimension is the grid's third axis, so every slice
/// is uploaded, launched, and downloaded in one dispatch instead of one per
/// slice. This is the high-precision reference path; the executor's default
/// path is [`matmul_wmma`].
pub(crate) fn matmul_fma(
    cache: &cache,
    lhs: &[f32],
    rhs: &[f32],
    m: usize,
    k: usize,
    n: usize,
    batch: usize,
) -> Result<Vec<f32>, gpuerror> {
    if batch == 0 || m == 0 || n == 0 {
        return Ok(Vec::new());
    }
    let total = m
        .checked_mul(n)
        .and_then(|value| value.checked_mul(batch))
        .ok_or(gpuerror::sizeoverflow)?;
    let grid = (
        (n as u32).div_ceil(source::THREADEDGE),
        (m as u32).div_ceil(source::THREADEDGE),
        batch as u32,
    );
    let block = (source::THREADEDGE, source::THREADEDGE, 1);
    let mut results = dispatch(
        cache,
        "cuda",
        source::MATMULENTRY,
        &[buf::host(lhs), buf::host(rhs), buf::out(total)],
        &[
            scalar::u32(m as u32),
            scalar::u32(k as u32),
            scalar::u32(n as u32),
            scalar::u32(batch as u32),
        ],
        grid,
        block,
        0,
    )?;
    Ok(results.remove(0))
}

/// Computes `[batch, m, k] * [batch, k, n]` with the Tensor-Core kernel.
///
/// The kernel accumulates in `f32` but rounds each operand to TF32 (a 10-bit
/// mantissa) before the Tensor-Core multiply, so its result differs from the
/// FMA reference path within the documented TF32 tolerance
/// (`docs/design.md` §21.8). It covers the same shapes, including a leading
/// batch dimension and edges that are not multiples of the 64x64 tile: the
/// kernel stages operands and results through guarded shared-memory tiles.
pub(crate) fn matmul_wmma(
    cache: &cache,
    lhs: &[f32],
    rhs: &[f32],
    m: usize,
    k: usize,
    n: usize,
    batch: usize,
) -> Result<Vec<f32>, gpuerror> {
    if batch == 0 || m == 0 || n == 0 {
        return Ok(Vec::new());
    }
    let total = m
        .checked_mul(n)
        .and_then(|value| value.checked_mul(batch))
        .ok_or(gpuerror::sizeoverflow)?;
    let grid = (
        (n as u32).div_ceil(source::MATMULWMMABN),
        (m as u32).div_ceil(source::MATMULWMMABM),
        batch as u32,
    );
    let block = (source::MATMULWMMATHREADS, 1, 1);
    let mut results = dispatch(
        cache,
        source::MATMULWMMASOURCEID,
        source::MATMULWMMAENTRY,
        &[buf::host(lhs), buf::host(rhs), buf::out(total)],
        &[
            scalar::u32(m as u32),
            scalar::u32(k as u32),
            scalar::u32(n as u32),
            scalar::u32(batch as u32),
        ],
        grid,
        block,
        0,
    )?;
    Ok(results.remove(0))
}

/// Fused RMSNorm (weight + per-row inverse RMS) followed by a Tensor-Core
/// matmul. The reciprocal-RMS kernel leaves its result on the device and the
/// matmul reads it directly, so the normalised activation is never
/// materialised.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rmsnormmatmul(
    cache: &cache,
    input: &[f32],
    weight: &[f32],
    rhs: &[f32],
    m: usize,
    k: usize,
    n: usize,
    batch: usize,
    eps: f32,
) -> Result<Vec<f32>, gpuerror> {
    if batch == 0 || m == 0 || n == 0 {
        return Ok(Vec::new());
    }
    let rows = batch.checked_mul(m).ok_or(gpuerror::sizeoverflow)?;
    let total = rows.checked_mul(n).ok_or(gpuerror::sizeoverflow)?;
    let inv = dispatchmixed(
        cache,
        source::MATMULWMMASOURCEID,
        source::RMSINVENTRY,
        &[buf::hostdevice(input), buf::outdevice(rows)],
        &[
            scalar::u32(rows as u32),
            scalar::u32(k as u32),
            scalar::f32(eps),
        ],
        (rows as u32, 1, 1),
        (source::RMSNORMTILE, 1, 1),
        source::RMSNORMTILE * 4,
    )?;
    let inputptr = inv.device[0].0;
    let invptr = inv.device[1].0;
    // The fused kernel is full-precision fp32 (the fused operator carries
    // a 1e-4 contract, not the TF32 2e-3 bound), so it uses the same 16x16
    // FMA tile as the reference matmul rather than the wave128 Tensor tile.
    let grid = (
        (n as u32).div_ceil(source::THREADEDGE),
        (m as u32).div_ceil(source::THREADEDGE),
        batch as u32,
    );
    let mut results = dispatch(
        cache,
        source::MATMULWMMASOURCEID,
        source::MATMULRMSENTRY,
        &[
            buf::device(inputptr),
            buf::host(weight),
            buf::device(invptr),
            buf::host(rhs),
            buf::out(total),
        ],
        &[
            scalar::u32(m as u32),
            scalar::u32(k as u32),
            scalar::u32(n as u32),
            scalar::u32(batch as u32),
        ],
        grid,
        (source::THREADEDGE, source::THREADEDGE, 1),
        0,
    )?;
    retiredevice(cache, inputptr, inv.device[0].1);
    retiredevice(cache, invptr, inv.device[1].1);
    Ok(results.remove(0))
}

/// Elementwise `lhs + rhs`.
pub(crate) fn add(cache: &cache, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, gpuerror> {
    let count = lhs.len();
    ewise(
        cache,
        source::ADDENTRY,
        &[buf::host(lhs), buf::host(rhs), buf::out(count)],
        count,
    )
}

/// Elementwise `lhs * rhs`.
pub(crate) fn mul(cache: &cache, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, gpuerror> {
    let count = lhs.len();
    ewise(
        cache,
        source::MULENTRY,
        &[buf::host(lhs), buf::host(rhs), buf::out(count)],
        count,
    )
}

/// Sigmoid linear unit, elementwise.
pub(crate) fn silu(cache: &cache, input: &[f32]) -> Result<Vec<f32>, gpuerror> {
    let count = input.len();
    ewise(
        cache,
        source::SILUENTRY,
        &[buf::host(input), buf::out(count)],
        count,
    )
}

/// Runs the flat elementwise kernel `entry` over `count` elements.
fn ewise(
    cache: &cache,
    entry: &'static str,
    buffers: &[buf<'_>],
    count: usize,
) -> Result<Vec<f32>, gpuerror> {
    let grid = ((count as u32).div_ceil(source::FLATTILE), 1, 1);
    let mut results = dispatch(
        cache,
        "cuda",
        entry,
        buffers,
        &[scalar::u32(count as u32)],
        grid,
        (source::FLATTILE, 1, 1),
        0,
    )?;
    Ok(results.remove(0))
}

/// RMS normalisation over the last axis.
pub(crate) fn rmsnorm(
    cache: &cache,
    input: &[f32],
    weight: &[f32],
    rows: usize,
    width: usize,
    eps: f32,
) -> Result<Vec<f32>, gpuerror> {
    let count = rows.checked_mul(width).ok_or(gpuerror::sizeoverflow)?;
    let grid = (rows as u32, 1, 1);
    let block = (source::RMSNORMTILE, 1, 1);
    let mut results = dispatch(
        cache,
        "cuda",
        source::RMSNORMENTRY,
        &[buf::host(input), buf::host(weight), buf::out(count)],
        &[scalar::u32(width as u32), scalar::f32(eps)],
        grid,
        block,
        source::RMSNORMTILE * 4,
    )?;
    Ok(results.remove(0))
}

/// Numerically stable softmax over the last axis.
pub(crate) fn softmax(
    cache: &cache,
    input: &[f32],
    rows: usize,
    width: usize,
) -> Result<Vec<f32>, gpuerror> {
    let count = rows.checked_mul(width).ok_or(gpuerror::sizeoverflow)?;
    let grid = (rows as u32, 1, 1);
    let block = (source::FLATTILE, 1, 1);
    let mut results = dispatch(
        cache,
        "cuda",
        source::SOFTMAXENTRY,
        &[buf::host(input), buf::out(count)],
        &[scalar::u32(width as u32)],
        grid,
        block,
        source::FLATTILE * 4,
    )?;
    Ok(results.remove(0))
}

/// Rotary position embedding.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rope(
    cache: &cache,
    input: &[f32],
    rows: usize,
    seqaxis: usize,
    heads: usize,
    headdim: usize,
    base: f32,
) -> Result<Vec<f32>, gpuerror> {
    let count = input.len();
    let grid = ((seqaxis as u32).div_ceil(source::FLATTILE), 1, 1);
    let block = (source::FLATTILE, 1, 1);
    let mut results = dispatch(
        cache,
        "cuda",
        source::ROPEENTRY,
        &[buf::host(input), buf::out(count)],
        &[
            scalar::u32(rows as u32),
            scalar::u32(seqaxis as u32),
            scalar::u32(heads as u32),
            scalar::u32(headdim as u32),
            scalar::f32(base),
        ],
        grid,
        block,
        0,
    )?;
    Ok(results.remove(0))
}

/// Absolute-position rotary embedding.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rope_at(
    cache: &cache,
    input: &[f32],
    position: usize,
    heads: usize,
    headdim: usize,
    base: f32,
) -> Result<Vec<f32>, gpuerror> {
    let count = input.len();
    let grid = (1, 1, 1);
    let block = (source::FLATTILE, 1, 1);
    let mut results = dispatch(
        cache,
        "cuda",
        source::ROPEATENTRY,
        &[buf::host(input), buf::out(count)],
        &[
            scalar::u32(heads as u32),
            scalar::u32(headdim as u32),
            scalar::u32(position as u32),
            scalar::f32(base),
        ],
        grid,
        block,
        0,
    )?;
    Ok(results.remove(0))
}

/// Gathers `rows` of `width` elements from `table` by `indices`.
pub(crate) fn embedding(
    cache: &cache,
    table: &[f32],
    indices: &[u64],
    width: usize,
) -> Result<Vec<f32>, gpuerror> {
    let rows = indices.len();
    let count = rows.checked_mul(width).ok_or(gpuerror::sizeoverflow)?;
    let grid = (rows as u32, 1, 1);
    let block = (source::FLATTILE, 1, 1);
    let mut results = dispatch(
        cache,
        "cuda",
        source::EMBEDDINGENTRY,
        &[buf::host(table), buf::hostu64(indices), buf::out(count)],
        &[scalar::u64(rows as u64), scalar::u64(width as u64)],
        grid,
        block,
        0,
    )?;
    Ok(results.remove(0))
}

/// Multi-head scaled dot-product attention.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention(
    cache: &cache,
    query: &[f32],
    key: &[f32],
    value: &[f32],
    qseq: usize,
    kvseq: usize,
    heads: usize,
    kvheads: usize,
    headdim: usize,
    causal: bool,
) -> Result<Vec<f32>, gpuerror> {
    let count = qseq
        .checked_mul(heads)
        .and_then(|value| value.checked_mul(headdim))
        .ok_or(gpuerror::sizeoverflow)?;
    let grid = (
        qseq.checked_mul(heads).ok_or(gpuerror::sizeoverflow)? as u32,
        1,
        1,
    );
    let block = (source::ATTENTIONLANES, 1, 1);
    let mut results = dispatch(
        cache,
        "cuda",
        source::ATTENTIONENTRY,
        &[
            buf::host(query),
            buf::host(key),
            buf::host(value),
            buf::out(count),
        ],
        &[
            scalar::u32(qseq as u32),
            scalar::u32(kvseq as u32),
            scalar::u32(heads as u32),
            scalar::u32(kvheads as u32),
            scalar::u32(headdim as u32),
            scalar::u32(causal as u32),
        ],
        grid,
        block,
        (2 * source::ATTENTIONLANES + headdim as u32) * 4,
    )?;
    Ok(results.remove(0))
}

/// Incremental-decode attention over a cached history plus a new row.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_cached(
    cache: &cache,
    query: &[f32],
    keycached: &[f32],
    keynew: &[f32],
    valuecached: &[f32],
    valuenew: &[f32],
    kvseq: usize,
    heads: usize,
    kvheads: usize,
    headdim: usize,
    causal: bool,
) -> Result<Vec<f32>, gpuerror> {
    let count = heads.checked_mul(headdim).ok_or(gpuerror::sizeoverflow)?;
    let grid = (heads as u32, 1, 1);
    let block = (source::ATTENTIONLANES, 1, 1);
    let mut results = dispatch(
        cache,
        "cuda",
        source::ATTENTIONCACHEDENTRY,
        &[
            buf::host(query),
            buf::host(keycached),
            buf::host(valuecached),
            buf::host(keynew),
            buf::host(valuenew),
            buf::out(count),
        ],
        &[
            scalar::u32(kvseq as u32),
            scalar::u32(heads as u32),
            scalar::u32(kvheads as u32),
            scalar::u32(headdim as u32),
            scalar::u32(causal as u32),
        ],
        grid,
        block,
        (2 * source::ATTENTIONLANES + headdim as u32) * 4,
    )?;
    Ok(results.remove(0))
}

/// Serial per-channel semigroup scan (the bit-for-bit reference path).
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_serial(
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
    let outcount = n.checked_mul(d).ok_or(gpuerror::sizeoverflow)?;
    let grid = (d as u32, 1, 1);
    let block = (source::FLATTILE, 1, 1);
    let mut results = dispatch(
        cache,
        "cuda",
        source::SCANENTRY,
        &[
            buf::host(input),
            buf::host(a),
            buf::host(b),
            buf::host(c),
            buf::host(skip),
            buf::host(init),
            buf::out(outcount),
            buf::out(statewidth.checked_mul(d).ok_or(gpuerror::sizeoverflow)?),
        ],
        &[
            scalar::u32(n as u32),
            scalar::u32(d as u32),
            scalar::u32(statewidth as u32),
        ],
        grid,
        block,
        0,
    )?;
    let out = results.remove(0);
    let state = results.remove(0);
    Ok((out, state))
}

/// Extracts the device pointer of an operand that must already be resident.
fn deviceptr(value: buf<'_>) -> Result<mudevptr, gpuerror> {
    match value {
        buf::device(ptr) => Ok(ptr),
        _ => Err(gpuerror::transfer {
            message: "scan state operand is not device-resident".into(),
        }),
    }
}

/// Records a serial scan into a device-resident chain, returning the output and
/// final-state slots.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_record(
    session: &mut session,
    input: buf<'_>,
    init: buf<'_>,
    a: buf<'_>,
    b: buf<'_>,
    c: buf<'_>,
    skip: buf<'_>,
    n: usize,
    d: usize,
    statewidth: usize,
) -> Result<(usize, usize), gpuerror> {
    let outcount = n.checked_mul(d).ok_or(gpuerror::sizeoverflow)?;
    let statecount = d.checked_mul(statewidth).ok_or(gpuerror::sizeoverflow)?;
    // The kernel updates its state operand in place. Stage a copy so the
    // caller's init slot (which may itself be a requested result) is preserved.
    let statebytes = statecount
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(gpuerror::sizeoverflow)?;
    let initslot = session.copy(deviceptr(init)?, statebytes)?;
    let initcopy = buf::device(session.slotptr(initslot));
    let slots = session.recordmulti(
        "cuda",
        source::SCANENTRY,
        &[
            input,
            a,
            b,
            c,
            skip,
            initcopy,
            buf::outdevice(outcount),
            buf::outdevice(statecount),
        ],
        &[
            scalar::u32(n as u32),
            scalar::u32(d as u32),
            scalar::u32(statewidth as u32),
        ],
        (d as u32, 1, 1),
        (source::FLATTILE, 1, 1),
        0,
    )?;
    Ok((slots[0], slots[1]))
}

/// Records a single-step scan into a device-resident chain, returning the
/// output and new-state slots.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_cached_record(
    session: &mut session,
    input: buf<'_>,
    state: buf<'_>,
    a: buf<'_>,
    b: buf<'_>,
    c: buf<'_>,
    skip: buf<'_>,
    d: usize,
    statewidth: usize,
) -> Result<(usize, usize), gpuerror> {
    let statecount = d.checked_mul(statewidth).ok_or(gpuerror::sizeoverflow)?;
    // The kernel updates its state operand in place; copy it so the caller's
    // state slot (often the previous scan's output) stays downloadable.
    let statebytes = statecount
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(gpuerror::sizeoverflow)?;
    let stateslot = session.copy(deviceptr(state)?, statebytes)?;
    let statecopy = buf::device(session.slotptr(stateslot));
    let slots = session.recordmulti(
        "cuda",
        source::SCANENTRY,
        &[
            input,
            a,
            b,
            c,
            skip,
            statecopy,
            buf::outdevice(d),
            buf::outdevice(statecount),
        ],
        &[
            scalar::u32(1),
            scalar::u32(d as u32),
            scalar::u32(statewidth as u32),
        ],
        (d as u32, 1, 1),
        (source::FLATTILE, 1, 1),
        0,
    )?;
    Ok((slots[0], slots[1]))
}

/// Parallel three-stage semigroup scan for state widths within
/// [`source::SCANMAXN`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_parallel(
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
    // The three passes chain on device-resident intermediates: pass 1 keeps
    // `dtotal` and the six shared operands on the device (uploaded once),
    // pass 2 binds them, keeps `dprefix` on the device and downloads only the
    // final state, and pass 3 binds the retained operands and downloads only
    // the output. The intermediates are released through `retiredevice`, whose
    // queue is drained inside the next handoff block, so the release adds no
    // driver call of its own (docs/musa-backend-plan.md §5.5).
    let blocks = n.div_ceil(source::SCANCHUNK as usize);
    let dntotal = d.checked_mul(statewidth).ok_or(gpuerror::sizeoverflow)?;
    let pairbytes = |count: usize| {
        count
            .checked_mul(2)
            .and_then(|pairs| pairs.checked_mul(dntotal))
            .and_then(|elements| elements.checked_mul(core::mem::size_of::<f32>()))
            .ok_or(gpuerror::sizeoverflow)
    };
    let aggsize = pairbytes(blocks)? / core::mem::size_of::<f32>();

    // Pass 1: per-block totals; every operand upload is retained.
    let totals = dispatchmixed(
        cache,
        "cuda",
        source::SCANAGGENTRY,
        &[
            buf::hostdevice(input),
            buf::hostdevice(a),
            buf::hostdevice(b),
            buf::hostdevice(c),
            buf::hostdevice(skip),
            buf::hostdevice(init),
            buf::outdevice(aggsize),
        ],
        &[
            scalar::u32(n as u32),
            scalar::u32(d as u32),
            scalar::u32(statewidth as u32),
            scalar::u32(source::SCANCHUNK),
            scalar::u32(blocks as u32),
        ],
        (blocks as u32, (d as u32).div_ceil(source::SCANBLOCKDIM), 1),
        (source::SCANBLOCKDIM, 1, 1),
        0,
    )?;
    // Device slots in buffer order: input, a, b, c, skip, init, dtotal.
    let dinput = totals.device[0];
    let da = totals.device[1];
    let db = totals.device[2];
    let dc = totals.device[3];
    let dskip = totals.device[4];
    let dinit = totals.device[5];
    let dtotal = totals.device[6];

    // Pass 2: inter-block prefix plus the whole-sequence final state.
    let prefixcount = blocks.checked_add(1).ok_or(gpuerror::sizeoverflow)?;
    let statecount = dntotal;
    let mut prefix = dispatchmixed(
        cache,
        "cuda",
        source::SCANPREFIXENTRY,
        &[
            buf::device(dtotal.0),
            buf::device(dinit.0),
            buf::out(statecount),
            buf::outdevice(pairbytes(prefixcount)? / core::mem::size_of::<f32>()),
        ],
        &[
            scalar::u32(n as u32),
            scalar::u32(d as u32),
            scalar::u32(statewidth as u32),
            scalar::u32(source::SCANCHUNK),
            scalar::u32(blocks as u32),
        ],
        ((statecount as u32).min(1024), 1, 1),
        ((statecount as u32).min(1024), 1, 1),
        0,
    )?;
    let state = prefix.host.remove(0);
    // Only producer-side slots are collected; the bound pass-throughs
    // (`device`) are not, so the single collected entry is `dprefix`.
    let dprefix = prefix.device[0];
    retiredevice(cache, dtotal.0, dtotal.1);

    // Pass 3: per-block apply; only the output is downloaded.
    let outcount = n.checked_mul(d).ok_or(gpuerror::sizeoverflow)?;
    let mut results = dispatch(
        cache,
        "cuda",
        source::SCANAPPLYENTRY,
        &[
            buf::device(dinput.0),
            buf::device(da.0),
            buf::device(db.0),
            buf::device(dc.0),
            buf::device(dskip.0),
            buf::device(dinit.0),
            buf::device(dprefix.0),
            buf::out(outcount),
        ],
        &[
            scalar::u32(n as u32),
            scalar::u32(d as u32),
            scalar::u32(statewidth as u32),
            scalar::u32(source::SCANCHUNK),
            scalar::u32(blocks as u32),
        ],
        (blocks as u32, (d as u32).div_ceil(source::SCANBLOCKDIM), 1),
        (source::SCANBLOCKDIM, 1, 1),
        0,
    )?;
    let out = results.remove(0);
    retiredevice(cache, dinput.0, dinput.1);
    retiredevice(cache, da.0, da.1);
    retiredevice(cache, db.0, db.1);
    retiredevice(cache, dc.0, dc.1);
    retiredevice(cache, dskip.0, dskip.1);
    retiredevice(cache, dinit.0, dinit.1);
    retiredevice(cache, dprefix.0, dprefix.1);
    Ok((out, state))
}

/// Semigroup scan: parallel for state widths within [`source::SCANMAXN`],
/// serial (the bit-for-bit reference) above it.
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
    // A single chunk (`n <= SCANCHUNK`) degenerates the three-pass parallel
    // scan to one block: pass 1 aggregates the whole sequence, pass 2 is a
    // trivial identity prefix, and pass 3 recomputes it — three dispatches for
    // no parallelism over the serial kernel, which already runs one thread per
    // channel. Route the single-chunk case to the serial reference, which is
    // the bit-for-bit continuation of the same recurrence.
    if statewidth > source::SCANMAXN as usize || n <= source::SCANCHUNK as usize {
        return scan_serial(cache, input, init, a, b, c, skip, n, d, statewidth);
    }
    scan_parallel(cache, input, init, a, b, c, skip, n, d, statewidth)
}

/// Single-step incremental scan (decode form).
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_cached(
    cache: &cache,
    input: &[f32],
    state: &[f32],
    a: &[f32],
    b: &[f32],
    c: &[f32],
    skip: &[f32],
    d: usize,
    statewidth: usize,
) -> Result<(Vec<f32>, Vec<f32>), gpuerror> {
    let outcount = d;
    let mut results = dispatch(
        cache,
        "cuda",
        source::SCANENTRY,
        &[
            buf::host(input),
            buf::host(a),
            buf::host(b),
            buf::host(c),
            buf::host(skip),
            buf::host(state),
            buf::out(outcount),
            buf::out(statewidth.checked_mul(d).ok_or(gpuerror::sizeoverflow)?),
        ],
        &[
            scalar::u32(1),
            scalar::u32(d as u32),
            scalar::u32(statewidth as u32),
        ],
        (d as u32, 1, 1),
        (source::FLATTILE, 1, 1),
        0,
    )?;
    let out = results.remove(0);
    let newstate = results.remove(0);
    Ok((out, newstate))
}
