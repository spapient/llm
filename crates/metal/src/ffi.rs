#![allow(unsafe_code)]
//! The single platform-FFI island of the Metal backend.
//!
//! Authority: docs/design.md §21.8. This is the only file in the crate that
//! declares `#![allow(unsafe_code)]`; the crate root keeps
//! `#![deny(unsafe_code)]`. Every platform call that the Objective-C bridge
//! marks `unsafe` is confined here, and each `unsafe` construct is directly
//! preceded by a `// SAFETY:` comment. The registered site count (four) is
//! enforced by `scripts/checkunsafe.sh`.
//!
//! The four sites are one per **class of platform interaction**, not one per
//! operator: uploading host data into a buffer, downloading a result out of a
//! buffer, binding operand buffers to the encoder, and binding the scalar
//! parameter buffer. Every kernel in [`crate::source`] reuses those same four
//! sites, so the count does not grow with the operator count.
//!
//! The safe functions expose a narrow, run-to-completion interface: upload host
//! slices, dispatch one kernel, download the result. All device, command, and
//! library objects are created locally and released by scope, so no GPU state
//! outlives a call.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};

use crate::error::metalerror;
use crate::geometry::attentionshape;
use crate::source;

/// The threadgroup edge used when dispatching the matmul kernel.
const THREADEDGE: usize = 16;

/// Computes `[m, k] * [k, n]` on the default Metal device.
///
/// `lhs` holds `m * k` row-major elements and `rhs` holds `k * n`. The result
/// has `m * n` elements. The call is synchronous: it returns only after the
/// kernel has completed and the result has been copied back to the host.
///
/// # Errors
///
/// Returns [`metalerror`] when no device is available, a buffer cannot be
/// allocated, the MSL source fails to compile, the compute pipeline cannot be
/// built, a dimension does not fit a `u32`, or the kernel fails to execute.
pub(crate) fn matmul(
    lhs: &[f32],
    rhs: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, metalerror> {
    if m == 0 || n == 0 {
        return Ok(Vec::new());
    }
    if k == 0 {
        return Ok(vec![0.0_f32; m * n]);
    }
    let dims = Extents::new(m, k, n)?;

    let device = opendevice()?;
    let bufa = alloc(&device, core::mem::size_of_val(lhs))?;
    let bufb = alloc(&device, core::mem::size_of_val(rhs))?;
    let bufc = alloc(&device, m * n * core::mem::size_of::<f32>())?;
    write(&bufa, lhs);
    write(&bufb, rhs);

    let buffers: [&ProtocolObject<dyn MTLBuffer>; 3] = [&bufa, &bufb, &bufc];
    let params = [dims.m, dims.k, dims.n];
    execute(
        &device,
        source::MATMULMSL,
        source::MATMULENTRY,
        &buffers,
        &params,
        MTLSize {
            width: dims.cols,
            height: dims.rows,
            depth: 1,
        },
        dims.threadgroup(),
    )?;
    read(&bufc, m * n)
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
pub(crate) fn add(lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, metalerror> {
    elementwise2(source::ADDENTRY, lhs, rhs)
}

/// Multiplies two equal-length `f32` operands elementwise.
///
/// # Errors
///
/// See [`add`].
pub(crate) fn mul(lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, metalerror> {
    elementwise2(source::MULENTRY, lhs, rhs)
}

/// Applies the sigmoid linear unit elementwise.
///
/// # Errors
///
/// See [`add`].
pub(crate) fn silu(input: &[f32]) -> Result<Vec<f32>, metalerror> {
    if input.is_empty() {
        return Ok(Vec::new());
    }
    let count = asu32(input.len())?;
    let device = opendevice()?;
    let da = alloc(&device, core::mem::size_of_val(input))?;
    let dc = alloc(&device, core::mem::size_of_val(input))?;
    write(&da, input);

    let buffers: [&ProtocolObject<dyn MTLBuffer>; 2] = [&da, &dc];
    let params = [count];
    execute(
        &device,
        source::ELEMENTWISE,
        source::SILUENTRY,
        &buffers,
        &params,
        flatgrid(input.len())?,
        flatthreads(input.len()),
    )?;
    read(&dc, input.len())
}

/// Runs the two-operand elementwise kernel named `entry` in [`ELEMENTWISE`].
fn elementwise2(entry: &str, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, metalerror> {
    if lhs.is_empty() {
        return Ok(Vec::new());
    }
    let count = asu32(lhs.len())?;
    let device = opendevice()?;
    let da = alloc(&device, core::mem::size_of_val(lhs))?;
    let db = alloc(&device, core::mem::size_of_val(rhs))?;
    let dc = alloc(&device, core::mem::size_of_val(lhs))?;
    write(&da, lhs);
    write(&db, rhs);

    let buffers: [&ProtocolObject<dyn MTLBuffer>; 3] = [&da, &db, &dc];
    let params = [count];
    execute(
        &device,
        source::ELEMENTWISE,
        entry,
        &buffers,
        &params,
        flatgrid(lhs.len())?,
        flatthreads(lhs.len()),
    )?;
    read(&dc, lhs.len())
}

/// Numerically stable softmax over the last axis.
///
/// `input` is read as `[rows, width]`. The row maximum is subtracted before
/// exponentiating, matching the scalar reference, and a row whose shifted
/// exponents all underflow to zero is left as zeros.
///
/// # Errors
///
/// Returns [`metalerror`] when no device is available, a buffer cannot be
/// allocated, the source fails to compile, an extent does not fit `u32`, or the
/// kernel fails to execute.
pub(crate) fn softmax(input: &[f32], rows: usize, width: usize) -> Result<Vec<f32>, metalerror> {
    if rows == 0 {
        return Ok(Vec::new());
    }
    let total = rows.checked_mul(width).ok_or(metalerror::sizeoverflow)?;
    let bytes = total
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(metalerror::sizeoverflow)?;
    let device = opendevice()?;
    let dx = alloc(&device, bytes)?;
    let dy = alloc(&device, bytes)?;
    write(&dx, input);

    let buffers: [&ProtocolObject<dyn MTLBuffer>; 2] = [&dx, &dy];
    let params = [asu32(width)?];
    execute(
        &device,
        source::SOFTMAXMSL,
        source::SOFTMAXENTRY,
        &buffers,
        &params,
        rowgrid(rows, source::SOFTMAXTILE)?,
        fullthreads(source::SOFTMAXTILE),
    )?;
    read(&dy, total)
}

/// RMS-normalises `rows` rows of `width` elements and scales them by `weight`.
///
/// # Errors
///
/// Returns [`metalerror`] when no device is available, a buffer cannot be
/// allocated, the source fails to compile, an extent does not fit `u32`, or the
/// kernel fails to execute.
pub(crate) fn rmsnorm(
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
    let bytes = total
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(metalerror::sizeoverflow)?;
    let device = opendevice()?;
    let dx = alloc(&device, bytes)?;
    let dw = alloc(&device, core::mem::size_of_val(weight))?;
    let dy = alloc(&device, bytes)?;
    write(&dx, input);
    write(&dw, weight);

    let buffers: [&ProtocolObject<dyn MTLBuffer>; 3] = [&dx, &dw, &dy];
    let params = [asu32(width)?, eps.to_bits()];
    execute(
        &device,
        source::RMSNORMMSL,
        source::RMSNORMENTRY,
        &buffers,
        &params,
        rowgrid(rows, source::RMSNORMTILE)?,
        fullthreads(source::RMSNORMTILE),
    )?;
    read(&dy, total)
}

/// Rotates every `headdim` slice of `rows` rows by its position angle.
///
/// # Errors
///
/// Returns [`metalerror`] when no device is available, a buffer cannot be
/// allocated, the source fails to compile, an extent does not fit `u32`, or the
/// kernel fails to execute.
pub(crate) fn rope(
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
    let device = opendevice()?;
    let dx = alloc(&device, core::mem::size_of_val(input))?;
    let dy = alloc(&device, core::mem::size_of_val(input))?;
    write(&dx, input);

    let buffers: [&ProtocolObject<dyn MTLBuffer>; 2] = [&dx, &dy];
    let params = [
        asu32(rows)?,
        asu32(seqaxis)?,
        asu32(heads)?,
        asu32(headdim)?,
        base.to_bits(),
    ];
    execute(
        &device,
        source::ROPEMSL,
        source::ROPEENTRY,
        &buffers,
        &params,
        flatgrid(pairs)?,
        flatthreads(pairs),
    )?;
    read(&dy, input.len())
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
    table: &[f32],
    indices: &[usize],
    width: usize,
) -> Result<Vec<f32>, metalerror> {
    let rows = indices.len();
    if rows == 0 {
        return Ok(Vec::new());
    }
    let total = rows.checked_mul(width).ok_or(metalerror::sizeoverflow)?;
    let bytes = total
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(metalerror::sizeoverflow)?;
    let devindices: Vec<u64> = indices.iter().map(|&index| index as u64).collect();
    let device = opendevice()?;
    let dtable = alloc(&device, core::mem::size_of_val(table))?;
    let dindices = alloc(&device, core::mem::size_of_val(&devindices))?;
    let dout = alloc(&device, bytes)?;
    write(&dtable, table);
    write(&dindices, &devindices);

    let buffers: [&ProtocolObject<dyn MTLBuffer>; 3] = [&dtable, &dindices, &dout];
    let params = [rows as u64, width as u64];
    execute(
        &device,
        source::EMBEDDINGMSL,
        source::EMBEDDINGENTRY,
        &buffers,
        &params,
        flatgrid(total)?,
        flatthreads(total),
    )?;
    read(&dout, total)
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
    query: &[f32],
    key: &[f32],
    value: &[f32],
    geometry: &attentionshape,
    causal: bool,
) -> Result<Vec<f32>, metalerror> {
    let seq = geometry.seq;
    if seq == 0 {
        return Ok(Vec::new());
    }
    let queryelements = seq
        .checked_mul(geometry.heads)
        .and_then(|rows| rows.checked_mul(geometry.headdim))
        .ok_or(metalerror::sizeoverflow)?;
    let bytes = queryelements
        .checked_mul(core::mem::size_of::<f32>())
        .ok_or(metalerror::sizeoverflow)?;
    let groups = seq
        .checked_mul(geometry.heads)
        .ok_or(metalerror::sizeoverflow)?;
    let device = opendevice()?;
    let dquery = alloc(&device, core::mem::size_of_val(query))?;
    let dkey = alloc(&device, core::mem::size_of_val(key))?;
    let dvalue = alloc(&device, core::mem::size_of_val(value))?;
    let dout = alloc(&device, bytes)?;
    write(&dquery, query);
    write(&dkey, key);
    write(&dvalue, value);

    let buffers: [&ProtocolObject<dyn MTLBuffer>; 4] = [&dquery, &dkey, &dvalue, &dout];
    let params = [
        asu32(seq)?,
        asu32(geometry.heads)?,
        asu32(geometry.kvheads)?,
        asu32(geometry.headdim)?,
        u32::from(causal),
    ];
    execute(
        &device,
        source::ATTENTIONMSL,
        source::ATTENTIONENTRY,
        &buffers,
        &params,
        rowgrid(groups, source::ATTENTIONLANES)?,
        fullthreads(source::ATTENTIONLANES),
    )?;
    read(&dout, queryelements)
}

/// Opens the system default Metal device.
fn opendevice() -> Result<Retained<ProtocolObject<dyn MTLDevice>>, metalerror> {
    objc2_metal::MTLCreateSystemDefaultDevice().ok_or(metalerror::nodevice)
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
            message: describe(&error),
        })?;
    let name = NSString::from_str(entry);
    let function = library
        .newFunctionWithName(&name)
        .ok_or_else(|| metalerror::compile {
            message: format!("entry point `{entry}` is missing from the compiled library"),
        })?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| metalerror::pipeline {
            message: describe(&error),
        })
}

/// Compiles, binds, and runs one kernel, then waits for completion.
///
/// `buffers` are bound to buffer indices `0..buffers.len()` in order, and
/// `params` is copied into the next index, so a kernel's operand buffers and
/// its single parameter buffer line up with its `[[buffer(i)]]` attributes.
///
/// # Errors
///
/// Returns [`metalerror`] when compilation or pipeline creation fails, or when
/// the device, queue, or command buffer declines to hand out the resources the
/// dispatch needs.
fn execute<T>(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
    entry: &str,
    buffers: &[&ProtocolObject<dyn MTLBuffer>],
    params: &T,
    grid: MTLSize,
    threads: MTLSize,
) -> Result<(), metalerror> {
    let pipeline = compile(device, source, entry)?;
    let queue = newqueue(device)?;
    let command = newcommand(&queue)?;
    let encoder = newencoder(&command)?;
    encoder.setComputePipelineState(&pipeline);
    bind(&encoder, buffers);
    setbytes(&encoder, params, buffers.len());
    encoder.dispatchThreads_threadsPerThreadgroup(grid, threads);
    encoder.endEncoding();
    command.commit();
    command.waitUntilCompleted();
    Ok(())
}

/// Creates the device's command queue.
fn newqueue(
    device: &ProtocolObject<dyn MTLDevice>,
) -> Result<Retained<ProtocolObject<dyn MTLCommandQueue>>, metalerror> {
    device
        .newCommandQueue()
        .ok_or_else(|| metalerror::execution {
            message: "the device returned no command queue".to_string(),
        })
}

/// Creates a command buffer on `queue`.
fn newcommand(
    queue: &ProtocolObject<dyn MTLCommandQueue>,
) -> Result<Retained<ProtocolObject<dyn MTLCommandBuffer>>, metalerror> {
    queue.commandBuffer().ok_or_else(|| metalerror::execution {
        message: "the queue returned no command buffer".to_string(),
    })
}

/// Creates a compute encoder on `command`.
fn newencoder(
    command: &ProtocolObject<dyn MTLCommandBuffer>,
) -> Result<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>, metalerror> {
    command
        .computeCommandEncoder()
        .ok_or_else(|| metalerror::execution {
            message: "the command buffer returned no compute encoder".to_string(),
        })
}

/// Binds the operand buffers to the kernel's buffer indices.
fn bind(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buffers: &[&ProtocolObject<dyn MTLBuffer>],
) {
    // SAFETY: `encoder` is a live compute encoder and each buffer was allocated
    // by `alloc` with the byte length the kernel reads or writes, so binding
    // buffer indices 0..buffers.len() to the pipeline's `[[buffer(i)]]`
    // attributes matches the MSL signature. The kernel bounds-tests every
    // element index it derives, and no index is dereferenced here.
    unsafe {
        for (index, buffer) in buffers.iter().enumerate() {
            encoder.setBuffer_offset_atIndex(Some(*buffer), 0, index);
        }
    }
}

/// Copies `params` into the kernel's scalar parameter buffer at `index`.
fn setbytes<T>(encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>, params: &T, index: usize) {
    // SAFETY: `params` is a live value whose address is valid for the duration
    // of the call, and `setBytes_length_atIndex` copies `size_of_val(params)`
    // bytes rather than retaining the pointer, so the reference need not
    // outlive the call. `index` is the number of operand buffers, which the
    // caller's kernel declares as its parameter buffer's `[[buffer(index)]]`.
    unsafe {
        encoder.setBytes_length_atIndex(
            core::ptr::NonNull::from(params).cast(),
            core::mem::size_of_val(params),
            index,
        );
    }
}

/// Allocates a shared-storage buffer of `bytes` bytes.
fn alloc(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, metalerror> {
    device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .ok_or(metalerror::bufferallocation)
}

/// Copies `data` into a shared-storage buffer of matching byte length.
///
/// The caller MUST allocate `target` from `data.len()`; a smaller target would
/// make the copy out of bounds, which is why this is private to the call sites
/// that establish that precondition.
fn write<T>(target: &ProtocolObject<dyn MTLBuffer>, data: &[T]) {
    // SAFETY: `target` was allocated by `alloc` with exactly
    // `size_of_val(data)` bytes of shared storage, so its `contents` pointer
    // covers the region that is copied into it. The buffer outlives the copy
    // and the two pointers are non-overlapping.
    unsafe {
        core::ptr::copy_nonoverlapping(
            data.as_ptr().cast::<u8>(),
            target.contents().as_ptr().cast::<u8>(),
            core::mem::size_of_val(data),
        );
    }
}

/// Copies `len` f32 elements out of a completed shared-storage buffer.
fn read(source: &ProtocolObject<dyn MTLBuffer>, len: usize) -> Result<Vec<f32>, metalerror> {
    let mut out = vec![0.0_f32; len];
    // SAFETY: `source` was allocated with at least `len` f32 elements and the
    // dispatch that wrote it has completed (the command buffer was waited on),
    // so its `contents` pointer yields exactly `len` initialized elements,
    // which `out` has room for.
    unsafe {
        core::ptr::copy_nonoverlapping(
            source.contents().as_ptr().cast::<f32>(),
            out.as_mut_ptr(),
            out.len(),
        );
    }
    Ok(out)
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
