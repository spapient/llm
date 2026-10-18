#![allow(unsafe_code)]
//! The Metal driver-handoff boundary: the only file in this island that
//! invokes Objective-C `unsafe` APIs.
//!
//! Authority: docs/design.md §21.8. Three registered `unsafe` blocks, one per
//! driver handoff: the synchronous [`dispatch`], the session encoder
//! ([`session::dispatch`]), and the session commit in [`session::apply`].
//! Each is directly preceded by a `// SAFETY:` comment.
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;

use objc2_metal::MTLBlitCommandEncoder;

/// A queued device-to-device row copy: `(destination, offset, source, bytes)`.
///
/// The tuple keeps the destination and its offset adjacent so the blit pass
/// in `apply` can encode each copy without re-deriving the row geometry.
type appendlist = Vec<(
    Retained<ProtocolObject<dyn MTLBuffer>>,
    usize,
    Retained<ProtocolObject<dyn MTLBuffer>>,
    usize,
)>;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLDevice, MTLResourceOptions, MTLSize,
};
use std::cell::RefCell;

use crate::engine::gpuerror;

use super::{buf, cache};

/// The allocation and transfer plan for one dispatch.
struct bufferplan<'a> {
    allocbytes: Vec<usize>,
    uploads: Vec<(usize, &'a [f32])>,
    uploads64: Vec<(usize, &'a [u64])>,
    outputs: Vec<(usize, usize)>,
}

/// Splits the buffer slots into allocation sizes, host uploads, and outputs.
fn planbuffers<'a>(buffers: &[buf<'a>]) -> Result<bufferplan<'a>, gpuerror> {
    let mut allocbytes = Vec::with_capacity(buffers.len());
    let mut uploads = Vec::new();
    let mut uploads64 = Vec::new();
    let mut outputs = Vec::new();
    for (index, slot) in buffers.iter().enumerate() {
        match *slot {
            buf::host(data) => {
                allocbytes.push(
                    data.len()
                        .checked_mul(core::mem::size_of::<f32>())
                        .ok_or(gpuerror::sizeoverflow)?,
                );
                uploads.push((index, data));
            }
            buf::hostu64(data) => {
                allocbytes.push(
                    data.len()
                        .checked_mul(core::mem::size_of::<u64>())
                        .ok_or(gpuerror::sizeoverflow)?,
                );
                uploads64.push((index, data));
            }
            buf::out(count) => {
                let bytes = count
                    .checked_mul(core::mem::size_of::<f32>())
                    .ok_or(gpuerror::sizeoverflow)?;
                allocbytes.push(bytes);
                outputs.push((index, count));
            }
            buf::readback(_, count) => {
                outputs.push((index, count));
            }
            buf::device(_) => {}
        }
    }
    Ok(bufferplan {
        allocbytes,
        uploads,
        uploads64,
        outputs,
    })
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
/// Returns [`gpuerror`] when compilation or pipeline creation fails, or when
/// the device, queue, or command buffer declines to hand out the resources the
/// dispatch needs. Output buffers are reserved before their contents are read,
/// so a host allocation failure surfaces as [`gpuerror::allocation`] rather
/// than aborting.
pub(super) fn dispatch<P>(
    cache: &cache,
    source: &'static str,
    entry: &'static str,
    buffers: &[buf<'_>],
    params: &P,
    grid: MTLSize,
    threads: MTLSize,
) -> Result<Vec<Vec<f32>>, gpuerror> {
    // The device, the pipeline, and the queue all come from the same cache, so
    // the pipeline cannot be built for one device and used on another. The
    // queue is opened once and reused; only the command buffer is per-dispatch
    // (Metal command buffers are single-use by contract).
    let device = cache.device()?;
    let pipeline = cache.pipeline(source, entry)?;
    let queue = cache.queue()?;
    let command = queue.commandBuffer().ok_or_else(|| gpuerror::execution {
        message: "the queue returned no command buffer".to_string(),
    })?;
    let encoder = command
        .computeCommandEncoder()
        .ok_or_else(|| gpuerror::execution {
            message: "the command buffer returned no compute encoder".to_string(),
        })?;
    encoder.setComputePipelineState(&pipeline);

    // Allocate the device buffers this dispatch owns in one pass, then line up
    // the bound references in a second pass. Splitting the passes keeps the
    // borrows disjoint: nothing borrows `owned` while it is still being pushed
    // to. `owned` lives for the whole function, so no buffer is freed while a
    // kernel can still touch it.
    let bufferplan {
        allocbytes,
        uploads,
        uploads64,
        outputs,
    } = planbuffers(buffers)?;
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
                .map_err(|_| gpuerror::allocation)?;
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
pub(super) fn alloc(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: usize,
) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, gpuerror> {
    device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .ok_or(gpuerror::allocation)
}

/// The pipeline key and thread geometry of one dispatch, packed so a launch
/// reads as one argument rather than four.
pub(super) struct dispatchspec {
    pub(super) source: &'static str,
    pub(super) entry: &'static str,
    pub(super) grid: MTLSize,
    pub(super) threads: MTLSize,
}

/// One buffer slot of an asynchronous session dispatch, in kernel binding
/// order. The last slot MUST be [`sbuf::out`], the output the slot records.
pub(super) enum sbuf<'a> {
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

/// The slot-space marker for persistent device buffers registered through
/// [`session::persist`]. Persistent slots live in their own table (they must
/// survive every `apply` without being downloaded as step outputs), so their
/// index carries this high bit and `getbuffer` routes on it.
const PERSIST_BASE: usize = 1usize << 63;

/// The allocation and upload plan for one session dispatch.
struct sessionplan<'a> {
    allocbytes: Vec<usize>,
    uploads64: Vec<(usize, &'a [u64])>,
}

/// Splits the session buffer slots into allocation sizes and index uploads.
fn plansessionbuffers<'a>(buffers: &[sbuf<'a>]) -> Result<sessionplan<'a>, gpuerror> {
    let mut allocbytes = Vec::with_capacity(buffers.len());
    let mut uploads64 = Vec::new();
    for (index, slot) in buffers.iter().enumerate() {
        match *slot {
            sbuf::hostu64(data) => {
                allocbytes.push(
                    data.len()
                        .checked_mul(core::mem::size_of::<u64>())
                        .ok_or(gpuerror::sizeoverflow)?,
                );
                uploads64.push((index, data));
            }
            sbuf::out(count) => {
                let bytes = count
                    .checked_mul(core::mem::size_of::<f32>())
                    .ok_or(gpuerror::sizeoverflow)?;
                allocbytes.push(bytes);
            }
            sbuf::device(_) => {}
        }
    }
    Ok(sessionplan {
        allocbytes,
        uploads64,
    })
}

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
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
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
    /// Persistent device buffers registered through [`session::persist`],
    /// retained across every apply so a device KV cache keeps its key/value
    /// rows resident. The pending queue flushes their contents on the next
    /// dispatch, exactly like ordinary uploads.
    persisted: Vec<Retained<ProtocolObject<dyn MTLBuffer>>>,
    /// Device-to-device row copies queued by [`session::recordappend`]: each
    /// `(destination, offset, source, bytes)` copies a new key/value row into a
    /// persistent cache buffer. The copies are encoded by `apply` as one blit
    /// pass after the compute encoder ends and before commit, inside the same
    /// unsafe block as the commit and the downloads, so the site count does
    /// not change.
    appends: appendlist,
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
    pub(crate) fn new(cache: &cache) -> Result<Self, gpuerror> {
        let device = cache.device()?;
        let queue = cache.queue()?;
        let command = queue.commandBuffer().ok_or_else(|| gpuerror::execution {
            message: "the queue returned no command buffer".to_string(),
        })?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| gpuerror::execution {
                message: "the command buffer returned no compute encoder".to_string(),
            })?;
        Ok(Self {
            device,
            queue,
            command,
            encoder,
            outputs: Vec::new(),
            pending: RefCell::new(Vec::new()),
            persisted: Vec::new(),
            appends: Vec::new(),
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
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, gpuerror> {
        let buffer = alloc(&self.device, core::mem::size_of_val(data))?;
        self.pending
            .borrow_mut()
            .push((buffer.clone(), data.to_vec()));
        Ok(buffer)
    }

    /// Registers `data` as a persistent device-resident buffer.
    ///
    /// The buffer is retained in the session's persistent table (never
    /// downloaded as a step output) and its contents are flushed into it by
    /// the next dispatch, inside that dispatch's single unsafe block. The
    /// returned slot carries [`PERSIST_BASE`], so `getbuffer` routes it to the
    /// persistent table even after `apply` rebuilt the output list.
    pub(crate) fn persist(&mut self, data: &[f32]) -> Result<usize, gpuerror> {
        let buffer = alloc(&self.device, core::mem::size_of_val(data))?;
        self.pending
            .borrow_mut()
            .push((buffer.clone(), data.to_vec()));
        let index = self.persisted.len();
        self.persisted.push(buffer);
        Ok(PERSIST_BASE | index)
    }

    /// Queues a device-to-device row copy into a persistent buffer.
    ///
    /// `bytes` is one row (`width * size_of::<f32>()`) and `offset` its
    /// destination row start; both ranges are validated here, and the copy
    /// itself is encoded by `apply` as one blit pass, so no host round trip
    /// ever carries a cached row.
    pub(crate) fn recordappend(
        &mut self,
        destination: &Retained<ProtocolObject<dyn MTLBuffer>>,
        offset: usize,
        source: &Retained<ProtocolObject<dyn MTLBuffer>>,
        bytes: usize,
    ) -> Result<(), gpuerror> {
        let end = offset.checked_add(bytes).ok_or(gpuerror::sizeoverflow)?;
        if end > destination.length() {
            return Err(gpuerror::execution {
                message: "append destination out of range".to_string(),
            });
        }
        if bytes > source.length() {
            return Err(gpuerror::execution {
                message: "append source out of range".to_string(),
            });
        }
        self.appends
            .push((destination.clone(), offset, source.clone(), bytes));
        Ok(())
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
    pub(super) fn dispatch<T>(
        &mut self,
        cache: &cache,
        spec: &dispatchspec,
        buffers: &[sbuf<'_>],
        params: &T,
        outlen: usize,
    ) -> Result<usize, gpuerror> {
        let pipeline = cache.pipeline(spec.source, spec.entry)?;
        self.encoder.setComputePipelineState(&pipeline);

        // Allocate the buffers this dispatch owns in one pass, then line up
        // the bound references in a second pass, keeping the borrows disjoint.
        let sessionplan {
            allocbytes,
            uploads64,
        } = plansessionbuffers(buffers)?;
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
        let out = out.ok_or_else(|| gpuerror::execution {
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
    pub(super) fn recordempty(&mut self) -> usize {
        let index = self.outputs.len();
        self.outputs.push(slot::empty);
        index
    }

    /// Records an all-zero output of `count` elements without touching the device.
    pub(super) fn recordzeros(&mut self, count: usize) -> usize {
        let index = self.outputs.len();
        self.outputs.push(slot::zeros(count));
        index
    }

    /// Returns the device buffer of slot `index`, when that slot holds one.
    pub(crate) fn getbuffer(
        &self,
        index: usize,
    ) -> Option<&Retained<ProtocolObject<dyn MTLBuffer>>> {
        if index & PERSIST_BASE != 0 {
            return self.persisted.get(index & !PERSIST_BASE);
        }
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
            _ => {
                // A persistent slot names a buffer that lives on the device and
                // is not part of the per-apply download list; copy it out here
                // on demand. The leading-window view re-tags such a slot, so a
                // caller downloading that view receives the real cache prefix.
                if index >= PERSIST_BASE {
                    let persist_index = index & !PERSIST_BASE;
                    let buffer = self.persisted.get(persist_index)?;
                    let len = buffer.length() / 4;
                    let mut out = vec![0.0_f32; len];
                    // SAFETY: `contents` is valid for the buffer's length and
                    // `out` is exactly that length in floats; the command
                    // buffer has already been committed and waited on by
                    // `apply`, so the read races nothing.
                    unsafe {
                        core::ptr::copy_nonoverlapping::<f32>(
                            buffer.contents().as_ptr().cast::<f32>(),
                            out.as_mut_ptr(),
                            len,
                        );
                    }
                    Some(out)
                } else {
                    None
                }
            }
        }
    }

    /// Ends encoding, commits, waits, then downloads every recorded output.
    ///
    /// The returned vectors are in slot order, so a slot index obtained from a
    /// `dispatch` / `recordempty` / `recordzeros` call indexes into this result.
    pub(crate) fn apply(&mut self) -> Result<Vec<Vec<f32>>, gpuerror> {
        self.encoder.endEncoding();
        self.ended = true;
        // `session` implements `Drop`, so fields cannot be moved out of `self`;
        // `take` leaves empty vectors behind and hands the slots over.
        let appends = core::mem::take(&mut self.appends);
        let outputs = core::mem::take(&mut self.outputs);
        let mut results: Vec<Vec<f32>> = Vec::with_capacity(outputs.len());
        // SAFETY: one unsafe block serializes the blit pass, the commit, and
        // the downloads on the same command buffer:
        // * every `(destination, offset, source, bytes)` append was validated
        //   by `recordappend` to fit both buffers, and the copies are encoded
        //   after the compute encoder ended, so each new key/value row lands in
        //   its persistent cache row after the kernel that produced it;
        // * `commit` submits that ordered work and `waitUntilCompleted` makes
        //   every download below read post-kernel data;
        // * every `slot::buffer` was allocated with at least `len` f32 elements
        //   by the dispatch that recorded it, so each `contents` pointer yields
        //   exactly `len` initialized elements, which the reserved `out` has
        //   room for.
        unsafe {
            // Flush the pending host uploads into their buffers first: a
            // persist-only session (no dispatch) still needs the initial
            // key/value rows resident before the blit pass reads them, and
            // every buffer here was allocated with exactly `size_of_val(data)`
            // bytes, so the copies are in bounds.
            let pending = core::mem::take(&mut *self.pending.borrow_mut());
            for (buffer, data) in pending {
                core::ptr::copy_nonoverlapping(
                    data.as_ptr().cast::<u8>(),
                    buffer.contents().as_ptr().cast::<u8>(),
                    core::mem::size_of_val(&data[..]),
                );
            }
            if !appends.is_empty() {
                let blit =
                    self.command
                        .blitCommandEncoder()
                        .ok_or_else(|| gpuerror::execution {
                            message: "the command buffer returned no blit encoder".to_string(),
                        })?;
                for (destination, offset, source, bytes) in appends {
                    blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                        &source,
                        0,
                        &destination,
                        offset,
                        bytes,
                    );
                }
                blit.endEncoding();
            }
            self.command.commit();
            self.command.waitUntilCompleted();
            for output in outputs {
                match output {
                    slot::buffer(buffer, len) => {
                        let mut out = Vec::new();
                        out.try_reserve_exact(len)
                            .map_err(|_| gpuerror::allocation)?;
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
        // A persistent session starts the next chain on a fresh command
        // buffer: Metal command buffers are single-use by contract.
        self.pending.borrow_mut().clear();
        let command = self
            .queue
            .commandBuffer()
            .ok_or_else(|| gpuerror::execution {
                message: "the queue returned no command buffer".to_string(),
            })?;
        let encoder = command
            .computeCommandEncoder()
            .ok_or_else(|| gpuerror::execution {
                message: "the command buffer returned no compute encoder".to_string(),
            })?;
        self.command = command;
        self.encoder = encoder;
        self.ended = false;
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
