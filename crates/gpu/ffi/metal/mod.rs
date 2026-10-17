//! The Metal backend FFI island, split across three sibling modules.
//!
//! This file holds the device-context declarations ([`cache`], the operand
//! [`buf`] descriptors, the launch geometry helpers, and the pipeline
//! builder); the narrow unsafe driver/session boundary lives in [`drive`],
//! and the safe operator surface in [`ops`].
//!
//! Authority: docs/design.md §21.8. The crate root keeps
//! `#![deny(unsafe_code)]`; the three driver-handoff `unsafe` blocks (one
//! synchronous dispatch, two session blocks) all live in [`drive`], each
//! directly preceded by a `// SAFETY:` comment.
mod drive;
mod ops;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSError, NSString};
use objc2_metal::{
    MTLBuffer, MTLCommandQueue, MTLComputePipelineState, MTLDevice, MTLLibrary, MTLSize,
};

use std::collections::HashMap;
use std::sync::{Mutex, PoisonError};

use crate::engine::gpuerror;
use crate::metal::source;

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
    /// [`crate::metal::source`] can never reuse a stale binary. The map is
    /// device-specific by construction: it lives beside the one device it was
    /// compiled for, so no device belongs in the key.
    pipelines: HashMap<
        (&'static str, &'static str),
        Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    >,
    /// One compiled library per source text.
    ///
    /// Every entry shares the single `grid/metal.c` source, so the library is
    /// compiled once and each entry only builds its pipeline from it.
    libraries: HashMap<&'static str, Retained<ProtocolObject<dyn MTLLibrary>>>,
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
    fn device(&self) -> Result<Retained<ProtocolObject<dyn MTLDevice>>, gpuerror> {
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
    ) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, gpuerror> {
        let mut state = self.lock();
        if let Some(compiled) = state.pipelines.get(&(source, entry)) {
            return Ok(compiled.clone());
        }
        let device = state.device()?;
        let library = match state.libraries.get(source).cloned() {
            Some(library) => library,
            None => {
                let library = compilelibrary(&device, source)?;
                state.libraries.insert(source, library.clone());
                library
            }
        };
        let compiled = pipelinefor(&device, &library, entry)?;
        state.pipelines.insert((source, entry), compiled.clone());
        Ok(compiled)
    }

    /// Returns the device's command queue, opening and caching it once.
    fn queue(&self) -> Result<Retained<ProtocolObject<dyn MTLCommandQueue>>, gpuerror> {
        let mut state = self.lock();
        if let Some(queue) = &state.queue {
            return Ok(queue.clone());
        }
        let device = state.device()?;
        let opened = device
            .newCommandQueue()
            .ok_or_else(|| gpuerror::execution {
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
    fn device(&mut self) -> Result<Retained<ProtocolObject<dyn MTLDevice>>, gpuerror> {
        if let Some(device) = &self.device {
            return Ok(device.clone());
        }
        let opened = objc2_metal::MTLCreateSystemDefaultDevice().ok_or(gpuerror::nodevice)?;
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
    fn new(m: usize, k: usize, n: usize) -> Result<Self, gpuerror> {
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

/// Compiles `source` into a library, once per source text.
fn compilelibrary(
    device: &ProtocolObject<dyn MTLDevice>,
    source: &str,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, gpuerror> {
    // The GRID header is already spliced in front of this source by
    // catalog.rs `msl::SOURCE` (Metal compiles at run time from a
    // string, so a file-system `#include` is unavailable).
    let text = NSString::from_str(source);
    device
        .newLibraryWithSource_options_error(&text, None)
        .map_err(|error| gpuerror::compile {
            kernel: "metal.c".to_string(),
            message: describe(&error),
        })
}

/// Builds the compute pipeline for `entry` from an already compiled library.
fn pipelinefor(
    device: &ProtocolObject<dyn MTLDevice>,
    library: &ProtocolObject<dyn MTLLibrary>,
    entry: &str,
) -> Result<Retained<ProtocolObject<dyn MTLComputePipelineState>>, gpuerror> {
    let name = NSString::from_str(entry);
    let function = library
        .newFunctionWithName(&name)
        .ok_or_else(|| gpuerror::compile {
            kernel: entry.to_string(),
            message: format!("entry point `{entry}` is missing from the compiled library"),
        })?;
    device
        .newComputePipelineStateWithFunction_error(&function)
        .map_err(|error| gpuerror::pipeline {
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

/// Converts a validated extent into the kernel's `uint` parameter.
fn asu32(value: usize) -> Result<u32, gpuerror> {
    u32::try_from(value).map_err(|_| gpuerror::sizeoverflow)
}

/// Builds the one-dimensional grid covering `count` elements.
///
/// The `uint` thread index a flat kernel reads cannot address more than
/// `u32::MAX`, so a larger count is rejected as a size overflow rather than
/// silently truncated.
fn flatgrid(count: usize) -> Result<MTLSize, gpuerror> {
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
fn rowgrid(rows: usize, tile: usize) -> Result<MTLSize, gpuerror> {
    let width = rows.checked_mul(tile).ok_or(gpuerror::sizeoverflow)?;
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

pub(crate) use drive::session;
pub(crate) use ops::*;
