//! The CUDA backend FFI island, split across three sibling modules.
//!
//! This file holds the device-context declarations ([`cache`] and its
//! retained state); the narrow unsafe driver boundary lives in [`drive`],
//! and the safe operator/accumulative-session surface in [`ops`].
//!
//! Authority: docs/design.md §21.8. The crate root keeps
//! `#![deny(unsafe_code)]`; exactly two constructs are `unsafe` — one per
//! **class of platform interaction**, not one per operator — and both live in
//! [`drive`], each directly preceded by a `// SAFETY:` comment.
//! `scripts/checkunsafe.sh` enforces both the count and the comments.
//!
//! The safe functions expose a narrow, run-to-completion interface: upload
//! host slices, run one kernel or one library call, download the result; the
//! retained [`cache`] is the executor-owned device context and compiled kernels
//! and holds no request-scoped value (docs/design.md §21.6).
mod drive;
mod ops;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use cudarc::cublas::CudaBlas;
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};
use cudarc::nvrtc::compile_ptx;

use crate::cuda::source;
use crate::engine::gpuerror;

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
    /// The one module compiled from the single CUDA-family translation unit
    /// [`crate::cuda::source::SOURCE`], loaded on first use.
    ///
    /// `None` means "not compiled yet or the last attempt failed", so a
    /// failure is retried on the next call rather than latched. Compiling
    /// once and resolving every entry through [`CudaModule::load_function`]
    /// turns fifteen NVRTC passes into one (docs/design.md §21.10.3 measured
    /// ~10.8 ms per kernel).
    module: Option<Arc<CudaModule>>,
    /// One resolved kernel per entry.
    ///
    /// The key is the kernel entry name only: every function lives in the
    /// single `module`, so no source text belongs in the key. The map is
    /// device-specific by construction: it lives beside the one device whose
    /// module it was resolved from, so no device belongs in the key either.
    /// Every `CudaFunction` keeps its `CudaModule` — and through it the
    /// context — alive, so a cached kernel cannot outlive the device it was
    /// compiled for.
    functions: HashMap<&'static str, CudaFunction>,
    /// The stream shared by the cuBLAS path.
    ///
    /// The cuBLAS handle is created once and bound to this stream, so every
    /// cuBLAS-served dispatch runs on it. Dispatches of different calls are
    /// therefore ordered by the stream instead of running concurrently, and
    /// each call still allocates its own buffers, so no call ever observes
    /// another call's data.
    librarystream: Option<Arc<CudaStream>>,
    /// The cuBLAS handle created on first use and reused by every
    /// [`matmul_cublas`] call.
    blas: Option<Arc<CudaBlas>>,
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
    fn stream(&self) -> Result<Arc<CudaStream>, gpuerror> {
        let device = self.device()?;
        device.new_stream().map_err(|error| gpuerror::device {
            message: error.to_string(),
        })
    }

    /// Returns the retained device, opening it once.
    fn device(&self) -> Result<Arc<CudaContext>, gpuerror> {
        self.lock().device()
    }

    /// Returns the shared cuBLAS handle and its stream, creating both once.
    ///
    /// The lock is held only across a creation miss; the handle and stream
    /// are cloned out and every transfer, launch, and synchronization happens
    /// after the lock is released, exactly like the resolved kernels.
    fn blas(&self) -> Result<(Arc<CudaBlas>, Arc<CudaStream>), gpuerror> {
        self.lock().blas()
    }

    /// Returns the resolved kernel for `entry`, compiling the module once.
    ///
    /// The first call compiles the whole translation unit with NVRTC and
    /// loads the module; every later call — for this entry or any other —
    /// resolves the entry out of the cached module, which is a driver lookup
    /// and not a compile. The lock is held across the miss, its NVRTC
    /// compilation, and the module load, which serializes concurrent first
    /// calls instead of compiling several times over; it is released before
    /// the caller moves any operand, so no transfer, launch, or
    /// synchronization ever happens under the lock.
    fn function(&self, entry: &'static str) -> Result<CudaFunction, gpuerror> {
        let mut state = self.lock();
        if let Some(compiled) = state.functions.get(entry) {
            return Ok(compiled.clone());
        }
        let device = state.device()?;
        if state.module.is_none() {
            let ptx = compile_ptx(source::SOURCE).map_err(|error| gpuerror::compile {
                kernel: entry.to_string(),
                message: error.to_string(),
            })?;
            let module = device.load_module(ptx).map_err(|error| gpuerror::module {
                kernel: entry.to_string(),
                message: error.to_string(),
            })?;
            state.module = Some(module);
        }
        let module = state.module.clone().ok_or_else(|| gpuerror::module {
            kernel: entry.to_string(),
            message: "module not compiled".into(),
        })?;
        let function = module
            .load_function(entry)
            .map_err(|error| gpuerror::module {
                kernel: entry.to_string(),
                message: error.to_string(),
            })?;
        state.functions.insert(entry, function.clone());
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
    fn device(&mut self) -> Result<Arc<CudaContext>, gpuerror> {
        if let Some(device) = &self.device {
            return Ok(device.clone());
        }
        let opened = CudaContext::new(0).map_err(|error| gpuerror::device {
            message: error.to_string(),
        })?;
        self.device = Some(opened.clone());
        Ok(opened)
    }

    /// Returns the shared library stream, creating it once.
    ///
    /// The stream owns a reference to the retained device, so it stays valid
    /// as long as the cache does.
    fn librarystream(&mut self) -> Result<Arc<CudaStream>, gpuerror> {
        if let Some(stream) = &self.librarystream {
            return Ok(stream.clone());
        }
        let device = self.device()?;
        let opened = device.new_stream().map_err(|error| gpuerror::device {
            message: error.to_string(),
        })?;
        self.librarystream = Some(opened.clone());
        Ok(opened)
    }

    /// Returns the cuBLAS handle bound to the shared library stream, creating
    /// it once. Creation happens under the cache lock, which serializes
    /// concurrent first calls instead of creating several handles.
    fn blas(&mut self) -> Result<(Arc<CudaBlas>, Arc<CudaStream>), gpuerror> {
        if self.blas.is_none() {
            let stream = self.librarystream()?;
            let opened = CudaBlas::new(stream.clone()).map_err(|error| gpuerror::blas {
                message: error.to_string(),
            })?;
            self.blas = Some(Arc::new(opened));
        }
        let blas = self.blas.clone().ok_or(gpuerror::blas {
            message: "cuBLAS handle not available".into(),
        })?;
        let stream = self.librarystream()?;
        Ok((blas, stream))
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

pub(crate) use ops::*;
