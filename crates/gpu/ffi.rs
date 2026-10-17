//! The CUDA and MUSA FFI islands.
//!
//! Both backends are Linux-only and drive their devices through a C API, so
//! their unsafe islands share one file with a module per backend. Each module
//! declares its own `#![allow(unsafe_code)]`, so the per-backend island model
//! is preserved; `scripts/checkunsafe.sh` registers the file's combined site
//! count (cuda 2 + musa 2 = 4).
//!
//! Metal's bindings are Objective-C and therefore stay in `metal/ffi.rs`.

#[cfg(target_os = "linux")]
pub(crate) mod cuda {
    #![allow(unsafe_code)]
    //! The single platform-FFI island of the CUDA backend.
    //!
    //! Authority: docs/design.md §21.8. This is the only file in the crate that
    //! declares `#![allow(unsafe_code)]`; the crate root keeps
    //! `#![deny(unsafe_code)]`. Exactly two constructs are `unsafe` — one per
    //! **class of platform interaction**, not one per operator — and each is
    //! directly preceded by a `// SAFETY:` comment. `scripts/checkunsafe.sh`
    //! enforces both the count and the comments.
    //!
    //! The two classes are:
    //!
    //! * [`submit`] — the NVRTC kernel launch. Every in-repo kernel reaches the
    //!   driver through it, so the count does not grow with the kernel count.
    //! * [`matmul_cublas`] — the cuBLAS `gemm` call.
    //!
    //! The safe functions expose a narrow, run-to-completion interface: upload host
    //! slices, run one kernel or one library call, download the result. NVRTC
    //! kernels take a fresh stream per call so concurrent calls proceed in
    //! parallel; the cuBLAS path shares one cached stream because its handle is
    //! created once and bound to it, which also orders concurrent cuBLAS calls.
    //! Every call still allocates and releases its own buffers by scope, so no
    //! execution state outlives a call. The only state that does survive is
    //! [`cache`] — the retained device context, the compiled kernels, and the
    //! cuBLAS handle — which the executor owns and which holds no request-scoped
    //! value (docs/design.md §21.6).
    //!
    //! The launch helpers take the validated plans produced by [`api::geometry`]
    //! rather than loose extents, so no number is re-derived after it was checked.

    use core::ffi::{c_int, c_longlong};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, PoisonError};

    use cudarc::cublas::{
        CudaBlas, Gemm, GemmConfig, StridedBatchedConfig, sys::cublasOperation_t,
    };
    use cudarc::driver::{
        CudaContext, CudaFunction, CudaSlice, CudaStream, DeviceRepr, LaunchArgs, LaunchConfig,
        PushKernelArg,
    };
    use cudarc::nvrtc::compile_ptx;

    use crate::cuda::source;
    use crate::engine::gpuerror;
    use api::geometry::{attentionshape, ropeshape};

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
        /// [`crate::cuda::source`] can never reuse a stale module. The map is
        /// device-specific by construction: it lives beside the one device whose
        /// module it was resolved from, so no device belongs in the key. Every
        /// `CudaFunction` keeps its `CudaModule` — and through it the context —
        /// alive, so a cached kernel cannot outlive the device it was compiled for.
        functions: HashMap<(&'static str, &'static str), CudaFunction>,
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

        /// Returns the resolved kernel for `(text, entry)`, compiling it once.
        ///
        /// The lock is held across a miss, its NVRTC compilation, and the module
        /// load. That serializes concurrent first calls for the same kernel instead
        /// of compiling it several times over, and it is released before the caller
        /// moves any operand: no transfer, launch, or synchronization ever happens
        /// under the lock.
        fn function(
            &self,
            text: &'static str,
            entry: &'static str,
        ) -> Result<CudaFunction, gpuerror> {
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

    /// Compiles `text` with NVRTC, loads the module, and resolves `entry`.
    fn compile(
        device: &Arc<CudaContext>,
        text: &str,
        entry: &str,
    ) -> Result<CudaFunction, gpuerror> {
        let ptx = compile_ptx(text).map_err(|error| gpuerror::compile {
            kernel: entry.to_string(),
            message: error.to_string(),
        })?;
        let module = device.load_module(ptx).map_err(|error| gpuerror::module {
            kernel: entry.to_string(),
            message: error.to_string(),
        })?;
        module
            .load_function(entry)
            .map_err(|error| gpuerror::module {
                kernel: entry.to_string(),
                message: error.to_string(),
            })
    }

    /// Converts a validated extent into the `c_int` a host-side library takes.
    fn asint(value: u32) -> Result<c_int, gpuerror> {
        c_int::try_from(value).map_err(|_| gpuerror::sizeoverflow)
    }

    /// Converts a validated extent into the `unsigned` a kernel parameter takes.
    fn asunsigned(value: usize) -> Result<u32, gpuerror> {
        u32::try_from(value).map_err(|_| gpuerror::sizeoverflow)
    }

    /// Builds the one-dimensional launch configuration covering `count` elements.
    fn flatconfig(count: usize) -> Result<LaunchConfig, gpuerror> {
        let tile = source::ELEMENTWISETILE;
        let grid = count.div_ceil(tile as usize);
        Ok(LaunchConfig {
            grid_dim: (asunsigned(grid)?, 1, 1),
            block_dim: (tile, 1, 1),
            shared_mem_bytes: 0,
        })
    }

    /// Copies a host slice to a fresh device allocation.
    fn upload<T: DeviceRepr>(
        stream: &Arc<CudaStream>,
        src: &[T],
    ) -> Result<CudaSlice<T>, gpuerror> {
        stream.clone_htod(src).map_err(|error| gpuerror::transfer {
            message: error.to_string(),
        })
    }

    /// Allocates `len` zeroed device elements.
    fn alloc(stream: &Arc<CudaStream>, len: usize) -> Result<CudaSlice<f32>, gpuerror> {
        stream
            .alloc_zeros::<f32>(len)
            .map_err(|error| gpuerror::transfer {
                message: error.to_string(),
            })
    }

    /// Waits for `stream` and copies a device allocation back to the host.
    fn syncanddownload(
        stream: &Arc<CudaStream>,
        src: &CudaSlice<f32>,
    ) -> Result<Vec<f32>, gpuerror> {
        stream.synchronize().map_err(|error| gpuerror::transfer {
            message: error.to_string(),
        })?;
        stream.clone_dtoh(src).map_err(|error| gpuerror::transfer {
            message: error.to_string(),
        })
    }

    /// Submits a prepared kernel launch to the driver.
    ///
    /// This is the crate's single kernel-launch site, and it is a function rather
    /// than an inline block so the registered count does not grow with the number
    /// of kernels: callers assemble the argument list with the safe
    /// `LaunchArgs::arg` builder and only the submit itself is unchecked.
    fn submit(args: &mut LaunchArgs<'_>, config: LaunchConfig) -> Result<(), gpuerror> {
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
            .map_err(|error| gpuerror::launch {
                message: error.to_string(),
            })
    }

    /// Runs one cuBLAS strided-batched `gemm`, the crate's single cuBLAS site.
    ///
    /// This is a function rather than an inline block so the registered count stays
    /// one: the single-operator [`matmul_cublas`] and the accumulative submission
    /// path both call it, and neither adds an `unsafe` site of its own. The batched
    /// form serves single matrices too (batch size one), so the one site covers
    /// both call shapes.
    fn gemm(
        blas: &CudaBlas,
        config: StridedBatchedConfig<f32>,
        db: &CudaSlice<f32>,
        da: &CudaSlice<f32>,
        dc: &mut CudaSlice<f32>,
    ) -> Result<(), gpuerror> {
        // SAFETY: `da`, `db`, and `dc` are live device allocations. `da` holds the
        // `[batch, m, k]` left operand and `db` the `[batch, k, n]` right operand;
        // read as column-major they are the `k x m` and `n x k` matrices the
        // `lda`/`ldb` leading dimensions describe, and `dc` is written as an
        // `n x m` column-major matrix with leading dimension `ldc`, i.e. the
        // row-major `[m, n]` result. The per-slice strides `stride_a`/`stride_b`/
        // `stride_c` are the exact element counts of one slice of each operand, so
        // every batch slice stays inside its allocation. Every dimension is at
        // least 1 and fits `c_int`, so no out-of-bounds access is possible.
        unsafe {
            blas.gemm_strided_batched(config, db, da, dc)
                .map_err(|error| gpuerror::blas {
                    message: error.to_string(),
                })?;
        }
        Ok(())
    }

    /// Computes `[m, k] * [k, n]` with the in-repo NVRTC kernel.
    ///
    /// This is the independent reference path: the kernel is compiled from
    /// [`MATMULCUDA`](crate::cuda::source::MATMULCUDA) at run time, so it shares no code
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

    /// Numerically stable softmax over the last axis with the in-repo NVRTC kernel.
    ///
    /// `input` is read as `[rows, width]`. The kernel is compiled from
    /// [`SOFTMAXCUDA`](crate::cuda::source::SOFTMAXCUDA) at run time through the cached
    /// kernel machinery, so every call after the first pays upload, launch, and
    /// download only. It is the crate's only softmax implementation — the cuDNN
    /// path was removed once this kernel reached parity (docs/design.md §21.8).
    ///
    /// An input containing any non-finite value is computed by [`scalarsoftmax`],
    /// the same formulation `cpu` and `metal` implement: the kernel seeds its
    /// maximum with the most negative finite `f32`, so an all-negative-infinity row
    /// would otherwise come back as zeros instead of propagating NaN.
    ///
    /// # Errors
    ///
    /// Returns [`gpuerror`] when the device cannot be opened, a transfer fails, a
    /// dimension does not fit `c_int`, or the kernel launch fails.
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
        if input.iter().any(|value| !value.is_finite()) {
            return Ok(scalarsoftmax(input, width));
        }
        let stream = cache.stream()?;
        let function = cache.function(source::SOFTMAXCUDA, source::SOFTMAXENTRY)?;
        let dx = upload(&stream, input)?;
        let mut dy = alloc(&stream, total)?;
        let widthu = asunsigned(width)?;
        let config = LaunchConfig {
            grid_dim: (asunsigned(rows)?, 1, 1),
            block_dim: (source::ROWTILE, 1, 1),
            shared_mem_bytes: source::ROWTILE * 4,
        };
        let mut args = stream.launch_builder(&function);
        args.arg(&dx).arg(&mut dy).arg(&widthu);
        submit(&mut args, config)?;
        syncanddownload(&stream, &dy)
    }

    /// The scalar max-subtracting softmax `cpu` and `metal` implement.
    ///
    /// Private to this file: the single-operator [`softmax`] defers to it for
    /// inputs outside the finite domain, where the NVRTC kernel's finite seed would
    /// turn a negative-infinity row into zeros. The row rule is the one
    /// `crates/cpu/src/softmax.rs` documents — subtract the row maximum,
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
    /// Returns [`gpuerror`] when the device cannot be opened, the source fails to
    /// compile, a transfer fails, the length does not fit `unsigned`, or the launch
    /// fails.
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
    ) -> Result<Vec<f32>, gpuerror> {
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
    /// Returns [`gpuerror`] when the device cannot be opened, the source fails to
    /// compile, a transfer fails, an extent does not fit its device type, or the
    /// launch fails.
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
    /// Returns [`gpuerror`] when the device cannot be opened, the source fails to
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
    ) -> Result<Vec<f32>, gpuerror> {
        if rows == 0 {
            return Ok(Vec::new());
        }
        let pairs = rows
            .checked_mul(heads)
            .and_then(|value| value.checked_mul(headdim / 2))
            .ok_or(gpuerror::sizeoverflow)?;
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
    /// Returns [`gpuerror`] when the device cannot be opened, the source fails to
    /// compile, a transfer fails, an extent does not fit its device type, or the
    /// launch fails.
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
    /// Returns [`gpuerror`] when the device cannot be opened, the source fails to
    /// compile, a transfer fails, the launch geometry or the shared-memory request
    /// overflows, or the launch fails.
    pub(crate) fn attention(
        cache: &cache,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        geometry: &attentionshape,
        causal: bool,
    ) -> Result<Vec<f32>, gpuerror> {
        let qseq = geometry.qseq;
        if qseq == 0 {
            return Ok(Vec::new());
        }
        let queryelements = qseq
            .checked_mul(geometry.heads)
            .and_then(|rows| rows.checked_mul(geometry.headdim))
            .ok_or(gpuerror::sizeoverflow)?;
        let stream = cache.stream()?;
        let function = cache.function(source::ATTENTIONCUDA, source::ATTENTIONENTRY)?;
        let dquery = upload(&stream, query)?;
        let dkey = upload(&stream, key)?;
        let dvalue = upload(&stream, value)?;
        let mut dout = alloc(&stream, queryelements)?;
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
        let mut args = stream.launch_builder(&function);
        args.arg(&dquery)
            .arg(&dkey)
            .arg(&dvalue)
            .arg(&mut dout)
            .arg(&qseqv)
            .arg(&kvseqv)
            .arg(&headsv)
            .arg(&kvheadsv)
            .arg(&dimv)
            .arg(&causalv);
        submit(&mut args, config)?;
        syncanddownload(&stream, &dout)
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
        let function = cache.function(source::SCANCUDA, source::SCANENTRY)?;
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
        let agg = cache.function(source::SCANPARALLELCUDA, source::SCANAGGENTRY)?;
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
        let prefix = cache.function(source::SCANPARALLELCUDA, source::SCANPREFIXENTRY)?;
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
        let apply = cache.function(source::SCANPARALLELCUDA, source::SCANAPPLYENTRY)?;
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
    pub(crate) struct session {
        stream: Arc<CudaStream>,
        outputs: Vec<slot>,
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

        /// Returns the device slice of slot `index`, when that slot holds one.
        pub(crate) fn getbuffer(&self, index: usize) -> Option<&CudaSlice<f32>> {
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
                _ => None,
            }
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
        let function = cache.function(source::MATMULCUDA, source::MATMULENTRY)?;
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
        let function = cache.function(source::ELEMENTWISE, entry)?;
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
        let function = cache.function(source::ELEMENTWISE, source::SILUENTRY)?;
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
        let function = cache.function(source::SOFTMAXCUDA, source::SOFTMAXENTRY)?;
        let mut dy = session.alloc(total)?;
        let widthu = asunsigned(width)?;
        let config = LaunchConfig {
            grid_dim: (asunsigned(rows)?, 1, 1),
            block_dim: (source::ROWTILE, 1, 1),
            shared_mem_bytes: source::ROWTILE * 4,
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
        let function = cache.function(source::RMSNORMCUDA, source::RMSNORMENTRY)?;
        let mut dy = session.alloc(total)?;
        let widthu = asunsigned(width)?;
        let config = LaunchConfig {
            grid_dim: (asunsigned(rows)?, 1, 1),
            block_dim: (source::ROWTILE, 1, 1),
            shared_mem_bytes: source::ROWTILE * 4,
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
        let function = cache.function(source::ROPECUDA, source::ROPEENTRY)?;
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
        let function = cache.function(source::ROPEATCUDA, source::ROPEATENTRY)?;
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
        let function = cache.function(source::EMBEDDINGCUDA, source::EMBEDDINGENTRY)?;
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
        let function = cache.function(source::ATTENTIONCUDA, source::ATTENTIONENTRY)?;
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
        let function = cache.function(source::ATTENTIONCACHEDCUDA, source::ATTENTIONCACHEDENTRY)?;
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
}

#[cfg(all(target_os = "linux", musa_toolkit))]
pub(crate) mod musa {
    #![allow(unsafe_code)]
    //! The MUSA platform FFI island: every `mu*` driver call in the crate lives
    //! here, and the crate root keeps `#![deny(unsafe_code)]`. Each `unsafe`
    //! construct is directly preceded by a `// SAFETY:` comment; the registered
    //! site count (two) is enforced by `scripts/checkunsafe.sh`.
    //!
    //! The two sites are the 2024-edition lower bound for a hand-written FFI
    //! (docs/musa-backend-plan.md §5.0, §5.4):
    //! 1. the `unsafe extern "C"` declaration island mirroring `musa.h`;
    //! 2. the single driver-handoff block inside [`dispatch`], which is lazy by
    //!    design: device open, module load, function resolution, allocation,
    //!    transfers, launch, synchronisation, and freeing all happen inside that
    //!    one block, so no second handoff site exists.
    //!
    //! Every safe function exposed to the executor validates shapes and budgets
    //! first, then describes its operands as a [`buf`] list and forwards to
    //! [`dispatch`] — a narrow run-to-completion interface like the `metal`
    //! backend's `dispatch`, keeping the three GPU backends structurally aligned.

    use std::collections::HashMap;
    use std::ffi::CString;
    use std::sync::Arc;
    use std::sync::{Mutex, PoisonError};

    use crate::engine::gpuerror;
    use crate::musa::registry::KERNELS;
    use crate::musa::source;

    // ---------------------------------------------------------------------------
    // MUSA driver ABI (musa.h, MUSA Toolkit 3.1.0). The opaque handles are
    // one-field repr(C) wrappers; the device pointer is a byte address.
    // ---------------------------------------------------------------------------

    /// The MUSA driver result code; `MUSA_SUCCESS` is zero.
    pub(crate) type muresult = i32;

    /// The success code every `mu*` call must return.
    const MUSA_SUCCESS: muresult = 0;

    /// An opaque MUSA device handle: the address of the driver's opaque device
    /// object, represented as `usize` so the cached state stays `Send + Sync`
    /// (the two-site unsafe budget does not allow an `unsafe impl`; the address
    /// is a plain integer, so no raw pointer crosses a thread boundary).
    pub(crate) type mudevice = usize;

    /// An opaque MUSA context handle (see [`mudevice`]).
    pub(crate) type mucontext = usize;

    /// An opaque MUSA module handle (see [`mudevice`]).
    pub(crate) type mumodule = usize;

    /// An opaque MUSA kernel handle (see [`mudevice`]).
    pub(crate) type mufunction = usize;

    /// A device memory pointer: a byte address into the device address space.
    pub(crate) type mudevptr = usize;

    /// The null stream: launches go to the default stream, which orders
    /// synchronously in this backend's run-to-completion contract.
    pub(crate) const MUNULLSTREAM: usize = 0;

    /// The maximum number of buffers a pool bucket keeps per byte size before an
    /// excess buffer is freed back to the driver. It bounds device memory to a
    /// small multiple of the largest operand set while still absorbing the
    /// per-dispatch allocation churn (docs/musa-backend-plan.md §5.5).
    const POOLCAP: usize = 4;

    // SAFETY: Every declaration mirrors the C ABI of `musa.h` (MUSA Toolkit
    // 3.1.0): the opaque handles are pointer-sized `repr(C)` structs, `MUdeviceptr`
    // is a byte address, and `muLaunchKernel`'s `kernelParams` is a pointer to an
    // array of pointers, one per kernel parameter, each pointing at a value whose
    // type matches the kernel signature the caller mirrors. All calls happen only
    // inside the single `unsafe` block of [`dispatch`], guarded by result-code
    // checks, so no unvalidated driver call can escape this island.
    unsafe extern "C" {
        /// Initialises the MUSA driver (idempotent; flags must be 0 per musa.h).
        fn muInit(flags: u32) -> muresult;
        /// Fills `device` with the `ordinal`-th device.
        fn muDeviceGet(device: *mut mudevice, ordinal: i32) -> muresult;
        /// Creates a context for `device`.
        fn muCtxCreate_v2(context: *mut mucontext, flags: u32, device: mudevice) -> muresult;
        /// Blocks until every preceding operation on `context` completes.
        fn muCtxSynchronize(context: mucontext) -> muresult;
        /// Frees a device allocation.
        fn muMemFree_v2(mem: mudevptr) -> muresult;
        /// Allocates `bytesize` device bytes into `mem`.
        fn muMemAlloc_v2(mem: *mut mudevptr, bytesize: usize) -> muresult;
        /// Copies `count` bytes from host `src` to device `dst`.
        fn muMemcpyHtoD_v2(dst: mudevptr, src: *const core::ffi::c_void, count: usize) -> muresult;
        /// Copies `count` bytes from device `src` to host `dst`.
        fn muMemcpyDtoH_v2(dst: *mut core::ffi::c_void, src: mudevptr, count: usize) -> muresult;
        /// Copies `count` bytes from device `src` to device `dst`.
        fn muMemcpyDtoD_v2(dst: mudevptr, src: mudevptr, count: usize) -> muresult;
        /// Loads the binary fatbin `data` into `module`.
        fn muModuleLoadData(module: *mut mumodule, data: *const core::ffi::c_void) -> muresult;
        /// Resolves `name` in `module` into `function`.
        fn muModuleGetFunction(
            function: *mut mufunction,
            module: mumodule,
            name: *const core::ffi::c_char,
        ) -> muresult;
        /// Launches `function` on the grid/block geometry with `kernelParams`.
        #[allow(clippy::too_many_arguments)]
        fn muLaunchKernel(
            function: mufunction,
            gridx: u32,
            gridy: u32,
            gridz: u32,
            blockx: u32,
            blocky: u32,
            blockz: u32,
            sharedbytes: u32,
            stream: usize,
            kernelparams: *mut *mut core::ffi::c_void,
            extra: *mut *mut core::ffi::c_void,
        ) -> muresult;
    }

    // ---------------------------------------------------------------------------
    // cache: the device handle, context, loaded modules, and resolved kernels.
    // ---------------------------------------------------------------------------

    /// The backend's device state, guarded by a single lock.
    ///
    /// `cache` owns the opened device and context plus every loaded module and
    /// resolved kernel function. It holds no operand or result, so one cache can
    /// be shared immutably across requests (docs/design.md §21.6). The lock is
    /// held across a dispatch: the default MUSA stream serialises launches
    /// anyway, and the lazy open/module/function state must stay consistent with
    /// the launches that use it.
    pub(crate) struct cache {
        state: Mutex<cachestate>,
    }

    /// The mutable half of [`cache`].
    struct cachestate {
        /// Whether [`open`](cachestate::ensureopen) has run.
        opened: bool,
        /// The opened device and context.
        device: Option<mudevice>,
        context: Option<mucontext>,
        /// Loaded modules by source id (`<stem>.cu`).
        modules: HashMap<&'static str, mumodule>,
        /// Resolved kernels by `(source id, entry)`.
        functions: HashMap<(&'static str, &'static str), mufunction>,
        /// Device buffers returned by earlier dispatches, keyed by exact byte
        /// size, reused by later dispatches instead of a fresh `muMemAlloc_v2`.
        /// The lock is held across a whole dispatch and the default stream
        /// serialises launches, so a pooled buffer is never checked out by two
        /// dispatches at once. The pool is bounded in practice by the fixed set
        /// of operand sizes the operators request, so it does not grow without
        /// limit for a fixed workload.
        pool: HashMap<usize, Vec<mudevptr>>,
        /// Device-resident buffers released by [`retiredevice`], waiting for the
        /// next driver handoff block to return them to the pool or free them.
        /// The queue keeps the release inside the single registered handoff, and
        /// its length is bounded by the live device-resident slots of one
        /// multi-stage operator.
        pendingfree: Vec<(mudevptr, usize)>,
    }

    impl cache {
        /// Creates a cache with nothing opened; the driver is initialised and the
        /// device/context opened lazily by the first dispatch.
        #[must_use]
        pub(crate) fn new() -> Self {
            Self {
                state: Mutex::new(cachestate {
                    opened: false,
                    device: None,
                    context: None,
                    modules: HashMap::new(),
                    functions: HashMap::new(),
                    pool: HashMap::new(),
                    pendingfree: Vec::new(),
                }),
            }
        }

        /// Returns how many kernels have been resolved so far.
        ///
        /// Kernels resolve lazily, once per `(source, entry)` pair, so the count
        /// rises to the number of distinct kernels a workload exercises and then
        /// stops. It is the observable that makes "resolved once" checkable.
        #[must_use]
        pub(crate) fn kernelcount(&self) -> usize {
            self.lock().functions.len()
        }

        /// Acquires the lock, recovering the contents if an earlier holder
        /// panicked (the panic would otherwise poison the shared state).
        fn lock(&self) -> std::sync::MutexGuard<'_, cachestate> {
            self.state.lock().unwrap_or_else(PoisonError::into_inner)
        }
    }

    impl core::fmt::Debug for cache {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            // Deliberately opaque: the handles are raw pointers with no useful
            // debug rendering, and only the kernel count is observable.
            f.debug_struct("cache")
                .field("kernels", &self.lock().functions.len())
                .finish()
        }
    }

    // ---------------------------------------------------------------------------
    // dispatch: the single driver-handoff block.
    // ---------------------------------------------------------------------------

    /// A scalar kernel argument. Every in-repo kernel takes its buffer operands
    /// first and its scalar parameters last, so the launch argument list is
    /// exactly the bound device pointers followed by these scalars.
    #[derive(Clone, Copy, Debug)]
    pub(crate) enum scalar {
        /// An `unsigned` kernel parameter.
        u32(u32),
        /// An `unsigned long long` kernel parameter.
        u64(u64),
        /// A `float` kernel parameter.
        f32(f32),
    }

    /// A bound operand of a dispatch.
    ///
    /// Every `dispatch` is run-to-completion: host operands are uploaded, the
    /// output is downloaded, and this dispatch's temporaries are returned to the
    /// pool inside the single handoff block. The device-resident variants serve
    /// multi-stage kernels (`scan_parallel` chains its passes): the producer hands
    /// a buffer back with [`buf::outdevice`] or [`buf::hostdevice`], later passes
    /// bind it with [`buf::device`], and the owner releases it through
    /// [`retiredevice`], which queues the return into the next handoff block — no
    /// second driver-handoff site exists for freeing.
    #[derive(Clone, Copy, Debug)]
    pub(crate) enum buf<'a> {
        /// Upload a host `f32` slice into a fresh device allocation.
        host(&'a [f32]),
        /// Upload a host `u64` slice into a fresh device allocation.
        hostu64(&'a [u64]),
        /// Upload a host `u64` slice and hand the allocation back as a device
        /// pointer instead of returning it to the pool, so a recorded step keeps
        /// its index buffer alive across the deferred synchronisation.
        hostu64device(&'a [u64]),
        /// Allocate the output buffer and download it after the kernel completes.
        out(usize),
        /// Bind an existing device allocation without uploading or returning it
        /// to the pool. The caller owns its lifetime and releases it through
        /// [`retiredevice`].
        device(mudevptr),
        /// Allocate the output buffer and hand it back as a device pointer
        /// instead of downloading it. The caller owns its lifetime and releases
        /// it through [`retiredevice`].
        outdevice(usize),
        /// Upload a host `f32` slice and hand the allocation back as a device
        /// pointer instead of returning it to the pool, so later passes reuse the
        /// upload. The caller owns its lifetime and releases it through
        /// [`retiredevice`].
        hostdevice(&'a [f32]),
        /// Allocate a fresh device buffer and copy `count` bytes into it from
        /// `src`, handing the allocation back to the caller. Used when an operand
        /// must be preserved across a kernel that updates it in place.
        copy(mudevptr, usize),
        /// Copy `count` bytes from `src` into an existing device buffer `dst`
        /// without allocating. Used to append a key/value row to a persistent
        /// cache buffer.
        copyinto(mudevptr, mudevptr, usize),
        /// Download an existing device allocation into a host vector without
        /// allocating or launching anything. Used by [`flush`], which runs the
        /// deferred synchronisation and the downloads inside the one registered
        /// handoff block.
        download(mudevptr, usize),
    }

    /// The two halves of one driver handoff: downloaded host outputs in buffer
    /// order, and handed-back device buffers as `(pointer, byte size)` pairs for
    /// every `device`-resident slot in buffer order.
    struct dispatchresult {
        host: Vec<Vec<f32>>,
        device: Vec<(mudevptr, usize)>,
    }

    /// Runs one kernel through the single driver-handoff block.
    ///
    /// The block is lazy by design: the first call initialises the driver, opens
    /// the device and context, loads the fatbin for `source_id` from the
    /// build-time registry, and resolves `entry`; later calls reuse all of it.
    /// Host operands are uploaded, the kernel is launched on the default stream,
    /// the context is synchronised, outputs are downloaded, and this dispatch's
    /// temporary allocations are freed — all inside the one `unsafe` block, so the
    /// registered handoff count stays exactly one (docs/musa-backend-plan.md
    /// §5.4).
    ///
    /// # Errors
    ///
    /// Returns [`gpuerror::device`] when the driver cannot be initialised or no
    /// device exists, [`gpuerror::module`] when a fatbin cannot be loaded or an
    /// entry resolved, [`gpuerror::transfer`] when a copy fails,
    /// [`gpuerror::launch`] when the launch or synchronisation fails, and
    /// [`gpuerror::sizeoverflow`] when a byte count overflows.
    /// Runs one kernel through the single driver-handoff block, downloading
    /// every [`buf::out`] slot. Device-resident variants are accepted but their
    /// handed-back pointers are discarded: callers of a mixed run use
    /// [`dispatchdevice`] instead.
    pub(crate) fn dispatch(
        cache: &cache,
        source_id: &'static str,
        entry: &'static str,
        buffers: &[buf<'_>],
        scalars: &[scalar],
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        sharedbytes: u32,
    ) -> Result<Vec<Vec<f32>>, gpuerror> {
        Ok(drive(
            cache,
            Some((source_id, entry)),
            buffers,
            scalars,
            grid,
            block,
            sharedbytes,
            true,
        )?
        .host)
    }

    /// Runs one kernel through the single driver-handoff block, handing back
    /// both halves: downloaded [`buf::out`] slots and every device-resident slot
    /// as a `(pointer, byte size)` pair, each in buffer order. The caller owns
    /// the device-resident buffers and releases them through [`retiredevice`].
    pub(crate) fn dispatchmixed(
        cache: &cache,
        source_id: &'static str,
        entry: &'static str,
        buffers: &[buf<'_>],
        scalars: &[scalar],
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        sharedbytes: u32,
    ) -> Result<dispatchresult, gpuerror> {
        drive(
            cache,
            Some((source_id, entry)),
            buffers,
            scalars,
            grid,
            block,
            sharedbytes,
            true,
        )
    }

    /// Queues a device-resident buffer for release at the next driver handoff.
    ///
    /// The release itself (pool return or `muMemFree_v2`) happens inside the one
    /// registered handoff block, so this function performs no driver call and adds
    /// no `unsafe` site. A zero pointer or zero size is a no-op.
    pub(crate) fn retiredevice(cache: &cache, mem: mudevptr, bytes: usize) {
        if mem != 0 && bytes != 0 {
            cache.lock().pendingfree.push((mem, bytes));
        }
    }

    /// Synchronises once and downloads the given device buffers, all inside the
    /// one registered handoff block.
    ///
    /// This is the submission path's `apply`: the per-step records ran with the
    /// synchronisation deferred, and this call drains the whole chain with a
    /// single `muCtxSynchronize` before copying the requested outputs out.
    pub(crate) fn flush(
        cache: &cache,
        downloads: &[(mudevptr, usize)],
    ) -> Result<Vec<Vec<f32>>, gpuerror> {
        let slots: Vec<buf<'_>> = downloads
            .iter()
            .map(|&(ptr, bytes)| buf::download(ptr, bytes))
            .collect();
        Ok(drive(cache, None, &slots, &[], (1, 1, 1), (1, 1, 1), 0, true)?.host)
    }

    /// A device-resident record chain: every step launches immediately on the
    /// default stream with the synchronisation deferred, and [`session::apply`]
    /// drains the chain with one synchronise plus the requested downloads.
    pub(crate) struct session {
        cache: Arc<cache>,
        /// Every device buffer the session owns, in creation order: uploaded
        /// inputs and recorded outputs alike. `apply` frees them all through
        /// [`retiredevice`], whose queue drains inside the next handoff block.
        slots: Vec<(mudevptr, usize)>,
        /// Parallel to `slots`: whether a slot survives
        /// [`applyretained`](Session::applyretained). Persistent slots hold the
        /// device KV-cache buffers a persistent session reuses across steps.
        persistent: Vec<bool>,
    }

    impl Drop for session {
        fn drop(&mut self) {
            for (ptr, bytes) in self.slots.drain(..) {
                retiredevice(&self.cache, ptr, bytes);
            }
        }
    }

    impl session {
        /// Creates an empty record chain over `cache`.
        #[must_use]
        pub(crate) fn new(cache: Arc<cache>) -> Self {
            Self {
                cache,
                slots: Vec::new(),
                persistent: Vec::new(),
            }
        }

        /// Returns the device pointer of `slot`.
        pub(crate) fn slotptr(&self, slot: usize) -> mudevptr {
            self.slots[slot].0
        }

        /// Uploads a host slice once and returns its slot.
        pub(crate) fn upload(&mut self, data: &[f32]) -> Result<usize, gpuerror> {
            let mut result = drive(
                &self.cache,
                None,
                &[buf::hostdevice(data)],
                &[],
                (1, 1, 1),
                (1, 1, 1),
                0,
                false,
            )?;
            let (ptr, bytes) = result.device.remove(0);
            self.slots.push((ptr, bytes));
            self.persistent.push(false);
            Ok(self.slots.len() - 1)
        }

        /// Uploads a host slice into a slot that survives `applyretained`.
        pub(crate) fn allocpersistent(&mut self, data: &[f32]) -> Result<usize, gpuerror> {
            let slot = self.upload(data)?;
            self.persistent[slot] = true;
            Ok(slot)
        }

        /// Allocates a zeroed persistent device buffer of `count` floats.
        pub(crate) fn alloczeros(&mut self, count: usize) -> Result<usize, gpuerror> {
            let zeros = vec![0.0_f32; count];
            self.allocpersistent(&zeros)
        }

        /// Reports whether `slot` survives `applyretained`.
        pub(crate) fn ispersistent(&self, slot: usize) -> bool {
            self.persistent.get(slot).copied().unwrap_or(false)
        }

        /// Uploads a host slice into an allocation the caller owns and returns its
        /// pointer and byte size. Used for inputs pinned for a persistent session;
        /// they are not session slots, so `apply` never releases them.
        pub(crate) fn uploadpersistent(
            cache: &cache,
            data: &[f32],
        ) -> Result<(mudevptr, usize), gpuerror> {
            let mut result = drive(
                cache,
                None,
                &[buf::hostdevice(data)],
                &[],
                (1, 1, 1),
                (1, 1, 1),
                0,
                false,
            )?;
            Ok(result.device.remove(0))
        }

        /// Copies `bytes` from device `src` into a fresh allocation and returns its
        /// slot. Used when a kernel updates an operand in place and the original
        /// value must remain downloadable.
        pub(crate) fn copy(&mut self, src: mudevptr, bytes: usize) -> Result<usize, gpuerror> {
            let mut result = drive(
                &self.cache,
                None,
                &[buf::copy(src, bytes)],
                &[],
                (1, 1, 1),
                (1, 1, 1),
                0,
                false,
            )?;
            let (ptr, size) = result.device.remove(0);
            self.slots.push((ptr, size));
            self.persistent.push(false);
            Ok(self.slots.len() - 1)
        }

        /// Copies `bytes` from device `src` into the existing device buffer `dst`.
        pub(crate) fn copyinto(
            &mut self,
            dst: mudevptr,
            src: mudevptr,
            bytes: usize,
        ) -> Result<(), gpuerror> {
            drive(
                &self.cache,
                None,
                &[buf::copyinto(dst, src, bytes)],
                &[],
                (1, 1, 1),
                (1, 1, 1),
                0,
                false,
            )?;
            Ok(())
        }

        /// Uploads a host `u64` slice once and returns its slot.
        pub(crate) fn uploadu64(&mut self, data: &[u64]) -> Result<usize, gpuerror> {
            let mut result = drive(
                &self.cache,
                None,
                &[buf::hostu64device(data)],
                &[],
                (1, 1, 1),
                (1, 1, 1),
                0,
                false,
            )?;
            let (ptr, bytes) = result.device.remove(0);
            self.slots.push((ptr, bytes));
            self.persistent.push(false);
            Ok(self.slots.len() - 1)
        }

        /// Records one kernel step without synchronising and returns the output
        /// slot. `hostdevice` inputs in `buffers` are uploaded once and kept;
        /// `device` inputs bind existing slots.
        pub(crate) fn record(
            &mut self,
            source_id: &'static str,
            entry: &'static str,
            buffers: &[buf<'_>],
            scalars: &[scalar],
            grid: (u32, u32, u32),
            block: (u32, u32, u32),
            sharedbytes: u32,
        ) -> Result<usize, gpuerror> {
            let mut result = drive(
                &self.cache,
                Some((source_id, entry)),
                buffers,
                scalars,
                grid,
                block,
                sharedbytes,
                false,
            )?;
            for (ptr, bytes) in result.device {
                self.slots.push((ptr, bytes));
                self.persistent.push(false);
            }
            Ok(self.slots.len() - 1)
        }

        /// Records one kernel step and returns **every** new device slot in buffer
        /// order.
        ///
        /// Used by operators with two device-resident outputs (`scan`'s output and
        /// final state); [`record`](Self::record) remains the single-output form.
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn recordmulti(
            &mut self,
            source_id: &'static str,
            entry: &'static str,
            buffers: &[buf<'_>],
            scalars: &[scalar],
            grid: (u32, u32, u32),
            block: (u32, u32, u32),
            sharedbytes: u32,
        ) -> Result<Vec<usize>, gpuerror> {
            let result = drive(
                &self.cache,
                Some((source_id, entry)),
                buffers,
                scalars,
                grid,
                block,
                sharedbytes,
                false,
            )?;
            let mut slots = Vec::with_capacity(result.device.len());
            for (ptr, bytes) in result.device {
                self.slots.push((ptr, bytes));
                self.persistent.push(false);
                slots.push(self.slots.len() - 1);
            }
            Ok(slots)
        }

        /// Drains one slot to host data mid-chain, for a caller that needs a host
        /// copy of a recorded result. The slot is released immediately and marked
        /// empty.
        pub(crate) fn take(&mut self, slot: usize) -> Result<Vec<f32>, gpuerror> {
            let (ptr, bytes) = self.slots[slot];
            let mut out = flush(&self.cache, &[(ptr, bytes)])?;
            retiredevice(&self.cache, ptr, bytes);
            self.slots[slot] = (0, 0);
            self.persistent[slot] = false;
            Ok(out.remove(0))
        }

        /// Synchronises once, downloads the requested slots in order, and releases
        /// every owned buffer through [`retiredevice`].
        pub(crate) fn apply(&mut self, downloads: &[usize]) -> Result<Vec<Vec<f32>>, gpuerror> {
            let wanted: Vec<(mudevptr, usize)> =
                downloads.iter().map(|&index| self.slots[index]).collect();
            let out = flush(&self.cache, &wanted)?;
            for (ptr, bytes) in self.slots.drain(..) {
                retiredevice(&self.cache, ptr, bytes);
            }
            self.persistent.clear();
            Ok(out)
        }

        /// Downloads the requested slots and frees every **non-persistent** buffer,
        /// leaving persistent slots (and their indices) in place for the next step.
        pub(crate) fn applyretained(
            &mut self,
            downloads: &[usize],
        ) -> Result<Vec<Vec<f32>>, gpuerror> {
            let wanted: Vec<(mudevptr, usize)> =
                downloads.iter().map(|&index| self.slots[index]).collect();
            let out = flush(&self.cache, &wanted)?;
            for index in 0..self.slots.len() {
                if self.persistent[index] {
                    continue;
                }
                let (ptr, bytes) = self.slots[index];
                if ptr != 0 && bytes != 0 {
                    retiredevice(&self.cache, ptr, bytes);
                }
                self.slots[index] = (0, 0);
            }
            Ok(out)
        }
    }

    /// Records one elementwise `add` into a device-resident chain.
    pub(crate) fn add_record(
        session: &mut session,
        lhs: buf<'_>,
        rhs: buf<'_>,
        count: usize,
    ) -> Result<usize, gpuerror> {
        let grid = ((count as u32).div_ceil(source::ELEMENTWISETILE), 1, 1);
        let block = (source::ELEMENTWISETILE, 1, 1);
        session.record(
            "elementwise",
            source::ADDENTRY,
            &[lhs, rhs, buf::outdevice(count)],
            &[scalar::u32(count as u32)],
            grid,
            block,
            0,
        )
    }

    /// Records one elementwise `mul` into a device-resident chain.
    pub(crate) fn mul_record(
        session: &mut session,
        lhs: buf<'_>,
        rhs: buf<'_>,
        count: usize,
    ) -> Result<usize, gpuerror> {
        let grid = ((count as u32).div_ceil(source::ELEMENTWISETILE), 1, 1);
        let block = (source::ELEMENTWISETILE, 1, 1);
        session.record(
            "elementwise",
            source::MULENTRY,
            &[lhs, rhs, buf::outdevice(count)],
            &[scalar::u32(count as u32)],
            grid,
            block,
            0,
        )
    }

    /// Records one `silu` into a device-resident chain.
    pub(crate) fn silu_record(
        session: &mut session,
        input: buf<'_>,
        count: usize,
    ) -> Result<usize, gpuerror> {
        let grid = ((count as u32).div_ceil(source::ELEMENTWISETILE), 1, 1);
        let block = (source::ELEMENTWISETILE, 1, 1);
        session.record(
            "elementwise",
            source::SILUENTRY,
            &[input, buf::outdevice(count)],
            &[scalar::u32(count as u32)],
            grid,
            block,
            0,
        )
    }

    /// Records one RMS normalization into a device-resident chain.
    pub(crate) fn rmsnorm_record(
        session: &mut session,
        input: buf<'_>,
        weight: buf<'_>,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<usize, gpuerror> {
        let count = rows.checked_mul(width).ok_or(gpuerror::sizeoverflow)?;
        let grid = (rows as u32, 1, 1);
        let block = (source::ROWTILE, 1, 1);
        session.record(
            "rmsnorm",
            source::RMSNORMENTRY,
            &[input, weight, buf::outdevice(count)],
            &[scalar::u32(width as u32), scalar::f32(eps)],
            grid,
            block,
            source::ROWTILE as u32 * 4,
        )
    }

    /// Records one softmax over the last axis into a device-resident chain.
    pub(crate) fn softmax_record(
        session: &mut session,
        input: buf<'_>,
        rows: usize,
        width: usize,
    ) -> Result<usize, gpuerror> {
        let count = rows.checked_mul(width).ok_or(gpuerror::sizeoverflow)?;
        let grid = (rows as u32, 1, 1);
        let block = (source::ELEMENTWISETILE, 1, 1);
        session.record(
            "softmax",
            source::SOFTMAXENTRY,
            &[input, buf::outdevice(count)],
            &[scalar::u32(width as u32)],
            grid,
            block,
            source::ELEMENTWISETILE as u32 * 4,
        )
    }

    /// Records one embedding gather into a device-resident chain.
    pub(crate) fn embedding_record(
        session: &mut session,
        table: buf<'_>,
        indices: buf<'_>,
        rows: usize,
        width: usize,
    ) -> Result<usize, gpuerror> {
        let count = rows.checked_mul(width).ok_or(gpuerror::sizeoverflow)?;
        session.record(
            "embedding",
            source::EMBEDDINGENTRY,
            &[table, indices, buf::outdevice(count)],
            &[scalar::u64(rows as u64), scalar::u64(width as u64)],
            (rows as u32, 1, 1),
            (source::ELEMENTWISETILE, 1, 1),
            0,
        )
    }

    /// Records one rotary position embedding into a device-resident chain.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rope_record(
        session: &mut session,
        input: buf<'_>,
        count: usize,
        rows: usize,
        seqaxis: usize,
        heads: usize,
        headdim: usize,
        base: f32,
    ) -> Result<usize, gpuerror> {
        let grid = ((seqaxis as u32).div_ceil(source::ELEMENTWISETILE), 1, 1);
        session.record(
            "rope",
            source::ROPEENTRY,
            &[input, buf::outdevice(count)],
            &[
                scalar::u32(rows as u32),
                scalar::u32(seqaxis as u32),
                scalar::u32(heads as u32),
                scalar::u32(headdim as u32),
                scalar::f32(base),
            ],
            grid,
            (source::ELEMENTWISETILE, 1, 1),
            0,
        )
    }

    /// Records one absolute-position rotary embedding into a device-resident
    /// chain.
    pub(crate) fn rope_at_record(
        session: &mut session,
        input: buf<'_>,
        count: usize,
        position: usize,
        heads: usize,
        headdim: usize,
        base: f32,
    ) -> Result<usize, gpuerror> {
        session.record(
            "at",
            source::ROPEATENTRY,
            &[input, buf::outdevice(count)],
            &[
                scalar::u32(heads as u32),
                scalar::u32(headdim as u32),
                scalar::u32(position as u32),
                scalar::f32(base),
            ],
            (1, 1, 1),
            (source::ELEMENTWISETILE, 1, 1),
            0,
        )
    }

    /// Records one multi-head attention into a device-resident chain.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_record(
        session: &mut session,
        query: buf<'_>,
        key: buf<'_>,
        value: buf<'_>,
        count: usize,
        qseq: usize,
        kvseq: usize,
        heads: usize,
        kvheads: usize,
        headdim: usize,
        causal: bool,
    ) -> Result<usize, gpuerror> {
        let gridx = qseq.checked_mul(heads).ok_or(gpuerror::sizeoverflow)?;
        session.record(
            "attention",
            source::ATTENTIONENTRY,
            &[query, key, value, buf::outdevice(count)],
            &[
                scalar::u32(qseq as u32),
                scalar::u32(kvseq as u32),
                scalar::u32(heads as u32),
                scalar::u32(kvheads as u32),
                scalar::u32(headdim as u32),
                scalar::u32(causal as u32),
            ],
            (gridx as u32, 1, 1),
            (source::ATTENTIONLANES, 1, 1),
            (2 * source::ATTENTIONLANES + headdim as u32) * 4,
        )
    }

    /// Records one incremental-decode attention into a device-resident chain.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn attention_cached_record(
        session: &mut session,
        query: buf<'_>,
        keycached: buf<'_>,
        valuecached: buf<'_>,
        keynew: buf<'_>,
        valuenew: buf<'_>,
        count: usize,
        kvseq: usize,
        heads: usize,
        kvheads: usize,
        headdim: usize,
        causal: bool,
    ) -> Result<usize, gpuerror> {
        session.record(
            "cached",
            source::ATTENTIONCACHEDENTRY,
            &[
                query,
                keycached,
                valuecached,
                keynew,
                valuenew,
                buf::outdevice(count),
            ],
            &[
                scalar::u32(kvseq as u32),
                scalar::u32(heads as u32),
                scalar::u32(kvheads as u32),
                scalar::u32(headdim as u32),
                scalar::u32(causal as u32),
            ],
            (heads as u32, 1, 1),
            (source::ATTENTIONLANES, 1, 1),
            (2 * source::ATTENTIONLANES + headdim as u32) * 4,
        )
    }

    /// The single registered driver-handoff block behind [`dispatch`] and
    /// [`dispatchdevice`].
    ///
    /// The block is lazy by design: the first call initialises the driver, opens
    /// the device and context, loads the fatbin for `source_id` from the
    /// build-time registry, and resolves `entry`; later calls reuse all of it.
    /// Host operands are uploaded, the kernel is launched on the default stream,
    /// the context is synchronised, outputs are downloaded, and this dispatch's
    /// temporary allocations are returned to the pool — all inside the one
    /// `unsafe` block, so the registered handoff count stays exactly one
    /// (docs/musa-backend-plan.md §5.4).
    #[allow(clippy::too_many_arguments)]
    fn drive(
        cache: &cache,
        kernel: Option<(&'static str, &'static str)>,
        buffers: &[buf<'_>],
        scalars: &[scalar],
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        sharedbytes: u32,
        sync: bool,
    ) -> Result<dispatchresult, gpuerror> {
        // The state lock is held across the whole handoff: the lazy open and
        // function resolution mutate the shared caches, and the default stream
        // serialises launches anyway.
        let mut state = cache.lock();

        // SAFETY: All driver calls in this block share one precondition: every
        // handle and pointer is either (a) a handle produced by an earlier call in
        // this block or held in `state` from a previous dispatch, or (b) a live
        // value owned by this call (host slices, the fatbin bytes, the C string
        // entry name, the argument lists). Every allocation size is derived from
        // the host slice length or the output element count with checked
        // arithmetic, and every copy length equals the allocation it touches. The
        // launch geometry comes from the caller's validated shapes and the kernels
        // bounds-test every index they derive. `kernelparams` points at the local
        // argument arrays, which outlive the launch. Every call's result is
        // checked against `MUSA_SUCCESS` and converted to `gpuerror`; no result
        // is ignored. The synchronisation orders the downloads after the kernel's
        // writes, and a buffer is returned to the pool only after its last use in
        // this handoff; the lock held across the whole block means no pooled
        // buffer is ever checked out by two dispatches at once, and the
        // device-resident variants bypass the pool return because the caller owns
        // them through [`retiredevice`].
        unsafe {
            // --- lazy open -----------------------------------------------------
            if !state.opened {
                if muInit(0) != MUSA_SUCCESS {
                    return Err(gpuerror::device {
                        message: "muInit failed".into(),
                    });
                }
                let mut device: mudevice = 0;
                if muDeviceGet(&mut device, 0) != MUSA_SUCCESS {
                    return Err(gpuerror::device {
                        message: "muDeviceGet(0) failed: no MUSA device".into(),
                    });
                }
                let mut context: mucontext = 0;
                if muCtxCreate_v2(&mut context, 0, device) != MUSA_SUCCESS {
                    return Err(gpuerror::device {
                        message: "muCtxCreate failed".into(),
                    });
                }
                state.opened = true;
                state.device = Some(device);
                state.context = Some(context);
            }
            let context = state.context.ok_or(gpuerror::device {
                message: "context not open".into(),
            })?;

            // --- deferred releases --------------------------------------------
            // Buffers released by `retiredevice` re-enter the pool (or are freed)
            // here, inside the one registered handoff block, before any new
            // allocation so a just-released slot can serve this very dispatch.
            let pending: Vec<(mudevptr, usize)> = state.pendingfree.drain(..).collect();
            for (mem, bytes) in pending {
                let bucket = state.pool.entry(bytes).or_default();
                if bucket.len() < POOLCAP {
                    bucket.push(mem);
                } else if muMemFree_v2(mem) != MUSA_SUCCESS {
                    return Err(gpuerror::transfer {
                        message: "muMemFree failed for released device buffer".into(),
                    });
                }
            }

            // --- lazy module load and kernel resolution ------------------------
            if let Some((source_id, entry)) = kernel {
                if !state.functions.contains_key(&(source_id, entry)) {
                    let fatbin = KERNELS
                    .iter()
                    .find(|(id, _)| *id == source_id)
                    .map(|(_, bytes)| *bytes)
                    .ok_or_else(|| gpuerror::module {
                        kernel: entry.to_string(),
                        message: format!(
                            "no fatbin for source `{source_id}` (was mcc available at build time?)"
                        ),
                    })?;
                    let module = if let Some(module) = state.modules.get(source_id) {
                        *module
                    } else {
                        let mut module: mumodule = 0;
                        if muModuleLoadData(&mut module, fatbin.as_ptr().cast()) != MUSA_SUCCESS {
                            return Err(gpuerror::module {
                                kernel: entry.to_string(),
                                message: format!("muModuleLoadData failed for `{source_id}`"),
                            });
                        }
                        state.modules.insert(source_id, module);
                        module
                    };
                    let name = CString::new(entry).map_err(|_| gpuerror::module {
                        kernel: entry.to_string(),
                        message: "entry name contains a NUL byte".into(),
                    })?;
                    let mut function: mufunction = 0;
                    if muModuleGetFunction(&mut function, module, name.as_ptr()) != MUSA_SUCCESS {
                        return Err(gpuerror::module {
                            kernel: entry.to_string(),
                            message: format!("muModuleGetFunction failed for `{entry}`"),
                        });
                    }
                    state.functions.insert((source_id, entry), function);
                }
            }
            let function = match kernel {
                Some((source_id, entry)) => {
                    Some(*state.functions.get(&(source_id, entry)).ok_or_else(|| {
                        gpuerror::module {
                            kernel: entry.to_string(),
                            message: "kernel not resolved".into(),
                        }
                    })?)
                }
                None => None,
            };

            // --- allocate and upload ------------------------------------------
            // Byte sizes first (all checked), then allocate, then copy, so a
            // failed allocation leaves nothing half-uploaded. Device-resident
            // slots (`device`/`outdevice`/`hostdevice`) are marked `keep`: they
            // skip the pool return and their `(pointer, byte size)` pair is
            // handed back to the caller in buffer order.
            let mut sizes: Vec<usize> = Vec::with_capacity(buffers.len());
            let mut owned: Vec<usize> = Vec::with_capacity(buffers.len());
            let mut keep: Vec<bool> = Vec::with_capacity(buffers.len());
            let mut allocate: Vec<bool> = Vec::with_capacity(buffers.len());
            for slot in buffers {
                match *slot {
                    buf::host(data) | buf::hostdevice(data) => {
                        sizes.push(
                            data.len()
                                .checked_mul(core::mem::size_of::<f32>())
                                .ok_or(gpuerror::sizeoverflow)?,
                        );
                        keep.push(matches!(*slot, buf::hostdevice(_)));
                        allocate.push(true);
                        owned.push(0);
                    }
                    buf::hostu64(data) | buf::hostu64device(data) => {
                        sizes.push(
                            data.len()
                                .checked_mul(core::mem::size_of::<u64>())
                                .ok_or(gpuerror::sizeoverflow)?,
                        );
                        keep.push(matches!(*slot, buf::hostu64device(_)));
                        allocate.push(true);
                        owned.push(0);
                    }
                    buf::out(count) | buf::outdevice(count) => {
                        sizes.push(
                            count
                                .checked_mul(core::mem::size_of::<f32>())
                                .ok_or(gpuerror::sizeoverflow)?,
                        );
                        keep.push(matches!(*slot, buf::outdevice(_)));
                        allocate.push(true);
                        owned.push(0);
                    }
                    buf::device(mem) => {
                        sizes.push(0);
                        keep.push(true);
                        allocate.push(false);
                        owned.push(mem);
                    }
                    buf::download(mem, bytes) => {
                        sizes.push(bytes);
                        keep.push(true);
                        allocate.push(false);
                        owned.push(mem);
                    }
                    buf::copy(_, bytes) => {
                        sizes.push(bytes);
                        keep.push(true);
                        allocate.push(true);
                        owned.push(0);
                    }
                    buf::copyinto(_, _, bytes) => {
                        sizes.push(bytes);
                        keep.push(true);
                        allocate.push(false);
                        owned.push(0);
                    }
                }
            }
            for (index, &size) in sizes.iter().enumerate() {
                if size == 0 || !allocate[index] {
                    continue;
                }
                let mem = if let Some(pooled) = state.pool.get_mut(&size).and_then(Vec::pop) {
                    pooled
                } else {
                    let mut mem: mudevptr = 0;
                    if muMemAlloc_v2(&mut mem, size) != MUSA_SUCCESS {
                        return Err(gpuerror::transfer {
                            message: format!("muMemAlloc failed for {size} bytes"),
                        });
                    }
                    mem
                };
                owned[index] = mem;
            }
            for (index, slot) in buffers.iter().enumerate() {
                if sizes[index] == 0 {
                    continue;
                }
                match *slot {
                    buf::host(data) | buf::hostdevice(data) => {
                        if muMemcpyHtoD_v2(owned[index], data.as_ptr().cast(), sizes[index])
                            != MUSA_SUCCESS
                        {
                            return Err(gpuerror::transfer {
                                message: "muMemcpyHtoD failed for f32 operand".into(),
                            });
                        }
                    }
                    buf::hostu64(data) | buf::hostu64device(data) => {
                        if muMemcpyHtoD_v2(owned[index], data.as_ptr().cast(), sizes[index])
                            != MUSA_SUCCESS
                        {
                            return Err(gpuerror::transfer {
                                message: "muMemcpyHtoD failed for u64 operand".into(),
                            });
                        }
                    }
                    buf::copy(src, _) => {
                        if muMemcpyDtoD_v2(owned[index], src, sizes[index]) != MUSA_SUCCESS {
                            return Err(gpuerror::transfer {
                                message: "muMemcpyDtoD failed for copy".into(),
                            });
                        }
                    }
                    buf::copyinto(dst, src, _) => {
                        if muMemcpyDtoD_v2(dst, src, sizes[index]) != MUSA_SUCCESS {
                            return Err(gpuerror::transfer {
                                message: "muMemcpyDtoD failed for copyinto".into(),
                            });
                        }
                    }
                    buf::out(_) | buf::outdevice(_) | buf::device(_) | buf::download(_, _) => {}
                }
            }
            // Buffer pointers: every bound operand is backed by an owned
            // allocation or a caller-owned device pointer. `ptrs` outlives the
            // launch.
            let mut ptrs: Vec<usize> = owned.clone();

            // --- launch on the default stream, then synchronise ----------------
            let mut rawargs: Vec<*mut core::ffi::c_void> =
                Vec::with_capacity(ptrs.len() + scalars.len());
            rawargs.extend(ptrs.iter_mut().map(|ptr| ptr as *mut usize as *mut _));
            let mut rawscalars: Vec<scalar> = scalars.to_vec();
            for value in rawscalars.iter_mut() {
                match value {
                    scalar::u32(number) => rawargs.push(number as *mut u32 as *mut _),
                    scalar::u64(number) => rawargs.push(number as *mut u64 as *mut _),
                    scalar::f32(number) => rawargs.push(number as *mut f32 as *mut _),
                }
            }
            if let Some(function) = function {
                if muLaunchKernel(
                    function,
                    grid.0,
                    grid.1,
                    grid.2,
                    block.0,
                    block.1,
                    block.2,
                    sharedbytes,
                    MUNULLSTREAM,
                    rawargs.as_mut_ptr(),
                    core::ptr::null_mut(),
                ) != MUSA_SUCCESS
                {
                    return Err(gpuerror::launch {
                        message: "muLaunchKernel failed".into(),
                    });
                }
            }
            if sync && muCtxSynchronize(context) != MUSA_SUCCESS {
                return Err(gpuerror::launch {
                    message: "muCtxSynchronize failed".into(),
                });
            }

            // --- download outputs ---------------------------------------------
            let mut host: Vec<Vec<f32>> = Vec::with_capacity(buffers.len());
            let mut device: Vec<(mudevptr, usize)> = Vec::with_capacity(buffers.len());
            for (index, slot) in buffers.iter().enumerate() {
                match *slot {
                    buf::out(count) => {
                        let mut out = Vec::new();
                        out.try_reserve_exact(count)
                            .map_err(|_| gpuerror::sizeoverflow)?;
                        out.resize(count, 0.0_f32);
                        if count != 0
                            && muMemcpyDtoH_v2(out.as_mut_ptr().cast(), owned[index], sizes[index])
                                != MUSA_SUCCESS
                        {
                            return Err(gpuerror::transfer {
                                message: "muMemcpyDtoH failed for output".into(),
                            });
                        }
                        host.push(out);
                    }
                    buf::outdevice(_)
                    | buf::hostdevice(_)
                    | buf::hostu64device(_)
                    | buf::copy(_, _) => {
                        device.push((owned[index], sizes[index]));
                    }
                    buf::device(_) | buf::copyinto(_, _, _) => {}
                    buf::download(ptr, bytes) => {
                        let mut out = Vec::new();
                        out.try_reserve_exact(bytes / core::mem::size_of::<f32>())
                            .map_err(|_| gpuerror::sizeoverflow)?;
                        out.resize(bytes / core::mem::size_of::<f32>(), 0.0_f32);
                        if bytes != 0
                            && muMemcpyDtoH_v2(out.as_mut_ptr().cast(), ptr, bytes) != MUSA_SUCCESS
                        {
                            return Err(gpuerror::transfer {
                                message: "muMemcpyDtoH failed for flush download".into(),
                            });
                        }
                        host.push(out);
                    }
                    buf::host(_) | buf::hostu64(_) => {}
                }
            }

            // --- return this dispatch's buffers to the pool --------------------
            // Every non-zero allocation this dispatch checked out goes back by its
            // exact byte size, so the next dispatch of the same size reuses it
            // instead of paying the driver allocation cost. The pool never hands a
            // buffer to two dispatches at once: the lock is held across this whole
            // block and the default stream serialises launches. A bucket that has
            // reached `POOLCAP` frees its excess instead, bounding device memory.
            for (index, &mem) in owned.iter().enumerate() {
                if sizes[index] == 0 || keep[index] {
                    continue;
                }
                let bucket = state.pool.entry(sizes[index]).or_default();
                if bucket.len() < POOLCAP {
                    bucket.push(mem);
                } else if muMemFree_v2(mem) != MUSA_SUCCESS {
                    return Err(gpuerror::transfer {
                        message: "muMemFree failed for excess pooled buffer".into(),
                    });
                }
            }
            Ok(dispatchresult { host, device })
        }
    }

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
            "matmul",
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
        let mut inv = dispatchmixed(
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
            (source::ROWTILE, 1, 1),
            source::ROWTILE as u32 * 4,
        )?;
        let inputptr = inv.device[0].0;
        let invptr = inv.device[1].0;
        let grid = (
            (n as u32).div_ceil(source::MATMULWMMABN),
            (m as u32).div_ceil(source::MATMULWMMABM),
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
            (source::MATMULWMMATHREADS, 1, 1),
            0,
        )?;
        retiredevice(cache, inputptr, inv.device[0].1);
        retiredevice(cache, invptr, inv.device[1].1);
        Ok(results.remove(0))
    }

    /// Records the per-row reciprocal-RMS step into a device-resident chain.
    pub(crate) fn rmsinv_record(
        session: &mut session,
        input: buf<'_>,
        rows: usize,
        width: usize,
        eps: f32,
    ) -> Result<usize, gpuerror> {
        session.record(
            source::MATMULWMMASOURCEID,
            source::RMSINVENTRY,
            &[input, buf::outdevice(rows)],
            &[
                scalar::u32(rows as u32),
                scalar::u32(width as u32),
                scalar::f32(eps),
            ],
            (rows as u32, 1, 1),
            (source::ROWTILE, 1, 1),
            source::ROWTILE as u32 * 4,
        )
    }

    /// Records the fused RMSNorm+matmul step into a device-resident chain.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn rmsnormmatmul_record(
        session: &mut session,
        input: buf<'_>,
        weight: buf<'_>,
        rhs: buf<'_>,
        invslot: usize,
        total: usize,
        m: usize,
        k: usize,
        n: usize,
        batch: usize,
    ) -> Result<usize, gpuerror> {
        let invptr = session.slotptr(invslot);
        let grid = (
            (n as u32).div_ceil(source::MATMULWMMABN),
            (m as u32).div_ceil(source::MATMULWMMABM),
            batch as u32,
        );
        session.record(
            source::MATMULWMMASOURCEID,
            source::MATMULRMSENTRY,
            &[
                input,
                weight,
                buf::device(invptr),
                rhs,
                buf::outdevice(total),
            ],
            &[
                scalar::u32(m as u32),
                scalar::u32(k as u32),
                scalar::u32(n as u32),
                scalar::u32(batch as u32),
            ],
            grid,
            (source::MATMULWMMATHREADS, 1, 1),
            0,
        )
    }

    /// Records one device-resident Tensor-Core matmul into a submission session.
    ///
    /// The operands bind existing device slots (uploaded once per submission) and
    /// the result stays in a device slot, so a chain of recorded matmuls performs
    /// one upload, no per-step download, and one synchronisation at `apply`.
    pub(crate) fn matmul_wmma_record(
        session: &mut session,
        lhs: buf<'_>,
        rhs: buf<'_>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<usize, gpuerror> {
        let total = m.checked_mul(n).ok_or(gpuerror::sizeoverflow)?;
        let grid = (
            (n as u32).div_ceil(source::MATMULWMMABN),
            (m as u32).div_ceil(source::MATMULWMMABM),
            1,
        );
        let block = (source::MATMULWMMATHREADS, 1, 1);
        session.record(
            source::MATMULWMMASOURCEID,
            source::MATMULWMMAENTRY,
            &[lhs, rhs, buf::outdevice(total)],
            &[
                scalar::u32(m as u32),
                scalar::u32(k as u32),
                scalar::u32(n as u32),
                scalar::u32(1),
            ],
            grid,
            block,
            0,
        )
    }

    /// Records one device-resident reference FMA matmul into a submission session.
    pub(crate) fn matmul_fma_record(
        session: &mut session,
        lhs: buf<'_>,
        rhs: buf<'_>,
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<usize, gpuerror> {
        let total = m.checked_mul(n).ok_or(gpuerror::sizeoverflow)?;
        let grid = (
            (n as u32).div_ceil(source::THREADEDGE),
            (m as u32).div_ceil(source::THREADEDGE),
            1,
        );
        let block = (source::THREADEDGE, source::THREADEDGE, 1);
        session.record(
            "matmul",
            source::MATMULENTRY,
            &[lhs, rhs, buf::outdevice(total)],
            &[
                scalar::u32(m as u32),
                scalar::u32(k as u32),
                scalar::u32(n as u32),
                scalar::u32(1),
            ],
            grid,
            block,
            0,
        )
    }

    /// Elementwise `lhs + rhs`.
    pub(crate) fn add(cache: &cache, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, gpuerror> {
        let count = lhs.len();
        let grid = ((count as u32).div_ceil(source::ELEMENTWISETILE), 1, 1);
        let block = (source::ELEMENTWISETILE, 1, 1);
        let mut results = dispatch(
            cache,
            "elementwise",
            source::ADDENTRY,
            &[buf::host(lhs), buf::host(rhs), buf::out(count)],
            &[scalar::u32(count as u32)],
            grid,
            block,
            0,
        )?;
        Ok(results.remove(0))
    }

    /// Elementwise `lhs * rhs`.
    pub(crate) fn mul(cache: &cache, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, gpuerror> {
        let count = lhs.len();
        let grid = ((count as u32).div_ceil(source::ELEMENTWISETILE), 1, 1);
        let block = (source::ELEMENTWISETILE, 1, 1);
        let mut results = dispatch(
            cache,
            "elementwise",
            source::MULENTRY,
            &[buf::host(lhs), buf::host(rhs), buf::out(count)],
            &[scalar::u32(count as u32)],
            grid,
            block,
            0,
        )?;
        Ok(results.remove(0))
    }

    /// Sigmoid linear unit, elementwise.
    pub(crate) fn silu(cache: &cache, input: &[f32]) -> Result<Vec<f32>, gpuerror> {
        let count = input.len();
        let grid = ((count as u32).div_ceil(source::ELEMENTWISETILE), 1, 1);
        let block = (source::ELEMENTWISETILE, 1, 1);
        let mut results = dispatch(
            cache,
            "elementwise",
            source::SILUENTRY,
            &[buf::host(input), buf::out(count)],
            &[scalar::u32(count as u32)],
            grid,
            block,
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
        let block = (source::ROWTILE, 1, 1);
        let mut results = dispatch(
            cache,
            "rmsnorm",
            source::RMSNORMENTRY,
            &[buf::host(input), buf::host(weight), buf::out(count)],
            &[scalar::u32(width as u32), scalar::f32(eps)],
            grid,
            block,
            source::ROWTILE as u32 * 4,
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
        let block = (source::ELEMENTWISETILE, 1, 1);
        let mut results = dispatch(
            cache,
            "softmax",
            source::SOFTMAXENTRY,
            &[buf::host(input), buf::out(count)],
            &[scalar::u32(width as u32)],
            grid,
            block,
            source::ELEMENTWISETILE as u32 * 4,
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
        let grid = ((seqaxis as u32).div_ceil(source::ELEMENTWISETILE), 1, 1);
        let block = (source::ELEMENTWISETILE, 1, 1);
        let mut results = dispatch(
            cache,
            "rope",
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
        let block = (source::ELEMENTWISETILE, 1, 1);
        let mut results = dispatch(
            cache,
            "at",
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
        let block = (source::ELEMENTWISETILE, 1, 1);
        let mut results = dispatch(
            cache,
            "embedding",
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
            "attention",
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
            (2 * source::ATTENTIONLANES as u32 + headdim as u32) * 4,
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
            "cached",
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
            (2 * source::ATTENTIONLANES as u32 + headdim as u32) * 4,
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
        let block = (source::ELEMENTWISETILE, 1, 1);
        let mut results = dispatch(
            cache,
            "scan",
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
            "scan",
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
            (source::ELEMENTWISETILE, 1, 1),
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
            "scan",
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
            (source::ELEMENTWISETILE, 1, 1),
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
        let mut totals = dispatchmixed(
            cache,
            "parallel",
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
            "parallel",
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
            "parallel",
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
            "scan",
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
            (source::ELEMENTWISETILE, 1, 1),
            0,
        )?;
        let out = results.remove(0);
        let newstate = results.remove(0);
        Ok((out, newstate))
    }
}

#[cfg(target_os = "macos")]
pub(crate) mod metal {
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
    //! Every kernel in [`crate::metal::source`] reuses those same sites, so the count does
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

    use crate::engine::gpuerror;
    use crate::metal::source;
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
        /// [`crate::metal::source`] can never reuse a stale binary. The map is
        /// device-specific by construction: it lives beside the one device it was
        /// compiled for, so no device belongs in the key.
        pipelines: HashMap<
            (&'static str, &'static str),
            Retained<ProtocolObject<dyn MTLComputePipelineState>>,
        >,
        /// One compiled library per source text.
        ///
        /// Every entry shares the single `metal/metal.c` source, so the library is
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

    /// Compiles `source` into a library, once per source text.
    fn compilelibrary(
        device: &ProtocolObject<dyn MTLDevice>,
        source: &str,
    ) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>, gpuerror> {
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
    fn dispatch<P>(
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
    fn alloc(
        device: &ProtocolObject<dyn MTLDevice>,
        bytes: usize,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, gpuerror> {
        device
            .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
            .ok_or(gpuerror::allocation)
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
        pub(crate) fn apply(&mut self) -> Result<Vec<Vec<f32>>, gpuerror> {
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
}
