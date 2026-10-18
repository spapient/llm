//! GRID: unified multi-vendor GPU backend.
//!
//! `grid` is the crate's single backend namespace for every vendor GPU: one
//! module per platform (`cuda`, `metal`, `musa`, `rocm`, `sycl`, `cann`,
//! `mlu`), each a Layer 1 backend implementing the frozen [`api::backend`]
//! and [`api::compute`] contracts for `f32` operands on its device. No backend
//! owns request state or model weights; every operator call creates, uses, and
//! releases its own device resources (docs/design.md §21.6).
//!
//! # Platforms
//!
//! Each module compiles only where its toolchain exists, through the cfg
//! attributes below plus the `*_toolkit` cfgs emitted by `build.rs`:
//!
//! | module | device | toolchain | kernels |
//! | --- | --- | --- | --- |
//! | `cuda` | NVIDIA | NVRTC / cuBLAS (runtime) | `grid/cuda.c` |
//! | `metal` | Apple | Metal runtime | `grid/metal.c` |
//! | `musa` | Moore Threads | `mcc` | `grid/musa.c` (+WMMA island) |
//! | `rocm` | AMD / Hygon | `hipcc` + hipBLAS | `grid/cuda.c` (HIP) |
//! | `sycl` | Intel | `icpx` | `grid/sycl.c` |
//! | `cann` | Huawei Ascend | `ccec`/`bisheng` | `grid/cann.c` |
//! | `mlu` | Cambricon | `cncc` | `grid/mlu.c` |
//!
//! All platforms execute the same 15-operator contract from the shared
//! `grid/` kernel family — one translation unit per vendor language, compiled
//! at build time by the vendor compiler or at run time (NVRTC / Metal
//! `newLibraryWithSource`), with the shared `grid/grid.h` contract header
//! single-written across dialects.
//!
//! # Unsafe
//!
//! Every backend is a carrier of the workspace's restricted `unsafe`
//! exemption (docs/design.md §21.8): the crate root uses
//! `#![deny(unsafe_code)]`; `ffi` declares each backend's allowance and
//! confines every platform call there, each preceded by a `// SAFETY:`
//! comment. `scripts/checkunsafe.sh` enforces the bounded site counts
//! mechanically.
//!
//! # Naming
//!
//! Per the repository no-separator convention, all public type, trait, enum,
//! variant, and function names are lowercase concatenated, and constants are
//! uppercase concatenated. Module file names differ from the public types
//! they export because a module and a re-exported type share one name
//! namespace inside the backend namespace.

#[cfg(target_os = "linux")]
pub mod cuda {
    //! NVIDIA CUDA GPU execution backend.
    //!
    //! `cuda` is a Layer 1 backend: it implements the frozen [`api::backend`] and
    //! [`api::compute`] contracts for `f32` operands on GPU device 0. It owns no
    //! request state and holds no model weights. An executor owns the retained
    //! device context and the kernels compiled from the sources at the crate root,
    //! while every operator call takes, uses, and releases its own execution stream
    //! (docs/design.md §21.6), so no operand or result survives a call.
    //!
    //! # Platform
    //!
    //! The module is compiled only on Linux; on every other target the crate
    //! exposes no `cuda` namespace and `cargo check --workspace --all-targets`
    //! stays green.
    //!
    //! # Two mechanisms
    //!
    //! The operator set is served by two CUDA-specific mechanisms, chosen per
    //! operator rather than per convenience (docs/design.md §21.2 decision 11):
    //!
    //! * **cuBLAS** serves `matmul` — the CUDA-specific fast path.
    //! * **In-repo CUDA C++** compiled at run time through NVRTC serves everything
    //!   else: `softmax`, `add`, `mul`, `silu`, `rmsnorm`, `rope`, `embedding`, and
    //!   a fused `attention`. The NVRTC matmul kernel remains available through
    //!   `cudaexecutor::newnvrtc` as the independent reference the cuBLAS path is
    //!   validated against.
    //!
    //! `softmax` uses the in-repo max-subtracting kernel rather than cuDNN: the
    //! kernel reuses the NVRTC launch site (no exemption growth), implements the
    //! same formulation as `cpu`/`metal`, and avoids cuDNN's per-call handle
    //! creation. Removing the cuDNN path reduced the backend from three `unsafe`
    //! sites to two.
    //!
    //! # Unsafe
    //!
    //! This backend is one of the carriers of the workspace's restricted `unsafe`
    //! exemption (see docs/design.md §21.8). The crate root uses
    //! `#![deny(unsafe_code)]`; `ffi` declares this backend's one allowance and
    //! confines every platform call there, each preceded by a `// SAFETY:` comment.
    //! The registered sites are two, one per class of platform interaction: the
    //! NVRTC kernel launch and the cuBLAS `gemm`. `scripts/checkunsafe.sh` enforces
    //! the exact site count and the SAFETY comments mechanically.
    //!
    //! # Naming
    //!
    //! Per the repository no-separator convention, all public type, trait, enum,
    //! variant, and function names are lowercase concatenated, and constants are
    //! uppercase concatenated. Module file names differ from the public types they
    //! export because a module and a re-exported type share one name namespace
    //! inside the backend namespace.

    use crate::ffi::cuda as ffi;

    /// The CUDA view of the shared kernel catalog.
    pub(crate) mod source {
        pub(crate) use crate::catalog::cu::{
            ADDENTRY, AGGENTRY as SCANAGGENTRY, APPLYENTRY as SCANAPPLYENTRY, ATTENTIONCACHEDENTRY,
            ATTENTIONENTRY, ATTENTIONLANES, ATTNLIMIT, CHUNK as SCANCHUNK, EMBEDDINGENTRY,
            FLATTILE, MULENTRY, PREFIXENTRY as SCANPREFIXENTRY, RMSNORMENTRY, RMSNORMTILE,
            ROPEATENTRY, ROPEENTRY, SCANBLOCKDIM, SCANENTRY, SCANMAXN, SILUENTRY, SOFTMAXENTRY,
            SOFTMAXTILE, SOURCE,
        };
        pub(crate) use crate::catalog::cu::{MATMULENTRY, THREADEDGE};
    }

    /// The maximum tensor rank the CUDA backend accepts.
    pub const CUDAMAXRANK: usize = crate::engine::MAXRANK;

    /// The maximum head dimension the attention kernel accepts.
    ///
    /// The value is a device limit rather than a contract limit: the kernel keeps
    /// one accumulator slot per head dimension in shared memory, so its request is
    /// `(2 * 256 + headdim) * 4` bytes. At this bound that is 34 KiB, which stays
    /// under the 48 KiB of dynamic shared memory a threadgroup may use without the
    /// opt-in that `cudarc` does not currently expose. Basis (GOV7): 34 KiB <
    /// 48 KiB leaves ~14 KiB of headroom for driver accounting.
    pub const CUDAMAXHEADDIM: usize = 8192;

    /// The CUDA device identity.
    pub struct cudadevice;

    impl crate::engine::devicespec for cudadevice {
        const NAME: &'static str = "cuda";
        const MAXHEADDIM: usize = CUDAMAXHEADDIM;

        fn device() -> api::device {
            api::device::cuda()
        }
    }

    /// The CUDA execution backend.
    pub type cudabackend = crate::engine::gpubackend<cudadevice>;

    use std::sync::Arc;

    use ::tensor::shape;
    use api::{backend, backenderror, capabilities};
    use cudarc::driver::CudaSlice;
    use lib::limits;

    use api::geometry::{attentionshape, axisshape, embeddingshape, matmulshape, ropeshape};

    use crate::engine::gpuerror;
    use crate::engine::{executor, platform};

    /// Which CUDA implementation serves `matmul`.
    ///
    /// The choice is made once, at construction, and stored as a field. Selecting
    /// it per call would add a run-time branch to the hottest path for no benefit,
    /// and would obscure which path a given measurement exercised.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum matmulpath {
        /// cuBLAS `gemm` — the CUDA-specific fast path, used by [`cudaexecutor::new`].
        cublas,
        /// The in-repo kernel compiled at run time through NVRTC, used by
        /// [`cudaexecutor::newnvrtc`] as the independent reference path.
        nvrtc,
    }

    /// The CUDA device primitives behind [`cudaexecutor`].
    #[derive(Clone, Debug)]
    pub struct cudaplatform {
        backend: cudabackend,
        path: matmulpath,
        cache: Arc<ffi::cache>,
    }

    impl cudaplatform {
        fn new(lim: limits, path: matmulpath) -> Self {
            Self {
                backend: cudabackend::new(lim),
                path,
                cache: Arc::new(ffi::cache::new()),
            }
        }

        /// Runs one block of `[m, k] * [k, n]` on the configured path.
        fn dispatch(
            &self,
            lhs: &[f32],
            rhs: &[f32],
            m: usize,
            k: usize,
            n: usize,
        ) -> Result<Vec<f32>, gpuerror> {
            match self.path {
                matmulpath::cublas => ffi::matmul_cublas(&self.cache, lhs, rhs, m, k, n, 1),
                matmulpath::nvrtc => ffi::matmul_nvrtc(&self.cache, lhs, rhs, m, k, n),
            }
        }
    }

    impl platform for cudaplatform {
        const NAME: &'static str = "cuda";
        const MAXHEADDIM: usize = CUDAMAXHEADDIM;

        fn capabilities(&self) -> &capabilities {
            self.backend.getcapabilities()
        }

        fn matmul(
            &self,
            lhs: &[f32],
            rhs: &[f32],
            geometry: &matmulshape,
        ) -> Result<Vec<f32>, backenderror> {
            match self.path {
                matmulpath::cublas => ffi::matmul_cublas(
                    &self.cache,
                    lhs,
                    rhs,
                    geometry.m,
                    geometry.k,
                    geometry.n,
                    geometry.batch,
                )
                .map_err(maptobackend),
                matmulpath::nvrtc => {
                    let mut data = Vec::new();
                    data.try_reserve_exact(geometry.m * geometry.n * geometry.batch)
                        .map_err(|_| backenderror::resourceexhausted)?;
                    for b in 0..geometry.batch {
                        let lstart = b * geometry.m * geometry.k;
                        let rstart = b * geometry.k * geometry.n;
                        let lblock = &lhs[lstart..lstart + geometry.m * geometry.k];
                        let rblock = &rhs[rstart..rstart + geometry.k * geometry.n];
                        let block = self
                            .dispatch(lblock, rblock, geometry.m, geometry.k, geometry.n)
                            .map_err(maptobackend)?;
                        data.extend_from_slice(&block);
                    }
                    Ok(data)
                }
            }
        }

        fn add(&self, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, backenderror> {
            ffi::add(&self.cache, lhs, rhs).map_err(maptobackend)
        }

        fn mul(&self, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, backenderror> {
            ffi::mul(&self.cache, lhs, rhs).map_err(maptobackend)
        }

        fn rmsnorm(
            &self,
            input: &[f32],
            weight: &[f32],
            geometry: &axisshape,
            eps: f32,
        ) -> Result<Vec<f32>, backenderror> {
            ffi::rmsnorm(
                &self.cache,
                input,
                weight,
                geometry.rows,
                geometry.width,
                eps,
            )
            .map_err(maptobackend)
        }

        fn softmax(&self, input: &[f32], geometry: &axisshape) -> Result<Vec<f32>, backenderror> {
            ffi::softmax(&self.cache, input, geometry.rows, geometry.width).map_err(maptobackend)
        }

        fn silu(&self, input: &[f32]) -> Result<Vec<f32>, backenderror> {
            ffi::silu(&self.cache, input).map_err(maptobackend)
        }

        fn embedding(
            &self,
            table: &[f32],
            indices: &[usize],
            geometry: &embeddingshape,
        ) -> Result<Vec<f32>, backenderror> {
            let index64: Vec<u64> = indices.iter().map(|&i| i as u64).collect();
            ffi::embedding(&self.cache, table, &index64, geometry.rows, geometry.width)
                .map_err(maptobackend)
        }

        fn rope(
            &self,
            input: &[f32],
            geometry: &ropeshape,
            base: f32,
        ) -> Result<Vec<f32>, backenderror> {
            ffi::rope(
                &self.cache,
                input,
                geometry.rows,
                geometry.seqaxis,
                geometry.heads,
                geometry.headdim,
                base,
            )
            .map_err(maptobackend)
        }

        fn rope_at(
            &self,
            input: &[f32],
            position: usize,
            geometry: &ropeshape,
            base: f32,
        ) -> Result<Vec<f32>, backenderror> {
            // CUDA has no absolute-position kernel: pad the history on the host and
            // reuse the relative kernel, which is exactly the reference semantics.
            let width = geometry
                .heads
                .checked_mul(geometry.headdim)
                .ok_or(backenderror::sizeoverflow)?;
            let rows = position.checked_add(1).ok_or(backenderror::sizeoverflow)?;
            let mut data =
                vec![0.0_f32; rows.checked_mul(width).ok_or(backenderror::sizeoverflow)?];
            data[position * width..].copy_from_slice(input);
            let padded = shape::new(vec![rows, width]).map_err(|_| backenderror::sizeoverflow)?;
            let paddedgeometry = ropeshape::new(&padded, geometry.heads)?;
            let rotated = ffi::rope(
                &self.cache,
                &data,
                paddedgeometry.rows,
                paddedgeometry.seqaxis,
                paddedgeometry.heads,
                paddedgeometry.headdim,
                base,
            )
            .map_err(maptobackend)?;
            Ok(rotated[position * width..position * width + width].to_vec())
        }

        fn attention(
            &self,
            query: &[f32],
            key: &[f32],
            value: &[f32],
            geometry: &attentionshape,
            causal: bool,
        ) -> Result<Vec<f32>, backenderror> {
            ffi::attention(
                &self.cache,
                query,
                key,
                value,
                geometry.qseq,
                geometry.kvseq,
                geometry.heads,
                geometry.kvheads,
                geometry.headdim,
                causal,
            )
            .map_err(maptobackend)
        }

        fn scan(
            &self,
            input: &[f32],
            init: &[f32],
            a: &[f32],
            b: &[f32],
            c: &[f32],
            skip: &[f32],
            n: usize,
            d: usize,
            statewidth: usize,
        ) -> Result<(Vec<f32>, Vec<f32>), backenderror> {
            ffi::scan(&self.cache, input, init, a, b, c, skip, n, d, statewidth)
                .map_err(maptobackend)
        }
    }

    /// A CUDA executor: the crate's [`compute`](api::compute) provider.
    ///
    /// The orchestration is shared with the other backends ([`crate::engine`]);
    /// `cudaexecutor` is that engine instantiated over [`cudaplatform`].
    pub type cudaexecutor = executor<cudaplatform>;

    impl executor<cudaplatform> {
        /// Creates an executor whose `matmul` is served by cuBLAS.
        #[must_use]
        pub fn new(lim: limits) -> Self {
            Self::newplatform(cudaplatform::new(lim, matmulpath::cublas))
        }

        /// Creates an executor whose `matmul` is served by the NVRTC kernel.
        ///
        /// This is the independent path the cuBLAS path is validated against.
        #[must_use]
        pub fn newnvrtc(lim: limits) -> Self {
            Self::newplatform(cudaplatform::new(lim, matmulpath::nvrtc))
        }

        /// Returns how many kernels this executor has compiled so far.
        #[must_use]
        pub fn getcachedkernelcount(&self) -> usize {
            self.getplatform().cache.kernelcount()
        }

        /// Returns the backend whose budget this executor enforces.
        #[must_use]
        pub fn getbackend(&self) -> &cudabackend {
            &self.getplatform().backend
        }

        /// Returns which implementation currently serves `matmul`.
        #[must_use]
        pub fn getpath(&self) -> matmulpath {
            self.getplatform().path
        }

        /// Creates a single-use accumulative submission over this executor's
        /// backend, matmul path, and compiled-kernel cache.
        ///
        /// # Errors
        ///
        /// Returns `gpuerror::device` when no CUDA device is available, the same
        /// first-call condition as any operator dispatch.
        pub fn newsubmission(&self) -> Result<cudasubmission, backenderror> {
            cudasubmission::new(self.getplatform().clone())
        }

        /// Creates a request-scoped persistent session over this executor's device
        /// state and compiled-kernel cache.
        ///
        /// A session records operator chains repeatedly over one device state, so a
        /// decode loop does not rebuild it on every step.
        ///
        /// # Errors
        ///
        /// Returns `backenderror::devicenotavailable` when no CUDA device is
        /// available, the same first-call condition as any operator dispatch.
        pub fn newsession(&self) -> Result<cudasession, backenderror> {
            cudasession::new(self.getplatform().clone())
        }
    }

    /// Maps a crate-local [`gpuerror`] onto the frozen `api::backenderror`.
    ///
    /// Device absence and kernel compilation/module loading are surfaced as their
    /// own variants so a caller can tell "no GPU on this host" from "the kernel did
    /// not compile"; the kernel entry point rides along so a compile failure
    /// identifies exactly which `(source, entry)` pair failed. Transfer, launch,
    /// and cuBLAS runtime failures stay under the generic `executionfailed`.
    fn maptobackend(error: gpuerror) -> backenderror {
        error.maptobackend("cuda")
    }

    /// A CUDA submission: one execution stream shared by every recorded step, so a
    /// whole operator chain launches in order and synchronises once
    /// (docs/design.md §21.13).
    pub type cudasubmission = crate::engine::submission<cudaplatform>;

    /// A CUDA persistent session: recorded chains applied repeatedly over one
    /// device state.
    pub type cudasession = crate::engine::session<cudaplatform>;

    /// The CUDA recording-session handle (implementation detail).
    #[doc(hidden)]
    pub struct cudarecordingsession(ffi::session);

    impl crate::engine::recordplatform for cudaplatform {
        type buffer = CudaSlice<f32>;
        type session = cudarecordingsession;

        fn newsession(&self) -> Result<Self::session, backenderror> {
            ffi::session::new(&self.cache)
                .map(cudarecordingsession)
                .map_err(maptobackend)
        }

        fn upload(
            &self,
            session: &Self::session,
            data: &[f32],
        ) -> Result<Self::buffer, backenderror> {
            session.0.upload(data).map_err(maptobackend)
        }

        fn getbuffer(&self, session: &Self::session, slot: usize) -> Option<Self::buffer> {
            session.0.getbuffer(slot).cloned()
        }

        fn recordpersist(
            &self,
            session: &mut Self::session,
            data: &[f32],
            _shape: &shape,
        ) -> Result<usize, backenderror> {
            session.0.persist(data).map_err(maptobackend)
        }

        fn gethostdata(&self, session: &Self::session, slot: usize) -> Option<Vec<f32>> {
            session.0.gethostdata(slot)
        }

        fn applysession(&self, session: &mut Self::session) -> Result<Vec<Vec<f32>>, backenderror> {
            session.0.apply().map_err(maptobackend)
        }

        fn recordmatmul(
            &self,
            session: &mut Self::session,
            lhs: &Self::buffer,
            rhs: &Self::buffer,
            geometry: &matmulshape,
        ) -> Result<usize, backenderror> {
            if geometry.batch != 1 {
                return Err(backenderror::unsupportedoperation { op: "matmul" });
            }
            match self.path {
                matmulpath::cublas => ffi::matmul_cublas_into(
                    &self.cache,
                    &mut session.0,
                    lhs,
                    rhs,
                    geometry.m,
                    geometry.k,
                    geometry.n,
                ),
                matmulpath::nvrtc => ffi::matmul_nvrtc_into(
                    &self.cache,
                    &mut session.0,
                    lhs,
                    rhs,
                    geometry.m,
                    geometry.k,
                    geometry.n,
                ),
            }
            .map_err(maptobackend)
        }

        fn recordadd(
            &self,
            session: &mut Self::session,
            lhs: &Self::buffer,
            rhs: &Self::buffer,
            count: usize,
        ) -> Result<usize, backenderror> {
            ffi::add_into(&self.cache, &mut session.0, lhs, rhs, count).map_err(maptobackend)
        }

        fn recordmul(
            &self,
            session: &mut Self::session,
            lhs: &Self::buffer,
            rhs: &Self::buffer,
            count: usize,
        ) -> Result<usize, backenderror> {
            ffi::mul_into(&self.cache, &mut session.0, lhs, rhs, count).map_err(maptobackend)
        }

        fn recordrmsnorm(
            &self,
            session: &mut Self::session,
            input: &Self::buffer,
            weight: &Self::buffer,
            geometry: &axisshape,
            eps: f32,
        ) -> Result<usize, backenderror> {
            ffi::rmsnorm_into(
                &self.cache,
                &mut session.0,
                input,
                weight,
                geometry.rows,
                geometry.width,
                eps,
            )
            .map_err(maptobackend)
        }

        fn recordsoftmax(
            &self,
            session: &mut Self::session,
            input: &Self::buffer,
            geometry: &axisshape,
        ) -> Result<usize, backenderror> {
            ffi::softmax_into(
                &self.cache,
                &mut session.0,
                input,
                geometry.rows,
                geometry.width,
            )
            .map_err(maptobackend)
        }

        fn recordsilu(
            &self,
            session: &mut Self::session,
            input: &Self::buffer,
            count: usize,
        ) -> Result<usize, backenderror> {
            ffi::silu_into(&self.cache, &mut session.0, input, count).map_err(maptobackend)
        }

        fn recordembedding(
            &self,
            session: &mut Self::session,
            table: &Self::buffer,
            indices: &[usize],
            geometry: &embeddingshape,
        ) -> Result<usize, backenderror> {
            ffi::embedding_into(&self.cache, &mut session.0, table, indices, geometry.width)
                .map_err(maptobackend)
        }

        fn recordrope(
            &self,
            session: &mut Self::session,
            input: &Self::buffer,
            geometry: &ropeshape,
            base: f32,
        ) -> Result<usize, backenderror> {
            ffi::rope_into(&self.cache, &mut session.0, input, geometry, base).map_err(maptobackend)
        }

        fn recordrope_at(
            &self,
            session: &mut Self::session,
            input: &Self::buffer,
            position: usize,
            geometry: &ropeshape,
            base: f32,
        ) -> Result<usize, backenderror> {
            ffi::rope_at_into(
                &self.cache,
                &mut session.0,
                input,
                position,
                geometry.heads,
                geometry.headdim,
                base,
            )
            .map_err(maptobackend)
        }

        fn recordattention(
            &self,
            session: &mut Self::session,
            query: &Self::buffer,
            key: &Self::buffer,
            value: &Self::buffer,
            geometry: &attentionshape,
            causal: bool,
        ) -> Result<usize, backenderror> {
            ffi::attention_into(
                &self.cache,
                &mut session.0,
                query,
                key,
                value,
                geometry,
                causal,
            )
            .map_err(maptobackend)
        }

        fn recordattention_cached(
            &self,
            session: &mut Self::session,
            query: &Self::buffer,
            keycached: &Self::buffer,
            keynew: &Self::buffer,
            valuecached: &Self::buffer,
            valuenew: &Self::buffer,
            geometry: &attentionshape,
            causal: bool,
        ) -> Result<usize, backenderror> {
            ffi::attention_cached_into(
                &self.cache,
                &mut session.0,
                query,
                ffi::bufferpair {
                    cached: keycached,
                    new: keynew,
                },
                ffi::bufferpair {
                    cached: valuecached,
                    new: valuenew,
                },
                geometry,
                causal,
            )
            .map_err(maptobackend)
        }
    }
}
#[cfg(target_os = "macos")]
pub mod metal {
    //! Apple Metal GPU execution backend.
    //!
    //! `metal` is a Layer 1 backend: it implements the frozen [`api::backend`] and
    //! [`api::compute`] contracts for `f32` operands on the default Metal device.
    //! It owns no request state and holds no model weights; every operator call
    //! creates, uses, and releases its own device resources.
    //!
    //! # Platform
    //!
    //! The module is compiled only on macOS; on every other target the crate
    //! exposes no `metal` namespace and `cargo check --workspace --all-targets`
    //! stays green.
    //!
    //! # Unsafe
    //!
    //! This backend is one of the carriers of the workspace's restricted `unsafe`
    //! exemption (see docs/design.md §21.8). The crate root uses
    //! `#![deny(unsafe_code)]`; `ffi` declares this backend's one allowance and
    //! confines every platform call there, each preceded by a `// SAFETY:` comment.
    //! `scripts/checkunsafe.sh` enforces the bounded site count and the SAFETY
    //! comments mechanically.
    //!
    //! # Naming
    //!
    //! Per the repository no-separator convention, all public type, trait, enum,
    //! variant, and function names are lowercase concatenated, and constants are
    //! uppercase concatenated. Module file names differ from the public types they
    //! export because a module and a re-exported type share one name namespace
    //! inside the backend namespace.

    use crate::ffi::metal as ffi;

    /// The Metal view of the shared kernel catalog.
    ///
    /// Every entry compiles from the single `grid/metal.c` translation unit, so
    /// all `*MSL` constants below are aliases of one source.
    pub(crate) mod source {
        pub(crate) use crate::catalog::msl::{
            ADDENTRY, ATTENTIONCACHEDENTRY, ATTENTIONENTRY, ATTENTIONLANES, ATTNLIMIT, FLATTILE,
            MULENTRY, RMSNORMENTRY, RMSNORMTILE, ROPEATENTRY, ROPEENTRY, SILUENTRY,
        };
        pub(crate) const ATTENTIONMSL: &str = crate::catalog::msl::SOURCE;
        pub(crate) const ATTENTIONCACHEDMSL: &str = crate::catalog::msl::SOURCE;
        pub(crate) const ELEMENTWISE: &str = crate::catalog::msl::SOURCE;
        pub(crate) const EMBEDDINGMSL: &str = crate::catalog::msl::SOURCE;
        pub(crate) const MATMULMSL: &str = crate::catalog::msl::SOURCE;
        pub(crate) const RMSNORMMSL: &str = crate::catalog::msl::SOURCE;
        pub(crate) const ROPEMSL: &str = crate::catalog::msl::SOURCE;
        pub(crate) const ROPEATMSL: &str = crate::catalog::msl::SOURCE;
        pub(crate) const SCANMSL: &str = crate::catalog::msl::SOURCE;
        pub(crate) const SCANPARALLELMSL: &str = crate::catalog::msl::SOURCE;
        pub(crate) const SOFTMAXMSL: &str = crate::catalog::msl::SOURCE;

        pub(crate) use crate::catalog::msl::{
            AGGENTRY as SCANAGGENTRY, APPLYENTRY as SCANAPPLYENTRY, CHUNK as SCANCHUNK,
            EMBEDDINGENTRY, MATMULENTRY, PREFIXENTRY as SCANPREFIXENTRY, SCANENTRY, SCANMAXN,
            SOFTMAXENTRY, SOFTMAXTILE,
        };
    }

    /// The maximum tensor rank the METAL backend accepts.
    pub const METALMAXRANK: usize = crate::engine::MAXRANK;

    /// The maximum head dimension the attention kernel accepts.
    ///
    /// The value is a device limit rather than a contract limit: the kernel keeps
    /// one accumulator slot per head dimension in threadgroup memory, so its
    /// request is `(2 * ATTENTIONLANES + METALMAXHEADDIM) * 4` bytes. Apple GPUs
    /// expose 32 KiB of threadgroup memory per threadgroup, so at this bound the
    /// request is 18 432 B, leaving about 14 KiB of headroom. Basis (GOV7):
    /// 18 432 B < 32 768 B. The bound MUST stay equal to the `accum` array in the
    /// attention kernel of `grid/metal.c`.
    pub const METALMAXHEADDIM: usize = 4096;

    /// The METAL device identity.
    pub struct metaldevice;

    impl crate::engine::devicespec for metaldevice {
        const NAME: &'static str = "metal";
        const MAXHEADDIM: usize = METALMAXHEADDIM;

        fn device() -> api::device {
            api::device::metal()
        }
    }

    /// The METAL execution backend.
    pub type metalbackend = crate::engine::gpubackend<metaldevice>;

    use std::sync::Arc;

    use ::tensor::shape;
    use api::{backend, backenderror, capabilities};
    use lib::limits;
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_metal::MTLBuffer;

    use api::geometry::{attentionshape, axisshape, embeddingshape, matmulshape, ropeshape};

    use crate::engine::gpuerror;
    use crate::engine::{executor, platform};

    /// The Metal device primitives behind [`metalexecutor`].
    #[derive(Clone, Debug)]
    pub struct metalplatform {
        backend: metalbackend,
        cache: Arc<ffi::cache>,
    }

    impl metalplatform {
        fn new(lim: limits) -> Self {
            Self {
                backend: metalbackend::new(lim),
                cache: Arc::new(ffi::cache::new()),
            }
        }
    }

    impl platform for metalplatform {
        const NAME: &'static str = "metal";
        const MAXHEADDIM: usize = METALMAXHEADDIM;

        fn capabilities(&self) -> &capabilities {
            self.backend.getcapabilities()
        }

        fn matmul(
            &self,
            lhs: &[f32],
            rhs: &[f32],
            geometry: &matmulshape,
        ) -> Result<Vec<f32>, backenderror> {
            // One dispatch covers every batch block: the block index travels in
            // the grid (docs/design.md §21.10.2).
            ffi::matmul(
                &self.cache,
                lhs,
                rhs,
                geometry.batch,
                geometry.m,
                geometry.k,
                geometry.n,
            )
            .map_err(maptobackend)
        }

        fn add(&self, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, backenderror> {
            ffi::add(&self.cache, lhs, rhs).map_err(maptobackend)
        }

        fn mul(&self, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, backenderror> {
            ffi::mul(&self.cache, lhs, rhs).map_err(maptobackend)
        }

        fn rmsnorm(
            &self,
            input: &[f32],
            weight: &[f32],
            geometry: &axisshape,
            eps: f32,
        ) -> Result<Vec<f32>, backenderror> {
            ffi::rmsnorm(
                &self.cache,
                input,
                weight,
                geometry.rows,
                geometry.width,
                eps,
            )
            .map_err(maptobackend)
        }

        fn softmax(&self, input: &[f32], geometry: &axisshape) -> Result<Vec<f32>, backenderror> {
            ffi::softmax(&self.cache, input, geometry.rows, geometry.width).map_err(maptobackend)
        }

        fn silu(&self, input: &[f32]) -> Result<Vec<f32>, backenderror> {
            ffi::silu(&self.cache, input).map_err(maptobackend)
        }

        fn embedding(
            &self,
            table: &[f32],
            indices: &[usize],
            geometry: &embeddingshape,
        ) -> Result<Vec<f32>, backenderror> {
            ffi::embedding(&self.cache, table, indices, geometry.width).map_err(maptobackend)
        }

        fn rope(
            &self,
            input: &[f32],
            geometry: &ropeshape,
            base: f32,
        ) -> Result<Vec<f32>, backenderror> {
            ffi::rope(
                &self.cache,
                input,
                geometry.rows,
                geometry.seqaxis,
                geometry.heads,
                geometry.headdim,
                base,
            )
            .map_err(maptobackend)
        }

        fn rope_at(
            &self,
            input: &[f32],
            position: usize,
            geometry: &ropeshape,
            base: f32,
        ) -> Result<Vec<f32>, backenderror> {
            ffi::rope_at(
                &self.cache,
                input,
                position,
                geometry.heads,
                geometry.headdim,
                base,
            )
            .map_err(maptobackend)
        }

        fn attention(
            &self,
            query: &[f32],
            key: &[f32],
            value: &[f32],
            geometry: &attentionshape,
            causal: bool,
        ) -> Result<Vec<f32>, backenderror> {
            ffi::attention(&self.cache, query, key, value, geometry, causal).map_err(maptobackend)
        }

        fn scan(
            &self,
            input: &[f32],
            init: &[f32],
            a: &[f32],
            b: &[f32],
            c: &[f32],
            skip: &[f32],
            n: usize,
            d: usize,
            statewidth: usize,
        ) -> Result<(Vec<f32>, Vec<f32>), backenderror> {
            ffi::scan(&self.cache, input, init, a, b, c, skip, n, d, statewidth)
                .map_err(maptobackend)
        }
    }

    /// A Metal executor: the crate's [`compute`](api::compute) provider.
    ///
    /// The orchestration is shared with the other backends ([`crate::engine`]);
    /// `metalexecutor` is that engine instantiated over [`metalplatform`].
    pub type metalexecutor = executor<metalplatform>;

    impl executor<metalplatform> {
        /// Creates an executor over a Metal backend configured with `lim`.
        #[must_use]
        pub fn new(lim: limits) -> Self {
            Self::newplatform(metalplatform::new(lim))
        }

        /// Returns the backend whose budget this executor enforces.
        #[must_use]
        pub fn getbackend(&self) -> &metalbackend {
            &self.getplatform().backend
        }

        /// Returns how many compute pipelines this executor has compiled so far.
        #[must_use]
        pub fn getcachedpipelinecount(&self) -> usize {
            self.getplatform().cache.pipelinecount()
        }

        /// Creates a single-use accumulative submission over this executor's backend
        /// and compiled-pipeline cache.
        ///
        /// # Errors
        ///
        /// Returns `gpuerror::nodevice` when no Metal device is available, the
        /// same first-call condition as any operator dispatch.
        pub fn newsubmission(&self) -> Result<metalsubmission, backenderror> {
            metalsubmission::new(self.getplatform().clone())
        }

        /// Creates a request-scoped persistent session over this executor's device
        /// state and compiled-pipeline cache.
        ///
        /// A session records operator chains repeatedly over one device state, so a
        /// decode loop does not rebuild it on every step.
        ///
        /// # Errors
        ///
        /// Returns `backenderror::devicenotavailable` when no Metal device is
        /// available, the same first-call condition as any operator dispatch.
        pub fn newsession(&self) -> Result<metalsession, backenderror> {
            metalsession::new(self.getplatform().clone())
        }
    }

    /// Maps a crate-local [`gpuerror`] onto the frozen `api::backenderror`.
    ///
    /// Device absence and kernel compilation are surfaced as their own variants so
    /// a caller can tell "no GPU on this host" from "the kernel did not compile";
    /// the kernel entry point rides along so a compile failure identifies exactly
    /// which `(source, entry)` pair failed. Allocation and runtime execution
    /// failures stay under the generic `executionfailed` because no caller action
    /// distinguishes them.
    fn maptobackend(error: gpuerror) -> backenderror {
        error.maptobackend("metal")
    }

    /// A Metal submission: one command buffer and encoder shared by every recorded
    /// step, so a whole operator chain encodes in order, submits once, and
    /// synchronises once (docs/design.md §21.13).
    pub type metalsubmission = crate::engine::submission<metalplatform>;

    /// A Metal persistent session: recorded chains applied repeatedly over one
    /// device state.
    pub type metalsession = crate::engine::session<metalplatform>;

    /// The Metal recording-session handle (implementation detail).
    #[doc(hidden)]
    pub struct metalrecordingsession(ffi::session);

    impl crate::engine::recordplatform for metalplatform {
        type buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
        type session = metalrecordingsession;

        const DEDUP_INPUTS: bool = true;

        fn newsession(&self) -> Result<Self::session, backenderror> {
            ffi::session::new(&self.cache)
                .map(metalrecordingsession)
                .map_err(maptobackend)
        }

        fn upload(
            &self,
            session: &Self::session,
            data: &[f32],
        ) -> Result<Self::buffer, backenderror> {
            session.0.upload(data).map_err(maptobackend)
        }

        fn getbuffer(&self, session: &Self::session, slot: usize) -> Option<Self::buffer> {
            session.0.getbuffer(slot).cloned()
        }

        fn recordpersist(
            &self,
            session: &mut Self::session,
            data: &[f32],
            _shape: &shape,
        ) -> Result<usize, backenderror> {
            session.0.persist(data).map_err(maptobackend)
        }

        fn recordappend(
            &self,
            session: &mut Self::session,
            cached: &Retained<ProtocolObject<dyn MTLBuffer>>,
            new: &Retained<ProtocolObject<dyn MTLBuffer>>,
            position: usize,
        ) -> Result<(), backenderror> {
            // One row is the whole `new` buffer: the cached-attention path
            // appends exactly the `[1, width]` key/value the kernel produced.
            let bytes = new.length();
            if bytes == 0 {
                return Err(backenderror::invalidparameter { op: "append" });
            }
            let offset = position
                .checked_mul(bytes)
                .ok_or(backenderror::sizeoverflow)?;
            session
                .0
                .recordappend(cached, offset, new, bytes)
                .map_err(maptobackend)
        }

        fn gethostdata(&self, session: &Self::session, slot: usize) -> Option<Vec<f32>> {
            session.0.gethostdata(slot)
        }

        fn applysession(&self, session: &mut Self::session) -> Result<Vec<Vec<f32>>, backenderror> {
            session.0.apply().map_err(maptobackend)
        }

        fn recordmatmul(
            &self,
            session: &mut Self::session,
            lhs: &Self::buffer,
            rhs: &Self::buffer,
            geometry: &matmulshape,
        ) -> Result<usize, backenderror> {
            ffi::matmul_into(&self.cache, &mut session.0, lhs, rhs, geometry).map_err(maptobackend)
        }

        fn recordadd(
            &self,
            session: &mut Self::session,
            lhs: &Self::buffer,
            rhs: &Self::buffer,
            count: usize,
        ) -> Result<usize, backenderror> {
            ffi::add_into(&self.cache, &mut session.0, lhs, rhs, count).map_err(maptobackend)
        }

        fn recordmul(
            &self,
            session: &mut Self::session,
            lhs: &Self::buffer,
            rhs: &Self::buffer,
            count: usize,
        ) -> Result<usize, backenderror> {
            ffi::mul_into(&self.cache, &mut session.0, lhs, rhs, count).map_err(maptobackend)
        }

        fn recordrmsnorm(
            &self,
            session: &mut Self::session,
            input: &Self::buffer,
            weight: &Self::buffer,
            geometry: &axisshape,
            eps: f32,
        ) -> Result<usize, backenderror> {
            ffi::rmsnorm_into(
                &self.cache,
                &mut session.0,
                input,
                weight,
                geometry.rows,
                geometry.width,
                eps,
            )
            .map_err(maptobackend)
        }

        fn recordsoftmax(
            &self,
            session: &mut Self::session,
            input: &Self::buffer,
            geometry: &axisshape,
        ) -> Result<usize, backenderror> {
            ffi::softmax_into(
                &self.cache,
                &mut session.0,
                input,
                geometry.rows,
                geometry.width,
            )
            .map_err(maptobackend)
        }

        fn recordsilu(
            &self,
            session: &mut Self::session,
            input: &Self::buffer,
            count: usize,
        ) -> Result<usize, backenderror> {
            ffi::silu_into(&self.cache, &mut session.0, input, count).map_err(maptobackend)
        }

        fn recordembedding(
            &self,
            session: &mut Self::session,
            table: &Self::buffer,
            indices: &[usize],
            geometry: &embeddingshape,
        ) -> Result<usize, backenderror> {
            ffi::embedding_into(&self.cache, &mut session.0, table, indices, geometry.width)
                .map_err(maptobackend)
        }

        fn recordrope(
            &self,
            session: &mut Self::session,
            input: &Self::buffer,
            geometry: &ropeshape,
            base: f32,
        ) -> Result<usize, backenderror> {
            ffi::rope_into(&self.cache, &mut session.0, input, geometry, base).map_err(maptobackend)
        }

        fn recordrope_at(
            &self,
            session: &mut Self::session,
            input: &Self::buffer,
            position: usize,
            geometry: &ropeshape,
            base: f32,
        ) -> Result<usize, backenderror> {
            ffi::rope_at_into(
                &self.cache,
                &mut session.0,
                input,
                position,
                geometry.heads,
                geometry.headdim,
                base,
            )
            .map_err(maptobackend)
        }

        fn recordattention(
            &self,
            session: &mut Self::session,
            query: &Self::buffer,
            key: &Self::buffer,
            value: &Self::buffer,
            geometry: &attentionshape,
            causal: bool,
        ) -> Result<usize, backenderror> {
            ffi::attention_into(
                &self.cache,
                &mut session.0,
                query,
                key,
                value,
                geometry,
                causal,
            )
            .map_err(maptobackend)
        }

        fn recordattention_cached(
            &self,
            session: &mut Self::session,
            query: &Self::buffer,
            keycached: &Self::buffer,
            keynew: &Self::buffer,
            valuecached: &Self::buffer,
            valuenew: &Self::buffer,
            geometry: &attentionshape,
            causal: bool,
        ) -> Result<usize, backenderror> {
            ffi::attention_cached_into(
                &self.cache,
                &mut session.0,
                query,
                ffi::bufferpair {
                    cached: keycached,
                    new: keynew,
                },
                ffi::bufferpair {
                    cached: valuecached,
                    new: valuenew,
                },
                geometry,
                causal,
            )
            .map_err(maptobackend)
        }
    }
}
#[cfg(all(target_os = "linux", musa_toolkit))]
pub mod musa {
    //! Moore Threads MUSA GPU execution backend.
    //!
    //! `musa` is a Layer 1 backend: it implements the frozen [`api::backend`] and
    //! [`api::compute`] contracts for `f32` operands on a Moore Threads MUSA device.
    //! It owns no request state and holds no model weights.
    //!
    //! # Platform
    //!
    //! The module is compiled only on Linux with the MUSA toolkit present (the
    //! `musa_toolkit` cfg emitted by the crate's build script). On every other
    //! host the crate exposes no `musa` namespace and
    //! `cargo check --workspace --all-targets` stays green.
    //!
    //! # Architecture (docs/musa-backend-plan.md §5)
    //!
    //! The MUSA backend converges on the workspace's unified GPU-backend terminal
    //! state (docs/gpu-backend-unification.md): **build-time compilation, zero
    //! runtime JIT, and the minimal unsafe surface**. Kernels are compiled at build
    //! time by `mcc` into ELF fatbins embedded with `include_bytes!` (see
    //! [`registry`]) and loaded at run time through the `mu*` driver API — no
    //! runtime compiler, no `muBLAS`/`muDNN` (matmul is served by an in-repo
    //! Tensor-Core (TF32) kernel with the scalar FMA kernel kept as the reference
    //! path, softmax by the max-subtracting kernel shared with `cpu`/`metal`),
    //! so the whole platform FFI collapses to one driver-handoff function. The
    //! registered unsafe site count is `musa = 2`: the `unsafe extern "C"` block
    //! (mandatory under edition 2024) and the driver-handoff `unsafe` block
    //! (docs/design.md §21.8). The handoff block is lazy by design: device open,
    //! module load, function resolution, transfers, launch, and synchronisation
    //! all live in the single `dispatch` block, so no second handoff site exists.
    //!
    //! # Unsafe
    //!
    //! This backend is one of the carriers of the workspace's restricted `unsafe`
    //! exemption (docs/design.md §21.8): the crate root uses
    //! `#![deny(unsafe_code)]`; `ffi` declares this backend's one allowance and
    //! confines every platform call there, each preceded by a `// SAFETY:` comment.
    //! `scripts/checkunsafe.sh` enforces the exact site count and the SAFETY
    //! comments mechanically.
    //!
    //! # Naming
    //!
    //! Per the repository no-separator convention, all public type, trait, enum,
    //! variant, and function names are lowercase concatenated, and constants are
    //! uppercase concatenated.

    use crate::ffi::musa as ffi;

    /// The MUSA view of the shared kernel catalog.
    pub(crate) mod source {
        pub(crate) use crate::catalog::cu::{
            ADDENTRY, ATTENTIONCACHEDENTRY, ATTENTIONENTRY, ATTENTIONLANES, FLATTILE, MULENTRY,
            RMSNORMENTRY, RMSNORMTILE, ROPEATENTRY, ROPEENTRY, SILUENTRY,
        };
        pub(crate) use crate::catalog::cu::{
            AGGENTRY as SCANAGGENTRY, APPLYENTRY as SCANAPPLYENTRY, CHUNK as SCANCHUNK,
            PREFIXENTRY as SCANPREFIXENTRY, SCANBLOCKDIM,
        };
        pub(crate) use crate::catalog::cu::{EMBEDDINGENTRY, SCANENTRY, SCANMAXN, SOFTMAXENTRY};
        pub(crate) use crate::catalog::cu::{MATMULENTRY, THREADEDGE};
        pub(crate) use crate::catalog::musa::{
            MATMULRMSENTRY, RMSINVENTRY, WMMABM as MATMULWMMABM, WMMABN as MATMULWMMABN,
            WMMAENTRY as MATMULWMMAENTRY, WMMASOURCEID as MATMULWMMASOURCEID,
            WMMATHREADS as MATMULWMMATHREADS,
        };
    }

    /// The maximum tensor rank the MUSA backend accepts.
    pub const MUSAMAXRANK: usize = crate::engine::MAXRANK;

    /// The maximum head dimension the attention kernel accepts.
    ///
    /// The value is a device limit rather than a contract limit: the attention
    /// kernel keeps one accumulator slot per head dimension in shared memory, so
    /// its request is `(2 * lanes + headdim) * 4` bytes. The kernel is ported from
    /// the CUDA backend unchanged (docs/musa-backend-plan.md §5.3), so the shared-
    /// memory budget and this bound are identical to the CUDA derivation
    /// (docs/design.md §21.1, GOV7).
    pub const MUSAMAXHEADDIM: usize = 8192;

    /// The MUSA device identity.
    pub struct musadevice;

    impl crate::engine::devicespec for musadevice {
        const NAME: &'static str = "musa";
        const MAXHEADDIM: usize = MUSAMAXHEADDIM;

        fn device() -> api::device {
            api::device::musa()
        }
    }

    /// The MUSA execution backend.
    pub type musabackend = crate::engine::gpubackend<musadevice>;

    use std::sync::{Arc, Mutex};

    use ::tensor::{shape, tensor, tensorview};
    use api::{
        backend, backenderror, capabilities, checkbudget, checkgatherbudget, kvpair, operand,
        operandpair, recorder, session, stepref, submission,
    };
    use lib::limits;

    use api::geometry::{
        ADDOP, MULOP, attentionshape, axisshape, binaryshape, embeddingshape, matmulshape,
        ropeshape,
    };

    use crate::engine::gpuerror;
    use crate::engine::{concatshape, embeddingoutshape, scanshape, wrap};
    use crate::engine::{executor, platform, rmsnormmatmulshape};
    use crate::ffi::musa::buf;

    /// Which MUSA implementation serves `matmul`.
    ///
    /// The choice is made once, at construction, and stored as a field, exactly as
    /// `cuda` stores its `matmulpath`: selecting it per call would add a run-time
    /// branch to the hottest path and obscure which path a measurement exercised.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum matmulpath {
        /// The in-repo Tensor-Core kernel (TF32), used by [`musaexecutor::new`].
        /// TF32 rounds each operand to a 10-bit mantissa, so its result differs
        /// from the FMA reference within the documented tolerance.
        wmma,
        /// The in-repo scalar FMA kernel, used by [`musaexecutor::newfma`] as the
        /// independent high-precision reference path.
        fma,
    }

    /// The MUSA device primitives behind [`musaexecutor`].
    #[derive(Clone, Debug)]
    pub struct musaplatform {
        backend: musabackend,
        path: matmulpath,
        cache: Arc<ffi::cache>,
    }

    impl musaplatform {
        fn new(lim: limits, path: matmulpath) -> Self {
            Self {
                backend: musabackend::new(lim),
                path,
                cache: Arc::new(ffi::cache::new()),
            }
        }
    }

    impl platform for musaplatform {
        const NAME: &'static str = "musa";
        const MAXHEADDIM: usize = MUSAMAXHEADDIM;
        const RMSNORMMATMUL: bool = true;
        const SCAN_CACHED: bool = true;

        fn capabilities(&self) -> &capabilities {
            self.backend.getcapabilities()
        }

        fn matmul(
            &self,
            lhs: &[f32],
            rhs: &[f32],
            geometry: &matmulshape,
        ) -> Result<Vec<f32>, backenderror> {
            match self.path {
                matmulpath::wmma => ffi::matmul_wmma(
                    &self.cache,
                    lhs,
                    rhs,
                    geometry.m,
                    geometry.k,
                    geometry.n,
                    geometry.batch,
                ),
                matmulpath::fma => ffi::matmul_fma(
                    &self.cache,
                    lhs,
                    rhs,
                    geometry.m,
                    geometry.k,
                    geometry.n,
                    geometry.batch,
                ),
            }
            .map_err(maptobackend)
        }

        fn add(&self, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, backenderror> {
            ffi::add(&self.cache, lhs, rhs).map_err(maptobackend)
        }

        fn mul(&self, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, backenderror> {
            ffi::mul(&self.cache, lhs, rhs).map_err(maptobackend)
        }

        fn rmsnorm(
            &self,
            input: &[f32],
            weight: &[f32],
            geometry: &axisshape,
            eps: f32,
        ) -> Result<Vec<f32>, backenderror> {
            ffi::rmsnorm(
                &self.cache,
                input,
                weight,
                geometry.rows,
                geometry.width,
                eps,
            )
            .map_err(maptobackend)
        }

        fn rmsnormmatmul(
            &self,
            input: &[f32],
            weight: &[f32],
            rhs: &[f32],
            geometry: &matmulshape,
            eps: f32,
        ) -> Result<Vec<f32>, backenderror> {
            ffi::rmsnormmatmul(
                &self.cache,
                input,
                weight,
                rhs,
                geometry.m,
                geometry.k,
                geometry.n,
                geometry.batch,
                eps,
            )
            .map_err(maptobackend)
        }

        fn softmax(&self, input: &[f32], geometry: &axisshape) -> Result<Vec<f32>, backenderror> {
            ffi::softmax(&self.cache, input, geometry.rows, geometry.width).map_err(maptobackend)
        }

        fn silu(&self, input: &[f32]) -> Result<Vec<f32>, backenderror> {
            ffi::silu(&self.cache, input).map_err(maptobackend)
        }

        fn embedding(
            &self,
            table: &[f32],
            indices: &[usize],
            geometry: &embeddingshape,
        ) -> Result<Vec<f32>, backenderror> {
            let index64 = tou64(indices);
            ffi::embedding(&self.cache, table, &index64, geometry.width).map_err(maptobackend)
        }

        fn rope(
            &self,
            input: &[f32],
            geometry: &ropeshape,
            base: f32,
        ) -> Result<Vec<f32>, backenderror> {
            ffi::rope(
                &self.cache,
                input,
                geometry.rows,
                geometry.seqaxis,
                geometry.heads,
                geometry.headdim,
                base,
            )
            .map_err(maptobackend)
        }

        fn rope_at(
            &self,
            input: &[f32],
            position: usize,
            geometry: &ropeshape,
            base: f32,
        ) -> Result<Vec<f32>, backenderror> {
            // The dedicated absolute-position kernel rotates the single token
            // directly, without materialising a padded history (contrast the
            // `cuda` platform's host-side pad-and-rotate).
            ffi::rope_at(
                &self.cache,
                input,
                position,
                geometry.heads,
                geometry.headdim,
                base,
            )
            .map_err(maptobackend)
        }

        fn attention(
            &self,
            query: &[f32],
            key: &[f32],
            value: &[f32],
            geometry: &attentionshape,
            causal: bool,
        ) -> Result<Vec<f32>, backenderror> {
            ffi::attention(
                &self.cache,
                query,
                key,
                value,
                geometry.qseq,
                geometry.kvseq,
                geometry.heads,
                geometry.kvheads,
                geometry.headdim,
                causal,
            )
            .map_err(maptobackend)
        }

        fn attention_cached(
            &self,
            query: &[f32],
            key: &kvpair<'_>,
            value: &kvpair<'_>,
            geometry: &attentionshape,
            causal: bool,
        ) -> Result<Vec<f32>, backenderror> {
            // Native kernel: the cached rows followed by the new row are read from
            // their own buffers (contrast the `cuda`/`metal` host-side concat).
            ffi::attention_cached(
                &self.cache,
                query,
                key.cached.getdata(),
                key.new.getdata(),
                value.cached.getdata(),
                value.new.getdata(),
                geometry.kvseq,
                geometry.heads,
                geometry.kvheads,
                geometry.headdim,
                causal,
            )
            .map_err(maptobackend)
        }

        fn scan(
            &self,
            input: &[f32],
            init: &[f32],
            a: &[f32],
            b: &[f32],
            c: &[f32],
            skip: &[f32],
            n: usize,
            d: usize,
            statewidth: usize,
        ) -> Result<(Vec<f32>, Vec<f32>), backenderror> {
            ffi::scan(&self.cache, input, init, a, b, c, skip, n, d, statewidth)
                .map_err(maptobackend)
        }

        fn scan_cached(
            &self,
            input: &[f32],
            state: &[f32],
            a: &[f32],
            b: &[f32],
            c: &[f32],
            skip: &[f32],
            d: usize,
            statewidth: usize,
        ) -> Result<(Vec<f32>, Vec<f32>), backenderror> {
            // The decode step runs the serial kernel with n = 1, the bitwise-equal
            // continuation of the serial scan over the same prefix.
            ffi::scan_cached(&self.cache, input, state, a, b, c, skip, d, statewidth)
                .map_err(maptobackend)
        }
    }

    /// A MUSA executor: the crate's [`compute`](api::compute) provider.
    ///
    /// The orchestration is shared with the other backends ([`crate::engine`]);
    /// `musaexecutor` is that engine instantiated over [`musaplatform`].
    pub type musaexecutor = executor<musaplatform>;

    impl executor<musaplatform> {
        /// Creates an executor over the MUSA backend with budget `lim`.
        ///
        /// `matmul` is served by the Tensor-Core (TF32) kernel. Nothing touches the
        /// device until the first operator call — `new` never fails on a host
        /// without a MUSA device.
        #[must_use]
        pub fn new(lim: limits) -> Self {
            Self::newplatform(musaplatform::new(lim, matmulpath::wmma))
        }

        /// Creates an executor whose `matmul` is served by the scalar FMA kernel.
        ///
        /// This is the independent high-precision path the Tensor-Core path is
        /// validated against.
        #[must_use]
        pub fn newfma(lim: limits) -> Self {
            Self::newplatform(musaplatform::new(lim, matmulpath::fma))
        }

        /// Returns which implementation serves `matmul`.
        #[must_use]
        pub fn getpath(&self) -> matmulpath {
            self.getplatform().path
        }

        /// Returns how many kernels this executor has resolved so far.
        #[must_use]
        pub fn getcachedkernelcount(&self) -> usize {
            self.getplatform().cache.kernelcount()
        }

        /// Returns the backend whose budget this executor enforces.
        #[must_use]
        pub fn getbackend(&self) -> &musabackend {
            &self.getplatform().backend
        }

        /// Creates a single-use accumulative submission over this executor's
        /// backend and compiled-kernel cache.
        #[must_use]
        pub fn newsubmission(&self) -> musasubmission {
            let platform = self.getplatform();
            musasubmission::new(
                platform.backend.clone(),
                platform.path,
                platform.cache.clone(),
            )
        }

        /// Creates a request-scoped persistent session over this executor's backend
        /// and compiled-kernel cache.
        ///
        /// The session pins model weights once and reuses them across every
        /// [`apply`](session::apply), so a decode loop does not re-upload the model
        /// on each token. Call [`session::reset`](session::reset) (or drop the
        /// session) to release the pinned allocations.
        #[must_use]
        pub fn newpersistent(&self) -> musasession {
            let platform = self.getplatform();
            musasession::new(
                platform.backend.clone(),
                platform.path,
                platform.cache.clone(),
            )
        }
    }

    /// Maps a crate-local [`gpuerror`] onto the frozen `api::backenderror`.
    ///
    /// Device absence and module loading are surfaced as their own variants so a
    /// caller can tell "no MUSA device on this host" from "the fatbin could not be
    /// loaded"; the kernel entry point rides along so a loading failure identifies
    /// exactly which `(source, entry)` pair failed. Transfer and launch runtime
    /// failures stay under the generic `executionfailed`. The compilation class is
    /// deliberately absent: kernels are compiled at build time by `mcc`, so a
    /// compilation failure is a build failure, never a run-time error
    /// (docs/musa-backend-plan.md §5.3).
    fn maptobackend(error: gpuerror) -> backenderror {
        error.maptobackend("musa")
    }

    /// Reports a step reference that a previous persistent apply released.
    fn stalestep() -> backenderror {
        backenderror::executionfailed {
            backend: "musa",
            kernel: "record".into(),
            message: "step was released by the previous apply".into(),
        }
    }

    /// Extracts a device pointer from an operand binding.
    fn bufferptr(value: buf<'_>) -> Result<ffi::mudevptr, backenderror> {
        if let buf::device(ptr) = value {
            Ok(ptr)
        } else {
            Err(backenderror::executionfailed {
                backend: "musa",
                kernel: "append".into(),
                message: "operand is not device-resident".into(),
            })
        }
    }

    /// Converts the API's `usize` row indices to the kernel's `u64` gather indices.
    fn tou64(indices: &[usize]) -> Vec<u64> {
        indices.iter().map(|index| *index as u64).collect()
    }

    /// Device-resident pinned inputs keyed by `(data pointer, length)`: each
    /// uniquely identified input is uploaded once and shared across a session.
    type pinnedmap = Arc<Mutex<std::collections::HashMap<(usize, usize), (ffi::mudevptr, usize)>>>;

    /// A MUSA submission: the [`submission`] contract implemented over the same
    /// executor-owned cache.
    ///
    /// `musasubmission` owns the recorded results and no model weights; it shares
    /// no mutable state between requests. Every supported step records a kernel
    /// that reads existing device slots and leaves its result in a new slot; only
    /// degenerate empty outputs stay on the host. [`apply`](submission::apply)
    /// rebuilds the requested outputs with one final synchronise and download.
    ///
    /// A batched `matmul` (a leading batch dimension greater than one) is rejected,
    /// mirroring `cuda`'s submission contract: the recorded path is one contraction
    /// block, and joining blocks back into one tensor would re-introduce host round
    /// trips, so it is left to the single-operator [`compute`] path.
    pub struct musasubmission {
        backend: musabackend,
        path: matmulpath,
        session: ffi::session,
        results: Vec<(resultdata, shape)>,
        /// Uploaded caller inputs keyed by `(data pointer, length)`, so an input
        /// operand referenced by several steps is uploaded exactly once per
        /// submission.
        inputs: std::collections::HashMap<(usize, usize), usize>,
        /// Inputs pinned by a persistent session, shared across submissions so the
        /// model weights stop being re-uploaded on every decode step.
        pinned: pinnedmap,
    }

    /// A recorded result: a device slot, host data for a degenerate empty output,
    /// or a step released by the previous persistent apply.
    enum resultdata {
        slot(usize),
        host(Vec<f32>),
        dead,
    }

    impl musasubmission {
        /// Creates a submission over `backend` using `cache`, selecting `path` for
        /// recorded `matmul` steps.
        pub(crate) fn new(backend: musabackend, path: matmulpath, cache: Arc<ffi::cache>) -> Self {
            Self::newpinned(
                backend,
                path,
                cache,
                Arc::new(Mutex::new(std::collections::HashMap::new())),
            )
        }

        /// Creates a submission that shares `pinned` with a persistent session.
        pub(crate) fn newpinned(
            backend: musabackend,
            path: matmulpath,
            cache: Arc<ffi::cache>,
            pinned: pinnedmap,
        ) -> Self {
            let session = ffi::session::new(cache);
            Self {
                backend,
                path,
                session,
                results: Vec::new(),
                inputs: std::collections::HashMap::new(),
                pinned,
            }
        }

        fn checkbudget(&self, s: &shape) -> Result<(), backenderror> {
            checkbudget(
                self.backend.getcapabilities(),
                s,
                core::mem::size_of::<f32>(),
            )
        }

        fn checkgatherbudget(&self, rows: usize, width: usize) -> Result<(), backenderror> {
            checkgatherbudget(
                self.backend.getcapabilities(),
                rows,
                width,
                core::mem::size_of::<f32>(),
            )
        }

        /// Resolves an operand to its output shape, without touching the device.
        fn shapeof(&self, op: operand<'_>) -> Result<shape, backenderror> {
            match op {
                operand::input(view) => Ok(view.getshape().clone()),
                operand::result(reference) => {
                    let index = reference.getindex();
                    match self.results.get(index) {
                        Some((resultdata::dead, _)) => Err(stalestep()),
                        Some((_, outshape)) => Ok(outshape.clone()),
                        None => Err(backenderror::indexoutofrange {
                            index,
                            len: self.results.len(),
                        }),
                    }
                }
            }
        }

        /// Resolves an operand to a device slot for the device-resident path, or
        /// returns `None` when the operand is host data (an operator not yet
        /// converted): caller inputs are uploaded once per identity, recorded
        /// results already live in slots.
        fn resolveslot(&mut self, op: operand<'_>) -> Result<Option<usize>, backenderror> {
            match op {
                operand::input(view) => {
                    let data = view.getdata();
                    let key = (data.as_ptr() as usize, data.len());
                    if let Some(&slot) = self.inputs.get(&key) {
                        return Ok(Some(slot));
                    }
                    let slot = self.session.upload(data).map_err(maptobackend)?;
                    self.inputs.insert(key, slot);
                    Ok(Some(slot))
                }
                operand::result(reference) => {
                    let index = reference.getindex();
                    match self.results.get(index) {
                        Some((resultdata::slot(slot), _)) => Ok(Some(*slot)),
                        Some((resultdata::host(_), _)) => Ok(None),
                        Some((resultdata::dead, _)) => Err(stalestep()),
                        None => Err(backenderror::indexoutofrange {
                            index,
                            len: self.results.len(),
                        }),
                    }
                }
            }
        }

        /// Resolves an operand to host data for the upload fallback (a degenerate
        /// empty result); a device-resident result is drained with one
        /// synchronise, which reintroduces a sync only on that fallback path.
        fn resolve(&mut self, op: operand<'_>) -> Result<Vec<f32>, backenderror> {
            match op {
                operand::input(view) => Ok(view.getdata().to_vec()),
                operand::result(reference) => {
                    let index = reference.getindex();
                    match self.results.get(index) {
                        Some((resultdata::host(data), _)) => Ok(data.clone()),
                        Some((resultdata::slot(slot), _)) => {
                            let slot = *slot;
                            self.session.take(slot).map_err(maptobackend)
                        }
                        Some((resultdata::dead, _)) => Err(stalestep()),
                        None => Err(backenderror::indexoutofrange {
                            index,
                            len: self.results.len(),
                        }),
                    }
                }
            }
        }

        /// Binds one operand for a device-resident record: a recorded slot binds
        /// directly, host data is uploaded once into a new slot, and a fresh
        /// caller input is uploaded once per identity.
        /// The returned bind never borrows host data: every operand is resolved
        /// to a device slot (uploaded on first use), so the `'static` lifetime
        /// parameter is a formality — `buf::device` carries a pointer only.
        fn operandsbuf(&mut self, op: operand<'_>) -> Result<buf<'static>, backenderror> {
            if let operand::input(view) = op {
                let data = view.getdata();
                let key = (data.as_ptr() as usize, data.len());
                let pinned = self
                    .pinned
                    .lock()
                    .map_err(|_| backenderror::executionfailed {
                        backend: "musa",
                        kernel: "pin".into(),
                        message: "pinned input cache poisoned".into(),
                    })?;
                if let Some(&(ptr, _)) = pinned.get(&key) {
                    return Ok(buf::device(ptr));
                }
            }
            if let Some(slot) = self.resolveslot(op)? {
                return Ok(buf::device(self.session.slotptr(slot)));
            }
            let data = self.resolve(op)?;
            let slot = self.session.upload(&data).map_err(maptobackend)?;
            Ok(buf::device(self.session.slotptr(slot)))
        }

        fn pushslot(&mut self, slot: usize, outshape: shape) -> stepref {
            let index = self.results.len();
            self.results.push((resultdata::slot(slot), outshape));
            stepref::fromindex(index)
        }

        fn push(&mut self, data: Vec<f32>, outshape: shape) -> stepref {
            let index = self.results.len();
            self.results.push((resultdata::host(data), outshape));
            stepref::fromindex(index)
        }
    }

    /// A request-scoped persistent MUSA session.
    ///
    /// Unlike [`musasubmission`], a session keeps its pinned inputs (the model
    /// weights) in device allocations that outlive each [`apply`](session::apply).
    /// Every apply consumes the current step buffer and starts a fresh one over the
    /// same pinned allocations, so a decode loop uploads the weights once and the
    /// transient activations and key/value rows are released after each step.
    pub struct musasession {
        backend: musabackend,
        path: matmulpath,
        cache: Arc<ffi::cache>,
        pinned: pinnedmap,
        current: musasubmission,
    }

    impl musasession {
        /// Creates an empty session over `backend`, selecting `path` for matmul.
        pub(crate) fn new(backend: musabackend, path: matmulpath, cache: Arc<ffi::cache>) -> Self {
            let pinned = Arc::new(Mutex::new(std::collections::HashMap::new()));
            let current =
                musasubmission::newpinned(backend.clone(), path, cache.clone(), pinned.clone());
            Self {
                backend,
                path,
                cache,
                pinned,
                current,
            }
        }

        /// Allocates a zeroed device buffer that survives every apply and returns a
        /// step handle to it.
        ///
        /// A device KV cache uses this for its per-layer key/value buffers, then
        /// exposes the first `len` rows through [`recorder::recordview`].
        ///
        /// # Errors
        ///
        /// Returns a backend error when the allocation fails or the shape overflows.
        pub fn zeroed(&mut self, shape: shape) -> Result<stepref, backenderror> {
            let count = shape
                .getelementcount()
                .map_err(|_| backenderror::sizeoverflow)?;
            let slot = self
                .current
                .session
                .alloczeros(count)
                .map_err(maptobackend)?;
            Ok(self.current.pushslot(slot, shape))
        }

        /// Releases every pinned allocation through the deferred free queue.
        fn retirepinned(&mut self) {
            let pinned = match self.pinned.lock() {
                Ok(mut map) => std::mem::take(&mut *map),
                Err(poisoned) => std::mem::take(&mut *poisoned.into_inner()),
            };
            for (_, (ptr, bytes)) in pinned {
                ffi::retiredevice(&self.cache, ptr, bytes);
            }
        }
    }

    impl Drop for musasession {
        fn drop(&mut self) {
            self.retirepinned();
        }
    }

    impl recorder for musasession {
        fn recordmatmul(
            &mut self,
            lhs: operand<'_>,
            rhs: operand<'_>,
        ) -> Result<stepref, backenderror> {
            submission::recordmatmul(&mut self.current, lhs, rhs)
        }

        fn recordadd(
            &mut self,
            lhs: operand<'_>,
            rhs: operand<'_>,
        ) -> Result<stepref, backenderror> {
            submission::recordadd(&mut self.current, lhs, rhs)
        }

        fn recordmul(
            &mut self,
            lhs: operand<'_>,
            rhs: operand<'_>,
        ) -> Result<stepref, backenderror> {
            submission::recordmul(&mut self.current, lhs, rhs)
        }

        fn recordrmsnorm(
            &mut self,
            input: operand<'_>,
            weight: operand<'_>,
            eps: f32,
        ) -> Result<stepref, backenderror> {
            submission::recordrmsnorm(&mut self.current, input, weight, eps)
        }

        fn recordrmsnormmatmul(
            &mut self,
            input: operand<'_>,
            weight: operand<'_>,
            rhs: operand<'_>,
            eps: f32,
        ) -> Result<stepref, backenderror> {
            submission::recordrmsnormmatmul(&mut self.current, input, weight, rhs, eps)
        }

        fn recordsoftmax(&mut self, input: operand<'_>) -> Result<stepref, backenderror> {
            submission::recordsoftmax(&mut self.current, input)
        }

        fn recordsilu(&mut self, input: operand<'_>) -> Result<stepref, backenderror> {
            submission::recordsilu(&mut self.current, input)
        }

        fn recordembedding(
            &mut self,
            table: operand<'_>,
            indices: &[usize],
        ) -> Result<stepref, backenderror> {
            submission::recordembedding(&mut self.current, table, indices)
        }

        fn recordrope(
            &mut self,
            input: operand<'_>,
            heads: usize,
            base: f32,
        ) -> Result<stepref, backenderror> {
            submission::recordrope(&mut self.current, input, heads, base)
        }

        fn recordrope_at(
            &mut self,
            input: operand<'_>,
            position: usize,
            heads: usize,
            base: f32,
        ) -> Result<stepref, backenderror> {
            submission::recordrope_at(&mut self.current, input, position, heads, base)
        }

        fn recordattention(
            &mut self,
            query: operand<'_>,
            key: operand<'_>,
            value: operand<'_>,
            heads: usize,
            kvheads: usize,
            causal: bool,
        ) -> Result<stepref, backenderror> {
            submission::recordattention(
                &mut self.current,
                query,
                key,
                value,
                heads,
                kvheads,
                causal,
            )
        }

        fn recordattention_cached(
            &mut self,
            query: operand<'_>,
            key: operandpair<'_>,
            value: operandpair<'_>,
            heads: usize,
            kvheads: usize,
            causal: bool,
        ) -> Result<stepref, backenderror> {
            submission::recordattention_cached(
                &mut self.current,
                query,
                key,
                value,
                heads,
                kvheads,
                causal,
            )
        }

        fn recordscan(
            &mut self,
            input: operand<'_>,
            init: operand<'_>,
            a: operand<'_>,
            b: operand<'_>,
            c: operand<'_>,
            skip: operand<'_>,
        ) -> Result<stepref, backenderror> {
            self.recordscan_pair(input, init, a, b, c, skip)
                .map(|(out, _)| out)
        }

        fn recordscan_cached(
            &mut self,
            input: operand<'_>,
            state: operand<'_>,
            a: operand<'_>,
            b: operand<'_>,
            c: operand<'_>,
            skip: operand<'_>,
        ) -> Result<stepref, backenderror> {
            self.recordscan_cached_pair(input, state, a, b, c, skip)
                .map(|(out, _)| out)
        }

        fn recordscan_pair(
            &mut self,
            input: operand<'_>,
            init: operand<'_>,
            a: operand<'_>,
            b: operand<'_>,
            c: operand<'_>,
            skip: operand<'_>,
        ) -> Result<(stepref, stepref), backenderror> {
            let inshape = self.current.shapeof(input)?;
            let initshape = self.current.shapeof(init)?;
            let ashape = self.current.shapeof(a)?;
            let bshape = self.current.shapeof(b)?;
            let cshape = self.current.shapeof(c)?;
            let skipshape = self.current.shapeof(skip)?;
            for shape in [&inshape, &initshape, &ashape, &bshape, &cshape, &skipshape] {
                self.current.checkbudget(shape)?;
            }
            let (n, d, statewidth) =
                scanshape(&inshape, &initshape, &ashape, &bshape, &cshape, &skipshape)?;
            // The spectral check needs host values. A recorded result is already on
            // the device, so it cannot be inspected without a download; the
            // submission contract validates caller inputs and leaves device
            // intermediates to the backend (docs/design.md §21.13).
            if let operand::input(view) = a {
                for &value in view.getdata() {
                    if value.abs() > 1.0 || value.is_nan() {
                        return Err(backenderror::invalidparameter { op: "scan" });
                    }
                }
            }
            let inputbuf = self.current.operandsbuf(input)?;
            let initbuf = self.current.operandsbuf(init)?;
            let abuf = self.current.operandsbuf(a)?;
            let bbuf = self.current.operandsbuf(b)?;
            let cbuf = self.current.operandsbuf(c)?;
            let skipbuf = self.current.operandsbuf(skip)?;
            let (outslot, stateslot) = ffi::scan_record(
                &mut self.current.session,
                inputbuf,
                initbuf,
                abuf,
                bbuf,
                cbuf,
                skipbuf,
                n,
                d,
                statewidth,
            )
            .map_err(maptobackend)?;
            let outshape = shape::new(vec![n, d]).map_err(|_| backenderror::sizeoverflow)?;
            let stateshape =
                shape::new(vec![d, statewidth]).map_err(|_| backenderror::sizeoverflow)?;
            Ok((
                self.current.pushslot(outslot, outshape),
                self.current.pushslot(stateslot, stateshape),
            ))
        }

        fn recordscan_cached_pair(
            &mut self,
            input: operand<'_>,
            state: operand<'_>,
            a: operand<'_>,
            b: operand<'_>,
            c: operand<'_>,
            skip: operand<'_>,
        ) -> Result<(stepref, stepref), backenderror> {
            let inshape = self.current.shapeof(input)?;
            let stateshapein = self.current.shapeof(state)?;
            let ashape = self.current.shapeof(a)?;
            let bshape = self.current.shapeof(b)?;
            let cshape = self.current.shapeof(c)?;
            let skipshape = self.current.shapeof(skip)?;
            for shape in [
                &inshape,
                &stateshapein,
                &ashape,
                &bshape,
                &cshape,
                &skipshape,
            ] {
                self.current.checkbudget(shape)?;
            }
            let (n, d, statewidth) = scanshape(
                &inshape,
                &stateshapein,
                &ashape,
                &bshape,
                &cshape,
                &skipshape,
            )?;
            if n != 1 {
                return Err(backenderror::shapemismatch { op: "scan_cached" });
            }
            if let operand::input(view) = a {
                for &value in view.getdata() {
                    if value.abs() > 1.0 || value.is_nan() {
                        return Err(backenderror::invalidparameter { op: "scan_cached" });
                    }
                }
            }
            let inputbuf = self.current.operandsbuf(input)?;
            let statebuf = self.current.operandsbuf(state)?;
            let abuf = self.current.operandsbuf(a)?;
            let bbuf = self.current.operandsbuf(b)?;
            let cbuf = self.current.operandsbuf(c)?;
            let skipbuf = self.current.operandsbuf(skip)?;
            let (outslot, newstate) = ffi::scan_cached_record(
                &mut self.current.session,
                inputbuf,
                statebuf,
                abuf,
                bbuf,
                cbuf,
                skipbuf,
                d,
                statewidth,
            )
            .map_err(maptobackend)?;
            let outshape = shape::new(vec![1, d]).map_err(|_| backenderror::sizeoverflow)?;
            let stateshape =
                shape::new(vec![d, statewidth]).map_err(|_| backenderror::sizeoverflow)?;
            Ok((
                self.current.pushslot(outslot, outshape),
                self.current.pushslot(newstate, stateshape),
            ))
        }

        fn recordview(&mut self, base: stepref, shape: shape) -> Result<stepref, backenderror> {
            let index = base.getindex();
            match self.current.results.get(index) {
                Some((resultdata::slot(slot), _)) => Ok(self.current.pushslot(*slot, shape)),
                Some((resultdata::dead, _)) => Err(stalestep()),
                Some((resultdata::host(_), _)) | None => Err(backenderror::indexoutofrange {
                    index,
                    len: self.current.results.len(),
                }),
            }
        }

        fn recordappend(
            &mut self,
            cached: operand<'_>,
            new: operand<'_>,
            position: usize,
        ) -> Result<(), backenderror> {
            let cachedbuf = self.current.operandsbuf(cached)?;
            let newbuf = self.current.operandsbuf(new)?;
            let cachedptr = bufferptr(cachedbuf)?;
            let newptr = bufferptr(newbuf)?;
            let newshape = self.current.shapeof(new)?;
            let width = *newshape
                .getdims()
                .last()
                .ok_or(backenderror::shapemismatch { op: "append" })?;
            let bytes = width
                .checked_mul(core::mem::size_of::<f32>())
                .ok_or(backenderror::sizeoverflow)?;
            let offset = position
                .checked_mul(bytes)
                .ok_or(backenderror::sizeoverflow)?;
            let dst = cachedptr
                .checked_add(offset)
                .ok_or(backenderror::sizeoverflow)?;
            ffi::session::copyinto(&mut self.current.session, dst, newptr, bytes)
                .map_err(maptobackend)?;
            Ok(())
        }

        fn pin(&mut self, view: tensorview<'_, f32>) -> Result<(), backenderror> {
            let data = view.getdata();
            let key = (data.as_ptr() as usize, data.len());
            let mut pinned = self
                .pinned
                .lock()
                .map_err(|_| backenderror::executionfailed {
                    backend: "musa",
                    kernel: "pin".into(),
                    message: "pinned input cache poisoned".into(),
                })?;
            if pinned.contains_key(&key) {
                return Ok(());
            }
            let (ptr, bytes) =
                ffi::session::uploadpersistent(&self.cache, data).map_err(maptobackend)?;
            pinned.insert(key, (ptr, bytes));
            Ok(())
        }
    }

    impl session for musasession {
        fn apply(&mut self, outputs: &[stepref]) -> Result<Vec<tensor<f32>>, backenderror> {
            self.current.applyref(outputs)
        }

        fn reset(&mut self) {
            self.retirepinned();
            self.current = musasubmission::newpinned(
                self.backend.clone(),
                self.path,
                self.cache.clone(),
                self.pinned.clone(),
            );
        }
    }

    impl submission for musasubmission {
        fn recordmatmul(
            &mut self,
            lhs: operand<'_>,
            rhs: operand<'_>,
        ) -> Result<stepref, backenderror> {
            let lhsshape = self.shapeof(lhs)?;
            let rhsshape = self.shapeof(rhs)?;
            self.checkbudget(&lhsshape)?;
            self.checkbudget(&rhsshape)?;
            let geometry = matmulshape::new(&lhsshape, &rhsshape)?;
            let outshape = geometry.outshape()?;
            self.checkbudget(&outshape)?;
            if geometry.batch != 1 {
                return Err(backenderror::unsupportedoperation { op: "matmul" });
            }
            if geometry.m == 0 || geometry.n == 0 {
                return Ok(self.push(Vec::new(), outshape));
            }
            let lhs = self.operandsbuf(lhs)?;
            let rhs = self.operandsbuf(rhs)?;
            let slot = match self.path {
                matmulpath::wmma => ffi::matmul_wmma_record(
                    &mut self.session,
                    lhs,
                    rhs,
                    geometry.m,
                    geometry.k,
                    geometry.n,
                ),
                matmulpath::fma => ffi::matmul_fma_record(
                    &mut self.session,
                    lhs,
                    rhs,
                    geometry.m,
                    geometry.k,
                    geometry.n,
                ),
            }
            .map_err(maptobackend)?;
            Ok(self.pushslot(slot, outshape))
        }

        fn recordadd(
            &mut self,
            lhs: operand<'_>,
            rhs: operand<'_>,
        ) -> Result<stepref, backenderror> {
            let lhsshape = self.shapeof(lhs)?;
            let rhsshape = self.shapeof(rhs)?;
            self.checkbudget(&lhsshape)?;
            self.checkbudget(&rhsshape)?;
            let outshape = binaryshape(&lhsshape, &rhsshape, ADDOP)?;
            let count = lhsshape
                .getelementcount()
                .map_err(|_| backenderror::sizeoverflow)?;
            let lbuf = self.operandsbuf(lhs)?;
            let rbuf = self.operandsbuf(rhs)?;
            let slot =
                ffi::add_record(&mut self.session, lbuf, rbuf, count).map_err(maptobackend)?;
            Ok(self.pushslot(slot, outshape))
        }

        fn recordmul(
            &mut self,
            lhs: operand<'_>,
            rhs: operand<'_>,
        ) -> Result<stepref, backenderror> {
            let lhsshape = self.shapeof(lhs)?;
            let rhsshape = self.shapeof(rhs)?;
            self.checkbudget(&lhsshape)?;
            self.checkbudget(&rhsshape)?;
            let outshape = binaryshape(&lhsshape, &rhsshape, MULOP)?;
            let count = lhsshape
                .getelementcount()
                .map_err(|_| backenderror::sizeoverflow)?;
            let lbuf = self.operandsbuf(lhs)?;
            let rbuf = self.operandsbuf(rhs)?;
            let slot =
                ffi::mul_record(&mut self.session, lbuf, rbuf, count).map_err(maptobackend)?;
            Ok(self.pushslot(slot, outshape))
        }

        fn recordrmsnorm(
            &mut self,
            input: operand<'_>,
            weight: operand<'_>,
            eps: f32,
        ) -> Result<stepref, backenderror> {
            let inshape = self.shapeof(input)?;
            let weightshape = self.shapeof(weight)?;
            self.checkbudget(&inshape)?;
            self.checkbudget(&weightshape)?;
            let geometry = axisshape::rmsnorm(&inshape, &weightshape)?;
            let ibuf = self.operandsbuf(input)?;
            let wbuf = self.operandsbuf(weight)?;
            let slot = ffi::rmsnorm_record(
                &mut self.session,
                ibuf,
                wbuf,
                geometry.rows,
                geometry.width,
                eps,
            )
            .map_err(maptobackend)?;
            Ok(self.pushslot(slot, inshape))
        }

        fn recordrmsnormmatmul(
            &mut self,
            input: operand<'_>,
            weight: operand<'_>,
            rhs: operand<'_>,
            eps: f32,
        ) -> Result<stepref, backenderror> {
            let inshape = self.shapeof(input)?;
            let weightshape = self.shapeof(weight)?;
            let rhsshape = self.shapeof(rhs)?;
            self.checkbudget(&inshape)?;
            self.checkbudget(&weightshape)?;
            self.checkbudget(&rhsshape)?;
            let (geometry, outshape) = rmsnormmatmulshape(&inshape, &weightshape, &rhsshape)?;
            self.checkbudget(&outshape)?;
            if geometry.batch == 0 || geometry.m == 0 || geometry.n == 0 {
                return Ok(self.push(Vec::new(), outshape));
            }
            let inputbuf = self.operandsbuf(input)?;
            let weightbuf = self.operandsbuf(weight)?;
            let rhsbuf = self.operandsbuf(rhs)?;
            let rows = geometry
                .batch
                .checked_mul(geometry.m)
                .ok_or(backenderror::sizeoverflow)?;
            let total = rows
                .checked_mul(geometry.n)
                .ok_or(backenderror::sizeoverflow)?;
            let invslot = ffi::rmsinv_record(&mut self.session, inputbuf, rows, geometry.k, eps)
                .map_err(maptobackend)?;
            let slot = ffi::rmsnormmatmul_record(
                &mut self.session,
                inputbuf,
                weightbuf,
                rhsbuf,
                invslot,
                total,
                geometry.m,
                geometry.k,
                geometry.n,
                geometry.batch,
            )
            .map_err(maptobackend)?;
            Ok(self.pushslot(slot, outshape))
        }

        fn recordsoftmax(&mut self, input: operand<'_>) -> Result<stepref, backenderror> {
            let inshape = self.shapeof(input)?;
            self.checkbudget(&inshape)?;
            let geometry = axisshape::softmax(&inshape)?;
            let ibuf = self.operandsbuf(input)?;
            let slot = ffi::softmax_record(&mut self.session, ibuf, geometry.rows, geometry.width)
                .map_err(maptobackend)?;
            Ok(self.pushslot(slot, inshape))
        }

        fn recordsilu(&mut self, input: operand<'_>) -> Result<stepref, backenderror> {
            let inshape = self.shapeof(input)?;
            self.checkbudget(&inshape)?;
            let count = inshape
                .getelementcount()
                .map_err(|_| backenderror::sizeoverflow)?;
            let ibuf = self.operandsbuf(input)?;
            let slot = ffi::silu_record(&mut self.session, ibuf, count).map_err(maptobackend)?;
            Ok(self.pushslot(slot, inshape))
        }

        fn recordembedding(
            &mut self,
            table: operand<'_>,
            indices: &[usize],
        ) -> Result<stepref, backenderror> {
            let tableshape = self.shapeof(table)?;
            self.checkbudget(&tableshape)?;
            let geometry = embeddingshape::new(&tableshape, indices)?;
            self.checkgatherbudget(geometry.rows, geometry.width)?;
            let outshape = embeddingoutshape(&tableshape, geometry.rows)?;
            if geometry.rows == 0 {
                return Ok(self.push(Vec::new(), outshape));
            }
            let tablebuf = self.operandsbuf(table)?;
            let index64 = tou64(indices);
            let indexslot = self.session.uploadu64(&index64).map_err(maptobackend)?;
            let indexbuf = buf::device(self.session.slotptr(indexslot));
            let slot = ffi::embedding_record(
                &mut self.session,
                tablebuf,
                indexbuf,
                geometry.rows,
                geometry.width,
            )
            .map_err(maptobackend)?;
            Ok(self.pushslot(slot, outshape))
        }

        fn recordrope(
            &mut self,
            input: operand<'_>,
            heads: usize,
            base: f32,
        ) -> Result<stepref, backenderror> {
            let inshape = self.shapeof(input)?;
            self.checkbudget(&inshape)?;
            let geometry = ropeshape::new(&inshape, heads)?;
            let count = inshape
                .getelementcount()
                .map_err(|_| backenderror::sizeoverflow)?;
            let inputbuf = self.operandsbuf(input)?;
            let slot = ffi::rope_record(
                &mut self.session,
                inputbuf,
                count,
                geometry.rows,
                geometry.seqaxis,
                geometry.heads,
                geometry.headdim,
                base,
            )
            .map_err(maptobackend)?;
            Ok(self.pushslot(slot, inshape))
        }

        fn recordrope_at(
            &mut self,
            input: operand<'_>,
            position: usize,
            heads: usize,
            base: f32,
        ) -> Result<stepref, backenderror> {
            let inshape = self.shapeof(input)?;
            self.checkbudget(&inshape)?;
            let geometry = ropeshape::new(&inshape, heads)?;
            let count = inshape
                .getelementcount()
                .map_err(|_| backenderror::sizeoverflow)?;
            let inputbuf = self.operandsbuf(input)?;
            let slot = ffi::rope_at_record(
                &mut self.session,
                inputbuf,
                count,
                position,
                geometry.heads,
                geometry.headdim,
                base,
            )
            .map_err(maptobackend)?;
            Ok(self.pushslot(slot, inshape))
        }

        fn recordattention(
            &mut self,
            query: operand<'_>,
            key: operand<'_>,
            value: operand<'_>,
            heads: usize,
            kvheads: usize,
            causal: bool,
        ) -> Result<stepref, backenderror> {
            let queryshape = self.shapeof(query)?;
            let keyshape = self.shapeof(key)?;
            let valueshape = self.shapeof(value)?;
            self.checkbudget(&queryshape)?;
            self.checkbudget(&keyshape)?;
            self.checkbudget(&valueshape)?;
            let geometry = attentionshape::new(
                &queryshape,
                &keyshape,
                &valueshape,
                heads,
                kvheads,
                MUSAMAXHEADDIM,
            )?;
            let count = queryshape
                .getelementcount()
                .map_err(|_| backenderror::sizeoverflow)?;
            let query = self.operandsbuf(query)?;
            let key = self.operandsbuf(key)?;
            let value = self.operandsbuf(value)?;
            let slot = ffi::attention_record(
                &mut self.session,
                query,
                key,
                value,
                count,
                geometry.qseq,
                geometry.kvseq,
                geometry.heads,
                geometry.kvheads,
                geometry.headdim,
                causal,
            )
            .map_err(maptobackend)?;
            Ok(self.pushslot(slot, queryshape))
        }

        fn recordattention_cached(
            &mut self,
            query: operand<'_>,
            key: operandpair<'_>,
            value: operandpair<'_>,
            heads: usize,
            kvheads: usize,
            causal: bool,
        ) -> Result<stepref, backenderror> {
            let queryshape = self.shapeof(query)?;
            let cachedkeyshape = self.shapeof(key.cached)?;
            let cachedvalueshape = self.shapeof(value.cached)?;
            let newkeyshape = self.shapeof(key.new)?;
            let newvalueshape = self.shapeof(value.new)?;
            self.checkbudget(&queryshape)?;
            self.checkbudget(&cachedkeyshape)?;
            self.checkbudget(&cachedvalueshape)?;
            self.checkbudget(&newkeyshape)?;
            self.checkbudget(&newvalueshape)?;
            let keyshape = concatshape(&cachedkeyshape, &newkeyshape)?;
            let valueshape = concatshape(&cachedvalueshape, &newvalueshape)?;
            let geometry = attentionshape::new(
                &queryshape,
                &keyshape,
                &valueshape,
                heads,
                kvheads,
                MUSAMAXHEADDIM,
            )?;
            let count = queryshape
                .getelementcount()
                .map_err(|_| backenderror::sizeoverflow)?;
            let query = self.operandsbuf(query)?;
            let cachedkey = self.operandsbuf(key.cached)?;
            let cachedvalue = self.operandsbuf(value.cached)?;
            let newkey = self.operandsbuf(key.new)?;
            let newvalue = self.operandsbuf(value.new)?;
            let slot = ffi::attention_cached_record(
                &mut self.session,
                query,
                cachedkey,
                cachedvalue,
                newkey,
                newvalue,
                count,
                geometry.kvseq,
                geometry.heads,
                geometry.kvheads,
                geometry.headdim,
                causal,
            )
            .map_err(maptobackend)?;
            Ok(self.pushslot(slot, queryshape))
        }

        fn recordscan(
            &mut self,
            _input: operand<'_>,
            _init: operand<'_>,
            _a: operand<'_>,
            _b: operand<'_>,
            _c: operand<'_>,
            _skip: operand<'_>,
        ) -> Result<stepref, backenderror> {
            Err(backenderror::unsupportedoperation { op: "scan" })
        }

        fn recordscan_cached(
            &mut self,
            _input: operand<'_>,
            _state: operand<'_>,
            _a: operand<'_>,
            _b: operand<'_>,
            _c: operand<'_>,
            _skip: operand<'_>,
        ) -> Result<stepref, backenderror> {
            Err(backenderror::unsupportedoperation { op: "scan_cached" })
        }

        fn apply(mut self, outputs: &[stepref]) -> Result<Vec<tensor<f32>>, backenderror> {
            let slotindices = self.requestslots(outputs)?;
            let downloaded = self.session.apply(&slotindices).map_err(maptobackend)?;
            self.materialize(outputs, downloaded)
        }
    }

    impl musasubmission {
        /// Maps the requested step references to their device slots.
        fn requestslots(&self, outputs: &[stepref]) -> Result<Vec<usize>, backenderror> {
            let mut slotindices = Vec::new();
            for reference in outputs {
                let index = reference.getindex();
                match self.results.get(index) {
                    Some((resultdata::slot(slot), _)) => slotindices.push(*slot),
                    Some((resultdata::host(_), _)) => {}
                    Some((resultdata::dead, _)) => return Err(stalestep()),
                    None => {
                        return Err(backenderror::indexoutofrange {
                            index,
                            len: self.results.len(),
                        });
                    }
                }
            }
            Ok(slotindices)
        }

        /// Rebuilds the requested outputs in request order.
        fn materialize(
            &self,
            outputs: &[stepref],
            downloaded: Vec<Vec<f32>>,
        ) -> Result<Vec<tensor<f32>>, backenderror> {
            let mut cursor = 0;
            let mut built = Vec::with_capacity(outputs.len());
            for reference in outputs {
                let index = reference.getindex();
                let (data, outshape) =
                    self.results
                        .get(index)
                        .ok_or(backenderror::indexoutofrange {
                            index,
                            len: self.results.len(),
                        })?;
                let data = match data {
                    resultdata::host(data) => data.clone(),
                    resultdata::slot(_) => {
                        let data = downloaded[cursor].clone();
                        cursor += 1;
                        data
                    }
                    resultdata::dead => return Err(stalestep()),
                };
                built.push(wrap("musa", outshape.clone(), data)?);
            }
            Ok(built)
        }

        /// Applies a persistent step without consuming the submission: downloads
        /// the requested outputs, frees every transient slot, and keeps the
        /// persistent cache buffers (and their step indices) alive.
        fn applyref(&mut self, outputs: &[stepref]) -> Result<Vec<tensor<f32>>, backenderror> {
            let slotindices = self.requestslots(outputs)?;
            let downloaded = self
                .session
                .applyretained(&slotindices)
                .map_err(maptobackend)?;
            let built = self.materialize(outputs, downloaded)?;
            for (data, _) in self.results.iter_mut() {
                match data {
                    resultdata::slot(slot) if !self.session.ispersistent(*slot) => {
                        *data = resultdata::dead;
                    }
                    resultdata::host(_) => *data = resultdata::dead,
                    _ => {}
                }
            }
            self.inputs.clear();
            Ok(built)
        }
    }
}
/// The AMD / Hygon ROCm backend.
#[cfg(all(target_os = "linux", rocm_toolkit))]
pub mod rocm {
    use crate::{GENBACKENDMOD, GENHOSTRECORD};
    GENBACKENDMOD!(rocm, "rocm", 8192);
    GENHOSTRECORD!();
}
/// The Intel oneAPI SYCL backend.
///
/// Unlike the other GRID-family islands, SYCL records on the device: every
/// step submits to the session's own in-order queue, intermediate results stay
/// in shared USM, and `apply` pays one wait plus the requested downloads. See
/// [`crate::ffi::sycl`] for the lifetime rules that keep an in-flight kernel
/// from outliving its operands.
#[cfg(all(any(target_os = "linux", target_os = "windows"), sycl_toolkit))]
pub mod sycl {
    use crate::GENBACKENDMOD;
    GENBACKENDMOD!(sycl, "sycl", 8192);

    /// The device-resident handle an operand resolves to.
    ///
    /// A thin public wrapper over the island's crate-private handle: an
    /// `impl` in a public module cannot name the crate-private type directly.
    #[derive(Clone)]
    pub struct devicebuffer(ffi::buffer);

    /// A recorded session over this platform, backed by the island's
    /// device-resident session.
    pub struct recordingsession(ffi::session);

    impl enginerecordplatform for platform {
        type buffer = devicebuffer;
        type session = recordingsession;

        // Shared USM has no implicit synchronisation on free: a caller upload
        // must stay alive from its record step until the one wait in `apply`.
        // The submission's input table is what provides that lifetime.
        const DEDUP_INPUTS: bool = true;

        fn newsession(&self) -> Result<Self::session, backenderror> {
            ffi::session::new(self.cache.clone())
                .map(recordingsession)
                .map_err(maptobackend)
        }

        fn upload(
            &self,
            session: &Self::session,
            data: &[f32],
        ) -> Result<Self::buffer, backenderror> {
            session
                .0
                .upload(data)
                .map(devicebuffer)
                .map_err(maptobackend)
        }

        fn getbuffer(&self, session: &Self::session, slot: usize) -> Option<Self::buffer> {
            session.0.getbuffer(slot).map(devicebuffer)
        }

        fn gethostdata(&self, session: &Self::session, slot: usize) -> Option<Vec<f32>> {
            session.0.gethostdata(slot)
        }

        fn applysession(&self, session: &mut Self::session) -> Result<Vec<Vec<f32>>, backenderror> {
            session.0.apply().map_err(maptobackend)
        }

        fn resetsession(&self, session: &mut Self::session) {
            session.0.reset();
        }

        fn recordmatmul(
            &self,
            session: &mut Self::session,
            lhs: &Self::buffer,
            rhs: &Self::buffer,
            geometry: &matmulshape,
        ) -> Result<usize, backenderror> {
            ffi::matmul_record(
                &mut session.0,
                &lhs.0,
                &rhs.0,
                geometry.batch,
                geometry.m,
                geometry.k,
                geometry.n,
            )
            .map_err(maptobackend)
        }

        fn recordadd(
            &self,
            session: &mut Self::session,
            lhs: &Self::buffer,
            rhs: &Self::buffer,
            count: usize,
        ) -> Result<usize, backenderror> {
            ffi::add_record(&mut session.0, &lhs.0, &rhs.0, count).map_err(maptobackend)
        }

        fn recordmul(
            &self,
            session: &mut Self::session,
            lhs: &Self::buffer,
            rhs: &Self::buffer,
            count: usize,
        ) -> Result<usize, backenderror> {
            ffi::mul_record(&mut session.0, &lhs.0, &rhs.0, count).map_err(maptobackend)
        }

        fn recordrmsnorm(
            &self,
            session: &mut Self::session,
            input: &Self::buffer,
            weight: &Self::buffer,
            geometry: &axisshape,
            eps: f32,
        ) -> Result<usize, backenderror> {
            ffi::rmsnorm_record(
                &mut session.0,
                &input.0,
                &weight.0,
                geometry.rows,
                geometry.width,
                eps,
            )
            .map_err(maptobackend)
        }

        fn recordsoftmax(
            &self,
            session: &mut Self::session,
            input: &Self::buffer,
            geometry: &axisshape,
        ) -> Result<usize, backenderror> {
            ffi::softmax_record(&mut session.0, &input.0, geometry.rows, geometry.width)
                .map_err(maptobackend)
        }

        fn recordsilu(
            &self,
            session: &mut Self::session,
            input: &Self::buffer,
            count: usize,
        ) -> Result<usize, backenderror> {
            ffi::silu_record(&mut session.0, &input.0, count).map_err(maptobackend)
        }

        fn recordembedding(
            &self,
            session: &mut Self::session,
            table: &Self::buffer,
            indices: &[usize],
            geometry: &embeddingshape,
        ) -> Result<usize, backenderror> {
            let index64 = tou64(indices);
            ffi::embedding_record(
                &mut session.0,
                &table.0,
                &index64,
                geometry.rows,
                geometry.width,
            )
            .map_err(maptobackend)
        }

        fn recordrope(
            &self,
            session: &mut Self::session,
            input: &Self::buffer,
            geometry: &ropeshape,
            base: f32,
        ) -> Result<usize, backenderror> {
            ffi::rope_record(
                &mut session.0,
                &input.0,
                geometry.rows,
                geometry.seqaxis,
                geometry.heads,
                geometry.headdim,
                base,
            )
            .map_err(maptobackend)
        }

        fn recordrope_at(
            &self,
            session: &mut Self::session,
            input: &Self::buffer,
            position: usize,
            geometry: &ropeshape,
            base: f32,
        ) -> Result<usize, backenderror> {
            ffi::rope_at_record(
                &mut session.0,
                &input.0,
                position,
                geometry.heads,
                geometry.headdim,
                base,
            )
            .map_err(maptobackend)
        }

        fn recordattention(
            &self,
            session: &mut Self::session,
            query: &Self::buffer,
            key: &Self::buffer,
            value: &Self::buffer,
            geometry: &attentionshape,
            causal: bool,
        ) -> Result<usize, backenderror> {
            ffi::attention_record(
                &mut session.0,
                &query.0,
                &key.0,
                &value.0,
                geometry.qseq,
                geometry.kvseq,
                geometry.heads,
                geometry.kvheads,
                geometry.headdim,
                causal,
            )
            .map_err(maptobackend)
        }

        fn recordattention_cached(
            &self,
            session: &mut Self::session,
            query: &Self::buffer,
            keycached: &Self::buffer,
            keynew: &Self::buffer,
            valuecached: &Self::buffer,
            valuenew: &Self::buffer,
            geometry: &attentionshape,
            causal: bool,
        ) -> Result<usize, backenderror> {
            ffi::attention_cached_record(
                &mut session.0,
                &query.0,
                &keycached.0,
                &valuecached.0,
                &keynew.0,
                &valuenew.0,
                geometry.kvseq,
                geometry.heads,
                geometry.kvheads,
                geometry.headdim,
                causal,
            )
            .map_err(maptobackend)
        }
    }
}
/// The Huawei CANN backend.
#[cfg(all(target_os = "linux", cann_toolkit))]
pub mod cann {
    use crate::{GENBACKENDMOD, GENHOSTRECORD};
    GENBACKENDMOD!(cann, "cann", 4096);
    GENHOSTRECORD!();
}
/// The Cambricon MLU backend.
#[cfg(all(target_os = "linux", mlu_toolkit))]
pub mod mlu {
    use crate::{GENBACKENDMOD, GENHOSTRECORD};
    GENBACKENDMOD!(mlu, "mlu", 4096);
    GENHOSTRECORD!();
}
