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
        ADDENTRY, AGGENTRY as SCANAGGENTRY, APPLYENTRY as SCANAPPLYENTRY,
        ATTENTION as ATTENTIONCUDA, ATTENTIONCACHED as ATTENTIONCACHEDCUDA, ATTENTIONCACHEDENTRY,
        ATTENTIONENTRY, ATTENTIONLANES, BLOCKDIM as SCANBLOCKDIM, CHUNK as SCANCHUNK, ELEMENTWISE,
        ELEMENTWISETILE, EMBEDDING as EMBEDDINGCUDA, EMBEDDINGENTRY, MULENTRY,
        PARALLEL as SCANPARALLELCUDA, PREFIXENTRY as SCANPREFIXENTRY, RMSNORM as RMSNORMCUDA,
        RMSNORMENTRY, ROPE as ROPECUDA, ROPEAT as ROPEATCUDA, ROPEATENTRY, ROPEENTRY, ROWTILE,
        SCAN as SCANCUDA, SCANENTRY, SCANMAXN, SILUENTRY, SOFTMAX as SOFTMAXCUDA, SOFTMAXENTRY,
    };
    pub(crate) use crate::catalog::cu::{MATMUL as MATMULCUDA, MATMULENTRY, THREADEDGE};
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
        // CUDA has no absolute-position kernel: pad the history on the host and
        // reuse the relative kernel, which is exactly the reference semantics.
        let width = geometry
            .heads
            .checked_mul(geometry.headdim)
            .ok_or(backenderror::sizeoverflow)?;
        let rows = position.checked_add(1).ok_or(backenderror::sizeoverflow)?;
        let mut data = vec![0.0_f32; rows.checked_mul(width).ok_or(backenderror::sizeoverflow)?];
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
        ffi::scan(&self.cache, input, init, a, b, c, skip, n, d, statewidth).map_err(maptobackend)
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

    fn upload(&self, session: &Self::session, data: &[f32]) -> Result<Self::buffer, backenderror> {
        session.0.upload(data).map_err(maptobackend)
    }

    fn getbuffer(&self, session: &Self::session, slot: usize) -> Option<Self::buffer> {
        session.0.getbuffer(slot).cloned()
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
