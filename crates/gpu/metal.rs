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
/// Every entry compiles from the single `metal/metal.c` translation unit, so
/// all `*MSL` constants below are aliases of one source.
pub(crate) mod source {
    pub(crate) use crate::catalog::msl::{
        ADDENTRY, ATTENTIONCACHEDENTRY, ATTENTIONENTRY, ATTENTIONLANES, FLATTILE, MULENTRY,
        RMSNORMENTRY, RMSNORMTILE, ROPEATENTRY, ROPEENTRY, SILUENTRY,
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
        AGGENTRY as SCANAGGENTRY, APPLYENTRY as SCANAPPLYENTRY, CHUNK as SCANCHUNK, EMBEDDINGENTRY,
        MATMULENTRY, PREFIXENTRY as SCANPREFIXENTRY, SCANENTRY, SCANMAXN, SOFTMAXENTRY,
        SOFTMAXTILE,
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
/// attention kernel of `metal/metal.c`.
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
        ffi::scan(&self.cache, input, init, a, b, c, skip, n, d, statewidth).map_err(maptobackend)
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
