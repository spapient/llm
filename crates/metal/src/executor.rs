use std::sync::Arc;

use ::tensor::{shape, tensor, tensorview};
use api::{backend, backenderror, checkbudget, checkgatherbudget, compute};
use lib::limits;

use crate::backend::metalbackend;
use crate::error::metalerror;
use crate::ffi;
use crate::geometry::{
    ADDOP, MULOP, attentionshape, axisshape, binaryshape, embeddingshape, matmulshape, ropeshape,
};

/// A Metal executor: the crate's [`compute`](api::compute) provider.
///
/// `metalexecutor` owns a [`metalbackend`] and is the single place where its
/// declared element and byte budget is enforced. It is stateless between calls
/// and holds no request-scoped state, so a single value can be shared
/// immutably across requests.
///
/// The operator set is complete: `matmul` is served by the in-repo MSL kernel,
/// and every remaining operator by the MSL kernels embedded in `source.rs`, all
/// compiled at run time through `newLibraryWithSource`. See docs/design.md
/// §21 for the operator-to-kernel mapping.
#[derive(Clone, Debug)]
pub struct metalexecutor {
    backend: metalbackend,
    cache: Arc<ffi::cache>,
}

impl metalexecutor {
    /// Creates an executor over a Metal backend configured with `lim`.
    ///
    /// The executor owns the device state and the compiled pipelines: a clone
    /// shares them, while an independently constructed executor compiles its own
    /// (docs/design.md §21.6).
    #[must_use]
    pub fn new(lim: limits) -> Self {
        Self {
            backend: metalbackend::new(lim),
            cache: Arc::new(ffi::cache::new()),
        }
    }

    /// Returns the backend whose budget this executor enforces.
    #[must_use]
    pub fn getbackend(&self) -> &metalbackend {
        &self.backend
    }

    /// Returns how many compute pipelines this executor has compiled so far.
    ///
    /// The MSL sources are compiled at run time, once per `(source, entry)`
    /// pair, and reused by every later call; the count therefore rises to the
    /// number of distinct kernels a workload exercises and then stops. It is the
    /// observable that makes "compiled once" checkable from outside the crate
    /// rather than only by timing.
    #[must_use]
    pub fn getcachedpipelinecount(&self) -> usize {
        self.cache.pipelinecount()
    }

    /// Validates a shape's rank, element count, and byte size against the
    /// backend budget.
    ///
    /// The predicate itself lives in `api` so that this crate and `cuda` cannot
    /// classify the same operand differently (docs/design.md §21.4).
    fn checkbudget(&self, s: &shape) -> Result<(), backenderror> {
        checkbudget(
            self.backend.getcapabilities(),
            s,
            core::mem::size_of::<f32>(),
        )
    }

    /// Validates a gathered output against the backend budget.
    ///
    /// The gather's output size is `rows * width`, where `rows` comes from the
    /// index list rather than from any operand shape, so it needs its own check.
    fn checkgatherbudget(&self, rows: usize, width: usize) -> Result<(), backenderror> {
        checkgatherbudget(
            self.backend.getcapabilities(),
            rows,
            width,
            core::mem::size_of::<f32>(),
        )
    }
}

/// Rebuilds a kernel result into a tensor of `outshape`.
fn wrap(outshape: shape, data: Vec<f32>) -> Result<tensor<f32>, backenderror> {
    tensor::new(outshape, data).map_err(|_| backenderror::executionfailed)
}

/// Maps a crate-local [`metalerror`] onto the frozen `api::backenderror`.
fn maptobackend(error: metalerror) -> backenderror {
    match error {
        metalerror::sizeoverflow => backenderror::sizeoverflow,
        metalerror::nodevice
        | metalerror::allocation
        | metalerror::compile { .. }
        | metalerror::pipeline { .. }
        | metalerror::execution { .. } => backenderror::executionfailed,
    }
}

impl compute for metalexecutor {
    fn matmul(
        &self,
        lhs: &tensorview<'_, f32>,
        rhs: &tensorview<'_, f32>,
    ) -> Result<tensor<f32>, backenderror> {
        self.checkbudget(lhs.getshape())?;
        self.checkbudget(rhs.getshape())?;
        let geometry = matmulshape::new(lhs.getshape(), rhs.getshape())?;
        // The result is allocated by this crate, so the budget covers it too:
        // the check runs before the host buffer and any device buffer exist.
        let outshape = geometry.outshape()?;
        self.checkbudget(&outshape)?;
        // One dispatch covers every batch block, so the operands are handed
        // over whole rather than sliced per block: the block index travels in
        // the grid (docs/design.md §21.10.2).
        let data = ffi::matmul(
            &self.cache,
            lhs.getdata(),
            rhs.getdata(),
            geometry.batch,
            geometry.m,
            geometry.k,
            geometry.n,
        )
        .map_err(maptobackend)?;

        wrap(outshape, data)
    }

    fn add(
        &self,
        lhs: &tensorview<'_, f32>,
        rhs: &tensorview<'_, f32>,
    ) -> Result<tensor<f32>, backenderror> {
        self.checkbudget(lhs.getshape())?;
        self.checkbudget(rhs.getshape())?;
        let outshape = binaryshape(lhs.getshape(), rhs.getshape(), ADDOP)?;
        let data = ffi::add(&self.cache, lhs.getdata(), rhs.getdata()).map_err(maptobackend)?;
        wrap(outshape, data)
    }

    fn mul(
        &self,
        lhs: &tensorview<'_, f32>,
        rhs: &tensorview<'_, f32>,
    ) -> Result<tensor<f32>, backenderror> {
        self.checkbudget(lhs.getshape())?;
        self.checkbudget(rhs.getshape())?;
        let outshape = binaryshape(lhs.getshape(), rhs.getshape(), MULOP)?;
        let data = ffi::mul(&self.cache, lhs.getdata(), rhs.getdata()).map_err(maptobackend)?;
        wrap(outshape, data)
    }

    fn rmsnorm(
        &self,
        input: &tensorview<'_, f32>,
        weight: &tensorview<'_, f32>,
        eps: f32,
    ) -> Result<tensor<f32>, backenderror> {
        self.checkbudget(input.getshape())?;
        self.checkbudget(weight.getshape())?;
        let geometry = axisshape::rmsnorm(input.getshape(), weight.getshape())?;
        let data = ffi::rmsnorm(
            &self.cache,
            input.getdata(),
            weight.getdata(),
            geometry.rows,
            geometry.width,
            eps,
        )
        .map_err(maptobackend)?;
        wrap(input.getshape().clone(), data)
    }

    fn softmax(&self, input: &tensorview<'_, f32>) -> Result<tensor<f32>, backenderror> {
        self.checkbudget(input.getshape())?;
        let geometry = axisshape::softmax(input.getshape())?;
        let data = ffi::softmax(&self.cache, input.getdata(), geometry.rows, geometry.width)
            .map_err(maptobackend)?;
        wrap(input.getshape().clone(), data)
    }

    fn silu(&self, input: &tensorview<'_, f32>) -> Result<tensor<f32>, backenderror> {
        self.checkbudget(input.getshape())?;
        let data = ffi::silu(&self.cache, input.getdata()).map_err(maptobackend)?;
        wrap(input.getshape().clone(), data)
    }

    fn embedding(
        &self,
        table: &tensorview<'_, f32>,
        indices: &[usize],
    ) -> Result<tensor<f32>, backenderror> {
        self.checkbudget(table.getshape())?;
        let geometry = embeddingshape::new(table.getshape(), indices)?;
        self.checkgatherbudget(geometry.rows, geometry.width)?;
        let data = ffi::embedding(&self.cache, table.getdata(), indices, geometry.width)
            .map_err(maptobackend)?;

        let mut outdims = Vec::with_capacity(table.getshape().getrank());
        outdims.push(geometry.rows);
        outdims.extend_from_slice(&table.getshape().getdims()[1..]);
        let outshape = shape::new(outdims).map_err(|_| backenderror::sizeoverflow)?;
        wrap(outshape, data)
    }

    fn rope(
        &self,
        input: &tensorview<'_, f32>,
        heads: usize,
        base: f32,
    ) -> Result<tensor<f32>, backenderror> {
        self.checkbudget(input.getshape())?;
        let geometry = ropeshape::new(input.getshape(), heads)?;
        let data = ffi::rope(
            &self.cache,
            input.getdata(),
            geometry.rows,
            geometry.seqaxis,
            geometry.heads,
            geometry.headdim,
            base,
        )
        .map_err(maptobackend)?;
        wrap(input.getshape().clone(), data)
    }

    fn attention(
        &self,
        query: &tensorview<'_, f32>,
        key: &tensorview<'_, f32>,
        value: &tensorview<'_, f32>,
        heads: usize,
        kvheads: usize,
        causal: bool,
    ) -> Result<tensor<f32>, backenderror> {
        self.checkbudget(query.getshape())?;
        self.checkbudget(key.getshape())?;
        self.checkbudget(value.getshape())?;
        let geometry = attentionshape::new(
            query.getshape(),
            key.getshape(),
            value.getshape(),
            heads,
            kvheads,
        )?;
        let data = ffi::attention(
            &self.cache,
            query.getdata(),
            key.getdata(),
            value.getdata(),
            &geometry,
            causal,
        )
        .map_err(maptobackend)?;
        wrap(query.getshape().clone(), data)
    }
}
