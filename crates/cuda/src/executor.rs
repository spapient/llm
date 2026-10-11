use std::sync::Arc;

use ::tensor::{shape, tensor, tensorview};
use api::{backend, backenderror, checkbudget, checkgatherbudget, compute};
use lib::limits;

use crate::backend::cudabackend;
use crate::error::cudaerror;
use crate::ffi;
use crate::geometry::{
    ADDOP, MULOP, attentionshape, axisshape, binaryshape, embeddingshape, matmulshape, ropeshape,
};

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

/// A CUDA executor: the crate's [`compute`](api::compute) provider.
///
/// `cudaexecutor` owns a [`cudabackend`] and is the single place where its
/// declared rank and byte budget is enforced. Besides the budget it holds only
/// device state and compiled kernels (kept in `ffi.rs`), never a request-scoped
/// value, so one value can be shared immutably across requests.
///
/// The operator set is complete: `matmul` is served by cuBLAS (or, on the
/// reference path, by an NVRTC kernel), `softmax` by cuDNN, and every remaining
/// operator by an in-repo CUDA C++ kernel compiled at run time through NVRTC.
/// See docs/design.md §21.2 decision 11 for why each operator uses the
/// mechanism it does.
#[derive(Clone, Debug)]
pub struct cudaexecutor {
    backend: cudabackend,
    path: matmulpath,
    cache: Arc<ffi::cache>,
}

impl cudaexecutor {
    /// Creates an executor whose `matmul` is served by cuBLAS.
    ///
    /// The executor owns the device state and the compiled kernels: a clone
    /// shares them, while an independently constructed executor retains its own
    /// device handle and compiles its own kernels (docs/design.md §21.6).
    #[must_use]
    pub fn new(lim: limits) -> Self {
        Self {
            backend: cudabackend::new(lim),
            path: matmulpath::cublas,
            cache: Arc::new(ffi::cache::new()),
        }
    }

    /// Creates an executor whose `matmul` is served by the NVRTC kernel.
    ///
    /// This is the independent path the cuBLAS path is validated against. It
    /// exists as a constructor rather than a flag so that a reference run can
    /// never be confused with a production run.
    #[must_use]
    pub fn newnvrtc(lim: limits) -> Self {
        Self {
            backend: cudabackend::new(lim),
            path: matmulpath::nvrtc,
            cache: Arc::new(ffi::cache::new()),
        }
    }

    /// Returns how many kernels this executor has compiled so far.
    ///
    /// The kernel sources are compiled at run time, once per `(source, entry)`
    /// pair, and reused by every later call; the count therefore rises to the
    /// number of distinct kernels a workload exercises and then stops. It is the
    /// observable that makes "compiled once" checkable from outside the crate
    /// rather than only by timing.
    #[must_use]
    pub fn getcachedkernelcount(&self) -> usize {
        self.cache.kernelcount()
    }

    /// Returns the backend whose budget this executor enforces.
    #[must_use]
    pub fn getbackend(&self) -> &cudabackend {
        &self.backend
    }

    /// Returns which implementation currently serves `matmul`.
    #[must_use]
    pub fn getpath(&self) -> matmulpath {
        self.path
    }

    /// Validates a shape's rank, element count, and byte size against the
    /// backend budget.
    ///
    /// The predicate itself lives in `api` so that this crate and `metal`
    /// cannot classify the same operand differently (docs/design.md §21.4).
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

    /// Runs one block of `[m, k] * [k, n]` on the configured path.
    fn dispatch(
        &self,
        lhs: &[f32],
        rhs: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>, cudaerror> {
        match self.path {
            matmulpath::cublas => ffi::matmul_cublas(&self.cache, lhs, rhs, m, k, n),
            matmulpath::nvrtc => ffi::matmul_nvrtc(&self.cache, lhs, rhs, m, k, n),
        }
    }
}

/// Rebuilds a kernel result into a tensor of `outshape`.
fn wrap(outshape: shape, data: Vec<f32>) -> Result<tensor<f32>, backenderror> {
    tensor::new(outshape, data).map_err(|_| backenderror::executionfailed)
}

/// Maps a crate-local [`cudaerror`] onto the frozen `api::backenderror`.
fn maptobackend(error: cudaerror) -> backenderror {
    match error {
        cudaerror::sizeoverflow => backenderror::sizeoverflow,
        cudaerror::device { .. }
        | cudaerror::compile { .. }
        | cudaerror::module { .. }
        | cudaerror::transfer { .. }
        | cudaerror::launch { .. }
        | cudaerror::blas { .. }
        | cudaerror::cudnn { .. } => backenderror::executionfailed,
    }
}

impl compute for cudaexecutor {
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
        let outelements = outshape
            .getelementcount()
            .map_err(|_| backenderror::sizeoverflow)?;

        let lhsdata = lhs.getdata();
        let rhsdata = rhs.getdata();
        let mut data = Vec::new();
        data.try_reserve_exact(outelements)
            .map_err(|_| backenderror::resourceexhausted)?;
        for b in 0..geometry.batch {
            let lstart = b * geometry.m * geometry.k;
            let rstart = b * geometry.k * geometry.n;
            let lblock = &lhsdata[lstart..lstart + geometry.m * geometry.k];
            let rblock = &rhsdata[rstart..rstart + geometry.k * geometry.n];
            let block = self
                .dispatch(lblock, rblock, geometry.m, geometry.k, geometry.n)
                .map_err(maptobackend)?;
            data.extend_from_slice(&block);
        }

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
