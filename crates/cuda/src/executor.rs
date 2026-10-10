use ::tensor::{shape, tensor, tensorview};
use api::{backend, backenderror, compute};
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
/// declared rank and byte budget is enforced. It is stateless between calls and
/// holds no request-scoped state, so one value can be shared immutably across
/// requests.
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
}

impl cudaexecutor {
    /// Creates an executor whose `matmul` is served by cuBLAS.
    #[must_use]
    pub fn new(lim: limits) -> Self {
        Self {
            backend: cudabackend::new(lim),
            path: matmulpath::cublas,
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
        }
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
    fn checkbudget(&self, s: &shape) -> Result<(), backenderror> {
        let caps = self.backend.getcapabilities();
        if s.getrank() > caps.getmaxrank() {
            return Err(backenderror::executionfailed);
        }
        let elements = s
            .getelementcount()
            .map_err(|_| backenderror::sizeoverflow)?;
        caps.getlimits()
            .checkelements(elements)
            .map_err(|_| backenderror::resourceexhausted)?;
        let bytes = elements
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(backenderror::sizeoverflow)?;
        caps.getlimits()
            .checkbytes(bytes)
            .map_err(|_| backenderror::resourceexhausted)?;
        Ok(())
    }

    /// Validates a gathered output against the backend budget.
    ///
    /// The gather's output size is `rows * width`, where `rows` comes from the
    /// index list rather than from any operand shape, so it needs its own check.
    fn checkgatherbudget(&self, rows: usize, width: usize) -> Result<(), backenderror> {
        let elements = rows.checked_mul(width).ok_or(backenderror::sizeoverflow)?;
        let caps = self.backend.getcapabilities();
        caps.getlimits()
            .checkelements(elements)
            .map_err(|_| backenderror::resourceexhausted)?;
        let bytes = elements
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(backenderror::sizeoverflow)?;
        caps.getlimits()
            .checkbytes(bytes)
            .map_err(|_| backenderror::resourceexhausted)?;
        Ok(())
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
            matmulpath::cublas => ffi::matmul_cublas(lhs, rhs, m, k, n),
            matmulpath::nvrtc => ffi::matmul_nvrtc(lhs, rhs, m, k, n),
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

        let outelements = geometry
            .batch
            .checked_mul(geometry.m)
            .and_then(|value| value.checked_mul(geometry.n))
            .ok_or(backenderror::sizeoverflow)?;

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

        let mut outdims = geometry.batchdims;
        outdims.push(geometry.m);
        outdims.push(geometry.n);
        let outshape = shape::new(outdims).map_err(|_| backenderror::executionfailed)?;
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
        let data = ffi::add(lhs.getdata(), rhs.getdata()).map_err(maptobackend)?;
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
        let data = ffi::mul(lhs.getdata(), rhs.getdata()).map_err(maptobackend)?;
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
        let data =
            ffi::softmax(input.getdata(), geometry.rows, geometry.width).map_err(maptobackend)?;
        wrap(input.getshape().clone(), data)
    }

    fn silu(&self, input: &tensorview<'_, f32>) -> Result<tensor<f32>, backenderror> {
        self.checkbudget(input.getshape())?;
        let data = ffi::silu(input.getdata()).map_err(maptobackend)?;
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
        let data =
            ffi::embedding(table.getdata(), indices, geometry.width).map_err(maptobackend)?;

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
