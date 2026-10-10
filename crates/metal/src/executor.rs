use ::tensor::{shape, tensor, tensorview};
use api::{backend, backenderror, compute};
use lib::limits;

use crate::backend::metalbackend;
use crate::error::metalerror;
use crate::ffi;
use crate::geometry::{
    ADDOP, MULOP, attentionshape, axisshape, binaryshape, embeddingshape, ropeshape,
};

/// The stable name of the matrix-multiply operator.
const MATMULOP: &str = "matmul";

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
}

impl metalexecutor {
    /// Creates an executor over a Metal backend configured with `lim`.
    #[must_use]
    pub fn new(lim: limits) -> Self {
        Self {
            backend: metalbackend::new(lim),
        }
    }

    /// Returns the backend whose budget this executor enforces.
    #[must_use]
    pub fn getbackend(&self) -> &metalbackend {
        &self.backend
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
        | metalerror::bufferallocation
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

        let ld = lhs.getshape().getdims();
        let rd = rhs.getshape().getdims();
        if ld.len() < 2 || rd.len() < 2 {
            return Err(backenderror::shapemismatch { op: MATMULOP });
        }
        let batchdims = &ld[..ld.len() - 2];
        if batchdims != &rd[..rd.len() - 2] {
            return Err(backenderror::shapemismatch { op: MATMULOP });
        }
        let m = ld[ld.len() - 2];
        let k = ld[ld.len() - 1];
        let rk = rd[rd.len() - 2];
        let n = rd[rd.len() - 1];
        if k != rk {
            return Err(backenderror::shapemismatch { op: MATMULOP });
        }

        let batch = batchdims
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or(backenderror::sizeoverflow)?;
        let outelements = batch
            .checked_mul(m)
            .and_then(|x| x.checked_mul(n))
            .ok_or(backenderror::sizeoverflow)?;

        let lhsdata = lhs.getdata();
        let rhsdata = rhs.getdata();
        let mut data = Vec::new();
        data.try_reserve_exact(outelements)
            .map_err(|_| backenderror::resourceexhausted)?;
        for b in 0..batch {
            let lblock = &lhsdata[b * m * k..(b + 1) * m * k];
            let rblock = &rhsdata[b * k * n..(b + 1) * k * n];
            let block = ffi::matmul(lblock, rblock, m, k, n).map_err(maptobackend)?;
            data.extend_from_slice(&block);
        }

        let mut outdims = batchdims.to_vec();
        outdims.push(m);
        outdims.push(n);
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
