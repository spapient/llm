use ::tensor::{shape, tensor, tensorview};
use api::{backenderror, binarykernel};

use crate::support::finish;

/// The stable name reported by [`cpumatmul`].
const MATMULOP: &str = "matmul";

/// Matrix multiplication over `f32` operands.
///
/// `lhs` and `rhs` MUST have equal rank `r >= 2` and equal leading batch
/// dimensions. The two trailing dimensions are read as `[m, k]` and `[k, n]`;
/// the result has shape `[batch..., m, n]`. The inner dimensions must agree.
///
/// No implicit broadcasting is performed: the leading batch dimensions of the
/// two operands must be equal element by element. Accumulation is a plain
/// left-to-right `f32` sum, so the result is deterministic for fixed inputs.
#[derive(Clone, Copy, Debug, Default)]
pub struct cpumatmul;

impl cpumatmul {
    /// Applies the operator using an already validated plan.
    ///
    /// The executor derives the plan itself so that it can validate the output
    /// against the backend budget before any allocation; [`Self::apply`]
    /// derives the same plan for callers that reach the kernel directly.
    pub(crate) fn applyplan(
        &self,
        plan: matmulshape,
        lhs: &tensorview<'_, f32>,
        rhs: &tensorview<'_, f32>,
    ) -> Result<tensor<f32>, backenderror> {
        let data = matmuldata(&plan, lhs, rhs)?;
        let outshape = shape::new(plan.dims).map_err(|_| backenderror::sizeoverflow)?;
        finish(&outshape, data)
    }
}

impl binarykernel<f32> for cpumatmul {
    fn getname(&self) -> &'static str {
        MATMULOP
    }

    fn apply(
        &self,
        lhs: &tensorview<'_, f32>,
        rhs: &tensorview<'_, f32>,
    ) -> Result<tensor<f32>, backenderror> {
        self.applyplan(matmulplan(lhs, rhs)?, lhs, rhs)
    }
}

/// Allocates the output and multiplies every batch block into it.
fn matmuldata(
    plan: &matmulshape,
    lhs: &tensorview<'_, f32>,
    rhs: &tensorview<'_, f32>,
) -> Result<Vec<f32>, backenderror> {
    let total = plan
        .batch
        .checked_mul(plan.m)
        .and_then(|x| x.checked_mul(plan.n))
        .ok_or(backenderror::sizeoverflow)?;
    let mut data = vec![0.0_f32; total];
    let lhsdata = lhs.getdata();
    let rhsdata = rhs.getdata();
    for b in 0..plan.batch {
        let a = &lhsdata[b * plan.m * plan.k..(b + 1) * plan.m * plan.k];
        let bt = &rhsdata[b * plan.k * plan.n..(b + 1) * plan.k * plan.n];
        let out = &mut data[b * plan.m * plan.n..(b + 1) * plan.m * plan.n];
        matmulblock(a, bt, out, plan.m, plan.k, plan.n);
    }
    Ok(data)
}

/// The validated shape plan of one batched matrix multiplication.
pub(crate) struct matmulshape {
    dims: Vec<usize>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
}

impl matmulshape {
    /// Returns the output shape, `[batch..., m, n]`.
    pub(crate) fn outshape(&self) -> Result<shape, backenderror> {
        shape::new(self.dims.clone()).map_err(|_| backenderror::sizeoverflow)
    }
}

/// Validates the operand shapes and builds the multiplication plan.
///
/// The plan is derived by the executor as well as by the kernel so that the
/// output budget can be checked before the output is allocated.
pub(crate) fn matmulplan(
    lhs: &tensorview<'_, f32>,
    rhs: &tensorview<'_, f32>,
) -> Result<matmulshape, backenderror> {
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
    let mut dims = batchdims.to_vec();
    dims.push(m);
    dims.push(n);
    Ok(matmulshape {
        dims,
        batch,
        m,
        k,
        n,
    })
}

/// Computes one `[m, k] * [k, n]` block into `out`.
fn matmulblock(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            let brow = &b[p * n..p * n + n];
            let orow = &mut out[i * n..i * n + n];
            for (o, &bv) in orow.iter_mut().zip(brow.iter()) {
                *o += av * bv;
            }
        }
    }
}
