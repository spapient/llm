use ::tensor::{shape, tensor, tensorview};
use api::{
    backend, backenderror, binarykernel, compute, gatherkernel, kernel, kvpair, weightedkernel,
};
use lib::limits;

use crate::activation::{cpugelutanh, cpurelu, cpusilu};
use crate::attention::cpuattention;
use crate::backend::cpubackend;
use crate::elementwise::{cpuadd, cpumul};
use crate::embedding::cpuembedding;
use crate::error::cpuerror;
use crate::matmul::{cpumatmul, matmulplan};
use crate::normalization::cpurmsnorm;
use crate::rotary::cpurope;
use crate::scan::cpuscan;
use crate::softmax::cpusoftmax;

/// Names a CPU operator for dispatch.
///
/// The variants are the operators this backend implements in the current
/// stage. The unary operators are dispatched by [`cpuexecutor::apply`], the
/// binary operators by [`cpuexecutor::applybinary`], the weighted operator by
/// [`cpuexecutor::applyweighted`], and the gather operator by
/// [`cpuexecutor::applygather`]. `rope` and `attention` carry per-call
/// attributes and are reachable through the [`compute`](api::compute)
/// implementation instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum cpuop {
    /// Rectified linear unit.
    relu,
    /// Sigmoid linear unit.
    silu,
    /// Tanh-approximation GELU.
    gelutanh,
    /// Numerically stable last-axis softmax.
    softmax,
    /// Matrix multiplication.
    matmul,
    /// Elementwise addition.
    add,
    /// Elementwise multiplication.
    mul,
    /// RMS normalization over the last axis with a weight.
    rmsnorm,
    /// Row gather from an embedding table.
    embedding,
}

impl cpuop {
    /// Returns the stable operator name.
    #[must_use]
    pub const fn getname(self) -> &'static str {
        match self {
            Self::relu => "relu",
            Self::silu => "silu",
            Self::gelutanh => "gelutanh",
            Self::softmax => "softmax",
            Self::matmul => "matmul",
            Self::add => "add",
            Self::mul => "mul",
            Self::rmsnorm => "rmsnorm",
            Self::embedding => "embedding",
        }
    }
}

/// A CPU executor that enforces the backend budget and dispatches to kernels.
///
/// The executor owns a [`cpubackend`] and is the single place where its
/// declared element, byte, and rank budget is enforced. The kernels themselves
/// are pure and stateless; they are composed here through static dispatch. The
/// executor is also the crate's [`compute`](api::compute) provider, so the
/// differentiation and neural-network layers reach the CPU through the `api`
/// contract rather than a direct dependency on this crate.
#[derive(Clone, Debug)]
pub struct cpuexecutor {
    backend: cpubackend,
}

impl cpuexecutor {
    /// Creates an executor over a CPU backend configured with `lim`.
    #[must_use]
    pub fn new(lim: limits) -> Self {
        Self {
            backend: cpubackend::new(lim),
        }
    }

    /// Returns the backend whose budget this executor enforces.
    #[must_use]
    pub fn getbackend(&self) -> &cpubackend {
        &self.backend
    }

    /// Validates `input` against the backend budget and applies the unary `op`.
    ///
    /// # Errors
    ///
    /// Returns [`cpuerror::rankexceeded`] when the input rank exceeds the
    /// declared maximum, [`cpuerror::limit`] when the element or byte budget is
    /// exceeded, and [`cpuerror::backend`] when the operator is not unary or the
    /// kernel itself rejects the input.
    pub fn apply(&self, op: cpuop, input: &tensorview<'_, f32>) -> Result<tensor<f32>, cpuerror> {
        self.checkbudget(input.getshape())?;
        match op {
            cpuop::relu => Ok(cpurelu.apply(input)?),
            cpuop::silu => Ok(cpusilu.apply(input)?),
            cpuop::gelutanh => Ok(cpugelutanh.apply(input)?),
            cpuop::softmax => Ok(cpusoftmax.apply(input)?),
            other => Err(cpuerror::backend(backenderror::unsupportedoperation {
                op: other.getname(),
            })),
        }
    }

    /// Validates both operands against the backend budget and applies the
    /// binary `op`.
    ///
    /// # Errors
    ///
    /// Returns the same budget errors as [`Self::apply`], or
    /// [`cpuerror::backend`] when `op` is not binary or the kernel rejects the
    /// operand shapes.
    pub fn applybinary(
        &self,
        op: cpuop,
        lhs: &tensorview<'_, f32>,
        rhs: &tensorview<'_, f32>,
    ) -> Result<tensor<f32>, cpuerror> {
        self.checkbudget(lhs.getshape())?;
        match op {
            cpuop::matmul => {
                self.checkbudget(rhs.getshape())?;
                // The result is allocated by this executor's kernel, so the
                // backend budget covers it too; the check runs before the
                // output vector exists.
                let plan = matmulplan(lhs, rhs)?;
                self.checkbudget(&plan.outshape()?)?;
                Ok(cpumatmul.applyplan(plan, lhs, rhs)?)
            }
            cpuop::add => {
                self.checkbudget(rhs.getshape())?;
                Ok(cpuadd.apply(lhs, rhs)?)
            }
            cpuop::mul => {
                self.checkbudget(rhs.getshape())?;
                Ok(cpumul.apply(lhs, rhs)?)
            }
            other => Err(cpuerror::backend(backenderror::unsupportedoperation {
                op: other.getname(),
            })),
        }
    }

    /// Validates the input and weight against the backend budget and applies
    /// the weighted `op`.
    ///
    /// # Errors
    ///
    /// Returns the same budget errors as [`Self::apply`], or
    /// [`cpuerror::backend`] when `op` is not weighted or the kernel rejects
    /// the weight shape.
    pub fn applyweighted(
        &self,
        op: cpuop,
        input: &tensorview<'_, f32>,
        weight: &tensorview<'_, f32>,
    ) -> Result<tensor<f32>, cpuerror> {
        self.checkbudget(input.getshape())?;
        match op {
            cpuop::rmsnorm => {
                self.checkbudget(weight.getshape())?;
                Ok(cpurmsnorm::default().apply(input, weight)?)
            }
            other => Err(cpuerror::backend(backenderror::unsupportedoperation {
                op: other.getname(),
            })),
        }
    }

    /// Validates the table and the gathered output size against the backend
    /// budget and applies the gather `op`.
    ///
    /// # Errors
    ///
    /// Returns the same budget errors as [`Self::apply`], or
    /// [`cpuerror::backend`] when `op` is not a gather or the kernel rejects an
    /// index.
    pub fn applygather(
        &self,
        op: cpuop,
        table: &tensorview<'_, f32>,
        indices: &[usize],
    ) -> Result<tensor<f32>, cpuerror> {
        self.checkbudget(table.getshape())?;
        match op {
            cpuop::embedding => {
                self.checkgatherbudget(table.getshape(), indices.len())?;
                Ok(cpuembedding.apply(table, indices)?)
            }
            other => Err(cpuerror::backend(backenderror::unsupportedoperation {
                op: other.getname(),
            })),
        }
    }

    fn checkbudget(&self, s: &shape) -> Result<(), cpuerror> {
        let caps = self.backend.getcapabilities();
        if s.getrank() > caps.getmaxrank() {
            return Err(cpuerror::rankexceeded {
                rank: s.getrank(),
                max: caps.getmaxrank(),
            });
        }
        let elements = s
            .getelementcount()
            .map_err(|_| backenderror::sizeoverflow)?;
        let lim = caps.getlimits();
        lim.checkelements(elements)?;
        let bytes = elements
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(backenderror::sizeoverflow)?;
        lim.checkbytes(bytes)?;
        Ok(())
    }

    fn checkgatherbudget(&self, table: &shape, rows: usize) -> Result<(), cpuerror> {
        let width = table.getdims()[1..]
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or(backenderror::sizeoverflow)?;
        let elements = rows.checked_mul(width).ok_or(backenderror::sizeoverflow)?;
        let lim = self.backend.getcapabilities().getlimits();
        lim.checkelements(elements)?;
        let bytes = elements
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(backenderror::sizeoverflow)?;
        lim.checkbytes(bytes)?;
        Ok(())
    }
}

impl compute for cpuexecutor {
    fn matmul(
        &self,
        lhs: &tensorview<'_, f32>,
        rhs: &tensorview<'_, f32>,
    ) -> Result<tensor<f32>, backenderror> {
        self.applybinary(cpuop::matmul, lhs, rhs)
            .map_err(maptobackend)
    }

    fn add(
        &self,
        lhs: &tensorview<'_, f32>,
        rhs: &tensorview<'_, f32>,
    ) -> Result<tensor<f32>, backenderror> {
        self.applybinary(cpuop::add, lhs, rhs).map_err(maptobackend)
    }

    fn mul(
        &self,
        lhs: &tensorview<'_, f32>,
        rhs: &tensorview<'_, f32>,
    ) -> Result<tensor<f32>, backenderror> {
        self.applybinary(cpuop::mul, lhs, rhs).map_err(maptobackend)
    }

    fn rmsnorm(
        &self,
        input: &tensorview<'_, f32>,
        weight: &tensorview<'_, f32>,
        eps: f32,
    ) -> Result<tensor<f32>, backenderror> {
        self.checkbudget(input.getshape()).map_err(maptobackend)?;
        self.checkbudget(weight.getshape()).map_err(maptobackend)?;
        cpurmsnorm::new(eps).apply(input, weight)
    }

    fn softmax(&self, input: &tensorview<'_, f32>) -> Result<tensor<f32>, backenderror> {
        self.apply(cpuop::softmax, input).map_err(maptobackend)
    }

    fn silu(&self, input: &tensorview<'_, f32>) -> Result<tensor<f32>, backenderror> {
        self.apply(cpuop::silu, input).map_err(maptobackend)
    }

    fn embedding(
        &self,
        table: &tensorview<'_, f32>,
        indices: &[usize],
    ) -> Result<tensor<f32>, backenderror> {
        self.applygather(cpuop::embedding, table, indices)
            .map_err(maptobackend)
    }

    fn rope(
        &self,
        input: &tensorview<'_, f32>,
        heads: usize,
        base: f32,
    ) -> Result<tensor<f32>, backenderror> {
        self.checkbudget(input.getshape()).map_err(maptobackend)?;
        cpurope::new(heads, base).apply(input)
    }

    fn rope_at(
        &self,
        input: &tensorview<'_, f32>,
        position: usize,
        heads: usize,
        base: f32,
    ) -> Result<tensor<f32>, backenderror> {
        let width = input.getshape().getdims()[1];
        let mut data = vec![0.0_f32; (position + 1) * width];
        data[position * width..].copy_from_slice(input.getdata());
        let padded = tensor::new(
            shape::new(vec![position + 1, width]).map_err(|_| backenderror::sizeoverflow)?,
            data,
        )
        .map_err(|_| backenderror::executionfailed)?;
        let rotated = self.rope(&padded.getview(), heads, base)?;
        let start = position * width;
        let last = rotated.getdata()[start..start + width].to_vec();
        tensor::new(
            shape::new(vec![1, width]).map_err(|_| backenderror::sizeoverflow)?,
            last,
        )
        .map_err(|_| backenderror::executionfailed)
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
        self.checkbudget(query.getshape()).map_err(maptobackend)?;
        self.checkbudget(key.getshape()).map_err(maptobackend)?;
        self.checkbudget(value.getshape()).map_err(maptobackend)?;
        cpuattention.apply(query, key, value, heads, kvheads, causal)
    }

    fn attention_cached(
        &self,
        query: &tensorview<'_, f32>,
        key: kvpair<'_>,
        value: kvpair<'_>,
        heads: usize,
        kvheads: usize,
        causal: bool,
    ) -> Result<tensor<f32>, backenderror> {
        let key = concat(&key.cached, &key.new)?;
        let value = concat(&value.cached, &value.new)?;
        self.attention(
            query,
            &key.getview(),
            &value.getview(),
            heads,
            kvheads,
            causal,
        )
    }

    fn scan(
        &self,
        input: &tensorview<'_, f32>,
        init: &tensorview<'_, f32>,
        a: &tensorview<'_, f32>,
        b: &tensorview<'_, f32>,
        c: &tensorview<'_, f32>,
        skip: &tensorview<'_, f32>,
    ) -> Result<(tensor<f32>, tensor<f32>), backenderror> {
        self.checkbudget(input.getshape()).map_err(maptobackend)?;
        self.checkbudget(init.getshape()).map_err(maptobackend)?;
        self.checkbudget(a.getshape()).map_err(maptobackend)?;
        self.checkbudget(b.getshape()).map_err(maptobackend)?;
        self.checkbudget(c.getshape()).map_err(maptobackend)?;
        self.checkbudget(skip.getshape()).map_err(maptobackend)?;
        cpuscan.apply(input, init, a, b, c, skip)
    }

    fn scan_cached(
        &self,
        input: &tensorview<'_, f32>,
        state: &tensorview<'_, f32>,
        a: &tensorview<'_, f32>,
        b: &tensorview<'_, f32>,
        c: &tensorview<'_, f32>,
        skip: &tensorview<'_, f32>,
    ) -> Result<(tensor<f32>, tensor<f32>), backenderror> {
        self.checkbudget(input.getshape()).map_err(maptobackend)?;
        self.checkbudget(state.getshape()).map_err(maptobackend)?;
        self.checkbudget(a.getshape()).map_err(maptobackend)?;
        self.checkbudget(b.getshape()).map_err(maptobackend)?;
        self.checkbudget(c.getshape()).map_err(maptobackend)?;
        self.checkbudget(skip.getshape()).map_err(maptobackend)?;
        cpuscan.apply_cached(input, state, a, b, c, skip)
    }
}

/// Concatenates two `[_, width]` views into one `[a + b, width]` tensor.
fn concat(a: &tensorview<'_, f32>, b: &tensorview<'_, f32>) -> Result<tensor<f32>, backenderror> {
    let width = a.getshape().getdims()[1];
    let mut data = a.getdata().to_vec();
    data.extend_from_slice(b.getdata());
    let len = data.len() / width;
    let outshape = shape::new(vec![len, width]).map_err(|_| backenderror::sizeoverflow)?;
    tensor::new(outshape, data).map_err(|_| backenderror::executionfailed)
}

/// Maps a CPU policy error onto the backend contract enforced by `api`.
///
/// A kernel failure passes through unchanged. A rejected element, byte, or rank
/// budget is a resource-exhaustion condition, which is the closest distinction
/// `backenderror` models; `cpuerror` already retained the exact bound.
fn maptobackend(error: cpuerror) -> backenderror {
    match error {
        cpuerror::backend(inner) => inner,
        cpuerror::limit(_) | cpuerror::rankexceeded { .. } => backenderror::resourceexhausted,
    }
}
