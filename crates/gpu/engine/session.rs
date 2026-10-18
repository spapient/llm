use ::tensor::{shape, tensor};
use api::{backenderror, operand, operandpair, stepref};

use super::{recordplatform, submission};

/// A persistent recording session: the same record primitives as a submission,
/// applied repeatedly over one device state.
pub struct session<P: recordplatform> {
    inner: submission<P>,
}

impl<P: recordplatform> session<P> {
    pub(crate) fn new(platform: P) -> Result<Self, backenderror> {
        Ok(Self {
            inner: submission::withpersistence(platform, true)?,
        })
    }
}

impl<P: recordplatform> api::recorder for session<P> {
    fn recordmatmul(
        &mut self,
        lhs: operand<'_>,
        rhs: operand<'_>,
    ) -> Result<stepref, backenderror> {
        api::submission::recordmatmul(&mut self.inner, lhs, rhs)
    }

    fn recordadd(&mut self, lhs: operand<'_>, rhs: operand<'_>) -> Result<stepref, backenderror> {
        api::submission::recordadd(&mut self.inner, lhs, rhs)
    }

    fn recordmul(&mut self, lhs: operand<'_>, rhs: operand<'_>) -> Result<stepref, backenderror> {
        api::submission::recordmul(&mut self.inner, lhs, rhs)
    }

    fn recordrmsnorm(
        &mut self,
        input: operand<'_>,
        weight: operand<'_>,
        eps: f32,
    ) -> Result<stepref, backenderror> {
        api::submission::recordrmsnorm(&mut self.inner, input, weight, eps)
    }

    fn recordrmsnormmatmul(
        &mut self,
        input: operand<'_>,
        weight: operand<'_>,
        rhs: operand<'_>,
        eps: f32,
    ) -> Result<stepref, backenderror> {
        api::submission::recordrmsnormmatmul(&mut self.inner, input, weight, rhs, eps)
    }

    fn recordsoftmax(&mut self, input: operand<'_>) -> Result<stepref, backenderror> {
        api::submission::recordsoftmax(&mut self.inner, input)
    }

    fn recordsilu(&mut self, input: operand<'_>) -> Result<stepref, backenderror> {
        api::submission::recordsilu(&mut self.inner, input)
    }

    fn recordembedding(
        &mut self,
        table: operand<'_>,
        indices: &[usize],
    ) -> Result<stepref, backenderror> {
        api::submission::recordembedding(&mut self.inner, table, indices)
    }

    fn recordrope(
        &mut self,
        input: operand<'_>,
        heads: usize,
        base: f32,
    ) -> Result<stepref, backenderror> {
        api::submission::recordrope(&mut self.inner, input, heads, base)
    }

    fn recordrope_at(
        &mut self,
        input: operand<'_>,
        position: usize,
        heads: usize,
        base: f32,
    ) -> Result<stepref, backenderror> {
        api::submission::recordrope_at(&mut self.inner, input, position, heads, base)
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
        api::submission::recordattention(&mut self.inner, query, key, value, heads, kvheads, causal)
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
        api::submission::recordattention_cached(
            &mut self.inner,
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
        api::submission::recordscan(&mut self.inner, input, init, a, b, c, skip)
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
        api::submission::recordscan_cached(&mut self.inner, input, state, a, b, c, skip)
    }

    fn recordview(&mut self, base: stepref, shape: shape) -> Result<stepref, backenderror> {
        api::submission::recordview(&mut self.inner, base, shape)
    }

    fn recordappend(
        &mut self,
        cached: operand<'_>,
        new: operand<'_>,
        position: usize,
    ) -> Result<(), backenderror> {
        api::submission::recordappend(&mut self.inner, cached, new, position)
    }

    fn recordpersist(&mut self, data: &[f32], shape: shape) -> Result<stepref, backenderror> {
        api::submission::recordpersist(&mut self.inner, data, shape)
    }
}

impl<P: recordplatform> api::session for session<P> {
    fn apply(&mut self, outputs: &[stepref]) -> Result<Vec<tensor<f32>>, backenderror> {
        self.inner.apply_mut(outputs)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}
