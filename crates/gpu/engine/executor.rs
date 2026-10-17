use ::tensor::{shape, tensor, tensorview};
use api::geometry::{
    ADDOP, MULOP, attentionshape, axisshape, binaryshape, embeddingshape, matmulshape, ropeshape,
};
use api::{backenderror, capabilities, checkbudget, checkgatherbudget, compute, kvpair};

use super::{concatdata, concatshape, embeddingoutshape, scanshape, scanstable, wrap};

#[doc(hidden)]
pub trait platform: Clone + core::fmt::Debug {
    /// The backend name used in diagnostics.
    const NAME: &'static str;

    /// The attention head-dimension bound this device admits.
    const MAXHEADDIM: usize;

    /// Whether the fused `rmsnormmatmul` primitive is implemented.
    const RMSNORMMATMUL: bool = false;

    /// Whether the cached-state `scan_cached` primitive is implemented.
    const SCAN_CACHED: bool = false;

    fn capabilities(&self) -> &capabilities;

    fn matmul(
        &self,
        lhs: &[f32],
        rhs: &[f32],
        geometry: &matmulshape,
    ) -> Result<Vec<f32>, backenderror>;

    fn add(&self, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, backenderror>;
    fn mul(&self, lhs: &[f32], rhs: &[f32]) -> Result<Vec<f32>, backenderror>;

    fn rmsnorm(
        &self,
        input: &[f32],
        weight: &[f32],
        geometry: &axisshape,
        eps: f32,
    ) -> Result<Vec<f32>, backenderror>;

    fn softmax(&self, input: &[f32], geometry: &axisshape) -> Result<Vec<f32>, backenderror>;

    fn silu(&self, input: &[f32]) -> Result<Vec<f32>, backenderror>;

    fn embedding(
        &self,
        table: &[f32],
        indices: &[usize],
        geometry: &embeddingshape,
    ) -> Result<Vec<f32>, backenderror>;

    fn rope(
        &self,
        input: &[f32],
        geometry: &ropeshape,
        base: f32,
    ) -> Result<Vec<f32>, backenderror>;

    fn rope_at(
        &self,
        input: &[f32],
        position: usize,
        geometry: &ropeshape,
        base: f32,
    ) -> Result<Vec<f32>, backenderror>;

    fn attention(
        &self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        geometry: &attentionshape,
        causal: bool,
    ) -> Result<Vec<f32>, backenderror>;

    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<(Vec<f32>, Vec<f32>), backenderror>;

    /// Fused RMSNorm+matmul. The default reports the operator as unsupported.
    #[allow(clippy::too_many_arguments)]
    fn rmsnormmatmul(
        &self,
        _input: &[f32],
        _weight: &[f32],
        _rhs: &[f32],
        _geometry: &matmulshape,
        _eps: f32,
    ) -> Result<Vec<f32>, backenderror> {
        Err(backenderror::unsupportedoperation {
            op: "rmsnormmatmul",
        })
    }

    /// Incremental-decode attention. The default concatenates the cached and
    /// new rows on the host and reuses [`platform::attention`]; a backend with
    /// a native cached kernel overrides it.
    fn attention_cached(
        &self,
        query: &[f32],
        key: &kvpair<'_>,
        value: &kvpair<'_>,
        geometry: &attentionshape,
        causal: bool,
    ) -> Result<Vec<f32>, backenderror> {
        let keys = concatdata(key.cached.getdata(), key.new.getdata());
        let values = concatdata(value.cached.getdata(), value.new.getdata());
        self.attention(query, &keys, &values, geometry, causal)
    }

    /// Cached-state scan. The default reports the operator as unsupported.
    #[allow(clippy::too_many_arguments)]
    fn scan_cached(
        &self,
        _input: &[f32],
        _state: &[f32],
        _a: &[f32],
        _b: &[f32],
        _c: &[f32],
        _skip: &[f32],
        _d: usize,
        _statewidth: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), backenderror> {
        Err(backenderror::unsupportedoperation { op: "scan_cached" })
    }
}

/// A backend executor: the [`compute`] provider built once over a platform.
#[derive(Clone, Debug)]
pub struct executor<P> {
    platform: P,
}

impl<P> executor<P> {
    pub(crate) fn newplatform(platform: P) -> Self {
        Self { platform }
    }

    pub(crate) fn getplatform(&self) -> &P {
        &self.platform
    }
}

impl<P: platform> executor<P> {
    /// Checks the budget of one shape.
    fn checkbudget(&self, s: &shape) -> Result<(), backenderror> {
        checkbudget(self.platform.capabilities(), s, core::mem::size_of::<f32>())
    }

    /// Checks the budget of every shape at once.
    fn checkmany(&self, shapes: &[&shape]) -> Result<(), backenderror> {
        for s in shapes {
            self.checkbudget(s)?;
        }
        Ok(())
    }

    fn checkgatherbudget(&self, rows: usize, width: usize) -> Result<(), backenderror> {
        checkgatherbudget(
            self.platform.capabilities(),
            rows,
            width,
            core::mem::size_of::<f32>(),
        )
    }
}

impl<P: platform> compute for executor<P> {
    fn matmul(
        &self,
        lhs: &tensorview<'_, f32>,
        rhs: &tensorview<'_, f32>,
    ) -> Result<tensor<f32>, backenderror> {
        self.checkmany(&[lhs.getshape(), rhs.getshape()])?;
        let geometry = matmulshape::new(lhs.getshape(), rhs.getshape())?;
        let outshape = geometry.outshape()?;
        self.checkbudget(&outshape)?;
        let data = self
            .platform
            .matmul(lhs.getdata(), rhs.getdata(), &geometry)?;
        wrap(P::NAME, outshape, data)
    }

    fn add(
        &self,
        lhs: &tensorview<'_, f32>,
        rhs: &tensorview<'_, f32>,
    ) -> Result<tensor<f32>, backenderror> {
        self.checkmany(&[lhs.getshape(), rhs.getshape()])?;
        let outshape = binaryshape(lhs.getshape(), rhs.getshape(), ADDOP)?;
        let data = self.platform.add(lhs.getdata(), rhs.getdata())?;
        wrap(P::NAME, outshape, data)
    }

    fn mul(
        &self,
        lhs: &tensorview<'_, f32>,
        rhs: &tensorview<'_, f32>,
    ) -> Result<tensor<f32>, backenderror> {
        self.checkmany(&[lhs.getshape(), rhs.getshape()])?;
        let outshape = binaryshape(lhs.getshape(), rhs.getshape(), MULOP)?;
        let data = self.platform.mul(lhs.getdata(), rhs.getdata())?;
        wrap(P::NAME, outshape, data)
    }

    fn rmsnorm(
        &self,
        input: &tensorview<'_, f32>,
        weight: &tensorview<'_, f32>,
        eps: f32,
    ) -> Result<tensor<f32>, backenderror> {
        self.checkmany(&[input.getshape(), weight.getshape()])?;
        let geometry = axisshape::rmsnorm(input.getshape(), weight.getshape())?;
        let data = self
            .platform
            .rmsnorm(input.getdata(), weight.getdata(), &geometry, eps)?;
        wrap(P::NAME, input.getshape().clone(), data)
    }

    fn rmsnormmatmul(
        &self,
        input: &tensorview<'_, f32>,
        weight: &tensorview<'_, f32>,
        rhs: &tensorview<'_, f32>,
        eps: f32,
    ) -> Result<tensor<f32>, backenderror> {
        if !P::RMSNORMMATMUL {
            return Err(backenderror::unsupportedoperation {
                op: "rmsnormmatmul",
            });
        }
        self.checkmany(&[input.getshape(), weight.getshape(), rhs.getshape()])?;
        let (geometry, outshape) =
            rmsnormmatmulshape(input.getshape(), weight.getshape(), rhs.getshape())?;
        self.checkbudget(&outshape)?;
        let data = self.platform.rmsnormmatmul(
            input.getdata(),
            weight.getdata(),
            rhs.getdata(),
            &geometry,
            eps,
        )?;
        wrap(P::NAME, outshape, data)
    }

    fn softmax(&self, input: &tensorview<'_, f32>) -> Result<tensor<f32>, backenderror> {
        self.checkbudget(input.getshape())?;
        let geometry = axisshape::softmax(input.getshape())?;
        let data = self.platform.softmax(input.getdata(), &geometry)?;
        wrap(P::NAME, input.getshape().clone(), data)
    }

    fn silu(&self, input: &tensorview<'_, f32>) -> Result<tensor<f32>, backenderror> {
        self.checkbudget(input.getshape())?;
        let data = self.platform.silu(input.getdata())?;
        wrap(P::NAME, input.getshape().clone(), data)
    }

    fn embedding(
        &self,
        table: &tensorview<'_, f32>,
        indices: &[usize],
    ) -> Result<tensor<f32>, backenderror> {
        self.checkbudget(table.getshape())?;
        let geometry = embeddingshape::new(table.getshape(), indices)?;
        self.checkgatherbudget(geometry.rows, geometry.width)?;
        let data = self
            .platform
            .embedding(table.getdata(), indices, &geometry)?;
        let outshape = embeddingoutshape(table.getshape(), geometry.rows)?;
        wrap(P::NAME, outshape, data)
    }

    fn rope(
        &self,
        input: &tensorview<'_, f32>,
        heads: usize,
        base: f32,
    ) -> Result<tensor<f32>, backenderror> {
        self.checkbudget(input.getshape())?;
        let geometry = ropeshape::new(input.getshape(), heads)?;
        let data = self.platform.rope(input.getdata(), &geometry, base)?;
        wrap(P::NAME, input.getshape().clone(), data)
    }

    fn rope_at(
        &self,
        input: &tensorview<'_, f32>,
        position: usize,
        heads: usize,
        base: f32,
    ) -> Result<tensor<f32>, backenderror> {
        self.checkbudget(input.getshape())?;
        let geometry = ropeshape::new(input.getshape(), heads)?;
        let data = self
            .platform
            .rope_at(input.getdata(), position, &geometry, base)?;
        wrap(P::NAME, input.getshape().clone(), data)
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
        self.checkmany(&[query.getshape(), key.getshape(), value.getshape()])?;
        let geometry = attentionshape::new(
            query.getshape(),
            key.getshape(),
            value.getshape(),
            heads,
            kvheads,
            P::MAXHEADDIM,
        )?;
        let data = self.platform.attention(
            query.getdata(),
            key.getdata(),
            value.getdata(),
            &geometry,
            causal,
        )?;
        wrap(P::NAME, query.getshape().clone(), data)
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
        self.checkbudget(query.getshape())?;
        self.checkbudget(key.cached.getshape())?;
        self.checkbudget(key.new.getshape())?;
        self.checkbudget(value.cached.getshape())?;
        self.checkbudget(value.new.getshape())?;
        let keyshape = concatshape(key.cached.getshape(), key.new.getshape())?;
        let valueshape = concatshape(value.cached.getshape(), value.new.getshape())?;
        let geometry = attentionshape::new(
            query.getshape(),
            &keyshape,
            &valueshape,
            heads,
            kvheads,
            P::MAXHEADDIM,
        )?;
        let data =
            self.platform
                .attention_cached(query.getdata(), &key, &value, &geometry, causal)?;
        wrap(P::NAME, query.getshape().clone(), data)
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
        self.checkmany(&[
            input.getshape(),
            init.getshape(),
            a.getshape(),
            b.getshape(),
            c.getshape(),
            skip.getshape(),
        ])?;
        let (n, d, statewidth) = scanshape(
            input.getshape(),
            init.getshape(),
            a.getshape(),
            b.getshape(),
            c.getshape(),
            skip.getshape(),
        )?;
        scanstable(a)?;
        let (out, state) = self.platform.scan(
            input.getdata(),
            init.getdata(),
            a.getdata(),
            b.getdata(),
            c.getdata(),
            skip.getdata(),
            n,
            d,
            statewidth,
        )?;
        let outshape = shape::new(vec![n, d]).map_err(|_| backenderror::sizeoverflow)?;
        let stateshape = shape::new(vec![d, statewidth]).map_err(|_| backenderror::sizeoverflow)?;
        Ok((
            wrap(P::NAME, outshape, out)?,
            wrap(P::NAME, stateshape, state)?,
        ))
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
        if !P::SCAN_CACHED {
            return Err(backenderror::unsupportedoperation { op: "scan_cached" });
        }
        self.checkmany(&[
            input.getshape(),
            state.getshape(),
            a.getshape(),
            b.getshape(),
            c.getshape(),
            skip.getshape(),
        ])?;
        let (n, d, statewidth) = scanshape(
            input.getshape(),
            state.getshape(),
            a.getshape(),
            b.getshape(),
            c.getshape(),
            skip.getshape(),
        )?;
        scanstable(a)?;
        if n != 1 {
            return Err(backenderror::shapemismatch { op: "scan_cached" });
        }
        let (out, newstate) = self.platform.scan_cached(
            input.getdata(),
            state.getdata(),
            a.getdata(),
            b.getdata(),
            c.getdata(),
            skip.getdata(),
            d,
            statewidth,
        )?;
        let outshape = shape::new(vec![1, d]).map_err(|_| backenderror::sizeoverflow)?;
        let stateshape = shape::new(vec![d, statewidth]).map_err(|_| backenderror::sizeoverflow)?;
        Ok((
            wrap(P::NAME, outshape, out)?,
            wrap(P::NAME, stateshape, newstate)?,
        ))
    }
}

/// Validates the fused RMSNorm+matmul shapes and returns the multiplication
/// plan and output shape.
pub(crate) fn rmsnormmatmulshape(
    input: &shape,
    weight: &shape,
    rhs: &shape,
) -> Result<(matmulshape, shape), backenderror> {
    // A zero contraction axis leaves the RMS denominator empty, so the fused
    // operator is unsupported. This must precede `matmulshape::new`: with k=0
    // the inner dimension also disagrees with the rhs's leading axis, and the
    // multiplication plan would otherwise report a vacuous `matmul` shape
    // error. This matches the CPU reference ordering (crates/cpu executor).
    let k = match input.getdims().last() {
        Some(&last) => last,
        None => {
            return Err(backenderror::shapemismatch {
                op: "rmsnormmatmul",
            });
        }
    };
    if k == 0 {
        return Err(backenderror::unsupportedoperation {
            op: "rmsnormmatmul",
        });
    }
    let geometry = matmulshape::new(input, rhs)?;
    let weightcount = weight
        .getelementcount()
        .map_err(|_| backenderror::sizeoverflow)?;
    if weightcount != geometry.k {
        return Err(backenderror::shapemismatch {
            op: "rmsnormmatmul",
        });
    }
    let outshape = geometry.outshape()?;
    Ok((geometry, outshape))
}
