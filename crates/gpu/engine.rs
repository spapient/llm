//! The shared execution engine: one `compute` implementation for all backends.
//!
//! Every backend performs the same orchestration — validate the budget, derive
//! the launch geometry, call the device primitive, rebuild the result tensor —
//! so it lives here once. A backend implements [`platform`] with only its
//! device primitives. Where an operator has a genuinely better native form,
//! the backend overrides that primitive: MUSA's fused `rmsnormmatmul` and
//! `scan_cached`, and its native `attention_cached` / `rope_at` kernels, stay
//! in `musa` and are selected through the same trait.
//!
//! Recorded execution follows the same split: [`recordplatform`] adds the
//! per-device recording primitives and [`submission`] implements
//! [`api::submission`] once. MUSA keeps its own submission because its
//! pinned-input and persistent-session state is a different resource model.

use ::tensor::{dtype, shape, tensor, tensorview};
use api::geometry::{
    ADDOP, MULOP, attentionshape, axisshape, binaryshape, embeddingshape, matmulshape, ropeshape,
};
use api::{
    backend, backenderror, capabilities, checkbudget, checkgatherbudget, compute, device, kvpair,
    operand, operandpair, stepref,
};
use lib::limits;

use std::fmt;

/// The device primitives one backend provides.
///
/// Every method is a direct hand-off to the backend's platform layer; none of
/// them validates shapes, which the engine has already done. A primitive that
/// a backend does not implement keeps the default and reports
/// `unsupportedoperation`.
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
    fn checkbudget(&self, s: &shape) -> Result<(), backenderror> {
        checkbudget(self.platform.capabilities(), s, core::mem::size_of::<f32>())
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
        self.checkbudget(lhs.getshape())?;
        self.checkbudget(rhs.getshape())?;
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
        self.checkbudget(lhs.getshape())?;
        self.checkbudget(rhs.getshape())?;
        let outshape = binaryshape(lhs.getshape(), rhs.getshape(), ADDOP)?;
        let data = self.platform.add(lhs.getdata(), rhs.getdata())?;
        wrap(P::NAME, outshape, data)
    }

    fn mul(
        &self,
        lhs: &tensorview<'_, f32>,
        rhs: &tensorview<'_, f32>,
    ) -> Result<tensor<f32>, backenderror> {
        self.checkbudget(lhs.getshape())?;
        self.checkbudget(rhs.getshape())?;
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
        self.checkbudget(input.getshape())?;
        self.checkbudget(weight.getshape())?;
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
        self.checkbudget(input.getshape())?;
        self.checkbudget(weight.getshape())?;
        self.checkbudget(rhs.getshape())?;
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
        self.checkbudget(query.getshape())?;
        self.checkbudget(key.getshape())?;
        self.checkbudget(value.getshape())?;
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
        self.checkbudget(input.getshape())?;
        self.checkbudget(init.getshape())?;
        self.checkbudget(a.getshape())?;
        self.checkbudget(b.getshape())?;
        self.checkbudget(c.getshape())?;
        self.checkbudget(skip.getshape())?;
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
        self.checkbudget(input.getshape())?;
        self.checkbudget(state.getshape())?;
        self.checkbudget(a.getshape())?;
        self.checkbudget(b.getshape())?;
        self.checkbudget(c.getshape())?;
        self.checkbudget(skip.getshape())?;
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
    let geometry = matmulshape::new(input, rhs)?;
    if geometry.k == 0 {
        return Err(backenderror::unsupportedoperation {
            op: "rmsnormmatmul",
        });
    }
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

/// The recording primitives one backend provides for [`submission`].
///
/// A backend implements this when its device model supports the recorded
/// execution form. MUSA keeps its own submission because its pinned-input and
/// persistent-session machinery is a different resource model; CUDA and Metal
/// share this one.
#[doc(hidden)]
pub trait recordplatform: platform {
    /// The device buffer handle an operand resolves to.
    type buffer: Clone;

    /// The per-submission recording state.
    type session;

    /// Whether caller inputs are deduplicated by `(pointer, length)` within
    /// one submission. Metal does this; CUDA uploads each reference.
    const DEDUP_INPUTS: bool = false;

    fn newsession(&self) -> Result<Self::session, backenderror>;

    fn upload(&self, session: &Self::session, data: &[f32]) -> Result<Self::buffer, backenderror>;

    fn getbuffer(&self, session: &Self::session, slot: usize) -> Option<Self::buffer>;

    fn gethostdata(&self, session: &Self::session, slot: usize) -> Option<Vec<f32>>;

    /// Synchronises and downloads every recorded slot, in slot order.
    ///
    /// The session stays usable: a persistent session records the next chain
    /// into the same device state.
    fn applysession(&self, session: &mut Self::session) -> Result<Vec<Vec<f32>>, backenderror>;

    /// Releases any device state the previous recorded chain left behind.
    fn resetsession(&self, _session: &mut Self::session) {}

    fn recordmatmul(
        &self,
        session: &mut Self::session,
        lhs: &Self::buffer,
        rhs: &Self::buffer,
        geometry: &matmulshape,
    ) -> Result<usize, backenderror>;

    fn recordadd(
        &self,
        session: &mut Self::session,
        lhs: &Self::buffer,
        rhs: &Self::buffer,
        count: usize,
    ) -> Result<usize, backenderror>;

    fn recordmul(
        &self,
        session: &mut Self::session,
        lhs: &Self::buffer,
        rhs: &Self::buffer,
        count: usize,
    ) -> Result<usize, backenderror>;

    fn recordrmsnorm(
        &self,
        session: &mut Self::session,
        input: &Self::buffer,
        weight: &Self::buffer,
        geometry: &axisshape,
        eps: f32,
    ) -> Result<usize, backenderror>;

    fn recordsoftmax(
        &self,
        session: &mut Self::session,
        input: &Self::buffer,
        geometry: &axisshape,
    ) -> Result<usize, backenderror>;

    fn recordsilu(
        &self,
        session: &mut Self::session,
        input: &Self::buffer,
        count: usize,
    ) -> Result<usize, backenderror>;

    fn recordembedding(
        &self,
        session: &mut Self::session,
        table: &Self::buffer,
        indices: &[usize],
        geometry: &embeddingshape,
    ) -> Result<usize, backenderror>;

    fn recordrope(
        &self,
        session: &mut Self::session,
        input: &Self::buffer,
        geometry: &ropeshape,
        base: f32,
    ) -> Result<usize, backenderror>;

    fn recordrope_at(
        &self,
        session: &mut Self::session,
        input: &Self::buffer,
        position: usize,
        geometry: &ropeshape,
        base: f32,
    ) -> Result<usize, backenderror>;

    fn recordattention(
        &self,
        session: &mut Self::session,
        query: &Self::buffer,
        key: &Self::buffer,
        value: &Self::buffer,
        geometry: &attentionshape,
        causal: bool,
    ) -> Result<usize, backenderror>;

    #[allow(clippy::too_many_arguments)]
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
    ) -> Result<usize, backenderror>;

    /// Fused RMSNorm+matmul in recorded form. Unsupported by default.
    fn recordrmsnormmatmul(
        &self,
        _session: &mut Self::session,
        _input: &Self::buffer,
        _weight: &Self::buffer,
        _rhs: &Self::buffer,
        _geometry: &matmulshape,
        _eps: f32,
    ) -> Result<usize, backenderror> {
        Err(backenderror::unsupportedoperation {
            op: "rmsnormmatmul",
        })
    }

    /// Recorded scan. Unsupported by default.
    #[allow(clippy::too_many_arguments)]
    fn recordscan(
        &self,
        _session: &mut Self::session,
        _input: &Self::buffer,
        _init: &Self::buffer,
        _a: &Self::buffer,
        _b: &Self::buffer,
        _c: &Self::buffer,
        _skip: &Self::buffer,
    ) -> Result<usize, backenderror> {
        Err(backenderror::unsupportedoperation { op: "scan" })
    }

    /// Recorded cached-state scan. Unsupported by default.
    #[allow(clippy::too_many_arguments)]
    fn recordscan_cached(
        &self,
        _session: &mut Self::session,
        _input: &Self::buffer,
        _state: &Self::buffer,
        _a: &Self::buffer,
        _b: &Self::buffer,
        _c: &Self::buffer,
        _skip: &Self::buffer,
    ) -> Result<usize, backenderror> {
        Err(backenderror::unsupportedoperation { op: "scan_cached" })
    }
}

/// A recorded submission over a platform: one synchronise, many kernels.
pub struct submission<P: recordplatform> {
    platform: P,
    session: P::session,
    results: Vec<(usize, shape)>,
    inputs: std::collections::HashMap<(usize, usize), P::buffer>,
    /// Whether caller inputs survive an apply (the persistent-session form).
    persistinputs: bool,
}

impl<P: recordplatform> submission<P> {
    pub(crate) fn new(platform: P) -> Result<Self, backenderror> {
        Self::withpersistence(platform, false)
    }

    fn withpersistence(platform: P, persistinputs: bool) -> Result<Self, backenderror> {
        let session = platform.newsession()?;
        Ok(Self {
            platform,
            session,
            results: Vec::new(),
            inputs: std::collections::HashMap::new(),
            persistinputs,
        })
    }

    fn checkbudget(&self, s: &shape) -> Result<(), backenderror> {
        checkbudget(self.platform.capabilities(), s, core::mem::size_of::<f32>())
    }

    fn checkgatherbudget(&self, rows: usize, width: usize) -> Result<(), backenderror> {
        checkgatherbudget(
            self.platform.capabilities(),
            rows,
            width,
            core::mem::size_of::<f32>(),
        )
    }

    /// Resolves an operand to its output shape, without touching the device.
    fn shapeof(&self, op: operand<'_>) -> Result<shape, backenderror> {
        match op {
            operand::input(view) => Ok(view.getshape().clone()),
            operand::result(reference) => {
                let index = reference.getindex();
                self.results
                    .get(index)
                    .map(|(_, outshape)| outshape.clone())
                    .ok_or(backenderror::indexoutofrange {
                        index,
                        len: self.results.len(),
                    })
            }
        }
    }

    /// Resolves an operand to a device buffer, uploading caller inputs and
    /// reusing or re-uploading recorded results in place.
    fn resolve(&mut self, op: operand<'_>) -> Result<P::buffer, backenderror> {
        match op {
            operand::input(view) => {
                let data = view.getdata();
                let key = (data.as_ptr() as usize, data.len());
                let dedup = self.persistinputs || P::DEDUP_INPUTS;
                if dedup && let Some(buffer) = self.inputs.get(&key) {
                    return Ok(buffer.clone());
                }
                let buffer = self.platform.upload(&self.session, data)?;
                if dedup {
                    self.inputs.insert(key, buffer.clone());
                }
                Ok(buffer)
            }
            operand::result(reference) => {
                let index = reference.getindex();
                let (slot, _) = self
                    .results
                    .get(index)
                    .ok_or(backenderror::indexoutofrange {
                        index,
                        len: self.results.len(),
                    })?;
                if let Some(buffer) = self.platform.getbuffer(&self.session, *slot) {
                    Ok(buffer)
                } else {
                    let data = self.platform.gethostdata(&self.session, *slot).ok_or(
                        backenderror::indexoutofrange {
                            index,
                            len: self.results.len(),
                        },
                    )?;
                    self.platform.upload(&self.session, &data)
                }
            }
        }
    }

    fn push(&mut self, slot: usize, outshape: shape) -> stepref {
        let index = self.results.len();
        self.results.push((slot, outshape));
        stepref::fromindex(index)
    }
}

impl<P: recordplatform> api::submission for submission<P> {
    fn recordmatmul(
        &mut self,
        lhs: operand<'_>,
        rhs: operand<'_>,
    ) -> Result<stepref, backenderror> {
        let lhsshape = self.shapeof(lhs)?;
        let rhsshape = self.shapeof(rhs)?;
        self.checkbudget(&lhsshape)?;
        self.checkbudget(&rhsshape)?;
        let geometry = matmulshape::new(&lhsshape, &rhsshape)?;
        let outshape = geometry.outshape()?;
        self.checkbudget(&outshape)?;
        let lhs = self.resolve(lhs)?;
        let rhs = self.resolve(rhs)?;
        let slot = self
            .platform
            .recordmatmul(&mut self.session, &lhs, &rhs, &geometry)?;
        Ok(self.push(slot, outshape))
    }

    fn recordadd(&mut self, lhs: operand<'_>, rhs: operand<'_>) -> Result<stepref, backenderror> {
        let lhsshape = self.shapeof(lhs)?;
        let rhsshape = self.shapeof(rhs)?;
        self.checkbudget(&lhsshape)?;
        self.checkbudget(&rhsshape)?;
        let outshape = binaryshape(&lhsshape, &rhsshape, ADDOP)?;
        let count = lhsshape
            .getelementcount()
            .map_err(|_| backenderror::sizeoverflow)?;
        let lhs = self.resolve(lhs)?;
        let rhs = self.resolve(rhs)?;
        let slot = self
            .platform
            .recordadd(&mut self.session, &lhs, &rhs, count)?;
        Ok(self.push(slot, outshape))
    }

    fn recordmul(&mut self, lhs: operand<'_>, rhs: operand<'_>) -> Result<stepref, backenderror> {
        let lhsshape = self.shapeof(lhs)?;
        let rhsshape = self.shapeof(rhs)?;
        self.checkbudget(&lhsshape)?;
        self.checkbudget(&rhsshape)?;
        let outshape = binaryshape(&lhsshape, &rhsshape, MULOP)?;
        let count = lhsshape
            .getelementcount()
            .map_err(|_| backenderror::sizeoverflow)?;
        let lhs = self.resolve(lhs)?;
        let rhs = self.resolve(rhs)?;
        let slot = self
            .platform
            .recordmul(&mut self.session, &lhs, &rhs, count)?;
        Ok(self.push(slot, outshape))
    }

    fn recordrmsnorm(
        &mut self,
        input: operand<'_>,
        weight: operand<'_>,
        eps: f32,
    ) -> Result<stepref, backenderror> {
        let inshape = self.shapeof(input)?;
        let weightshape = self.shapeof(weight)?;
        self.checkbudget(&inshape)?;
        self.checkbudget(&weightshape)?;
        let geometry = axisshape::rmsnorm(&inshape, &weightshape)?;
        let input = self.resolve(input)?;
        let weight = self.resolve(weight)?;
        let slot =
            self.platform
                .recordrmsnorm(&mut self.session, &input, &weight, &geometry, eps)?;
        Ok(self.push(slot, inshape))
    }

    fn recordrmsnormmatmul(
        &mut self,
        input: operand<'_>,
        weight: operand<'_>,
        rhs: operand<'_>,
        eps: f32,
    ) -> Result<stepref, backenderror> {
        let inshape = self.shapeof(input)?;
        let weightshape = self.shapeof(weight)?;
        let rhsshape = self.shapeof(rhs)?;
        self.checkbudget(&inshape)?;
        self.checkbudget(&weightshape)?;
        self.checkbudget(&rhsshape)?;
        let (geometry, outshape) = rmsnormmatmulshape(&inshape, &weightshape, &rhsshape)?;
        self.checkbudget(&outshape)?;
        let input = self.resolve(input)?;
        let weight = self.resolve(weight)?;
        let rhs = self.resolve(rhs)?;
        let slot = self.platform.recordrmsnormmatmul(
            &mut self.session,
            &input,
            &weight,
            &rhs,
            &geometry,
            eps,
        )?;
        Ok(self.push(slot, outshape))
    }

    fn recordsoftmax(&mut self, input: operand<'_>) -> Result<stepref, backenderror> {
        let inshape = self.shapeof(input)?;
        self.checkbudget(&inshape)?;
        let geometry = axisshape::softmax(&inshape)?;
        let input = self.resolve(input)?;
        let slot = self
            .platform
            .recordsoftmax(&mut self.session, &input, &geometry)?;
        Ok(self.push(slot, inshape))
    }

    fn recordsilu(&mut self, input: operand<'_>) -> Result<stepref, backenderror> {
        let inshape = self.shapeof(input)?;
        self.checkbudget(&inshape)?;
        let count = inshape
            .getelementcount()
            .map_err(|_| backenderror::sizeoverflow)?;
        let input = self.resolve(input)?;
        let slot = self.platform.recordsilu(&mut self.session, &input, count)?;
        Ok(self.push(slot, inshape))
    }

    fn recordembedding(
        &mut self,
        table: operand<'_>,
        indices: &[usize],
    ) -> Result<stepref, backenderror> {
        let tableshape = self.shapeof(table)?;
        self.checkbudget(&tableshape)?;
        let geometry = embeddingshape::new(&tableshape, indices)?;
        self.checkgatherbudget(geometry.rows, geometry.width)?;
        let outshape = embeddingoutshape(&tableshape, geometry.rows)?;
        let table = self.resolve(table)?;
        let slot = self
            .platform
            .recordembedding(&mut self.session, &table, indices, &geometry)?;
        Ok(self.push(slot, outshape))
    }

    fn recordrope(
        &mut self,
        input: operand<'_>,
        heads: usize,
        base: f32,
    ) -> Result<stepref, backenderror> {
        let inshape = self.shapeof(input)?;
        self.checkbudget(&inshape)?;
        let geometry = ropeshape::new(&inshape, heads)?;
        let input = self.resolve(input)?;
        let slot = self
            .platform
            .recordrope(&mut self.session, &input, &geometry, base)?;
        Ok(self.push(slot, inshape))
    }

    fn recordrope_at(
        &mut self,
        input: operand<'_>,
        position: usize,
        heads: usize,
        base: f32,
    ) -> Result<stepref, backenderror> {
        let inshape = self.shapeof(input)?;
        self.checkbudget(&inshape)?;
        let geometry = ropeshape::new(&inshape, heads)?;
        let input = self.resolve(input)?;
        let slot =
            self.platform
                .recordrope_at(&mut self.session, &input, position, &geometry, base)?;
        Ok(self.push(slot, inshape))
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
        let queryshape = self.shapeof(query)?;
        let keyshape = self.shapeof(key)?;
        let valueshape = self.shapeof(value)?;
        self.checkbudget(&queryshape)?;
        self.checkbudget(&keyshape)?;
        self.checkbudget(&valueshape)?;
        let geometry = attentionshape::new(
            &queryshape,
            &keyshape,
            &valueshape,
            heads,
            kvheads,
            P::MAXHEADDIM,
        )?;
        let query = self.resolve(query)?;
        let key = self.resolve(key)?;
        let value = self.resolve(value)?;
        let slot = self.platform.recordattention(
            &mut self.session,
            &query,
            &key,
            &value,
            &geometry,
            causal,
        )?;
        Ok(self.push(slot, queryshape))
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
        let queryshape = self.shapeof(query)?;
        let cachedkeyshape = self.shapeof(key.cached)?;
        let cachedvalueshape = self.shapeof(value.cached)?;
        let newkeyshape = self.shapeof(key.new)?;
        let newvalueshape = self.shapeof(value.new)?;
        self.checkbudget(&queryshape)?;
        self.checkbudget(&cachedkeyshape)?;
        self.checkbudget(&cachedvalueshape)?;
        self.checkbudget(&newkeyshape)?;
        self.checkbudget(&newvalueshape)?;
        let keyshape = concatshape(&cachedkeyshape, &newkeyshape)?;
        let valueshape = concatshape(&cachedvalueshape, &newvalueshape)?;
        let geometry = attentionshape::new(
            &queryshape,
            &keyshape,
            &valueshape,
            heads,
            kvheads,
            P::MAXHEADDIM,
        )?;
        let query = self.resolve(query)?;
        let cachedkey = self.resolve(key.cached)?;
        let cachedvalue = self.resolve(value.cached)?;
        let newkey = self.resolve(key.new)?;
        let newvalue = self.resolve(value.new)?;
        let slot = self.platform.recordattention_cached(
            &mut self.session,
            &query,
            &cachedkey,
            &newkey,
            &cachedvalue,
            &newvalue,
            &geometry,
            causal,
        )?;
        Ok(self.push(slot, queryshape))
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
        let inshape = self.shapeof(input)?;
        let initshape = self.shapeof(init)?;
        let ashape = self.shapeof(a)?;
        let bshape = self.shapeof(b)?;
        let cshape = self.shapeof(c)?;
        let skipshape = self.shapeof(skip)?;
        self.checkbudget(&inshape)?;
        self.checkbudget(&initshape)?;
        self.checkbudget(&ashape)?;
        self.checkbudget(&bshape)?;
        self.checkbudget(&cshape)?;
        self.checkbudget(&skipshape)?;
        scanshape(&inshape, &initshape, &ashape, &bshape, &cshape, &skipshape)?;
        let input = self.resolve(input)?;
        let init = self.resolve(init)?;
        let a = self.resolve(a)?;
        let b = self.resolve(b)?;
        let c = self.resolve(c)?;
        let skip = self.resolve(skip)?;
        let slot = self
            .platform
            .recordscan(&mut self.session, &input, &init, &a, &b, &c, &skip)?;
        Ok(self.push(slot, inshape))
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
        let inshape = self.shapeof(input)?;
        let stateshape = self.shapeof(state)?;
        let ashape = self.shapeof(a)?;
        let bshape = self.shapeof(b)?;
        let cshape = self.shapeof(c)?;
        let skipshape = self.shapeof(skip)?;
        self.checkbudget(&inshape)?;
        self.checkbudget(&stateshape)?;
        self.checkbudget(&ashape)?;
        self.checkbudget(&bshape)?;
        self.checkbudget(&cshape)?;
        self.checkbudget(&skipshape)?;
        scanshape(&inshape, &stateshape, &ashape, &bshape, &cshape, &skipshape)?;
        let input = self.resolve(input)?;
        let state = self.resolve(state)?;
        let a = self.resolve(a)?;
        let b = self.resolve(b)?;
        let c = self.resolve(c)?;
        let skip = self.resolve(skip)?;
        let slot = self.platform.recordscan_cached(
            &mut self.session,
            &input,
            &state,
            &a,
            &b,
            &c,
            &skip,
        )?;
        Ok(self.push(slot, inshape))
    }

    fn apply(self, outputs: &[stepref]) -> Result<Vec<tensor<f32>>, backenderror> {
        let mut this = self;
        this.apply_mut(outputs)
    }
}

impl<P: recordplatform> submission<P> {
    /// Downloads the requested outputs and starts a fresh recorded chain.
    fn apply_mut(&mut self, outputs: &[stepref]) -> Result<Vec<tensor<f32>>, backenderror> {
        let slots = self.platform.applysession(&mut self.session)?;
        let results = std::mem::take(&mut self.results);
        outputs
            .iter()
            .map(|reference| {
                let index = reference.getindex();
                let (slot, outshape) = results.get(index).ok_or(backenderror::indexoutofrange {
                    index,
                    len: results.len(),
                })?;
                let data = slots
                    .get(*slot)
                    .cloned()
                    .ok_or(backenderror::indexoutofrange {
                        index,
                        len: results.len(),
                    })?;
                wrap(P::NAME, outshape.clone(), data)
            })
            .collect()
    }

    fn reset(&mut self) {
        self.results.clear();
        self.inputs.clear();
        self.platform.resetsession(&mut self.session);
    }
}

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
}

impl<P: recordplatform> api::session for session<P> {
    fn apply(&mut self, outputs: &[stepref]) -> Result<Vec<tensor<f32>>, backenderror> {
        self.inner.apply_mut(outputs)
    }

    fn reset(&mut self) {
        self.inner.reset();
    }
}

/// The maximum tensor rank every GPU backend accepts.
pub const MAXRANK: usize = 8;

/// The device-specific half of a backend declaration.
#[doc(hidden)]
pub trait devicespec {
    /// The backend name used in diagnostics.
    const NAME: &'static str;

    /// The attention head-dimension bound this device admits.
    const MAXHEADDIM: usize;

    /// The device variant this backend reports.
    fn device() -> device;
}

/// The identity and capability declaration required by the [`backend`] contract.
///
/// It owns no execution logic and holds no request state; the executor enforces
/// the budget and drives the kernels.
pub struct gpubackend<D> {
    caps: capabilities,
    // `fn() -> D` keeps the device marker out of the value's auto traits.
    marker: core::marker::PhantomData<fn() -> D>,
}

impl<D> Clone for gpubackend<D> {
    fn clone(&self) -> Self {
        Self {
            caps: self.caps.clone(),
            marker: core::marker::PhantomData,
        }
    }
}

impl<D> core::fmt::Debug for gpubackend<D> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("gpubackend")
            .field("caps", &self.caps)
            .finish()
    }
}

impl<D: devicespec> gpubackend<D> {
    /// Creates a backend that declares `f32` support and `lim`.
    #[must_use]
    pub fn new(lim: limits) -> Self {
        Self {
            caps: capabilities {
                dtypes: vec![dtype::f32],
                maxrank: MAXRANK,
                limits: lim,
            },
            marker: core::marker::PhantomData,
        }
    }
}

impl<D: devicespec> backend for gpubackend<D> {
    fn getname(&self) -> &'static str {
        D::NAME
    }

    fn getdevice(&self) -> device {
        D::device()
    }

    fn getcapabilities(&self) -> &capabilities {
        &self.caps
    }
}

/// Failures owned by a GPU backend.
///
/// Each variant is constructed only by the backend that can produce it, so a
/// given target sees the other variants as unused.
#[allow(dead_code)]
#[derive(Debug)]
pub enum gpuerror {
    /// No usable GPU device could be opened.
    device {
        /// The driver-provided diagnostic.
        message: String,
    },

    /// No device is available on this system.
    nodevice,

    /// A device allocation failed.
    allocation,

    /// Kernel compilation failed.
    compile {
        /// The kernel entry point whose source failed to compile.
        kernel: String,
        /// The compiler-provided diagnostic.
        message: String,
    },

    /// Loading the compiled module or resolving a kernel failed.
    module {
        /// The kernel entry point that could not be loaded or resolved.
        kernel: String,
        /// The driver-provided diagnostic.
        message: String,
    },

    /// Compute pipeline creation failed.
    pipeline {
        /// The kernel entry point whose pipeline could not be created.
        kernel: String,
        /// The driver-provided diagnostic.
        message: String,
    },

    /// Copying data between host and device failed.
    transfer {
        /// The driver-provided diagnostic.
        message: String,
    },

    /// Kernel launch failed.
    launch {
        /// The driver-provided diagnostic.
        message: String,
    },

    /// A vendor BLAS call failed.
    blas {
        /// The library-provided diagnostic.
        message: String,
    },

    /// Kernel execution failed.
    execution {
        /// The driver-provided diagnostic.
        message: String,
    },

    /// A size computation exceeded the addressable range.
    sizeoverflow,
}

impl gpuerror {
    /// Maps this failure onto the frozen `api::backenderror`.
    ///
    /// `backend` names the backend in the diagnostic, so the shared mapping
    /// still identifies the device that failed.
    pub(crate) fn maptobackend(self, backend: &'static str) -> api::backenderror {
        use api::backenderror;
        match self {
            gpuerror::sizeoverflow => backenderror::sizeoverflow,
            gpuerror::device { .. } | gpuerror::nodevice => backenderror::devicenotavailable,
            gpuerror::compile { kernel, message }
            | gpuerror::module { kernel, message }
            | gpuerror::pipeline { kernel, message } => backenderror::kernelcompilefailed {
                backend,
                kernel,
                message,
            },
            gpuerror::transfer { message } => backenderror::executionfailed {
                backend,
                kernel: "transfer".into(),
                message,
            },
            gpuerror::launch { message } => backenderror::executionfailed {
                backend,
                kernel: "launch".into(),
                message,
            },
            gpuerror::blas { message } => backenderror::executionfailed {
                backend,
                kernel: "blas".into(),
                message,
            },
            gpuerror::allocation => backenderror::executionfailed {
                backend,
                kernel: "alloc".into(),
                message: "device allocation failed".into(),
            },
            gpuerror::execution { message } => backenderror::executionfailed {
                backend,
                kernel: "run".into(),
                message,
            },
        }
    }
}

impl fmt::Display for gpuerror {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::device { message } => write!(f, "no usable GPU device: {message}"),
            Self::nodevice => f.write_str("no GPU device is available"),
            Self::allocation => f.write_str("failed to allocate device memory"),
            Self::compile { kernel, message } => {
                write!(f, "kernel compilation failed ({kernel}): {message}")
            }
            Self::module { kernel, message } => {
                write!(f, "module load failed ({kernel}): {message}")
            }
            Self::pipeline { kernel, message } => {
                write!(f, "compute pipeline creation failed ({kernel}): {message}")
            }
            Self::transfer { message } => write!(f, "host/device transfer failed: {message}"),
            Self::launch { message } => write!(f, "kernel launch failed: {message}"),
            Self::blas { message } => write!(f, "BLAS call failed: {message}"),
            Self::execution { message } => write!(f, "kernel execution failed: {message}"),
            Self::sizeoverflow => f.write_str("size overflow"),
        }
    }
}

impl std::error::Error for gpuerror {}

/// Rebuilds a kernel result into a tensor of `outshape`.
///
/// `backend` names the backend in the failure diagnostic.
pub(crate) fn wrap(
    backend: &'static str,
    outshape: shape,
    data: Vec<f32>,
) -> Result<tensor<f32>, backenderror> {
    tensor::new(outshape, data).map_err(|e| backenderror::executionfailed {
        backend,
        kernel: "construct".into(),
        message: e.to_string(),
    })
}

/// Concatenates two flat buffers.
pub(crate) fn concatdata(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut data = Vec::with_capacity(a.len() + b.len());
    data.extend_from_slice(a);
    data.extend_from_slice(b);
    data
}

/// Concatenates two rank-two shapes along their sequence axis.
pub(crate) fn concatshape(a: &shape, b: &shape) -> Result<shape, backenderror> {
    let ad = a.getdims();
    let bd = b.getdims();
    if ad.len() != 2 || bd.len() != 2 || ad[1] != bd[1] {
        return Err(backenderror::shapemismatch { op: "attention" });
    }
    shape::new(vec![ad[0] + bd[0], ad[1]]).map_err(|_| backenderror::sizeoverflow)
}

/// Validates scan ranks, channel alignment, and state-width alignment,
/// returning `(n, d, statewidth)`.
///
/// The spectral precondition on `a` is a separate predicate ([`scanstable`]):
/// the single-operator path of all three backends applies it; `musa`'s
/// recorded path does not read operand data and therefore skips it.
pub(crate) fn scanshape(
    input: &shape,
    init: &shape,
    a: &shape,
    b: &shape,
    c: &shape,
    skip: &shape,
) -> Result<(usize, usize, usize), backenderror> {
    let id = input.getdims();
    let initd = init.getdims();
    let ad = a.getdims();
    let bd = b.getdims();
    let cd = c.getdims();
    let skipd = skip.getdims();
    if id.len() != 2
        || initd.len() != 2
        || ad.len() != 2
        || bd.len() != 2
        || cd.len() != 2
        || skipd.len() != 1
    {
        return Err(backenderror::shapemismatch { op: "scan" });
    }
    let d = id[1];
    if ad[0] != d || bd[0] != d || cd[0] != d || skipd[0] != d || initd[0] != d {
        return Err(backenderror::shapemismatch { op: "scan" });
    }
    let statewidth = initd[1];
    if ad[1] != statewidth || bd[1] != statewidth || cd[1] != statewidth {
        return Err(backenderror::shapemismatch { op: "scan" });
    }
    Ok((id[0], d, statewidth))
}

/// Enforces the scan's spectral precondition: every `a` entry is finite and
/// inside `[-1, 1]`.
pub(crate) fn scanstable(a: &tensorview<'_, f32>) -> Result<(), backenderror> {
    for &value in a.getdata() {
        if value.abs() > 1.0 || value.is_nan() {
            return Err(backenderror::invalidparameter { op: "scan" });
        }
    }
    Ok(())
}

/// Builds the gather output shape: `rows` replaces the table's first dimension.
pub(crate) fn embeddingoutshape(tableshape: &shape, rows: usize) -> Result<shape, backenderror> {
    let mut outdims = Vec::with_capacity(tableshape.getrank());
    outdims.push(rows);
    outdims.extend_from_slice(&tableshape.getdims()[1..]);
    shape::new(outdims).map_err(|_| backenderror::sizeoverflow)
}
