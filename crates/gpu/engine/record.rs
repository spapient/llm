use ::tensor::{shape, tensor};
use api::geometry::{
    ADDOP, MULOP, attentionshape, axisshape, binaryshape, embeddingshape, matmulshape, ropeshape,
};
use api::{backenderror, checkbudget, checkgatherbudget, operand, operandpair, stepref};

use super::platform;
use super::{concatshape, embeddingoutshape, rmsnormmatmulshape, scanshape, wrap};

/// The per-device recording primitives one backend provides, layered on [`platform`].
///
/// Recorded execution has the same split as direct execution: the device primitives
/// live in the backend, the orchestration — start, enqueue, commit, wait — lives here
/// once. MUSA keeps its own submission because its pinned-input and persistent-session
/// machinery is a different resource model; CUDA and Metal share this one.
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

    /// Registers caller-owned data as a persistent device-resident slot.
    ///
    /// The returned slot stays resolvable through later applies (the
    /// persistent-session form) or at least for the current apply (the
    /// single-use form), and `getbuffer` on it yields the registered data.
    /// Unsupported by default.
    fn recordpersist(
        &self,
        _session: &mut Self::session,
        _data: &[f32],
        _shape: &shape,
    ) -> Result<usize, backenderror> {
        Err(backenderror::unsupportedoperation {
            op: "recordpersist",
        })
    }

    /// Appends `new` into `cached` at row `position`, both device buffers.
    /// Unsupported by default.
    fn recordappend(
        &self,
        _session: &mut Self::session,
        _cached: &Self::buffer,
        _new: &Self::buffer,
        _position: usize,
    ) -> Result<(), backenderror> {
        Err(backenderror::unsupportedoperation { op: "recordappend" })
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
    /// Caller data registered through [`recordpersist`], keyed by
    /// `(pointer, length)` so a device KV cache that hands the same host
    /// buffer back every step reuses one device allocation.
    persistent: std::collections::HashMap<(usize, usize), usize>,
}

impl<P: recordplatform> submission<P> {
    pub(crate) fn new(platform: P) -> Result<Self, backenderror> {
        Self::withpersistence(platform, false)
    }

    pub(crate) fn withpersistence(platform: P, persistinputs: bool) -> Result<Self, backenderror> {
        let session = platform.newsession()?;
        Ok(Self {
            platform,
            session,
            results: Vec::new(),
            inputs: std::collections::HashMap::new(),
            persistinputs,
            persistent: std::collections::HashMap::new(),
        })
    }

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
        self.checkmany(&[&lhsshape, &rhsshape])?;
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
        self.checkmany(&[&lhsshape, &rhsshape])?;
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
        self.checkmany(&[&lhsshape, &rhsshape])?;
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
        self.checkmany(&[&inshape, &weightshape])?;
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
        self.checkmany(&[&inshape, &weightshape, &rhsshape])?;
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
        self.checkmany(&[&queryshape, &keyshape, &valueshape])?;
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
        self.checkmany(&[
            &queryshape,
            &cachedkeyshape,
            &cachedvalueshape,
            &newkeyshape,
            &newvalueshape,
        ])?;
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
        self.checkmany(&[&inshape, &initshape, &ashape, &bshape, &cshape, &skipshape])?;
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
        self.checkmany(&[&inshape, &stateshape, &ashape, &bshape, &cshape, &skipshape])?;
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

    fn recordpersist(&mut self, data: &[f32], shape: shape) -> Result<stepref, backenderror> {
        self.checkbudget(&shape)?;
        let key = (data.as_ptr() as usize, data.len());
        let slot = match self.persistent.get(&key) {
            Some(&slot) => slot,
            None => {
                let slot = self
                    .platform
                    .recordpersist(&mut self.session, data, &shape)?;
                self.persistent.insert(key, slot);
                slot
            }
        };
        Ok(self.push(slot, shape))
    }

    fn recordappend(
        &mut self,
        cached: operand<'_>,
        new: operand<'_>,
        position: usize,
    ) -> Result<(), backenderror> {
        let cached = self.resolve(cached)?;
        let new = self.resolve(new)?;
        self.platform
            .recordappend(&mut self.session, &cached, &new, position)
    }

    fn recordview(&mut self, base: stepref, shape: shape) -> Result<stepref, backenderror> {
        let index = base.getindex();
        let (slot, _) = self
            .results
            .get(index)
            .ok_or(backenderror::indexoutofrange {
                index,
                len: self.results.len(),
            })?;
        self.checkbudget(&shape)?;
        Ok(self.push(*slot, shape))
    }

    fn apply(self, outputs: &[stepref]) -> Result<Vec<tensor<f32>>, backenderror> {
        let mut this = self;
        this.apply_mut(outputs)
    }
}

impl<P: recordplatform> submission<P> {
    /// Downloads the requested outputs and starts a fresh recorded chain.
    pub(crate) fn apply_mut(
        &mut self,
        outputs: &[stepref],
    ) -> Result<Vec<tensor<f32>>, backenderror> {
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

    pub(crate) fn reset(&mut self) {
        // Teardown the platform first: a device-resident session may still
        // have kernels in flight over the retained input buffers. Its reset
        // drains the queue, so the handles cleared below are released only
        // after every kernel that could touch them has completed. Backends
        // whose record operations run to completion use the default no-op
        // `resetsession`, so the ordering is unobservable there.
        self.platform.resetsession(&mut self.session);
        self.results.clear();
        self.inputs.clear();
        self.persistent.clear();
    }
}
