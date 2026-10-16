use std::sync::Arc;

use ::tensor::{shape, tensor, tensorview};
use api::{
    backend, backenderror, checkbudget, checkgatherbudget, compute, kvpair, operand, operandpair,
    stepref, submission,
};
use lib::limits;

use api::geometry::{
    ADDOP, MULOP, attentionshape, axisshape, binaryshape, embeddingshape, matmulshape, ropeshape,
};

use crate::backend::{MUSAMAXHEADDIM, musabackend};
use crate::error::musaerror;
use crate::ffi::{self, buf};

/// A MUSA executor: the crate's [`compute`](api::compute) provider.
///
/// `musaexecutor` owns a [`musabackend`] and is the single place where its
/// declared rank and byte budget is enforced. Besides the budget it holds only
/// device state and build-time-compiled kernels (kept in `ffi.rs`), never a
/// request-scoped value, so one value can be shared immutably across requests.
///
/// The operator set is complete: all thirteen `api::compute` operators are
/// served by in-repo kernels compiled at build time by `mcc` — matmul by the
/// reference FMA kernel (the WMMA Tensor-Core kernel is a separate measurable
/// stage), softmax by the max-subtracting kernel, scan by the serial/parallel
/// pair. There is deliberately no vendor acceleration library and no runtime
/// JIT (docs/musa-backend-plan.md §5.3): build failures surface at build time,
/// not as a `kernelcompilefailed` at run time.
#[derive(Clone, Debug)]
pub struct musaexecutor {
    backend: musabackend,
    cache: Arc<ffi::cache>,
}

impl musaexecutor {
    /// Creates an executor over the MUSA backend with budget `lim`.
    ///
    /// The executor owns the device state and the compiled kernels: a clone
    /// shares them, while an independently constructed executor retains its own
    /// device handle and loads its own kernels. Nothing touches the device
    /// until the first operator call — `new` never fails on a host without a
    /// MUSA device.
    #[must_use]
    pub fn new(lim: limits) -> Self {
        Self {
            backend: musabackend::new(lim),
            cache: Arc::new(ffi::cache::new()),
        }
    }

    /// Returns how many kernels this executor has resolved so far.
    ///
    /// Kernels are loaded at run time from the embedded fatbins, once per
    /// `(source, entry)` pair, and reused by every later call; the count
    /// therefore rises to the number of distinct kernels a workload exercises
    /// and then stops. It is the observable that makes "loaded once" checkable
    /// from outside the crate rather than only by timing.
    #[must_use]
    pub fn getcachedkernelcount(&self) -> usize {
        self.cache.kernelcount()
    }

    /// Returns the backend whose budget this executor enforces.
    #[must_use]
    pub fn getbackend(&self) -> &musabackend {
        &self.backend
    }

    /// Validates a shape's rank, element count, and byte size against the
    /// backend budget.
    ///
    /// The predicate itself lives in `api` so that this crate, `metal`, and
    /// `cuda` cannot classify the same operand differently (docs/design.md
    /// §21.4).
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
    tensor::new(outshape, data).map_err(|e| backenderror::executionfailed {
        backend: "musa",
        kernel: "construct".into(),
        message: e.to_string(),
    })
}

/// Maps a crate-local [`musaerror`] onto the frozen `api::backenderror`.
///
/// Device absence and module loading are surfaced as their own variants so a
/// caller can tell "no MUSA device on this host" from "the fatbin could not be
/// loaded"; the kernel entry point rides along so a loading failure identifies
/// exactly which `(source, entry)` pair failed. Transfer and launch runtime
/// failures stay under the generic `executionfailed`. The compilation class is
/// deliberately absent: kernels are compiled at build time by `mcc`, so a
/// compilation failure is a build failure, never a run-time error
/// (docs/musa-backend-plan.md §5.3).
fn maptobackend(error: musaerror) -> backenderror {
    match error {
        musaerror::sizeoverflow => backenderror::sizeoverflow,
        musaerror::device { .. } => backenderror::devicenotavailable,
        musaerror::module { kernel, message } => backenderror::kernelcompilefailed {
            backend: "musa",
            kernel,
            message,
        },
        musaerror::transfer { message } => backenderror::executionfailed {
            backend: "musa",
            kernel: "transfer".into(),
            message,
        },
        musaerror::launch { message } => backenderror::executionfailed {
            backend: "musa",
            kernel: "launch".into(),
            message,
        },
    }
}

/// Converts the API's `usize` row indices to the kernel's `u64` gather indices.
fn tou64(indices: &[usize]) -> Vec<u64> {
    indices.iter().map(|index| *index as u64).collect()
}

impl compute for musaexecutor {
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

        let lhsdata = lhs.getdata();
        let rhsdata = rhs.getdata();
        let data = ffi::matmul(
            &self.cache,
            lhsdata,
            rhsdata,
            geometry.m,
            geometry.k,
            geometry.n,
            geometry.batch,
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
        let index64 = tou64(indices);
        let data = ffi::embedding(&self.cache, table.getdata(), &index64, geometry.width)
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

    fn rope_at(
        &self,
        input: &tensorview<'_, f32>,
        position: usize,
        heads: usize,
        base: f32,
    ) -> Result<tensor<f32>, backenderror> {
        self.checkbudget(input.getshape())?;
        let geometry = ropeshape::new(input.getshape(), heads)?;
        // The dedicated absolute-position kernel rotates the single token
        // directly, without materialising a padded history (contrast the
        // `cuda` executor's host-side pad-and-rotate).
        let data = ffi::rope_at(
            &self.cache,
            input.getdata(),
            position,
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
            MUSAMAXHEADDIM,
        )?;
        let data = ffi::attention(
            &self.cache,
            query.getdata(),
            key.getdata(),
            value.getdata(),
            geometry.qseq,
            geometry.kvseq,
            geometry.heads,
            geometry.kvheads,
            geometry.headdim,
            causal,
        )
        .map_err(maptobackend)?;
        wrap(query.getshape().clone(), data)
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
        // Validate the concatenated shapes without materialising them: the
        // kernel reads the cached rows followed by the new row from their own
        // buffers (docs/musa-backend-plan.md §5.3, contrast the `cuda`
        // executor's host-side concatenation).
        let keyshape = concatshape(key.cached.getshape(), key.new.getshape())?;
        let valueshape = concatshape(value.cached.getshape(), value.new.getshape())?;
        let geometry = attentionshape::new(
            query.getshape(),
            &keyshape,
            &valueshape,
            heads,
            kvheads,
            MUSAMAXHEADDIM,
        )?;
        let data = ffi::attention_cached(
            &self.cache,
            query.getdata(),
            key.cached.getdata(),
            key.new.getdata(),
            value.cached.getdata(),
            value.new.getdata(),
            geometry.kvseq,
            geometry.heads,
            geometry.kvheads,
            geometry.headdim,
            causal,
        )
        .map_err(maptobackend)?;
        wrap(query.getshape().clone(), data)
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
        let (n, d, statewidth) = scanshape(input, init, a, b, c, skip)?;
        let (out, state) = ffi::scan(
            &self.cache,
            input.getdata(),
            init.getdata(),
            a.getdata(),
            b.getdata(),
            c.getdata(),
            skip.getdata(),
            n,
            d,
            statewidth,
        )
        .map_err(maptobackend)?;
        let outshape = shape::new(vec![n, d]).map_err(|_| backenderror::sizeoverflow)?;
        let stateshape = shape::new(vec![d, statewidth]).map_err(|_| backenderror::sizeoverflow)?;
        Ok((wrap(outshape, out)?, wrap(stateshape, state)?))
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
        self.checkbudget(input.getshape())?;
        self.checkbudget(state.getshape())?;
        self.checkbudget(a.getshape())?;
        self.checkbudget(b.getshape())?;
        self.checkbudget(c.getshape())?;
        self.checkbudget(skip.getshape())?;
        let (n, d, statewidth) = scanshape(input, state, a, b, c, skip)?;
        if n != 1 {
            return Err(backenderror::shapemismatch { op: "scan_cached" });
        }
        // The decode step runs the serial kernel with n = 1, which is the
        // bitwise-equal continuation of the serial scan over the same prefix
        // (api::compute::scan_cached contract, M11).
        let (out, newstate) = ffi::scan_cached(
            &self.cache,
            input.getdata(),
            state.getdata(),
            a.getdata(),
            b.getdata(),
            c.getdata(),
            skip.getdata(),
            d,
            statewidth,
        )
        .map_err(maptobackend)?;
        let outshape = shape::new(vec![1, d]).map_err(|_| backenderror::sizeoverflow)?;
        let stateshape = shape::new(vec![d, statewidth]).map_err(|_| backenderror::sizeoverflow)?;
        Ok((wrap(outshape, out)?, wrap(stateshape, newstate)?))
    }
}

/// Concatenates two rank-two shapes along their sequence axis.
fn concatshape(a: &shape, b: &shape) -> Result<shape, backenderror> {
    let ad = a.getdims();
    let bd = b.getdims();
    if ad.len() != 2 || bd.len() != 2 || ad[1] != bd[1] {
        return Err(backenderror::shapemismatch { op: "attention" });
    }
    shape::new(vec![ad[0] + bd[0], ad[1]]).map_err(|_| backenderror::sizeoverflow)
}

/// Validates scan ranks, channel alignment, state-width alignment, and the
/// spectral precondition, returning `(n, d, statewidth)`.
fn scanshape(
    input: &tensorview<'_, f32>,
    init: &tensorview<'_, f32>,
    a: &tensorview<'_, f32>,
    b: &tensorview<'_, f32>,
    c: &tensorview<'_, f32>,
    skip: &tensorview<'_, f32>,
) -> Result<(usize, usize, usize), backenderror> {
    let id = input.getshape().getdims();
    let initd = init.getshape().getdims();
    let ad = a.getshape().getdims();
    let bd = b.getshape().getdims();
    let cd = c.getshape().getdims();
    let skipd = skip.getshape().getdims();
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
    for &value in a.getdata() {
        if value.abs() > 1.0 || value.is_nan() {
            return Err(backenderror::invalidparameter { op: "scan" });
        }
    }
    Ok((id[0], d, statewidth))
}

impl musaexecutor {
    /// Creates a single-use accumulative submission over this executor's
    /// backend and compiled-kernel cache.
    ///
    /// The MUSA submission records each step synchronously on the default
    /// stream and keeps results on the host, so construction never fails —
    /// nothing touches the device until the first record. (The device-buffer
    /// accumulation of the `cuda`/`metal` submissions is the C2 optimization
    /// path, a separate measurable stage.)
    #[must_use]
    pub fn newsubmission(&self) -> musasubmission {
        musasubmission::new(self.backend.clone(), self.cache.clone())
    }
}

/// A MUSA submission: the [`submission`] contract implemented over the same
/// executor-owned cache.
///
/// `musasubmission` owns the host-side results of its recorded steps and no
/// model weights; it shares no mutable state between requests. Each recorded
/// step resolves its operands to host data — copying caller inputs and reusing
/// recorded results — then runs the operator through the single `dispatch`
/// handoff and stores the result on the host. [`apply`](submission::apply)
/// rebuilds the requested outputs from those results.
///
/// A batched `matmul` (a leading batch dimension greater than one) is rejected,
/// mirroring `cuda`'s submission contract: the recorded path is one contraction
/// block, and joining blocks back into one tensor would re-introduce host round
/// trips, so it is left to the single-operator [`compute`] path.
pub struct musasubmission {
    backend: musabackend,
    cache: Arc<ffi::cache>,
    session: ffi::session,
    results: Vec<(resultdata, shape)>,
    /// Uploaded caller inputs keyed by `(data pointer, length)`, so an input
    /// operand referenced by several steps is uploaded exactly once per
    /// submission.
    inputs: std::collections::HashMap<(usize, usize), usize>,
}

/// A recorded result: either a device slot drained by [`session::apply`]
/// (the device-resident path) or host data (the operators not yet converted).
enum resultdata {
    slot(usize),
    host(Vec<f32>),
}

impl musasubmission {
    /// Creates a submission over `backend` using `cache`.
    pub(crate) fn new(backend: musabackend, cache: Arc<ffi::cache>) -> Self {
        let session = ffi::session::new(cache.clone());
        Self {
            backend,
            cache,
            session,
            results: Vec::new(),
            inputs: std::collections::HashMap::new(),
        }
    }

    fn checkbudget(&self, s: &shape) -> Result<(), backenderror> {
        checkbudget(
            self.backend.getcapabilities(),
            s,
            core::mem::size_of::<f32>(),
        )
    }

    fn checkgatherbudget(&self, rows: usize, width: usize) -> Result<(), backenderror> {
        checkgatherbudget(
            self.backend.getcapabilities(),
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

    /// Resolves an operand to a device slot for the device-resident path, or
    /// returns `None` when the operand is host data (an operator not yet
    /// converted): caller inputs are uploaded once per identity, recorded
    /// results already live in slots.
    fn resolveslot(&mut self, op: operand<'_>) -> Result<Option<usize>, backenderror> {
        match op {
            operand::input(view) => {
                let data = view.getdata();
                let key = (data.as_ptr() as usize, data.len());
                if let Some(&slot) = self.inputs.get(&key) {
                    return Ok(Some(slot));
                }
                let slot = self.session.upload(data).map_err(maptobackend)?;
                self.inputs.insert(key, slot);
                Ok(Some(slot))
            }
            operand::result(reference) => {
                let index = reference.getindex();
                match self.results.get(index) {
                    Some((resultdata::slot(slot), _)) => Ok(Some(*slot)),
                    Some((resultdata::host(_), _)) => Ok(None),
                    None => Err(backenderror::indexoutofrange {
                        index,
                        len: self.results.len(),
                    }),
                }
            }
        }
    }

    /// Resolves an operand to host data, for the operators still on the
    /// host-replay path. A device-resident result is drained mid-chain with a
    /// single synchronise, which is correct but reintroduces one sync.
    fn resolve(&mut self, op: operand<'_>) -> Result<Vec<f32>, backenderror> {
        match op {
            operand::input(view) => Ok(view.getdata().to_vec()),
            operand::result(reference) => {
                let index = reference.getindex();
                match self.results.get(index) {
                    Some((resultdata::host(data), _)) => Ok(data.clone()),
                    Some((resultdata::slot(slot), _)) => {
                        let slot = *slot;
                        self.session.take(slot).map_err(maptobackend)
                    }
                    None => Err(backenderror::indexoutofrange {
                        index,
                        len: self.results.len(),
                    }),
                }
            }
        }
    }

    /// Binds one operand for a device-resident record: a recorded slot binds
    /// directly, host data is uploaded once into a new slot, and a fresh
    /// caller input is uploaded once per identity.
    /// The returned bind never borrows host data: every operand is resolved
    /// to a device slot (uploaded on first use), so the `'static` lifetime
    /// parameter is a formality — `buf::device` carries a pointer only.
    fn operandsbuf(&mut self, op: operand<'_>) -> Result<buf<'static>, backenderror> {
        if let Some(slot) = self.resolveslot(op)? {
            return Ok(buf::device(self.session.slotptr(slot)));
        }
        let data = self.resolve(op)?;
        let slot = self.session.upload(&data).map_err(maptobackend)?;
        Ok(buf::device(self.session.slotptr(slot)))
    }

    fn pushslot(&mut self, slot: usize, outshape: shape) -> stepref {
        let index = self.results.len();
        self.results.push((resultdata::slot(slot), outshape));
        stepref::fromindex(index)
    }

    fn push(&mut self, data: Vec<f32>, outshape: shape) -> stepref {
        let index = self.results.len();
        self.results.push((resultdata::host(data), outshape));
        stepref::fromindex(index)
    }
}

impl submission for musasubmission {
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
        if geometry.batch != 1 {
            return Err(backenderror::unsupportedoperation { op: "matmul" });
        }
        let lhs = self.resolve(lhs)?;
        let rhs = self.resolve(rhs)?;
        let data = ffi::matmul(
            &self.cache,
            &lhs,
            &rhs,
            geometry.m,
            geometry.k,
            geometry.n,
            1,
        )
        .map_err(maptobackend)?;
        Ok(self.push(data, outshape))
    }

    fn recordadd(&mut self, lhs: operand<'_>, rhs: operand<'_>) -> Result<stepref, backenderror> {
        let lhsshape = self.shapeof(lhs)?;
        let rhsshape = self.shapeof(rhs)?;
        self.checkbudget(&lhsshape)?;
        self.checkbudget(&rhsshape)?;
        let outshape = binaryshape(&lhsshape, &rhsshape, ADDOP)?;
        let count = lhsshape.getelementcount().map_err(|_| backenderror::sizeoverflow)?;
        let lbuf = self.operandsbuf(lhs)?;
        let rbuf = self.operandsbuf(rhs)?;
        let slot = ffi::add_record(&mut self.session, lbuf, rbuf, count).map_err(maptobackend)?;
        Ok(self.pushslot(slot, outshape))
    }

    fn recordmul(&mut self, lhs: operand<'_>, rhs: operand<'_>) -> Result<stepref, backenderror> {
        let lhsshape = self.shapeof(lhs)?;
        let rhsshape = self.shapeof(rhs)?;
        self.checkbudget(&lhsshape)?;
        self.checkbudget(&rhsshape)?;
        let outshape = binaryshape(&lhsshape, &rhsshape, MULOP)?;
        let count = lhsshape.getelementcount().map_err(|_| backenderror::sizeoverflow)?;
        let lbuf = self.operandsbuf(lhs)?;
        let rbuf = self.operandsbuf(rhs)?;
        let slot = ffi::mul_record(&mut self.session, lbuf, rbuf, count).map_err(maptobackend)?;
        Ok(self.pushslot(slot, outshape))
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
        let ibuf = self.operandsbuf(input)?;
        let wbuf = self.operandsbuf(weight)?;
        let slot = ffi::rmsnorm_record(&mut self.session, ibuf, wbuf, geometry.rows, geometry.width, eps)
            .map_err(maptobackend)?;
        Ok(self.pushslot(slot, inshape))
    }

    fn recordsoftmax(&mut self, input: operand<'_>) -> Result<stepref, backenderror> {
        let inshape = self.shapeof(input)?;
        self.checkbudget(&inshape)?;
        let geometry = axisshape::softmax(&inshape)?;
        let ibuf = self.operandsbuf(input)?;
        let slot = ffi::softmax_record(&mut self.session, ibuf, geometry.rows, geometry.width)
            .map_err(maptobackend)?;
        Ok(self.pushslot(slot, inshape))
    }

    fn recordsilu(&mut self, input: operand<'_>) -> Result<stepref, backenderror> {
        let inshape = self.shapeof(input)?;
        self.checkbudget(&inshape)?;
        let count = inshape.getelementcount().map_err(|_| backenderror::sizeoverflow)?;
        let ibuf = self.operandsbuf(input)?;
        let slot = ffi::silu_record(&mut self.session, ibuf, count).map_err(maptobackend)?;
        Ok(self.pushslot(slot, inshape))
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
        let table = self.resolve(table)?;
        let index64 = tou64(indices);
        let data =
            ffi::embedding(&self.cache, &table, &index64, geometry.width).map_err(maptobackend)?;
        let mut outdims = Vec::with_capacity(tableshape.getrank());
        outdims.push(geometry.rows);
        outdims.extend_from_slice(&tableshape.getdims()[1..]);
        let outshape = shape::new(outdims).map_err(|_| backenderror::sizeoverflow)?;
        Ok(self.push(data, outshape))
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
        let data = ffi::rope(
            &self.cache,
            &input,
            geometry.rows,
            geometry.seqaxis,
            geometry.heads,
            geometry.headdim,
            base,
        )
        .map_err(maptobackend)?;
        Ok(self.push(data, inshape))
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
        let data = ffi::rope_at(
            &self.cache,
            &input,
            position,
            geometry.heads,
            geometry.headdim,
            base,
        )
        .map_err(maptobackend)?;
        Ok(self.push(data, inshape))
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
            MUSAMAXHEADDIM,
        )?;
        let query = self.resolve(query)?;
        let key = self.resolve(key)?;
        let value = self.resolve(value)?;
        let data = ffi::attention(
            &self.cache,
            &query,
            &key,
            &value,
            geometry.qseq,
            geometry.kvseq,
            geometry.heads,
            geometry.kvheads,
            geometry.headdim,
            causal,
        )
        .map_err(maptobackend)?;
        Ok(self.push(data, queryshape))
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
            MUSAMAXHEADDIM,
        )?;
        let query = self.resolve(query)?;
        let cachedkey = self.resolve(key.cached)?;
        let cachedvalue = self.resolve(value.cached)?;
        let newkey = self.resolve(key.new)?;
        let newvalue = self.resolve(value.new)?;
        let data = ffi::attention_cached(
            &self.cache,
            &query,
            &cachedkey,
            &newkey,
            &cachedvalue,
            &newvalue,
            geometry.kvseq,
            geometry.heads,
            geometry.kvheads,
            geometry.headdim,
            causal,
        )
        .map_err(maptobackend)?;
        Ok(self.push(data, queryshape))
    }

    fn recordscan(
        &mut self,
        _input: operand<'_>,
        _init: operand<'_>,
        _a: operand<'_>,
        _b: operand<'_>,
        _c: operand<'_>,
        _skip: operand<'_>,
    ) -> Result<stepref, backenderror> {
        Err(backenderror::unsupportedoperation { op: "scan" })
    }

    fn recordscan_cached(
        &mut self,
        _input: operand<'_>,
        _state: operand<'_>,
        _a: operand<'_>,
        _b: operand<'_>,
        _c: operand<'_>,
        _skip: operand<'_>,
    ) -> Result<stepref, backenderror> {
        Err(backenderror::unsupportedoperation { op: "scan_cached" })
    }

    fn apply(mut self, outputs: &[stepref]) -> Result<Vec<tensor<f32>>, backenderror> {
        // Download every requested device slot in one synchronise, then
        // rebuild the outputs in request order.
        let mut slotindices: Vec<usize> = Vec::new();
        for reference in outputs {
            let index = reference.getindex();
            match self.results.get(index) {
                Some((resultdata::slot(slot), _)) => slotindices.push(*slot),
                Some((resultdata::host(_), _)) => {}
                None => {
                    return Err(backenderror::indexoutofrange {
                        index,
                        len: self.results.len(),
                    });
                }
            }
        }
        let downloaded = self.session.apply(&slotindices).map_err(maptobackend)?;
        let mut cursor = 0;
        outputs
            .iter()
            .map(|reference| {
                let index = reference.getindex();
                let (data, outshape) = self.results.get(index).ok_or(
                    backenderror::indexoutofrange {
                        index,
                        len: self.results.len(),
                    },
                )?;
                let data = match data {
                    resultdata::host(data) => data.clone(),
                    resultdata::slot(_) => {
                        let data = downloaded[cursor].clone();
                        cursor += 1;
                        data
                    }
                };
                wrap(outshape.clone(), data)
            })
            .collect()
    }
}
