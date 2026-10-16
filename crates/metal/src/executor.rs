use std::sync::Arc;

use ::tensor::{shape, tensor, tensorview};
use api::{
    backend, backenderror, checkbudget, checkgatherbudget, compute, kvpair, operand, operandpair,
    stepref, submission,
};
use lib::limits;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;

use api::geometry::{
    ADDOP, MULOP, attentionshape, axisshape, binaryshape, embeddingshape, matmulshape, ropeshape,
};

use crate::backend::{METALMAXHEADDIM, metalbackend};
use crate::error::metalerror;
use crate::ffi;

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
    cache: Arc<ffi::cache>,
}

impl metalexecutor {
    /// Creates an executor over a Metal backend configured with `lim`.
    ///
    /// The executor owns the device state and the compiled pipelines: a clone
    /// shares them, while an independently constructed executor compiles its own
    /// (docs/design.md §21.6).
    #[must_use]
    pub fn new(lim: limits) -> Self {
        Self {
            backend: metalbackend::new(lim),
            cache: Arc::new(ffi::cache::new()),
        }
    }

    /// Returns the backend whose budget this executor enforces.
    #[must_use]
    pub fn getbackend(&self) -> &metalbackend {
        &self.backend
    }

    /// Returns how many compute pipelines this executor has compiled so far.
    ///
    /// The MSL sources are compiled at run time, once per `(source, entry)`
    /// pair, and reused by every later call; the count therefore rises to the
    /// number of distinct kernels a workload exercises and then stops. It is the
    /// observable that makes "compiled once" checkable from outside the crate
    /// rather than only by timing.
    #[must_use]
    pub fn getcachedpipelinecount(&self) -> usize {
        self.cache.pipelinecount()
    }

    /// Validates a shape's rank, element count, and byte size against the
    /// backend budget.
    ///
    /// The predicate itself lives in `api` so that this crate and `cuda` cannot
    /// classify the same operand differently (docs/design.md §21.4).
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
        backend: "metal",
        kernel: "construct".into(),
        message: e.to_string(),
    })
}

/// Maps a crate-local [`metalerror`] onto the frozen `api::backenderror`.
///
/// Device absence and kernel compilation are surfaced as their own variants so
/// a caller can tell "no GPU on this host" from "the kernel did not compile";
/// the kernel entry point rides along so a compile failure identifies exactly
/// which `(source, entry)` pair failed. Allocation and runtime execution
/// failures stay under the generic `executionfailed` because no caller action
/// distinguishes them.
fn maptobackend(error: metalerror) -> backenderror {
    match error {
        metalerror::sizeoverflow => backenderror::sizeoverflow,
        metalerror::nodevice => backenderror::devicenotavailable,
        metalerror::compile { kernel, message } | metalerror::pipeline { kernel, message } => {
            backenderror::kernelcompilefailed {
                backend: "metal",
                kernel,
                message,
            }
        }
        metalerror::allocation => backenderror::executionfailed {
            backend: "metal",
            kernel: "alloc".into(),
            message: "device allocation failed".into(),
        },
        metalerror::execution { message } => backenderror::executionfailed {
            backend: "metal",
            kernel: "run".into(),
            message,
        },
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
        let geometry = matmulshape::new(lhs.getshape(), rhs.getshape())?;
        // The result is allocated by this crate, so the budget covers it too:
        // the check runs before the host buffer and any device buffer exist.
        let outshape = geometry.outshape()?;
        self.checkbudget(&outshape)?;
        // One dispatch covers every batch block, so the operands are handed
        // over whole rather than sliced per block: the block index travels in
        // the grid (docs/design.md §21.10.2).
        let data = ffi::matmul(
            &self.cache,
            lhs.getdata(),
            rhs.getdata(),
            geometry.batch,
            geometry.m,
            geometry.k,
            geometry.n,
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

    fn rope_at(
        &self,
        input: &tensorview<'_, f32>,
        position: usize,
        heads: usize,
        base: f32,
    ) -> Result<tensor<f32>, backenderror> {
        self.checkbudget(input.getshape())?;
        let geometry = ropeshape::new(input.getshape(), heads)?;
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
            METALMAXHEADDIM,
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
        _input: &tensorview<'_, f32>,
        _state: &tensorview<'_, f32>,
        _a: &tensorview<'_, f32>,
        _b: &tensorview<'_, f32>,
        _c: &tensorview<'_, f32>,
        _skip: &tensorview<'_, f32>,
    ) -> Result<(tensor<f32>, tensor<f32>), backenderror> {
        Err(backenderror::unsupportedoperation { op: "scan_cached" })
    }
}

/// Concatenates two `[_, width]` views into one `[a + b, width]` tensor.
fn concat(a: &tensorview<'_, f32>, b: &tensorview<'_, f32>) -> Result<tensor<f32>, backenderror> {
    let width = a.getshape().getdims()[1];
    let mut data = a.getdata().to_vec();
    data.extend_from_slice(b.getdata());
    let len = data.len() / width;
    let outshape = shape::new(vec![len, width]).map_err(|_| backenderror::sizeoverflow)?;
    tensor::new(outshape, data).map_err(|e| backenderror::executionfailed {
        backend: "metal",
        kernel: "construct".into(),
        message: e.to_string(),
    })
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

/// Concatenates two rank-two shapes along their sequence axis.
fn concatshape(a: &shape, b: &shape) -> Result<shape, backenderror> {
    let ad = a.getdims();
    let bd = b.getdims();
    if ad.len() != 2 || bd.len() != 2 || ad[1] != bd[1] {
        return Err(backenderror::shapemismatch { op: "attention" });
    }
    shape::new(vec![ad[0] + bd[0], ad[1]]).map_err(|_| backenderror::sizeoverflow)
}

impl metalexecutor {
    /// Creates a single-use accumulative submission over this executor's backend
    /// and compiled-pipeline cache.
    ///
    /// # Errors
    ///
    /// Returns `metalerror::nodevice` when no Metal device is available, the
    /// same first-call condition as any operator dispatch.
    pub fn newsubmission(&self) -> Result<metalsubmission, metalerror> {
        metalsubmission::new(self.backend.clone(), self.cache.clone())
    }
}

/// A Metal submission: one command buffer and encoder shared by every recorded
/// step, so a whole operator chain submits and waits once (docs/design.md §21.13).
///
/// `metalsubmission` is the Metal implementation of [`submission`]. It owns its
/// command buffer and encoder (through `ffi::session`) and the device buffers
/// of its intermediate results; it owns no model weights and shares no mutable
/// state between requests. Compiled pipelines still come from the executor-owned
/// cache, which holds no request-scoped value.
///
/// Each recorded step resolves its operands to device buffers — uploading
/// caller inputs and reusing earlier results in place — then encodes its dispatch
/// into the shared encoder. [`apply`](submission::apply) submits once, waits, and
/// downloads only the requested outputs.
pub struct metalsubmission {
    backend: metalbackend,
    cache: Arc<ffi::cache>,
    session: ffi::session,
    results: Vec<(usize, shape)>,
    /// Uploaded caller inputs keyed by `(data pointer, length)`, so an input
    /// operand referenced by several steps is allocated, copied, and uploaded
    /// exactly once per submission. Operands are immutable views, so identical
    /// identity means identical bytes.
    inputs: std::collections::HashMap<(usize, usize), Retained<ProtocolObject<dyn MTLBuffer>>>,
}

impl metalsubmission {
    /// Creates a submission over `backend` using `cache` for compiled pipelines.
    ///
    /// # Errors
    ///
    /// Returns [`metalerror::nodevice`] when no Metal device is available.
    pub(crate) fn new(backend: metalbackend, cache: Arc<ffi::cache>) -> Result<Self, metalerror> {
        let session = ffi::session::new(&cache)?;
        Ok(Self {
            backend,
            cache,
            session,
            results: Vec::new(),
            inputs: std::collections::HashMap::new(),
        })
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

    /// Resolves an operand to a device buffer, uploading caller inputs and
    /// reusing or re-uploading recorded results in place.
    fn resolve(
        &mut self,
        op: operand<'_>,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, backenderror> {
        match op {
            operand::input(view) => {
                let data = view.getdata();
                let key = (data.as_ptr() as usize, data.len());
                if let Some(buffer) = self.inputs.get(&key) {
                    return Ok(buffer.clone());
                }
                let buffer = self.session.upload(data).map_err(maptobackend)?;
                self.inputs.insert(key, buffer.clone());
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
                if let Some(buffer) = self.session.getbuffer(*slot) {
                    Ok(buffer.clone())
                } else {
                    let data =
                        self.session
                            .gethostdata(*slot)
                            .ok_or(backenderror::indexoutofrange {
                                index,
                                len: self.results.len(),
                            })?;
                    self.session.upload(&data).map_err(maptobackend)
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

impl submission for metalsubmission {
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
        let slot = ffi::matmul_into(&self.cache, &mut self.session, &lhs, &rhs, &geometry)
            .map_err(maptobackend)?;
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
        let slot = ffi::add_into(&self.cache, &mut self.session, &lhs, &rhs, count)
            .map_err(maptobackend)?;
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
        let slot = ffi::mul_into(&self.cache, &mut self.session, &lhs, &rhs, count)
            .map_err(maptobackend)?;
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
        let slot = ffi::rmsnorm_into(
            &self.cache,
            &mut self.session,
            &input,
            &weight,
            geometry.rows,
            geometry.width,
            eps,
        )
        .map_err(maptobackend)?;
        Ok(self.push(slot, inshape))
    }

    fn recordsoftmax(&mut self, input: operand<'_>) -> Result<stepref, backenderror> {
        let inshape = self.shapeof(input)?;
        self.checkbudget(&inshape)?;
        let geometry = axisshape::softmax(&inshape)?;
        let input = self.resolve(input)?;
        let slot = ffi::softmax_into(
            &self.cache,
            &mut self.session,
            &input,
            geometry.rows,
            geometry.width,
        )
        .map_err(maptobackend)?;
        Ok(self.push(slot, inshape))
    }

    fn recordsilu(&mut self, input: operand<'_>) -> Result<stepref, backenderror> {
        let inshape = self.shapeof(input)?;
        self.checkbudget(&inshape)?;
        let count = inshape
            .getelementcount()
            .map_err(|_| backenderror::sizeoverflow)?;
        let input = self.resolve(input)?;
        let slot =
            ffi::silu_into(&self.cache, &mut self.session, &input, count).map_err(maptobackend)?;
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
        let table = self.resolve(table)?;
        let slot = ffi::embedding_into(
            &self.cache,
            &mut self.session,
            &table,
            indices,
            geometry.width,
        )
        .map_err(maptobackend)?;
        let mut outdims = Vec::with_capacity(tableshape.getrank());
        outdims.push(geometry.rows);
        outdims.extend_from_slice(&tableshape.getdims()[1..]);
        let outshape = shape::new(outdims).map_err(|_| backenderror::sizeoverflow)?;
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
        let slot = ffi::rope_into(&self.cache, &mut self.session, &input, &geometry, base)
            .map_err(maptobackend)?;
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
        let slot = ffi::rope_at_into(
            &self.cache,
            &mut self.session,
            &input,
            position,
            geometry.heads,
            geometry.headdim,
            base,
        )
        .map_err(maptobackend)?;
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
            METALMAXHEADDIM,
        )?;
        let query = self.resolve(query)?;
        let key = self.resolve(key)?;
        let value = self.resolve(value)?;
        let slot = ffi::attention_into(
            &self.cache,
            &mut self.session,
            &query,
            &key,
            &value,
            &geometry,
            causal,
        )
        .map_err(maptobackend)?;
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
            METALMAXHEADDIM,
        )?;
        let query = self.resolve(query)?;
        let cachedkey = self.resolve(key.cached)?;
        let cachedvalue = self.resolve(value.cached)?;
        let newkey = self.resolve(key.new)?;
        let newvalue = self.resolve(value.new)?;
        let slot = ffi::attention_cached_into(
            &self.cache,
            &mut self.session,
            &query,
            ffi::bufferpair {
                cached: &cachedkey,
                new: &newkey,
            },
            ffi::bufferpair {
                cached: &cachedvalue,
                new: &newvalue,
            },
            &geometry,
            causal,
        )
        .map_err(maptobackend)?;
        Ok(self.push(slot, queryshape))
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

    fn apply(self, outputs: &[stepref]) -> Result<Vec<tensor<f32>>, backenderror> {
        let slots = self.session.apply().map_err(maptobackend)?;
        let results = self.results;
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
                wrap(outshape.clone(), data)
            })
            .collect()
    }
}
