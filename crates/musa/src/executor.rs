use std::sync::{Arc, Mutex};

use ::tensor::{shape, tensor, tensorview};
use api::{
    backend, backenderror, checkbudget, checkgatherbudget, compute, kvpair, operand, operandpair,
    recorder, session, stepref, submission,
};
use lib::limits;

use api::geometry::{
    ADDOP, MULOP, attentionshape, axisshape, binaryshape, embeddingshape, matmulshape, ropeshape,
};

use crate::backend::{MUSAMAXHEADDIM, musabackend};
use crate::error::musaerror;
use crate::ffi::{self, buf};

/// Which MUSA implementation serves `matmul`.
///
/// The choice is made once, at construction, and stored as a field, exactly as
/// `cuda` stores its `matmulpath`: selecting it per call would add a run-time
/// branch to the hottest path and obscure which path a measurement exercised.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum matmulpath {
    /// The in-repo Tensor-Core kernel (TF32), used by [`musaexecutor::new`].
    /// TF32 rounds each operand to a 10-bit mantissa, so its result differs
    /// from the FMA reference within the documented tolerance.
    wmma,
    /// The in-repo scalar FMA kernel, used by [`musaexecutor::newfma`] as the
    /// independent high-precision reference path.
    fma,
}

/// A MUSA executor: the crate's [`compute`](api::compute) provider.
///
/// `musaexecutor` owns a [`musabackend`] and is the single place where its
/// declared rank and byte budget is enforced. Besides the budget it holds only
/// device state and build-time-compiled kernels (kept in `ffi.rs`), never a
/// request-scoped value, so one value can be shared immutably across requests.
///
/// The operator set is complete: all thirteen `api::compute` operators are
/// served by in-repo kernels compiled at build time by `mcc` — matmul by the
/// WMMA Tensor-Core kernel (or, on the reference path, the FMA kernel),
/// softmax by the max-subtracting kernel, scan by the serial/parallel pair.
/// There is deliberately no vendor acceleration library and no runtime JIT
/// (docs/musa-backend-plan.md §5.3): build failures surface at build time, not
/// as a `kernelcompilefailed` at run time.
#[derive(Clone, Debug)]
pub struct musaexecutor {
    backend: musabackend,
    path: matmulpath,
    cache: Arc<ffi::cache>,
}

impl musaexecutor {
    /// Creates an executor over the MUSA backend with budget `lim`.
    ///
    /// `matmul` is served by the Tensor-Core (TF32) kernel. The executor owns
    /// the device state and the compiled kernels: a clone shares them, while
    /// an independently constructed executor retains its own device handle and
    /// loads its own kernels. Nothing touches the device until the first
    /// operator call — `new` never fails on a host without a MUSA device.
    #[must_use]
    pub fn new(lim: limits) -> Self {
        Self {
            backend: musabackend::new(lim),
            path: matmulpath::wmma,
            cache: Arc::new(ffi::cache::new()),
        }
    }

    /// Creates an executor whose `matmul` is served by the scalar FMA kernel.
    ///
    /// This is the independent high-precision path the Tensor-Core path is
    /// validated against. It exists as a constructor rather than a flag so a
    /// reference run can never be confused with a production run.
    #[must_use]
    pub fn newfma(lim: limits) -> Self {
        Self {
            backend: musabackend::new(lim),
            path: matmulpath::fma,
            cache: Arc::new(ffi::cache::new()),
        }
    }

    /// Returns which implementation serves `matmul`.
    #[must_use]
    pub fn getpath(&self) -> matmulpath {
        self.path
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

/// Reports a step reference that a previous persistent apply released.
fn stalestep() -> backenderror {
    backenderror::executionfailed {
        backend: "musa",
        kernel: "record".into(),
        message: "step was released by the previous apply".into(),
    }
}

/// Extracts a device pointer from an operand binding.
fn bufferptr(value: buf<'_>) -> Result<ffi::mudevptr, backenderror> {
    if let buf::device(ptr) = value {
        Ok(ptr)
    } else {
        Err(backenderror::executionfailed {
            backend: "musa",
            kernel: "append".into(),
            message: "operand is not device-resident".into(),
        })
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
        let data = match self.path {
            matmulpath::wmma => ffi::matmul_wmma(
                &self.cache,
                lhsdata,
                rhsdata,
                geometry.m,
                geometry.k,
                geometry.n,
                geometry.batch,
            ),
            matmulpath::fma => ffi::matmul_fma(
                &self.cache,
                lhsdata,
                rhsdata,
                geometry.m,
                geometry.k,
                geometry.n,
                geometry.batch,
            ),
        }
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

    fn rmsnormmatmul(
        &self,
        input: &tensorview<'_, f32>,
        weight: &tensorview<'_, f32>,
        rhs: &tensorview<'_, f32>,
        eps: f32,
    ) -> Result<tensor<f32>, backenderror> {
        self.checkbudget(input.getshape())?;
        self.checkbudget(weight.getshape())?;
        self.checkbudget(rhs.getshape())?;
        let (geometry, outshape) =
            rmsnormmatmulshape(input.getshape(), weight.getshape(), rhs.getshape())?;
        self.checkbudget(&outshape)?;
        let data = ffi::rmsnormmatmul(
            &self.cache,
            input.getdata(),
            weight.getdata(),
            rhs.getdata(),
            geometry.m,
            geometry.k,
            geometry.n,
            geometry.batch,
            eps,
        )
        .map_err(maptobackend)?;
        wrap(outshape, data)
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
/// Validates the fused RMSNorm+matmul shapes and returns the multiplication
/// plan and output shape.
fn rmsnormmatmulshape(
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

fn scanshape(
    input: &tensorview<'_, f32>,
    init: &tensorview<'_, f32>,
    a: &tensorview<'_, f32>,
    b: &tensorview<'_, f32>,
    c: &tensorview<'_, f32>,
    skip: &tensorview<'_, f32>,
) -> Result<(usize, usize, usize), backenderror> {
    let geometry = scanshapes(
        input.getshape(),
        init.getshape(),
        a.getshape(),
        b.getshape(),
        c.getshape(),
        skip.getshape(),
    )?;
    for &value in a.getdata() {
        if value.abs() > 1.0 || value.is_nan() {
            return Err(backenderror::invalidparameter { op: "scan" });
        }
    }
    Ok(geometry)
}

/// Validates scan ranks and state-width alignment without reading operand data.
fn scanshapes(
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

impl musaexecutor {
    /// Creates a single-use accumulative submission over this executor's
    /// backend and compiled-kernel cache.
    ///
    /// Every supported operator records a device-resident kernel without
    /// synchronising and keeps its result in a device slot (`matmul`,
    /// `add`/`mul`/`silu`, `rmsnorm`, `softmax`, `embedding`, `rope`/
    /// `rope_at`, `attention`/`attention_cached`). `scan`/`scan_cached` and
    /// `rmsnormmatmul` remain unsupported. `apply` synchronises once and
    /// downloads the requested outputs, so construction never fails — nothing
    /// touches the device until the first record.
    #[must_use]
    pub fn newsubmission(&self) -> musasubmission {
        musasubmission::new(self.backend.clone(), self.path, self.cache.clone())
    }

    /// Creates a request-scoped persistent session over this executor's backend
    /// and compiled-kernel cache.
    ///
    /// The session pins model weights once and reuses them across every
    /// [`apply`](session::apply), so a decode loop does not re-upload the model
    /// on each token. Call [`session::reset`] (or drop the session) to release
    /// the pinned allocations.
    #[must_use]
    pub fn newpersistent(&self) -> musasession {
        musasession::new(self.backend.clone(), self.path, self.cache.clone())
    }
}

/// A MUSA submission: the [`submission`] contract implemented over the same
/// executor-owned cache.
///
/// `musasubmission` owns the recorded results and no model weights; it shares
/// no mutable state between requests. Every supported step records a kernel
/// that reads existing device slots and leaves its result in a new slot; only
/// degenerate empty outputs stay on the host. [`apply`](submission::apply)
/// rebuilds the requested outputs with one final synchronise and download.
///
/// A batched `matmul` (a leading batch dimension greater than one) is rejected,
/// mirroring `cuda`'s submission contract: the recorded path is one contraction
/// block, and joining blocks back into one tensor would re-introduce host round
/// trips, so it is left to the single-operator [`compute`] path.
pub struct musasubmission {
    backend: musabackend,
    path: matmulpath,
    session: ffi::session,
    results: Vec<(resultdata, shape)>,
    /// Uploaded caller inputs keyed by `(data pointer, length)`, so an input
    /// operand referenced by several steps is uploaded exactly once per
    /// submission.
    inputs: std::collections::HashMap<(usize, usize), usize>,
    /// Inputs pinned by a persistent session, shared across submissions so the
    /// model weights stop being re-uploaded on every decode step.
    pinned: Arc<Mutex<std::collections::HashMap<(usize, usize), (ffi::mudevptr, usize)>>>,
}

/// A recorded result: a device slot, host data for a degenerate empty output,
/// or a step released by the previous persistent apply.
enum resultdata {
    slot(usize),
    host(Vec<f32>),
    dead,
}

impl musasubmission {
    /// Creates a submission over `backend` using `cache`, selecting `path` for
    /// recorded `matmul` steps.
    pub(crate) fn new(backend: musabackend, path: matmulpath, cache: Arc<ffi::cache>) -> Self {
        Self::newpinned(
            backend,
            path,
            cache,
            Arc::new(Mutex::new(std::collections::HashMap::new())),
        )
    }

    /// Creates a submission that shares `pinned` with a persistent session.
    pub(crate) fn newpinned(
        backend: musabackend,
        path: matmulpath,
        cache: Arc<ffi::cache>,
        pinned: Arc<Mutex<std::collections::HashMap<(usize, usize), (ffi::mudevptr, usize)>>>,
    ) -> Self {
        let session = ffi::session::new(cache);
        Self {
            backend,
            path,
            session,
            results: Vec::new(),
            inputs: std::collections::HashMap::new(),
            pinned,
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
                match self.results.get(index) {
                    Some((resultdata::dead, _)) => Err(stalestep()),
                    Some((_, outshape)) => Ok(outshape.clone()),
                    None => Err(backenderror::indexoutofrange {
                        index,
                        len: self.results.len(),
                    }),
                }
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
                    Some((resultdata::dead, _)) => Err(stalestep()),
                    None => Err(backenderror::indexoutofrange {
                        index,
                        len: self.results.len(),
                    }),
                }
            }
        }
    }

    /// Resolves an operand to host data for the upload fallback (a degenerate
    /// empty result); a device-resident result is drained with one
    /// synchronise, which reintroduces a sync only on that fallback path.
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
                    Some((resultdata::dead, _)) => Err(stalestep()),
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
        if let operand::input(view) = op {
            let data = view.getdata();
            let key = (data.as_ptr() as usize, data.len());
            let pinned = self
                .pinned
                .lock()
                .map_err(|_| backenderror::executionfailed {
                    backend: "musa",
                    kernel: "pin".into(),
                    message: "pinned input cache poisoned".into(),
                })?;
            if let Some(&(ptr, _)) = pinned.get(&key) {
                return Ok(buf::device(ptr));
            }
        }
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

/// A request-scoped persistent MUSA session.
///
/// Unlike [`musasubmission`], a session keeps its pinned inputs (the model
/// weights) in device allocations that outlive each [`apply`](session::apply).
/// Every apply consumes the current step buffer and starts a fresh one over the
/// same pinned allocations, so a decode loop uploads the weights once and the
/// transient activations and key/value rows are released after each step.
pub struct musasession {
    backend: musabackend,
    path: matmulpath,
    cache: Arc<ffi::cache>,
    pinned: Arc<Mutex<std::collections::HashMap<(usize, usize), (ffi::mudevptr, usize)>>>,
    current: musasubmission,
}

impl musasession {
    /// Creates an empty session over `backend`, selecting `path` for matmul.
    pub(crate) fn new(backend: musabackend, path: matmulpath, cache: Arc<ffi::cache>) -> Self {
        let pinned = Arc::new(Mutex::new(std::collections::HashMap::new()));
        let current =
            musasubmission::newpinned(backend.clone(), path, cache.clone(), pinned.clone());
        Self {
            backend,
            path,
            cache,
            pinned,
            current,
        }
    }

    /// Allocates a zeroed device buffer that survives every apply and returns a
    /// step handle to it.
    ///
    /// A device KV cache uses this for its per-layer key/value buffers, then
    /// exposes the first `len` rows through [`recorder::recordview`].
    ///
    /// # Errors
    ///
    /// Returns a backend error when the allocation fails or the shape overflows.
    pub fn zeroed(&mut self, shape: shape) -> Result<stepref, backenderror> {
        let count = shape
            .getelementcount()
            .map_err(|_| backenderror::sizeoverflow)?;
        let slot = self
            .current
            .session
            .alloczeros(count)
            .map_err(maptobackend)?;
        Ok(self.current.pushslot(slot, shape))
    }

    /// Releases every pinned allocation through the deferred free queue.
    fn retirepinned(&mut self) {
        let pinned = match self.pinned.lock() {
            Ok(mut map) => std::mem::take(&mut *map),
            Err(poisoned) => std::mem::take(&mut *poisoned.into_inner()),
        };
        for (_, (ptr, bytes)) in pinned {
            ffi::retiredevice(&self.cache, ptr, bytes);
        }
    }
}

impl Drop for musasession {
    fn drop(&mut self) {
        self.retirepinned();
    }
}

impl recorder for musasession {
    fn recordmatmul(
        &mut self,
        lhs: operand<'_>,
        rhs: operand<'_>,
    ) -> Result<stepref, backenderror> {
        submission::recordmatmul(&mut self.current, lhs, rhs)
    }

    fn recordadd(&mut self, lhs: operand<'_>, rhs: operand<'_>) -> Result<stepref, backenderror> {
        submission::recordadd(&mut self.current, lhs, rhs)
    }

    fn recordmul(&mut self, lhs: operand<'_>, rhs: operand<'_>) -> Result<stepref, backenderror> {
        submission::recordmul(&mut self.current, lhs, rhs)
    }

    fn recordrmsnorm(
        &mut self,
        input: operand<'_>,
        weight: operand<'_>,
        eps: f32,
    ) -> Result<stepref, backenderror> {
        submission::recordrmsnorm(&mut self.current, input, weight, eps)
    }

    fn recordrmsnormmatmul(
        &mut self,
        input: operand<'_>,
        weight: operand<'_>,
        rhs: operand<'_>,
        eps: f32,
    ) -> Result<stepref, backenderror> {
        submission::recordrmsnormmatmul(&mut self.current, input, weight, rhs, eps)
    }

    fn recordsoftmax(&mut self, input: operand<'_>) -> Result<stepref, backenderror> {
        submission::recordsoftmax(&mut self.current, input)
    }

    fn recordsilu(&mut self, input: operand<'_>) -> Result<stepref, backenderror> {
        submission::recordsilu(&mut self.current, input)
    }

    fn recordembedding(
        &mut self,
        table: operand<'_>,
        indices: &[usize],
    ) -> Result<stepref, backenderror> {
        submission::recordembedding(&mut self.current, table, indices)
    }

    fn recordrope(
        &mut self,
        input: operand<'_>,
        heads: usize,
        base: f32,
    ) -> Result<stepref, backenderror> {
        submission::recordrope(&mut self.current, input, heads, base)
    }

    fn recordrope_at(
        &mut self,
        input: operand<'_>,
        position: usize,
        heads: usize,
        base: f32,
    ) -> Result<stepref, backenderror> {
        submission::recordrope_at(&mut self.current, input, position, heads, base)
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
        submission::recordattention(&mut self.current, query, key, value, heads, kvheads, causal)
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
        submission::recordattention_cached(
            &mut self.current,
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
        self.recordscan_pair(input, init, a, b, c, skip)
            .map(|(out, _)| out)
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
        self.recordscan_cached_pair(input, state, a, b, c, skip)
            .map(|(out, _)| out)
    }

    fn recordscan_pair(
        &mut self,
        input: operand<'_>,
        init: operand<'_>,
        a: operand<'_>,
        b: operand<'_>,
        c: operand<'_>,
        skip: operand<'_>,
    ) -> Result<(stepref, stepref), backenderror> {
        let inshape = self.current.shapeof(input)?;
        let initshape = self.current.shapeof(init)?;
        let ashape = self.current.shapeof(a)?;
        let bshape = self.current.shapeof(b)?;
        let cshape = self.current.shapeof(c)?;
        let skipshape = self.current.shapeof(skip)?;
        for shape in [&inshape, &initshape, &ashape, &bshape, &cshape, &skipshape] {
            self.current.checkbudget(shape)?;
        }
        let (n, d, statewidth) =
            scanshapes(&inshape, &initshape, &ashape, &bshape, &cshape, &skipshape)?;
        // The spectral check needs host values. A recorded result is already on
        // the device, so it cannot be inspected without a download; the
        // submission contract validates caller inputs and leaves device
        // intermediates to the backend (docs/design.md §21.13).
        if let operand::input(view) = a {
            for &value in view.getdata() {
                if value.abs() > 1.0 || value.is_nan() {
                    return Err(backenderror::invalidparameter { op: "scan" });
                }
            }
        }
        let inputbuf = self.current.operandsbuf(input)?;
        let initbuf = self.current.operandsbuf(init)?;
        let abuf = self.current.operandsbuf(a)?;
        let bbuf = self.current.operandsbuf(b)?;
        let cbuf = self.current.operandsbuf(c)?;
        let skipbuf = self.current.operandsbuf(skip)?;
        let (outslot, stateslot) = ffi::scan_record(
            &mut self.current.session,
            inputbuf,
            initbuf,
            abuf,
            bbuf,
            cbuf,
            skipbuf,
            n,
            d,
            statewidth,
        )
        .map_err(maptobackend)?;
        let outshape = shape::new(vec![n, d]).map_err(|_| backenderror::sizeoverflow)?;
        let stateshape = shape::new(vec![d, statewidth]).map_err(|_| backenderror::sizeoverflow)?;
        Ok((
            self.current.pushslot(outslot, outshape),
            self.current.pushslot(stateslot, stateshape),
        ))
    }

    fn recordscan_cached_pair(
        &mut self,
        input: operand<'_>,
        state: operand<'_>,
        a: operand<'_>,
        b: operand<'_>,
        c: operand<'_>,
        skip: operand<'_>,
    ) -> Result<(stepref, stepref), backenderror> {
        let inshape = self.current.shapeof(input)?;
        let stateshapein = self.current.shapeof(state)?;
        let ashape = self.current.shapeof(a)?;
        let bshape = self.current.shapeof(b)?;
        let cshape = self.current.shapeof(c)?;
        let skipshape = self.current.shapeof(skip)?;
        for shape in [
            &inshape,
            &stateshapein,
            &ashape,
            &bshape,
            &cshape,
            &skipshape,
        ] {
            self.current.checkbudget(shape)?;
        }
        let (n, d, statewidth) = scanshapes(
            &inshape,
            &stateshapein,
            &ashape,
            &bshape,
            &cshape,
            &skipshape,
        )?;
        if n != 1 {
            return Err(backenderror::shapemismatch { op: "scan_cached" });
        }
        if let operand::input(view) = a {
            for &value in view.getdata() {
                if value.abs() > 1.0 || value.is_nan() {
                    return Err(backenderror::invalidparameter { op: "scan_cached" });
                }
            }
        }
        let inputbuf = self.current.operandsbuf(input)?;
        let statebuf = self.current.operandsbuf(state)?;
        let abuf = self.current.operandsbuf(a)?;
        let bbuf = self.current.operandsbuf(b)?;
        let cbuf = self.current.operandsbuf(c)?;
        let skipbuf = self.current.operandsbuf(skip)?;
        let (outslot, newstate) = ffi::scan_cached_record(
            &mut self.current.session,
            inputbuf,
            statebuf,
            abuf,
            bbuf,
            cbuf,
            skipbuf,
            d,
            statewidth,
        )
        .map_err(maptobackend)?;
        let outshape = shape::new(vec![1, d]).map_err(|_| backenderror::sizeoverflow)?;
        let stateshape = shape::new(vec![d, statewidth]).map_err(|_| backenderror::sizeoverflow)?;
        Ok((
            self.current.pushslot(outslot, outshape),
            self.current.pushslot(newstate, stateshape),
        ))
    }

    fn recordview(&mut self, base: stepref, shape: shape) -> Result<stepref, backenderror> {
        let index = base.getindex();
        match self.current.results.get(index) {
            Some((resultdata::slot(slot), _)) => Ok(self.current.pushslot(*slot, shape)),
            Some((resultdata::dead, _)) => Err(stalestep()),
            Some((resultdata::host(_), _)) | None => Err(backenderror::indexoutofrange {
                index,
                len: self.current.results.len(),
            }),
        }
    }

    fn recordappend(
        &mut self,
        cached: operand<'_>,
        new: operand<'_>,
        position: usize,
    ) -> Result<(), backenderror> {
        let cachedbuf = self.current.operandsbuf(cached)?;
        let newbuf = self.current.operandsbuf(new)?;
        let cachedptr = bufferptr(cachedbuf)?;
        let newptr = bufferptr(newbuf)?;
        let newshape = self.current.shapeof(new)?;
        let width = *newshape
            .getdims()
            .last()
            .ok_or(backenderror::shapemismatch { op: "append" })?;
        let bytes = width
            .checked_mul(core::mem::size_of::<f32>())
            .ok_or(backenderror::sizeoverflow)?;
        let offset = position
            .checked_mul(bytes)
            .ok_or(backenderror::sizeoverflow)?;
        let dst = cachedptr
            .checked_add(offset)
            .ok_or(backenderror::sizeoverflow)?;
        ffi::session::copyinto(&mut self.current.session, dst, newptr, bytes)
            .map_err(maptobackend)?;
        Ok(())
    }

    fn pin(&mut self, view: tensorview<'_, f32>) -> Result<(), backenderror> {
        let data = view.getdata();
        let key = (data.as_ptr() as usize, data.len());
        let mut pinned = self
            .pinned
            .lock()
            .map_err(|_| backenderror::executionfailed {
                backend: "musa",
                kernel: "pin".into(),
                message: "pinned input cache poisoned".into(),
            })?;
        if pinned.contains_key(&key) {
            return Ok(());
        }
        let (ptr, bytes) =
            ffi::session::uploadpersistent(&self.cache, data).map_err(maptobackend)?;
        pinned.insert(key, (ptr, bytes));
        Ok(())
    }
}

impl session for musasession {
    fn apply(&mut self, outputs: &[stepref]) -> Result<Vec<tensor<f32>>, backenderror> {
        self.current.applyref(outputs)
    }

    fn reset(&mut self) {
        self.retirepinned();
        self.current = musasubmission::newpinned(
            self.backend.clone(),
            self.path,
            self.cache.clone(),
            self.pinned.clone(),
        );
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
        if geometry.m == 0 || geometry.n == 0 {
            return Ok(self.push(Vec::new(), outshape));
        }
        let lhs = self.operandsbuf(lhs)?;
        let rhs = self.operandsbuf(rhs)?;
        let slot = match self.path {
            matmulpath::wmma => ffi::matmul_wmma_record(
                &mut self.session,
                lhs,
                rhs,
                geometry.m,
                geometry.k,
                geometry.n,
            ),
            matmulpath::fma => ffi::matmul_fma_record(
                &mut self.session,
                lhs,
                rhs,
                geometry.m,
                geometry.k,
                geometry.n,
            ),
        }
        .map_err(maptobackend)?;
        Ok(self.pushslot(slot, outshape))
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
        let count = lhsshape
            .getelementcount()
            .map_err(|_| backenderror::sizeoverflow)?;
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
        let slot = ffi::rmsnorm_record(
            &mut self.session,
            ibuf,
            wbuf,
            geometry.rows,
            geometry.width,
            eps,
        )
        .map_err(maptobackend)?;
        Ok(self.pushslot(slot, inshape))
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
        if geometry.batch == 0 || geometry.m == 0 || geometry.n == 0 {
            return Ok(self.push(Vec::new(), outshape));
        }
        let inputbuf = self.operandsbuf(input)?;
        let weightbuf = self.operandsbuf(weight)?;
        let rhsbuf = self.operandsbuf(rhs)?;
        let rows = geometry
            .batch
            .checked_mul(geometry.m)
            .ok_or(backenderror::sizeoverflow)?;
        let total = rows
            .checked_mul(geometry.n)
            .ok_or(backenderror::sizeoverflow)?;
        let invslot = ffi::rmsinv_record(&mut self.session, inputbuf, rows, geometry.k, eps)
            .map_err(maptobackend)?;
        let slot = ffi::rmsnormmatmul_record(
            &mut self.session,
            inputbuf,
            weightbuf,
            rhsbuf,
            invslot,
            total,
            geometry.m,
            geometry.k,
            geometry.n,
            geometry.batch,
        )
        .map_err(maptobackend)?;
        Ok(self.pushslot(slot, outshape))
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
        let count = inshape
            .getelementcount()
            .map_err(|_| backenderror::sizeoverflow)?;
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
        let mut outdims = Vec::with_capacity(tableshape.getrank());
        outdims.push(geometry.rows);
        outdims.extend_from_slice(&tableshape.getdims()[1..]);
        let outshape = shape::new(outdims).map_err(|_| backenderror::sizeoverflow)?;
        if geometry.rows == 0 {
            return Ok(self.push(Vec::new(), outshape));
        }
        let tablebuf = self.operandsbuf(table)?;
        let index64 = tou64(indices);
        let indexslot = self.session.uploadu64(&index64).map_err(maptobackend)?;
        let indexbuf = buf::device(self.session.slotptr(indexslot));
        let slot = ffi::embedding_record(
            &mut self.session,
            tablebuf,
            indexbuf,
            geometry.rows,
            geometry.width,
        )
        .map_err(maptobackend)?;
        Ok(self.pushslot(slot, outshape))
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
        let count = inshape
            .getelementcount()
            .map_err(|_| backenderror::sizeoverflow)?;
        let inputbuf = self.operandsbuf(input)?;
        let slot = ffi::rope_record(
            &mut self.session,
            inputbuf,
            count,
            geometry.rows,
            geometry.seqaxis,
            geometry.heads,
            geometry.headdim,
            base,
        )
        .map_err(maptobackend)?;
        Ok(self.pushslot(slot, inshape))
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
        let count = inshape
            .getelementcount()
            .map_err(|_| backenderror::sizeoverflow)?;
        let inputbuf = self.operandsbuf(input)?;
        let slot = ffi::rope_at_record(
            &mut self.session,
            inputbuf,
            count,
            position,
            geometry.heads,
            geometry.headdim,
            base,
        )
        .map_err(maptobackend)?;
        Ok(self.pushslot(slot, inshape))
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
        let count = queryshape
            .getelementcount()
            .map_err(|_| backenderror::sizeoverflow)?;
        let query = self.operandsbuf(query)?;
        let key = self.operandsbuf(key)?;
        let value = self.operandsbuf(value)?;
        let slot = ffi::attention_record(
            &mut self.session,
            query,
            key,
            value,
            count,
            geometry.qseq,
            geometry.kvseq,
            geometry.heads,
            geometry.kvheads,
            geometry.headdim,
            causal,
        )
        .map_err(maptobackend)?;
        Ok(self.pushslot(slot, queryshape))
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
        let count = queryshape
            .getelementcount()
            .map_err(|_| backenderror::sizeoverflow)?;
        let query = self.operandsbuf(query)?;
        let cachedkey = self.operandsbuf(key.cached)?;
        let cachedvalue = self.operandsbuf(value.cached)?;
        let newkey = self.operandsbuf(key.new)?;
        let newvalue = self.operandsbuf(value.new)?;
        let slot = ffi::attention_cached_record(
            &mut self.session,
            query,
            cachedkey,
            cachedvalue,
            newkey,
            newvalue,
            count,
            geometry.kvseq,
            geometry.heads,
            geometry.kvheads,
            geometry.headdim,
            causal,
        )
        .map_err(maptobackend)?;
        Ok(self.pushslot(slot, queryshape))
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
        let slotindices = self.requestslots(outputs)?;
        let downloaded = self.session.apply(&slotindices).map_err(maptobackend)?;
        self.materialize(outputs, downloaded)
    }
}

impl musasubmission {
    /// Maps the requested step references to their device slots.
    fn requestslots(&self, outputs: &[stepref]) -> Result<Vec<usize>, backenderror> {
        let mut slotindices = Vec::new();
        for reference in outputs {
            let index = reference.getindex();
            match self.results.get(index) {
                Some((resultdata::slot(slot), _)) => slotindices.push(*slot),
                Some((resultdata::host(_), _)) => {}
                Some((resultdata::dead, _)) => return Err(stalestep()),
                None => {
                    return Err(backenderror::indexoutofrange {
                        index,
                        len: self.results.len(),
                    });
                }
            }
        }
        Ok(slotindices)
    }

    /// Rebuilds the requested outputs in request order.
    fn materialize(
        &self,
        outputs: &[stepref],
        downloaded: Vec<Vec<f32>>,
    ) -> Result<Vec<tensor<f32>>, backenderror> {
        let mut cursor = 0;
        let mut built = Vec::with_capacity(outputs.len());
        for reference in outputs {
            let index = reference.getindex();
            let (data, outshape) =
                self.results
                    .get(index)
                    .ok_or(backenderror::indexoutofrange {
                        index,
                        len: self.results.len(),
                    })?;
            let data = match data {
                resultdata::host(data) => data.clone(),
                resultdata::slot(_) => {
                    let data = downloaded[cursor].clone();
                    cursor += 1;
                    data
                }
                resultdata::dead => return Err(stalestep()),
            };
            built.push(wrap(outshape.clone(), data)?);
        }
        Ok(built)
    }

    /// Applies a persistent step without consuming the submission: downloads
    /// the requested outputs, frees every transient slot, and keeps the
    /// persistent cache buffers (and their step indices) alive.
    fn applyref(&mut self, outputs: &[stepref]) -> Result<Vec<tensor<f32>>, backenderror> {
        let slotindices = self.requestslots(outputs)?;
        let downloaded = self
            .session
            .applyretained(&slotindices)
            .map_err(maptobackend)?;
        let built = self.materialize(outputs, downloaded)?;
        for (data, _) in self.results.iter_mut() {
            match data {
                resultdata::slot(slot) if !self.session.ispersistent(*slot) => {
                    *data = resultdata::dead;
                }
                resultdata::host(_) => *data = resultdata::dead,
                _ => {}
            }
        }
        self.inputs.clear();
        Ok(built)
    }
}
