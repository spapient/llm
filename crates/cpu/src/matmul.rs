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
/// two operands must be equal element by element. Accumulation uses a
/// register-blocked SIMD micro-kernel with fused multiply-add, so the reduction
/// is not a plain left-to-right `f32` sum; the result is deterministic for
/// fixed inputs but matches the scalar reference only within tolerance, not
/// bit-for-bit.
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
        self.applyplanwith(plan, lhs, rhs, 1)
    }

    /// Applies the operator using an already validated plan, letting the
    /// blocked packed path use up to `workers` scoped threads.
    ///
    /// Every output element's fused accumulation sequence is identical to the
    /// serial kernel, so the result is bit-for-bit independent of `workers`.
    pub(crate) fn applyplanwith(
        &self,
        plan: matmulshape,
        lhs: &tensorview<'_, f32>,
        rhs: &tensorview<'_, f32>,
        workers: usize,
    ) -> Result<tensor<f32>, backenderror> {
        let data = matmuldata(&plan, lhs, rhs, workers)?;
        let outshape = shape::new(plan.dims).map_err(|_| backenderror::sizeoverflow)?;
        finish(&outshape, data)
    }

    /// The fused `rmsnormmatmul` entry over an already validated plan.
    ///
    /// The RMS weight folds into the operands exactly once, outside the hot
    /// loops: the direct path (narrow rows) pre-folds it into a scratch copy
    /// of the left operand, because its loads re-read every row per column
    /// tile and an in-loop fold would repeat the multiply per tile; the
    /// blocked packed path folds it into the packed right operand at pack
    /// time, amortised over all row blocks. Every stored row is then scaled
    /// by its reciprocal RMS on the final inner chunk only, so the
    /// normalised activation is never materialised and the per-element
    /// accumulation sequence is unchanged.
    pub(crate) fn applyplanfused(
        &self,
        plan: matmulshape,
        input: &tensorview<'_, f32>,
        weight: &[f32],
        rhs: &tensorview<'_, f32>,
        rowscale: &[f32],
        workers: usize,
    ) -> Result<tensor<f32>, backenderror> {
        let data = if plan.m < PACKROWS {
            let xw: Vec<f32> = input
                .getdata()
                .chunks(plan.k)
                .flat_map(|row| row.iter().zip(weight.iter()).map(|(&x, &w)| x * w))
                .collect();
            let xwtensor = tensor::new(
                shape::new(input.getshape().getdims().to_vec())
                    .map_err(|_| backenderror::sizeoverflow)?,
                xw,
            )
            .map_err(|e| backenderror::executionfailed {
                backend: "cpu",
                kernel: MATMULOP.into(),
                message: e.to_string(),
            })?;
            matmuldatarows(
                &plan,
                &xwtensor.getview(),
                rhs,
                workers,
                Some(rowscale),
                None,
            )?
        } else {
            matmuldatarows(&plan, input, rhs, workers, Some(rowscale), Some(weight))?
        };
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
    workers: usize,
) -> Result<Vec<f32>, backenderror> {
    matmuldatarows(plan, lhs, rhs, workers, None, None)
}

/// [`matmuldata`] with a per-output-row scale applied inside the kernel
/// stores, so no separate output pass exists. `rowscale` carries one
/// reciprocal RMS per row of the left operand, in row order across the batch.
fn matmuldatarows(
    plan: &matmulshape,
    lhs: &tensorview<'_, f32>,
    rhs: &tensorview<'_, f32>,
    workers: usize,
    rowscale: Option<&[f32]>,
    rowweight: Option<&[f32]>,
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
        matmulblock(
            a, bt, out, plan.m, plan.k, plan.n, workers, rowscale, rowweight,
        )?;
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

/// Register-tile row count of the SIMD micro-kernel.
const BLOCKROWS: usize = 8;

/// Row count at or above which the blocked packed path replaces the direct
/// path. Below it the right operand is re-walked at most `PACKROWS / BLOCKROWS`
/// times, so the packing copy would cost more than the re-walk it saves.
const PACKROWS: usize = 32;

/// Inner-dimension chunk of one packed right-operand tile. The tile stays in
/// the packed buffer and is reused by every `BLOCKROWS`-row block, so its size
/// is chosen to remain cache-resident on the documented baseline machine.
const KCPANEL: usize = 512;

/// Column count of one packed right-operand tile.
const NCPANEL: usize = 256;

/// Computes one `[m, k] * [k, n]` block into `out` with a register-blocked
/// SIMD micro-kernel dispatched through `simd`.
///
/// Wide row counts take the blocked packed path: the right operand is copied
/// into a contiguous tile of `KCPANEL * NCPANEL` at most, and every
/// `BLOCKROWS`-row block then streams that tile instead of re-walking the
/// strided operand. With more than one worker the packed path splits the row
/// blocks over scoped threads, each with a private packed tile and a private
/// partial output that is merged afterwards. Narrow row counts (decode) take
/// the direct path, which already walks the right operand exactly once and
/// would only pay for the extra copy. The per-element fused accumulation
/// order is identical on every path and thread layout, so the choice never
/// changes the floating-point result.
#[allow(clippy::too_many_arguments)]
fn matmulblock(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    workers: usize,
    rowscale: Option<&[f32]>,
    rowweight: Option<&[f32]>,
) -> Result<(), backenderror> {
    if m >= PACKROWS {
        if workers > 1 && m > BLOCKROWS {
            matmulblockedparallel(a, b, out, m, k, n, workers, rowscale, rowweight)
        } else {
            matmulblocked(a, b, out, m, k, n, rowscale, rowweight);
            Ok(())
        }
    } else {
        simd::matmuldirect(simd::detect(), a, b, out, m, k, n, rowscale);
        Ok(())
    }
}

/// The blocked packed micro-kernel behind [`matmulblock`].
///
/// The right operand is walked in column panels of at most `NCPANEL` and inner
/// chunks of at most `KCPANEL`. Each chunk is copied once into the contiguous
/// packed tile and then streamed by every `BLOCKROWS`-row block, so the right
/// operand is read from its strided layout exactly once instead of once per
/// row block. Output tiles are loaded and stored once per inner chunk, which
/// keeps the per-element accumulation order identical to [`matmuldirect`]:
/// the same ascending inner-dimension sequence of fused multiply-adds.
#[allow(clippy::too_many_arguments)]
fn matmulblocked(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    rowscale: Option<&[f32]>,
    rowweight: Option<&[f32]>,
) {
    let mut packed = vec![0.0_f32; KCPANEL * NCPANEL];
    for jp in (0..n).step_by(NCPANEL) {
        let panel = (n - jp).min(NCPANEL);
        for kc0 in (0..k).step_by(KCPANEL) {
            let kchunk = (k - kc0).min(KCPANEL);
            packpanel(b, &mut packed, jp, panel, kc0, kchunk, n, rowweight);
            for i0 in (0..m).step_by(BLOCKROWS) {
                let rows = (m - i0).min(BLOCKROWS);
                simd::matmultile(
                    simd::detect(),
                    a,
                    b,
                    &packed,
                    &mut out[i0 * n..],
                    i0,
                    rows,
                    jp,
                    panel,
                    kc0,
                    kchunk,
                    k,
                    n,
                    rowscale,
                    kc0 + kchunk >= k,
                );
            }
        }
    }
}

/// The blocked packed micro-kernel with the row blocks split over `workers`
/// scoped threads.
///
/// Each thread packs its own copy of every tile and accumulates into a private
/// partial output for its strided row blocks; the partials are merged into
/// `out` after the threads join, in block order. No mutable state is shared
/// between threads, and the per-element accumulation sequence is identical to
/// [`matmulblocked`], so the result is bit-for-bit independent of the worker
/// count.
// The geometry parameters are loop invariants the caller owns; passing them
// flat keeps the hot inner loops free of descriptor indirection (same trade
// as `cpuscan::apply_chunked`).
#[allow(clippy::too_many_arguments)]
fn matmulblockedparallel(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    workers: usize,
    rowscale: Option<&[f32]>,
    rowweight: Option<&[f32]>,
) -> Result<(), backenderror> {
    let nblocks = m.div_ceil(BLOCKROWS);
    let isa = simd::detect();
    std::thread::scope(|scope| {
        let mut slots: Vec<(usize, usize, Vec<f32>)> = Vec::with_capacity(workers);
        let mut handles = Vec::with_capacity(workers);
        for t in 0..workers {
            let blocks = (nblocks - t).div_ceil(workers);
            if blocks == 0 {
                continue;
            }
            let mut part = vec![0.0_f32; blocks * BLOCKROWS * n];
            handles.push(scope.spawn(move || {
                let mut packed = vec![0.0_f32; KCPANEL * NCPANEL];
                for jp in (0..n).step_by(NCPANEL) {
                    let panel = (n - jp).min(NCPANEL);
                    for kc0 in (0..k).step_by(KCPANEL) {
                        let kchunk = (k - kc0).min(KCPANEL);
                        packpanel(b, &mut packed, jp, panel, kc0, kchunk, n, rowweight);
                        for lb in 0..blocks {
                            let i0 = (t + lb * workers) * BLOCKROWS;
                            let rows = (m - i0).min(BLOCKROWS);
                            simd::matmultile(
                                isa,
                                a,
                                b,
                                &packed,
                                &mut part[lb * BLOCKROWS * n..],
                                i0,
                                rows,
                                jp,
                                panel,
                                kc0,
                                kchunk,
                                k,
                                n,
                                rowscale,
                                kc0 + kchunk >= k,
                            );
                        }
                    }
                }
                (t, blocks, part)
            }));
        }
        let mut failed = false;
        for handle in handles {
            match handle.join() {
                Ok(slot) => slots.push(slot),
                Err(_) => {
                    failed = true;
                    break;
                }
            }
        }
        if !failed {
            for (t, blocks, part) in &slots {
                for lb in 0..*blocks {
                    let i0 = (t + lb * workers) * BLOCKROWS;
                    let rows = (m - i0).min(BLOCKROWS);
                    out[i0 * n..(i0 + rows) * n]
                        .copy_from_slice(&part[lb * BLOCKROWS * n..(lb * BLOCKROWS + rows) * n]);
                }
            }
        }
        if failed {
            return Err(backenderror::executionfailed {
                backend: "cpu",
                kernel: MATMULOP.into(),
                message: "a matmul worker thread panicked".into(),
            });
        }
        Ok(())
    })
}

/// Copies one inner chunk of one column panel of `b` into the packed tile.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn packpanel(
    b: &[f32],
    packed: &mut [f32],
    jp: usize,
    panel: usize,
    kc0: usize,
    kchunk: usize,
    n: usize,
    rowweight: Option<&[f32]>,
) {
    for p in 0..kchunk {
        let src = &b[(kc0 + p) * n + jp..(kc0 + p) * n + jp + panel];
        let dst = &mut packed[p * panel..p * panel + panel];
        dst.copy_from_slice(src);
        if let Some(w) = rowweight {
            let s = w[kc0 + p];
            for entry in dst.iter_mut() {
                *entry *= s;
            }
        }
    }
}
