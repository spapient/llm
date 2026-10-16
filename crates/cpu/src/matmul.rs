use ::tensor::{shape, tensor, tensorview};
use api::{backenderror, binarykernel};
use archmage::prelude::*;

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
        matmulblock(a, bt, out, plan.m, plan.k, plan.n, workers)?;
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
/// SIMD micro-kernel dispatched through `archmage`.
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
fn matmulblock(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    workers: usize,
) -> Result<(), backenderror> {
    if m >= PACKROWS {
        if workers > 1 && m > BLOCKROWS {
            matmulblockedparallel(a, b, out, m, k, n, workers)
        } else {
            matmulblocked(a, b, out, m, k, n);
            Ok(())
        }
    } else {
        incant!(matmuldirect(a, b, out, m, k, n), [v3, neon, scalar]);
        Ok(())
    }
}

/// The direct register-blocked SIMD micro-kernel behind [`matmulblock`].
///
/// Tiles the output into `BLOCKROWS` by lane blocks whose accumulators stay in
/// SIMD registers across the inner dimension. Every row count uses the SIMD
/// path (a partial final block just holds fewer accumulators), so the fused
/// accumulation order depends only on the inner dimension, never on the batch
/// width. Only columns past the last full lane block fall back to a scalar dot
/// product. The lane width differs per tier (8 on AVX2, 4 on NEON, 1 scalar),
/// but every output element accumulates over `k` in ascending order with fused
/// multiply-add, so the result is bit-for-bit identical to the previous
/// `fearless_simd` 4-lane kernel.
#[arcane(import_intrinsics)]
fn matmuldirect_v3(
    _: X64V3Token,
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let ntile = n / 8 * 8;
    for i0 in (0..m).step_by(BLOCKROWS) {
        let rows = (m - i0).min(BLOCKROWS);
        for j0 in (0..ntile).step_by(8) {
            let mut acc = [_mm256_setzero_ps(); BLOCKROWS];
            for p in 0..k {
                let mut bl = [0.0_f32; 8];
                bl.copy_from_slice(&b[p * n + j0..p * n + j0 + 8]);
                let brow = _mm256_loadu_ps(&bl);
                for r in 0..rows {
                    acc[r] = _mm256_fmadd_ps(brow, _mm256_set1_ps(a[(i0 + r) * k + p]), acc[r]);
                }
            }
            for r in 0..rows {
                let mut outl = [0.0_f32; 8];
                _mm256_storeu_ps(&mut outl, acc[r]);
                out[(i0 + r) * n + j0..(i0 + r) * n + j0 + 8].copy_from_slice(&outl);
            }
        }
        for r in 0..rows {
            for j in ntile..n {
                out[(i0 + r) * n + j] = scalardot(a, b, i0 + r, j, k, n);
            }
        }
    }
}

/// AArch64 4-lane direct variant.
#[arcane(import_intrinsics)]
fn matmuldirect_neon(
    _: NeonToken,
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    let ntile = n / 4 * 4;
    for i0 in (0..m).step_by(BLOCKROWS) {
        let rows = (m - i0).min(BLOCKROWS);
        for j0 in (0..ntile).step_by(4) {
            let mut acc = [vdupq_n_f32(0.0); BLOCKROWS];
            for p in 0..k {
                let mut bl = [0.0_f32; 4];
                bl.copy_from_slice(&b[p * n + j0..p * n + j0 + 4]);
                let brow = vld1q_f32(&bl);
                for r in 0..rows {
                    acc[r] = vfmaq_f32(acc[r], brow, vdupq_n_f32(a[(i0 + r) * k + p]));
                }
            }
            for r in 0..rows {
                let mut outl = [0.0_f32; 4];
                vst1q_f32(&mut outl, acc[r]);
                out[(i0 + r) * n + j0..(i0 + r) * n + j0 + 4].copy_from_slice(&outl);
            }
        }
        for r in 0..rows {
            for j in ntile..n {
                out[(i0 + r) * n + j] = scalardot(a, b, i0 + r, j, k, n);
            }
        }
    }
}

/// Scalar direct variant with fused multiply-add, bit-identical to the SIMD
/// tiers per element.
#[arcane(import_intrinsics)]
fn matmuldirect_scalar(
    _: ScalarToken,
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) {
    for i0 in (0..m).step_by(BLOCKROWS) {
        let rows = (m - i0).min(BLOCKROWS);
        for r in 0..rows {
            for j in 0..n {
                let mut acc = 0.0_f32;
                for p in 0..k {
                    acc = f32::mul_add(a[(i0 + r) * k + p], b[p * n + j], acc);
                }
                out[(i0 + r) * n + j] = acc;
            }
        }
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
fn matmulblocked(a: &[f32], b: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    let mut packed = vec![0.0_f32; KCPANEL * NCPANEL];
    for jp in (0..n).step_by(NCPANEL) {
        let panel = (n - jp).min(NCPANEL);
        for kc0 in (0..k).step_by(KCPANEL) {
            let kchunk = (k - kc0).min(KCPANEL);
            packpanel(b, &mut packed, jp, panel, kc0, kchunk, n);
            for i0 in (0..m).step_by(BLOCKROWS) {
                let rows = (m - i0).min(BLOCKROWS);
                incant!(
                    matmultile(
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
                        n
                    ),
                    [v3, neon, scalar]
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
) -> Result<(), backenderror> {
    let nblocks = m.div_ceil(BLOCKROWS);
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
                        packpanel(b, &mut packed, jp, panel, kc0, kchunk, n);
                        for lb in 0..blocks {
                            let i0 = (t + lb * workers) * BLOCKROWS;
                            let rows = (m - i0).min(BLOCKROWS);
                            incant!(
                                matmultile(
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
                                ),
                                [v3, neon, scalar]
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
fn packpanel(
    b: &[f32],
    packed: &mut [f32],
    jp: usize,
    panel: usize,
    kc0: usize,
    kchunk: usize,
    n: usize,
) {
    for p in 0..kchunk {
        let src = &b[(kc0 + p) * n + jp..(kc0 + p) * n + jp + panel];
        packed[p * panel..p * panel + panel].copy_from_slice(src);
    }
}

/// Accumulates one packed inner chunk into `part[0..rows * n]`, the local
/// row-major block of the logical rows starting at `i0base`, reading the left
/// operand at its global row offsets.
///
/// The lane width differs per tier (8 on AVX2, 4 on NEON, 1 scalar), so the
/// per-element fused accumulation stays identical to the previous 4-lane
/// kernel for the tiled columns; only the columns past the last full tile fall
/// back to the same scalar dot product the old tail path used.
#[allow(clippy::too_many_arguments)]
#[arcane(import_intrinsics)]
fn matmultile_v3(
    _: X64V3Token,
    a: &[f32],
    b: &[f32],
    packed: &[f32],
    part: &mut [f32],
    i0base: usize,
    rows: usize,
    jp: usize,
    panel: usize,
    kc0: usize,
    kchunk: usize,
    k: usize,
    n: usize,
) {
    let ntile = panel / 8 * 8;
    for j0 in (0..ntile).step_by(8) {
        let mut acc = [_mm256_setzero_ps(); BLOCKROWS];
        for r in 0..rows {
            let mut pl = [0.0_f32; 8];
            pl.copy_from_slice(&part[r * n + jp + j0..r * n + jp + j0 + 8]);
            acc[r] = _mm256_loadu_ps(&pl);
        }
        for p in 0..kchunk {
            let mut bl = [0.0_f32; 8];
            bl.copy_from_slice(&packed[p * panel + j0..p * panel + j0 + 8]);
            let brow = _mm256_loadu_ps(&bl);
            for r in 0..rows {
                acc[r] =
                    _mm256_fmadd_ps(brow, _mm256_set1_ps(a[(i0base + r) * k + kc0 + p]), acc[r]);
            }
        }
        for r in 0..rows {
            let mut outl = [0.0_f32; 8];
            _mm256_storeu_ps(&mut outl, acc[r]);
            part[r * n + jp + j0..r * n + jp + j0 + 8].copy_from_slice(&outl);
        }
    }
    for r in 0..rows {
        for j in ntile..panel {
            part[r * n + jp + j] = scalardot(a, b, i0base + r, jp + j, k, n);
        }
    }
}

/// AArch64 4-lane packed variant.
#[allow(clippy::too_many_arguments)]
#[arcane(import_intrinsics)]
fn matmultile_neon(
    _: NeonToken,
    a: &[f32],
    b: &[f32],
    packed: &[f32],
    part: &mut [f32],
    i0base: usize,
    rows: usize,
    jp: usize,
    panel: usize,
    kc0: usize,
    kchunk: usize,
    k: usize,
    n: usize,
) {
    let ntile = panel / 4 * 4;
    for j0 in (0..ntile).step_by(4) {
        let mut acc = [vdupq_n_f32(0.0); BLOCKROWS];
        for r in 0..rows {
            let mut pl = [0.0_f32; 4];
            pl.copy_from_slice(&part[r * n + jp + j0..r * n + jp + j0 + 4]);
            acc[r] = vld1q_f32(&pl);
        }
        for p in 0..kchunk {
            let mut bl = [0.0_f32; 4];
            bl.copy_from_slice(&packed[p * panel + j0..p * panel + j0 + 4]);
            let brow = vld1q_f32(&bl);
            for r in 0..rows {
                acc[r] = vfmaq_f32(acc[r], brow, vdupq_n_f32(a[(i0base + r) * k + kc0 + p]));
            }
        }
        for r in 0..rows {
            let mut outl = [0.0_f32; 4];
            vst1q_f32(&mut outl, acc[r]);
            part[r * n + jp + j0..r * n + jp + j0 + 4].copy_from_slice(&outl);
        }
    }
    for r in 0..rows {
        for j in ntile..panel {
            part[r * n + jp + j] = scalardot(a, b, i0base + r, jp + j, k, n);
        }
    }
}

/// Scalar packed variant with fused multiply-add per element.
#[allow(clippy::too_many_arguments)]
#[arcane(import_intrinsics)]
fn matmultile_scalar(
    _: ScalarToken,
    a: &[f32],
    b: &[f32],
    _packed: &[f32],
    part: &mut [f32],
    i0base: usize,
    rows: usize,
    jp: usize,
    panel: usize,
    _kc0: usize,
    _kchunk: usize,
    k: usize,
    n: usize,
) {
    for r in 0..rows {
        for j in 0..panel {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc = f32::mul_add(a[(i0base + r) * k + p], b[p * n + jp + j], acc);
            }
            part[r * n + jp + j] = acc;
        }
    }
}

/// Scalar dot product of row `i` of `a` with column `j` of `b`, for the tail.
#[inline(always)]
fn scalardot(a: &[f32], b: &[f32], i: usize, j: usize, k: usize, n: usize) -> f32 {
    let mut acc = 0.0_f32;
    for p in 0..k {
        acc += a[i * k + p] * b[p * n + j];
    }
    acc
}
