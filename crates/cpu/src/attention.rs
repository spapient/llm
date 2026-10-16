use ::tensor::{shape, tensor, tensorview};
use api::backenderror;
use archmage::prelude::*;

use crate::support::finish;

/// The stable name reported by [`cpuattention`].
const ATTENTIONOP: &str = "attention";

/// Query row-block size of the blocked kernel.
const QBLOCK: usize = 32;

/// Key tile size of the online-rescaled softmax.
const KVBLOCK: usize = 256;

/// Grouped-query attention.
///
/// The query rows are evaluated in blocks of at most [`QBLOCK`] rows. Every
/// key tile walks at most [`KVBLOCK`] key positions and maintains the online
/// softmax state (`max`, `sum`, output accumulator) described below, so the
/// full `head × head` attention matrix is never materialized.
///
/// When `causal` is true, query position `s` may attend to keys up to
/// `offset + s` where `offset = kvseq - qseq`; the kernel skips the dot
/// products of excluded positions entirely. Non-causal attention attends to
/// every key.
#[derive(Clone, Copy, Debug, Default)]
pub struct cpuattention;

impl cpuattention {
    /// Applies grouped-query attention serially.
    ///
    /// # Errors
    ///
    /// Returns [`backenderror::shapemismatch`] when the ranks are not two, the
    /// sequence lengths differ, the head counts are inconsistent, or the query
    /// last axis is not a multiple of `heads`.
    pub fn apply(
        &self,
        query: &tensorview<'_, f32>,
        key: &tensorview<'_, f32>,
        value: &tensorview<'_, f32>,
        heads: usize,
        kvheads: usize,
        causal: bool,
    ) -> Result<tensor<f32>, backenderror> {
        self.applywith(query, key, value, heads, kvheads, causal, 1)
    }

    /// Applies grouped-query attention, letting the blocked kernel use up to
    /// `workers` scoped threads over query blocks.
    ///
    /// Every row's arithmetic sequence is identical to the serial kernel, so
    /// the result is bit-for-bit independent of `workers`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn applywith(
        &self,
        query: &tensorview<'_, f32>,
        key: &tensorview<'_, f32>,
        value: &tensorview<'_, f32>,
        heads: usize,
        kvheads: usize,
        causal: bool,
        workers: usize,
    ) -> Result<tensor<f32>, backenderror> {
        let plan = attentionplan(query, key, value, heads, kvheads, causal)?;
        let mut data = vec![0.0_f32; plan.qseq * plan.qstride];
        attentionblocked(
            &plan,
            query.getdata(),
            key.getdata(),
            value.getdata(),
            &mut data,
            workers,
        )?;
        finish(
            &shape::new(vec![plan.qseq, plan.qstride]).map_err(|_| backenderror::sizeoverflow)?,
            data,
        )
    }
}

/// Computes one attention call into `out` with a query-blocked kernel, picking
/// the widest CPU tier available.
///
/// The tier proof is summoned once per call; every query block then reuses the
/// same token, so the `#[target_feature]` boundary is crossed once per query
/// block instead of once per query-key pair.
fn attentionblocked(
    plan: &attentionshape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    workers: usize,
) -> Result<(), backenderror> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    if let Some(token) = X64V3Token::summon() {
        return attentionblocked_v3(token, plan, q, k, v, out, workers);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(token) = NeonToken::summon() {
        return attentionblocked_neon(token, plan, q, k, v, out, workers);
    }
    attentionblocked_scalar(ScalarToken, plan, q, k, v, out, workers)
}

/// AVX2 8-lane blocked kernel.
#[arcane(import_intrinsics)]
fn attentionblocked_v3(
    token: X64V3Token,
    plan: &attentionshape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    workers: usize,
) -> Result<(), backenderror> {
    if workers > 1 && plan.qseq >= 2 * QBLOCK {
        attentionblockedparallel_v3(token, plan, q, k, v, out, workers)
    } else {
        for qs in (0..plan.qseq).step_by(QBLOCK) {
            let qe = (qs + QBLOCK).min(plan.qseq);
            attentionqblock_v3(
                token,
                plan,
                q,
                k,
                v,
                &mut out[qs * plan.qstride..qe * plan.qstride],
                qs,
                qe,
            );
        }
        Ok(())
    }
}

/// AArch64 4-lane blocked kernel.
#[arcane(import_intrinsics)]
fn attentionblocked_neon(
    token: NeonToken,
    plan: &attentionshape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    workers: usize,
) -> Result<(), backenderror> {
    if workers > 1 && plan.qseq >= 2 * QBLOCK {
        attentionblockedparallel_neon(token, plan, q, k, v, out, workers)
    } else {
        for qs in (0..plan.qseq).step_by(QBLOCK) {
            let qe = (qs + QBLOCK).min(plan.qseq);
            attentionqblock_neon(
                token,
                plan,
                q,
                k,
                v,
                &mut out[qs * plan.qstride..qe * plan.qstride],
                qs,
                qe,
            );
        }
        Ok(())
    }
}

/// Scalar blocked kernel.
#[arcane(import_intrinsics)]
fn attentionblocked_scalar(
    token: ScalarToken,
    plan: &attentionshape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    workers: usize,
) -> Result<(), backenderror> {
    if workers > 1 && plan.qseq >= 2 * QBLOCK {
        attentionblockedparallel_scalar(token, plan, q, k, v, out, workers)
    } else {
        for qs in (0..plan.qseq).step_by(QBLOCK) {
            let qe = (qs + QBLOCK).min(plan.qseq);
            attentionqblock_scalar(
                token,
                plan,
                q,
                k,
                v,
                &mut out[qs * plan.qstride..qe * plan.qstride],
                qs,
                qe,
            );
        }
        Ok(())
    }
}

/// The blocked kernel with the query blocks split over `workers` scoped
/// threads.
///
/// Each thread evaluates its strided query blocks into a private partial
/// output laid out row-major like `out`; the partials are merged after the
/// threads join, in block order. No mutable state is shared between threads.
/// Workers inherit the caller's SIMD tier token, so each query block still
/// crosses the feature boundary only once.
#[arcane(import_intrinsics)]
fn attentionblockedparallel_v3(
    token: X64V3Token,
    plan: &attentionshape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    workers: usize,
) -> Result<(), backenderror> {
    let nblocks = plan.qseq.div_ceil(QBLOCK);
    std::thread::scope(|scope| {
        let slots: Vec<(usize, usize, Vec<f32>)> = Vec::with_capacity(workers);
        let mut handles = Vec::with_capacity(workers);
        for t in 0..workers {
            let blocks = (nblocks - t).div_ceil(workers);
            if blocks == 0 {
                continue;
            }
            let mut part = vec![0.0_f32; blocks * QBLOCK * plan.qstride];
            handles.push(scope.spawn(move || {
                for lb in 0..blocks {
                    let qs = (t + lb * workers) * QBLOCK;
                    let qe = (qs + QBLOCK).min(plan.qseq);
                    attentionqblock_v3(
                        token,
                        plan,
                        q,
                        k,
                        v,
                        &mut part[lb * QBLOCK * plan.qstride..],
                        qs,
                        qe,
                    );
                }
                (t, blocks, part)
            }));
        }
        mergepartials(plan, out, workers, handles, slots)
    })
}

/// NEON parallel variant.
#[arcane(import_intrinsics)]
fn attentionblockedparallel_neon(
    token: NeonToken,
    plan: &attentionshape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    workers: usize,
) -> Result<(), backenderror> {
    let nblocks = plan.qseq.div_ceil(QBLOCK);
    std::thread::scope(|scope| {
        let slots: Vec<(usize, usize, Vec<f32>)> = Vec::with_capacity(workers);
        let mut handles = Vec::with_capacity(workers);
        for t in 0..workers {
            let blocks = (nblocks - t).div_ceil(workers);
            if blocks == 0 {
                continue;
            }
            let mut part = vec![0.0_f32; blocks * QBLOCK * plan.qstride];
            handles.push(scope.spawn(move || {
                for lb in 0..blocks {
                    let qs = (t + lb * workers) * QBLOCK;
                    let qe = (qs + QBLOCK).min(plan.qseq);
                    attentionqblock_neon(
                        token,
                        plan,
                        q,
                        k,
                        v,
                        &mut part[lb * QBLOCK * plan.qstride..],
                        qs,
                        qe,
                    );
                }
                (t, blocks, part)
            }));
        }
        mergepartials(plan, out, workers, handles, slots)
    })
}

/// Scalar parallel variant.
#[arcane(import_intrinsics)]
fn attentionblockedparallel_scalar(
    token: ScalarToken,
    plan: &attentionshape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    workers: usize,
) -> Result<(), backenderror> {
    let nblocks = plan.qseq.div_ceil(QBLOCK);
    std::thread::scope(|scope| {
        let slots: Vec<(usize, usize, Vec<f32>)> = Vec::with_capacity(workers);
        let mut handles = Vec::with_capacity(workers);
        for t in 0..workers {
            let blocks = (nblocks - t).div_ceil(workers);
            if blocks == 0 {
                continue;
            }
            let mut part = vec![0.0_f32; blocks * QBLOCK * plan.qstride];
            handles.push(scope.spawn(move || {
                for lb in 0..blocks {
                    let qs = (t + lb * workers) * QBLOCK;
                    let qe = (qs + QBLOCK).min(plan.qseq);
                    attentionqblock_scalar(
                        token,
                        plan,
                        q,
                        k,
                        v,
                        &mut part[lb * QBLOCK * plan.qstride..],
                        qs,
                        qe,
                    );
                }
                (t, blocks, part)
            }));
        }
        mergepartials(plan, out, workers, handles, slots)
    })
}

/// Joins the worker handles and copies the partial row blocks into `out` in
/// block order. Returns an error if any worker panicked.
#[allow(clippy::type_complexity)]
fn mergepartials(
    plan: &attentionshape,
    out: &mut [f32],
    workers: usize,
    handles: Vec<std::thread::ScopedJoinHandle<'_, (usize, usize, Vec<f32>)>>,
    mut slots: Vec<(usize, usize, Vec<f32>)>,
) -> Result<(), backenderror> {
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
                let qs = (t + lb * workers) * QBLOCK;
                let qe = (qs + QBLOCK).min(plan.qseq);
                let rows = qe - qs;
                out[qs * plan.qstride..qe * plan.qstride].copy_from_slice(
                    &part[lb * QBLOCK * plan.qstride..(lb * QBLOCK + rows) * plan.qstride],
                );
            }
        }
    }
    if failed {
        return Err(backenderror::executionfailed {
            backend: "cpu",
            kernel: ATTENTIONOP.into(),
            message: "an attention worker thread panicked".into(),
        });
    }
    Ok(())
}

/// Evaluates one query block `[qs, qe)` into `part[0..(qe - qs) * qstride]`,
/// where `part` is the block's row-major output slice. The key axis is walked
/// in tiles with the online rescale described on [`cpuattention`]: for every
/// tile the running maximum advances and the accumulated output is rescaled by
/// `exp(old - new)`, then the tile's value rows are fused into the accumulator.
#[allow(clippy::too_many_arguments)]
#[arcane(import_intrinsics)]
fn attentionqblock_v3(
    token: X64V3Token,
    plan: &attentionshape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    part: &mut [f32],
    qs: usize,
    qe: usize,
) {
    let headdim = plan.headdim;
    let ntile = headdim / 8 * 8;
    let rows = qe - qs;
    let mut maxv = vec![0.0_f32; QBLOCK * plan.heads];
    let mut sumv = vec![0.0_f32; QBLOCK * plan.heads];
    let mut acc = vec![0.0_f32; QBLOCK * plan.heads * headdim];
    let mut weights = [0.0_f32; KVBLOCK];
    for i in 0..rows {
        for head in 0..plan.heads {
            maxv[i * plan.heads + head] = f32::NEG_INFINITY;
            sumv[i * plan.heads + head] = 0.0;
            for d in 0..headdim {
                acc[(i * plan.heads + head) * headdim + d] = 0.0;
            }
        }
    }
    for ks in (0..plan.kvseq).step_by(KVBLOCK) {
        let ke = (ks + KVBLOCK).min(plan.kvseq);
        // A causal tile entirely after every row's limit contributes
        // nothing to this query block.
        if plan.causal && plan.offset + qe <= ks {
            continue;
        }
        for kvhead in 0..plan.kvheads {
            for head in kvhead * plan.group..(kvhead + 1) * plan.group {
                for i in 0..rows {
                    let s = qs + i;
                    let end = attentionlimit(plan.causal, s, plan.kvseq, plan.qseq).min(ke);
                    if end <= ks {
                        continue;
                    }
                    let state = i * plan.heads + head;
                    let qrow = &q[s * plan.qstride + head * headdim..][..headdim];
                    let mut rowmax = f32::NEG_INFINITY;
                    for t in ks..end {
                        let krow = &k[t * plan.kstride + kvhead * headdim..][..headdim];
                        let score = attentiondot_v3(token, qrow, krow, ntile, headdim) * plan.scale;
                        weights[t - ks] = score;
                        if score > rowmax {
                            rowmax = score;
                        }
                    }
                    let newmax = if rowmax > maxv[state] {
                        rowmax
                    } else {
                        maxv[state]
                    };
                    let alpha = (maxv[state] - newmax).exp();
                    let mut tilesum = 0.0_f32;
                    for t in ks..end {
                        let w = (weights[t - ks] - newmax).exp();
                        weights[t - ks] = w;
                        tilesum += w;
                    }
                    // Rescale the accumulated output by the old/new
                    // maximum ratio, then add the tile's value rows in
                    // key order so the value operand streams row-major.
                    let alphav = _mm256_set1_ps(alpha);
                    for d in (0..ntile).step_by(8) {
                        let mut al = [0.0_f32; 8];
                        al.copy_from_slice(&acc[state * headdim + d..state * headdim + d + 8]);
                        let a8 = _mm256_loadu_ps(&al);
                        let mut outl = [0.0_f32; 8];
                        _mm256_storeu_ps(&mut outl, _mm256_mul_ps(a8, alphav));
                        acc[state * headdim + d..state * headdim + d + 8].copy_from_slice(&outl);
                    }
                    for d in ntile..headdim {
                        acc[state * headdim + d] *= alpha;
                    }
                    for t in ks..end {
                        let vrow = &v[t * plan.vstride + kvhead * headdim..][..headdim];
                        let w = weights[t - ks];
                        let wv = _mm256_set1_ps(w);
                        for d in (0..ntile).step_by(8) {
                            let mut al = [0.0_f32; 8];
                            let mut vl = [0.0_f32; 8];
                            al.copy_from_slice(&acc[state * headdim + d..state * headdim + d + 8]);
                            vl.copy_from_slice(&vrow[d..d + 8]);
                            let a8 = _mm256_loadu_ps(&al);
                            let v8 = _mm256_loadu_ps(&vl);
                            let mut outl = [0.0_f32; 8];
                            _mm256_storeu_ps(&mut outl, _mm256_fmadd_ps(v8, wv, a8));
                            acc[state * headdim + d..state * headdim + d + 8]
                                .copy_from_slice(&outl);
                        }
                        for d in ntile..headdim {
                            acc[state * headdim + d] += w * vrow[d];
                        }
                    }
                    maxv[state] = newmax;
                    sumv[state] = sumv[state] * alpha + tilesum;
                }
            }
        }
    }
    for i in 0..rows {
        let outrow = &mut part[i * plan.qstride..(i + 1) * plan.qstride];
        for head in 0..plan.heads {
            let state = i * plan.heads + head;
            if sumv[state] > 0.0 {
                for d in 0..headdim {
                    outrow[head * headdim + d] = acc[state * headdim + d] / sumv[state];
                }
            }
        }
    }
}

/// NEON query-block variant.
#[allow(clippy::too_many_arguments)]
#[arcane(import_intrinsics)]
fn attentionqblock_neon(
    token: NeonToken,
    plan: &attentionshape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    part: &mut [f32],
    qs: usize,
    qe: usize,
) {
    let headdim = plan.headdim;
    let ntile = headdim / 4 * 4;
    let rows = qe - qs;
    let mut maxv = vec![0.0_f32; QBLOCK * plan.heads];
    let mut sumv = vec![0.0_f32; QBLOCK * plan.heads];
    let mut acc = vec![0.0_f32; QBLOCK * plan.heads * headdim];
    let mut weights = [0.0_f32; KVBLOCK];
    for i in 0..rows {
        for head in 0..plan.heads {
            maxv[i * plan.heads + head] = f32::NEG_INFINITY;
            sumv[i * plan.heads + head] = 0.0;
            for d in 0..headdim {
                acc[(i * plan.heads + head) * headdim + d] = 0.0;
            }
        }
    }
    for ks in (0..plan.kvseq).step_by(KVBLOCK) {
        let ke = (ks + KVBLOCK).min(plan.kvseq);
        if plan.causal && plan.offset + qe <= ks {
            continue;
        }
        for kvhead in 0..plan.kvheads {
            for head in kvhead * plan.group..(kvhead + 1) * plan.group {
                for i in 0..rows {
                    let s = qs + i;
                    let end = attentionlimit(plan.causal, s, plan.kvseq, plan.qseq).min(ke);
                    if end <= ks {
                        continue;
                    }
                    let state = i * plan.heads + head;
                    let qrow = &q[s * plan.qstride + head * headdim..][..headdim];
                    let mut rowmax = f32::NEG_INFINITY;
                    for t in ks..end {
                        let krow = &k[t * plan.kstride + kvhead * headdim..][..headdim];
                        let score =
                            attentiondot_neon(token, qrow, krow, ntile, headdim) * plan.scale;
                        weights[t - ks] = score;
                        if score > rowmax {
                            rowmax = score;
                        }
                    }
                    let newmax = if rowmax > maxv[state] {
                        rowmax
                    } else {
                        maxv[state]
                    };
                    let alpha = (maxv[state] - newmax).exp();
                    let mut tilesum = 0.0_f32;
                    for t in ks..end {
                        let w = (weights[t - ks] - newmax).exp();
                        weights[t - ks] = w;
                        tilesum += w;
                    }
                    let alphav = vdupq_n_f32(alpha);
                    for d in (0..ntile).step_by(4) {
                        let mut al = [0.0_f32; 4];
                        al.copy_from_slice(&acc[state * headdim + d..state * headdim + d + 4]);
                        let a4 = vld1q_f32(&al);
                        let mut outl = [0.0_f32; 4];
                        vst1q_f32(&mut outl, vmulq_f32(a4, alphav));
                        acc[state * headdim + d..state * headdim + d + 4].copy_from_slice(&outl);
                    }
                    for d in ntile..headdim {
                        acc[state * headdim + d] *= alpha;
                    }
                    for t in ks..end {
                        let vrow = &v[t * plan.vstride + kvhead * headdim..][..headdim];
                        let w = weights[t - ks];
                        let wv = vdupq_n_f32(w);
                        for d in (0..ntile).step_by(4) {
                            let mut al = [0.0_f32; 4];
                            let mut vl = [0.0_f32; 4];
                            al.copy_from_slice(&acc[state * headdim + d..state * headdim + d + 4]);
                            vl.copy_from_slice(&vrow[d..d + 4]);
                            let a4 = vld1q_f32(&al);
                            let v4 = vld1q_f32(&vl);
                            let mut outl = [0.0_f32; 4];
                            vst1q_f32(&mut outl, vfmaq_f32(a4, v4, wv));
                            acc[state * headdim + d..state * headdim + d + 4]
                                .copy_from_slice(&outl);
                        }
                        for d in ntile..headdim {
                            acc[state * headdim + d] += w * vrow[d];
                        }
                    }
                    maxv[state] = newmax;
                    sumv[state] = sumv[state] * alpha + tilesum;
                }
            }
        }
    }
    for i in 0..rows {
        let outrow = &mut part[i * plan.qstride..(i + 1) * plan.qstride];
        for head in 0..plan.heads {
            let state = i * plan.heads + head;
            if sumv[state] > 0.0 {
                for d in 0..headdim {
                    outrow[head * headdim + d] = acc[state * headdim + d] / sumv[state];
                }
            }
        }
    }
}

/// Scalar query-block variant (fused multiply-add keeps the accumulation
/// sequence identical to the SIMD tiers per element).
#[allow(clippy::too_many_arguments)]
#[arcane(import_intrinsics)]
fn attentionqblock_scalar(
    token: ScalarToken,
    plan: &attentionshape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    part: &mut [f32],
    qs: usize,
    qe: usize,
) {
    let headdim = plan.headdim;
    let rows = qe - qs;
    let mut maxv = vec![0.0_f32; QBLOCK * plan.heads];
    let mut sumv = vec![0.0_f32; QBLOCK * plan.heads];
    let mut acc = vec![0.0_f32; QBLOCK * plan.heads * headdim];
    let mut weights = [0.0_f32; KVBLOCK];
    for i in 0..rows {
        for head in 0..plan.heads {
            maxv[i * plan.heads + head] = f32::NEG_INFINITY;
            sumv[i * plan.heads + head] = 0.0;
            for d in 0..headdim {
                acc[(i * plan.heads + head) * headdim + d] = 0.0;
            }
        }
    }
    for ks in (0..plan.kvseq).step_by(KVBLOCK) {
        let ke = (ks + KVBLOCK).min(plan.kvseq);
        if plan.causal && plan.offset + qe <= ks {
            continue;
        }
        for kvhead in 0..plan.kvheads {
            for head in kvhead * plan.group..(kvhead + 1) * plan.group {
                for i in 0..rows {
                    let s = qs + i;
                    let end = attentionlimit(plan.causal, s, plan.kvseq, plan.qseq).min(ke);
                    if end <= ks {
                        continue;
                    }
                    let state = i * plan.heads + head;
                    let qrow = &q[s * plan.qstride + head * headdim..][..headdim];
                    let mut rowmax = f32::NEG_INFINITY;
                    for t in ks..end {
                        let krow = &k[t * plan.kstride + kvhead * headdim..][..headdim];
                        let score = attentiondot_scalar(token, qrow, krow, 0, headdim) * plan.scale;
                        weights[t - ks] = score;
                        if score > rowmax {
                            rowmax = score;
                        }
                    }
                    let newmax = if rowmax > maxv[state] {
                        rowmax
                    } else {
                        maxv[state]
                    };
                    let alpha = (maxv[state] - newmax).exp();
                    let mut tilesum = 0.0_f32;
                    for t in ks..end {
                        let w = (weights[t - ks] - newmax).exp();
                        weights[t - ks] = w;
                        tilesum += w;
                    }
                    for d in 0..headdim {
                        acc[state * headdim + d] *= alpha;
                    }
                    for t in ks..end {
                        let vrow = &v[t * plan.vstride + kvhead * headdim..][..headdim];
                        let w = weights[t - ks];
                        for d in 0..headdim {
                            acc[state * headdim + d] =
                                f32::mul_add(w, vrow[d], acc[state * headdim + d]);
                        }
                    }
                    maxv[state] = newmax;
                    sumv[state] = sumv[state] * alpha + tilesum;
                }
            }
        }
    }
    for i in 0..rows {
        let outrow = &mut part[i * plan.qstride..(i + 1) * plan.qstride];
        for head in 0..plan.heads {
            let state = i * plan.heads + head;
            if sumv[state] > 0.0 {
                for d in 0..headdim {
                    outrow[head * headdim + d] = acc[state * headdim + d] / sumv[state];
                }
            }
        }
    }
}

/// Dot product of one query row and one key row, vectorized over the leading
/// `ntile` elements and scalar over the tail. The lane reduction is a
/// fixed-order tree on the caller's tier; the tail follows the same
/// left-to-right accumulation.
#[arcane(import_intrinsics)]
fn attentiondot_v3(_: X64V3Token, a: &[f32], b: &[f32], ntile: usize, len: usize) -> f32 {
    let mut acc = _mm256_setzero_ps();
    for i in (0..ntile).step_by(8) {
        let mut al = [0.0_f32; 8];
        let mut bl = [0.0_f32; 8];
        al.copy_from_slice(&a[i..i + 8]);
        bl.copy_from_slice(&b[i..i + 8]);
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(&al), _mm256_loadu_ps(&bl), acc);
    }
    let low = _mm256_castps256_ps128(acc);
    let high = _mm256_extractf128_ps::<1>(acc);
    let q = _mm_add_ps(low, high);
    let shuf = _mm_movehdup_ps(q);
    let sums = _mm_add_ps(q, shuf);
    let shuf = _mm_movehl_ps(shuf, sums);
    let mut dot = _mm_cvtss_f32(_mm_add_ss(sums, shuf));
    for i in ntile..len {
        dot += a[i] * b[i];
    }
    dot
}

/// NEON dot-product variant.
#[arcane(import_intrinsics)]
fn attentiondot_neon(_: NeonToken, a: &[f32], b: &[f32], ntile: usize, len: usize) -> f32 {
    let mut acc = vdupq_n_f32(0.0);
    for i in (0..ntile).step_by(4) {
        let mut al = [0.0_f32; 4];
        let mut bl = [0.0_f32; 4];
        al.copy_from_slice(&a[i..i + 4]);
        bl.copy_from_slice(&b[i..i + 4]);
        acc = vfmaq_f32(acc, vld1q_f32(&al), vld1q_f32(&bl));
    }
    let mut dot = vaddvq_f32(acc);
    for i in ntile..len {
        dot += a[i] * b[i];
    }
    dot
}

/// Scalar dot-product variant with fused multiply-add.
#[arcane(import_intrinsics)]
fn attentiondot_scalar(_: ScalarToken, a: &[f32], b: &[f32], _ntile: usize, len: usize) -> f32 {
    let mut dot = 0.0_f32;
    for i in 0..len {
        dot = f32::mul_add(a[i], b[i], dot);
    }
    dot
}

/// Returns how many key positions a query position may attend to.
///
/// Causal attention lets query position `s` see keys up to `offset + s`, where
/// `offset = kvseq - qseq` is the gap between the query and key lengths; keys
/// after that are excluded, so their dot products are never computed and their
/// weight slots are never written. Non-causal attention attends to every key.
fn attentionlimit(causal: bool, s: usize, kvseq: usize, qseq: usize) -> usize {
    if causal {
        let offset = kvseq - qseq;
        (offset + s + 1).min(kvseq)
    } else {
        kvseq
    }
}

/// The validated shape plan of one attention call.
struct attentionshape {
    qseq: usize,
    kvseq: usize,
    qstride: usize,
    kstride: usize,
    vstride: usize,
    heads: usize,
    kvheads: usize,
    headdim: usize,
    group: usize,
    scale: f32,
    offset: usize,
    causal: bool,
}

fn attentionplan(
    query: &tensorview<'_, f32>,
    key: &tensorview<'_, f32>,
    value: &tensorview<'_, f32>,
    heads: usize,
    kvheads: usize,
    causal: bool,
) -> Result<attentionshape, backenderror> {
    let qd = query.getshape().getdims();
    let kd = key.getshape().getdims();
    let vd = value.getshape().getdims();
    if qd.len() != 2 || kd.len() != 2 || vd.len() != 2 {
        return Err(backenderror::shapemismatch { op: ATTENTIONOP });
    }
    if kd != vd || qd[0] > kd[0] {
        return Err(backenderror::shapemismatch { op: ATTENTIONOP });
    }
    if heads == 0 || kvheads == 0 || !heads.is_multiple_of(kvheads) {
        return Err(backenderror::shapemismatch { op: ATTENTIONOP });
    }
    if !qd[1].is_multiple_of(heads) {
        return Err(backenderror::shapemismatch { op: ATTENTIONOP });
    }
    let headdim = qd[1] / heads;
    if headdim == 0 {
        return Err(backenderror::shapemismatch { op: ATTENTIONOP });
    }
    let kwidth = kvheads
        .checked_mul(headdim)
        .ok_or(backenderror::sizeoverflow)?;
    if kd[1] != kwidth {
        return Err(backenderror::shapemismatch { op: ATTENTIONOP });
    }
    Ok(attentionshape {
        qseq: qd[0],
        kvseq: kd[0],
        qstride: qd[1],
        kstride: kd[1],
        vstride: vd[1],
        heads,
        kvheads,
        headdim,
        group: heads / kvheads,
        scale: 1.0 / (headdim as f32).sqrt(),
        offset: kd[0] - qd[0],
        causal,
    })
}
