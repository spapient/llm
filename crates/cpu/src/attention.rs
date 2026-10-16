use ::tensor::{shape, tensor, tensorview};
use api::backenderror;
use fearless_simd::{Level, Simd, dispatch, f32x4, prelude::*};

use crate::support::finish;

/// The stable name reported by [`cpuattention`].
const ATTENTIONOP: &str = "attention";

/// Query positions evaluated per block.
const QBLOCK: usize = 32;

/// Key/value positions evaluated per tile.
const KVBLOCK: usize = 256;

/// Multi-head scaled dot-product attention with grouped-query key/value heads.
///
/// `query` has shape `[seq, heads * headdim]`. `key` and `value` share shape
/// `[seq, kvheads * headdim]`, and `heads` MUST be a positive multiple of
/// `kvheads`: each key/value head serves `heads / kvheads` consecutive query
/// heads. When `causal` is set, query position `s` attends only to key
/// positions `t <= s`; masked positions are removed before normalization.
///
/// The scale is `1 / sqrt(headdim)`. Rows whose attention weights would be
/// entirely masked are left as zeros rather than dividing by zero.
///
/// Scores are evaluated in [`QBLOCK`]-row query blocks and [`KVBLOCK`]-key
/// tiles with an online maximum/sum rescale (one pass over the key axis per
/// tile, no materialised score matrix), so the softmax reassociation matches a
/// scalar reference within tolerance, not bit-for-bit. The result is
/// deterministic for fixed inputs, and every row's arithmetic sequence depends
/// only on its own attended key range, never on the query width or block
/// partitioning: an incremental single-token call reproduces the corresponding
/// full-sequence row bit-for-bit. Non-causal calls still evaluate every key
/// once per query position, so the worst-case complexity remains
/// `O(seq^2 * heads * headdim)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct cpuattention;

impl cpuattention {
    /// Applies grouped-query attention to `query`, `key`, and `value`.
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
        let plan = attentionplan(query, key, value, heads, kvheads, causal)?;
        let mut data = vec![0.0_f32; plan.qseq * plan.qstride];
        let level = Level::new();
        dispatch!(level, simd => attentionblocked(
            simd,
            &plan,
            query.getdata(),
            key.getdata(),
            value.getdata(),
            &mut data,
        ));
        let outshape =
            shape::new(vec![plan.qseq, plan.qstride]).map_err(|_| backenderror::sizeoverflow)?;
        finish(&outshape, data)
    }
}

/// Evaluates attention in query blocks of [`QBLOCK`] rows over key tiles of
/// [`KVBLOCK`] positions with an online softmax rescale.
///
/// Per row the running maximum, sum, and output accumulator are kept across
/// tiles; each tile rescales the accumulated state by the old/new maximum
/// ratio and adds the tile's contribution, so the key axis is read once per
/// query row and the value rows once per output dim. Heads of one group share
/// the same key/value tile and are evaluated back to back so the tile stays
/// resident across the group.
#[inline(always)]
fn attentionblocked<S: Simd>(
    simd: S,
    plan: &attentionshape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
) {
    let headdim = plan.headdim;
    let ntile = headdim / 4 * 4;
    let mut maxv = vec![0.0_f32; QBLOCK * plan.heads];
    let mut sumv = vec![0.0_f32; QBLOCK * plan.heads];
    let mut acc = vec![0.0_f32; QBLOCK * plan.heads * headdim];
    let mut weights = [0.0_f32; KVBLOCK];
    for qs in (0..plan.qseq).step_by(QBLOCK) {
        let qe = (qs + QBLOCK).min(plan.qseq);
        let rows = qe - qs;
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
                            let score = attentiondot(simd, qrow, krow, ntile) * plan.scale;
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
                        for d in (0..ntile).step_by(4) {
                            let a4 = f32x4::<S>::from_slice(
                                simd,
                                &acc[state * headdim + d..state * headdim + d + 4],
                            );
                            (a4 * f32x4::<S>::splat(simd, alpha)).store_slice(
                                &mut acc[state * headdim + d..state * headdim + d + 4],
                            );
                        }
                        for d in ntile..headdim {
                            acc[state * headdim + d] *= alpha;
                        }
                        for t in ks..end {
                            let vrow = &v[t * plan.vstride + kvhead * headdim..][..headdim];
                            let w = weights[t - ks];
                            for d in (0..ntile).step_by(4) {
                                let a4 = f32x4::<S>::from_slice(
                                    simd,
                                    &acc[state * headdim + d..state * headdim + d + 4],
                                );
                                let v4 = f32x4::<S>::from_slice(simd, &vrow[d..d + 4]);
                                (v4.mul_add(f32x4::<S>::splat(simd, w), a4)).store_slice(
                                    &mut acc[state * headdim + d..state * headdim + d + 4],
                                );
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
            let s = qs + i;
            let outrow = &mut out[s * plan.qstride..s * plan.qstride + plan.qstride];
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
}

/// Dot product of one query row and one key row, vectorized over the leading
/// `ntile` elements and scalar over the tail.
#[inline(always)]
fn attentiondot<S: Simd>(simd: S, a: &[f32], b: &[f32], ntile: usize) -> f32 {
    let mut acc = f32x4::<S>::splat(simd, 0.0);
    for i in (0..ntile).step_by(4) {
        let x = f32x4::<S>::from_slice(simd, &a[i..i + 4]);
        let y = f32x4::<S>::from_slice(simd, &b[i..i + 4]);
        acc = x.mul_add(y, acc);
    }
    let mut dot = acc.reduce_sum();
    for i in ntile..a.len() {
        dot += a[i] * b[i];
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
