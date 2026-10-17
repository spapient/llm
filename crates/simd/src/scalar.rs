//! Scalar fallback tier of the SIMD library.
//!
//! Plain safe code; the compiler's auto-vectorizer applies to these simple
//! loops, so the fallback is never wasted. The per-element arithmetic order is
//! identical to the SIMD tiers (fused multiply-add, left-to-right tail
//! accumulation), so results are bit-for-bit comparable.

use crate::AttentionShape;

/// RMS-normalizes one `row` into `out`, scaled by `weight`, epsilon `eps`.
pub fn rmsnormrow(row: &[f32], out: &mut [f32], weight: &[f32], eps: f32) {
    let n = row.len();
    let mut sumsq = 0.0_f32;
    for &x in row {
        sumsq += x * x;
    }
    let inv = 1.0 / (sumsq / n as f32 + eps).sqrt();
    for i in 0..n {
        out[i] = row[i] * inv * weight[i];
    }
}

/// Reciprocal RMS scale of one row.
pub fn rowinv(row: &[f32], eps: f32) -> f32 {
    let n = row.len();
    let mut sumsq = 0.0_f32;
    for &x in row {
        sumsq += x * x;
    }
    1.0 / (sumsq / n as f32 + eps).sqrt()
}

/// The scalar softmax row: `f32::max` fold (skips NaN), libm `exp`, and the
/// zero-sum guard that propagates the `-inf` row's NaN.
pub fn softmaxrow(row: &[f32], dst: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for (d, &x) in dst.iter_mut().zip(row.iter()) {
        let e = (x - max).exp();
        *d = e;
        sum += e;
    }
    if sum > 0.0 {
        for d in dst.iter_mut() {
            *d /= sum;
        }
    }
}

/// Softmax-normalizes rows `[start, end)` of `src` into `data`, width `width`.
pub fn softmaxrowspan(src: &[f32], data: &mut [f32], width: usize, start: usize, end: usize) {
    for row in start..end {
        softmaxrow(
            &src[row * width..(row + 1) * width],
            &mut data[row * width..(row + 1) * width],
        );
    }
}

/// Scalar dot product with fused multiply-add.
pub fn attentiondot(a: &[f32], b: &[f32], _ntile: usize, len: usize) -> f32 {
    let mut dot = 0.0_f32;
    for i in 0..len {
        dot = f32::mul_add(a[i], b[i], dot);
    }
    dot
}

/// Scalar grouped-query attention query block `[qs, qe)` (fused multiply-add
/// keeps the accumulation sequence identical to the SIMD tiers per element).
#[allow(clippy::too_many_arguments)]
pub fn attentionqblock(
    plan: &AttentionShape,
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
                        let score = attentiondot(qrow, krow, 0, headdim) * plan.scale;
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

/// Query row-block size of the blocked attention kernel.
const QBLOCK: usize = 32;

/// Key tile size of the online-rescaled attention softmax.
const KVBLOCK: usize = 256;

/// Returns how many key positions a query position may attend to.
fn attentionlimit(causal: bool, s: usize, kvseq: usize, qseq: usize) -> usize {
    if causal {
        let offset = kvseq - qseq;
        (offset + s + 1).min(kvseq)
    } else {
        kvseq
    }
}

/// Scalar direct register-blocked matmul micro-kernel.
#[allow(clippy::too_many_arguments)]
pub fn matmuldirect(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    rowscale: Option<&[f32]>,
) {
    for i0 in (0..m).step_by(8) {
        let rows = (m - i0).min(8);
        for r in 0..rows {
            let s = rowscale.map(|scales| scales[i0 + r]);
            for j in 0..n {
                let mut acc = 0.0_f32;
                for p in 0..k {
                    acc = f32::mul_add(a[(i0 + r) * k + p], b[p * n + j], acc);
                }
                out[(i0 + r) * n + j] = match s {
                    Some(s) => acc * s,
                    None => acc,
                };
            }
        }
    }
}

/// Scalar blocked matmul micro-kernel with fused multiply-add per element.
/// The packed tile is not used on this tier (matching the source kernel);
/// the left operand is read at its global row offsets and the right operand
/// directly from `b`'s strided layout.
#[allow(clippy::too_many_arguments)]
pub fn matmultile(
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
    rowscale: Option<&[f32]>,
    lastchunk: bool,
) {
    for r in 0..rows {
        let s = (lastchunk)
            .then(|| rowscale.map(|scales| scales[i0base + r]))
            .flatten();
        for j in 0..panel {
            let mut acc = 0.0_f32;
            for p in 0..k {
                acc = f32::mul_add(a[(i0base + r) * k + p], b[p * n + jp + j], acc);
            }
            part[r * n + jp + j] = match s {
                Some(s) => acc * s,
                None => acc,
            };
        }
    }
}
