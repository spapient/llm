use ::tensor::{shape, tensor, tensorview};
use api::backenderror;

use crate::support::finish;

/// The stable name reported by [`cpuattention`].
const ATTENTIONOP: &str = "attention";

/// Multi-head scaled dot-product attention with grouped-query key/value heads.
///
/// `query` has shape `[seq, heads * headdim]`. `key` and `value` share shape
/// `[seq, kvheads * headdim]`, and `heads` MUST be a positive multiple of
/// `kvheads`: each key/value head serves `heads / kvheads` consecutive query
/// heads. When `causal` is set, query position `s` attends only to key
/// positions `t <= s`; masked positions are removed before normalization.
///
/// The scale is `1 / sqrt(headdim)`. Rows whose attention weights would be
/// entirely masked are left as zeros rather than dividing by zero. The
/// implementation is the naive O(seq^2 * heads * headdim) algorithm; a
/// device-specific fused kernel is a later, measured optimization.
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
        let plan = attentionplan(query, key, value, heads, kvheads)?;
        let mut data = vec![0.0_f32; plan.seq * plan.qstride];
        let mut weights = vec![0.0_f32; plan.seq];
        let q = query.getdata();
        let k = key.getdata();
        let v = value.getdata();
        for head in 0..heads {
            let kvhead = head / plan.group;
            for s in 0..plan.seq {
                attentionrow(
                    Attention {
                        q: &q[s * plan.qstride + head * plan.headdim..][..plan.headdim],
                        k,
                        kstride: plan.kstride,
                        kvhead,
                        headdim: plan.headdim,
                        scale: plan.scale,
                        causal,
                    },
                    s,
                    plan.seq,
                    &mut weights,
                );
                let out = &mut data[s * plan.qstride + head * plan.headdim..][..plan.headdim];
                for (t, &p) in weights.iter().enumerate() {
                    if p != 0.0 {
                        let vrow = &v[t * plan.vstride + kvhead * plan.headdim..][..plan.headdim];
                        for (o, &x) in out.iter_mut().zip(vrow.iter()) {
                            *o += p * x;
                        }
                    }
                }
            }
        }
        let outshape =
            shape::new(vec![plan.seq, plan.qstride]).map_err(|_| backenderror::sizeoverflow)?;
        finish(&outshape, data)
    }
}

/// The immutable operands of one attention row, grouped to keep the row
/// function's parameter list small.
struct Attention<'a> {
    q: &'a [f32],
    k: &'a [f32],
    kstride: usize,
    kvhead: usize,
    headdim: usize,
    scale: f32,
    causal: bool,
}

/// Computes the attention weights of one query position into `weights`.
fn attentionrow(att: Attention<'_>, s: usize, seq: usize, weights: &mut [f32]) {
    let mut max = f32::NEG_INFINITY;
    for (t, w) in weights.iter_mut().enumerate().take(seq) {
        let value = if att.causal && t > s {
            f32::NEG_INFINITY
        } else {
            let krow = &att.k[t * att.kstride + att.kvhead * att.headdim..][..att.headdim];
            let dot: f32 = att.q.iter().zip(krow.iter()).map(|(&a, &b)| a * b).sum();
            dot * att.scale
        };
        *w = value;
        if value > max {
            max = value;
        }
    }
    let mut sum = 0.0_f32;
    for w in weights.iter_mut().take(seq) {
        let e = (*w - max).exp();
        *w = e;
        sum += e;
    }
    if sum > 0.0 {
        for w in weights.iter_mut().take(seq) {
            *w /= sum;
        }
    } else {
        for w in weights.iter_mut().take(seq) {
            *w = 0.0;
        }
    }
}

/// The validated shape plan of one attention call.
struct attentionshape {
    seq: usize,
    qstride: usize,
    kstride: usize,
    vstride: usize,
    headdim: usize,
    group: usize,
    scale: f32,
}

fn attentionplan(
    query: &tensorview<'_, f32>,
    key: &tensorview<'_, f32>,
    value: &tensorview<'_, f32>,
    heads: usize,
    kvheads: usize,
) -> Result<attentionshape, backenderror> {
    let qd = query.getshape().getdims();
    let kd = key.getshape().getdims();
    let vd = value.getshape().getdims();
    if qd.len() != 2 || kd.len() != 2 || vd.len() != 2 {
        return Err(backenderror::shapemismatch { op: ATTENTIONOP });
    }
    if kd != vd || kd[0] != qd[0] {
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
        seq: qd[0],
        qstride: qd[1],
        kstride: kd[1],
        vstride: vd[1],
        headdim,
        group: heads / kvheads,
        scale: 1.0 / (headdim as f32).sqrt(),
    })
}
