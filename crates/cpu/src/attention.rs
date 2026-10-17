use ::tensor::{shape, tensor, tensorview};
use api::backenderror;
use simd;

use crate::support::finish;

/// The stable name reported by [`cpuattention`].
const ATTENTIONOP: &str = "attention";

/// Query row-block size of the blocked kernel.
const QBLOCK: usize = 32;

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
    plan: &simd::AttentionShape,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    workers: usize,
) -> Result<(), backenderror> {
    let isa = simd::detect();
    if workers > 1 && plan.qseq >= 2 * QBLOCK {
        attentionblockedparallel(isa, plan, q, k, v, out, workers)
    } else {
        for qs in (0..plan.qseq).step_by(QBLOCK) {
            let qe = (qs + QBLOCK).min(plan.qseq);
            simd::attentionqblock(
                isa,
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
/// Workers inherit the caller's detected tier, so each query block still
/// crosses the feature boundary only once.
fn attentionblockedparallel(
    isa: simd::Isa,
    plan: &simd::AttentionShape,
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
                    simd::attentionqblock(
                        isa,
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
    plan: &simd::AttentionShape,
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

fn attentionplan(
    query: &tensorview<'_, f32>,
    key: &tensorview<'_, f32>,
    value: &tensorview<'_, f32>,
    heads: usize,
    kvheads: usize,
    causal: bool,
) -> Result<simd::AttentionShape, backenderror> {
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
    Ok(simd::AttentionShape {
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
