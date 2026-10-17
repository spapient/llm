//! The device-resident accumulation layer: the [`session`] that chains
//! launches with deferred synchronisation, and every `*_record` step.
//!
//! All steps are safe: they describe their operands as [`buf`] and forward
//! to [`drive`](drive::drive) or the safe [`flush`] / [`retiredevice`]
//! wrappers, so this file needs no `allow(unsafe_code)`.
use std::sync::Arc;

use super::drive::drive;
use super::*;
use crate::engine::gpuerror;
use crate::musa::source;

/// A device-resident record chain: every step launches immediately on the
/// default stream with the synchronisation deferred, and [`session::apply`]
/// drains the chain with one synchronise plus the requested downloads.
pub(crate) struct session {
    cache: Arc<cache>,
    /// Every device buffer the session owns, in creation order: uploaded
    /// inputs and recorded outputs alike. `apply` frees them all through
    /// [`retiredevice`], whose queue drains inside the next handoff block.
    slots: Vec<(mudevptr, usize)>,
    /// Parallel to `slots`: whether a slot survives
    /// [`applyretained`](Session::applyretained). Persistent slots hold the
    /// device KV-cache buffers a persistent session reuses across steps.
    persistent: Vec<bool>,
}

impl Drop for session {
    fn drop(&mut self) {
        for (ptr, bytes) in self.slots.drain(..) {
            retiredevice(&self.cache, ptr, bytes);
        }
    }
}

impl session {
    /// Creates an empty record chain over `cache`.
    #[must_use]
    pub(crate) fn new(cache: Arc<cache>) -> Self {
        Self {
            cache,
            slots: Vec::new(),
            persistent: Vec::new(),
        }
    }

    /// Returns the device pointer of `slot`.
    pub(crate) fn slotptr(&self, slot: usize) -> mudevptr {
        self.slots[slot].0
    }

    /// Uploads a host slice once and returns its slot.
    pub(crate) fn upload(&mut self, data: &[f32]) -> Result<usize, gpuerror> {
        let mut result = drive(
            &self.cache,
            None,
            &[buf::hostdevice(data)],
            &[],
            (1, 1, 1),
            (1, 1, 1),
            0,
            false,
        )?;
        let (ptr, bytes) = result.device.remove(0);
        self.slots.push((ptr, bytes));
        self.persistent.push(false);
        Ok(self.slots.len() - 1)
    }

    /// Uploads a host slice into a slot that survives `applyretained`.
    pub(crate) fn allocpersistent(&mut self, data: &[f32]) -> Result<usize, gpuerror> {
        let slot = self.upload(data)?;
        self.persistent[slot] = true;
        Ok(slot)
    }

    /// Allocates a zeroed persistent device buffer of `count` floats.
    pub(crate) fn alloczeros(&mut self, count: usize) -> Result<usize, gpuerror> {
        let zeros = vec![0.0_f32; count];
        self.allocpersistent(&zeros)
    }

    /// Reports whether `slot` survives `applyretained`.
    pub(crate) fn ispersistent(&self, slot: usize) -> bool {
        self.persistent.get(slot).copied().unwrap_or(false)
    }

    /// Uploads a host slice into an allocation the caller owns and returns its
    /// pointer and byte size. Used for inputs pinned for a persistent session;
    /// they are not session slots, so `apply` never releases them.
    pub(crate) fn uploadpersistent(
        cache: &cache,
        data: &[f32],
    ) -> Result<(mudevptr, usize), gpuerror> {
        let mut result = drive(
            cache,
            None,
            &[buf::hostdevice(data)],
            &[],
            (1, 1, 1),
            (1, 1, 1),
            0,
            false,
        )?;
        Ok(result.device.remove(0))
    }

    /// Copies `bytes` from device `src` into a fresh allocation and returns its
    /// slot. Used when a kernel updates an operand in place and the original
    /// value must remain downloadable.
    pub(crate) fn copy(&mut self, src: mudevptr, bytes: usize) -> Result<usize, gpuerror> {
        let mut result = drive(
            &self.cache,
            None,
            &[buf::copy(src, bytes)],
            &[],
            (1, 1, 1),
            (1, 1, 1),
            0,
            false,
        )?;
        let (ptr, size) = result.device.remove(0);
        self.slots.push((ptr, size));
        self.persistent.push(false);
        Ok(self.slots.len() - 1)
    }

    /// Copies `bytes` from device `src` into the existing device buffer `dst`.
    pub(crate) fn copyinto(
        &mut self,
        dst: mudevptr,
        src: mudevptr,
        bytes: usize,
    ) -> Result<(), gpuerror> {
        drive(
            &self.cache,
            None,
            &[buf::copyinto(dst, src, bytes)],
            &[],
            (1, 1, 1),
            (1, 1, 1),
            0,
            false,
        )?;
        Ok(())
    }

    /// Uploads a host `u64` slice once and returns its slot.
    pub(crate) fn uploadu64(&mut self, data: &[u64]) -> Result<usize, gpuerror> {
        let mut result = drive(
            &self.cache,
            None,
            &[buf::hostu64device(data)],
            &[],
            (1, 1, 1),
            (1, 1, 1),
            0,
            false,
        )?;
        let (ptr, bytes) = result.device.remove(0);
        self.slots.push((ptr, bytes));
        self.persistent.push(false);
        Ok(self.slots.len() - 1)
    }

    /// Records one kernel step without synchronising and returns the output
    /// slot. `hostdevice` inputs in `buffers` are uploaded once and kept;
    /// `device` inputs bind existing slots.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record(
        &mut self,
        source_id: &'static str,
        entry: &'static str,
        buffers: &[buf<'_>],
        scalars: &[scalar],
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        sharedbytes: u32,
    ) -> Result<usize, gpuerror> {
        let result = drive(
            &self.cache,
            Some((source_id, entry)),
            buffers,
            scalars,
            grid,
            block,
            sharedbytes,
            false,
        )?;
        for (ptr, bytes) in result.device {
            self.slots.push((ptr, bytes));
            self.persistent.push(false);
        }
        Ok(self.slots.len() - 1)
    }

    /// Records one kernel step and returns **every** new device slot in buffer
    /// order.
    ///
    /// Used by operators with two device-resident outputs (`scan`'s output and
    /// final state); [`record`](Self::record) remains the single-output form.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn recordmulti(
        &mut self,
        source_id: &'static str,
        entry: &'static str,
        buffers: &[buf<'_>],
        scalars: &[scalar],
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        sharedbytes: u32,
    ) -> Result<Vec<usize>, gpuerror> {
        let result = drive(
            &self.cache,
            Some((source_id, entry)),
            buffers,
            scalars,
            grid,
            block,
            sharedbytes,
            false,
        )?;
        let mut slots = Vec::with_capacity(result.device.len());
        for (ptr, bytes) in result.device {
            self.slots.push((ptr, bytes));
            self.persistent.push(false);
            slots.push(self.slots.len() - 1);
        }
        Ok(slots)
    }

    /// Drains one slot to host data mid-chain, for a caller that needs a host
    /// copy of a recorded result. The slot is released immediately and marked
    /// empty.
    pub(crate) fn take(&mut self, slot: usize) -> Result<Vec<f32>, gpuerror> {
        let (ptr, bytes) = self.slots[slot];
        let mut out = flush(&self.cache, &[(ptr, bytes)])?;
        retiredevice(&self.cache, ptr, bytes);
        self.slots[slot] = (0, 0);
        self.persistent[slot] = false;
        Ok(out.remove(0))
    }

    /// Synchronises once, downloads the requested slots in order, and releases
    /// every owned buffer through [`retiredevice`].
    pub(crate) fn apply(&mut self, downloads: &[usize]) -> Result<Vec<Vec<f32>>, gpuerror> {
        let wanted: Vec<(mudevptr, usize)> =
            downloads.iter().map(|&index| self.slots[index]).collect();
        let out = flush(&self.cache, &wanted)?;
        for (ptr, bytes) in self.slots.drain(..) {
            retiredevice(&self.cache, ptr, bytes);
        }
        self.persistent.clear();
        Ok(out)
    }

    /// Downloads the requested slots and frees every **non-persistent** buffer,
    /// leaving persistent slots (and their indices) in place for the next step.
    pub(crate) fn applyretained(&mut self, downloads: &[usize]) -> Result<Vec<Vec<f32>>, gpuerror> {
        let wanted: Vec<(mudevptr, usize)> =
            downloads.iter().map(|&index| self.slots[index]).collect();
        let out = flush(&self.cache, &wanted)?;
        for index in 0..self.slots.len() {
            if self.persistent[index] {
                continue;
            }
            let (ptr, bytes) = self.slots[index];
            if ptr != 0 && bytes != 0 {
                retiredevice(&self.cache, ptr, bytes);
            }
            self.slots[index] = (0, 0);
        }
        Ok(out)
    }
}

/// Records one elementwise `add` into a device-resident chain.
pub(crate) fn add_record(
    session: &mut session,
    lhs: buf<'_>,
    rhs: buf<'_>,
    count: usize,
) -> Result<usize, gpuerror> {
    let grid = ((count as u32).div_ceil(source::FLATTILE), 1, 1);
    let block = (source::FLATTILE, 1, 1);
    session.record(
        "cuda",
        source::ADDENTRY,
        &[lhs, rhs, buf::outdevice(count)],
        &[scalar::u32(count as u32)],
        grid,
        block,
        0,
    )
}

/// Records one elementwise `mul` into a device-resident chain.
pub(crate) fn mul_record(
    session: &mut session,
    lhs: buf<'_>,
    rhs: buf<'_>,
    count: usize,
) -> Result<usize, gpuerror> {
    let grid = ((count as u32).div_ceil(source::FLATTILE), 1, 1);
    let block = (source::FLATTILE, 1, 1);
    session.record(
        "cuda",
        source::MULENTRY,
        &[lhs, rhs, buf::outdevice(count)],
        &[scalar::u32(count as u32)],
        grid,
        block,
        0,
    )
}

/// Records one `silu` into a device-resident chain.
pub(crate) fn silu_record(
    session: &mut session,
    input: buf<'_>,
    count: usize,
) -> Result<usize, gpuerror> {
    let grid = ((count as u32).div_ceil(source::FLATTILE), 1, 1);
    let block = (source::FLATTILE, 1, 1);
    session.record(
        "cuda",
        source::SILUENTRY,
        &[input, buf::outdevice(count)],
        &[scalar::u32(count as u32)],
        grid,
        block,
        0,
    )
}

/// Records one RMS normalization into a device-resident chain.
pub(crate) fn rmsnorm_record(
    session: &mut session,
    input: buf<'_>,
    weight: buf<'_>,
    rows: usize,
    width: usize,
    eps: f32,
) -> Result<usize, gpuerror> {
    let count = rows.checked_mul(width).ok_or(gpuerror::sizeoverflow)?;
    let grid = (rows as u32, 1, 1);
    let block = (source::RMSNORMTILE, 1, 1);
    session.record(
        "cuda",
        source::RMSNORMENTRY,
        &[input, weight, buf::outdevice(count)],
        &[scalar::u32(width as u32), scalar::f32(eps)],
        grid,
        block,
        source::RMSNORMTILE * 4,
    )
}

/// Records one softmax over the last axis into a device-resident chain.
pub(crate) fn softmax_record(
    session: &mut session,
    input: buf<'_>,
    rows: usize,
    width: usize,
) -> Result<usize, gpuerror> {
    let count = rows.checked_mul(width).ok_or(gpuerror::sizeoverflow)?;
    let grid = (rows as u32, 1, 1);
    let block = (source::FLATTILE, 1, 1);
    session.record(
        "cuda",
        source::SOFTMAXENTRY,
        &[input, buf::outdevice(count)],
        &[scalar::u32(width as u32)],
        grid,
        block,
        source::FLATTILE * 4,
    )
}

/// Records one embedding gather into a device-resident chain.
pub(crate) fn embedding_record(
    session: &mut session,
    table: buf<'_>,
    indices: buf<'_>,
    rows: usize,
    width: usize,
) -> Result<usize, gpuerror> {
    let count = rows.checked_mul(width).ok_or(gpuerror::sizeoverflow)?;
    session.record(
        "cuda",
        source::EMBEDDINGENTRY,
        &[table, indices, buf::outdevice(count)],
        &[scalar::u64(rows as u64), scalar::u64(width as u64)],
        (rows as u32, 1, 1),
        (source::FLATTILE, 1, 1),
        0,
    )
}

/// Records one rotary position embedding into a device-resident chain.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rope_record(
    session: &mut session,
    input: buf<'_>,
    count: usize,
    rows: usize,
    seqaxis: usize,
    heads: usize,
    headdim: usize,
    base: f32,
) -> Result<usize, gpuerror> {
    let grid = ((seqaxis as u32).div_ceil(source::FLATTILE), 1, 1);
    session.record(
        "cuda",
        source::ROPEENTRY,
        &[input, buf::outdevice(count)],
        &[
            scalar::u32(rows as u32),
            scalar::u32(seqaxis as u32),
            scalar::u32(heads as u32),
            scalar::u32(headdim as u32),
            scalar::f32(base),
        ],
        grid,
        (source::FLATTILE, 1, 1),
        0,
    )
}

/// Records one absolute-position rotary embedding into a device-resident
/// chain.
pub(crate) fn rope_at_record(
    session: &mut session,
    input: buf<'_>,
    count: usize,
    position: usize,
    heads: usize,
    headdim: usize,
    base: f32,
) -> Result<usize, gpuerror> {
    session.record(
        "cuda",
        source::ROPEATENTRY,
        &[input, buf::outdevice(count)],
        &[
            scalar::u32(heads as u32),
            scalar::u32(headdim as u32),
            scalar::u32(position as u32),
            scalar::f32(base),
        ],
        (1, 1, 1),
        (source::FLATTILE, 1, 1),
        0,
    )
}

/// Records one multi-head attention into a device-resident chain.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_record(
    session: &mut session,
    query: buf<'_>,
    key: buf<'_>,
    value: buf<'_>,
    count: usize,
    qseq: usize,
    kvseq: usize,
    heads: usize,
    kvheads: usize,
    headdim: usize,
    causal: bool,
) -> Result<usize, gpuerror> {
    let gridx = qseq.checked_mul(heads).ok_or(gpuerror::sizeoverflow)?;
    session.record(
        "cuda",
        source::ATTENTIONENTRY,
        &[query, key, value, buf::outdevice(count)],
        &[
            scalar::u32(qseq as u32),
            scalar::u32(kvseq as u32),
            scalar::u32(heads as u32),
            scalar::u32(kvheads as u32),
            scalar::u32(headdim as u32),
            scalar::u32(causal as u32),
        ],
        (gridx as u32, 1, 1),
        (source::ATTENTIONLANES, 1, 1),
        (2 * source::ATTENTIONLANES + headdim as u32) * 4,
    )
}

/// Records one incremental-decode attention into a device-resident chain.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_cached_record(
    session: &mut session,
    query: buf<'_>,
    keycached: buf<'_>,
    valuecached: buf<'_>,
    keynew: buf<'_>,
    valuenew: buf<'_>,
    count: usize,
    kvseq: usize,
    heads: usize,
    kvheads: usize,
    headdim: usize,
    causal: bool,
) -> Result<usize, gpuerror> {
    session.record(
        "cuda",
        source::ATTENTIONCACHEDENTRY,
        &[
            query,
            keycached,
            valuecached,
            keynew,
            valuenew,
            buf::outdevice(count),
        ],
        &[
            scalar::u32(kvseq as u32),
            scalar::u32(heads as u32),
            scalar::u32(kvheads as u32),
            scalar::u32(headdim as u32),
            scalar::u32(causal as u32),
        ],
        (heads as u32, 1, 1),
        (source::ATTENTIONLANES, 1, 1),
        (2 * source::ATTENTIONLANES + headdim as u32) * 4,
    )
}

/// Records the per-row reciprocal-RMS step into a device-resident chain.
pub(crate) fn rmsinv_record(
    session: &mut session,
    input: buf<'_>,
    rows: usize,
    width: usize,
    eps: f32,
) -> Result<usize, gpuerror> {
    session.record(
        source::MATMULWMMASOURCEID,
        source::RMSINVENTRY,
        &[input, buf::outdevice(rows)],
        &[
            scalar::u32(rows as u32),
            scalar::u32(width as u32),
            scalar::f32(eps),
        ],
        (rows as u32, 1, 1),
        (source::RMSNORMTILE, 1, 1),
        source::RMSNORMTILE * 4,
    )
}

/// Records the fused RMSNorm+matmul step into a device-resident chain.
#[allow(clippy::too_many_arguments)]
pub(crate) fn rmsnormmatmul_record(
    session: &mut session,
    input: buf<'_>,
    weight: buf<'_>,
    rhs: buf<'_>,
    invslot: usize,
    total: usize,
    m: usize,
    k: usize,
    n: usize,
    batch: usize,
) -> Result<usize, gpuerror> {
    let invptr = session.slotptr(invslot);
    // Full-precision fp32 fused kernel: 16x16 FMA tile, matching the
    // synchronous path's 1e-4 contract (not the TF32 2e-3 bound).
    let grid = (
        (n as u32).div_ceil(source::THREADEDGE),
        (m as u32).div_ceil(source::THREADEDGE),
        batch as u32,
    );
    session.record(
        source::MATMULWMMASOURCEID,
        source::MATMULRMSENTRY,
        &[
            input,
            weight,
            buf::device(invptr),
            rhs,
            buf::outdevice(total),
        ],
        &[
            scalar::u32(m as u32),
            scalar::u32(k as u32),
            scalar::u32(n as u32),
            scalar::u32(batch as u32),
        ],
        grid,
        (source::THREADEDGE, source::THREADEDGE, 1),
        0,
    )
}

/// Records one device-resident Tensor-Core matmul into a submission session.
///
/// The operands bind existing device slots (uploaded once per submission) and
/// the result stays in a device slot, so a chain of recorded matmuls performs
/// one upload, no per-step download, and one synchronisation at `apply`.
pub(crate) fn matmul_wmma_record(
    session: &mut session,
    lhs: buf<'_>,
    rhs: buf<'_>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<usize, gpuerror> {
    let total = m.checked_mul(n).ok_or(gpuerror::sizeoverflow)?;
    let grid = (
        (n as u32).div_ceil(source::MATMULWMMABN),
        (m as u32).div_ceil(source::MATMULWMMABM),
        1,
    );
    let block = (source::MATMULWMMATHREADS, 1, 1);
    session.record(
        source::MATMULWMMASOURCEID,
        source::MATMULWMMAENTRY,
        &[lhs, rhs, buf::outdevice(total)],
        &[
            scalar::u32(m as u32),
            scalar::u32(k as u32),
            scalar::u32(n as u32),
            scalar::u32(1),
        ],
        grid,
        block,
        0,
    )
}

/// Records one device-resident reference FMA matmul into a submission session.
pub(crate) fn matmul_fma_record(
    session: &mut session,
    lhs: buf<'_>,
    rhs: buf<'_>,
    m: usize,
    k: usize,
    n: usize,
) -> Result<usize, gpuerror> {
    let total = m.checked_mul(n).ok_or(gpuerror::sizeoverflow)?;
    let grid = (
        (n as u32).div_ceil(source::THREADEDGE),
        (m as u32).div_ceil(source::THREADEDGE),
        1,
    );
    let block = (source::THREADEDGE, source::THREADEDGE, 1);
    session.record(
        "cuda",
        source::MATMULENTRY,
        &[lhs, rhs, buf::outdevice(total)],
        &[
            scalar::u32(m as u32),
            scalar::u32(k as u32),
            scalar::u32(n as u32),
            scalar::u32(1),
        ],
        grid,
        block,
        0,
    )
}
