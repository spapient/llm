//! The validated geometry of one operator call.
//!
//! Each plan type turns an already budget-checked operand shape (or shape
//! triple) into the plain numbers a kernel needs, and rejects exactly the
//! inputs the scalar reference rejects — the same error variant, for the same
//! condition, in the same order. Keeping the shape reasoning here rather than
//! in the executor keeps both reviewable: this module owns *what is a legal
//! call*, the executor owns *how a legal call is dispatched*.
//!
//! No plan allocates or touches the device, so every one of them is exercised
//! unconditionally by the contract suite, on a machine with or without a GPU.
//!
//! `matmul` carries a plan here too, mirroring the CUDA backend so that a
//! boundary fix cannot cover one backend and miss the other.

use api::backenderror;
use tensor::shape;

use crate::backend::METALMAXHEADDIM;

/// The stable name of the matrix-multiply operator.
pub(crate) const MATMULOP: &str = "matmul";

/// The stable name of the elementwise addition operator.
pub(crate) const ADDOP: &str = "add";

/// The stable name of the elementwise multiplication operator.
pub(crate) const MULOP: &str = "mul";

/// The stable name of the RMS-normalization operator.
pub(crate) const RMSNORMOP: &str = "rmsnorm";

/// The stable name of the numerically stable softmax operator.
pub(crate) const SOFTMAXOP: &str = "softmax";

/// The stable name of the rotary-position-embedding operator.
pub(crate) const ROPEOP: &str = "rope";

/// The stable name of the grouped-query attention operator.
pub(crate) const ATTENTIONOP: &str = "attention";

/// Returns the number of rows an operand flattened to `[rows, width]` has.
fn rowcount(input: &shape, width: usize) -> Result<usize, backenderror> {
    let elements = input
        .getelementcount()
        .map_err(|_| backenderror::sizeoverflow)?;
    Ok(elements / width)
}

/// Returns the shared shape of two equal-shaped operands.
///
/// No broadcasting is performed: the dimensions must match exactly, matching
/// the scalar reference. The result is the output shape, which for these
/// operators is the operand shape itself.
pub(crate) fn binaryshape(
    lhs: &shape,
    rhs: &shape,
    op: &'static str,
) -> Result<shape, backenderror> {
    if lhs.getdims() != rhs.getdims() {
        return Err(backenderror::shapemismatch { op });
    }
    Ok(lhs.clone())
}

/// The validated geometry of a batched `matmul`.
pub(crate) struct matmulshape {
    /// The leading dimensions of every operand, shared by both sides.
    pub(crate) batchdims: Vec<usize>,
    /// The product of `batchdims`, or 1 for unbatched operands.
    pub(crate) batch: usize,
    /// The left operand's row count.
    pub(crate) m: usize,
    /// The shared contraction extent.
    pub(crate) k: usize,
    /// The right operand's column count.
    pub(crate) n: usize,
}

impl matmulshape {
    /// Validates the `matmul` contract's shape preconditions.
    pub(crate) fn new(lhs: &shape, rhs: &shape) -> Result<Self, backenderror> {
        let ld = lhs.getdims();
        let rd = rhs.getdims();
        if ld.len() < 2 || rd.len() < 2 {
            return Err(backenderror::shapemismatch { op: MATMULOP });
        }
        let batchdims = &ld[..ld.len() - 2];
        if batchdims != &rd[..rd.len() - 2] {
            return Err(backenderror::shapemismatch { op: MATMULOP });
        }
        let m = ld[ld.len() - 2];
        let k = ld[ld.len() - 1];
        if k != rd[rd.len() - 2] {
            return Err(backenderror::shapemismatch { op: MATMULOP });
        }
        let n = rd[rd.len() - 1];
        let batch = batchdims
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or(backenderror::sizeoverflow)?;
        Ok(Self {
            batchdims: batchdims.to_vec(),
            batch,
            m,
            k,
            n,
        })
    }

    /// Returns the output shape `[batch..., m, n]`.
    ///
    /// The plan carries the batch dimensions so the output shape is rebuilt
    /// here rather than re-derived by the executor; the constructor of a `shape`
    /// fails only for an empty dimension list, which a plan cannot hold.
    pub(crate) fn outshape(&self) -> Result<shape, backenderror> {
        let mut dims = self.batchdims.clone();
        dims.push(self.m);
        dims.push(self.n);
        shape::new(dims).map_err(|_| backenderror::sizeoverflow)
    }
}

/// The validated geometry of a last-axis normalization.
pub(crate) struct axisshape {
    /// How many contiguous last-axis slices the operand holds.
    pub(crate) rows: usize,
    /// The length of one slice.
    pub(crate) width: usize,
}

impl axisshape {
    /// Validates that `input` has a non-empty last axis, naming `op` in the
    /// rejection the way the scalar reference does.
    fn new(input: &shape, op: &'static str) -> Result<Self, backenderror> {
        match input.getdims().last() {
            Some(&width) if width > 0 => Ok(Self {
                rows: rowcount(input, width)?,
                width,
            }),
            _ => Err(backenderror::unsupportedoperation { op }),
        }
    }

    /// Validates a softmax operand: a non-empty last axis, nothing else.
    pub(crate) fn softmax(input: &shape) -> Result<Self, backenderror> {
        Self::new(input, SOFTMAXOP)
    }

    /// Validates an RMS-normalization operand and its weight.
    ///
    /// The weight must hold exactly one row; the scalar reference compares its
    /// element count rather than its shape, so a `[width]` and a `[1, width]`
    /// weight are both accepted.
    pub(crate) fn rmsnorm(input: &shape, weight: &shape) -> Result<Self, backenderror> {
        let geometry = Self::new(input, RMSNORMOP)?;
        let weights = weight
            .getelementcount()
            .map_err(|_| backenderror::sizeoverflow)?;
        if weights != geometry.width {
            return Err(backenderror::shapemismatch { op: RMSNORMOP });
        }
        Ok(geometry)
    }
}

/// The validated geometry of a rotary-position-embedding call.
///
/// The plan carries only what the kernel call needs; the frequency base stays a
/// call parameter because it is a configuration value the plan does not
/// reinterpret.
pub(crate) struct ropeshape {
    /// How many `heads * headdim` rows the operand holds.
    pub(crate) rows: usize,
    /// The position-axis extent: the position of a row is its index modulo it.
    pub(crate) seqaxis: usize,
    /// The number of attention heads packed into the last axis.
    pub(crate) heads: usize,
    /// The per-head slice length, always even.
    pub(crate) headdim: usize,
}

impl ropeshape {
    /// Validates the `rope` contract's shape preconditions.
    pub(crate) fn new(input: &shape, heads: usize) -> Result<Self, backenderror> {
        let dims = input.getdims();
        let rank = dims.len();
        if rank < 2 || heads == 0 {
            return Err(backenderror::shapemismatch { op: ROPEOP });
        }
        let last = dims[rank - 1];
        if !last.is_multiple_of(heads) {
            return Err(backenderror::shapemismatch { op: ROPEOP });
        }
        let headdim = last / heads;
        if headdim == 0 || !headdim.is_multiple_of(2) {
            return Err(backenderror::shapemismatch { op: ROPEOP });
        }
        Ok(Self {
            rows: rowcount(input, last)?,
            seqaxis: dims[rank - 2],
            heads,
            headdim,
        })
    }
}

/// The validated geometry of an embedding gather.
pub(crate) struct embeddingshape {
    /// How many rows the gathered output holds.
    pub(crate) rows: usize,
    /// The element count of one gathered row.
    pub(crate) width: usize,
}

impl embeddingshape {
    /// Validates the table shape and every index against it.
    ///
    /// The table always has rank at least one, so its first axis is its row
    /// space; a rank-one table therefore has a row width of one, matching the
    /// scalar reference.
    pub(crate) fn new(table: &shape, indices: &[usize]) -> Result<Self, backenderror> {
        let dims = table.getdims();
        let vocab = dims[0];
        let width = dims[1..]
            .iter()
            .try_fold(1usize, |acc, &d| acc.checked_mul(d))
            .ok_or(backenderror::sizeoverflow)?;
        for &index in indices {
            if index >= vocab {
                return Err(backenderror::indexoutofrange { index, len: vocab });
            }
        }
        Ok(Self {
            rows: indices.len(),
            width,
        })
    }
}

/// The validated geometry of one attention call.
///
/// The plan carries every extent the launch needs, so the launch neither
/// re-derives nor re-checks a number the plan has already established. The
/// dot-product scale is the one quantity deliberately absent: the kernel
/// computes `1 / sqrt(headdim)` itself from the head dimension it is given.
pub(crate) struct attentionshape {
    /// The shared sequence length of query, key, and value.
    pub(crate) seq: usize,
    /// How many query heads the last axis is split into.
    pub(crate) heads: usize,
    /// How many key and value heads the last axis is split into.
    pub(crate) kvheads: usize,
    /// The per-head slice length, `qstride / heads`.
    pub(crate) headdim: usize,
}

impl attentionshape {
    /// Validates the three operand shapes and the head split.
    ///
    /// The last check is a device limit, not a contract limit: the kernel keeps
    /// one accumulator slot per head dimension in threadgroup memory, so a head
    /// dimension beyond [`METALMAXHEADDIM`] cannot be launched. It is rejected
    /// as resource exhaustion rather than shape mismatch because the call is
    /// well formed and only the device is too small for it.
    pub(crate) fn new(
        query: &shape,
        key: &shape,
        value: &shape,
        heads: usize,
        kvheads: usize,
    ) -> Result<Self, backenderror> {
        let qd = query.getdims();
        let kd = key.getdims();
        let vd = value.getdims();
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
        let kwidth = kvheads
            .checked_mul(headdim)
            .ok_or(backenderror::sizeoverflow)?;
        if headdim == 0 || kd[1] != kwidth {
            return Err(backenderror::shapemismatch { op: ATTENTIONOP });
        }
        if headdim > METALMAXHEADDIM {
            return Err(backenderror::resourceexhausted);
        }
        Ok(Self {
            seq: qd[0],
            heads,
            kvheads,
            headdim,
        })
    }
}
