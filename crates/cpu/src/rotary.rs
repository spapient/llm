use ::tensor::{tensor, tensorview};
use api::{backenderror, kernel};

use crate::support::finish;

/// The stable name reported by [`cpurope`].
const ROPEOP: &str = "rope";

/// Rotary position embedding (RoPE) over the last axis.
///
/// `input` has shape `[..., seq, heads * headdim]`. The second-to-last axis is
/// the position axis: the position of a row is its index along that axis. Each
/// contiguous `headdim` slice of the last axis is rotated by the tangent of the
/// position-scaled angle `theta_i = base^(-2i / headdim)`, applied to the pair
/// `(2i, 2i + 1)`:
///
/// ```text
/// out[2i]     = x[2i] * cos(p * theta_i) - x[2i+1] * sin(p * theta_i)
/// out[2i+1]   = x[2i] * sin(p * theta_i) + x[2i+1] * cos(p * theta_i)
/// ```
///
/// The device is f32-only. `base` MUST be finite and positive; a non-positive
/// base produces a non-finite frequency and is a caller precondition. The
/// inverse-frequency table is recomputed per call; this is a documented
/// correctness-first allocation, not a measured hot path.
#[derive(Clone, Copy, Debug)]
pub struct cpurope {
    heads: usize,
    base: f32,
}

impl cpurope {
    /// Creates a RoPE kernel for `heads` attention heads with `base` frequency.
    #[must_use]
    pub const fn new(heads: usize, base: f32) -> Self {
        Self { heads, base }
    }

    /// Returns the configured head count.
    #[must_use]
    pub const fn getheads(self) -> usize {
        self.heads
    }

    /// Returns the configured rotary base.
    #[must_use]
    pub const fn getbase(self) -> f32 {
        self.base
    }
}

impl kernel<f32> for cpurope {
    fn getname(&self) -> &'static str {
        ROPEOP
    }

    fn apply(&self, input: &tensorview<'_, f32>) -> Result<tensor<f32>, backenderror> {
        let dims = input.getshape().getdims();
        let rank = dims.len();
        if rank < 2 || self.heads == 0 {
            return Err(backenderror::shapemismatch { op: ROPEOP });
        }
        let last = dims[rank - 1];
        if !last.is_multiple_of(self.heads) {
            return Err(backenderror::shapemismatch { op: ROPEOP });
        }
        let headdim = last / self.heads;
        if headdim == 0 || !headdim.is_multiple_of(2) {
            return Err(backenderror::shapemismatch { op: ROPEOP });
        }
        let seqaxis = dims[rank - 2];
        let half = headdim / 2;
        let mut invfreq = Vec::with_capacity(half);
        for i in 0..half {
            invfreq.push(self.base.powf(-2.0 * i as f32 / headdim as f32));
        }
        let src = input.getdata();
        let mut data = vec![0.0_f32; src.len()];
        for (row, out) in data.chunks_mut(last).enumerate() {
            let position = if seqaxis == 0 { 0 } else { row % seqaxis };
            let inrow = &src[row * last..(row + 1) * last];
            // Each (position, i) angle is computed once and reused by every
            // head; the per-element arithmetic is unchanged, so the result is
            // bit-identical to the head-outer evaluation order.
            for (i, &freq) in invfreq.iter().enumerate() {
                let angle = position as f32 * freq;
                let (sin, cos) = angle.sin_cos();
                for head in 0..self.heads {
                    let offset = head * headdim + 2 * i;
                    let even = inrow[offset];
                    let odd = inrow[offset + 1];
                    out[offset] = even * cos - odd * sin;
                    out[offset + 1] = even * sin + odd * cos;
                }
            }
        }
        finish(input.getshape(), data)
    }
}
