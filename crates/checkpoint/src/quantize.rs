//! Symmetric int8 quantization of `f32` weights.
//!
//! Quantization converts a `f32` slice into `i8` values plus one scale per
//! group, matching the `quantdesc` contract of the `.u` format (scale is a
//! first-class tensor). Dequantization is the approximate inverse; the
//! reconstruction error is bounded by half a scale step.

use crate::error::checkpointerror;

/// Returns the number of groups a quantization of `len` elements produces.
fn groupcount(len: usize, groupsize: usize) -> usize {
    if groupsize == 0 {
        1
    } else {
        len.div_ceil(groupsize)
    }
}

/// Symmetrically quantizes `data` to `i8`, returning the quantized values and
/// one scale per group.
///
/// `groupsize = 0` quantizes the whole slice with a single scale (per-tensor);
/// otherwise every `groupsize` elements share one scale (per-group/per-channel).
/// The scale of a group is `max(|x|) / 127`, so a full-scale value maps to
/// `127`.
///
/// # Errors
///
/// Returns [`checkpointerror::invalidmanifest`] for an empty input or a
/// `groupsize` larger than the input length.
pub fn quantize(data: &[f32], groupsize: usize) -> Result<(Vec<i8>, Vec<f32>), checkpointerror> {
    if data.is_empty() {
        return Err(checkpointerror::invalidmanifest {
            reason: "cannot quantize an empty slice".to_string(),
        });
    }
    if groupsize != 0 && groupsize > data.len() {
        return Err(checkpointerror::invalidmanifest {
            reason: "groupsize exceeds the input length".to_string(),
        });
    }
    let size = if groupsize == 0 {
        data.len()
    } else {
        groupsize
    };
    let ngroup = data.len().div_ceil(size);
    let mut scale = vec![0.0f32; ngroup];
    for (gi, chunk) in data.chunks(size).enumerate() {
        let mut maxabs = 0.0f32;
        for &x in chunk {
            maxabs = maxabs.max(x.abs());
        }
        scale[gi] = maxabs / 127.0;
    }
    let mut q = vec![0i8; data.len()];
    for (gi, chunk) in data.chunks(size).enumerate() {
        let s = scale[gi];
        for (j, &x) in chunk.iter().enumerate() {
            let scaled = if s == 0.0 { 0.0 } else { x / s };
            q[gi * size + j] = scaled.round().clamp(-127.0, 127.0) as i8;
        }
    }
    Ok((q, scale))
}

/// Dequantizes `i8` values with one scale per group back to `f32`.
///
/// # Errors
///
/// Returns [`checkpointerror::invalidmanifest`] for an empty input, a
/// `groupsize` larger than the input length, or a scale count that does not
/// match the group count.
pub fn dequantize(q: &[i8], scale: &[f32], groupsize: usize) -> Result<Vec<f32>, checkpointerror> {
    if q.is_empty() {
        return Err(checkpointerror::invalidmanifest {
            reason: "cannot dequantize an empty slice".to_string(),
        });
    }
    if groupsize != 0 && groupsize > q.len() {
        return Err(checkpointerror::invalidmanifest {
            reason: "groupsize exceeds the input length".to_string(),
        });
    }
    let ngroup = groupcount(q.len(), groupsize);
    if scale.len() != ngroup {
        return Err(checkpointerror::invalidmanifest {
            reason: "scale count does not match the group count".to_string(),
        });
    }
    let size = if groupsize == 0 { q.len() } else { groupsize };
    let mut data = vec![0.0f32; q.len()];
    for (gi, chunk) in q.chunks(size).enumerate() {
        let s = scale[gi];
        for (j, &qi) in chunk.iter().enumerate() {
            data[gi * size + j] = f32::from(qi) * s;
        }
    }
    Ok(data)
}
