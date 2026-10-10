use ::tensor::{shape, tensor};
use models::kvcache;
use models::modelerror;

use crate::error::infererror;

/// A request-owned key/value cache.
///
/// `kvstate` implements the [`models::kvcache`] contract owned by `models`. It
/// holds one key and one value tensor per layer, each of shape
/// `[cached_len, width]`, and advances every layer in lockstep. The cache is the
/// only mutable attention state a request carries; it is created by, owned by,
/// and reset by a [`generationrequest`](crate::generationrequest) and is never
/// stored in shared model weights (OW2).
///
/// The correctness path recomputes the full context and does not read or write
/// this cache yet; it exists so a cache-aware decode can be added against the
/// current full-recompute baseline without changing the ownership model.
#[derive(Clone, Debug)]
pub struct kvstate {
    layers: usize,
    width: usize,
    maxseq: usize,
    lens: Vec<usize>,
    keys: Vec<Option<tensor<f32>>>,
    values: Vec<Option<tensor<f32>>>,
}

impl kvstate {
    /// Builds an empty cache for `layers` layers of width `width`, holding at
    /// most `maxseq` positions per layer.
    ///
    /// # Errors
    ///
    /// Returns [`infererror::invalidrequest`] when any dimension is zero.
    pub fn new(layers: usize, width: usize, maxseq: usize) -> Result<Self, infererror> {
        if layers == 0 {
            return Err(infererror::invalidrequest {
                field: "kvlayers",
                detail: "must be non-zero".to_string(),
            });
        }
        if width == 0 {
            return Err(infererror::invalidrequest {
                field: "kvwidth",
                detail: "must be non-zero".to_string(),
            });
        }
        if maxseq == 0 {
            return Err(infererror::invalidrequest {
                field: "maxseq",
                detail: "must be non-zero".to_string(),
            });
        }
        Ok(Self {
            layers,
            width,
            maxseq,
            lens: vec![0; layers],
            keys: vec![None; layers],
            values: vec![None; layers],
        })
    }

    /// Returns the per-layer key/value width.
    #[must_use]
    pub const fn getwidth(&self) -> usize {
        self.width
    }

    /// Clears every layer and resets the cached length to zero.
    pub fn clear(&mut self) {
        self.keys.fill(None);
        self.values.fill(None);
        self.lens.clear();
        self.lens.resize(self.layers, 0);
    }

    /// Appends one position's key and value to every layer, advancing them
    /// together.
    ///
    /// # Errors
    ///
    /// Returns a [`models::modelerror`] when a layer is out of range, `key` or
    /// `value` is not a `[1, width]` tensor, or the cache is full.
    pub fn appendall(&mut self, key: &tensor<f32>, value: &tensor<f32>) -> Result<(), modelerror> {
        for layer in 0..self.layers {
            self.append(layer, key, value)?;
        }
        Ok(())
    }
}

impl models::kvcache for kvstate {
    fn getlayers(&self) -> usize {
        self.layers
    }

    fn getlen(&self) -> usize {
        self.lens.iter().copied().max().unwrap_or(0)
    }

    fn getmaxseq(&self) -> usize {
        self.maxseq
    }

    fn append(
        &mut self,
        layer: usize,
        key: &tensor<f32>,
        value: &tensor<f32>,
    ) -> Result<(), modelerror> {
        let current = self
            .lens
            .get(layer)
            .copied()
            .ok_or(modelerror::config("kv layer", "out of range"))?;
        checkrow(key, self.width, "kv key")?;
        checkrow(value, self.width, "kv value")?;
        if current >= self.maxseq {
            return Err(modelerror::config("kv cache", "capacity exceeded"));
        }
        let newlen = current + 1;
        self.keys[layer] = Some(concatrow(
            self.keys[layer].as_ref(),
            key.getdata(),
            newlen,
            self.width,
        )?);
        self.values[layer] = Some(concatrow(
            self.values[layer].as_ref(),
            value.getdata(),
            newlen,
            self.width,
        )?);
        self.lens[layer] = newlen;
        Ok(())
    }

    fn keys(&self, layer: usize) -> Option<&tensor<f32>> {
        self.keys.get(layer).and_then(Option::as_ref)
    }

    fn values(&self, layer: usize) -> Option<&tensor<f32>> {
        self.values.get(layer).and_then(Option::as_ref)
    }
}

/// Validates that `t` is a `[1, width]` row.
fn checkrow(t: &tensor<f32>, width: usize, context: &'static str) -> Result<(), modelerror> {
    let dims = t.getshape().getdims();
    if dims.len() != 2 || dims[0] != 1 || dims[1] != width {
        return Err(modelerror::shapemismatch {
            context,
            expected: vec![1, width],
            actual: dims.to_vec(),
        });
    }
    Ok(())
}

/// Appends `row` to `existing` and rebuilds a `[newlen, width]` tensor.
fn concatrow(
    existing: Option<&tensor<f32>>,
    row: &[f32],
    newlen: usize,
    width: usize,
) -> Result<tensor<f32>, modelerror> {
    let mut data = match existing {
        Some(t) => t.getdata().to_vec(),
        None => Vec::new(),
    };
    data.extend_from_slice(row);
    let s = shape::new(vec![newlen, width])?;
    Ok(tensor::new(s, data)?)
}
