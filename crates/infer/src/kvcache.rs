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
    /// together. All validation and tensor construction completes before the
    /// cache is mutated, so reported errors cannot leave a partial append.
    ///
    /// # Errors
    ///
    /// Returns a [`models::modelerror`] when `key` or `value` is not a
    /// `[1, width]` tensor, layer lengths have diverged, or the cache is full.
    pub fn appendall(&mut self, key: &tensor<f32>, value: &tensor<f32>) -> Result<(), modelerror> {
        checkrow(key, self.width, "kv key")?;
        checkrow(value, self.width, "kv value")?;

        let current = self.lens.first().copied().unwrap_or(0);
        if self.lens.iter().any(|&len| len != current) {
            return Err(modelerror::config(
                "kv cache",
                "layer lengths diverged; clear the cache before reuse",
            ));
        }
        if current >= self.maxseq {
            return Err(modelerror::config("kv cache", "capacity exceeded"));
        }

        let newlen = current + 1;
        let mut replacements = Vec::with_capacity(self.layers);
        for layer in 0..self.layers {
            let newkey = concatrow(
                self.keys[layer].as_ref(),
                key.getdata(),
                newlen,
                self.width,
            )?;
            let newvalue = concatrow(
                self.values[layer].as_ref(),
                value.getdata(),
                newlen,
                self.width,
            )?;
            replacements.push((newkey, newvalue));
        }

        for (layer, (newkey, newvalue)) in replacements.into_iter().enumerate() {
            self.keys[layer] = Some(newkey);
            self.values[layer] = Some(newvalue);
            self.lens[layer] = newlen;
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
        // Build both tensors before committing either side.
        let newkey = concatrow(
            self.keys[layer].as_ref(),
            key.getdata(),
            newlen,
            self.width,
        )?;
        let newvalue = concatrow(
            self.values[layer].as_ref(),
            value.getdata(),
            newlen,
            self.width,
        )?;
        self.keys[layer] = Some(newkey);
        self.values[layer] = Some(newvalue);
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


#[cfg(test)]
mod tests {
    use super::{kvstate, shape, tensor};
    use models::kvcache;

    use std::error::Error;

    fn row(values: &[f32]) -> Result<tensor<f32>, Box<dyn Error>> {
        Ok(tensor::new(
            shape::new(vec![1, values.len()])?,
            values.to_vec(),
        )?)
    }

    #[test]
    fn appendall_commits_every_layer_together() -> Result<(), Box<dyn Error>> {
        let mut cache = kvstate::new(2, 2, 3)?;
        let key = row(&[1.0, 2.0])?;
        let value = row(&[3.0, 4.0])?;

        cache.appendall(&key, &value)?;

        assert_eq!(cache.getlen(), 1);
        for layer in 0..cache.getlayers() {
            assert_eq!(cache.keys(layer).map(|t| t.getshape().getdims()), Some(&[1, 2][..]));
            assert_eq!(cache.values(layer).map(|t| t.getshape().getdims()), Some(&[1, 2][..]));
        }
        Ok(())
    }

    #[test]
    fn appendall_rejects_diverged_layers_without_partial_mutation(
    ) -> Result<(), Box<dyn Error>> {
        let mut cache = kvstate::new(2, 2, 3)?;
        let key = row(&[1.0, 2.0])?;
        let value = row(&[3.0, 4.0])?;
        cache.append(0, &key, &value)?;
        cache.append(0, &key, &value)?;

        assert!(cache.appendall(&key, &value).is_err());
        assert_eq!(cache.getlen(), 2);
        assert_eq!(cache.keys(1), None);
        assert_eq!(cache.values(1), None);
        Ok(())
    }

    #[test]
    fn appendall_rejects_capacity_before_mutating_any_layer(
    ) -> Result<(), Box<dyn Error>> {
        let mut cache = kvstate::new(2, 2, 1)?;
        let key = row(&[1.0, 2.0])?;
        let value = row(&[3.0, 4.0])?;

        // Fill every layer uniformly so the next append fails specifically
        // because the cache is at capacity, not because layer lengths differ.
        cache.appendall(&key, &value)?;
        let keys_before: Vec<Option<Vec<f32>>> = (0..cache.getlayers())
            .map(|layer| cache.keys(layer).map(|tensor| tensor.getdata().to_vec()))
            .collect();
        let values_before: Vec<Option<Vec<f32>>> = (0..cache.getlayers())
            .map(|layer| cache.values(layer).map(|tensor| tensor.getdata().to_vec()))
            .collect();

        assert!(cache.appendall(&key, &value).is_err());
        assert_eq!(cache.getlen(), 1);
        for layer in 0..cache.getlayers() {
            assert_eq!(
                cache.keys(layer).map(|tensor| tensor.getdata().to_vec()),
                keys_before[layer]
            );
            assert_eq!(
                cache.values(layer).map(|tensor| tensor.getdata().to_vec()),
                values_before[layer]
            );
        }
        Ok(())
    }
}
