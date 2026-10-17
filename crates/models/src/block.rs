use ::tensor::{shape, tensor};
use api::{compute, operand, operandpair, recorder, stepref};
use nn::{attention, rotary};

use crate::causal::{devicekvcache, kvcache};
use crate::config::llamaconfig;
use crate::error::modelerror;
use crate::weights::llamalayerweights;

/// Pins every layer weight for a recorder.
///
/// A single-use submission ignores this; a persistent session uploads each
/// weight once and reuses the device allocation across decode steps.
fn pinweights<R: recorder>(sub: &mut R, weights: &llamalayerweights) -> Result<(), modelerror> {
    sub.pin(weights.getattnnorm().getweight().getvalue().getview())?;
    sub.pin(weights.getwq().getweight().getvalue().getview())?;
    sub.pin(weights.getwk().getweight().getvalue().getview())?;
    sub.pin(weights.getwv().getweight().getvalue().getview())?;
    sub.pin(weights.getwo().getweight().getvalue().getview())?;
    sub.pin(weights.getffnnorm().getweight().getvalue().getview())?;
    sub.pin(weights.getmlp().getgate().getweight().getvalue().getview())?;
    sub.pin(weights.getmlp().getup().getweight().getvalue().getview())?;
    sub.pin(weights.getmlp().getdown().getweight().getvalue().getview())?;
    Ok(())
}

/// The configuration-derived operators of one Llama transformer block.
///
/// A block's attention configuration and rotary embeddings depend only on the
/// architecture, not on the weights, so they are built once and shared by every
/// layer. The per-layer weights are passed to [`Self::forward`], keeping the
/// immutable weights separate from the operator configuration (OW4).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct llamablock {
    attention: attention,
    rotaryq: rotary,
    rotaryk: rotary,
}

impl llamablock {
    /// Builds the block operators from `config`.
    ///
    /// # Errors
    ///
    /// Returns a [`modelerror`] when the head layout or rotary base is invalid
    /// for the configuration.
    pub fn new(config: &llamaconfig) -> Result<Self, modelerror> {
        Ok(Self {
            attention: attention::new(config.getnheads(), config.getnkvheads(), true)?,
            rotaryq: rotary::new(config.getnheads(), config.getropebase())?,
            rotaryk: rotary::new(config.getnkvheads(), config.getropebase())?,
        })
    }

    /// Runs one pre-norm transformer block over `x` of shape `[seq, dim]`.
    ///
    /// # Errors
    ///
    /// Returns a [`modelerror`] when `x` is not rank two with the model
    /// dimension, or when a projection, normalization, rotary, attention, or
    /// residual operation fails.
    pub fn forward<C: compute>(
        &self,
        engine: &C,
        weights: &llamalayerweights,
        x: &tensor<f32>,
    ) -> Result<tensor<f32>, modelerror> {
        let dims = x.getshape().getdims();
        if dims.len() != 2 {
            return Err(modelerror::config(
                "block",
                "input must be a rank-two [seq, dim] tensor",
            ));
        }
        let normalized = weights.getattnnorm().forward(engine, x)?;
        let q = weights.getwq().forward(engine, &normalized)?;
        let k = weights.getwk().forward(engine, &normalized)?;
        let v = weights.getwv().forward(engine, &normalized)?;
        let q = self.rotaryq.forward(engine, &q)?;
        let k = self.rotaryk.forward(engine, &k)?;
        let attended = self.attention.forward(engine, &q, &k, &v)?;
        let projected = weights.getwo().forward(engine, &attended)?;
        let residual = engine.add(&x.getview(), &projected.getview())?;
        let normalized = weights.getffnnorm().forward(engine, &residual)?;
        let feedforward = weights.getmlp().forward(engine, &normalized)?;
        Ok(engine.add(&residual.getview(), &feedforward.getview())?)
    }

    /// Records one pre-norm transformer block into a submission, mirroring
    /// [`forward`](Self::forward) operator for operator, and returns the residual
    /// output's handle.
    ///
    /// The caller submits the whole chain once and resolves the returned
    /// [`stepref`]; intermediate results stay on the backend's device.
    pub fn forwardsubmission<R: recorder>(
        &self,
        sub: &mut R,
        weights: &llamalayerweights,
        x: stepref,
    ) -> Result<stepref, modelerror> {
        pinweights(sub, weights)?;
        let normalized = sub.recordrmsnorm(
            operand::result(x),
            operand::input(weights.getattnnorm().getweight().getvalue().getview()),
            weights.getattnnorm().geteps(),
        )?;
        let q = sub.recordmatmul(
            operand::result(normalized),
            operand::input(weights.getwq().getweight().getvalue().getview()),
        )?;
        let k = sub.recordmatmul(
            operand::result(normalized),
            operand::input(weights.getwk().getweight().getvalue().getview()),
        )?;
        let v = sub.recordmatmul(
            operand::result(normalized),
            operand::input(weights.getwv().getweight().getvalue().getview()),
        )?;
        let q = sub.recordrope(
            operand::result(q),
            self.rotaryq.getheads(),
            self.rotaryq.getbase(),
        )?;
        let k = sub.recordrope(
            operand::result(k),
            self.rotaryk.getheads(),
            self.rotaryk.getbase(),
        )?;
        let attended = sub.recordattention(
            operand::result(q),
            operand::result(k),
            operand::result(v),
            self.attention.getheads(),
            self.attention.getkvheads(),
            self.attention.getcausal(),
        )?;
        let projected = sub.recordmatmul(
            operand::result(attended),
            operand::input(weights.getwo().getweight().getvalue().getview()),
        )?;
        let residual = sub.recordadd(operand::result(x), operand::result(projected))?;
        let normalized = sub.recordrmsnorm(
            operand::result(residual),
            operand::input(weights.getffnnorm().getweight().getvalue().getview()),
            weights.getffnnorm().geteps(),
        )?;
        let gate = sub.recordmatmul(
            operand::result(normalized),
            operand::input(weights.getmlp().getgate().getweight().getvalue().getview()),
        )?;
        let up = sub.recordmatmul(
            operand::result(normalized),
            operand::input(weights.getmlp().getup().getweight().getvalue().getview()),
        )?;
        let silu = sub.recordsilu(operand::result(gate))?;
        let gated = sub.recordmul(operand::result(silu), operand::result(up))?;
        let feedforward = sub.recordmatmul(
            operand::result(gated),
            operand::input(weights.getmlp().getdown().getweight().getvalue().getview()),
        )?;
        Ok(sub.recordadd(operand::result(residual), operand::result(feedforward))?)
    }

    /// Records one pre-norm transformer block for a single new token into a
    /// submission, reading the cached keys and values and returning the new
    /// key/value handles for the caller to append after `apply`.
    pub fn forwardsubmission_cached<R: recorder, K: kvcache>(
        &self,
        sub: &mut R,
        cache: &K,
        weights: &llamalayerweights,
        x: stepref,
        layer: usize,
        position: usize,
    ) -> Result<(stepref, stepref, stepref), modelerror> {
        pinweights(sub, weights)?;
        let normalized = sub.recordrmsnorm(
            operand::result(x),
            operand::input(weights.getattnnorm().getweight().getvalue().getview()),
            weights.getattnnorm().geteps(),
        )?;
        let q = sub.recordmatmul(
            operand::result(normalized),
            operand::input(weights.getwq().getweight().getvalue().getview()),
        )?;
        let k = sub.recordmatmul(
            operand::result(normalized),
            operand::input(weights.getwk().getweight().getvalue().getview()),
        )?;
        let v = sub.recordmatmul(
            operand::result(normalized),
            operand::input(weights.getwv().getweight().getvalue().getview()),
        )?;
        let q = sub.recordrope_at(
            operand::result(q),
            position,
            self.rotaryq.getheads(),
            self.rotaryq.getbase(),
        )?;
        let k = sub.recordrope_at(
            operand::result(k),
            position,
            self.rotaryk.getheads(),
            self.rotaryk.getbase(),
        )?;
        // The first prefill token has no cached rows. Attention over an empty
        // cache plus the new row is well-defined, so pass zero-length views
        // instead of requiring the caller to seed the cache.
        let width = *weights
            .getwk()
            .getweight()
            .getvalue()
            .getshape()
            .getdims()
            .get(1)
            .ok_or_else(|| modelerror::config("kv cache", "projection weight is not rank two"))?;
        let emptykey = if cache.keys(layer).is_none() {
            Some(tensor::new(shape::new(vec![0, width])?, Vec::new())?)
        } else {
            None
        };
        let emptyvalue = if cache.values(layer).is_none() {
            Some(tensor::new(shape::new(vec![0, width])?, Vec::new())?)
        } else {
            None
        };
        let cached_key = match (cache.keys(layer), emptykey.as_ref()) {
            (Some(key), _) => key.getview(),
            (None, Some(key)) => key.getview(),
            (None, None) => return Err(modelerror::config("kv cache", "missing cached key")),
        };
        let cached_value = match (cache.values(layer), emptyvalue.as_ref()) {
            (Some(value), _) => value.getview(),
            (None, Some(value)) => value.getview(),
            (None, None) => return Err(modelerror::config("kv cache", "missing cached value")),
        };
        let attended = sub.recordattention_cached(
            operand::result(q),
            operandpair {
                cached: operand::input(cached_key),
                new: operand::result(k),
            },
            operandpair {
                cached: operand::input(cached_value),
                new: operand::result(v),
            },
            self.attention.getheads(),
            self.attention.getkvheads(),
            self.attention.getcausal(),
        )?;
        let projected = sub.recordmatmul(
            operand::result(attended),
            operand::input(weights.getwo().getweight().getvalue().getview()),
        )?;
        let residual = sub.recordadd(operand::result(x), operand::result(projected))?;
        let normalized = sub.recordrmsnorm(
            operand::result(residual),
            operand::input(weights.getffnnorm().getweight().getvalue().getview()),
            weights.getffnnorm().geteps(),
        )?;
        let gate = sub.recordmatmul(
            operand::result(normalized),
            operand::input(weights.getmlp().getgate().getweight().getvalue().getview()),
        )?;
        let up = sub.recordmatmul(
            operand::result(normalized),
            operand::input(weights.getmlp().getup().getweight().getvalue().getview()),
        )?;
        let silu = sub.recordsilu(operand::result(gate))?;
        let gated = sub.recordmul(operand::result(silu), operand::result(up))?;
        let feedforward = sub.recordmatmul(
            operand::result(gated),
            operand::input(weights.getmlp().getdown().getweight().getvalue().getview()),
        )?;
        let out = sub.recordadd(operand::result(residual), operand::result(feedforward))?;
        Ok((out, k, v))
    }

    /// Records one cache-aware block step against a device-resident cache.
    ///
    /// Attention reads the `[position, width]` view the cache exposes, and the
    /// new key/value row is appended with a device copy, so no cached row
    /// crosses the host boundary. Returns the residual output and the new
    /// key/value handles.
    pub fn forwardsubmission_cached_device<R: recorder, K: devicekvcache>(
        &self,
        sub: &mut R,
        cache: &mut K,
        weights: &llamalayerweights,
        x: stepref,
        layer: usize,
        position: usize,
    ) -> Result<(stepref, stepref, stepref), modelerror> {
        pinweights(sub, weights)?;
        let normalized = sub.recordrmsnorm(
            operand::result(x),
            operand::input(weights.getattnnorm().getweight().getvalue().getview()),
            weights.getattnnorm().geteps(),
        )?;
        let q = sub.recordmatmul(
            operand::result(normalized),
            operand::input(weights.getwq().getweight().getvalue().getview()),
        )?;
        let k = sub.recordmatmul(
            operand::result(normalized),
            operand::input(weights.getwk().getweight().getvalue().getview()),
        )?;
        let v = sub.recordmatmul(
            operand::result(normalized),
            operand::input(weights.getwv().getweight().getvalue().getview()),
        )?;
        let q = sub.recordrope_at(
            operand::result(q),
            position,
            self.rotaryq.getheads(),
            self.rotaryq.getbase(),
        )?;
        let k = sub.recordrope_at(
            operand::result(k),
            position,
            self.rotaryk.getheads(),
            self.rotaryk.getbase(),
        )?;
        let width = *weights
            .getwk()
            .getweight()
            .getvalue()
            .getshape()
            .getdims()
            .get(1)
            .ok_or_else(|| modelerror::config("kv cache", "projection weight is not rank two"))?;
        let cachedkey = cache.keys(sub, layer)?;
        let cachedvalue = cache.values(sub, layer)?;
        let emptykey = if cachedkey.is_none() {
            Some(tensor::new(shape::new(vec![0, width])?, Vec::new())?)
        } else {
            None
        };
        let emptyvalue = if cachedvalue.is_none() {
            Some(tensor::new(shape::new(vec![0, width])?, Vec::new())?)
        } else {
            None
        };
        let key_operand = match (cachedkey, emptykey.as_ref()) {
            (Some(handle), _) => operand::result(handle),
            (None, Some(key)) => operand::input(key.getview()),
            (None, None) => return Err(modelerror::config("kv cache", "missing cached key")),
        };
        let value_operand = match (cachedvalue, emptyvalue.as_ref()) {
            (Some(handle), _) => operand::result(handle),
            (None, Some(value)) => operand::input(value.getview()),
            (None, None) => return Err(modelerror::config("kv cache", "missing cached value")),
        };
        let attended = sub.recordattention_cached(
            operand::result(q),
            operandpair {
                cached: key_operand,
                new: operand::result(k),
            },
            operandpair {
                cached: value_operand,
                new: operand::result(v),
            },
            self.attention.getheads(),
            self.attention.getkvheads(),
            self.attention.getcausal(),
        )?;
        cache.append(sub, layer, k, v)?;
        let projected = sub.recordmatmul(
            operand::result(attended),
            operand::input(weights.getwo().getweight().getvalue().getview()),
        )?;
        let residual = sub.recordadd(operand::result(x), operand::result(projected))?;
        let normalized = sub.recordrmsnorm(
            operand::result(residual),
            operand::input(weights.getffnnorm().getweight().getvalue().getview()),
            weights.getffnnorm().geteps(),
        )?;
        let gate = sub.recordmatmul(
            operand::result(normalized),
            operand::input(weights.getmlp().getgate().getweight().getvalue().getview()),
        )?;
        let up = sub.recordmatmul(
            operand::result(normalized),
            operand::input(weights.getmlp().getup().getweight().getvalue().getview()),
        )?;
        let silu = sub.recordsilu(operand::result(gate))?;
        let gated = sub.recordmul(operand::result(silu), operand::result(up))?;
        let feedforward = sub.recordmatmul(
            operand::result(gated),
            operand::input(weights.getmlp().getdown().getweight().getvalue().getview()),
        )?;
        let out = sub.recordadd(operand::result(residual), operand::result(feedforward))?;
        Ok((out, k, v))
    }

    /// Runs one pre-norm transformer block for a single new token, reading the
    /// cached keys and values and appending the new ones.
    ///
    /// The cached context length is `position`, so the new token sits at
    /// absolute position `position`; the query is `[1, dim]` while the key and
    /// value are the cached `[position, kvwidth]` concatenated with the new row.
    pub fn forward_cached<C: compute, K: kvcache>(
        &self,
        provider: &C,
        cache: &mut K,
        weights: &llamalayerweights,
        x: &tensor<f32>,
        layer: usize,
        position: usize,
    ) -> Result<tensor<f32>, modelerror> {
        let normalized = provider.rmsnorm(
            &x.getview(),
            &weights.getattnnorm().getweight().getvalue().getview(),
            weights.getattnnorm().geteps(),
        )?;
        let q = provider.matmul(
            &normalized.getview(),
            &weights.getwq().getweight().getvalue().getview(),
        )?;
        let k = provider.matmul(
            &normalized.getview(),
            &weights.getwk().getweight().getvalue().getview(),
        )?;
        let v = provider.matmul(
            &normalized.getview(),
            &weights.getwv().getweight().getvalue().getview(),
        )?;
        let q = ropeat(
            provider,
            &q,
            position,
            self.rotaryq.getheads(),
            self.rotaryq.getbase(),
        )?;
        let k = ropeat(
            provider,
            &k,
            position,
            self.rotaryk.getheads(),
            self.rotaryk.getbase(),
        )?;
        let kfull = concatkv(cache.keys(layer), &k)?;
        let vfull = concatkv(cache.values(layer), &v)?;
        let attended = provider.attention(
            &q.getview(),
            &kfull.getview(),
            &vfull.getview(),
            self.attention.getheads(),
            self.attention.getkvheads(),
            self.attention.getcausal(),
        )?;
        cache.append(layer, &k, &v)?;
        let projected = provider.matmul(
            &attended.getview(),
            &weights.getwo().getweight().getvalue().getview(),
        )?;
        let residual = provider.add(&x.getview(), &projected.getview())?;
        let normalized = provider.rmsnorm(
            &residual.getview(),
            &weights.getffnnorm().getweight().getvalue().getview(),
            weights.getffnnorm().geteps(),
        )?;
        let gate = provider.matmul(
            &normalized.getview(),
            &weights.getmlp().getgate().getweight().getvalue().getview(),
        )?;
        let up = provider.matmul(
            &normalized.getview(),
            &weights.getmlp().getup().getweight().getvalue().getview(),
        )?;
        let silu = provider.silu(&gate.getview())?;
        let gated = provider.mul(&silu.getview(), &up.getview())?;
        let feedforward = provider.matmul(
            &gated.getview(),
            &weights.getmlp().getdown().getweight().getvalue().getview(),
        )?;
        provider
            .add(&residual.getview(), &feedforward.getview())
            .map_err(modelerror::from)
    }
}

/// Applies rotary position embedding to a single `[1, width]` token at absolute
/// `position`, by padding a `[position + 1, width]` operand and taking the last
/// row. Reuses the `rope` contract unchanged; a dedicated absolute-position
/// kernel is a later optimization.
fn ropeat<C: compute>(
    provider: &C,
    x: &tensor<f32>,
    position: usize,
    heads: usize,
    base: f32,
) -> Result<tensor<f32>, modelerror> {
    let width = x.getshape().getdims()[1];
    let mut data = vec![0.0_f32; (position + 1) * width];
    data[position * width..].copy_from_slice(x.getdata());
    let padded = tensor::new(shape::new(vec![position + 1, width])?, data)?;
    let rotated = provider.rope(&padded.getview(), heads, base)?;
    let start = position * width;
    let last = rotated.getdata()[start..start + width].to_vec();
    tensor::new(shape::new(vec![1, width])?, last).map_err(modelerror::from)
}

/// Concatenates a cached `[len, width]` tensor with a new `[1, width]` row.
fn concatkv(existing: Option<&tensor<f32>>, new: &tensor<f32>) -> Result<tensor<f32>, modelerror> {
    let width = new.getshape().getdims()[1];
    let mut data = match existing {
        Some(t) => t.getdata().to_vec(),
        None => Vec::new(),
    };
    data.extend_from_slice(new.getdata());
    let len = data.len() / width;
    tensor::new(shape::new(vec![len, width])?, data).map_err(modelerror::from)
}
