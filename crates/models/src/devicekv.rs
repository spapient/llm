//! A backend-agnostic device-resident key/value cache.
//!
//! [`devicecache`] implements the [`devicekvcache`] contract against the
//! recorder's persistent-input primitives, so the same code drives the CPU
//! reference, the Metal backend, and any other recorder:
//!
//! * `recordpersist` registers each layer's `[maxseq, width]` buffer once
//!   (the engine and CPU reference deduplicate on `(pointer, length)`, so the
//!   same host buffer handed back every step reuses one device allocation);
//! * `recordview` exposes the leading `[position, width]` window to the
//!   cached attention path — the kernel reads the leading rows, which is
//!   exactly the prefix a cache-row window needs;
//! * `recordappend` copies the new row into the device buffer without a host
//!   round trip, so no cached row ever crosses the host boundary.
//!
//! The host buffers stay zeroed; the real rows live on the device, appended in
//! place. `getlen`/`advance` mirror the host `kvstate` position so the model's
//! `forwardsubmission_cached_device` can place each token at its absolute row.
use ::tensor::shape;
use api::{backenderror, operand, recorder, stepref};

use crate::causal::devicekvcache;
use crate::error::modelerror;

/// One layer's persistent key/value host buffers.
///
/// `keydata`/`valuedata` are the zeroed `[maxseq, width]` buffers whose
/// `(pointer, length)` is the deduplication key every `recordpersist` call
/// reuses: the engine's persistent table and the CPU reference both map the
/// same registered buffer back to the same device slot, so one forward pass
/// pays at most one upload per layer no matter how many steps reference it.
/// A `stepref` is submission-local, so the cache never stores one; it
/// re-registers through the recorder each call and lets the dedup table name
/// the slot.
struct layerbuffers {
    keydata: Vec<f32>,
    valuedata: Vec<f32>,
}

/// A device-resident key/value cache for incremental decoding.
///
/// Created by and owned by the caller (a generation request), never stored in
/// model weights (OW2). Position advances in lockstep across layers; the
/// appended rows live on the device and are never downloaded as step outputs.
pub struct devicecache {
    layers: usize,
    width: usize,
    maxseq: usize,
    position: usize,
    buffers: Vec<layerbuffers>,
}

impl devicecache {
    /// Builds an empty device-resident cache for `layers` layers of width
    /// `width`, holding at most `maxseq` positions per layer.
    ///
    /// # Errors
    ///
    /// Returns [`modelerror::invalidconfig`] when any dimension is zero.
    pub fn new(layers: usize, width: usize, maxseq: usize) -> Result<Self, modelerror> {
        if layers == 0 {
            return Err(modelerror::invalidconfig {
                field: "kvlayers",
                detail: "must be non-zero".to_string(),
            });
        }
        if width == 0 {
            return Err(modelerror::invalidconfig {
                field: "kvwidth",
                detail: "must be non-zero".to_string(),
            });
        }
        if maxseq == 0 {
            return Err(modelerror::invalidconfig {
                field: "maxseq",
                detail: "must be non-zero".to_string(),
            });
        }
        let buffers = (0..layers)
            .map(|_| layerbuffers {
                keydata: vec![0.0; maxseq * width],
                valuedata: vec![0.0; maxseq * width],
            })
            .collect();
        Ok(Self {
            layers,
            width,
            maxseq,
            position: 0,
            buffers,
        })
    }

    /// Returns the per-layer key/value width.
    #[must_use]
    pub const fn getwidth(&self) -> usize {
        self.width
    }

    /// Resets the position.
    ///
    /// The host buffers stay allocated and their deduplication keys unchanged,
    /// so a reused cache re-registers against the same device slots.
    pub fn clear(&mut self) {
        self.position = 0;
    }

    /// Registers a layer's buffers through `sub`.
    ///
    /// Registration happens on every call because a `stepref` is valid only
    /// for the submission that produced it; the recorder's deduplication table
    /// (engine persistent table / CPU reference) maps the same host buffer
    /// back to the same slot, so repeated registration within one submission
    /// is a hash lookup, not a re-upload.
    ///
    /// # Errors
    ///
    /// Returns a backend error when either registration fails.
    fn ensure_registered<R: recorder>(
        &self,
        sub: &mut R,
        layer: usize,
    ) -> Result<(stepref, stepref), backenderror> {
        let buffer = &self.buffers[layer];
        let shape = shape::new(vec![self.maxseq, self.width])
            .map_err(|_| backenderror::invalidparameter { op: "persist" })?;
        let k = sub.recordpersist(&buffer.keydata, shape.clone())?;
        let v = sub.recordpersist(&buffer.valuedata, shape)?;
        Ok((k, v))
    }
}

impl devicekvcache for devicecache {
    fn getlayers(&self) -> usize {
        self.layers
    }

    fn getlen(&self) -> usize {
        self.position
    }

    fn getmaxseq(&self) -> usize {
        self.maxseq
    }

    fn keys<R: recorder>(
        &mut self,
        sub: &mut R,
        layer: usize,
    ) -> Result<Option<stepref>, backenderror> {
        if self.position == 0 {
            return Ok(None);
        }
        let (keyslot, _) = self.ensure_registered(sub, layer)?;
        let view = sub.recordview(
            keyslot,
            shape::new(vec![self.position, self.width])
                .map_err(|_| backenderror::invalidparameter { op: "view" })?,
        )?;
        Ok(Some(view))
    }

    fn values<R: recorder>(
        &mut self,
        sub: &mut R,
        layer: usize,
    ) -> Result<Option<stepref>, backenderror> {
        if self.position == 0 {
            return Ok(None);
        }
        let (_, valueslot) = self.ensure_registered(sub, layer)?;
        let view = sub.recordview(
            valueslot,
            shape::new(vec![self.position, self.width])
                .map_err(|_| backenderror::invalidparameter { op: "view" })?,
        )?;
        Ok(Some(view))
    }

    fn append<R: recorder>(
        &mut self,
        sub: &mut R,
        layer: usize,
        key: stepref,
        value: stepref,
    ) -> Result<(), backenderror> {
        let (keyslot, valueslot) = self.ensure_registered(sub, layer)?;
        sub.recordappend(
            operand::result(keyslot),
            operand::result(key),
            self.position,
        )?;
        sub.recordappend(
            operand::result(valueslot),
            operand::result(value),
            self.position,
        )
    }

    fn advance(&mut self) {
        self.position += 1;
    }
}
