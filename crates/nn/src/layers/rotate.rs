use ::tensor::tensor;
use api::compute;

use crate::dictionary::statedict;
use crate::error::nnerror;
use crate::modules::module;

/// The default rotary frequency base.
pub const DEFAULTBASE: f32 = 10_000.0;

/// A rotary position embedding with no learned parameters.
///
/// The input has shape `[..., seq, heads * headdim]`. Each contiguous `headdim`
/// slice of the last axis is rotated by its position-dependent angle through
/// `api::compute::rope`. The layer validates the head layout and forwards the
/// rotation base.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct rotary {
    heads: usize,
    base: f32,
}

impl rotary {
    /// Builds a rotary embedding.
    ///
    /// # Errors
    ///
    /// Returns [`nnerror::invalidconfig`] when `heads` is zero or `base` is not
    /// finite and positive.
    pub fn new(heads: usize, base: f32) -> Result<Self, nnerror> {
        if heads == 0 {
            return Err(nnerror::config("heads", "must be non-zero"));
        }
        if !base.is_finite() || base <= 0.0 {
            return Err(nnerror::config("base", "must be finite and positive"));
        }
        Ok(Self { heads, base })
    }

    /// Returns the head count.
    #[must_use]
    pub fn getheads(&self) -> usize {
        self.heads
    }

    /// Returns the frequency base.
    #[must_use]
    pub fn getbase(&self) -> f32 {
        self.base
    }

    /// Applies the rotation to `input`.
    ///
    /// # Errors
    ///
    /// Returns [`nnerror::shapemismatch`] when the rank is below two, the last
    /// dimension is not a multiple of the head count, or the head dimension is
    /// odd; or forwards the provider's error.
    pub fn forward<C: compute>(
        &self,
        engine: &C,
        input: &tensor<f32>,
    ) -> Result<tensor<f32>, nnerror> {
        let dims = input.getshape().getdims();
        if dims.len() < 2 {
            return Err(nnerror::shapemismatch {
                layer: "rotary",
                expected: vec![0, self.heads * 2],
                actual: dims.to_vec(),
            });
        }
        let last = dims[dims.len() - 1];
        if !last.is_multiple_of(self.heads) || !(last / self.heads).is_multiple_of(2) {
            return Err(nnerror::shapemismatch {
                layer: "rotary",
                expected: vec![0, self.heads * 2],
                actual: dims.to_vec(),
            });
        }
        Ok(engine.rope(&input.getview(), self.heads, self.base)?)
    }
}

impl module for rotary {
    fn getname(&self) -> &'static str {
        "rotary"
    }

    fn collect(&self, _prefix: &str, _dict: &mut statedict) -> Result<(), nnerror> {
        Ok(())
    }
}
