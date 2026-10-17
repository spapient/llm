use crate::error::modelerror;

/// The architecture configuration of a Llama-family causal language model.
///
/// Every field is validated at construction, so a `llamaconfig` that exists is
/// a configuration whose derived dimensions are consistent: the model dimension
/// divides evenly into query heads, the query head count is a multiple of the
/// key/value head count, and the head dimension is even so rotary pairs exist.
/// The derived values ([`Self::getheaddim`], [`Self::getkvwidth`]) are computed
/// from the stored fields rather than stored twice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct llamaconfig {
    vocab: usize,
    dim: usize,
    nlayers: usize,
    nheads: usize,
    nkvheads: usize,
    hidden: usize,
    rmseps: f32,
    ropebase: f32,
    maxseq: usize,
}

impl llamaconfig {
    /// Builds a validated configuration.
    ///
    /// # Errors
    ///
    /// Returns [`modelerror::invalidconfig`] when any dimension is zero, the
    /// model dimension is not divisible by the head count, the head counts are
    /// not divisible, the head dimension is odd, or `rmseps`/`ropebase` are not
    /// finite and non-negative/positive respectively.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        vocab: usize,
        dim: usize,
        nlayers: usize,
        nheads: usize,
        nkvheads: usize,
        hidden: usize,
        rmseps: f32,
        ropebase: f32,
        maxseq: usize,
    ) -> Result<Self, modelerror> {
        if vocab == 0 {
            return Err(modelerror::config("vocab", "must be non-zero"));
        }
        if dim == 0 {
            return Err(modelerror::config("dim", "must be non-zero"));
        }
        if nlayers == 0 {
            return Err(modelerror::config("nlayers", "must be non-zero"));
        }
        if nheads == 0 || nkvheads == 0 {
            return Err(modelerror::config("nheads", "must be non-zero"));
        }
        if !dim.is_multiple_of(nheads) {
            return Err(modelerror::config("dim", "must be divisible by nheads"));
        }
        if !nheads.is_multiple_of(nkvheads) {
            return Err(modelerror::config(
                "nheads",
                "must be a multiple of nkvheads",
            ));
        }
        if !(dim / nheads).is_multiple_of(2) {
            return Err(modelerror::config("dim", "head dimension must be even"));
        }
        if hidden == 0 {
            return Err(modelerror::config("hidden", "must be non-zero"));
        }
        if !rmseps.is_finite() || rmseps < 0.0 {
            return Err(modelerror::config(
                "rmseps",
                "must be finite and non-negative",
            ));
        }
        if !ropebase.is_finite() || ropebase <= 0.0 {
            return Err(modelerror::config(
                "ropebase",
                "must be finite and positive",
            ));
        }
        if maxseq == 0 {
            return Err(modelerror::config("maxseq", "must be non-zero"));
        }
        Ok(Self {
            vocab,
            dim,
            nlayers,
            nheads,
            nkvheads,
            hidden,
            rmseps,
            ropebase,
            maxseq,
        })
    }

    /// Returns the vocabulary size.
    #[must_use]
    pub const fn getvocab(&self) -> usize {
        self.vocab
    }

    /// Returns the model dimension.
    #[must_use]
    pub const fn getdim(&self) -> usize {
        self.dim
    }

    /// Returns the number of transformer layers.
    #[must_use]
    pub const fn getnlayers(&self) -> usize {
        self.nlayers
    }

    /// Returns the query head count.
    #[must_use]
    pub const fn getnheads(&self) -> usize {
        self.nheads
    }

    /// Returns the key/value head count.
    #[must_use]
    pub const fn getnkvheads(&self) -> usize {
        self.nkvheads
    }

    /// Returns the feed-forward hidden dimension.
    #[must_use]
    pub const fn gethidden(&self) -> usize {
        self.hidden
    }

    /// Returns the RMS normalization epsilon.
    #[must_use]
    pub const fn getrmseps(&self) -> f32 {
        self.rmseps
    }

    /// Returns the rotary frequency base.
    #[must_use]
    pub const fn getropebase(&self) -> f32 {
        self.ropebase
    }

    /// Returns the maximum sequence length.
    #[must_use]
    pub const fn getmaxseq(&self) -> usize {
        self.maxseq
    }

    /// Returns the per-head dimension, `dim / nheads`.
    #[must_use]
    pub const fn getheaddim(&self) -> usize {
        self.dim / self.nheads
    }

    /// Returns the key/value width, `nkvheads * head_dim`.
    #[must_use]
    pub const fn getkvwidth(&self) -> usize {
        self.nkvheads * self.getheaddim()
    }
}
