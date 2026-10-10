use ::tensor::tensor;

use crate::error::opterror;

/// The gradient-clipping method held by a [`clipconfig`].
#[derive(Clone, Copy, Debug, PartialEq)]
enum clipkind {
    /// No clipping.
    none,
    /// Clip the global L2 norm across every gradient element to `maxnorm`.
    bynorm { maxnorm: f32 },
    /// Clamp every gradient element into `[min, max]`.
    byvalue { min: f32, max: f32 },
}

/// A validated gradient-clipping configuration.
///
/// A value is built through one of the constructors, each of which rejects an
/// invalid domain, so the stored configuration is always well-formed and
/// [`clipgradients`] cannot fail. The default is `none`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct clipconfig {
    kind: clipkind,
}

impl clipconfig {
    /// Returns a configuration that performs no clipping.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            kind: clipkind::none,
        }
    }

    /// Returns a global-norm clipping configuration.
    ///
    /// # Errors
    ///
    /// Returns [`opterror::invalidhyperparameter`] when `maxnorm` is not finite
    /// or is negative.
    pub fn bynorm(maxnorm: f32) -> Result<Self, opterror> {
        if !maxnorm.is_finite() || maxnorm < 0.0 {
            return Err(opterror::hyperparameter(
                "maxnorm",
                "must be finite and non-negative",
            ));
        }
        Ok(Self {
            kind: clipkind::bynorm { maxnorm },
        })
    }

    /// Returns an elementwise value-clipping configuration.
    ///
    /// # Errors
    ///
    /// Returns [`opterror::invalidhyperparameter`] when either bound is not
    /// finite or when `min` is greater than `max`.
    pub fn byvalue(min: f32, max: f32) -> Result<Self, opterror> {
        if !min.is_finite() || !max.is_finite() {
            return Err(opterror::hyperparameter("bounds", "must be finite"));
        }
        if min > max {
            return Err(opterror::hyperparameter(
                "bounds",
                "min must not exceed max",
            ));
        }
        Ok(Self {
            kind: clipkind::byvalue { min, max },
        })
    }

    /// Returns `true` when this configuration changes gradients.
    #[must_use]
    pub fn isenabled(&self) -> bool {
        !matches!(self.kind, clipkind::none)
    }
}

/// Applies `config` to `grads` in place.
///
/// * [`clipconfig::none`] leaves the gradients unchanged.
/// * By-value clipping clamps every element independently into the configured
///   `[min, max]`.
/// * By-norm clipping treats all elements of every gradient as one vector and,
///   when its L2 norm exceeds `maxnorm`, scales every element by
///   `maxnorm / norm`. The norm is accumulated in `f64` for determinism and
///   numerical headroom. A zero norm, or a norm within `maxnorm`, leaves the
///   gradients unchanged; `maxnorm = 0` therefore zeroes a non-zero gradient
///   exactly, with no division by zero.
///
/// The function is infallible because the configuration it accepts is already
/// validated by [`clipconfig`]'s constructors.
pub fn clipgradients(grads: &mut [tensor<f32>], config: clipconfig) {
    match config.kind {
        clipkind::none => {}
        clipkind::byvalue { min, max } => {
            for grad in grads.iter_mut() {
                for value in grad.getdatamut() {
                    *value = value.clamp(min, max);
                }
            }
        }
        clipkind::bynorm { maxnorm } => {
            let mut sumsq = 0.0_f64;
            for grad in grads.iter() {
                for value in grad.getdata() {
                    sumsq += f64::from(*value) * f64::from(*value);
                }
            }
            let norm = sumsq.sqrt();
            let limit = f64::from(maxnorm);
            if norm <= limit || norm == 0.0 {
                return;
            }
            let scale = (limit / norm) as f32;
            for grad in grads.iter_mut() {
                for value in grad.getdatamut() {
                    *value *= scale;
                }
            }
        }
    }
}
