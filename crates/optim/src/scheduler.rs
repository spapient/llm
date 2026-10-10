use core::f64::consts::PI;

use crate::error::opterror;

/// The concrete schedule held by an [`lrschedule`].
#[derive(Clone, Copy, Debug, PartialEq)]
enum schedulekind {
    /// A fixed learning rate.
    constant { lr: f32 },
    /// Multiply the initial rate by `gamma` every `steplen` steps.
    steplr {
        initial: f32,
        steplen: u64,
        gamma: f32,
        min: f32,
    },
    /// Multiply the initial rate by `gamma` on every step.
    exponential { initial: f32, gamma: f32, min: f32 },
    /// Cosine decay from `initial` to `min` over `period` steps.
    cosineannealing { initial: f32, min: f32, period: u64 },
    /// Linear warmup from `initial` to `peak`, then cosine decay to `min`.
    warmupcosine {
        initial: f32,
        peak: f32,
        min: f32,
        warmup: u64,
        period: u64,
    },
}

/// A validated learning-rate schedule.
///
/// A schedule is a pure function from a zero-based step index to a learning
/// rate, so it carries no mutable state; the training loop queries
/// [`Self::getlr`] and applies the result through
/// [`optimizer::setlr`](crate::optimizer::setlr). Values are built through the
/// constructors, each of which rejects an invalid domain, so `getlr` is total.
///
/// # Equations
///
/// Let `s` be the step index and `s_0` the index of the first step after
/// warmup.
///
/// ```text
/// constant:          lr
/// steplr:            max(min, initial * gamma^(s / steplen))
/// exponential:       max(min, initial * gamma^s)
/// cosineannealing:   min + (initial - min) * (1 + cos(pi * min(s, period) / period)) / 2
/// warmupcosine:      initial + (peak - initial) * s / warmup                  if s < warmup
///                    min + (peak - min) * (1 + cos(pi * min(s - warmup, period) / period)) / 2   otherwise
/// ```
///
/// Every schedule with a `min` bound is monotone non-increasing after any
/// warmup and never falls below `min`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct lrschedule {
    kind: schedulekind,
}

impl lrschedule {
    /// Returns a schedule with a fixed learning rate.
    ///
    /// # Errors
    ///
    /// Returns [`opterror::invalidhyperparameter`] when `lr` is not finite or
    /// is negative.
    pub fn constant(lr: f32) -> Result<Self, opterror> {
        nonnegative("lr", lr)?;
        Ok(Self {
            kind: schedulekind::constant { lr },
        })
    }

    /// Returns a step-decay schedule.
    ///
    /// # Errors
    ///
    /// Returns [`opterror::invalidhyperparameter`] when `initial` or `min` is
    /// not finite or negative, when `gamma` is not finite or outside `[0, 1]`,
    /// or when `steplen` is zero.
    pub fn steplr(initial: f32, steplen: u64, gamma: f32, min: f32) -> Result<Self, opterror> {
        nonnegative("initial", initial)?;
        nonnegative("min", min)?;
        fraction("gamma", gamma)?;
        if steplen == 0 {
            return Err(opterror::hyperparameter("steplen", "must be at least one"));
        }
        Ok(Self {
            kind: schedulekind::steplr {
                initial,
                steplen,
                gamma,
                min,
            },
        })
    }

    /// Returns an exponential-decay schedule.
    ///
    /// # Errors
    ///
    /// Returns [`opterror::invalidhyperparameter`] when `initial` or `min` is
    /// not finite or negative, or when `gamma` is not finite or outside `[0, 1]`.
    pub fn exponential(initial: f32, gamma: f32, min: f32) -> Result<Self, opterror> {
        nonnegative("initial", initial)?;
        nonnegative("min", min)?;
        fraction("gamma", gamma)?;
        Ok(Self {
            kind: schedulekind::exponential {
                initial,
                gamma,
                min,
            },
        })
    }

    /// Returns a cosine-annealing schedule.
    ///
    /// # Errors
    ///
    /// Returns [`opterror::invalidhyperparameter`] when `initial` or `min` is
    /// not finite or negative, or when `period` is zero.
    pub fn cosineannealing(initial: f32, min: f32, period: u64) -> Result<Self, opterror> {
        nonnegative("initial", initial)?;
        nonnegative("min", min)?;
        if period == 0 {
            return Err(opterror::hyperparameter("period", "must be at least one"));
        }
        Ok(Self {
            kind: schedulekind::cosineannealing {
                initial,
                min,
                period,
            },
        })
    }

    /// Returns a linear-warmup-plus-cosine schedule.
    ///
    /// # Errors
    ///
    /// Returns [`opterror::invalidhyperparameter`] when `initial`, `peak`, or
    /// `min` is not finite or negative, when `period` is zero, or when
    /// `warmup` exceeds `period`.
    pub fn linearwarmupcosine(
        initial: f32,
        peak: f32,
        min: f32,
        warmup: u64,
        period: u64,
    ) -> Result<Self, opterror> {
        nonnegative("initial", initial)?;
        nonnegative("peak", peak)?;
        nonnegative("min", min)?;
        if period == 0 {
            return Err(opterror::hyperparameter("period", "must be at least one"));
        }
        if warmup > period {
            return Err(opterror::hyperparameter("warmup", "must not exceed period"));
        }
        Ok(Self {
            kind: schedulekind::warmupcosine {
                initial,
                peak,
                min,
                warmup,
                period,
            },
        })
    }

    /// Returns the learning rate for zero-based step `step`.
    #[must_use]
    pub fn getlr(&self, step: u64) -> f32 {
        match self.kind {
            schedulekind::constant { lr } => lr,
            schedulekind::steplr {
                initial,
                steplen,
                gamma,
                min,
            } => {
                let decays = step / steplen;
                let exponent = i32::try_from(decays).unwrap_or(i32::MAX);
                let factor = f64::from(gamma).powi(exponent);
                floor(f64::from(initial) * factor, f64::from(min))
            }
            schedulekind::exponential {
                initial,
                gamma,
                min,
            } => {
                let factor = f64::from(gamma).powf(step as f64);
                floor(f64::from(initial) * factor, f64::from(min))
            }
            schedulekind::cosineannealing {
                initial,
                min,
                period,
            } => {
                let elapsed = step.min(period);
                cosine(f64::from(initial), f64::from(min), elapsed, period)
            }
            schedulekind::warmupcosine {
                initial,
                peak,
                min,
                warmup,
                period,
            } => {
                if warmup > 0 && step < warmup {
                    let fraction = step as f64 / warmup as f64;
                    let value =
                        f64::from(initial) + (f64::from(peak) - f64::from(initial)) * fraction;
                    value as f32
                } else {
                    let elapsed = step.saturating_sub(warmup).min(period);
                    cosine(f64::from(peak), f64::from(min), elapsed, period)
                }
            }
        }
    }
}

/// Returns `max(value, min)` as `f32`.
fn floor(value: f64, min: f64) -> f32 {
    let bounded = if value < min { min } else { value };
    bounded as f32
}

/// Evaluates the cosine interpolation from `high` to `low` at `elapsed` of `period`.
fn cosine(high: f64, low: f64, elapsed: u64, period: u64) -> f32 {
    let angle = PI * (elapsed as f64) / (period as f64);
    (low + (high - low) * (1.0 + angle.cos()) / 2.0) as f32
}

/// Rejects a value that is not finite and non-negative.
fn nonnegative(field: &'static str, value: f32) -> Result<(), opterror> {
    if !value.is_finite() || value < 0.0 {
        return Err(opterror::hyperparameter(
            field,
            "must be finite and non-negative",
        ));
    }
    Ok(())
}

/// Rejects a value that is not finite or outside `[0, 1]`.
fn fraction(field: &'static str, value: f32) -> Result<(), opterror> {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        return Err(opterror::hyperparameter(
            field,
            "must be finite and in [0, 1]",
        ));
    }
    Ok(())
}
