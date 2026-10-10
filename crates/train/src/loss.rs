use ::tensor::{shape, tensor};

use crate::error::trainerror;

/// The scalar loss and its gradient with respect to the logits.
///
/// The gradient has the same shape as the logits it was computed from, so it can
/// be handed directly to an optimizer that is keyed by parameter position. The
/// pair is returned together so a caller never recomputes the forward pass to
/// obtain the backward value.
#[derive(Clone, Debug)]
pub struct lossresult {
    loss: f32,
    gradient: tensor<f32>,
}

impl lossresult {
    /// Builds a loss result from its scalar and gradient.
    #[must_use]
    pub fn new(loss: f32, gradient: tensor<f32>) -> Self {
        Self { loss, gradient }
    }

    /// Returns the scalar loss.
    #[must_use]
    pub const fn getloss(&self) -> f32 {
        self.loss
    }

    /// Returns the gradient with respect to the logits.
    #[must_use]
    pub const fn getgradient(&self) -> &tensor<f32> {
        &self.gradient
    }

    /// Splits the result into its scalar loss and gradient.
    #[must_use]
    pub fn intoparts(self) -> (f32, tensor<f32>) {
        (self.loss, self.gradient)
    }
}

/// Computes the mean cross-entropy of `logits` against integer class targets.
///
/// `logits` is a rank-two matrix `[rows, classes]`; `targets[i]` selects the
/// correct class for row `i` and must be strictly less than `classes`. The
/// returned loss is the mean over rows of the negative log-probability of the
/// target class under a softmax, and the returned gradient is the exact
/// analytic derivative with respect to the logits, already divided by the row
/// count so that it averages with the loss.
///
/// The softmax is evaluated with the max-subtraction (log-sum-exp) form, so a
/// large logit magnitude shifts the logits toward zero instead of overflowing
/// `f32`.
///
/// # Errors
///
/// Returns [`trainerror::lossnotmatrix`] when `logits` is not rank two,
/// [`trainerror::lossemptymatrix`] when it has no rows or no classes,
/// [`trainerror::targetcountmismatch`] when `targets` does not have one entry
/// per row, [`trainerror::targetoutofrange`] when a target exceeds the class
/// count, and [`trainerror::lossnonfinite`] when a computed loss or gradient is
/// not finite. A rejected call does not mutate any input.
pub fn crossentropy(logits: &tensor<f32>, targets: &[usize]) -> Result<lossresult, trainerror> {
    let dims = logits.getshape().getdims();
    if dims.len() != 2 {
        return Err(trainerror::lossnotmatrix {
            dims: dims.to_vec(),
        });
    }
    let rows = dims[0];
    let classes = dims[1];
    if rows == 0 || classes == 0 {
        return Err(trainerror::lossemptymatrix {
            dims: dims.to_vec(),
        });
    }
    if targets.len() != rows {
        return Err(trainerror::targetcountmismatch {
            rows,
            targets: targets.len(),
        });
    }
    for (index, &label) in targets.iter().enumerate() {
        if label >= classes {
            return Err(trainerror::targetoutofrange {
                index,
                label,
                classes,
            });
        }
    }

    let values = logits.getdata();
    let mut gradient = vec![0.0_f32; values.len()];
    let mut total = 0.0_f32;
    for (row, &label) in targets.iter().enumerate() {
        let base = row
            .checked_mul(classes)
            .ok_or_else(|| trainerror::state("cross-entropy row offset overflows usize"))?;
        let window = values
            .get(base..base + classes)
            .ok_or_else(|| trainerror::state("cross-entropy row exceeds the logits buffer"))?;
        let maximum = window.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for &value in window {
            sum += (value - maximum).exp();
        }
        if !sum.is_finite() || sum <= 0.0 {
            return Err(trainerror::lossnonfinite);
        }
        let logsumexp = maximum + sum.ln();
        total += logsumexp - window[label];

        let outbase = base;
        let scale = 1.0 / (sum * rows as f32);
        for (class, slot) in gradient[outbase..outbase + classes].iter_mut().enumerate() {
            *slot = (window[class] - maximum).exp() * scale;
        }
        gradient[outbase + label] -= 1.0 / rows as f32;
    }
    let loss = total / rows as f32;
    if !loss.is_finite() || gradient.iter().any(|value| !value.is_finite()) {
        return Err(trainerror::lossnonfinite);
    }
    Ok(lossresult::new(
        loss,
        tensor::new(shape::new(dims.to_vec())?, gradient)?,
    ))
}
