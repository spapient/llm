use tensor::tensor;

use crate::error::trainerror;
use crate::loss::crossentropy;

/// One evaluation sample: logits and the target class of every row.
///
/// The pair is supplied by the caller because `train` does not own a forward
/// pass; the caller builds logits (for example with a forward-only `nn` model)
/// and hands them to [`evaluate`] together with the classes they should predict.
#[derive(Clone, Debug)]
pub struct evalsample {
    logits: tensor<f32>,
    targets: Vec<usize>,
}

impl evalsample {
    /// Pairs logits with their per-row target classes.
    #[must_use]
    pub fn new(logits: tensor<f32>, targets: Vec<usize>) -> Self {
        Self { logits, targets }
    }

    /// Returns the logits.
    #[must_use]
    pub const fn getlogits(&self) -> &tensor<f32> {
        &self.logits
    }

    /// Returns the target classes.
    #[must_use]
    pub fn gettargets(&self) -> &[usize] {
        &self.targets
    }
}

/// The aggregate result of an evaluation pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct evalresult {
    loss: f32,
    perplexity: f32,
    rows: usize,
}

impl evalresult {
    /// Returns the row-weighted mean cross-entropy across all samples.
    #[must_use]
    pub const fn getloss(&self) -> f32 {
        self.loss
    }

    /// Returns the perplexity, `exp(loss)`.
    ///
    /// A loss large enough to overflow `f32` saturates the perplexity to
    /// positive infinity; that is a representational limit, not a validation
    /// failure, so it is reported rather than rejected.
    #[must_use]
    pub const fn getperplexity(&self) -> f32 {
        self.perplexity
    }

    /// Returns the total number of scored rows.
    #[must_use]
    pub const fn getrows(&self) -> usize {
        self.rows
    }
}

/// Aggregates cross-entropy over evaluation samples.
///
/// Each sample's cross-entropy is weighted by its row count, so the reported
/// loss is the mean over all rows rather than the mean of per-sample means, and
/// the perplexity is `exp` of that mean.
///
/// # Errors
///
/// Returns [`trainerror::invalidstate`] when `samples` is empty, and any error
/// [`crossentropy`] raises for an individual sample.
pub fn evaluate(samples: &[evalsample]) -> Result<evalresult, trainerror> {
    if samples.is_empty() {
        return Err(trainerror::state("evaluation requires at least one sample"));
    }
    let mut total = 0.0_f32;
    let mut rows = 0_usize;
    for sample in samples {
        let result = crossentropy(&sample.logits, &sample.targets)?;
        let count = sample.targets.len();
        total += result.getloss() * count as f32;
        rows += count;
    }
    if rows == 0 {
        return Err(trainerror::state(
            "evaluation samples contain no rows to score",
        ));
    }
    let loss = total / rows as f32;
    if !loss.is_finite() {
        return Err(trainerror::lossnonfinite);
    }
    Ok(evalresult {
        loss,
        perplexity: loss.exp(),
        rows,
    })
}
