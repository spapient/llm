use ::tensor::tensor;
use api::compute;
use models::causallm;

use crate::error::infererror;
use crate::stopping::stopreason;

/// The outcome of one generation attempt.
///
/// The result owns the generated tokens (the prompt is not repeated), the
/// reason generation stopped, and the number of forward passes performed. It is
/// a value, so it remains valid after the request that produced it is reset or
/// dropped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct generationresult {
    tokens: Vec<usize>,
    reason: stopreason,
    steps: usize,
}

impl generationresult {
    pub(crate) fn new(tokens: Vec<usize>, reason: stopreason, steps: usize) -> Self {
        Self {
            tokens,
            reason,
            steps,
        }
    }

    /// Returns the generated tokens in order.
    #[must_use]
    pub fn gettokens(&self) -> &[usize] {
        &self.tokens
    }

    /// Returns the number of generated tokens.
    #[must_use]
    pub fn getcount(&self) -> usize {
        self.tokens.len()
    }

    /// Returns why generation stopped.
    #[must_use]
    pub const fn getreason(&self) -> stopreason {
        self.reason
    }

    /// Returns the number of forward passes performed.
    #[must_use]
    pub const fn getsteps(&self) -> usize {
        self.steps
    }
}

/// Runs one forward pass over `tokens` and returns the next-token logits of the
/// last position.
///
/// Both the prefill and decode phases reduce to this computation while the
/// correctness path recomputes the full context. The prefill and decode entry
/// points keep the two phases separately named so a cache-aware decode can
/// replace one of them without changing the other.
pub(crate) fn nextlogits<M: causallm, C: compute>(
    model: &M,
    provider: &C,
    tokens: &[usize],
) -> Result<Vec<f32>, infererror> {
    let logits = model.forward(provider, tokens)?;
    lastrow(&logits)
}

/// Returns the last row of a `[rows, vocab]` logits tensor as owned values.
fn lastrow(logits: &tensor<f32>) -> Result<Vec<f32>, infererror> {
    let vocab = logits
        .getshape()
        .getdims()
        .last()
        .copied()
        .ok_or(infererror::emptylogits)?;
    if vocab == 0 {
        return Err(infererror::emptylogits);
    }
    let data = logits.getdata();
    let start = data
        .len()
        .checked_sub(vocab)
        .ok_or(infererror::degeneratelogits)?;
    Ok(data[start..].to_vec())
}
