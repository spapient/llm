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
