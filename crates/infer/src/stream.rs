use crate::stopping::stopreason;

/// A sink for tokens as they are generated.
///
/// The generation engine calls [`Self::ongenerated`] once per emitted token, in
/// order, and [`Self::onstop`] exactly once when generation ends normally. A
/// failed or cancelled generation does not call [`Self::onstop`]; the error is
/// returned from the engine instead, so a stream never silently terminates in a
/// way that implies success (`docs/engineering.md`).
pub trait streamsink {
    /// Receives generated token `token` at zero-based position `index`.
    ///
    /// Returns `true` to continue generation or `false` to request a stop.
    fn ongenerated(&mut self, index: usize, token: usize) -> bool;

    /// Receives the terminal reason of a normally completed generation.
    fn onstop(&mut self, reason: stopreason);
}

/// A sink that records every generated token and the terminal reason.
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub struct collectingsink {
    tokens: Vec<usize>,
    reason: Option<stopreason>,
}

impl collectingsink {
    /// Creates an empty collecting sink.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the collected tokens in generation order.
    #[must_use]
    pub fn gettokens(&self) -> &[usize] {
        &self.tokens
    }

    /// Returns the terminal reason, or `None` when generation did not stop
    /// normally.
    #[must_use]
    pub fn getreason(&self) -> Option<stopreason> {
        self.reason
    }
}

impl streamsink for collectingsink {
    fn ongenerated(&mut self, _index: usize, token: usize) -> bool {
        self.tokens.push(token);
        true
    }

    fn onstop(&mut self, reason: stopreason) {
        self.reason = Some(reason);
    }
}
