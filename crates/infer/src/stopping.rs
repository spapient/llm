use crate::error::infererror;

/// Why generation stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum stopreason {
    /// A configured end-of-sequence token was selected.
    endoftoken,
    /// The configured maximum number of new tokens was reached.
    maxlength,
    /// The output sink requested that generation stop.
    stopped,
}

/// The termination criteria of a generation request.
///
/// A stop condition names the end-of-sequence tokens that end generation early
/// and the maximum number of new tokens to emit. Both are explicit upper bounds;
/// `maxnewtokens` of zero is rejected because a request that can emit nothing is
/// a wiring error rather than a valid limit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct stopcondition {
    eostokens: Vec<usize>,
    maxnewtokens: usize,
}

impl stopcondition {
    /// Builds a stop condition from end-of-sequence tokens and a new-token cap.
    ///
    /// # Errors
    ///
    /// Returns [`infererror::invalidrequest`] when `maxnewtokens` is zero.
    pub fn new(eostokens: Vec<usize>, maxnewtokens: usize) -> Result<Self, infererror> {
        if maxnewtokens == 0 {
            return Err(infererror::invalidrequest {
                field: "maxnewtokens",
                detail: "must be non-zero".to_string(),
            });
        }
        Ok(Self {
            eostokens,
            maxnewtokens,
        })
    }

    /// Returns the end-of-sequence tokens.
    #[must_use]
    pub fn geteostokens(&self) -> &[usize] {
        &self.eostokens
    }

    /// Returns the maximum number of new tokens.
    #[must_use]
    pub const fn getmaxnewtokens(&self) -> usize {
        self.maxnewtokens
    }

    /// Reports whether `token` is a configured end-of-sequence token.
    #[must_use]
    pub fn iseos(&self, token: usize) -> bool {
        self.eostokens.contains(&token)
    }
}
