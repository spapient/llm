use lib::tokenid;

/// A single dataset record: an ordered sequence of token identifiers.
///
/// A record owns its token buffer. It carries no position, label, or request
/// state; downstream code supplies any such metadata. The buffer length is the
/// record length and may be zero.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct record {
    tokens: Vec<tokenid>,
}

impl record {
    /// Creates a record that owns `tokens`.
    #[must_use]
    pub fn new(tokens: Vec<tokenid>) -> Self {
        Self { tokens }
    }

    /// Returns the record's tokens in order.
    #[must_use]
    pub fn gettokens(&self) -> &[tokenid] {
        &self.tokens
    }

    /// Returns the number of tokens in the record.
    #[must_use]
    pub fn getlen(&self) -> usize {
        self.tokens.len()
    }

    /// Returns `true` when the record contains no tokens.
    #[must_use]
    pub fn isempty(&self) -> bool {
        self.tokens.is_empty()
    }
}
