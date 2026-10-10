/// Identifies a token within a tokenizer vocabulary.
///
/// The wrapped value is an opaque index into a vocabulary; it carries no
/// interpretation beyond identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct tokenid(u32);

impl tokenid {
    /// Creates a token identifier from its raw index.
    #[must_use]
    pub const fn new(raw: u32) -> Self {
        Self(raw)
    }

    /// Returns the raw index.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}
