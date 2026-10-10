/// A reproducible random seed.
///
/// The wrapped value is passed to a deterministic generator owner; `lib`
/// only carries it without interpreting its distribution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct seed(u64);

impl seed {
    /// Creates a seed from its raw value.
    #[must_use]
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    /// Returns the raw value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}
