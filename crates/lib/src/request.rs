/// Identifies an inference or training request.
///
/// The wrapped value is an opaque, monotonically comparable identifier assigned
/// by the request owner; it carries no interpretation beyond identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct requestid(u64);

impl requestid {
    /// Creates a request identifier from its raw value.
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
