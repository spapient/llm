use thiserror::Error;

/// Failures produced while validating a requested resource budget against a
/// [`limits`](crate::limits) configuration.
///
/// Each variant carries the requested value and the configured limit so that a
/// caller can report the exact rejection without re-deriving it.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum limiterror {
    /// The requested element count exceeds the configured maximum.
    #[error("requested element count {requested} exceeds limit {limit}")]
    maxelementsexceeded {
        /// The requested element count.
        requested: usize,
        /// The configured maximum element count.
        limit: usize,
    },

    /// The requested byte size exceeds the configured maximum.
    #[error("requested byte size {requested} exceeds limit {limit}")]
    maxbytesexceeded {
        /// The requested byte size.
        requested: usize,
        /// The configured maximum byte size.
        limit: usize,
    },

    /// The requested sequence length exceeds the configured maximum.
    #[error("requested sequence length {requested} exceeds limit {limit}")]
    maxsequencelenexceeded {
        /// The requested sequence length.
        requested: usize,
        /// The configured maximum sequence length.
        limit: usize,
    },

    /// The requested batch size exceeds the configured maximum.
    #[error("requested batch size {requested} exceeds limit {limit}")]
    maxbatchexceeded {
        /// The requested batch size.
        requested: usize,
        /// The configured maximum batch size.
        limit: usize,
    },
}
