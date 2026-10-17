/// Failures produced while validating a requested resource budget against a
/// [`limits`](crate::limits) configuration.
///
/// Each variant carries the requested value and the configured limit so that a
/// caller can report the exact rejection without re-deriving it.
#[derive(Debug, PartialEq, Eq)]
pub enum limiterror {
    /// The requested element count exceeds the configured maximum.
    maxelementsexceeded {
        /// The requested element count.
        requested: usize,
        /// The configured maximum element count.
        limit: usize,
    },

    /// The requested byte size exceeds the configured maximum.
    maxbytesexceeded {
        /// The requested byte size.
        requested: usize,
        /// The configured maximum byte size.
        limit: usize,
    },

    /// The requested sequence length exceeds the configured maximum.
    maxsequencelenexceeded {
        /// The requested sequence length.
        requested: usize,
        /// The configured maximum sequence length.
        limit: usize,
    },

    /// The requested batch size exceeds the configured maximum.
    maxbatchexceeded {
        /// The requested batch size.
        requested: usize,
        /// The configured maximum batch size.
        limit: usize,
    },
}

impl std::fmt::Display for limiterror {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::maxelementsexceeded { requested, limit } => write!(
                f,
                "requested element count {requested} exceeds limit {limit}"
            ),
            Self::maxbytesexceeded { requested, limit } => {
                write!(f, "requested byte size {requested} exceeds limit {limit}")
            }
            Self::maxsequencelenexceeded { requested, limit } => write!(
                f,
                "requested sequence length {requested} exceeds limit {limit}"
            ),
            Self::maxbatchexceeded { requested, limit } => {
                write!(f, "requested batch size {requested} exceeds limit {limit}")
            }
        }
    }
}

impl std::error::Error for limiterror {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
