use std::path::PathBuf;

use thiserror::Error;

/// Failures produced while loading, saving, or validating a checkpoint.
///
/// The type separates the three categories a caller must act on differently:
///
/// * **Recoverable I/O failure** ([`Self::io`]) — the filesystem operation
///   failed; the operation may be retried or reported and no artifact was
///   published.
/// * **Structural incompatibility** (unsupported version, missing tensor,
///   dtype or shape mismatch) — the artifact is a valid checkpoint but does not
///   match the expected model or training state.
/// * **Malformed artifact** (bad magic, truncation, invalid manifest, an
///   out-of-range byte range) — the bytes are not a valid checkpoint.
///
/// Use [`Self::isrecoverable`] and [`Self::isincompatibility`] to classify a
/// value without matching on variants; the remaining cases are malformed
/// artifacts.
#[derive(Debug, Error)]
pub enum checkpointerror {
    /// A filesystem operation failed at the named path.
    #[error("i/o error at {path}: {source}")]
    io {
        /// The path the failed operation targeted.
        path: PathBuf,
        /// The underlying filesystem error.
        #[source]
        source: std::io::Error,
    },

    /// The artifact does not begin with the checkpoint magic bytes.
    #[error("not a checkpoint artifact: bad magic")]
    badmagic,

    /// The artifact declares a checkpoint version this build does not support.
    #[error("unsupported checkpoint version {found} (supported {supported})")]
    unsupportedversion {
        /// The version declared by the artifact.
        found: u32,
        /// The version this build supports.
        supported: u32,
    },

    /// The input ended before the declared content was available.
    #[error("input truncated: needed {needed} bytes but only {available} are available")]
    truncated {
        /// The number of bytes the parser required.
        needed: usize,
        /// The number of bytes actually supplied.
        available: usize,
    },

    /// The manifest section is not valid UTF-8 JSON.
    #[error("invalid manifest JSON at line {line}, column {column}: {message}")]
    invalidmanifestjson {
        /// One-based line number of the failure.
        line: usize,
        /// One-based column number of the failure.
        column: usize,
        /// The parser's diagnostic message.
        message: String,
    },

    /// The manifest is internally inconsistent.
    #[error("invalid manifest: {reason}")]
    invalidmanifest {
        /// Why the manifest was rejected.
        reason: String,
    },

    /// A tensor's element count does not fit in `usize`.
    #[error("element count overflows usize for tensor {name:?}")]
    shapeoverflow {
        /// The tensor name.
        name: String,
    },

    /// A tensor's declared byte length does not match its shape and dtype.
    #[error("tensor {name:?} byte length {actual} does not match expected {expected}")]
    lengthmismatch {
        /// The tensor name.
        name: String,
        /// The byte length implied by the shape and dtype.
        expected: u64,
        /// The byte length actually supplied.
        actual: u64,
    },

    /// The manifest declares a dtype this crate does not support.
    #[error("unsupported dtype {dtype:?} for tensor {name:?}")]
    dtypeunsupported {
        /// The tensor name.
        name: String,
        /// The declared dtype token.
        dtype: String,
    },

    /// Two tensors share the same name.
    #[error("duplicate tensor name {name:?}")]
    duplicatetensor {
        /// The duplicated tensor name.
        name: String,
    },

    /// A tensor's byte range is not contained in the data section.
    #[error("offset {offset} with length {nbytes} is out of range for data length {datalength}")]
    offsetoutofrange {
        /// The tensor's start offset.
        offset: u64,
        /// The tensor's byte length.
        nbytes: u64,
        /// The total length of the data section.
        datalength: u64,
    },

    /// Two tensors' byte ranges overlap.
    #[error("tensor byte ranges overlap")]
    overlappingtensors,

    /// The checkpoint is missing a tensor the caller requires.
    #[error("checkpoint is missing expected tensor {name:?}")]
    missingtensor {
        /// The missing tensor name.
        name: String,
    },

    /// A tensor's dtype does not match the expected dtype.
    #[error("tensor {name:?} dtype mismatch: expected {expected:?}, actual {actual:?}")]
    dtypemismatch {
        /// The tensor name.
        name: String,
        /// The expected dtype token.
        expected: String,
        /// The actual dtype token.
        actual: String,
    },

    /// A tensor's shape does not match the expected shape.
    #[error("tensor {name:?} shape mismatch: expected {expected:?}, actual {actual:?}")]
    shapemismatch {
        /// The tensor name.
        name: String,
        /// The expected dimensions.
        expected: Vec<usize>,
        /// The actual dimensions.
        actual: Vec<usize>,
    },

    /// An error owned by the `format` crate propagated unchanged.
    #[error(transparent)]
    format(#[from] format::formaterror),
}

impl checkpointerror {
    /// Returns `true` when the failure is a recoverable filesystem error.
    #[must_use]
    pub fn isrecoverable(&self) -> bool {
        matches!(self, Self::io { .. })
    }

    /// Returns `true` when the failure is a structural incompatibility with the
    /// expected model or training state.
    #[must_use]
    pub fn isincompatibility(&self) -> bool {
        matches!(
            self,
            Self::unsupportedversion { .. }
                | Self::missingtensor { .. }
                | Self::dtypemismatch { .. }
                | Self::shapemismatch { .. }
        )
    }
}
