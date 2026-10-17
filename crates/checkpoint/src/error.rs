use std::path::PathBuf;

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
#[derive(Debug)]
pub enum checkpointerror {
    /// A filesystem operation failed at the named path.
    io {
        /// The path the failed operation targeted.
        path: PathBuf,
        /// The underlying filesystem error.
        source: std::io::Error,
    },

    /// The artifact does not begin with the checkpoint magic bytes.
    badmagic,

    /// The artifact declares a checkpoint version this build does not support.
    unsupportedversion {
        /// The version declared by the artifact.
        found: u32,
        /// The version this build supports.
        supported: u32,
    },

    /// The input ended before the declared content was available.
    truncated {
        /// The number of bytes the parser required.
        needed: usize,
        /// The number of bytes actually supplied.
        available: usize,
    },

    /// The manifest section is not valid UTF-8 JSON.
    invalidmanifestjson {
        /// One-based line number of the failure.
        line: usize,
        /// One-based column number of the failure.
        column: usize,
        /// The parser's diagnostic message.
        message: String,
    },

    /// The manifest is internally inconsistent.
    invalidmanifest {
        /// Why the manifest was rejected.
        reason: String,
    },

    /// A tensor's element count does not fit in `usize`.
    shapeoverflow {
        /// The tensor name.
        name: String,
    },

    /// A tensor's declared byte length does not match its shape and dtype.
    lengthmismatch {
        /// The tensor name.
        name: String,
        /// The byte length implied by the shape and dtype.
        expected: u64,
        /// The byte length actually supplied.
        actual: u64,
    },

    /// The manifest declares a dtype this crate does not support.
    dtypeunsupported {
        /// The tensor name.
        name: String,
        /// The declared dtype token.
        dtype: String,
    },

    /// Two tensors share the same name.
    duplicatetensor {
        /// The duplicated tensor name.
        name: String,
    },

    /// A tensor's byte range is not contained in the data section.
    offsetoutofrange {
        /// The tensor's start offset.
        offset: u64,
        /// The tensor's byte length.
        nbytes: u64,
        /// The total length of the data section.
        datalength: u64,
    },

    /// Two tensors' byte ranges overlap.
    overlappingtensors,

    /// The checkpoint is missing a tensor the caller requires.
    missingtensor {
        /// The missing tensor name.
        name: String,
    },

    /// A tensor's dtype does not match the expected dtype.
    dtypemismatch {
        /// The tensor name.
        name: String,
        /// The expected dtype token.
        expected: String,
        /// The actual dtype token.
        actual: String,
    },

    /// A tensor's shape does not match the expected shape.
    shapemismatch {
        /// The tensor name.
        name: String,
        /// The expected dimensions.
        expected: Vec<usize>,
        /// The actual dimensions.
        actual: Vec<usize>,
    },

    /// The snapshot's root hash does not match its declared value.
    rootmismatch {
        /// The root hash computed over the canonical manifest.
        expected: [u8; 32],
        /// The root hash the artifact declared.
        found: [u8; 32],
    },

    /// A block's CRC32C does not match its declared checksum.
    blockcrcmismatch {
        /// The zero-based block index.
        position: usize,
    },

    /// A block's BLAKE3-128 content identity does not match its declaration.
    blockhashmismatch {
        /// The zero-based block index.
        position: usize,
        /// The expected content identity.
        expected: [u8; 16],
        /// The content identity computed over the stored bytes.
        found: [u8; 16],
    },

    /// A block's bytes are shorter than its declared length.
    truncatedblock {
        /// The zero-based block index.
        position: usize,
        /// The block's declared byte length.
        needed: usize,
        /// The bytes actually remaining.
        available: usize,
    },

    /// A raw tensor section's CRC32C does not match its declared checksum.
    tensorcrcmismatch {
        /// The tensor name.
        name: String,
    },

    /// A block's declared byte range is inconsistent with the data section.
    blockoutofrange {
        /// The zero-based block index.
        position: usize,
        /// The total number of blocks declared.
        blockcount: usize,
    },

    /// A block's content identity resolves neither to the delta nor to the base.
    missingblock {
        /// The unresolved content identity.
        hash: [u8; 16],
    },

    /// The base snapshot a delta references is absent.
    missingbase {
        /// The base snapshot's root hash.
        base_id: [u8; 32],
    },

    /// A base snapshot's root hash does not match the delta's reference.
    basehashmismatch {
        /// The root hash the delta references.
        expected: [u8; 32],
        /// The root hash of the base that was found.
        found: [u8; 32],
    },

    /// The Ed25519 signing seed is invalid (all zero).
    invalidsignaturekey,

    /// A snapshot carries no Ed25519 signature but verification was requested.
    signaturemissing,

    /// The Ed25519 signature over the snapshot's root hash does not verify.
    signaturemismatch,

    /// The AES-256-GCM authentication tag did not verify (wrong key or tampered
    /// ciphertext).
    decryptionfailed,

    /// An encrypted snapshot was opened without the decryption key.
    missingkey,

    /// An error owned by the `format` crate propagated unchanged.
    format(format::formaterror),

    /// A tensor decoding failure propagated from the `tensor` crate unchanged.
    ///
    /// The underlying [`tensor::tensorerror`] records which validation failed
    /// (element count, byte length, or alignment), so the caller does not lose
    /// the precise cause.
    tensor(tensor::tensorerror),
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
                | Self::missingbase { .. }
                | Self::basehashmismatch { .. }
                | Self::signaturemissing
                | Self::missingkey
        )
    }

    /// Returns `true` when the failure is a corruption of the artifact's
    /// integrity — a checksum or hash mismatch, a truncated block, or an
    /// inconsistent block table — as opposed to a structural incompatibility or
    /// a recoverable I/O error. Corruption is never auto-retried.
    #[must_use]
    pub fn iscorruption(&self) -> bool {
        matches!(
            self,
            Self::rootmismatch { .. }
                | Self::blockcrcmismatch { .. }
                | Self::blockhashmismatch { .. }
                | Self::truncatedblock { .. }
                | Self::blockoutofrange { .. }
                | Self::missingblock { .. }
                | Self::tensorcrcmismatch { .. }
                | Self::signaturemismatch
                | Self::decryptionfailed
        )
    }
}

impl std::fmt::Display for checkpointerror {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::io { path, source } => write!(f, "i/o error at {}: {source}", path.display()),
            Self::badmagic => f.write_str("not a checkpoint artifact: bad magic"),
            Self::unsupportedversion { found, supported } => write!(
                f,
                "unsupported checkpoint version {found} (supported {supported})"
            ),
            Self::truncated { needed, available } => write!(
                f,
                "input truncated: needed {needed} bytes but only {available} are available"
            ),
            Self::invalidmanifestjson {
                line,
                column,
                message,
            } => write!(
                f,
                "invalid manifest JSON at line {line}, column {column}: {message}"
            ),
            Self::invalidmanifest { reason } => write!(f, "invalid manifest: {reason}"),
            Self::shapeoverflow { name } => {
                write!(f, "element count overflows usize for tensor {name:?}")
            }
            Self::lengthmismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "tensor {name:?} byte length {actual} does not match expected {expected}"
            ),
            Self::dtypeunsupported { name, dtype } => {
                write!(f, "unsupported dtype {dtype:?} for tensor {name:?}")
            }
            Self::duplicatetensor { name } => write!(f, "duplicate tensor name {name:?}"),
            Self::offsetoutofrange {
                offset,
                nbytes,
                datalength,
            } => write!(
                f,
                "offset {offset} with length {nbytes} is out of range for data length {datalength}"
            ),
            Self::overlappingtensors => f.write_str("tensor byte ranges overlap"),
            Self::missingtensor { name } => {
                write!(f, "checkpoint is missing expected tensor {name:?}")
            }
            Self::dtypemismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "tensor {name:?} dtype mismatch: expected {expected:?}, actual {actual:?}"
            ),
            Self::shapemismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "tensor {name:?} shape mismatch: expected {expected:?}, actual {actual:?}"
            ),
            Self::rootmismatch { expected, found } => write!(
                f,
                "root hash mismatch: expected {expected:?}, found {found:?}"
            ),
            Self::blockcrcmismatch { position } => write!(f, "block {position} crc32c mismatch"),
            Self::blockhashmismatch { position, .. } => {
                write!(f, "block {position} content hash mismatch")
            }
            Self::truncatedblock {
                position,
                needed,
                available,
            } => write!(
                f,
                "block {position} truncated: needed {needed} bytes, {available} available"
            ),
            Self::tensorcrcmismatch { name } => write!(f, "tensor {name:?} crc32c mismatch"),
            Self::blockoutofrange {
                position,
                blockcount,
            } => write!(f, "block {position} out of range for {blockcount} blocks"),
            Self::missingblock { hash } => write!(
                f,
                "block {hash:?} is missing from both the delta and the base"
            ),
            Self::missingbase { base_id } => write!(f, "base snapshot {base_id:?} is missing"),
            Self::basehashmismatch { expected, found } => write!(
                f,
                "base root hash mismatch: expected {expected:?}, found {found:?}"
            ),
            Self::invalidsignaturekey => f.write_str("invalid ed25519 signing seed (all zero)"),
            Self::signaturemissing => f.write_str("snapshot has no ed25519 signature"),
            Self::signaturemismatch => {
                f.write_str("ed25519 signature does not match the snapshot's root hash")
            }
            Self::decryptionfailed => {
                f.write_str("aes-gcm decryption failed: authentication tag mismatch")
            }
            Self::missingkey => f.write_str("encrypted snapshot requires a decryption key"),
            Self::format(f0) => std::fmt::Display::fmt(f0, f),
            Self::tensor(f0) => std::fmt::Display::fmt(f0, f),
        }
    }
}

impl std::error::Error for checkpointerror {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::io { source, .. } => Some(source),
            Self::format(f0) => Some(f0),
            Self::tensor(f0) => Some(f0),
            _ => None,
        }
    }
}

impl From<format::formaterror> for checkpointerror {
    fn from(v: format::formaterror) -> Self {
        Self::format(v)
    }
}

impl From<tensor::tensorerror> for checkpointerror {
    fn from(v: tensor::tensorerror) -> Self {
        Self::tensor(v)
    }
}
