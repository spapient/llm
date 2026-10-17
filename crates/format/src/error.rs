/// Failures produced while parsing or structurally validating an external
/// model artifact.
///
/// Every variant describes a rejection owned by `format`: malformed or
/// truncated bytes, an unsupported dtype, an inconsistent element count or byte
/// range, overlapping tensors, or an unsupported model configuration. The type
/// carries the values needed to report the exact rejection so a caller can
/// react without re-deriving them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum formaterror {
    /// The input ended before the required number of bytes was available.
    truncated {
        /// The number of bytes the parser required.
        needed: usize,
        /// The number of bytes actually supplied.
        available: usize,
    },

    /// The declared safetensors header length exceeds the available input.
    headertoolarge {
        /// The header length declared by the artifact.
        declared: u64,
        /// The number of bytes available after the length prefix.
        available: usize,
    },

    /// The header or configuration is not valid UTF-8 JSON.
    invalidjson {
        /// One-based line number of the failure.
        line: usize,
        /// One-based column number of the failure.
        column: usize,
        /// The parser's diagnostic message.
        message: String,
    },

    /// A tensor entry does not match the expected safetensors schema.
    invalidentry {
        /// The tensor name whose entry was rejected.
        name: String,
        /// Why the entry was rejected.
        reason: String,
    },

    /// The artifact declares a dtype that this crate does not support.
    unsupporteddtype {
        /// The tensor name.
        name: String,
        /// The declared dtype token.
        dtype: String,
    },

    /// A tensor declares an empty (rank-zero) shape, which is unsupported.
    emptyshape {
        /// The tensor name.
        name: String,
    },

    /// The product of a tensor's dimensions does not fit in `usize`.
    shapeoverflow {
        /// The tensor name.
        name: String,
    },

    /// The tensor's element count exceeds the configured element budget.
    elementlimitexceeded {
        /// The tensor name.
        name: String,
        /// The requested element count.
        requested: usize,
        /// The configured maximum element count.
        limit: usize,
    },

    /// A tensor's byte range is not contained in the data section.
    offsetsoutofrange {
        /// The tensor name.
        name: String,
        /// The inclusive start offset.
        begin: u64,
        /// The exclusive end offset.
        end: u64,
        /// The total length of the data section.
        datalength: u64,
    },

    /// A tensor's byte range does not match its element count and dtype width.
    bytelengthmismatch {
        /// The tensor name.
        name: String,
        /// The byte length implied by the shape and dtype.
        expected: u64,
        /// The byte length declared by the offsets.
        actual: u64,
    },

    /// Two tensor byte ranges overlap.
    overlappingtensors {
        /// The tensor that starts first.
        first: String,
        /// The tensor whose range overlaps the first.
        second: String,
    },

    /// The reserved `__metadata__` section is not a string map.
    invalidmetadata {
        /// Why the metadata section was rejected.
        reason: String,
    },

    /// A model configuration is missing a required field.
    missingfield {
        /// The name of the missing field.
        field: &'static str,
    },

    /// A model configuration field holds an invalid value.
    invalidvalue {
        /// The name of the offending field.
        field: &'static str,
        /// Why the value was rejected.
        reason: String,
    },

    /// The model type is not supported by this workspace.
    unsupportedmodeltype {
        /// The declared model type.
        found: String,
    },

    /// The declared torch dtype is not supported.
    unsupportedtorchdtype {
        /// The declared torch dtype.
        found: String,
    },
}

impl std::fmt::Display for formaterror {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::truncated { needed, available } => write!(
                f,
                "input truncated: needed {needed} bytes but only {available} are available"
            ),
            Self::headertoolarge {
                declared,
                available,
            } => write!(
                f,
                "declared header length {declared} exceeds available {available}"
            ),
            Self::invalidjson {
                line,
                column,
                message,
            } => write!(f, "invalid JSON at line {line}, column {column}: {message}"),
            Self::invalidentry { name, reason } => {
                write!(f, "invalid safetensors entry {name:?}: {reason}")
            }
            Self::unsupporteddtype { name, dtype } => {
                write!(f, "unsupported dtype {dtype:?} for tensor {name:?}")
            }
            Self::emptyshape { name } => write!(f, "tensor {name:?} has an empty shape"),
            Self::shapeoverflow { name } => {
                write!(f, "element count overflows usize for tensor {name:?}")
            }
            Self::elementlimitexceeded {
                name,
                requested,
                limit,
            } => write!(
                f,
                "requested element count {requested} exceeds limit {limit} for tensor {name:?}"
            ),
            Self::offsetsoutofrange {
                name,
                begin,
                end,
                datalength,
            } => write!(
                f,
                "offsets [{begin}, {end}) are out of range for data length {datalength} on tensor {name:?}"
            ),
            Self::bytelengthmismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "tensor {name:?} byte length {actual} does not match expected {expected}"
            ),
            Self::overlappingtensors { first, second } => {
                write!(f, "tensors {first:?} and {second:?} overlap")
            }
            Self::invalidmetadata { reason } => write!(f, "invalid __metadata__ section: {reason}"),
            Self::missingfield { field } => {
                write!(f, "model configuration is missing required field {field:?}")
            }
            Self::invalidvalue { field, reason } => write!(
                f,
                "model configuration field {field:?} is invalid: {reason}"
            ),
            Self::unsupportedmodeltype { found } => write!(f, "unsupported model type {found:?}"),
            Self::unsupportedtorchdtype { found } => write!(f, "unsupported torch dtype {found:?}"),
        }
    }
}

impl std::error::Error for formaterror {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        None
    }
}
