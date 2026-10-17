use std::path::PathBuf;

/// Exit code for a command that completed successfully.
pub const SUCCESSEXITCODE: u8 = 0;

/// Exit code for an operational failure: I/O, an invalid path, an oversized or
/// unrecognized artifact, or a failure propagated from a workflow crate.
pub const OPERATIONALEXITCODE: u8 = 1;

/// Exit code for a usage error: a parse diagnostic or an invalid argument
/// value.
pub const USAGEEXITCODE: u8 = 2;

/// Exit code for a structurally invalid target artifact.
pub const VALIDATIONEXITCODE: u8 = 3;

/// Exit code for a declared, structural limitation (for example the missing
/// backward path needed to train a Llama end to end).
pub const UNSUPPORTEDEXITCODE: u8 = 4;

/// Failures produced by the command-line application.
///
/// The adapter owns the failures it can describe itself (an unreadable path, a
/// file larger than the configured bound, bytes that match no known artifact, an
/// invalid argument value, a malformed training-configuration document, and the
/// declared unsupported limitation). Failures owned by a workflow crate are
/// carried transparently with `` so no diagnostic context is lost and no
/// underlying error type is collapsed into an application variant (ER2).
#[derive(Debug)]
pub enum clierror {
    /// A filesystem operation failed at the named path.
    io {
        /// The path the failed operation targeted.
        path: PathBuf,
        /// The underlying filesystem error.
        source: std::io::Error,
    },

    /// A supplied path is not an existing regular file.
    invalidpath {
        /// The rejected path.
        path: PathBuf,
        /// Why the path was rejected.
        detail: &'static str,
    },

    /// A file exceeds the configured byte bound and was not read.
    sizeexceeded {
        /// The oversized path.
        path: PathBuf,
        /// The observed size in bytes.
        observed: u64,
        /// The configured inclusive byte bound.
        limit: usize,
    },

    /// The bytes match no supported artifact: not JSON, not a checkpoint, and
    /// not a structurally valid safetensors header.
    unrecognizedartifact {
        /// The path whose bytes were not recognized.
        path: PathBuf,
    },

    /// An argument value is present but semantically invalid.
    invalidarguments {
        /// A description of the rejected value.
        detail: String,
    },

    /// A training-configuration document is not valid JSON for the schema.
    invalidconfigjson {
        /// One-based line number of the failure.
        line: usize,
        /// One-based column number of the failure.
        column: usize,
        /// The parser's diagnostic message.
        message: String,
    },

    /// The requested operation is a declared, structural limitation.
    notsupported {
        /// The operation that cannot be performed.
        operation: &'static str,
        /// Why the operation is unsupported.
        detail: &'static str,
    },

    /// An artifact parsing or validation failure owned by `format`.
    format(format::formaterror),

    /// A checkpoint load or validation failure owned by `checkpoint`.
    checkpoint(checkpoint::checkpointerror),

    /// A model configuration or forward-pass failure owned by `models`.
    model(models::modelerror),

    /// A generation failure owned by `infer`.
    infer(infer::infererror),

    /// A training configuration or orchestration failure owned by `train`.
    train(train::trainerror),

    /// An execution-control failure owned by `context`.
    context(context::contexterror),

    /// A tokenization failure owned by `tokenizer`.
    tokenizer(tokenizer::tokenizererror),

    /// An optimizer configuration failure owned by `optim`.
    optim(optim::opterror),
}

/// Maps a `clierror` to its process exit code.
///
/// This is the single mapping from the crate error to an exit code (the
/// doc comment on the crate root states the table). The usage code is the named
/// constant [`USAGEEXITCODE`] so the binary and the tests share one value.
#[must_use]
pub fn exitcode(error: &clierror) -> u8 {
    match error {
        clierror::invalidarguments { .. } | clierror::invalidconfigjson { .. } => USAGEEXITCODE,
        clierror::format(_) | clierror::checkpoint(_) => VALIDATIONEXITCODE,
        clierror::notsupported { .. } => UNSUPPORTEDEXITCODE,
        _ => OPERATIONALEXITCODE,
    }
}

impl std::fmt::Display for clierror {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::io { path, source } => {
                write!(f, "i/o error at {}: {source}", path.display())
            }
            Self::invalidpath { path, detail } => {
                write!(f, "invalid path {}: {detail}", path.display())
            }
            Self::sizeexceeded {
                path,
                observed,
                limit,
            } => write!(
                f,
                "file {} is {observed} bytes, exceeding the {limit}-byte limit",
                path.display()
            ),
            Self::unrecognizedartifact { path } => write!(
                f,
                "unrecognized artifact at {}: not a config, checkpoint, or safetensors file",
                path.display()
            ),
            Self::invalidarguments { detail } => write!(f, "invalid argument: {detail}"),
            Self::invalidconfigjson {
                line,
                column,
                message,
            } => write!(
                f,
                "invalid training configuration at line {line}, column {column}: {message}"
            ),
            Self::notsupported { operation, detail } => {
                write!(f, "{operation} is not supported: {detail}")
            }
            Self::format(f0) => std::fmt::Display::fmt(f0, f),
            Self::checkpoint(f0) => std::fmt::Display::fmt(f0, f),
            Self::model(f0) => std::fmt::Display::fmt(f0, f),
            Self::infer(f0) => std::fmt::Display::fmt(f0, f),
            Self::train(f0) => std::fmt::Display::fmt(f0, f),
            Self::context(f0) => std::fmt::Display::fmt(f0, f),
            Self::tokenizer(f0) => std::fmt::Display::fmt(f0, f),
            Self::optim(f0) => std::fmt::Display::fmt(f0, f),
        }
    }
}

impl std::error::Error for clierror {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::format(f0) => Some(f0),
            Self::checkpoint(f0) => Some(f0),
            Self::model(f0) => Some(f0),
            Self::infer(f0) => Some(f0),
            Self::train(f0) => Some(f0),
            Self::context(f0) => Some(f0),
            Self::tokenizer(f0) => Some(f0),
            Self::optim(f0) => Some(f0),
            _ => None,
        }
    }
}

impl From<format::formaterror> for clierror {
    fn from(v: format::formaterror) -> Self {
        Self::format(v)
    }
}

impl From<checkpoint::checkpointerror> for clierror {
    fn from(v: checkpoint::checkpointerror) -> Self {
        Self::checkpoint(v)
    }
}

impl From<models::modelerror> for clierror {
    fn from(v: models::modelerror) -> Self {
        Self::model(v)
    }
}

impl From<infer::infererror> for clierror {
    fn from(v: infer::infererror) -> Self {
        Self::infer(v)
    }
}

impl From<train::trainerror> for clierror {
    fn from(v: train::trainerror) -> Self {
        Self::train(v)
    }
}

impl From<context::contexterror> for clierror {
    fn from(v: context::contexterror) -> Self {
        Self::context(v)
    }
}

impl From<tokenizer::tokenizererror> for clierror {
    fn from(v: tokenizer::tokenizererror) -> Self {
        Self::tokenizer(v)
    }
}

impl From<optim::opterror> for clierror {
    fn from(v: optim::opterror) -> Self {
        Self::optim(v)
    }
}
