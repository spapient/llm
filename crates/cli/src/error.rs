use std::path::PathBuf;

use thiserror::Error;

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
/// carried transparently with `#[from]` so no diagnostic context is lost and no
/// underlying error type is collapsed into an application variant (ER2).
#[derive(Debug, Error)]
pub enum clierror {
    /// A filesystem operation failed at the named path.
    #[error("i/o error at {path}: {source}")]
    io {
        /// The path the failed operation targeted.
        path: PathBuf,
        /// The underlying filesystem error.
        source: std::io::Error,
    },

    /// A supplied path is not an existing regular file.
    #[error("invalid path {path}: {detail}")]
    invalidpath {
        /// The rejected path.
        path: PathBuf,
        /// Why the path was rejected.
        detail: &'static str,
    },

    /// A file exceeds the configured byte bound and was not read.
    #[error("file {path} is {observed} bytes, exceeding the {limit}-byte limit")]
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
    #[error("unrecognized artifact at {path}: not a config, checkpoint, or safetensors file")]
    unrecognizedartifact {
        /// The path whose bytes were not recognized.
        path: PathBuf,
    },

    /// An argument value is present but semantically invalid.
    #[error("invalid argument: {detail}")]
    invalidarguments {
        /// A description of the rejected value.
        detail: String,
    },

    /// A training-configuration document is not valid JSON for the schema.
    #[error("invalid training configuration at line {line}, column {column}: {message}")]
    invalidconfigjson {
        /// One-based line number of the failure.
        line: usize,
        /// One-based column number of the failure.
        column: usize,
        /// The parser's diagnostic message.
        message: String,
    },

    /// The requested operation is a declared, structural limitation.
    #[error("{operation} is not supported: {detail}")]
    notsupported {
        /// The operation that cannot be performed.
        operation: &'static str,
        /// Why the operation is unsupported.
        detail: &'static str,
    },

    /// An artifact parsing or validation failure owned by `format`.
    #[error(transparent)]
    format(#[from] format::formaterror),

    /// A checkpoint load or validation failure owned by `checkpoint`.
    #[error(transparent)]
    checkpoint(#[from] checkpoint::checkpointerror),

    /// A model configuration or forward-pass failure owned by `models`.
    #[error(transparent)]
    model(#[from] models::modelerror),

    /// A generation failure owned by `infer`.
    #[error(transparent)]
    infer(#[from] infer::infererror),

    /// A training configuration or orchestration failure owned by `train`.
    #[error(transparent)]
    train(#[from] train::trainerror),

    /// An execution-control failure owned by `context`.
    #[error(transparent)]
    context(#[from] context::contexterror),

    /// A tokenization failure owned by `tokenizer`.
    #[error(transparent)]
    tokenizer(#[from] tokenizer::tokenizererror),

    /// An optimizer configuration failure owned by `optim`.
    #[error(transparent)]
    optim(#[from] optim::opterror),
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
