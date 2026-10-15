mod decode;
mod generate;
mod inspect;
mod train;
mod validate;

use std::path::Path;

use crate::args::command;
use crate::clierror;
use crate::commandoutput;

/// The recognized on-disk artifact kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum artifactkind {
    /// A JSON model configuration (`config.json`).
    config,
    /// A checkpoint artifact (`LLMCKPT1`).
    checkpoint,
    /// A safetensors container.
    safetensors,
}

/// The checkpoint artifact magic, repeated here so classification does not read
/// an unvalidated checkpoint.
const CHECKPOINTMAGIC: &[u8] = b"LLMCKPT1";

/// Dispatches a parsed subcommand to its handler.
pub(crate) fn dispatch(command: command) -> Result<commandoutput, clierror> {
    match command {
        command::inspect(args) => inspect::run(&args),
        command::validate(args) => validate::run(&args),
        command::generate(args) => generate::run(&args),
        command::decode(args) => decode::run(&args),
        command::train(args) => train::run(&args),
    }
}

/// Classifies artifact bytes without trusting them.
///
/// The first non-whitespace byte selects JSON; the exact checkpoint magic
/// selects a checkpoint; anything else is treated as a safetensors container.
/// The bytes are still parsed and validated by the selected parser, so a wrong
/// guess produces a structured error rather than an acceptance.
pub(crate) fn classify(bytes: &[u8]) -> artifactkind {
    if firstnonwhitespace(bytes) == Some(b'{') {
        artifactkind::config
    } else if bytes.starts_with(CHECKPOINTMAGIC) {
        artifactkind::checkpoint
    } else {
        artifactkind::safetensors
    }
}

/// Returns the first non-ASCII-whitespace byte, if any.
fn firstnonwhitespace(bytes: &[u8]) -> Option<u8> {
    bytes
        .iter()
        .find(|byte| !byte.is_ascii_whitespace())
        .copied()
}

/// Reads `path` into memory after validating it is a regular file within
/// `limit` bytes, and re-checks the read length (IN1/IN2/IN3/IN4).
///
/// # Errors
///
/// Returns [`clierror::io`] when the metadata or read fails,
/// [`clierror::invalidpath`] when `path` is not a regular file, and
/// [`clierror::sizeexceeded`] when the file exceeds `limit`.
pub(crate) fn readartifact(path: &Path, limit: usize) -> Result<Vec<u8>, clierror> {
    let metadata = std::fs::metadata(path).map_err(|source| clierror::io {
        path: path.to_path_buf(),
        source,
    })?;
    if !metadata.is_file() {
        return Err(clierror::invalidpath {
            path: path.to_path_buf(),
            detail: "not a regular file",
        });
    }
    let declared = metadata.len();
    if declared > limit as u64 {
        return Err(clierror::sizeexceeded {
            path: path.to_path_buf(),
            observed: declared,
            limit,
        });
    }
    let bytes = std::fs::read(path).map_err(|source| clierror::io {
        path: path.to_path_buf(),
        source,
    })?;
    if bytes.len() > limit {
        return Err(clierror::sizeexceeded {
            path: path.to_path_buf(),
            observed: bytes.len() as u64,
            limit,
        });
    }
    Ok(bytes)
}
