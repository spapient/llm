use std::path::Path;

use crate::error::checkpointerror;
use crate::manifests::checkpoint;

/// Loads a checkpoint from `path`.
///
/// The file is read into memory and then validated by [`checkpoint::decode`].
/// A recoverable filesystem failure is reported as [`checkpointerror::io`]; a
/// malformed or structurally invalid artifact is reported with the decode
/// error, so a caller can distinguish "the read failed" from "the bytes are not
/// a valid checkpoint".
///
/// # Errors
///
/// Returns [`checkpointerror::io`] when the file cannot be read, or the decode
/// error from [`checkpoint::decode`].
pub fn loadcheckpoint(path: &Path) -> Result<checkpoint, checkpointerror> {
    let bytes = std::fs::read(path).map_err(|source| checkpointerror::io {
        path: path.to_path_buf(),
        source,
    })?;
    let ckpt = checkpoint::decode(&bytes)?;

    Ok(ckpt)
}
