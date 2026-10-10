use std::path::Path;

use tracing::debug;

use crate::error::checkpointerror;
use crate::manifests::checkpoint;

/// Saves a checkpoint to `path` atomically.
///
/// The checkpoint is encoded to its on-disk layout and then published through
/// [`writeatomic`](crate::writeatomic), so a reader never observes a partial
/// artifact and an existing checkpoint is replaced only after the new bytes are
/// fully written and synced.
///
/// # Errors
///
/// Returns the encoding failure from [`checkpoint::encode`], or
/// [`checkpointerror::io`] when the staging write or rename fails.
pub fn savecheckpoint(path: &Path, ckpt: &checkpoint) -> Result<(), checkpointerror> {
    let bytes = ckpt.encode()?;
    crate::atomicwrite::writeatomic(path, &bytes)?;
    debug!(path = %path.display(), bytes = bytes.len(), "checkpoint saved");
    Ok(())
}
