use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::error::checkpointerror;

/// Maximum number of distinct staging-file names tried before giving up.
///
/// A staging name is `.{file}.tmp.{pid}.{attempt}`. The process id already
/// separates concurrent processes; the bounded attempt counter resolves the
/// rare case where a stale staging file from a crashed run still holds a name.
/// Rationale for the bound: 16 attempts make an accidental collision
/// vanishingly unlikely without looping without limit.
const MAXTEMPATTEMPTS: u32 = 16;

/// Writes `bytes` to `path` atomically via a staging file and rename.
///
/// The bytes are written to a unique staging file in the same directory, flushed
/// and synced, and then renamed onto `path`. A reader therefore observes either
/// the previous `path` contents or the complete new contents, never a partial
/// write. If any step fails, the staging file is removed (best effort) and the
/// error is returned; the existing `path` is left untouched.
///
/// # Guarantees And Limits
///
/// A same-directory rename is atomic on the filesystems this workspace targets
/// (POSIX `rename`). Durability of the rename itself requires a directory
/// `fsync`, which is not portable across the supported platforms and is not
/// performed; the guarantee is atomic replacement, not crash durability of the
/// directory entry.
///
/// # Errors
///
/// Returns [`checkpointerror::io`] when the staging file cannot be created, the
/// bytes cannot be written or synced, or the rename fails.
pub fn writeatomic(path: &Path, bytes: &[u8]) -> Result<(), checkpointerror> {
    let dir = parentdir(path);
    let (staging, mut file) = stagingfrompath(path, &dir)?;
    let written = file.write_all(bytes).and_then(|()| file.sync_all());
    drop(file);
    if let Err(source) = written {
        cleanup(&staging);
        return Err(checkpointerror::io {
            path: staging,
            source,
        });
    }
    if let Err(source) = std::fs::rename(&staging, path) {
        cleanup(&staging);
        return Err(checkpointerror::io {
            path: path.to_path_buf(),
            source,
        });
    }
    Ok(())
}

/// Returns the directory that contains `path`, defaulting to `"."`.
fn parentdir(path: &Path) -> PathBuf {
    match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent.to_path_buf(),
        _ => PathBuf::from("."),
    }
}

/// Creates a uniquely named staging file in `dir`, returning it and its handle.
fn stagingfrompath(path: &Path, dir: &Path) -> Result<(PathBuf, std::fs::File), checkpointerror> {
    let name = match path.file_name() {
        Some(name) => name.to_string_lossy().into_owned(),
        None => {
            return Err(checkpointerror::io {
                path: path.to_path_buf(),
                source: std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "target path has no file name",
                ),
            });
        }
    };
    for attempt in 0..MAXTEMPATTEMPTS {
        let candidate = dir.join(format!(".{name}.tmp.{}.{attempt}", std::process::id()));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => return Ok((candidate, file)),
            Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(source) => {
                return Err(checkpointerror::io {
                    path: candidate,
                    source,
                });
            }
        }
    }
    Err(checkpointerror::io {
        path: path.to_path_buf(),
        source: std::io::Error::new(
            std::io::ErrorKind::AlreadyExists,
            "could not create a unique staging file",
        ),
    })
}

/// Removes a staging file, ignoring failure.
///
/// Cleanup failure is intentionally ignored: the primary error is already being
/// reported, and a leftover staging file never shadows the committed artifact
/// because staging names are never target names.
fn cleanup(staging: &Path) {
    if std::fs::remove_file(staging).is_err() {
        // Best-effort cleanup; nothing actionable on failure.
    }
}
