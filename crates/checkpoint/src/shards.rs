//! Sharded `.u` snapshot persistence: a manifest prefix plus fixed-size data
//! shards.
//!
//! Sharding splits only the data section of an encoded snapshot; the manifest
//! prefix (magic, manifest length, manifest) stays whole, and every block's
//! `block_offset` remains an absolute offset within the logical data section.
//! The shards are contiguous slices of that data section, so joining them in
//! order reconstructs the original artifact byte-for-byte. This keeps the
//! sharding layer orthogonal to compression, encryption, and signing.

use crate::error::checkpointerror;
use crate::uformat::parseartifact;

/// A sharded snapshot: the manifest prefix plus the data section split into
/// fixed-size shards.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct sharded {
    manifest: Vec<u8>,
    shards: Vec<Vec<u8>>,
}

impl sharded {
    /// Returns the manifest prefix (magic, manifest length, manifest).
    #[must_use]
    pub fn getmanifest(&self) -> &[u8] {
        &self.manifest
    }

    /// Returns the data shards, in order.
    #[must_use]
    pub fn getshards(&self) -> &[Vec<u8>] {
        &self.shards
    }
}

/// Splits an encoded snapshot into a manifest prefix and `shardsize`-bounded
/// data shards.
///
/// # Errors
///
/// Returns [`checkpointerror::invalidmanifest`] for a zero shard size, or the
/// parse failures of the artifact.
pub fn split(bytes: &[u8], shardsize: usize) -> Result<sharded, checkpointerror> {
    if shardsize == 0 {
        return Err(checkpointerror::invalidmanifest {
            reason: "shard size must be non-zero".to_string(),
        });
    }
    let parsed = parseartifact(bytes)?;
    let manifest = bytes[..parsed.dataoffset].to_vec();
    let shards = bytes[parsed.dataoffset..]
        .chunks(shardsize)
        .map(<[u8]>::to_vec)
        .collect();
    Ok(sharded { manifest, shards })
}

/// Reassembles an encoded snapshot from a manifest prefix and its data shards.
///
/// The shards are concatenated in order onto the manifest prefix; the result is
/// the original artifact, so any reader or decoder can consume it unchanged.
#[must_use]
pub fn join(manifest: &[u8], shards: &[Vec<u8>]) -> Vec<u8> {
    let total = manifest.len() + shards.iter().map(Vec::len).sum::<usize>();
    let mut bytes = Vec::with_capacity(total);
    bytes.extend_from_slice(manifest);
    for shard in shards {
        bytes.extend_from_slice(shard);
    }
    bytes
}
