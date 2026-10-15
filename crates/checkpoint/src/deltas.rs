use std::collections::HashSet;

use crate::blocks::{blockref, chunk};
use crate::crc32c::crc32c;
use crate::error::checkpointerror;
use crate::hash::blake3hash;
use crate::uformat::{
    FLAGDELTA, MAGIC, PREFIXLEN, canonicalmanifest, pushu32, pushu64, snapshot, tensorentry,
};

/// One block physically stored in a delta artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct storedblock {
    hash: [u8; 16],
    bytes: Vec<u8>,
}

impl storedblock {
    /// Returns the block's 128-bit content identity.
    #[must_use]
    pub fn gethash(&self) -> [u8; 16] {
        self.hash
    }

    /// Returns the block's uncompressed bytes.
    #[must_use]
    pub fn getbytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// An incremental (delta) snapshot: the full logical manifest of a new state
/// plus only the blocks that differ from a base snapshot.
///
/// Unchanged blocks are referenced by content identity from the base, so the
/// delta stores only the new bytes. Its root hash equals the new snapshot's
/// root hash, and its `base_id` references the base snapshot's root hash.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct delta {
    version: u32,
    base_id: [u8; 32],
    tensors: Vec<tensorentry>,
    blocks: Vec<blockref>,
    stored: Vec<storedblock>,
    root_hash: [u8; 32],
}

impl delta {
    /// Builds the delta from `base` to `new`.
    ///
    /// A block is stored only when its content identity is absent from the
    /// base; otherwise the base's copy is referenced. Deduplication is by
    /// content hash, so an insertion that shifts offsets produces no false
    /// changes.
    #[must_use]
    pub fn fromsnapshots(base: &snapshot, new: &snapshot) -> Self {
        let newblocks = chunk(new.getdata());
        let basehashes: HashSet<[u8; 16]> =
            chunk(base.getdata()).iter().map(|b| b.gethash()).collect();
        let mut stored = Vec::new();
        let mut offset = 0usize;
        for block in &newblocks {
            let len = block.getlen() as usize;
            if !basehashes.contains(&block.gethash()) {
                stored.push(storedblock {
                    hash: block.gethash(),
                    bytes: new.getdata()[offset..offset + len].to_vec(),
                });
            }
            offset += len;
        }
        let canonical = canonicalmanifest(new.getversion(), new.gettensors(), &newblocks);
        let root_hash = blake3hash(&canonical);
        Self {
            version: new.getversion(),
            base_id: base.roothash(),
            tensors: new.gettensors().to_vec(),
            blocks: newblocks,
            stored,
            root_hash,
        }
    }

    /// Returns the base snapshot's root hash.
    #[must_use]
    pub fn getbaseid(&self) -> [u8; 32] {
        self.base_id
    }

    /// Returns the blocks physically stored in this delta.
    #[must_use]
    pub fn getstored(&self) -> &[storedblock] {
        &self.stored
    }

    /// Returns the delta's root hash, equal to the new snapshot's root hash.
    #[must_use]
    pub fn getroothash(&self) -> [u8; 32] {
        self.root_hash
    }

    /// Encodes the delta into its on-disk byte layout.
    ///
    /// The layout is magic, a little-endian `u64` manifest length, a binary
    /// manifest (tensors, block list, delta flag, base identity, root hash,
    /// physical block table), and the stored blocks.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::invalidmanifest`] if the total length
    /// overflows `usize`.
    pub fn encode(&self) -> Result<Vec<u8>, checkpointerror> {
        let canonical = canonicalmanifest(self.version, &self.tensors, &self.blocks);
        let root = blake3hash(&canonical);

        let mut manifest = canonical;
        pushu32(&mut manifest, FLAGDELTA);
        manifest.extend_from_slice(&self.base_id);
        manifest.extend_from_slice(&root);
        pushu64(&mut manifest, self.stored.len() as u64);
        let mut offset = 0u64;
        for block in &self.stored {
            let checksum = crc32c(&block.bytes);
            manifest.extend_from_slice(&block.hash);
            pushu64(&mut manifest, offset);
            // block.bytes.len() <= BLOCKSIZE < u32::MAX, so the cast is safe.
            pushu32(&mut manifest, block.bytes.len() as u32);
            pushu32(&mut manifest, checksum);
            offset += block.bytes.len() as u64;
        }

        let mut datalen = 0usize;
        for block in &self.stored {
            datalen += block.bytes.len();
        }
        let total = PREFIXLEN
            .checked_add(manifest.len())
            .and_then(|v| v.checked_add(datalen))
            .ok_or_else(|| checkpointerror::invalidmanifest {
                reason: "delta size overflows usize".to_string(),
            })?;
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&(manifest.len() as u64).to_le_bytes());
        out.extend_from_slice(&manifest);
        for block in &self.stored {
            out.extend_from_slice(&block.bytes);
        }
        Ok(out)
    }
}
