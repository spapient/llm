use std::collections::HashSet;

use crate::blocks::{blockref, chunk};
use crate::crc32c::crc32c;
use crate::error::checkpointerror;
use crate::hash::{blake3hash, contenthash};
use crate::uformat::{
    FLAGDELTA, MAGIC, PREFIXLEN, canonicalmanifest, pushu32, pushu64, readblocks, readbytes,
    readstored, readtensors, readu32, snapshot, storedentry, tensorentry,
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

    /// Returns the snapshot version.
    #[must_use]
    pub fn getversion(&self) -> u32 {
        self.version
    }

    /// Returns the tensor entries in insertion order.
    #[must_use]
    pub fn gettensors(&self) -> &[tensorentry] {
        &self.tensors
    }

    /// Returns the full logical block list.
    #[must_use]
    pub fn getblocks(&self) -> &[blockref] {
        &self.blocks
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

    /// Decodes and validates a delta from its on-disk byte layout.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::badmagic`], [`checkpointerror::truncated`],
    /// [`checkpointerror::rootmismatch`], [`checkpointerror::missingblock`],
    /// [`checkpointerror::blockcrcmismatch`], [`checkpointerror::blockhashmismatch`],
    /// [`checkpointerror::truncatedblock`], or [`checkpointerror::blockoutofrange`].
    pub fn decode(bytes: &[u8]) -> Result<Self, checkpointerror> {
        if bytes.len() < MAGIC.len() {
            return Err(checkpointerror::truncated {
                needed: MAGIC.len(),
                available: bytes.len(),
            });
        }
        if bytes[..MAGIC.len()] != MAGIC {
            return Err(checkpointerror::badmagic);
        }
        if bytes.len() < PREFIXLEN {
            return Err(checkpointerror::truncated {
                needed: PREFIXLEN,
                available: bytes.len(),
            });
        }
        let mut lengthbytes = [0u8; 8];
        lengthbytes.copy_from_slice(&bytes[MAGIC.len()..PREFIXLEN]);
        let manifestlen = u64::from_le_bytes(lengthbytes);
        let manifestlen = usize::try_from(manifestlen).map_err(|_| checkpointerror::truncated {
            needed: usize::MAX,
            available: bytes.len(),
        })?;
        let manifestend = PREFIXLEN
            .checked_add(manifestlen)
            .ok_or(checkpointerror::truncated {
                needed: usize::MAX,
                available: bytes.len(),
            })?;
        if manifestend > bytes.len() {
            return Err(checkpointerror::truncated {
                needed: manifestend,
                available: bytes.len(),
            });
        }
        let manifest = &bytes[PREFIXLEN..manifestend];
        let data = &bytes[manifestend..];

        let mut pos = 0usize;
        let version = readu32(manifest, &mut pos)?;
        let tensors = readtensors(manifest, &mut pos)?;
        let blocks = readblocks(manifest, &mut pos)?;
        let canonical = &manifest[..pos];

        let flags = readu32(manifest, &mut pos)?;
        if flags != FLAGDELTA {
            return Err(checkpointerror::invalidmanifest {
                reason: format!("unsupported snapshot flags {flags}"),
            });
        }
        let basebytes = readbytes(manifest, &mut pos, 32)?;
        let mut base_id = [0u8; 32];
        base_id.copy_from_slice(basebytes);
        let rootbytes = readbytes(manifest, &mut pos, 32)?;
        let mut root_hash = [0u8; 32];
        root_hash.copy_from_slice(rootbytes);
        let expectedroot = blake3hash(canonical);
        if root_hash != expectedroot {
            return Err(checkpointerror::rootmismatch {
                expected: expectedroot,
                found: root_hash,
            });
        }

        let entries = readstored(manifest, &mut pos)?;
        let stored = verifystored(data, &blocks, &entries)?;

        Ok(Self {
            version,
            base_id,
            tensors,
            blocks,
            stored,
            root_hash,
        })
    }
}

/// Verifies the stored blocks against the data section and the logical block
/// list, returning the stored blocks with their bytes.
fn verifystored(
    data: &[u8],
    blocks: &[blockref],
    entries: &[storedentry],
) -> Result<Vec<storedblock>, checkpointerror> {
    let known: HashSet<[u8; 16]> = blocks.iter().map(|b| b.gethash()).collect();
    let mut stored = Vec::with_capacity(entries.len());
    for (position, entry) in entries.iter().enumerate() {
        if !known.contains(&entry.hash) {
            return Err(checkpointerror::missingblock { hash: entry.hash });
        }
        let offset =
            usize::try_from(entry.offset).map_err(|_| checkpointerror::blockoutofrange {
                position,
                blockcount: blocks.len(),
            })?;
        let size = entry.size as usize;
        let end = offset
            .checked_add(size)
            .ok_or(checkpointerror::blockoutofrange {
                position,
                blockcount: blocks.len(),
            })?;
        if end > data.len() {
            return Err(checkpointerror::truncatedblock {
                position,
                needed: size,
                available: data.len() - offset,
            });
        }
        let bytes = &data[offset..end];
        if crc32c(bytes) != entry.crc32c {
            return Err(checkpointerror::blockcrcmismatch { position });
        }
        let hash = contenthash(bytes);
        if hash != entry.hash {
            return Err(checkpointerror::blockhashmismatch {
                position,
                expected: entry.hash,
                found: hash,
            });
        }
        stored.push(storedblock {
            hash: entry.hash,
            bytes: bytes.to_vec(),
        });
    }
    Ok(stored)
}
