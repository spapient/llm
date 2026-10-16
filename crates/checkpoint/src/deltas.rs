use std::borrow::Cow;
use std::collections::HashSet;

use crate::blocks::{TRANSFORMNONE, blockref, chunk, compressblock, decompressblock};
use crate::crc32c::crc32c;
use crate::encryption;
use crate::error::checkpointerror;
use crate::hash::{blake3hash, contenthash};
use crate::signature;
use crate::uformat::{
    FLAGDELTA, FLAGENCRYPTED, FLAGSIGNED, FLAGTRANSFORMED, MAGIC, PREFIXLEN, canonicalmanifest,
    checkflags, parseartifact, pushu32, pushu64, snapshot, storedentry, tensorentry,
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
    signature: Option<[u8; 64]>,
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

        // Each block's dedup decision and byte copy are independent, so spread
        // them over available cores into disjoint slots of `stored`.
        let nthreads = match std::thread::available_parallelism() {
            Ok(n) => n.get().min(newblocks.len()),
            Err(_) => 1,
        };
        let mut cumoffsets = Vec::with_capacity(newblocks.len() + 1);
        let mut acc = 0usize;
        cumoffsets.push(acc);
        for block in &newblocks {
            acc += block.getlen() as usize;
            cumoffsets.push(acc);
        }
        let mut spans = Vec::with_capacity(nthreads);
        let mut start = 0usize;
        for t in 0..nthreads {
            let len = newblocks.len() / nthreads + usize::from(t < newblocks.len() % nthreads);
            let end = start + len;
            spans.push(start..end);
            start = end;
        }
        let mut stored: Vec<Option<storedblock>> = vec![None; newblocks.len()];
        let newblocksref: &[blockref] = &newblocks;
        let basehashesref: &HashSet<[u8; 16]> = &basehashes;
        let dataref: &[u8] = new.getdata();
        let cumoffsetsref: &[usize] = &cumoffsets;
        std::thread::scope(|scope| {
            let mut cursor: &mut [Option<storedblock>] = &mut stored;
            let mut handles = Vec::with_capacity(nthreads);
            for span in &spans {
                let (head, tail) = cursor.split_at_mut(span.len());
                cursor = tail;
                handles.push(scope.spawn(move || {
                    deltablockspan(newblocksref, basehashesref, dataref, cumoffsetsref, span.clone(), head);
                }));
            }
            for handle in handles {
                let _ = handle.join();
            }
        });
        let stored: Vec<storedblock> = stored.into_iter().flatten().collect();
        let canonical = canonicalmanifest(new.getversion(), new.gettensors(), &newblocks);
        let root_hash = blake3hash(&canonical);
        Self {
            version: new.getversion(),
            base_id: base.roothash(),
            tensors: new.gettensors().to_vec(),
            blocks: newblocks,
            stored,
            root_hash,
            signature: None,
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

    /// Returns the Ed25519 signature carried by the delta, when present.
    #[must_use]
    pub fn getsignature(&self) -> Option<[u8; 64]> {
        self.signature
    }

    /// Verifies the delta's Ed25519 signature over its root hash against a
    /// caller-supplied public key.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::signaturemissing`] when the delta carries no
    /// signature, or [`checkpointerror::signaturemismatch`] when the signature
    /// does not verify.
    pub fn verifysignature(&self, public: &[u8; 32]) -> Result<(), checkpointerror> {
        let signature = self.signature.ok_or(checkpointerror::signaturemissing)?;
        if signature::verify(public, &self.root_hash, &signature) {
            Ok(())
        } else {
            Err(checkpointerror::signaturemismatch)
        }
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

    /// Encodes the delta into its on-disk byte layout with no signature and no
    /// encryption.
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
        self.encode_with(None, None)
    }

    /// Encodes the delta signed with `seed`.
    ///
    /// The signature covers the delta's root hash, which commits to every block
    /// through the Merkle DAG.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::invalidsignaturekey`] for an all-zero seed,
    /// or [`checkpointerror::invalidmanifest`] if the total length overflows
    /// `usize`.
    pub fn encode_signed(&self, seed: &[u8; 32]) -> Result<Vec<u8>, checkpointerror> {
        self.encode_with(Some(seed), None)
    }

    /// Encodes the delta encrypted with `key`.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::invalidmanifest`] if the total length
    /// overflows `usize`.
    pub fn encode_encrypted(&self, key: &[u8; 32]) -> Result<Vec<u8>, checkpointerror> {
        self.encode_with(None, Some(key))
    }

    /// Encodes the delta signed with `seed` and encrypted with `key`.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::invalidsignaturekey`] for an all-zero seed,
    /// or [`checkpointerror::invalidmanifest`] if the total length overflows
    /// `usize`.
    pub fn encode_signed_encrypted(
        &self,
        seed: &[u8; 32],
        key: &[u8; 32],
    ) -> Result<Vec<u8>, checkpointerror> {
        self.encode_with(Some(seed), Some(key))
    }

    /// Encodes the delta, signing and encrypting it as requested.
    fn encode_with(
        &self,
        seed: Option<&[u8; 32]>,
        key: Option<&[u8; 32]>,
    ) -> Result<Vec<u8>, checkpointerror> {
        let canonical = canonicalmanifest(self.version, &self.tensors, &self.blocks);
        let root = blake3hash(&canonical);
        let mut flags = FLAGDELTA;
        let signature = match seed {
            Some(seed) => {
                flags |= FLAGSIGNED;
                Some(signature::sign(seed, &root)?)
            }
            None => None,
        };
        if key.is_some() {
            flags |= FLAGENCRYPTED;
        }

        // Phase 1: compress every stored block before the manifest header is
        // written, so the snapshot-level transform flag can be set when any
        // block uses the bit-shuffle pre-pass.
        let mut storedblocks = Vec::with_capacity(self.stored.len());
        let mut has_transform = false;
        for block in &self.stored {
            let (codec, transform, compressed) = compressblock(&block.bytes);
            if transform != TRANSFORMNONE {
                has_transform = true;
            }
            storedblocks.push((codec, transform, compressed));
        }
        if has_transform {
            flags |= FLAGTRANSFORMED;
        }

        let mut manifest = canonical;
        pushu32(&mut manifest, flags);
        manifest.extend_from_slice(&self.base_id);
        manifest.extend_from_slice(&root);
        if let Some(sig) = signature {
            manifest.extend_from_slice(&sig);
        }
        pushu64(&mut manifest, self.stored.len() as u64);
        let mut storeddata = Vec::new();
        for (block, (codec, transform, compressed)) in self.stored.iter().zip(storedblocks) {
            let (stored, tag) = match key {
                Some(key) => {
                    let nonce = encryption::nonceof(&block.hash);
                    encryption::encryptblock(key, &nonce, &compressed)?
                }
                None => (compressed, [0u8; 16]),
            };
            let checksum = crc32c(&stored);
            manifest.extend_from_slice(&block.hash);
            pushu64(&mut manifest, storeddata.len() as u64);
            pushu32(&mut manifest, stored.len() as u32);
            pushu32(&mut manifest, checksum);
            manifest.push(codec);
            if flags & FLAGTRANSFORMED != 0 {
                manifest.push(transform);
            }
            if key.is_some() {
                manifest.extend_from_slice(&tag);
            }
            storeddata.extend_from_slice(&stored);
        }

        let total = PREFIXLEN
            .checked_add(manifest.len())
            .and_then(|v| v.checked_add(storeddata.len()))
            .ok_or_else(|| checkpointerror::invalidmanifest {
                reason: "delta size overflows usize".to_string(),
            })?;
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&(manifest.len() as u64).to_le_bytes());
        out.extend_from_slice(&manifest);
        out.extend_from_slice(&storeddata);
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
        Self::decode_with(bytes, None)
    }

    /// Decodes and validates an encrypted delta, decrypting its stored blocks
    /// with `key`.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::missingkey`] when the delta is encrypted but
    /// no key is provided, or the decode failures of [`Self::decode`].
    pub fn decode_encrypted(bytes: &[u8], key: &[u8; 32]) -> Result<Self, checkpointerror> {
        Self::decode_with(bytes, Some(key))
    }

    fn decode_with(bytes: &[u8], key: Option<&[u8; 32]>) -> Result<Self, checkpointerror> {
        let parsed = parseartifact(bytes)?;
        checkflags(parsed.flags, FLAGDELTA)?;
        let encrypted = parsed.flags & FLAGENCRYPTED != 0;
        let key = match (encrypted, key) {
            (true, Some(key)) => Some(key),
            (true, None) => return Err(checkpointerror::missingkey),
            (false, _) => None,
        };
        let data = &bytes[parsed.dataoffset..];
        let stored = verifystored(data, &parsed.blocks, &parsed.stored, key)?;
        Ok(Self {
            version: parsed.version,
            base_id: parsed.base_id,
            tensors: parsed.tensors,
            blocks: parsed.blocks,
            stored,
            root_hash: parsed.root_hash,
            signature: parsed.signature,
        })
    }
}

/// Fills one contiguous span of a delta's stored blocks: each block whose hash
/// is absent from `basehashes` gets its bytes copied into the matching slot of
/// `out`.
fn deltablockspan(
    newblocks: &[blockref],
    basehashes: &HashSet<[u8; 16]>,
    data: &[u8],
    cumoffsets: &[usize],
    span: std::ops::Range<usize>,
    out: &mut [Option<storedblock>],
) {
    for (i, position) in span.enumerate() {
        let block = &newblocks[position];
        if !basehashes.contains(&block.gethash()) {
            let start = cumoffsets[position];
            let len = block.getlen() as usize;
            out[i] = Some(storedblock {
                hash: block.gethash(),
                bytes: data[start..start + len].to_vec(),
            });
        }
    }
}

/// Verifies the stored blocks against the data section and the logical block
/// list, returning the stored blocks with their bytes.
fn verifystored(
    data: &[u8],
    blocks: &[blockref],
    entries: &[storedentry],
    key: Option<&[u8; 32]>,
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
                available: data.len().saturating_sub(offset),
            });
        }
        let bytes = &data[offset..end];
        if crc32c(bytes) != entry.crc32c {
            return Err(checkpointerror::blockcrcmismatch { position });
        }
        let plain = match key {
            Some(key) => {
                let nonce = encryption::nonceof(&entry.hash);
                Cow::Owned(encryption::decryptblock(key, &nonce, bytes, &entry.tag)?)
            }
            None => Cow::Borrowed(bytes),
        };
        let raw = decompressblock(entry.codec, entry.transform, &plain)?;
        let hash = contenthash(&raw);
        if hash != entry.hash {
            return Err(checkpointerror::blockhashmismatch {
                position,
                expected: entry.hash,
                found: hash,
            });
        }
        stored.push(storedblock {
            hash: entry.hash,
            bytes: raw.into_owned(),
        });
    }
    Ok(stored)
}
