use format::{dtypetoken, tokendtype};
use tensor::{dtype, shape};

use crate::blocks::{blockref, chunk, compressblock, decompressblock};
use crate::crc32c::crc32c;
use crate::error::checkpointerror;
use crate::hash::{blake3hash, contenthash};

/// Magic identifying a `.u` snapshot artifact.
pub(crate) const MAGIC: [u8; 8] = *b"UWFSNAP1";

/// The number of bytes in the fixed prefix: magic plus manifest length.
pub(crate) const PREFIXLEN: usize = 16;

/// Flags value for a full (self-contained) snapshot.
pub(crate) const FLAGSFULL: u32 = 2;

/// Flags value for an incremental (delta) snapshot.
pub(crate) const FLAGDELTA: u32 = 1;

/// One entry of the physical block table: where a stored block lives.
pub(crate) struct storedentry {
    pub(crate) hash: [u8; 16],
    pub(crate) offset: u64,
    pub(crate) size: u32,
    pub(crate) crc32c: u32,
    pub(crate) codec: u8,
}

/// Describes one tensor persisted in a `.u` snapshot.
///
/// A value records the tensor's name, storage [`dtype`], logical [`shape`], and
/// the half-open byte range `[offset, offset + nbytes)` it occupies within the
/// snapshot's logical data section.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct tensorentry {
    name: String,
    dtype: dtype,
    shape: shape,
    offset: u64,
    nbytes: u64,
}

impl tensorentry {
    /// Returns the tensor name.
    #[must_use]
    pub fn getname(&self) -> &str {
        &self.name
    }

    /// Returns the storage dtype.
    #[must_use]
    pub fn getdtype(&self) -> dtype {
        self.dtype
    }

    /// Returns the logical shape.
    #[must_use]
    pub fn getshape(&self) -> &shape {
        &self.shape
    }

    /// Returns the start offset within the logical data section.
    #[must_use]
    pub fn getoffset(&self) -> u64 {
        self.offset
    }

    /// Returns the tensor's byte length.
    #[must_use]
    pub fn getnbytes(&self) -> u64 {
        self.nbytes
    }

    /// Returns the exclusive end offset within the logical data section.
    #[must_use]
    pub fn getend(&self) -> u64 {
        self.offset + self.nbytes
    }
}

/// An in-memory block-addressed `.u` snapshot: tensors plus their concatenated
/// bytes, persisted with per-block CRC32C and BLAKE3 verification.
///
/// The on-disk layout is magic, a little-endian `u64` manifest length, a binary
/// manifest (tensors, block list, flags, base identity, root hash, physical
/// block table), and the data section.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct snapshot {
    version: u32,
    tensors: Vec<tensorentry>,
    data: Vec<u8>,
}

impl snapshot {
    /// Creates an empty snapshot with the given version.
    #[must_use]
    pub fn new(version: u32) -> Self {
        Self {
            version,
            tensors: Vec::new(),
            data: Vec::new(),
        }
    }

    /// Constructs a snapshot from its parts (used by the merge path).
    pub(crate) fn fromparts(version: u32, tensors: Vec<tensorentry>, data: Vec<u8>) -> Self {
        Self {
            version,
            tensors,
            data,
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

    /// Returns the flat logical data section.
    #[must_use]
    pub fn getdata(&self) -> &[u8] {
        &self.data
    }

    /// Returns the snapshot's root hash: the BLAKE3-256 content identity of its
    /// tensors and block list, independent of storage form (full or delta).
    #[must_use]
    pub fn roothash(&self) -> [u8; 32] {
        let blocks = chunk(&self.data);
        blake3hash(&canonicalmanifest(self.version, &self.tensors, &blocks))
    }

    /// Appends a tensor's bytes and records its metadata.
    ///
    /// The byte length must equal the shape's element count multiplied by the
    /// dtype width; the offset is the current end of the data section, so the
    /// ranges are contiguous and non-overlapping by construction.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::shapeoverflow`], [`checkpointerror::lengthmismatch`],
    /// or [`checkpointerror::duplicatetensor`]. The data section is unchanged on
    /// any error.
    pub fn addtensor(
        &mut self,
        name: String,
        value: dtype,
        dims: shape,
        bytes: &[u8],
    ) -> Result<(), checkpointerror> {
        let count = dims
            .getelementcount()
            .map_err(|_| checkpointerror::shapeoverflow { name: name.clone() })?;
        let expected = (count as u64)
            .checked_mul(value.sizeinbytes() as u64)
            .ok_or_else(|| checkpointerror::shapeoverflow { name: name.clone() })?;
        let actual = bytes.len() as u64;
        if actual != expected {
            return Err(checkpointerror::lengthmismatch {
                name,
                expected,
                actual,
            });
        }
        if self.tensors.iter().any(|t| t.name == name) {
            return Err(checkpointerror::duplicatetensor { name });
        }
        let offset = self.data.len() as u64;
        let entry = tensorentry {
            name,
            dtype: value,
            shape: dims,
            offset,
            nbytes: actual,
        };
        self.tensors.push(entry);
        self.data.extend_from_slice(bytes);
        Ok(())
    }

    /// Copies a tensor's bytes out of the data section.
    ///
    /// Returns `None` when the tensor is absent or its range does not fit the
    /// stored data.
    #[must_use]
    pub fn readtensor(&self, name: &str) -> Option<Vec<u8>> {
        let entry = self.tensors.iter().find(|t| t.name == name)?;
        let start = usize::try_from(entry.offset).ok()?;
        let end = usize::try_from(entry.getend()).ok()?;
        self.data.get(start..end).map(<[u8]>::to_vec)
    }

    /// Encodes the snapshot as a full (self-contained) artifact.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::invalidmanifest`] if the total length
    /// overflows `usize`.
    pub fn encode(&self) -> Result<Vec<u8>, checkpointerror> {
        let blocks = chunk(&self.data);
        let canonical = canonicalmanifest(self.version, &self.tensors, &blocks);
        let root = blake3hash(&canonical);

        let mut manifest = canonical;
        pushu32(&mut manifest, FLAGSFULL);
        manifest.extend_from_slice(&[0u8; 32]);
        manifest.extend_from_slice(&root);
        pushu64(&mut manifest, blocks.len() as u64);
        let mut storeddata = Vec::new();
        let mut offset = 0usize;
        for block in &blocks {
            let len = block.getlen() as usize;
            let (codec, stored) = compressblock(&self.data[offset..offset + len]);
            let checksum = crc32c(&stored);
            manifest.extend_from_slice(&block.gethash());
            pushu64(&mut manifest, storeddata.len() as u64);
            pushu32(&mut manifest, stored.len() as u32);
            pushu32(&mut manifest, checksum);
            manifest.push(codec);
            storeddata.extend_from_slice(&stored);
            offset += len;
        }

        let total = PREFIXLEN
            .checked_add(manifest.len())
            .and_then(|v| v.checked_add(storeddata.len()))
            .ok_or_else(|| checkpointerror::invalidmanifest {
                reason: "snapshot size overflows usize".to_string(),
            })?;
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&(manifest.len() as u64).to_le_bytes());
        out.extend_from_slice(&manifest);
        out.extend_from_slice(&storeddata);
        Ok(out)
    }

    /// Decodes and validates a full snapshot from its on-disk byte layout.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::badmagic`], [`checkpointerror::truncated`],
    /// [`checkpointerror::rootmismatch`], [`checkpointerror::blockcrcmismatch`],
    /// [`checkpointerror::blockhashmismatch`], [`checkpointerror::truncatedblock`],
    /// [`checkpointerror::blockoutofrange`], or the tensor validation failures.
    pub fn decode(bytes: &[u8]) -> Result<Self, checkpointerror> {
        let parsed = parseartifact(bytes)?;
        if parsed.flags != FLAGSFULL {
            return Err(checkpointerror::invalidmanifest {
                reason: format!("unsupported snapshot flags {}", parsed.flags),
            });
        }
        if parsed.base_id.iter().any(|&b| b != 0) {
            return Err(checkpointerror::invalidmanifest {
                reason: "full snapshot has a non-zero base identity".to_string(),
            });
        }
        let data = &bytes[parsed.dataoffset..];
        let rawdata = verifyfull(data, &parsed.blocks, &parsed.stored)?;
        verifytensors(&rawdata, &parsed.tensors)?;
        Ok(Self {
            version: parsed.version,
            tensors: parsed.tensors,
            data: rawdata,
        })
    }
}

/// The parsed manifest of a `.u` artifact: everything except the data section.
pub(crate) struct parsedmanifest {
    pub(crate) version: u32,
    pub(crate) tensors: Vec<tensorentry>,
    pub(crate) blocks: Vec<blockref>,
    pub(crate) flags: u32,
    pub(crate) base_id: [u8; 32],
    pub(crate) root_hash: [u8; 32],
    pub(crate) stored: Vec<storedentry>,
    pub(crate) dataoffset: usize,
}

/// Parses an artifact's manifest — magic, manifest length, tensors, block list,
/// flags, base identity, root hash, and physical block table — without loading
/// the data section, so a reader can access blocks on demand.
pub(crate) fn parseartifact(bytes: &[u8]) -> Result<parsedmanifest, checkpointerror> {
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

    let mut pos = 0usize;
    let version = readu32(manifest, &mut pos)?;
    let tensors = readtensors(manifest, &mut pos)?;
    let blocks = readblocks(manifest, &mut pos)?;
    let canonical = &manifest[..pos];
    let flags = readu32(manifest, &mut pos)?;
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
    let stored = readstored(manifest, &mut pos)?;
    Ok(parsedmanifest {
        version,
        tensors,
        blocks,
        flags,
        base_id,
        root_hash,
        stored,
        dataoffset: manifestend,
    })
}

/// Builds the canonical (hashed) manifest bytes: version, tensors, and the
/// ordered block list. Flags, base identity, and the physical block table are
/// storage metadata and are excluded so a full snapshot and its delta share a
/// root hash.
pub(crate) fn canonicalmanifest(
    version: u32,
    tensors: &[tensorentry],
    blocks: &[blockref],
) -> Vec<u8> {
    let mut buf = Vec::new();
    pushu32(&mut buf, version);
    pushu64(&mut buf, tensors.len() as u64);
    for entry in tensors {
        pushstring(&mut buf, &entry.name);
        pushstring(&mut buf, dtypetoken(entry.dtype));
        let dims = entry.shape.getdims();
        pushu64(&mut buf, dims.len() as u64);
        for &dim in dims {
            pushu64(&mut buf, dim as u64);
        }
        pushu64(&mut buf, entry.offset);
        pushu64(&mut buf, entry.nbytes);
    }
    pushu64(&mut buf, blocks.len() as u64);
    for block in blocks {
        buf.extend_from_slice(&block.gethash());
        pushu32(&mut buf, block.getlen());
    }
    buf
}

/// Parses and validates the tensor entries.
pub(crate) fn readtensors(
    manifest: &[u8],
    pos: &mut usize,
) -> Result<Vec<tensorentry>, checkpointerror> {
    let count = readcount(manifest, pos)?;
    let mut tensors = Vec::with_capacity(count);
    for _ in 0..count {
        let name = readstring(manifest, pos)?;
        let token = readstring(manifest, pos)?;
        let value = tokendtype(&token).ok_or_else(|| checkpointerror::dtypeunsupported {
            name: name.clone(),
            dtype: token.clone(),
        })?;
        let dimcount = readcount(manifest, pos)?;
        let mut dims = Vec::with_capacity(dimcount);
        for _ in 0..dimcount {
            let dim = readu64(manifest, pos)?;
            dims.push(
                usize::try_from(dim).map_err(|_| checkpointerror::invalidmanifest {
                    reason: format!("dimension {dim} overflows usize"),
                })?,
            );
        }
        let dims = shape::new(dims).map_err(|_| checkpointerror::invalidmanifest {
            reason: format!("tensor {name:?} has an invalid shape"),
        })?;
        let offset = readu64(manifest, pos)?;
        let nbytes = readu64(manifest, pos)?;
        let count = dims
            .getelementcount()
            .map_err(|_| checkpointerror::shapeoverflow { name: name.clone() })?;
        let expected = (count as u64)
            .checked_mul(value.sizeinbytes() as u64)
            .ok_or_else(|| checkpointerror::shapeoverflow { name: name.clone() })?;
        if nbytes != expected {
            return Err(checkpointerror::lengthmismatch {
                name,
                expected,
                actual: nbytes,
            });
        }
        tensors.push(tensorentry {
            name,
            dtype: value,
            shape: dims,
            offset,
            nbytes,
        });
    }
    Ok(tensors)
}

/// Parses the ordered block list.
pub(crate) fn readblocks(
    manifest: &[u8],
    pos: &mut usize,
) -> Result<Vec<blockref>, checkpointerror> {
    let count = readcount(manifest, pos)?;
    let mut blocks = Vec::with_capacity(count);
    for _ in 0..count {
        let hashbytes = readbytes(manifest, pos, 16)?;
        let mut hash = [0u8; 16];
        hash.copy_from_slice(hashbytes);
        let len = readu32(manifest, pos)?;
        blocks.push(blockref::new(hash, len));
    }
    Ok(blocks)
}

/// Parses the physical block table.
pub(crate) fn readstored(
    manifest: &[u8],
    pos: &mut usize,
) -> Result<Vec<storedentry>, checkpointerror> {
    let count = readcount(manifest, pos)?;
    let mut stored = Vec::with_capacity(count);
    for _ in 0..count {
        let hashbytes = readbytes(manifest, pos, 16)?;
        let mut hash = [0u8; 16];
        hash.copy_from_slice(hashbytes);
        let offset = readu64(manifest, pos)?;
        let size = readu32(manifest, pos)?;
        let crc32c = readu32(manifest, pos)?;
        let codec = readbytes(manifest, pos, 1)?[0];
        stored.push(storedentry {
            hash,
            offset,
            size,
            crc32c,
            codec,
        });
    }
    Ok(stored)
}

/// Verifies a full snapshot: every stored entry matches its logical block and
/// the data section passes CRC32C and content-identity checks.
fn verifyfull(
    data: &[u8],
    blocks: &[blockref],
    stored: &[storedentry],
) -> Result<Vec<u8>, checkpointerror> {
    if stored.len() != blocks.len() {
        return Err(checkpointerror::invalidmanifest {
            reason: "stored block count does not match the block list".to_string(),
        });
    }
    let mut rawdata = Vec::new();
    let mut offset = 0usize;
    for (position, block) in blocks.iter().enumerate() {
        let entry = &stored[position];
        if entry.hash != block.gethash() || entry.offset != offset as u64 {
            return Err(checkpointerror::invalidmanifest {
                reason: format!("stored entry {position} is inconsistent with the block list"),
            });
        }
        let end =
            offset
                .checked_add(entry.size as usize)
                .ok_or(checkpointerror::blockoutofrange {
                    position,
                    blockcount: blocks.len(),
                })?;
        if end > data.len() {
            return Err(checkpointerror::truncatedblock {
                position,
                needed: entry.size as usize,
                available: data.len().saturating_sub(offset),
            });
        }
        let bytes = &data[offset..end];
        if crc32c(bytes) != entry.crc32c {
            return Err(checkpointerror::blockcrcmismatch { position });
        }
        let raw = decompressblock(entry.codec, bytes)?;
        if raw.len() != block.getlen() as usize {
            return Err(checkpointerror::invalidmanifest {
                reason: format!("block {position} decompressed length mismatch"),
            });
        }
        let hash = contenthash(&raw);
        if hash != block.gethash() {
            return Err(checkpointerror::blockhashmismatch {
                position,
                expected: block.gethash(),
                found: hash,
            });
        }
        rawdata.extend_from_slice(&raw);
        offset = end;
    }
    if offset != data.len() {
        return Err(checkpointerror::blockoutofrange {
            position: offset,
            blockcount: data.len(),
        });
    }
    Ok(rawdata)
}

/// Verifies every tensor's byte range lies within the data section without
/// overlapping another tensor.
fn verifytensors(data: &[u8], tensors: &[tensorentry]) -> Result<(), checkpointerror> {
    let datalength = data.len() as u64;
    let mut spans: Vec<(u64, u64)> = tensors
        .iter()
        .filter(|e| e.getend() > e.getoffset())
        .map(|e| (e.getoffset(), e.getend()))
        .collect();
    spans.sort_unstable_by_key(|span| span.0);
    let mut maxend = 0u64;
    for (begin, end) in spans {
        if begin < maxend {
            return Err(checkpointerror::overlappingtensors);
        }
        if end > datalength {
            return Err(checkpointerror::offsetoutofrange {
                offset: begin,
                nbytes: end - begin,
                datalength,
            });
        }
        maxend = maxend.max(end);
    }
    Ok(())
}

/// Reads a count, rejecting values that do not fit `usize`.
fn readcount(bytes: &[u8], pos: &mut usize) -> Result<usize, checkpointerror> {
    let count = readu64(bytes, pos)?;
    usize::try_from(count).map_err(|_| checkpointerror::invalidmanifest {
        reason: format!("count {count} overflows usize"),
    })
}

/// Reads a length-prefixed UTF-8 string.
fn readstring(bytes: &[u8], pos: &mut usize) -> Result<String, checkpointerror> {
    let len = readcount(bytes, pos)?;
    let slice = readbytes(bytes, pos, len)?;
    String::from_utf8(slice.to_vec()).map_err(|_| checkpointerror::invalidmanifest {
        reason: "string is not valid UTF-8".to_string(),
    })
}

pub(crate) fn pushu32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

pub(crate) fn pushu64(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn pushstring(buf: &mut Vec<u8>, value: &str) {
    pushu64(buf, value.len() as u64);
    buf.extend_from_slice(value.as_bytes());
}

pub(crate) fn readu32(bytes: &[u8], pos: &mut usize) -> Result<u32, checkpointerror> {
    let slice = readbytes(bytes, pos, 4)?;
    let mut arr = [0u8; 4];
    arr.copy_from_slice(slice);
    Ok(u32::from_le_bytes(arr))
}

pub(crate) fn readu64(bytes: &[u8], pos: &mut usize) -> Result<u64, checkpointerror> {
    let slice = readbytes(bytes, pos, 8)?;
    let mut arr = [0u8; 8];
    arr.copy_from_slice(slice);
    Ok(u64::from_le_bytes(arr))
}

pub(crate) fn readbytes<'a>(
    bytes: &'a [u8],
    pos: &mut usize,
    len: usize,
) -> Result<&'a [u8], checkpointerror> {
    let end = pos.checked_add(len).ok_or(checkpointerror::truncated {
        needed: usize::MAX,
        available: bytes.len(),
    })?;
    if end > bytes.len() {
        return Err(checkpointerror::truncated {
            needed: end,
            available: bytes.len(),
        });
    }
    let slice = &bytes[*pos..end];
    *pos = end;
    Ok(slice)
}
