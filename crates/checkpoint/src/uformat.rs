use format::{dtypetoken, tokendtype};
use tensor::{dtype, shape};

use crate::blocks::{blockref, chunk};
use crate::crc32c::crc32c;
use crate::error::checkpointerror;
use crate::hash::{blake3hash, contenthash};

/// Magic identifying a `.u` snapshot artifact.
const MAGIC: [u8; 8] = *b"UWFSNAP1";

/// The number of bytes in the fixed prefix: magic plus manifest length.
const PREFIXLEN: usize = 16;

/// Flags value for a full (self-contained) snapshot.
const FLAGSFULL: u32 = 2;

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
/// The data section is split into [`BLOCKSIZE`]-sized blocks; the on-disk
/// layout is magic, a little-endian `u64` manifest length, a binary manifest
/// (tensors, block list, root hash, block checksums), and the data section.
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

    /// Encodes the snapshot into its on-disk byte layout.
    ///
    /// The layout is the 8-byte magic, the little-endian `u64` manifest length,
    /// the binary manifest, and the data section. The manifest carries the
    /// tensors, the ordered block list, the root hash, and a CRC32C per block.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::invalidmanifest`] if the total length
    /// overflows `usize`.
    pub fn encode(&self) -> Result<Vec<u8>, checkpointerror> {
        let blocks = chunk(&self.data);
        let canonical = self.canonicalmanifest(&blocks);
        let root = blake3hash(&canonical);

        let mut manifest = canonical;
        manifest.extend_from_slice(&root);
        let mut offset = 0usize;
        for block in &blocks {
            let len = block.getlen() as usize;
            let checksum = crc32c(&self.data[offset..offset + len]);
            manifest.extend_from_slice(&checksum.to_le_bytes());
            offset += len;
        }

        let total = PREFIXLEN
            .checked_add(manifest.len())
            .and_then(|v| v.checked_add(self.data.len()))
            .ok_or_else(|| checkpointerror::invalidmanifest {
                reason: "snapshot size overflows usize".to_string(),
            })?;
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&(manifest.len() as u64).to_le_bytes());
        out.extend_from_slice(&manifest);
        out.extend_from_slice(&self.data);
        Ok(out)
    }

    /// Decodes and validates a snapshot from its on-disk byte layout.
    ///
    /// The magic, manifest length, root hash, every block's CRC32C and content
    /// identity, and every tensor's byte range are all validated with checked
    /// arithmetic before any bytes are trusted.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::badmagic`], [`checkpointerror::truncated`],
    /// [`checkpointerror::rootmismatch`], [`checkpointerror::blockcrcmismatch`],
    /// [`checkpointerror::blockhashmismatch`], [`checkpointerror::truncatedblock`],
    /// [`checkpointerror::blockoutofrange`], or the tensor validation failures.
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
        let flags = readu32(manifest, &mut pos)?;
        if flags != FLAGSFULL {
            return Err(checkpointerror::invalidmanifest {
                reason: format!("unsupported snapshot flags {flags}"),
            });
        }

        let tensors = readtensors(manifest, &mut pos)?;
        let blocks = readblocks(manifest, &mut pos)?;

        let canonical = &manifest[..pos];
        let rootbytes = readbytes(manifest, &mut pos, 32)?;
        let mut root = [0u8; 32];
        root.copy_from_slice(rootbytes);
        let expectedroot = blake3hash(canonical);
        if root != expectedroot {
            return Err(checkpointerror::rootmismatch {
                expected: expectedroot,
                found: root,
            });
        }

        let mut checksums = Vec::with_capacity(blocks.len());
        for _ in 0..blocks.len() {
            checksums.push(readu32(manifest, &mut pos)?);
        }

        verifyblocks(data, &blocks, &checksums)?;
        verifytensors(data, &tensors)?;

        Ok(Self {
            version,
            tensors,
            data: data.to_vec(),
        })
    }

    /// Builds the canonical (hashed) manifest bytes: everything in the manifest
    /// except the root hash and the block checksums.
    fn canonicalmanifest(&self, blocks: &[blockref]) -> Vec<u8> {
        let mut buf = Vec::new();
        pushu32(&mut buf, self.version);
        pushu32(&mut buf, FLAGSFULL);
        pushu64(&mut buf, self.tensors.len() as u64);
        for entry in &self.tensors {
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
}

/// Parses and validates the tensor entries.
fn readtensors(manifest: &[u8], pos: &mut usize) -> Result<Vec<tensorentry>, checkpointerror> {
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
fn readblocks(manifest: &[u8], pos: &mut usize) -> Result<Vec<blockref>, checkpointerror> {
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

/// Verifies every block's CRC32C and content identity against the data section.
fn verifyblocks(
    data: &[u8],
    blocks: &[blockref],
    checksums: &[u32],
) -> Result<(), checkpointerror> {
    let mut offset = 0usize;
    for (position, block) in blocks.iter().enumerate() {
        let len = block.getlen() as usize;
        let end = offset
            .checked_add(len)
            .ok_or(checkpointerror::blockoutofrange {
                position,
                blockcount: blocks.len(),
            })?;
        if end > data.len() {
            return Err(checkpointerror::truncatedblock {
                position,
                needed: len,
                available: data.len() - offset,
            });
        }
        let bytes = &data[offset..end];
        if crc32c(bytes) != checksums[position] {
            return Err(checkpointerror::blockcrcmismatch { position });
        }
        let hash = contenthash(bytes);
        if hash != block.gethash() {
            return Err(checkpointerror::blockhashmismatch {
                position,
                expected: block.gethash(),
                found: hash,
            });
        }
        offset = end;
    }
    if offset != data.len() {
        return Err(checkpointerror::blockoutofrange {
            position: offset,
            blockcount: data.len(),
        });
    }
    Ok(())
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

fn pushu32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn pushu64(buf: &mut Vec<u8>, value: u64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn pushstring(buf: &mut Vec<u8>, value: &str) {
    pushu64(buf, value.len() as u64);
    buf.extend_from_slice(value.as_bytes());
}

fn readu32(bytes: &[u8], pos: &mut usize) -> Result<u32, checkpointerror> {
    let slice = readbytes(bytes, pos, 4)?;
    let mut arr = [0u8; 4];
    arr.copy_from_slice(slice);
    Ok(u32::from_le_bytes(arr))
}

fn readu64(bytes: &[u8], pos: &mut usize) -> Result<u64, checkpointerror> {
    let slice = readbytes(bytes, pos, 8)?;
    let mut arr = [0u8; 8];
    arr.copy_from_slice(slice);
    Ok(u64::from_le_bytes(arr))
}

fn readbytes<'a>(
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
