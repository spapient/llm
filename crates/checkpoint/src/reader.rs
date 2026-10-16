use std::borrow::Cow;
use std::collections::HashMap;
use std::path::Path;

use ::tensor::{element, tensor};
use format::dtypetoken;

use crate::blocks::{CODECRAW, TRANSFORMNONE, blockref, decompressblock};
use crate::crc32c::crc32c;
use crate::encryption;
use crate::error::checkpointerror;
use crate::hash::contenthash;
use crate::uformat::{
    FLAGDELTA, FLAGENCRYPTED, FLAGSFULL, MAGIC, PREFIXLEN, checkflags, parseartifact,
    parsedmanifest, sparseformat, storedentry, tensorentry,
};

/// A source of block bytes addressed by an absolute offset within a data
/// section. The abstraction reserves the extension point for mmap and
/// compressed blocks.
pub trait blocksource {
    /// Reads `size` bytes starting at `offset`.
    ///
    /// Sources may return a borrowed slice (zero-copy mmap or buffer) or an
    /// owned buffer.
    fn read(&self, offset: u64, size: usize) -> Result<Cow<'_, [u8]>, checkpointerror>;
}

/// A block source backed by an owned byte buffer.
pub struct bytesource {
    bytes: Vec<u8>,
}

impl bytesource {
    /// Wraps an owned byte buffer.
    #[must_use]
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }
}

impl blocksource for bytesource {
    fn read(&self, offset: u64, size: usize) -> Result<Cow<'_, [u8]>, checkpointerror> {
        let start = usize::try_from(offset).map_err(|_| checkpointerror::blockoutofrange {
            position: 0,
            blockcount: self.bytes.len(),
        })?;
        let end = start
            .checked_add(size)
            .ok_or(checkpointerror::blockoutofrange {
                position: start,
                blockcount: self.bytes.len(),
            })?;
        self.bytes
            .get(start..end)
            .map(Cow::Borrowed)
            .ok_or(checkpointerror::blockoutofrange {
                position: start,
                blockcount: self.bytes.len(),
            })
    }
}

/// A block source backed by a memory-mapped file, returning borrowed slices.
pub struct mmapsource {
    data: mmap_guard::FileData,
    base: u64,
}

impl mmapsource {
    /// Wraps a mapped file; `base` is the file offset of the data section.
    #[must_use]
    pub fn new(data: mmap_guard::FileData, base: u64) -> Self {
        Self { data, base }
    }
}

impl blocksource for mmapsource {
    fn read(&self, offset: u64, size: usize) -> Result<Cow<'_, [u8]>, checkpointerror> {
        let start = self
            .base
            .checked_add(offset)
            .ok_or(checkpointerror::blockoutofrange {
                position: 0,
                blockcount: 0,
            })?;
        let start = usize::try_from(start).map_err(|_| checkpointerror::blockoutofrange {
            position: 0,
            blockcount: 0,
        })?;
        let end = start
            .checked_add(size)
            .ok_or(checkpointerror::blockoutofrange {
                position: 0,
                blockcount: 0,
            })?;
        let slice: &[u8] = &self.data;
        slice
            .get(start..end)
            .map(Cow::Borrowed)
            .ok_or(checkpointerror::blockoutofrange {
                position: 0,
                blockcount: slice.len(),
            })
    }
}

/// Parses an artifact's manifest from a memory-mapped file slice.
fn parsemmap(data: &[u8]) -> Result<parsedmanifest, checkpointerror> {
    if data.len() < PREFIXLEN {
        return Err(checkpointerror::truncated {
            needed: PREFIXLEN,
            available: data.len(),
        });
    }
    let mut lenbytes = [0u8; 8];
    lenbytes.copy_from_slice(&data[MAGIC.len()..PREFIXLEN]);
    let manifestlen = u64::from_le_bytes(lenbytes);
    let manifestlen = usize::try_from(manifestlen).map_err(|_| checkpointerror::truncated {
        needed: usize::MAX,
        available: data.len(),
    })?;
    let manifestend = PREFIXLEN
        .checked_add(manifestlen)
        .ok_or(checkpointerror::truncated {
            needed: usize::MAX,
            available: data.len(),
        })?;
    if manifestend > data.len() {
        return Err(checkpointerror::truncated {
            needed: manifestend,
            available: data.len(),
        });
    }
    parseartifact(&data[..manifestend])
}

/// Memory-maps a file for read-only access.
fn mapfile(path: &Path) -> Result<mmap_guard::FileData, checkpointerror> {
    let strpath = path.to_str().ok_or_else(|| checkpointerror::io {
        path: path.to_path_buf(),
        source: std::io::Error::new(std::io::ErrorKind::InvalidInput, "path is not valid utf-8"),
    })?;
    mmap_guard::map_file(strpath).map_err(|source| checkpointerror::io {
        path: path.to_path_buf(),
        source,
    })
}

/// Resolves each logical block to a source, delta first then base.
fn resolvelocations(
    delta: &[storedentry],
    base: &[storedentry],
    blocks: &[blockref],
) -> Result<Vec<blocklocation>, checkpointerror> {
    let mut deltamap: HashMap<[u8; 16], blocklocation> = HashMap::new();
    for entry in delta {
        deltamap.insert(
            entry.hash,
            blocklocation {
                source: 0,
                offset: entry.offset,
                size: entry.size,
                crc32c: entry.crc32c,
                codec: entry.codec,
                transform: entry.transform,
                tag: entry.tag,
            },
        );
    }
    let mut basemap: HashMap<[u8; 16], blocklocation> = HashMap::new();
    for entry in base {
        basemap.insert(
            entry.hash,
            blocklocation {
                source: 1,
                offset: entry.offset,
                size: entry.size,
                crc32c: entry.crc32c,
                codec: entry.codec,
                transform: entry.transform,
                tag: entry.tag,
            },
        );
    }
    let mut locations = Vec::with_capacity(blocks.len());
    for block in blocks {
        let location = if let Some(location) = deltamap.get(&block.gethash()) {
            *location
        } else if let Some(location) = basemap.get(&block.gethash()) {
            *location
        } else {
            return Err(checkpointerror::missingblock {
                hash: block.gethash(),
            });
        };
        locations.push(location);
    }
    Ok(locations)
}

/// Where a logical block's bytes live.
#[derive(Clone, Copy)]
struct blocklocation {
    source: usize,
    offset: u64,
    size: u32,
    crc32c: u32,
    codec: u8,
    transform: u8,
    tag: [u8; 16],
}

/// Resolves the decryption key for a snapshot, requiring it when the snapshot's
/// flags mark it encrypted.
fn resolvekey(flags: u32, key: Option<[u8; 32]>) -> Result<Option<[u8; 32]>, checkpointerror> {
    match (flags & FLAGENCRYPTED != 0, key) {
        (true, Some(key)) => Ok(Some(key)),
        (true, None) => Err(checkpointerror::missingkey),
        (false, _) => Ok(None),
    }
}

/// Reads tensors from a full snapshot or a base plus a delta, touching only the
/// blocks a requested tensor spans.
pub struct checkpointreader {
    tensors: Vec<tensorentry>,
    blocks: Vec<blockref>,
    locations: Vec<blocklocation>,
    sources: Vec<Box<dyn blocksource>>,
    key: Option<[u8; 32]>,
}

impl checkpointreader {
    /// Opens a full snapshot's encoded bytes.
    ///
    /// # Errors
    ///
    /// Returns the manifest parse failures, or [`checkpointerror::invalidmanifest`]
    /// when the bytes are not a full snapshot.
    pub fn openfull(bytes: Vec<u8>) -> Result<Self, checkpointerror> {
        Self::openfull_with(bytes, None)
    }

    /// Opens an encrypted full snapshot's encoded bytes, decrypting blocks on
    /// demand with `key`.
    ///
    /// # Errors
    ///
    /// Returns the manifest parse failures, or [`checkpointerror::missingkey`]
    /// when the snapshot is not encrypted.
    pub fn openfull_encrypted(bytes: Vec<u8>, key: &[u8; 32]) -> Result<Self, checkpointerror> {
        Self::openfull_with(bytes, Some(*key))
    }

    fn openfull_with(bytes: Vec<u8>, key: Option<[u8; 32]>) -> Result<Self, checkpointerror> {
        let parsed = parseartifact(&bytes)?;
        checkflags(parsed.flags, FLAGSFULL)?;
        if parsed.base_id.iter().any(|&b| b != 0) {
            return Err(checkpointerror::invalidmanifest {
                reason: "full snapshot has a non-zero base identity".to_string(),
            });
        }
        let key = resolvekey(parsed.flags, key)?;
        let data = bytes[parsed.dataoffset..].to_vec();
        let locations = parsed
            .stored
            .iter()
            .map(|entry| blocklocation {
                source: 0,
                offset: entry.offset,
                size: entry.size,
                crc32c: entry.crc32c,
                codec: entry.codec,
                transform: entry.transform,
                tag: entry.tag,
            })
            .collect();
        Ok(Self {
            tensors: parsed.tensors,
            blocks: parsed.blocks,
            locations,
            sources: vec![Box::new(bytesource::new(data))],
            key,
        })
    }

    /// Opens a delta together with its base.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::basehashmismatch`] when the base's root hash
    /// does not match the delta's reference, [`checkpointerror::missingblock`]
    /// when a logical block resolves to neither, or the parse failures.
    pub fn opendelta(base: Vec<u8>, delta: Vec<u8>) -> Result<Self, checkpointerror> {
        Self::opendelta_with(base, delta, None)
    }

    /// Opens an encrypted delta together with its encrypted base, decrypting
    /// blocks on demand with `key`.
    pub fn opendelta_encrypted(
        base: Vec<u8>,
        delta: Vec<u8>,
        key: &[u8; 32],
    ) -> Result<Self, checkpointerror> {
        Self::opendelta_with(base, delta, Some(*key))
    }

    fn opendelta_with(
        base: Vec<u8>,
        delta: Vec<u8>,
        key: Option<[u8; 32]>,
    ) -> Result<Self, checkpointerror> {
        let parsed = parseartifact(&delta)?;
        checkflags(parsed.flags, FLAGDELTA)?;
        let baseparsed = parseartifact(&base)?;
        checkflags(baseparsed.flags, FLAGSFULL)?;
        if baseparsed.root_hash != parsed.base_id {
            return Err(checkpointerror::basehashmismatch {
                expected: parsed.base_id,
                found: baseparsed.root_hash,
            });
        }
        let deltaenc = parsed.flags & FLAGENCRYPTED != 0;
        let baseenc = baseparsed.flags & FLAGENCRYPTED != 0;
        if deltaenc != baseenc {
            return Err(checkpointerror::invalidmanifest {
                reason: "delta and base have inconsistent encryption".to_string(),
            });
        }
        let key = resolvekey(parsed.flags, key)?;

        let locations = resolvelocations(&parsed.stored, &baseparsed.stored, &parsed.blocks)?;

        let deltadata = delta[parsed.dataoffset..].to_vec();
        let basedata = base[baseparsed.dataoffset..].to_vec();
        Ok(Self {
            tensors: parsed.tensors,
            blocks: parsed.blocks,
            locations,
            sources: vec![
                Box::new(bytesource::new(deltadata)),
                Box::new(bytesource::new(basedata)),
            ],
            key,
        })
    }

    /// Opens a full snapshot file, reading blocks on demand.
    ///
    /// # Errors
    ///
    /// Returns the I/O or manifest parse failures, or
    /// [`checkpointerror::invalidmanifest`] when the file is not a full snapshot.
    pub fn openfullfile(path: &Path) -> Result<Self, checkpointerror> {
        Self::openfullfile_with(path, None)
    }

    /// Opens an encrypted full snapshot file, decrypting blocks on demand with
    /// `key`.
    pub fn openfullfile_encrypted(path: &Path, key: &[u8; 32]) -> Result<Self, checkpointerror> {
        Self::openfullfile_with(path, Some(*key))
    }

    fn openfullfile_with(path: &Path, key: Option<[u8; 32]>) -> Result<Self, checkpointerror> {
        let data = mapfile(path)?;
        let parsed = parsemmap(&data)?;
        checkflags(parsed.flags, FLAGSFULL)?;
        if parsed.base_id.iter().any(|&b| b != 0) {
            return Err(checkpointerror::invalidmanifest {
                reason: "full snapshot has a non-zero base identity".to_string(),
            });
        }
        let key = resolvekey(parsed.flags, key)?;
        let locations = parsed
            .stored
            .iter()
            .map(|entry| blocklocation {
                source: 0,
                offset: entry.offset,
                size: entry.size,
                crc32c: entry.crc32c,
                codec: entry.codec,
                transform: entry.transform,
                tag: entry.tag,
            })
            .collect();
        Ok(Self {
            tensors: parsed.tensors,
            blocks: parsed.blocks,
            locations,
            sources: vec![Box::new(mmapsource::new(data, parsed.dataoffset as u64))],
            key,
        })
    }

    /// Opens a delta file together with its base file, reading blocks on demand.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::basehashmismatch`] when the base's root hash
    /// does not match the delta's reference, [`checkpointerror::missingblock`]
    /// when a logical block resolves to neither, or the I/O and parse failures.
    pub fn opendeltafile(base_path: &Path, delta_path: &Path) -> Result<Self, checkpointerror> {
        Self::opendeltafile_with(base_path, delta_path, None)
    }

    /// Opens an encrypted delta file together with its encrypted base file,
    /// decrypting blocks on demand with `key`.
    pub fn opendeltafile_encrypted(
        base_path: &Path,
        delta_path: &Path,
        key: &[u8; 32],
    ) -> Result<Self, checkpointerror> {
        Self::opendeltafile_with(base_path, delta_path, Some(*key))
    }

    fn opendeltafile_with(
        base_path: &Path,
        delta_path: &Path,
        key: Option<[u8; 32]>,
    ) -> Result<Self, checkpointerror> {
        let deltadata = mapfile(delta_path)?;
        let parsed = parsemmap(&deltadata)?;
        checkflags(parsed.flags, FLAGDELTA)?;
        let basedata = mapfile(base_path)?;
        let baseparsed = parsemmap(&basedata)?;
        checkflags(baseparsed.flags, FLAGSFULL)?;
        if baseparsed.root_hash != parsed.base_id {
            return Err(checkpointerror::basehashmismatch {
                expected: parsed.base_id,
                found: baseparsed.root_hash,
            });
        }
        let deltaenc = parsed.flags & FLAGENCRYPTED != 0;
        let baseenc = baseparsed.flags & FLAGENCRYPTED != 0;
        if deltaenc != baseenc {
            return Err(checkpointerror::invalidmanifest {
                reason: "delta and base have inconsistent encryption".to_string(),
            });
        }
        let key = resolvekey(parsed.flags, key)?;
        let locations = resolvelocations(&parsed.stored, &baseparsed.stored, &parsed.blocks)?;
        Ok(Self {
            tensors: parsed.tensors,
            blocks: parsed.blocks,
            locations,
            sources: vec![
                Box::new(mmapsource::new(deltadata, parsed.dataoffset as u64)),
                Box::new(mmapsource::new(basedata, baseparsed.dataoffset as u64)),
            ],
            key,
        })
    }

    /// Reads a tensor's bytes, touching only the blocks it spans.
    ///
    /// Returns `Ok(None)` when the tensor is absent.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::blockcrcmismatch`] or
    /// [`checkpointerror::blockhashmismatch`] when a touched block is corrupt,
    /// or [`checkpointerror::offsetoutofrange`] when the tensor's range exceeds
    /// the data section.
    pub fn readtensor(&self, name: &str) -> Result<Option<Vec<u8>>, checkpointerror> {
        Ok(self.readtensorcow(name)?.map(Cow::into_owned))
    }

    /// Reads a tensor's bytes, returning a borrowed slice when the tensor is a
    /// single uncompressed block from a zero-copy source, and an owned buffer
    /// otherwise.
    ///
    /// Returns `Ok(None)` when the tensor is absent.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::blockcrcmismatch`] or
    /// [`checkpointerror::blockhashmismatch`] when a touched block is corrupt,
    /// or [`checkpointerror::offsetoutofrange`] when the tensor's range exceeds
    /// the data section.
    pub fn readtensorcow(&self, name: &str) -> Result<Option<Cow<'_, [u8]>>, checkpointerror> {
        let entry = match self.tensors.iter().find(|t| t.getname() == name) {
            Some(entry) => entry,
            None => return Ok(None),
        };
        let total = self.totallen();
        let start =
            usize::try_from(entry.getoffset()).map_err(|_| checkpointerror::offsetoutofrange {
                offset: entry.getoffset(),
                nbytes: entry.getnbytes(),
                datalength: total as u64,
            })?;
        let end =
            usize::try_from(entry.getend()).map_err(|_| checkpointerror::offsetoutofrange {
                offset: entry.getoffset(),
                nbytes: entry.getnbytes(),
                datalength: total as u64,
            })?;
        if end > total {
            return Err(checkpointerror::offsetoutofrange {
                offset: entry.getoffset(),
                nbytes: entry.getnbytes(),
                datalength: total as u64,
            });
        }

        let mut out = Vec::with_capacity(end - start);
        let mut running = 0usize;
        for (position, block) in self.blocks.iter().enumerate() {
            let len = block.getlen() as usize;
            let blockend = running + len;
            if blockend <= start {
                running = blockend;
                continue;
            }
            if running >= end {
                break;
            }
            let location = &self.locations[position];
            let blockbytes =
                self.sources[location.source].read(location.offset, location.size as usize)?;
            if crc32c(&blockbytes) != location.crc32c {
                return Err(checkpointerror::blockcrcmismatch { position });
            }
            let plain = match self.key {
                Some(key) => {
                    let nonce = encryption::nonceof(&block.gethash());
                    Cow::Owned(encryption::decryptblock(
                        &key,
                        &nonce,
                        &blockbytes,
                        &location.tag,
                    )?)
                }
                None => blockbytes,
            };
            let from = start.saturating_sub(running);
            let to = if blockend > end { end - running } else { len };
            match plain {
                Cow::Borrowed(slice)
                    if out.is_empty()
                        && blockend >= end
                        && location.codec == CODECRAW
                        && location.transform == TRANSFORMNONE =>
                {
                    let hash = contenthash(slice);
                    if hash != block.gethash() {
                        return Err(checkpointerror::blockhashmismatch {
                            position,
                            expected: block.gethash(),
                            found: hash,
                        });
                    }
                    return Ok(Some(Cow::Borrowed(&slice[from..to])));
                }
                bytes => {
                    let raw = decompressblock(location.codec, location.transform, &bytes)?;
                    let hash = contenthash(&raw);
                    if hash != block.gethash() {
                        return Err(checkpointerror::blockhashmismatch {
                            position,
                            expected: block.gethash(),
                            found: hash,
                        });
                    }
                    out.extend_from_slice(&raw[from..to]);
                }
            }
            running = blockend;
        }
        Ok(Some(Cow::Owned(out)))
    }

    /// Reads a tensor as a typed, owned tensor.
    ///
    /// Verifies that the stored storage dtype equals `T::DTYPE` before
    /// decoding, and carries the manifest shape into the returned tensor, so a
    /// caller cannot observe a silently wrong dtype or shape.
    ///
    /// Returns `Ok(None)` when the tensor is absent.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::dtypemismatch`] when the stored dtype
    /// differs, the corruption and range errors of [`Self::readtensorcow`], or
    /// the decoding error from [`tensor::tryfrombytes`] propagated as
    /// [`checkpointerror::tensor`].
    pub fn readtensortyped<T: element + bytemuck::Pod>(
        &self,
        name: &str,
    ) -> Result<Option<tensor<T>>, checkpointerror> {
        let entry = match self.tensors.iter().find(|t| t.getname() == name) {
            Some(entry) => entry,
            None => return Ok(None),
        };
        if entry.getdtype() != T::DTYPE {
            return Err(checkpointerror::dtypemismatch {
                name: name.to_string(),
                expected: dtypetoken(T::DTYPE).to_string(),
                actual: dtypetoken(entry.getdtype()).to_string(),
            });
        }
        let bytes = match self.readtensorcow(name)? {
            Some(bytes) => bytes,
            None => return Ok(None),
        };
        let dense = match entry.getsparse() {
            Some(sp) => {
                let count = entry
                    .getshape()
                    .getelementcount()
                    .map_err(checkpointerror::tensor)?;
                let width = entry.getdtype().sizeinbytes();
                let nnz =
                    usize::try_from(sp.getnnz()).map_err(|_| checkpointerror::invalidmanifest {
                        reason: "sparse non-zero count overflows usize".to_string(),
                    })?;
                let dense = match sp.getformat() {
                    sparseformat::coo => desparsify(&bytes, nnz, width, count)?,
                    sparseformat::bitmap => desparsifybitmap(&bytes, nnz, width, count)?,
                };
                Cow::Owned(dense)
            }
            None => bytes,
        };
        let decoded = match entry.getlayout() {
            Some(l) => tensor::tryfrombyteswithlayout(entry.getshape().clone(), l.clone(), &dense)?,
            None => tensor::tryfrombytes(entry.getshape().clone(), &dense)?,
        };
        Ok(Some(decoded))
    }

    /// Returns the total logical data length in bytes.
    fn totallen(&self) -> usize {
        self.blocks.iter().map(|b| b.getlen() as usize).sum()
    }
}

/// Reconstructs a dense byte buffer from a sparse encoding (indices first,
/// then values).
fn desparsify(
    bytes: &[u8],
    nnz: usize,
    width: usize,
    count: usize,
) -> Result<Vec<u8>, checkpointerror> {
    let indexbytes = nnz
        .checked_mul(4)
        .ok_or_else(|| checkpointerror::invalidmanifest {
            reason: "sparse index section overflows usize".to_string(),
        })?;
    let valuebytes = nnz
        .checked_mul(width)
        .ok_or_else(|| checkpointerror::invalidmanifest {
            reason: "sparse value section overflows usize".to_string(),
        })?;
    if bytes.len() != indexbytes + valuebytes {
        return Err(checkpointerror::invalidmanifest {
            reason: "sparse data length mismatch".to_string(),
        });
    }
    let dense_len = count
        .checked_mul(width)
        .ok_or_else(|| checkpointerror::invalidmanifest {
            reason: "dense buffer overflows usize".to_string(),
        })?;
    let mut dense = vec![0u8; dense_len];
    for i in 0..nnz {
        let idx = u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().map_err(|_| {
            checkpointerror::invalidmanifest {
                reason: "sparse index decode failed".to_string(),
            }
        })?) as usize;
        if idx >= count {
            return Err(checkpointerror::invalidmanifest {
                reason: "sparse index out of range".to_string(),
            });
        }
        let src = &bytes[indexbytes + i * width..indexbytes + (i + 1) * width];
        dense[idx * width..idx * width + width].copy_from_slice(src);
    }
    Ok(dense)
}

/// Reconstructs a dense byte buffer from a bitmap-sparse encoding (bitset then
/// values).
fn desparsifybitmap(
    bytes: &[u8],
    nnz: usize,
    width: usize,
    count: usize,
) -> Result<Vec<u8>, checkpointerror> {
    let bitmapbytes = count.div_ceil(8);
    let valuebytes = nnz
        .checked_mul(width)
        .ok_or_else(|| checkpointerror::invalidmanifest {
            reason: "sparse value section overflows usize".to_string(),
        })?;
    if bytes.len() != bitmapbytes + valuebytes {
        return Err(checkpointerror::invalidmanifest {
            reason: "sparse data length mismatch".to_string(),
        });
    }
    let dense_len = count
        .checked_mul(width)
        .ok_or_else(|| checkpointerror::invalidmanifest {
            reason: "dense buffer overflows usize".to_string(),
        })?;
    let bitmap = &bytes[..bitmapbytes];
    let values = &bytes[bitmapbytes..];
    let mut dense = vec![0u8; dense_len];
    let mut vi = 0usize;
    for i in 0..count {
        if (bitmap[i / 8] >> (i % 8)) & 1 != 0 {
            let src = &values[vi * width..(vi + 1) * width];
            dense[i * width..(i + 1) * width].copy_from_slice(src);
            vi += 1;
        }
    }
    Ok(dense)
}
