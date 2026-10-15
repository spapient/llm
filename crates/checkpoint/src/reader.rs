use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use ::tensor::{element, tensor};
use format::dtypetoken;

use crate::blocks::{CODECRAW, blockref, decompressblock};
use crate::crc32c::crc32c;
use crate::error::checkpointerror;
use crate::hash::contenthash;
use crate::uformat::{
    FLAGDELTA, FLAGSFULL, MAGIC, PREFIXLEN, checkflags, parseartifact, parsedmanifest,
    storedentry, tensorentry,
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

/// A block source backed by a file, read at absolute offsets.
pub struct filesource {
    file: File,
    base: u64,
    path: PathBuf,
}

impl filesource {
    /// Wraps an opened file; `base` is the file offset of the data section.
    #[must_use]
    pub fn new(file: File, base: u64, path: PathBuf) -> Self {
        Self { file, base, path }
    }
}

impl blocksource for filesource {
    fn read(&self, offset: u64, size: usize) -> Result<Cow<'_, [u8]>, checkpointerror> {
        let at = self
            .base
            .checked_add(offset)
            .ok_or(checkpointerror::blockoutofrange {
                position: 0,
                blockcount: 0,
            })?;
        let mut buf = vec![0u8; size];
        self.file
            .read_exact_at(&mut buf, at)
            .map_err(|source| checkpointerror::io {
                path: self.path.clone(),
                source,
            })?;
        Ok(Cow::Owned(buf))
    }
}

/// Parses an artifact's manifest from a file without loading its data section.
fn parsefileartifact(file: &File, path: &Path) -> Result<parsedmanifest, checkpointerror> {
    let mut prefix = [0u8; PREFIXLEN];
    readat(file, &mut prefix, 0, path)?;
    let mut lenbytes = [0u8; 8];
    lenbytes.copy_from_slice(&prefix[MAGIC.len()..PREFIXLEN]);
    let manifestlen = u64::from_le_bytes(lenbytes);
    let manifestlen = usize::try_from(manifestlen).map_err(|_| checkpointerror::truncated {
        needed: usize::MAX,
        available: 0,
    })?;
    let mut manifest = vec![0u8; manifestlen];
    readat(file, &mut manifest, PREFIXLEN as u64, path)?;
    let mut bytes = Vec::with_capacity(PREFIXLEN + manifestlen);
    bytes.extend_from_slice(&prefix);
    bytes.extend_from_slice(&manifest);
    parseartifact(&bytes)
}

/// Reads exactly into `buf` at `offset`, mapping failures to an I/O error.
fn readat(file: &File, buf: &mut [u8], offset: u64, path: &Path) -> Result<(), checkpointerror> {
    file.read_exact_at(buf, offset)
        .map_err(|source| checkpointerror::io {
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
}

/// Reads tensors from a full snapshot or a base plus a delta, touching only the
/// blocks a requested tensor spans.
pub struct checkpointreader {
    tensors: Vec<tensorentry>,
    blocks: Vec<blockref>,
    locations: Vec<blocklocation>,
    sources: Vec<Box<dyn blocksource>>,
}

impl checkpointreader {
    /// Opens a full snapshot's encoded bytes.
    ///
    /// # Errors
    ///
    /// Returns the manifest parse failures, or [`checkpointerror::invalidmanifest`]
    /// when the bytes are not a full snapshot.
    pub fn openfull(bytes: Vec<u8>) -> Result<Self, checkpointerror> {
        let parsed = parseartifact(&bytes)?;
        checkflags(parsed.flags, FLAGSFULL)?;
        if parsed.base_id.iter().any(|&b| b != 0) {
            return Err(checkpointerror::invalidmanifest {
                reason: "full snapshot has a non-zero base identity".to_string(),
            });
        }
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
            })
            .collect();
        Ok(Self {
            tensors: parsed.tensors,
            blocks: parsed.blocks,
            locations,
            sources: vec![Box::new(bytesource::new(data))],
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
        })
    }

    /// Opens a full snapshot file, reading blocks on demand.
    ///
    /// # Errors
    ///
    /// Returns the I/O or manifest parse failures, or
    /// [`checkpointerror::invalidmanifest`] when the file is not a full snapshot.
    pub fn openfullfile(path: &Path) -> Result<Self, checkpointerror> {
        let file = File::open(path).map_err(|source| checkpointerror::io {
            path: path.to_path_buf(),
            source,
        })?;
        let parsed = parsefileartifact(&file, path)?;
        checkflags(parsed.flags, FLAGSFULL)?;
        if parsed.base_id.iter().any(|&b| b != 0) {
            return Err(checkpointerror::invalidmanifest {
                reason: "full snapshot has a non-zero base identity".to_string(),
            });
        }
        let locations = parsed
            .stored
            .iter()
            .map(|entry| blocklocation {
                source: 0,
                offset: entry.offset,
                size: entry.size,
                crc32c: entry.crc32c,
                codec: entry.codec,
            })
            .collect();
        Ok(Self {
            tensors: parsed.tensors,
            blocks: parsed.blocks,
            locations,
            sources: vec![Box::new(filesource::new(
                file,
                parsed.dataoffset as u64,
                path.to_path_buf(),
            ))],
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
        let deltafile = File::open(delta_path).map_err(|source| checkpointerror::io {
            path: delta_path.to_path_buf(),
            source,
        })?;
        let parsed = parsefileartifact(&deltafile, delta_path)?;
        checkflags(parsed.flags, FLAGDELTA)?;
        let basefile = File::open(base_path).map_err(|source| checkpointerror::io {
            path: base_path.to_path_buf(),
            source,
        })?;
        let baseparsed = parsefileartifact(&basefile, base_path)?;
        checkflags(baseparsed.flags, FLAGSFULL)?;
        if baseparsed.root_hash != parsed.base_id {
            return Err(checkpointerror::basehashmismatch {
                expected: parsed.base_id,
                found: baseparsed.root_hash,
            });
        }
        let locations = resolvelocations(&parsed.stored, &baseparsed.stored, &parsed.blocks)?;
        Ok(Self {
            tensors: parsed.tensors,
            blocks: parsed.blocks,
            locations,
            sources: vec![
                Box::new(filesource::new(
                    deltafile,
                    parsed.dataoffset as u64,
                    delta_path.to_path_buf(),
                )),
                Box::new(filesource::new(
                    basefile,
                    baseparsed.dataoffset as u64,
                    base_path.to_path_buf(),
                )),
            ],
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
            let from = start.saturating_sub(running);
            let to = if blockend > end { end - running } else { len };
            match blockbytes {
                Cow::Borrowed(slice)
                    if out.is_empty() && blockend >= end && location.codec == CODECRAW =>
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
                    let raw = decompressblock(location.codec, &bytes)?;
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
        let decoded = tensor::tryfrombytes(entry.getshape().clone(), &bytes)?;
        Ok(Some(decoded))
    }

    /// Returns the total logical data length in bytes.
    fn totallen(&self) -> usize {
        self.blocks.iter().map(|b| b.getlen() as usize).sum()
    }
}
