//! Inference-only raw weights artifact: page-aligned, uncompressed sections.
//!
//! The raw format is the inference-side counterpart of the training `.u`
//! snapshot. Where `.u` prioritizes space and incremental updates (content
//! addressing, LZ4 compression, deltas), the raw format prioritizes load speed
//! and addressability: tensor sections are stored uncompressed, every section
//! starts on a page boundary, and each section carries only a CRC32C for
//! corruption detection. The layout is deterministic and self-describing, so a
//! loader can map or read sections without parsing a JSON manifest.
//!
//! # Layout
//!
//! ```text
//! [0..8)    magic "LLMRAW01"
//! [8..12)   u32 version
//! [12..16)  u32 tensor count
//! [16..24)  u64 data section offset (page-aligned)
//! [24..)    tensor directory, one entry per tensor:
//!             u32 name length, UTF-8 name
//!             u8  storage dtype token
//!             u32 rank, u64 rank dimensions
//!             u64 section offset within the data section (page-aligned)
//!             u64 byte length
//!             u32 CRC32C of the section
//! [padding to the data section offset)
//! [data offset..) tensor sections in directory order, each starting on a
//!                 page boundary
//! ```
//!
//! # Scope
//!
//! The raw format stores a flat set of tensors. It carries no quantization
//! descriptors, training state, or delta/base relationship; scale and
//! zero-point tensors are exported as ordinary tensors whose association the
//! loader derives by naming. The `.u` snapshot remains the training and
//! checkpointing format.

use std::path::Path;

use tensor::{dtype, shape};

use crate::crc32c::crc32c;
use crate::error::checkpointerror;
use crate::manifests::checkpoint;
use crate::uformat::{pushu32, pushu64, readbytes, readu32, readu64};

/// The 8-byte magic that begins a raw weights artifact.
pub(crate) const RAWMAGIC: &[u8; 8] = b"LLMRAW01";

/// The raw artifact version this build writes and accepts.
pub const RAWFORMATVERSION: u32 = 1;

/// The alignment unit for tensor sections, in bytes.
///
/// One standard OS page (4 KiB). Every tensor section starts at an offset that
/// is a multiple of this unit, so a loader can map or align its reads at page
/// granularity.
pub const PAGESIZE: usize = 4096;

/// The fixed prefix length: magic (8) + version (4) + count (4) + data offset (8).
pub(crate) const RAWPREFIXLEN: usize = 24;

/// Publishes a raw artifact to a file through the atomic writer.
///
/// # Errors
///
/// Returns the encoding failure from [`rawsnapshot::encode`], or
/// [`checkpointerror::io`] when the staging write or rename fails.
pub fn saveraw(path: &Path, snap: &rawsnapshot) -> Result<(), checkpointerror> {
    let bytes = snap.encode()?;
    crate::atomicwrite::writeatomic(path, &bytes)?;
    Ok(())
}

/// Exports a checkpoint file to the inference raw weights format.
///
/// The checkpoint is loaded and validated in memory, its tensors are laid out
/// page-aligned and uncompressed, and the artifact is published atomically at
/// `raw_path`.
///
/// # Errors
///
/// Returns the load errors of [`loadcheckpoint`], the export errors of
/// [`rawsnapshot::fromcheckpoint`], or the save errors of [`saveraw`].
pub fn rawexport(checkpoint_path: &Path, raw_path: &Path) -> Result<(), checkpointerror> {
    let ckpt = crate::load::loadcheckpoint(checkpoint_path)?;
    let raw = rawsnapshot::fromcheckpoint(&ckpt)?;
    saveraw(raw_path, &raw)
}

/// Rounds `n` up to the next multiple of [`PAGESIZE`], with checked arithmetic.
fn alignup(n: u64) -> Result<u64, checkpointerror> {
    let page = PAGESIZE as u64;
    n.checked_add(page - 1)
        .map(|v| v / page * page)
        .ok_or_else(|| checkpointerror::invalidmanifest {
            reason: "length overflows when page-aligning".to_string(),
        })
}

/// Maps a storage [`dtype`] to its one-byte raw format token.
const fn dtypetobyte(value: dtype) -> u8 {
    match value {
        dtype::f32 => 0,
        dtype::f64 => 1,
        dtype::f16 => 2,
        dtype::bf16 => 3,
        dtype::i8 => 4,
        dtype::u8 => 5,
        dtype::i32 => 6,
        dtype::i64 => 7,
    }
}

/// Maps a raw format dtype token back to a storage [`dtype`].
///
/// # Errors
///
/// Returns [`checkpointerror::dtypeunsupported`] for an unknown token.
fn dtypefrombyte(token: u8, name: &str) -> Result<dtype, checkpointerror> {
    match token {
        0 => Ok(dtype::f32),
        1 => Ok(dtype::f64),
        2 => Ok(dtype::f16),
        3 => Ok(dtype::bf16),
        4 => Ok(dtype::i8),
        5 => Ok(dtype::u8),
        6 => Ok(dtype::i32),
        7 => Ok(dtype::i64),
        _ => Err(checkpointerror::dtypeunsupported {
            name: name.to_string(),
            dtype: format!("raw format token {token}"),
        }),
    }
}

/// One tensor's record in a raw weights artifact.
///
/// The value records the tensor's name, storage [`dtype`], logical [`shape`],
/// the page-aligned half-open byte range `[offset, offset + nbytes)` it occupies
/// within the artifact's data section, and the CRC32C of its bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct rawtensorentry {
    name: String,
    dtype: dtype,
    shape: shape,
    offset: u64,
    nbytes: u64,
    crc: u32,
}

impl rawtensorentry {
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

    /// Returns the section start offset within the data section.
    #[must_use]
    pub fn getoffset(&self) -> u64 {
        self.offset
    }

    /// Returns the section byte length.
    #[must_use]
    pub fn getnbytes(&self) -> u64 {
        self.nbytes
    }

    /// Returns the exclusive end offset within the data section.
    #[must_use]
    pub fn getend(&self) -> u64 {
        self.offset + self.nbytes
    }

    /// Returns the section's CRC32C.
    #[must_use]
    pub fn getcrc(&self) -> u32 {
        self.crc
    }
}

/// An in-memory raw weights artifact: tensors plus their page-aligned,
/// uncompressed data section.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct rawsnapshot {
    version: u32,
    tensors: Vec<rawtensorentry>,
    data: Vec<u8>,
}

impl rawsnapshot {
    /// Exports a checkpoint's tensors into a page-aligned raw artifact.
    ///
    /// Sections are laid out in manifest order, each starting on a page
    /// boundary. The byte length of every tensor is revalidated against its
    /// shape and dtype before the section is recorded.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::shapeoverflow`] when an element count or
    /// length overflows, [`checkpointerror::lengthmismatch`] when a tensor's
    /// stored bytes disagree with its manifest, or
    /// [`checkpointerror::missingtensor`] when a manifest entry has no bytes.
    pub fn fromcheckpoint(ckpt: &checkpoint) -> Result<Self, checkpointerror> {
        let mut tensors = Vec::new();
        let mut data = Vec::new();
        let mut cursor = 0u64;
        for entry in ckpt.getmanifest().gettensors() {
            let name = entry.getname();
            let count =
                entry
                    .getshape()
                    .getelementcount()
                    .map_err(|_| checkpointerror::shapeoverflow {
                        name: name.to_string(),
                    })?;
            let expected = (count as u64)
                .checked_mul(entry.getdtype().sizeinbytes() as u64)
                .ok_or_else(|| checkpointerror::shapeoverflow {
                    name: name.to_string(),
                })?;
            let bytes = ckpt
                .readtensor(name)
                .ok_or_else(|| checkpointerror::missingtensor {
                    name: name.to_string(),
                })?;
            if bytes.len() as u64 != expected {
                return Err(checkpointerror::lengthmismatch {
                    name: name.to_string(),
                    expected,
                    actual: bytes.len() as u64,
                });
            }
            tensors.push(rawtensorentry {
                name: name.to_string(),
                dtype: entry.getdtype(),
                shape: entry.getshape().clone(),
                offset: cursor,
                nbytes: expected,
                crc: crc32c(&bytes),
            });
            data.resize(cursor as usize, 0);
            data.extend_from_slice(&bytes);
            cursor = alignup(cursor.checked_add(expected).ok_or_else(|| {
                checkpointerror::shapeoverflow {
                    name: name.to_string(),
                }
            })?)?;
        }
        Ok(Self {
            version: RAWFORMATVERSION,
            tensors,
            data,
        })
    }

    /// Returns the artifact version.
    #[must_use]
    pub fn getversion(&self) -> u32 {
        self.version
    }

    /// Returns the tensor entries in directory order.
    #[must_use]
    pub fn gettensors(&self) -> &[rawtensorentry] {
        &self.tensors
    }

    /// Returns the flat data section, with sections laid out page-aligned.
    #[must_use]
    pub fn getdata(&self) -> &[u8] {
        &self.data
    }

    /// Decomposes the snapshot into its parts.
    pub(crate) fn intoinner(self) -> (Vec<rawtensorentry>, Vec<u8>) {
        (self.tensors, self.data)
    }

    /// Encodes the artifact into its on-disk byte layout.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::invalidmanifest`] when a count, length, or
    /// offset overflows the format's fixed-width fields.
    pub fn encode(&self) -> Result<Vec<u8>, checkpointerror> {
        let count =
            u32::try_from(self.tensors.len()).map_err(|_| checkpointerror::invalidmanifest {
                reason: "tensor count overflows u32".to_string(),
            })?;
        let mut dir = Vec::new();
        for t in &self.tensors {
            let namelen =
                u32::try_from(t.name.len()).map_err(|_| checkpointerror::invalidmanifest {
                    reason: "tensor name length overflows u32".to_string(),
                })?;
            pushu32(&mut dir, namelen);
            dir.extend_from_slice(t.name.as_bytes());
            dir.push(dtypetobyte(t.dtype));
            let rank = u32::try_from(t.shape.getdims().len()).map_err(|_| {
                checkpointerror::invalidmanifest {
                    reason: "tensor rank overflows u32".to_string(),
                }
            })?;
            pushu32(&mut dir, rank);
            for dim in t.shape.getdims() {
                let dim = u64::try_from(*dim).map_err(|_| checkpointerror::invalidmanifest {
                    reason: "tensor dimension overflows u64".to_string(),
                })?;
                pushu64(&mut dir, dim);
            }
            pushu64(&mut dir, t.offset);
            pushu64(&mut dir, t.nbytes);
            pushu32(&mut dir, t.crc);
        }
        let dirlen = dir.len() as u64;
        let dataoffset = alignup((RAWPREFIXLEN as u64).checked_add(dirlen).ok_or_else(|| {
            checkpointerror::invalidmanifest {
                reason: "directory length overflows".to_string(),
            }
        })?)?;
        let capacity = usize::try_from(dataoffset)
            .ok()
            .and_then(|n| n.checked_add(self.data.len()))
            .ok_or_else(|| checkpointerror::invalidmanifest {
                reason: "artifact length overflows usize".to_string(),
            })?;
        let mut out = Vec::with_capacity(capacity);
        out.extend_from_slice(RAWMAGIC);
        pushu32(&mut out, self.version);
        pushu32(&mut out, count);
        pushu64(&mut out, dataoffset);
        out.extend_from_slice(&dir);
        out.resize(dataoffset as usize, 0);
        out.extend_from_slice(&self.data);
        Ok(out)
    }

    /// Decodes and validates an artifact.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::badmagic`] for a missing magic,
    /// [`checkpointerror::truncated`] when the input ends early,
    /// [`checkpointerror::unsupportedversion`] for an unknown version,
    /// [`checkpointerror::invalidmanifest`] for a malformed directory, or the
    /// range and overlap errors from section validation.
    pub fn decode(bytes: &[u8]) -> Result<Self, checkpointerror> {
        let (version, tensors, dataoffset) = parsehead(bytes)?;
        if version != RAWFORMATVERSION {
            return Err(checkpointerror::unsupportedversion {
                found: version,
                supported: RAWFORMATVERSION,
            });
        }
        let start = usize::try_from(dataoffset).map_err(|_| checkpointerror::truncated {
            needed: usize::MAX,
            available: bytes.len(),
        })?;
        let data = bytes.get(start..).ok_or(checkpointerror::truncated {
            needed: start,
            available: bytes.len(),
        })?;
        verifyrawtensors(data, &tensors)?;
        Ok(Self {
            version,
            tensors,
            data: data.to_vec(),
        })
    }
}

/// Parses the fixed prefix and the tensor directory, without loading the data
/// section, so a reader can access sections on demand.
pub(crate) fn parsehead(bytes: &[u8]) -> Result<(u32, Vec<rawtensorentry>, u64), checkpointerror> {
    if bytes.len() < RAWPREFIXLEN {
        return Err(checkpointerror::truncated {
            needed: RAWPREFIXLEN,
            available: bytes.len(),
        });
    }
    if &bytes[..RAWMAGIC.len()] != RAWMAGIC {
        return Err(checkpointerror::badmagic);
    }
    let mut pos = RAWMAGIC.len();
    let version = readu32(bytes, &mut pos)?;
    let count = readu32(bytes, &mut pos)?;
    let dataoffset = readu64(bytes, &mut pos)?;
    let count = usize::try_from(count).map_err(|_| checkpointerror::invalidmanifest {
        reason: format!("tensor count {count} overflows usize"),
    })?;
    let mut tensors = Vec::with_capacity(count);
    for _ in 0..count {
        tensors.push(readentry(bytes, &mut pos)?);
    }
    if pos > dataoffset as usize {
        return Err(checkpointerror::invalidmanifest {
            reason: "tensor directory extends past the data section offset".to_string(),
        });
    }
    Ok((version, tensors, dataoffset))
}

/// Reads one directory entry at `pos`, advancing it.
fn readentry(bytes: &[u8], pos: &mut usize) -> Result<rawtensorentry, checkpointerror> {
    let namelen = readu32(bytes, pos)?;
    let namelen = usize::try_from(namelen).map_err(|_| checkpointerror::invalidmanifest {
        reason: "tensor name length overflows usize".to_string(),
    })?;
    let namebytes = readbytes(bytes, pos, namelen)?;
    let name =
        String::from_utf8(namebytes.to_vec()).map_err(|_| checkpointerror::invalidmanifest {
            reason: "tensor name is not valid UTF-8".to_string(),
        })?;
    let tok = *bytes.get(*pos).ok_or_else(|| checkpointerror::truncated {
        needed: *pos + 1,
        available: bytes.len(),
    })?;
    *pos += 1;
    let value = dtypefrombyte(tok, &name)?;
    let rank = readu32(bytes, pos)?;
    let rank = usize::try_from(rank).map_err(|_| checkpointerror::invalidmanifest {
        reason: "tensor rank overflows usize".to_string(),
    })?;
    let mut dims = Vec::with_capacity(rank);
    for _ in 0..rank {
        let dim = readu64(bytes, pos)?;
        let dim = usize::try_from(dim).map_err(|_| checkpointerror::invalidmanifest {
            reason: format!("tensor {name:?} dimension overflows usize"),
        })?;
        dims.push(dim);
    }
    let dims = shape::new(dims)?;
    let offset = readu64(bytes, pos)?;
    let nbytes = readu64(bytes, pos)?;
    let crc = readu32(bytes, pos)?;
    Ok(rawtensorentry {
        name,
        dtype: value,
        shape: dims,
        offset,
        nbytes,
        crc,
    })
}

/// Validates that every non-empty section lies within the data section, is
/// page-aligned, and does not overlap another section.
fn verifyrawtensors(data: &[u8], tensors: &[rawtensorentry]) -> Result<(), checkpointerror> {
    let mut sorted: Vec<&rawtensorentry> = tensors.iter().filter(|e| e.nbytes > 0).collect();
    sorted.sort_by_key(|e| e.offset);
    let datalen = data.len() as u64;
    let mut maxend = 0u64;
    for entry in sorted {
        if entry.offset % PAGESIZE as u64 != 0 {
            return Err(checkpointerror::invalidmanifest {
                reason: format!(
                    "tensor {name:?} section is not page-aligned",
                    name = entry.name
                ),
            });
        }
        if entry.getend() > datalen {
            return Err(checkpointerror::offsetoutofrange {
                offset: entry.offset,
                nbytes: entry.nbytes,
                datalength: datalen,
            });
        }
        if entry.offset < maxend {
            return Err(checkpointerror::overlappingtensors);
        }
        maxend = maxend.max(entry.getend());
    }
    Ok(())
}
