use std::borrow::Cow;

use format::{dtypetoken, tokendtype};
use tensor::{dtype, layout, shape};

use crate::blocks::{TRANSFORMNONE, blockref, chunk, compressblock, decompressblock};
use crate::crc32c::crc32c;
use crate::encryption;
use crate::error::checkpointerror;
use crate::hash::{blake3hash, contenthash};
use crate::signature;

/// Magic identifying a `.u` snapshot artifact.
pub(crate) const MAGIC: [u8; 8] = *b"UWFSNAP1";

/// The number of bytes in the fixed prefix: magic plus manifest length.
pub(crate) const PREFIXLEN: usize = 16;

/// Flags value for a full (self-contained) snapshot.
pub(crate) const FLAGSFULL: u32 = 2;

/// Flags value for an incremental (delta) snapshot.
pub(crate) const FLAGDELTA: u32 = 1;

/// Flags value for a snapshot carrying an Ed25519 signature.
pub(crate) const FLAGSIGNED: u32 = 4;

/// Flags value for a snapshot whose blocks are AES-256-GCM encrypted.
pub(crate) const FLAGENCRYPTED: u32 = 8;

/// Flags value for a snapshot whose block table carries per-entry transform
/// bytes (the bit-shuffle + delta pre-pass).
pub(crate) const FLAGTRANSFORMED: u32 = 16;

/// One entry of the physical block table: where a stored block lives.
pub(crate) struct storedentry {
    pub(crate) hash: [u8; 16],
    pub(crate) offset: u64,
    pub(crate) size: u32,
    pub(crate) crc32c: u32,
    pub(crate) codec: u8,
    pub(crate) transform: u8,
    pub(crate) tag: [u8; 16],
}

/// Names the tensors carrying a quantized tensor's scale and zero-point, so a
/// reader can dequantize without relying on naming conventions.
///
/// The scale is a first-class tensor: block-addressed, deduplicated, and
/// compressed like any other tensor. `groupsize` is the number of elements
/// sharing one scale (0 = per-tensor); the scale tensor's own shape determines
/// whether it is per-channel, per-group, or per-block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct quantdesc {
    groupsize: u32,
    scale: String,
    zeropoint: Option<String>,
}

impl quantdesc {
    /// Creates an affine quantization descriptor with no zero-point.
    #[must_use]
    pub fn new(groupsize: u32, scale: String) -> Self {
        Self {
            groupsize,
            scale,
            zeropoint: None,
        }
    }

    /// Sets the zero-point tensor name.
    #[must_use]
    pub fn withzeropoint(mut self, zeropoint: String) -> Self {
        self.zeropoint = Some(zeropoint);
        self
    }

    /// Returns the group size in elements (0 = per-tensor).
    #[must_use]
    pub fn getgroupsize(&self) -> u32 {
        self.groupsize
    }

    /// Returns the scale tensor name.
    #[must_use]
    pub fn getscale(&self) -> &str {
        &self.scale
    }

    /// Returns the zero-point tensor name, when present.
    #[must_use]
    pub fn getzeropoint(&self) -> Option<&str> {
        self.zeropoint.as_deref()
    }
}

/// Describes a sparsely stored tensor: only the non-zero values and their
/// linear indices are persisted in the data section.
///
/// The data section stores the indices first (each a little-endian `u32` linear
/// element position) followed by the non-zero values, so a reader reconstructs
/// the dense tensor by zero-filling and scattering the values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct sparsedesc {
    nnz: u64,
}

impl sparsedesc {
    /// Creates a descriptor for a tensor with `nnz` non-zero elements.
    #[must_use]
    pub fn new(nnz: u64) -> Self {
        Self { nnz }
    }

    /// Returns the number of non-zero elements.
    #[must_use]
    pub fn getnnz(&self) -> u64 {
        self.nnz
    }
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
    quant: Option<quantdesc>,
    layout: Option<layout>,
    sparse: Option<sparsedesc>,
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

    /// Returns the quantization descriptor, when the tensor is quantized.
    #[must_use]
    pub fn getquant(&self) -> Option<&quantdesc> {
        self.quant.as_ref()
    }

    /// Returns the explicit layout, when it differs from the canonical
    /// contiguous (row-major) layout.
    #[must_use]
    pub fn getlayout(&self) -> Option<&layout> {
        self.layout.as_ref()
    }

    /// Returns the sparse-storage descriptor, when the tensor is sparsely
    /// stored.
    #[must_use]
    pub fn getsparse(&self) -> Option<&sparsedesc> {
        self.sparse.as_ref()
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
    signature: Option<[u8; 64]>,
}

impl snapshot {
    /// Creates an empty snapshot with the given version.
    #[must_use]
    pub fn new(version: u32) -> Self {
        Self {
            version,
            tensors: Vec::new(),
            data: Vec::new(),
            signature: None,
        }
    }

    /// Constructs a snapshot from its parts (used by the merge path).
    pub(crate) fn fromparts(version: u32, tensors: Vec<tensorentry>, data: Vec<u8>) -> Self {
        Self {
            version,
            tensors,
            data,
            signature: None,
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

    /// Returns the Ed25519 signature carried by the snapshot, when present.
    #[must_use]
    pub fn getsignature(&self) -> Option<[u8; 64]> {
        self.signature
    }

    /// Verifies the snapshot's Ed25519 signature over its root hash against a
    /// caller-supplied public key.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::signaturemissing`] when the snapshot carries
    /// no signature, or [`checkpointerror::signaturemismatch`] when the
    /// signature does not verify.
    pub fn verifysignature(&self, public: &[u8; 32]) -> Result<(), checkpointerror> {
        let signature = self.signature.ok_or(checkpointerror::signaturemissing)?;
        let root = self.roothash();
        if signature::verify(public, &root, &signature) {
            Ok(())
        } else {
            Err(checkpointerror::signaturemismatch)
        }
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
            quant: None,
            layout: None,
            sparse: None,
        };
        self.tensors.push(entry);
        self.data.extend_from_slice(bytes);
        Ok(())
    }

    /// Appends a sparsely stored tensor: `indices` are little-endian `u32`
    /// linear element positions and `values` are the matching non-zero bytes.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::shapeoverflow`], [`checkpointerror::lengthmismatch`],
    /// or [`checkpointerror::invalidmanifest`] for an out-of-range index.
    pub fn addsparse(
        &mut self,
        name: String,
        value: dtype,
        dims: shape,
        indices: &[u32],
        values: &[u8],
    ) -> Result<(), checkpointerror> {
        let count = dims
            .getelementcount()
            .map_err(|_| checkpointerror::shapeoverflow { name: name.clone() })?;
        let nnz = indices.len() as u64;
        let valuebytes = nnz
            .checked_mul(value.sizeinbytes() as u64)
            .ok_or_else(|| checkpointerror::shapeoverflow { name: name.clone() })?;
        if values.len() as u64 != valuebytes {
            return Err(checkpointerror::lengthmismatch {
                name: name.clone(),
                expected: valuebytes,
                actual: values.len() as u64,
            });
        }
        for &idx in indices {
            if (idx as u64) >= (count as u64) {
                return Err(checkpointerror::invalidmanifest {
                    reason: format!("sparse index {idx} out of range for tensor {name:?}"),
                });
            }
        }
        if self.tensors.iter().any(|t| t.name == name) {
            return Err(checkpointerror::duplicatetensor { name });
        }
        let offset = self.data.len() as u64;
        let nbytes = nnz
            .checked_mul(4)
            .and_then(|v| v.checked_add(valuebytes))
            .ok_or_else(|| checkpointerror::shapeoverflow { name: name.clone() })?;
        let entry = tensorentry {
            name,
            dtype: value,
            shape: dims,
            offset,
            nbytes,
            quant: None,
            layout: None,
            sparse: Some(sparsedesc::new(nnz)),
        };
        self.tensors.push(entry);
        for &idx in indices {
            self.data.extend_from_slice(&idx.to_le_bytes());
        }
        self.data.extend_from_slice(values);
        Ok(())
    }

    /// Attaches a quantization descriptor to an existing tensor.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::missingtensor`] when the tensor is absent.
    pub fn setquant(&mut self, name: &str, quant: quantdesc) -> Result<(), checkpointerror> {
        let entry = self
            .tensors
            .iter_mut()
            .find(|t| t.name == name)
            .ok_or_else(|| checkpointerror::missingtensor {
                name: name.to_string(),
            })?;
        entry.quant = Some(quant);
        Ok(())
    }

    /// Attaches an explicit layout to an existing tensor.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::missingtensor`] when the tensor is absent, or
    /// [`checkpointerror::invalidmanifest`] when the layout is not valid for
    /// the tensor's shape.
    pub fn setlayout(&mut self, name: &str, l: layout) -> Result<(), checkpointerror> {
        let entry = self
            .tensors
            .iter_mut()
            .find(|t| t.name == name)
            .ok_or_else(|| checkpointerror::missingtensor {
                name: name.to_string(),
            })?;
        if !l.isvalidfor(&entry.shape) {
            return Err(checkpointerror::invalidmanifest {
                reason: format!("layout is not valid for tensor {name:?}"),
            });
        }
        entry.layout = Some(l);
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

    /// Encodes the snapshot as a full (self-contained) artifact with no
    /// signature and no encryption.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::invalidmanifest`] if the total length
    /// overflows `usize`.
    pub fn encode(&self) -> Result<Vec<u8>, checkpointerror> {
        self.encode_with(None, None)
    }

    /// Encodes the snapshot as a full artifact signed with `seed`.
    ///
    /// The signature covers the snapshot's root hash, which commits to every
    /// block through the Merkle DAG.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::invalidsignaturekey`] for an all-zero seed,
    /// or [`checkpointerror::invalidmanifest`] if the total length overflows
    /// `usize`.
    pub fn encode_signed(&self, seed: &[u8; 32]) -> Result<Vec<u8>, checkpointerror> {
        self.encode_with(Some(seed), None)
    }

    /// Encodes the snapshot as a full artifact encrypted with `key`.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::invalidmanifest`] if the total length
    /// overflows `usize`.
    pub fn encode_encrypted(&self, key: &[u8; 32]) -> Result<Vec<u8>, checkpointerror> {
        self.encode_with(None, Some(key))
    }

    /// Encodes the snapshot as a full artifact signed with `seed` and encrypted
    /// with `key`.
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

    /// Encodes the snapshot, signing and encrypting it as requested.
    fn encode_with(
        &self,
        seed: Option<&[u8; 32]>,
        key: Option<&[u8; 32]>,
    ) -> Result<Vec<u8>, checkpointerror> {
        let blocks = chunk(&self.data);
        let canonical = canonicalmanifest(self.version, &self.tensors, &blocks);
        let root = blake3hash(&canonical);
        let mut flags = FLAGSFULL;
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

        // Phase 1: compress every block before the manifest header is written,
        // so the snapshot-level transform flag can be set when any block uses
        // the bit-shuffle pre-pass.
        let mut storedblocks = Vec::with_capacity(blocks.len());
        let mut has_transform = false;
        let mut offset = 0usize;
        for block in &blocks {
            let len = block.getlen() as usize;
            let (codec, transform, compressed) = compressblock(&self.data[offset..offset + len]);
            if transform != TRANSFORMNONE {
                has_transform = true;
            }
            storedblocks.push((codec, transform, compressed));
            offset += len;
        }
        if has_transform {
            flags |= FLAGTRANSFORMED;
        }

        let mut manifest = canonical;
        pushu32(&mut manifest, flags);
        manifest.extend_from_slice(&[0u8; 32]);
        manifest.extend_from_slice(&root);
        if let Some(sig) = signature {
            manifest.extend_from_slice(&sig);
        }
        pushu64(&mut manifest, blocks.len() as u64);
        let mut storeddata = Vec::new();
        for (block, (codec, transform, compressed)) in blocks.iter().zip(storedblocks) {
            let (stored, tag) = match key {
                Some(key) => {
                    let nonce = encryption::nonceof(&block.gethash());
                    encryption::encryptblock(key, &nonce, &compressed)?
                }
                None => (compressed, [0u8; 16]),
            };
            let checksum = crc32c(&stored);
            manifest.extend_from_slice(&block.gethash());
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
        Self::decode_with(bytes, None)
    }

    /// Decodes and validates an encrypted full snapshot, decrypting its blocks
    /// with `key`.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::missingkey`] when the snapshot is not
    /// encrypted, or the decode failures of [`Self::decode`].
    pub fn decode_encrypted(bytes: &[u8], key: &[u8; 32]) -> Result<Self, checkpointerror> {
        Self::decode_with(bytes, Some(key))
    }

    fn decode_with(bytes: &[u8], key: Option<&[u8; 32]>) -> Result<Self, checkpointerror> {
        let parsed = parseartifact(bytes)?;
        checkflags(parsed.flags, FLAGSFULL)?;
        if parsed.base_id.iter().any(|&b| b != 0) {
            return Err(checkpointerror::invalidmanifest {
                reason: "full snapshot has a non-zero base identity".to_string(),
            });
        }
        let encrypted = parsed.flags & FLAGENCRYPTED != 0;
        let key = match (encrypted, key) {
            (true, Some(key)) => Some(key),
            (true, None) => return Err(checkpointerror::missingkey),
            (false, _) => None,
        };
        let data = &bytes[parsed.dataoffset..];
        let rawdata = verifyfull(data, &parsed.blocks, &parsed.stored, key)?;
        verifytensors(&rawdata, &parsed.tensors)?;
        Ok(Self {
            version: parsed.version,
            tensors: parsed.tensors,
            data: rawdata,
            signature: parsed.signature,
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
    pub(crate) signature: Option<[u8; 64]>,
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
    let signature = if flags & FLAGSIGNED != 0 {
        let sigbytes = readbytes(manifest, &mut pos, 64)?;
        let mut sig = [0u8; 64];
        sig.copy_from_slice(sigbytes);
        Some(sig)
    } else {
        None
    };
    let stored = readstored(
        manifest,
        &mut pos,
        flags & FLAGENCRYPTED != 0,
        flags & FLAGTRANSFORMED != 0,
    )?;
    Ok(parsedmanifest {
        version,
        tensors,
        blocks,
        flags,
        base_id,
        root_hash,
        signature,
        stored,
        dataoffset: manifestend,
    })
}

/// Validates snapshot flags against an expected kind (full or delta), rejecting
/// unknown bits and a mixed full/delta declaration.
pub(crate) fn checkflags(flags: u32, expected: u32) -> Result<(), checkpointerror> {
    let known = FLAGSFULL | FLAGDELTA | FLAGSIGNED | FLAGENCRYPTED | FLAGTRANSFORMED;
    if flags & !known != 0 {
        return Err(checkpointerror::invalidmanifest {
            reason: format!("unsupported snapshot flags {flags}"),
        });
    }
    if flags & FLAGSFULL != 0 && flags & FLAGDELTA != 0 {
        return Err(checkpointerror::invalidmanifest {
            reason: "snapshot is both full and delta".to_string(),
        });
    }
    if flags & expected == 0 {
        return Err(checkpointerror::invalidmanifest {
            reason: format!("expected snapshot flags {expected}, found {flags}"),
        });
    }
    Ok(())
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
        match &entry.quant {
            None => pushu64(&mut buf, 0),
            Some(quant) => {
                pushu64(&mut buf, 1);
                pushu32(&mut buf, quant.groupsize);
                pushstring(&mut buf, &quant.scale);
                match &quant.zeropoint {
                    None => pushu64(&mut buf, 0),
                    Some(zp) => {
                        pushu64(&mut buf, 1);
                        pushstring(&mut buf, zp);
                    }
                }
            }
        }
        match &entry.layout {
            None => pushu64(&mut buf, 0),
            Some(l) => {
                pushu64(&mut buf, 1);
                pushu64(&mut buf, l.getoffset() as u64);
                let strides = l.getstrides();
                pushu64(&mut buf, strides.len() as u64);
                for &st in strides {
                    pushu64(&mut buf, st as u64);
                }
            }
        }
        match &entry.sparse {
            None => pushu64(&mut buf, 0),
            Some(sp) => {
                pushu64(&mut buf, 1);
                pushu64(&mut buf, sp.getnnz());
            }
        }
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
        let quant = match readu64(manifest, pos)? {
            0 => None,
            1 => {
                let groupsize = readu32(manifest, pos)?;
                let scale = readstring(manifest, pos)?;
                let zeropoint = match readu64(manifest, pos)? {
                    0 => None,
                    1 => Some(readstring(manifest, pos)?),
                    _ => {
                        return Err(checkpointerror::invalidmanifest {
                            reason: "invalid zero-point flag".to_string(),
                        });
                    }
                };
                Some(quantdesc {
                    groupsize,
                    scale,
                    zeropoint,
                })
            }
            _ => {
                return Err(checkpointerror::invalidmanifest {
                    reason: "invalid quant flag".to_string(),
                });
            }
        };
        let layout = match readu64(manifest, pos)? {
            0 => None,
            1 => {
                let layoutoffset = usize::try_from(readu64(manifest, pos)?).map_err(|_| {
                    checkpointerror::invalidmanifest {
                        reason: format!("tensor {name:?} layout offset overflows usize"),
                    }
                })?;
                let stridecount = readcount(manifest, pos)?;
                let mut strides = Vec::with_capacity(stridecount);
                for _ in 0..stridecount {
                    let st = usize::try_from(readu64(manifest, pos)?).map_err(|_| {
                        checkpointerror::invalidmanifest {
                            reason: format!("tensor {name:?} stride overflows usize"),
                        }
                    })?;
                    strides.push(st);
                }
                Some(layout::new(strides, layoutoffset))
            }
            _ => {
                return Err(checkpointerror::invalidmanifest {
                    reason: "invalid layout flag".to_string(),
                });
            }
        };
        if let Some(l) = &layout
            && !l.isvalidfor(&dims)
        {
            return Err(checkpointerror::invalidmanifest {
                reason: format!("tensor {name:?} has an invalid layout"),
            });
        }
        let sparse = match readu64(manifest, pos)? {
            0 => None,
            1 => Some(sparsedesc::new(readu64(manifest, pos)?)),
            _ => {
                return Err(checkpointerror::invalidmanifest {
                    reason: "invalid sparse flag".to_string(),
                });
            }
        };
        let count = dims
            .getelementcount()
            .map_err(|_| checkpointerror::shapeoverflow { name: name.clone() })?;
        let expected = match &sparse {
            Some(sp) => {
                let valuebytes = sp
                    .getnnz()
                    .checked_mul(value.sizeinbytes() as u64)
                    .ok_or_else(|| checkpointerror::shapeoverflow { name: name.clone() })?;
                sp.getnnz()
                    .checked_mul(4)
                    .and_then(|v| v.checked_add(valuebytes))
                    .ok_or_else(|| checkpointerror::shapeoverflow { name: name.clone() })?
            }
            None => (count as u64)
                .checked_mul(value.sizeinbytes() as u64)
                .ok_or_else(|| checkpointerror::shapeoverflow { name: name.clone() })?,
        };
        if nbytes != expected {
            return Err(checkpointerror::lengthmismatch {
                name,
                expected,
                actual: nbytes,
            });
        }
        if let Some(sp) = &sparse
            && sp.getnnz() > count as u64
        {
            return Err(checkpointerror::invalidmanifest {
                reason: format!("tensor {name:?} has more non-zeros than elements"),
            });
        }
        tensors.push(tensorentry {
            name,
            dtype: value,
            shape: dims,
            offset,
            nbytes,
            quant,
            layout,
            sparse,
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

/// Parses the physical block table, reading each entry's transform byte when
/// the snapshot is transformed and its authentication tag when encrypted.
pub(crate) fn readstored(
    manifest: &[u8],
    pos: &mut usize,
    encrypted: bool,
    transformed: bool,
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
        let transform = if transformed {
            readbytes(manifest, pos, 1)?[0]
        } else {
            TRANSFORMNONE
        };
        let tag = if encrypted {
            let tagbytes = readbytes(manifest, pos, 16)?;
            let mut tag = [0u8; 16];
            tag.copy_from_slice(tagbytes);
            tag
        } else {
            [0u8; 16]
        };
        stored.push(storedentry {
            hash,
            offset,
            size,
            crc32c,
            codec,
            transform,
            tag,
        });
    }
    Ok(stored)
}

/// Verifies a full snapshot: every stored entry matches its logical block and
/// the data section passes CRC32C and content-identity checks.
///
/// For snapshots with at least [`PARALLELMINBLOCKS`] blocks and
/// [`PARALLELMINBYTES`] of data section, the heavy per-block work (CRC,
/// decryption, decompression, transform inversion, content hashing) runs in
/// parallel across the available cores, each worker writing into its own
/// disjoint span of one preallocated output buffer. Metadata consistency and
/// data bounds are checked serially before the workers start, so manifest
/// errors keep their original ordering; a data error reports the first failing
/// block within the earliest failing worker span.
fn verifyfull(
    data: &[u8],
    blocks: &[blockref],
    stored: &[storedentry],
    key: Option<&[u8; 32]>,
) -> Result<Vec<u8>, checkpointerror> {
    if stored.len() != blocks.len() {
        return Err(checkpointerror::invalidmanifest {
            reason: "stored block count does not match the block list".to_string(),
        });
    }
    if blocks.len() >= PARALLELMINBLOCKS && data.len() >= PARALLELMINBYTES {
        return verifyfull_parallel(data, blocks, stored, key);
    }
    verifyfull_serial(data, blocks, stored, key)
}

/// Minimum block count and data-section size that trigger parallel decode;
/// below these thresholds thread creation would dominate the per-block work.
const PARALLELMINBLOCKS: usize = 8;
const PARALLELMINBYTES: usize = 1024 * 1024;

/// Serial decode: validates each block in order and returns the first error
/// encountered, exactly as snapshots were decoded before parallel decode.
fn verifyfull_serial(
    data: &[u8],
    blocks: &[blockref],
    stored: &[storedentry],
    key: Option<&[u8; 32]>,
) -> Result<Vec<u8>, checkpointerror> {
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
        let plain = match key {
            Some(key) => {
                let nonce = encryption::nonceof(&block.gethash());
                Cow::Owned(encryption::decryptblock(key, &nonce, bytes, &entry.tag)?)
            }
            None => Cow::Borrowed(bytes),
        };
        let raw = decompressblock(entry.codec, entry.transform, &plain)?;
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

/// Parallel decode: validates the manifest metadata serially, then decodes
/// every block concurrently into disjoint spans of one preallocated buffer.
fn verifyfull_parallel(
    data: &[u8],
    blocks: &[blockref],
    stored: &[storedentry],
    key: Option<&[u8; 32]>,
) -> Result<Vec<u8>, checkpointerror> {
    let mut rawtotal = 0usize;
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
        rawtotal = rawtotal
            .checked_add(block.getlen() as usize)
            .ok_or(checkpointerror::invalidmanifest {
                reason: format!("block {position} raw length overflows"),
            })?;
        offset = end;
    }
    if offset != data.len() {
        return Err(checkpointerror::blockoutofrange {
            position: offset,
            blockcount: data.len(),
        });
    }

    let nthreads = match std::thread::available_parallelism() {
        Ok(n) => n.get().min(blocks.len()),
        Err(_) => 1,
    };
    // One contiguous block span per worker, sized as evenly as possible.
    let mut spans = Vec::with_capacity(nthreads);
    let mut start = 0usize;
    for t in 0..nthreads {
        let len = blocks.len() / nthreads + usize::from(t < blocks.len() % nthreads);
        let end = start + len;
        spans.push(start..end);
        start = end;
    }

    let mut rawdata = vec![0u8; rawtotal];
    std::thread::scope(|scope| {
        // Preallocate the output and split it into disjoint per-span write
        // regions; the borrow ends when the scope returns.
        let mut cursor: &mut [u8] = &mut rawdata;
        let mut segs: Vec<&mut [u8]> = Vec::with_capacity(nthreads);
        for span in &spans {
            let seglen: usize = blocks[span.clone()]
                .iter()
                .map(|b| b.getlen() as usize)
                .sum();
            let (head, tail) = cursor.split_at_mut(seglen);
            segs.push(head);
            cursor = tail;
        }
        let mut handles = Vec::with_capacity(nthreads);
        for (span, out) in spans.into_iter().zip(segs) {
            handles.push(
                scope.spawn(move || verifyblockspan(data, blocks, stored, key, span, out)),
            );
        }
        for handle in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) => return Err(error),
                Err(_) => {
                    return Err(checkpointerror::invalidmanifest {
                        reason: "decode worker panicked".to_string(),
                    });
                }
            }
        }
        Ok(())
    })?;
    Ok(rawdata)
}

/// Decodes one contiguous span of blocks into `out`, which must hold exactly
/// the combined raw length of the span's blocks. The caller guarantees the
/// span's metadata and data bounds were already validated.
fn verifyblockspan(
    data: &[u8],
    blocks: &[blockref],
    stored: &[storedentry],
    key: Option<&[u8; 32]>,
    span: std::ops::Range<usize>,
    out: &mut [u8],
) -> Result<(), checkpointerror> {
    let mut written = 0usize;
    for position in span {
        let block = &blocks[position];
        let entry = &stored[position];
        let start = entry.offset as usize;
        let end = start + entry.size as usize;
        let bytes = &data[start..end];
        if crc32c(bytes) != entry.crc32c {
            return Err(checkpointerror::blockcrcmismatch { position });
        }
        let plain = match key {
            Some(key) => {
                let nonce = encryption::nonceof(&block.gethash());
                Cow::Owned(encryption::decryptblock(key, &nonce, bytes, &entry.tag)?)
            }
            None => Cow::Borrowed(bytes),
        };
        let raw = decompressblock(entry.codec, entry.transform, &plain)?;
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
        let dst = &mut out[written..written + raw.len()];
        dst.copy_from_slice(&raw);
        written += raw.len();
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
