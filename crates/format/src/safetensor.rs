use std::collections::BTreeMap;

use lib::limits;
use serde_json::Value;

use crate::error::formaterror;
use crate::metadata::tensormetadata;
use crate::validation::{checkoverlaps, parseentry};

/// The size, in bytes, of the little-endian header-length prefix.
pub const HEADERLENGTHSIZE: usize = 8;

/// The reserved safetensors key for a string-to-string metadata map.
const METADATAKEY: &str = "__metadata__";

/// A structurally validated safetensors artifact.
///
/// A value records the parsed header length, the data-section length, the
/// validated tensor metadata list, and the optional `__metadata__` string map.
/// It does not own the artifact bytes; callers pass the original buffer to
/// [`Self::readtensor`], which re-validates the absolute range before copying.
///
/// Construction goes exclusively through [`Self::parse`], so the header length,
/// tensor offsets, dtype support, element budget, and non-overlap invariants
/// hold for every value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct safetensorfile {
    headerlength: usize,
    datalength: u64,
    tensors: Vec<tensormetadata>,
    metadata: BTreeMap<String, String>,
}

impl safetensorfile {
    /// Parses and structurally validates a safetensors byte buffer.
    ///
    /// The buffer is treated as untrusted: the 8-byte little-endian header
    /// length is checked against the actual input length; the JSON header is
    /// parsed into the expected schema; every non-`__metadata__` entry is
    /// validated for dtype support, shape, element budget, and byte range; and
    /// non-empty tensor ranges are checked for overlap. All length and offset
    /// arithmetic is checked, and any inconsistency yields a [`formaterror`]
    /// rather than a panic.
    ///
    /// # Errors
    ///
    /// Returns a [`formaterror`] for truncated input, an oversized header,
    /// malformed JSON, an unsupported dtype, an inconsistent shape, an element
    /// budget rejection, an out-of-range or mismatched byte range, overlapping
    /// tensors, or a malformed `__metadata__` section.
    pub fn parse(bytes: &[u8], lim: &limits) -> Result<Self, formaterror> {
        if bytes.len() < HEADERLENGTHSIZE {
            return Err(formaterror::truncated {
                needed: HEADERLENGTHSIZE,
                available: bytes.len(),
            });
        }
        let mut lengthbytes = [0u8; HEADERLENGTHSIZE];
        lengthbytes.copy_from_slice(&bytes[..HEADERLENGTHSIZE]);
        let headerlength = u64::from_le_bytes(lengthbytes);
        let available = bytes.len() - HEADERLENGTHSIZE;
        if headerlength > available as u64 {
            return Err(formaterror::headertoolarge {
                declared: headerlength,
                available,
            });
        }
        let headerlength = headerlength as usize;
        let headerend = HEADERLENGTHSIZE + headerlength;
        let parsed: BTreeMap<String, Value> =
            serde_json::from_slice(&bytes[HEADERLENGTHSIZE..headerend]).map_err(|e| {
                formaterror::invalidjson {
                    line: e.line(),
                    column: e.column(),
                    message: e.to_string(),
                }
            })?;
        let datalength = (bytes.len() - headerend) as u64;
        let metadata = extractmetadata(&parsed)?;
        let mut tensors = Vec::with_capacity(parsed.len());
        for (name, value) in &parsed {
            if name == METADATAKEY {
                continue;
            }
            tensors.push(parseentry(name, value, datalength, lim)?);
        }
        checkoverlaps(&tensors)?;
        Ok(Self {
            headerlength,
            datalength,
            tensors,
            metadata,
        })
    }

    /// Returns the parsed header length in bytes.
    #[must_use]
    pub fn getheaderlength(&self) -> usize {
        self.headerlength
    }

    /// Returns the length of the data section in bytes.
    #[must_use]
    pub fn getdatalength(&self) -> u64 {
        self.datalength
    }

    /// Returns the validated tensor metadata, ordered by tensor name.
    #[must_use]
    pub fn gettensors(&self) -> &[tensormetadata] {
        &self.tensors
    }

    /// Returns the `__metadata__` string map, empty when absent.
    #[must_use]
    pub fn getmetadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// Looks up a tensor by name.
    #[must_use]
    pub fn gettensor(&self, name: &str) -> Option<&tensormetadata> {
        self.tensors.iter().find(|t| t.getname() == name)
    }

    /// Copies a tensor's bytes out of the original artifact buffer.
    ///
    /// The absolute byte range is recomputed from the parsed header length and
    /// the tensor's offsets with checked arithmetic, then validated against the
    /// supplied buffer before copying. This follows copy-validate-use: the
    /// bytes are not trusted merely because they were parsed earlier.
    ///
    /// # Errors
    ///
    /// Returns [`formaterror::invalidentry`] when the name is unknown, or
    /// [`formaterror::offsetsoutofrange`] when the supplied buffer does not
    /// contain the validated range.
    pub fn readtensor(&self, bytes: &[u8], name: &str) -> Result<Vec<u8>, formaterror> {
        let meta = self
            .gettensor(name)
            .ok_or_else(|| formaterror::invalidentry {
                name: name.to_string(),
                reason: "tensor not found".to_string(),
            })?;
        let begin = meta.getbegin();
        let end = meta.getend();
        let outofrange = |begin: u64, end: u64| formaterror::offsetsoutofrange {
            name: name.to_string(),
            begin,
            end,
            datalength: bytes.len() as u64,
        };
        let base = HEADERLENGTHSIZE
            .checked_add(self.headerlength)
            .ok_or_else(|| outofrange(begin, end))?;
        let start = usize::try_from(begin)
            .ok()
            .and_then(|offset| base.checked_add(offset))
            .ok_or_else(|| outofrange(begin, end))?;
        let stop = usize::try_from(end)
            .ok()
            .and_then(|offset| base.checked_add(offset))
            .ok_or_else(|| outofrange(begin, end))?;
        let slice = bytes
            .get(start..stop)
            .ok_or_else(|| outofrange(begin, end))?;
        Ok(slice.to_vec())
    }
}

/// Extracts the optional `__metadata__` string map from a parsed header.
fn extractmetadata(
    parsed: &BTreeMap<String, Value>,
) -> Result<BTreeMap<String, String>, formaterror> {
    match parsed.get(METADATAKEY) {
        None => Ok(BTreeMap::new()),
        Some(value) => {
            serde_json::from_value::<BTreeMap<String, String>>(value.clone()).map_err(|e| {
                formaterror::invalidmetadata {
                    reason: e.to_string(),
                }
            })
        }
    }
}
