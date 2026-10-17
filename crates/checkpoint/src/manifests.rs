use std::collections::BTreeMap;

use format::{dtypetoken, tokendtype};
use lib::{jsonnumber, jsonvalue, parse, tostring};
use tensor::{dtype, shape};

use crate::error::checkpointerror;
use crate::training::trainingstate;

/// Magic prefix identifying a checkpoint artifact.
const MAGIC: [u8; 8] = *b"LLMCKPT1";

/// The number of bytes in the fixed prefix: magic plus manifest length.
const PREFIXLEN: usize = 16;

/// Describes one tensor persisted in a checkpoint.
///
/// A value records the tensor's name, storage [`dtype`], logical [`shape`], and
/// the half-open byte range `[offset, offset + nbytes)` it occupies within the
/// checkpoint's data section. The byte length always equals the element count
/// multiplied by the dtype width, and the range always lies within the data
/// section; this is enforced by [`manifest::addtensor`] on save and by
/// [`checkpoint::decode`] on load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct manifestentry {
    name: String,
    dtype: dtype,
    shape: shape,
    offset: u64,
    nbytes: u64,
}

impl manifestentry {
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

    /// Returns the start offset within the data section.
    #[must_use]
    pub fn getoffset(&self) -> u64 {
        self.offset
    }

    /// Returns the tensor's byte length.
    #[must_use]
    pub fn getnbytes(&self) -> u64 {
        self.nbytes
    }

    /// Returns the exclusive end offset within the data section.
    ///
    /// The sum is guaranteed not to overflow by the construction invariant.
    #[must_use]
    pub fn getend(&self) -> u64 {
        self.offset + self.nbytes
    }
}

/// The JSON manifest describing a checkpoint's contents.
///
/// The manifest is the single source of truth for the artifact: it names the
/// version, the optional training state, an optional string metadata map, and
/// every tensor with its dtype, shape, and byte range. Tensor order in
/// [`Self::gettensors`] matches the order entries were added and the order the
/// data is laid out on disk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct manifest {
    version: u32,
    training: Option<trainingstate>,
    metadata: BTreeMap<String, String>,
    tensors: Vec<manifestentry>,
}

impl manifest {
    /// Creates an empty manifest with the given version.
    #[must_use]
    pub fn new(version: u32) -> Self {
        Self {
            version,
            training: None,
            metadata: BTreeMap::new(),
            tensors: Vec::new(),
        }
    }

    /// Returns the checkpoint version.
    #[must_use]
    pub fn getversion(&self) -> u32 {
        self.version
    }

    /// Returns the training state, when present.
    #[must_use]
    pub fn gettraining(&self) -> Option<trainingstate> {
        self.training
    }

    /// Returns the string metadata map.
    #[must_use]
    pub fn getmetadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// Returns the tensor entries in insertion order.
    #[must_use]
    pub fn gettensors(&self) -> &[manifestentry] {
        &self.tensors
    }

    /// Returns the number of tensor entries.
    #[must_use]
    pub fn getlen(&self) -> usize {
        self.tensors.len()
    }

    /// Looks up a tensor entry by name.
    #[must_use]
    pub fn gettensor(&self, name: &str) -> Option<&manifestentry> {
        self.tensors.iter().find(|e| e.name == name)
    }

    /// Sets the training-state snapshot.
    pub fn settraining(&mut self, state: trainingstate) {
        self.training = Some(state);
    }

    /// Inserts a string metadata entry, replacing any existing value.
    pub fn addmetadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Appends a validated tensor entry.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::duplicatetensor`] when an entry with the same
    /// name already exists.
    pub fn addtensor(&mut self, entry: manifestentry) -> Result<(), checkpointerror> {
        if self.gettensor(&entry.name).is_some() {
            return Err(checkpointerror::duplicatetensor { name: entry.name });
        }
        self.tensors.push(entry);
        Ok(())
    }

    /// Serializes the manifest to canonical UTF-8 JSON.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::invalidmanifest`] if serialization fails,
    /// which cannot occur for a well-formed manifest but is handled explicitly
    /// rather than panicking.
    pub fn tobytes(&self) -> Result<Vec<u8>, checkpointerror> {
        let mut manifest = BTreeMap::new();
        manifest.insert(
            "version".to_string(),
            jsonvalue::number(jsonnumber::integer(self.version as i64)),
        );
        manifest.insert(
            "training".to_string(),
            match self.training {
                Some(state) => state.tojson(),
                None => jsonvalue::null,
            },
        );
        let mut metadata = BTreeMap::new();
        for (key, value) in &self.metadata {
            metadata.insert(key.clone(), jsonvalue::string(value.clone()));
        }
        manifest.insert("metadata".to_string(), jsonvalue::object(metadata));
        let tensors = self
            .tensors
            .iter()
            .map(entrytojson)
            .collect::<Vec<jsonvalue>>();
        manifest.insert("tensors".to_string(), jsonvalue::array(tensors));
        Ok(tostring(&jsonvalue::object(manifest)).into_bytes())
    }

    /// Parses and validates a manifest from UTF-8 JSON.
    ///
    /// Every entry's dtype token is resolved, its shape is rebuilt, and its
    /// declared byte length is checked against the element count and dtype
    /// width. Names must be unique. Offsets are checked against the data
    /// section by [`checkpoint::decode`], which knows the data length.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::invalidmanifestjson`] for malformed JSON,
    /// [`checkpointerror::dtypeunsupported`] for an unsupported dtype, or
    /// [`checkpointerror::invalidmanifest`] / [`checkpointerror::shapeoverflow`]
    /// / [`checkpointerror::lengthmismatch`] for inconsistent entries.
    pub fn frombytes(bytes: &[u8]) -> Result<Self, checkpointerror> {
        let root = parse(bytes).map_err(|error| checkpointerror::invalidmanifestjson {
            line: error.getline(),
            column: error.getcolumn(),
            message: error.to_string(),
        })?;
        let object = root
            .getobject()
            .ok_or_else(|| checkpointerror::invalidmanifest {
                reason: "manifest is not a JSON object".to_string(),
            })?;
        let version = object
            .get("version")
            .and_then(jsonvalue::getinteger)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| checkpointerror::invalidmanifest {
                reason: "manifest 'version' is not a non-negative integer".to_string(),
            })?;
        let training = match object.get("training") {
            Some(value) if !value.isnull() => Some(trainingstate::fromjson(value)?),
            _ => None,
        };
        let mut metadata = BTreeMap::new();
        if let Some(values) = object.get("metadata").and_then(jsonvalue::getobject) {
            for (key, value) in values {
                let text = value
                    .getstr()
                    .ok_or_else(|| checkpointerror::invalidmanifest {
                        reason: format!("metadata {key:?} is not a string"),
                    })?;
                metadata.insert(key.clone(), text.to_string());
            }
        }
        let raws = object
            .get("tensors")
            .and_then(jsonvalue::getarray)
            .ok_or_else(|| checkpointerror::invalidmanifest {
                reason: "manifest has no 'tensors' array".to_string(),
            })?;
        let mut tensors = Vec::with_capacity(raws.len());
        for entry in raws {
            tensors.push(jsonentrytoentry(entry)?);
        }
        let manifest = Self {
            version,
            training,
            metadata,
            tensors,
        };
        manifest.checkuniquenames()?;
        Ok(manifest)
    }

    /// Rejects duplicate tensor names.
    fn checkuniquenames(&self) -> Result<(), checkpointerror> {
        let mut seen: Vec<&str> = Vec::with_capacity(self.tensors.len());
        for entry in &self.tensors {
            if seen.contains(&entry.name.as_str()) {
                return Err(checkpointerror::duplicatetensor {
                    name: entry.name.clone(),
                });
            }
            seen.push(&entry.name);
        }
        Ok(())
    }
}

/// Converts a typed entry to its JSON value.
fn entrytojson(entry: &manifestentry) -> jsonvalue {
    let mut object = BTreeMap::new();
    object.insert("name".to_string(), jsonvalue::string(entry.name.clone()));
    object.insert(
        "dtype".to_string(),
        jsonvalue::string(dtypetoken(entry.dtype).to_string()),
    );
    let shape = entry
        .shape
        .getdims()
        .iter()
        .map(|dim| jsonvalue::number(jsonnumber::integer(*dim as i64)))
        .collect();
    object.insert("shape".to_string(), jsonvalue::array(shape));
    object.insert(
        "offset".to_string(),
        jsonvalue::number(jsonnumber::integer(entry.offset as i64)),
    );
    object.insert(
        "nbytes".to_string(),
        jsonvalue::number(jsonnumber::integer(entry.nbytes as i64)),
    );
    jsonvalue::object(object)
}

/// Validates a JSON entry value into a typed entry.
fn jsonentrytoentry(entry: &jsonvalue) -> Result<manifestentry, checkpointerror> {
    let object = entry
        .getobject()
        .ok_or_else(|| checkpointerror::invalidmanifest {
            reason: "manifest tensor entry is not a JSON object".to_string(),
        })?;
    let name = object
        .get("name")
        .and_then(jsonvalue::getstr)
        .ok_or_else(|| checkpointerror::invalidmanifest {
            reason: "manifest tensor entry has no string 'name'".to_string(),
        })?
        .to_string();
    let dtype = object
        .get("dtype")
        .and_then(jsonvalue::getstr)
        .ok_or_else(|| checkpointerror::invalidmanifest {
            reason: format!("tensor {name:?} has no string 'dtype'"),
        })?;
    let value = tokendtype(dtype).ok_or_else(|| checkpointerror::dtypeunsupported {
        name: name.clone(),
        dtype: dtype.to_string(),
    })?;
    let shape = object
        .get("shape")
        .and_then(jsonvalue::getarray)
        .ok_or_else(|| checkpointerror::invalidmanifest {
            reason: format!("tensor {name:?} has no 'shape' array"),
        })?;
    if shape.is_empty() {
        return Err(checkpointerror::invalidmanifest {
            reason: format!("tensor {name:?} has an empty shape"),
        });
    }
    let dims = shape
        .iter()
        .map(|dim| {
            dim.getinteger()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| checkpointerror::invalidmanifest {
                    reason: format!("tensor {name:?} has a non-integer shape dimension"),
                })
        })
        .collect::<Result<Vec<usize>, checkpointerror>>()?;
    let dims = shape::new(dims).map_err(|_| checkpointerror::invalidmanifest {
        reason: format!("tensor {name:?} has an invalid shape"),
    })?;
    let count = dims
        .getelementcount()
        .map_err(|_| checkpointerror::shapeoverflow { name: name.clone() })?;
    let expected = (count as u64)
        .checked_mul(value.sizeinbytes() as u64)
        .ok_or_else(|| checkpointerror::shapeoverflow { name: name.clone() })?;
    let offset = object
        .get("offset")
        .and_then(jsonvalue::getinteger)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| checkpointerror::invalidmanifest {
            reason: format!("tensor {name:?} has a non-integer 'offset'"),
        })?;
    let nbytes = object
        .get("nbytes")
        .and_then(jsonvalue::getinteger)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or_else(|| checkpointerror::invalidmanifest {
            reason: format!("tensor {name:?} has a non-integer 'nbytes'"),
        })?;
    if nbytes != expected {
        return Err(checkpointerror::lengthmismatch {
            name,
            expected,
            actual: nbytes,
        });
    }
    Ok(manifestentry {
        name,
        dtype: value,
        shape: dims,
        offset,
        nbytes,
    })
}

/// An in-memory checkpoint: a manifest plus its contiguous data section.
///
/// The container owns the tensor bytes as one flat buffer and records, for each
/// tensor, the dtype, shape, and byte range in the [`manifest`]. Tensor bytes
/// are appended in insertion order, so the data section layout is deterministic
/// and independent of the JSON key order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct checkpoint {
    manifest: manifest,
    data: Vec<u8>,
}

impl checkpoint {
    /// Creates an empty checkpoint with the given version.
    #[must_use]
    pub fn new(version: u32) -> Self {
        Self {
            manifest: manifest::new(version),
            data: Vec::new(),
        }
    }

    /// Returns the manifest.
    #[must_use]
    pub fn getmanifest(&self) -> &manifest {
        &self.manifest
    }

    /// Returns the flat data section.
    #[must_use]
    pub fn getdata(&self) -> &[u8] {
        &self.data
    }

    /// Sets the training-state snapshot.
    pub fn settraining(&mut self, state: trainingstate) {
        self.manifest.settraining(state);
    }

    /// Inserts a string metadata entry.
    pub fn addmetadata(&mut self, key: String, value: String) {
        self.manifest.addmetadata(key, value);
    }

    /// Appends a tensor's bytes and records its metadata.
    ///
    /// The byte length must equal the shape's element count multiplied by the
    /// dtype width. The offset is the current end of the data section, so the
    /// ranges are contiguous and non-overlapping by construction.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::shapeoverflow`] when the element count
    /// overflows `usize`, [`checkpointerror::lengthmismatch`] when the byte
    /// length is wrong, or [`checkpointerror::duplicatetensor`] for a repeated
    /// name. The data section is left unchanged on any error.
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
        let offset = self.data.len() as u64;
        let entry = manifestentry {
            name,
            dtype: value,
            shape: dims,
            offset,
            nbytes: actual,
        };
        self.manifest.addtensor(entry)?;
        self.data.extend_from_slice(bytes);
        Ok(())
    }

    /// Looks up a tensor entry by name.
    #[must_use]
    pub fn gettensor(&self, name: &str) -> Option<&manifestentry> {
        self.manifest.gettensor(name)
    }

    /// Copies a tensor's bytes out of the data section.
    ///
    /// Returns `None` when the tensor is absent or its range does not fit the
    /// stored data, so callers never observe a panic or an out-of-bounds slice.
    #[must_use]
    pub fn readtensor(&self, name: &str) -> Option<Vec<u8>> {
        let entry = self.manifest.gettensor(name)?;
        let start = usize::try_from(entry.getoffset()).ok()?;
        let end = usize::try_from(entry.getend()).ok()?;
        self.data.get(start..end).map(<[u8]>::to_vec)
    }

    /// Encodes the checkpoint into its on-disk byte layout.
    ///
    /// The layout is the 8-byte magic, the little-endian `u64` manifest length,
    /// the manifest JSON, and the data section. All length arithmetic is
    /// checked.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::invalidmanifest`] if the manifest cannot be
    /// serialized or the total length overflows `usize`.
    pub fn encode(&self) -> Result<Vec<u8>, checkpointerror> {
        let manifestbytes = self.manifest.tobytes()?;
        let total = PREFIXLEN
            .checked_add(manifestbytes.len())
            .and_then(|v| v.checked_add(self.data.len()))
            .ok_or_else(|| checkpointerror::invalidmanifest {
                reason: "checkpoint size overflows usize".to_string(),
            })?;
        let mut out = Vec::with_capacity(total);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&(manifestbytes.len() as u64).to_le_bytes());
        out.extend_from_slice(&manifestbytes);
        out.extend_from_slice(&self.data);
        Ok(out)
    }

    /// Decodes and validates a checkpoint from its on-disk byte layout.
    ///
    /// The magic, prefix length, and declared manifest length are checked with
    /// checked arithmetic; the manifest is validated; and every tensor's byte
    /// range is checked to lie inside the data section without overlapping.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::badmagic`], [`checkpointerror::truncated`],
    /// [`checkpointerror::invalidmanifestjson`], and the entry validation
    /// failures described on [`manifest::frombytes`], plus
    /// [`checkpointerror::offsetoutofrange`] or
    /// [`checkpointerror::overlappingtensors`].
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
        let available = bytes.len() - PREFIXLEN;
        let manifestlen = match usize::try_from(manifestlen) {
            Ok(value) if value <= available => value,
            _ => {
                let needed = usize::try_from(manifestlen)
                    .ok()
                    .and_then(|value| PREFIXLEN.checked_add(value))
                    .unwrap_or(usize::MAX);
                return Err(checkpointerror::truncated {
                    needed,
                    available: bytes.len(),
                });
            }
        };
        let manifestend = PREFIXLEN + manifestlen;
        let manifest = manifest::frombytes(&bytes[PREFIXLEN..manifestend])?;
        let data = bytes[manifestend..].to_vec();
        let datalength = data.len() as u64;
        for entry in manifest.gettensors() {
            let end = entry
                .getoffset()
                .checked_add(entry.getnbytes())
                .ok_or_else(|| checkpointerror::offsetoutofrange {
                    offset: entry.getoffset(),
                    nbytes: entry.getnbytes(),
                    datalength,
                })?;
            if end > datalength {
                return Err(checkpointerror::offsetoutofrange {
                    offset: entry.getoffset(),
                    nbytes: entry.getnbytes(),
                    datalength,
                });
            }
        }
        checkoverlaps(manifest.gettensors(), datalength)?;
        Ok(Self { manifest, data })
    }
}

/// Verifies that no two non-empty tensor ranges overlap and all lie in range.
fn checkoverlaps(tensors: &[manifestentry], datalength: u64) -> Result<(), checkpointerror> {
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
