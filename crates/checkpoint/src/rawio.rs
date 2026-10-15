//! On-demand reader for the raw weights artifact.
//!
//! [`rawreader`] reads a raw artifact's sections through a [`blocksource`],
//! verifying each section's CRC32C as it is read, so a file-backed reader
//! touches only the sections a caller requests. It mirrors
//! [`checkpointreader`](crate::checkpointreader) but without block tables,
//! deltas, or compression.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::Path;

use ::tensor::{element, tensor};
use format::dtypetoken;

use crate::crc32c::crc32c;
use crate::error::checkpointerror;
use crate::rawfmt::{
    RAWFORMATVERSION, RAWMAGIC, RAWPREFIXLEN, parsehead, rawsnapshot, rawtensorentry,
};
use crate::reader::{blocksource, bytesource, filesource};

/// Reads tensors from a raw weights artifact, touching only the sections a
/// requested tensor spans.
pub struct rawreader {
    tensors: Vec<rawtensorentry>,
    datalen: u64,
    source: Box<dyn blocksource>,
}

impl rawreader {
    /// Opens a raw artifact from an owned byte buffer.
    ///
    /// # Errors
    ///
    /// Returns the decode failures of [`rawsnapshot::decode`].
    pub fn open(bytes: Vec<u8>) -> Result<Self, checkpointerror> {
        let snap = rawsnapshot::decode(&bytes)?;
        let (tensors, data) = snap.intoinner();
        let datalen = data.len() as u64;
        Ok(Self {
            tensors,
            datalen,
            source: Box::new(bytesource::new(data)),
        })
    }

    /// Opens a raw artifact file, reading sections on demand.
    ///
    /// # Errors
    ///
    /// Returns the I/O failures opening or reading the file, or the header
    /// parse failures.
    pub fn fromfile(path: &Path) -> Result<Self, checkpointerror> {
        let file = File::open(path).map_err(|source| checkpointerror::io {
            path: path.to_path_buf(),
            source,
        })?;
        let mut prefix = [0u8; RAWPREFIXLEN];
        file.read_exact_at(&mut prefix, 0)
            .map_err(|source| checkpointerror::io {
                path: path.to_path_buf(),
                source,
            })?;
        if &prefix[..RAWMAGIC.len()] != RAWMAGIC {
            return Err(checkpointerror::badmagic);
        }
        let dataoffset = u64::from_le_bytes(
            prefix[16..24]
                .try_into()
                .map_err(|_| checkpointerror::badmagic)?,
        );
        let heaplen = usize::try_from(dataoffset).map_err(|_| checkpointerror::truncated {
            needed: usize::MAX,
            available: 0,
        })?;
        let mut head = vec![0u8; heaplen];
        file.read_exact_at(&mut head, 0)
            .map_err(|source| checkpointerror::io {
                path: path.to_path_buf(),
                source,
            })?;
        let (version, tensors, _) = parsehead(&head)?;
        if version != RAWFORMATVERSION {
            return Err(checkpointerror::unsupportedversion {
                found: version,
                supported: RAWFORMATVERSION,
            });
        }
        let filelen = file
            .metadata()
            .map_err(|source| checkpointerror::io {
                path: path.to_path_buf(),
                source,
            })?
            .len();
        let datalen =
            filelen
                .checked_sub(dataoffset)
                .ok_or_else(|| checkpointerror::invalidmanifest {
                    reason: "file is shorter than its declared data section".to_string(),
                })?;
        Ok(Self {
            tensors,
            datalen,
            source: Box::new(filesource::new(file, dataoffset, path.to_path_buf())),
        })
    }

    /// Returns the number of tensors declared in the directory.
    #[must_use]
    pub fn getcount(&self) -> usize {
        self.tensors.len()
    }

    /// Returns the tensor directory in file order.
    #[must_use]
    pub fn gettensors(&self) -> &[rawtensorentry] {
        &self.tensors
    }

    /// Reads every section in a single sequential pass over the data section.
    ///
    /// The whole data section is read as one large, contiguous I/O — the
    /// sequential-prefetch analogue of `posix_fadvise` that stays within safe
    /// Rust — and each section is then verified against its CRC32C in directory
    /// order. This is the fast path for loading a complete model, where N
    /// independent small reads would otherwise defeat the kernel's sequential
    /// read-ahead.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::offsetoutofrange`] when a section range
    /// exceeds the data section, [`checkpointerror::tensorcrcmismatch`] when a
    /// section is corrupt, or the source's I/O errors.
    pub fn readall(&self) -> Result<Vec<(String, Vec<u8>)>, checkpointerror> {
        let whole = self.source.read(0, self.datalen as usize)?;
        let mut out = Vec::with_capacity(self.tensors.len());
        for entry in &self.tensors {
            let start = usize::try_from(entry.getoffset()).map_err(|_| {
                checkpointerror::offsetoutofrange {
                    offset: entry.getoffset(),
                    nbytes: entry.getnbytes(),
                    datalength: self.datalen,
                }
            })?;
            let end =
                usize::try_from(entry.getend()).map_err(|_| checkpointerror::offsetoutofrange {
                    offset: entry.getoffset(),
                    nbytes: entry.getnbytes(),
                    datalength: self.datalen,
                })?;
            let section =
                whole
                    .get(start..end)
                    .ok_or_else(|| checkpointerror::offsetoutofrange {
                        offset: entry.getoffset(),
                        nbytes: entry.getnbytes(),
                        datalength: self.datalen,
                    })?;
            if crc32c(section) != entry.getcrc() {
                return Err(checkpointerror::tensorcrcmismatch {
                    name: entry.getname().to_string(),
                });
            }
            out.push((entry.getname().to_string(), section.to_vec()));
        }
        Ok(out)
    }

    /// Reads a tensor's bytes, verifying the section's CRC32C.
    ///
    /// Returns `Ok(None)` when the tensor is absent.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::offsetoutofrange`] when the section range
    /// exceeds the data section, [`checkpointerror::tensorcrcmismatch`] when
    /// the section is corrupt, or the source's I/O errors.
    pub fn readtensor(&self, name: &str) -> Result<Option<Vec<u8>>, checkpointerror> {
        let entry = match self.tensors.iter().find(|t| t.getname() == name) {
            Some(entry) => entry,
            None => return Ok(None),
        };
        if entry.getend() > self.datalen {
            return Err(checkpointerror::offsetoutofrange {
                offset: entry.getoffset(),
                nbytes: entry.getnbytes(),
                datalength: self.datalen,
            });
        }
        let bytes = self
            .source
            .read(entry.getoffset(), entry.getnbytes() as usize)?;
        if crc32c(&bytes) != entry.getcrc() {
            return Err(checkpointerror::tensorcrcmismatch {
                name: name.to_string(),
            });
        }
        Ok(Some(bytes.into_owned()))
    }

    /// Reads a tensor as a typed, owned tensor.
    ///
    /// Verifies that the stored storage dtype equals `T::DTYPE` before
    /// decoding, and carries the directory shape into the returned tensor, so a
    /// caller cannot observe a silently wrong dtype or shape.
    ///
    /// Returns `Ok(None)` when the tensor is absent.
    ///
    /// # Errors
    ///
    /// Returns [`checkpointerror::dtypemismatch`] when the stored dtype
    /// differs, the range and corruption errors of [`Self::readtensor`], or the
    /// decoding error from [`tensor::tryfrombytes`] propagated as
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
        let bytes = match self.readtensor(name)? {
            Some(bytes) => bytes,
            None => return Ok(None),
        };
        let decoded = tensor::tryfrombytes(entry.getshape().clone(), &bytes)?;
        Ok(Some(decoded))
    }
}
