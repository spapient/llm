use std::collections::HashMap;

use crate::blocks::blockref;
use crate::crc32c::crc32c;
use crate::error::checkpointerror;
use crate::hash::contenthash;
use crate::uformat::{FLAGDELTA, FLAGSFULL, parseartifact, tensorentry};

/// A source of block bytes addressed by an absolute offset within a data
/// section. The abstraction reserves the extension point for mmap and
/// compressed blocks.
pub trait blocksource {
    /// Reads `size` bytes starting at `offset`.
    fn read(&self, offset: u64, size: usize) -> Result<Vec<u8>, checkpointerror>;
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
    fn read(&self, offset: u64, size: usize) -> Result<Vec<u8>, checkpointerror> {
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
            .map(<[u8]>::to_vec)
            .ok_or(checkpointerror::blockoutofrange {
                position: start,
                blockcount: self.bytes.len(),
            })
    }
}

/// Where a logical block's bytes live.
#[derive(Clone, Copy)]
struct blocklocation {
    source: usize,
    offset: u64,
    size: u32,
    crc32c: u32,
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
        if parsed.flags != FLAGSFULL {
            return Err(checkpointerror::invalidmanifest {
                reason: "expected a full snapshot".to_string(),
            });
        }
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
        if parsed.flags != FLAGDELTA {
            return Err(checkpointerror::invalidmanifest {
                reason: "expected a delta snapshot".to_string(),
            });
        }
        let baseparsed = parseartifact(&base)?;
        if baseparsed.flags != FLAGSFULL {
            return Err(checkpointerror::invalidmanifest {
                reason: "expected a full base snapshot".to_string(),
            });
        }
        if baseparsed.root_hash != parsed.base_id {
            return Err(checkpointerror::basehashmismatch {
                expected: parsed.base_id,
                found: baseparsed.root_hash,
            });
        }

        let mut deltamap: HashMap<[u8; 16], blocklocation> = HashMap::new();
        for entry in &parsed.stored {
            deltamap.insert(
                entry.hash,
                blocklocation {
                    source: 0,
                    offset: entry.offset,
                    size: entry.size,
                    crc32c: entry.crc32c,
                },
            );
        }
        let mut basemap: HashMap<[u8; 16], blocklocation> = HashMap::new();
        for entry in &baseparsed.stored {
            basemap.insert(
                entry.hash,
                blocklocation {
                    source: 1,
                    offset: entry.offset,
                    size: entry.size,
                    crc32c: entry.crc32c,
                },
            );
        }
        let mut locations = Vec::with_capacity(parsed.blocks.len());
        for block in &parsed.blocks {
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
            let hash = contenthash(&blockbytes);
            if hash != block.gethash() {
                return Err(checkpointerror::blockhashmismatch {
                    position,
                    expected: block.gethash(),
                    found: hash,
                });
            }
            let from = start.saturating_sub(running);
            let to = if blockend > end { end - running } else { len };
            out.extend_from_slice(&blockbytes[from..to]);
            running = blockend;
        }
        Ok(Some(out))
    }

    /// Returns the total logical data length in bytes.
    fn totallen(&self) -> usize {
        self.blocks.iter().map(|b| b.getlen() as usize).sum()
    }
}
