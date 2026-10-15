use std::collections::HashMap;
use std::path::Path;

use crate::atomicwrite::writeatomic;
use crate::blocks::chunk;
use crate::deltas::delta;
use crate::error::checkpointerror;
use crate::uformat::snapshot;

/// Merges a delta onto its base, producing the full snapshot the delta
/// represents.
///
/// The base's root hash must equal the delta's base identity. Each logical
/// block is resolved from the delta's stored blocks first, then from the base,
/// by content identity; a block that resolves nowhere is a corruption. The
/// result is a full snapshot with the delta's tensors and the reconstructed
/// data, whose root hash equals the delta's root hash.
///
/// # Errors
///
/// Returns [`checkpointerror::basehashmismatch`] when the base does not match
/// the delta's reference, or [`checkpointerror::missingblock`] when a logical
/// block is present in neither the delta nor the base.
pub fn merge(base: &snapshot, delta: &delta) -> Result<snapshot, checkpointerror> {
    let baseroot = base.roothash();
    if baseroot != delta.getbaseid() {
        return Err(checkpointerror::basehashmismatch {
            expected: delta.getbaseid(),
            found: baseroot,
        });
    }

    let mut basemap: HashMap<[u8; 16], &[u8]> = HashMap::new();
    let baseblocks = chunk(base.getdata());
    let mut offset = 0usize;
    for block in &baseblocks {
        let len = block.getlen() as usize;
        basemap.insert(block.gethash(), &base.getdata()[offset..offset + len]);
        offset += len;
    }

    let mut storedmap: HashMap<[u8; 16], &[u8]> = HashMap::new();
    for block in delta.getstored() {
        storedmap.insert(block.gethash(), block.getbytes());
    }

    let mut data = Vec::new();
    for block in delta.getblocks() {
        let bytes = if let Some(bytes) = storedmap.get(&block.gethash()) {
            bytes
        } else if let Some(bytes) = basemap.get(&block.gethash()) {
            bytes
        } else {
            return Err(checkpointerror::missingblock {
                hash: block.gethash(),
            });
        };
        data.extend_from_slice(bytes);
    }

    Ok(snapshot::fromparts(
        delta.getversion(),
        delta.gettensors().to_vec(),
        data,
    ))
}

/// Merges a delta file onto its base file, writing the full snapshot to
/// `out_path` atomically.
///
/// # Errors
///
/// Returns [`checkpointerror::missingbase`] when the base file is absent,
/// [`checkpointerror::basehashmismatch`] when the base does not match the
/// delta's reference, [`checkpointerror::missingblock`] when a block resolves
/// nowhere, or the decode and I/O failures.
pub fn mergefiles(
    base_path: &Path,
    delta_path: &Path,
    out_path: &Path,
) -> Result<(), checkpointerror> {
    let deltabytes = std::fs::read(delta_path).map_err(|source| checkpointerror::io {
        path: delta_path.to_path_buf(),
        source,
    })?;
    let delta = delta::decode(&deltabytes)?;
    let basebytes = match std::fs::read(base_path) {
        Ok(bytes) => bytes,
        Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
            return Err(checkpointerror::missingbase {
                base_id: delta.getbaseid(),
            });
        }
        Err(source) => {
            return Err(checkpointerror::io {
                path: base_path.to_path_buf(),
                source,
            });
        }
    };
    let base = snapshot::decode(&basebytes)?;
    let merged = merge(&base, &delta)?;
    let out = merged.encode()?;
    writeatomic(out_path, &out)
}
